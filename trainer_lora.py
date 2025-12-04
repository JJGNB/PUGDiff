

import os, sys, math, time, random, datetime
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils.import_utils import is_xformers_available
# import torchvision
import cv2
from datapipe.datasets import create_dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import util_pol
from utils import util_net
from utils import util_common
from utils import util_image
from utils.util_ops import append_dims
import pyiqa
from PIL import Image
from basicsr.arch.unet import CorrectNet
from diffusers import EulerDiscreteScheduler
from diffusers.models.autoencoders.vae import  DiagonalGaussianDistribution

def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
def get_torch_dtype(torch_dtype: str):
        if torch_dtype == 'torch.float16':
            return torch.float16
        elif torch_dtype == 'torch.bfloat16':
            return torch.bfloat16
        elif torch_dtype == 'torch.float32':
            return torch.float32
        else:
            raise ValueError(f'Unexpected torch dtype:{torch_dtype}')
class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.get('global_seeding', False)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        project_id=str(self.configs.project_id)
        save_dir = Path(self.configs.save_dir) / project_id
        if not save_dir.exists() and self.rank == 0:
            save_dir.mkdir(parents=True)
        else:
            save_dir = Path(self.configs.save_dir) / (project_id+"_"+datetime.datetime.now().strftime("%H-%M"))
            save_dir.mkdir(parents=True)
        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val','test']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val','test']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate if 'ema_rate' in self.configs.train else None
            # self.ema_rate_vae = self.configs.train.ema_rate_vae if 'ema_rate_vae' in self.configs.train else None
            # assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
                (image_dir / 'test').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        if self.configs.resume:
            if self.configs.sd_pipe.lora_ckpt_path is not None:
                if self.rank == 0:
                    self.logger.info(f"=> Loading checkpoint from {self.configs.sd_pipe.lora_ckpt_path}")
                lora_ckpt=torch.load(self.configs.sd_pipe.lora_ckpt_path)
            else:
                if self.rank == 0:
                    self.logger.info(f"=> Loading checkpoint from {self.configs.unet.ckpt_path}")
                lora_ckpt=torch.load(self.configs.unet.ckpt_path)
            torch.cuda.empty_cache()

            # learning rate scheduler
            if 'iters_start' in lora_ckpt:
                self.iters_start = lora_ckpt['iters_start']
            else:
                self.iters_start = 0
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                if 'log_step' in lora_ckpt and 'log_step_img' in lora_ckpt:
                    self.log_step = lora_ckpt['log_step']
                    self.log_step_img = lora_ckpt['log_step_img']
            if self.amp_scaler_lora is not None and self.configs.resume_opt:
                if "amp_scaler_lora" in lora_ckpt:
                    self.amp_scaler_lora.load_state_dict(lora_ckpt["amp_scaler_lora"])
                    self.amp_scaler.load_state_dict(lora_ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")
        else:
            self.iters_start = 0
    def setup_optimization(self):
        self.layers_to_opt = []
        if self.configs.get('use_unet', True) and not self.configs.get('use_sd', False):
            self.logger.info('add psrnet to optimize') 
            self.layers_to_opt= list(self.psr_net.parameters())
            self.optimizer= torch.optim.AdamW(self.layers_to_opt,
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)
        if self.configs.get('use_fusion', False):
            self.logger.info('add fusion_net to optimize')
            self.layers_to_opt += list(self.fusion_net.parameters())
            self.optimizer = torch.optim.AdamW(self.layers_to_opt,
                                            lr=self.configs.train.lr,
                                            weight_decay=self.configs.train.weight_decay)
        elif self.configs.sd_pipe.get('use_lora', False) and self.configs.get('use_sd', True):
            self.logger.info('add sd_unet to optimize')
            for n, _p in self.sd_pipe.unet.named_parameters():
                if "lora" in n:
                    self.layers_to_opt.append(_p)
            self.logger.info('add sd_vae to optimize')
            for n, _p in self.sd_pipe.vae.named_parameters():
                if "lora" in n:
                    self.layers_to_opt.append(_p)
            self.layers_to_opt += list(self.sd_pipe.unet.conv_in.parameters())
            self.optimizer = torch.optim.AdamW(self.layers_to_opt,
                                            lr=self.configs.train.lr,
                                            weight_decay=self.configs.train.weight_decay)
        # amp settings
        self.amp_scaler = torch.amp.GradScaler('cuda') if self.configs.train.use_amp else None
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )

    def prepare_compiling(self):
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#stable-diffusion-3
        if not hasattr(self, "prepare_compiling_well") or (not self.prepare_compiling_well):
            torch.set_float32_matmul_precision("high")
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
            self.prepare_compiling_well = True

    def set_grad_checkpointing(self, model):
        if hasattr(model, 'down_blocks'):
            for module in model.down_blocks:
                module.gradient_checkpointing = True
                module.training = True

        if hasattr(model, 'up_blocks'):
            for module in model.up_blocks:
                module.gradient_checkpointing = True
                module.training = True

        if hasattr(model, 'mid_block'):
            model.mid_block.gradient_checkpointing = True
            model.mid_block.training = True

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.validate.batch,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        all_param=0
        trainable_params=0
        for _, param in self.psr_net.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        self.logger.info(
                f"psrnet_trainable params: {trainable_params:,d} || "
                f"psrnet_all params: {all_param:,d} || "
                f"psrnet_trainable%: {100 * trainable_params / all_param:.4f}"
            )
        if self.configs.get('use_fusion', False):
            all_param=0
            trainable_params=0
            for _, param in self.fusion_net.named_parameters():
                num_params = param.numel()
                # if using DS Zero 3 and the weights are initialized empty
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel

                # Due to the design of 4bit linear layers from bitsandbytes
                # one needs to multiply the number of parameters by 2 to get
                # the correct number of parameters
                if param.__class__.__name__ == "Params4bit":
                    num_params = num_params * 2

                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            self.logger.info(
                    f"correctnet_trainable params: {trainable_params:,d} || "
                    f"correctnet_all params: {all_param:,d} || "
                    f"correctnet_trainable%: {100 * trainable_params / all_param:.4f}"
                )
        if self.configs.get('use_sd', True):
            all_param=0
            trainable_params=0
            for _, param in self.sd_pipe.vae.named_parameters():
                num_params = param.numel()
                # if using DS Zero 3 and the weights are initialized empty
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel

                # Due to the design of 4bit linear layers from bitsandbytes
                # one needs to multiply the number of parameters by 2 to get
                # the correct number of parameters
                if param.__class__.__name__ == "Params4bit":
                    num_params = num_params * 2

                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            self.logger.info(
                    f"vae_trainable params: {trainable_params:,d} || "
                    f"vae_all params: {all_param:,d} || "
                    f"vae_trainable%: {100 * trainable_params / all_param:.4f}"
                )
            all_param=0
            trainable_params=0
            for _, param in self.sd_pipe.unet.named_parameters():
                num_params = param.numel()
                # if using DS Zero 3 and the weights are initialized empty
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel

                # Due to the design of 4bit linear layers from bitsandbytes
                # one needs to multiply the number of parameters by 2 to get
                # the correct number of parameters
                if param.__class__.__name__ == "Params4bit":
                    num_params = num_params * 2

                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            self.logger.info(
                    f"unet_trainable params: {trainable_params:,d} || "
                    f"unet_all params: {all_param:,d} || "
                    f"unet_trainable%: {100 * trainable_params / all_param:.4f}"
                )

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass
    def test(self):
        pass
    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimization() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary
        if self.configs.get('use_sd', True) and self.configs.sd_pipe.get('use_lora', False):
            self.sd_pipe.unet.train()
            self.sd_pipe.vae.train()
            self.psr_net.eval()
            if self.configs.get('use_fusion', False):
                self.sd_pipe.unet.eval()
                self.sd_pipe.vae.eval()
                self.psr_net.train()
                self.fusion_net.train()
        else:
            self.psr_net.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch)
        self.best_psnr=0
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']), phase='train')
            
            
            # training phase
           
            self.training_step(data)

            # update ema model
            if hasattr(self.configs.train, 'ema_rate') and self.rank == 0:
                self.update_ema_model(self.psr_net,self.ema_model, self.configs.train.ema_rate)
            # validation phase
            if ((ii+1) % self.configs.train.save_freq == 0 and
                'val' in self.dataloaders and
                self.rank == 0
                ):
                # pass
                self.validation()
                if self.configs.data.test.test_for_real:
                    self.test()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0 and self.rank == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        self.close_logger()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.get("warmup_iterations", 0)
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def save_ckpt(self):
        ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
        if self.configs.sd_pipe.get('use_lora', False) and self.configs.get('use_sd', True):
            ckpt = {
                    'vae_lora_encoder_modules': self.sd_pipe.lora_vae_modules_encoder,
                    'unet_lora_encoder_modules': self.sd_pipe.lora_unet_modules_encoder,
                    'unet_lora_decoder_modules': self.sd_pipe.lora_unet_modules_decoder,
                    'unet_lora_others_modules': self.sd_pipe.lora_unet_others,
                    'iters_start': self.current_iters,
                    'log_step': {phase:self.log_step[phase] for phase in ['train', 'val','test']},
                    'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val','test']},
                    'lora_rank': self.configs.sd_pipe.lora_rank,
                    'state_dict_vae': {k: v for k, v in self.sd_pipe.vae.state_dict().items() if "lora" in k},
                    'state_dict_unet': {k: v for k, v in self.sd_pipe.unet.state_dict().items() if "lora" in k or "conv_in" in k},
                    }
            if self.configs.get('use_fusion',False):
                ckpt['state_dict_correct']=self.fusion_net.state_dict()
                # ckpt['state_dict_psrnet']=self.psr_net.state_dict()
        else:
            ckpt = {
                'iters_start': self.current_iters,
                'log_step': {phase:self.log_step[phase] for phase in ['train', 'val','test']},
                'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val','test']},
                'state_dict': self.psr_net.state_dict(),
                }
        if self.amp_scaler is not None:
            ckpt['amp_scaler'] = self.amp_scaler.state_dict()
        torch.save(ckpt, ckpt_path)
        if hasattr(self.configs.train, 'ema_rate'):
            ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
            torch.save(self.ema_model.state_dict(), ema_ckpt_path)
            
    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8,colormap=None):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path,colormap=colormap)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1



    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def load_model(self, model, ckpt_path=None, tag='model'):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
        if self.rank == 0:
            self.logger.info('Loaded Done')

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def unfreeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = True

    @torch.no_grad()
    def update_ema_model(self,net,net_ema,ema_rate):
        decay = min(ema_rate, (1 + self.current_iters) / (10 + self.current_iters))
        target_params = dict(net.named_parameters())

        one_minus_decay = 1.0 - decay

        for key, source_value in net_ema.named_parameters():
            target_value = target_params[key]
            if target_value.requires_grad:
                source_value.sub_(one_minus_decay * (source_value - target_value.data))

class TrainerBasePSR(TrainerBase):
    def build_model(self):
        if self.configs.train.get("compile", True):
            self.prepare_compiling()

        params = self.configs.unet.get('params', dict)
        psr_net = util_common.get_obj_from_str(self.configs.unet.target)(**params)
        # setting the base_branch
        if self.configs.unet.get('ckpt_path', None):   # initialize if necessary
            ckpt_path = self.configs.unet.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(psr_net, ckpt)
        psr_net.cuda()
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(psr_net)
            self.freeze_model(self.ema_model)
        self.psr_net = psr_net

        # setting the uncertainty model
        if self.configs.unet.get('teacher_ckpt_path', None):
            self.use_teacher = True
            params = self.configs.unet.get('params', dict)
            teacher = util_common.get_obj_from_str(self.configs.unet.target)(**params)
            ckpt_path = self.configs.unet.teacher_ckpt_path
            if self.rank == 0:
                self.logger.info(f"Teacher initializing from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(teacher, ckpt)
            teacher.cuda()
            self.freeze_model(teacher)
            self.teacher = teacher
            teacher.eval()
        else:
            self.use_teacher = False

        # setting the sd_branch
        if self.configs.get('use_sd', True) :
            # build the stable diffusion
            params = dict(self.configs.sd_pipe.params)
            torch_dtype = params.pop('torch_dtype')
            params['torch_dtype'] = get_torch_dtype(torch_dtype)
            # loading the fp16 robust vae for sdxl: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
            if self.configs.get('vae_fp16', None) is not None:
                params_vae = dict(self.configs.vae_fp16.params)
                params_vae['torch_dtype'] = torch.float16
                pipe_id = self.configs.vae_fp16.params.pretrained_model_name_or_path
                if self.rank == 0:
                    self.logger.info(f'Loading improved vae from {pipe_id}...')
                vae_pipe = util_common.get_obj_from_str(self.configs.vae_fp16.target).from_pretrained(**params_vae)
                if self.rank == 0:
                    self.logger.info('Loaded Done')
                params['vae'] = vae_pipe
            if ("StableDiffusion3" in self.configs.sd_pipe.target.split('.')[-1]
                and self.configs.sd_pipe.get("model_quantization", False)):
                if self.rank == 0:
                    self.logger.info(f'Loading the quantized transformer for SD3...')
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                params_model = dict(self.configs.model_nf4.params)
                torch_dtype = params_model.pop('torch_dtype')
                params_model['torch_dtype'] = get_torch_dtype(torch_dtype)
                params_model['quantization_config'] = nf4_config
                model_nf4 = util_common.get_obj_from_str(self.configs.model_nf4.target).from_pretrained(
                    **params_model
                )
                params['transformer'] = model_nf4
            sd_pipe = util_common.get_obj_from_str(self.configs.sd_pipe.target).from_pretrained(**params)
            # del sd_pipe.vae.decoder
            if self.configs.get('scheduler', None) is not None:
                pipe_id = self.configs.scheduler.target.split('.')[-1]
                if self.rank == 0:
                    self.logger.info(f'Loading scheduler of {pipe_id}...')
                sd_pipe.scheduler = util_common.get_obj_from_str(self.configs.scheduler.target).from_config(
                    sd_pipe.scheduler.config
                )
                if self.rank == 0:
                    self.logger.info('Loaded Done')
            if ("StableDiffusion3" in self.configs.sd_pipe.target.split('.')[-1]
                and self.configs.sd_pipe.get("model_quantization", False)):
                sd_pipe.enable_model_cpu_offload(gpu_id=self.rank,device='cuda')
            else:
                sd_pipe.to(f"cuda:{self.rank}")
            if not self.configs.sd_pipe.get('use_lora', False):
                # freezing model parameters
                if hasattr(sd_pipe, 'unet'):
                    self.freeze_model(sd_pipe.unet)
                if hasattr(sd_pipe, 'transformer'):
                    self.freeze_model(sd_pipe.transformer)
                self.freeze_model(sd_pipe.vae)
            if not self.configs.get('use_unet', True):
                self.freeze_model(self.psr_net)
            # compiling
            if self.configs.sd_pipe.get('compile', True):
                if self.rank == 0:
                    self.logger.info('Compile the SD model...')
                sd_pipe.set_progress_bar_config(disable=True)
                if hasattr(sd_pipe, 'unet'):
                    sd_pipe.unet.to(memory_format=torch.channels_last)
                    sd_pipe.unet = torch.compile(sd_pipe.unet, mode="max-autotune", fullgraph=False)
                if hasattr(sd_pipe, 'transformer'):
                    sd_pipe.transformer.to(memory_format=torch.channels_last)
                    sd_pipe.transformer = torch.compile(sd_pipe.transformer, mode="max-autotune", fullgraph=False)
                sd_pipe.vae.to(memory_format=torch.channels_last)
                sd_pipe.vae = torch.compile(sd_pipe.vae, mode="max-autotune", fullgraph=True)
            # setting gradient checkpoint for vae
            if self.configs.sd_pipe.get("enable_grad_checkpoint_vae", True):
                if self.rank == 0:
                    self.logger.info("Activating gradient checkpointing for VAE...")
                sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae,True)
                sd_pipe.vae._set_gradient_checkpointing(self.psr_net,True)

            # setting gradient checkpoint for diffusion model
            if self.configs.sd_pipe.enable_grad_checkpoint:
                if self.rank == 0:
                    self.logger.info("Activating gradient checkpointing for SD...")
                if hasattr(sd_pipe, 'unet'):
                    self.set_grad_checkpointing(sd_pipe.unet)
                if hasattr(sd_pipe, 'transformer'):
                    self.set_grad_checkpointing(sd_pipe.transformer)
                    # setting xformer for vae and diffusion
            if self.configs.sd_pipe.enable_xformers_memory_efficient_attention:
                if is_xformers_available():
                    sd_pipe.vae.enable_xformers_memory_efficient_attention()
                    sd_pipe.unet.enable_xformers_memory_efficient_attention()
                else:
                    raise ValueError("xformers is not available, please install it by running `pip install xformers`")
            # setting lora
            if self.configs.sd_pipe.get('use_lora', False):
                self.freeze_model(sd_pipe.vae)
                self.freeze_model(sd_pipe.unet)
                util_net.initialize_vae(self.configs.sd_pipe.lora_rank, sd_pipe,self.configs)
                util_net.initialize_unet(self.configs.sd_pipe.lora_rank, sd_pipe,self.configs)
                util_net.set_train_vae(sd_pipe)
                util_net.set_train_unet(sd_pipe)
                if self.configs.get('use_fusion', False):
                    self.freeze_model(sd_pipe.vae)
                    self.freeze_model(sd_pipe.unet)
                    self.freeze_model(self.psr_net)
            self.sd_pipe = sd_pipe
            # setting the fusion module
            if self.configs.get('use_fusion',False):
                self.logger.info("use fusion module...")
                self.fusion_net=CorrectNet().cuda()
                if self.configs.unet.get('ckpt_fusion_path', None):   # initialize if necessary
                    ckpt_path = self.configs.unet.ckpt_fusion_path
                    if self.rank == 0:
                        self.logger.info(f"Initializing fusion_net from {ckpt_path}")
                    ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
                    if 'state_dict_correct' in ckpt:
                        ckpt = ckpt['state_dict_correct']
                    util_net.reload_model(self.fusion_net, ckpt)
        # model information
        self.print_model_info()
        torch.cuda.empty_cache()
    def add_noise(self, latents, sigmas, model_pred=None):
        if sigmas.ndim < latents.ndim:
            sigmas = append_dims(sigmas, latents.ndim)

        if model_pred is None:
            zt_noisy=latents
        else:
            if self.configs.train.loss_coef.get('rkl', 0) > 0:
                mean, std = model_pred.mean, model_pred.std
                zt_noisy = latents + mean + sigmas * std * torch.randn_like(latents)
            else:
                mps=model_pred.sample()
                mps_3dim=util_pol.chunkdim2batch(mps)
                zt_noisy = latents + sigmas * mps_3dim

        return zt_noisy
    def prepare_lq_latents(
        self,
        image_lq: torch.Tensor,
        lq_downsample: torch.Tensor,
        timestep: torch.Tensor,
        start_noise_predictor: torch.nn.Module = None,
    ):
        """
        Input:
            image_lq: low-quality image, torch.Tensor, range in [0, 1]
            hight, width: resolution for high-quality image

        """
        image_lq_up = image_lq
        init_latents = self.encode_first_stage(
            image_lq_up, deterministic=True, center_input_sample=True,
        )

        if start_noise_predictor is None:
            model_pred = None
        else:
            model_pred,_ = start_noise_predictor(
                lq_downsample, timestep, sample_posterior=False, center_input_sample=True,
            )
        timestep=timestep.repeat(4)
        # get latents
        sigmas = self.sigmas_cache[timestep]
        sigmas = append_dims(sigmas, init_latents.ndim)
        latents = self.add_noise(init_latents, sigmas, model_pred)

        return latents
    def prepare_data(self, data, phase='train'):
        if phase == 'train':
            tfs=transforms.CenterCrop(int(self.configs.train.gt_size))
            data['gt']=tfs(data['gt'])
            # syn cpfa
            if not self.configs.data.train.params.data_source.source1.use_cpfa:
                self.cpfa=util_pol.Gen_CPFA(data['gt'])
            else:
                self.cpfa=data['cpfa']
            
            # gen init lq
            if not self.configs.data.train.params.data_source.source1.use_lq:
                self.lq=util_pol.init_CPDM(self.cpfa)
            else:
                self.lq=data['lq']
            self.gt=data['gt']
            gt_clone=data['gt'].clone()
            lq_clone=self.lq.clone()
            s0_gt,dolp_gt,_=util_pol.I2S012(gt_clone,save_img=True,use_loss=False,improve_contrast=False,clip=True)
            s0_lq,dolp_lq,_=util_pol.I2S012(lq_clone,save_img=True,use_loss=False,improve_contrast=False,clip=True)
        else:
            if self.configs.data.val.type=="CPDM":
                # syn cpfa
                if not self.configs.data.val.params.data_source.source1.use_cpfa:
                    self.cpfa=util_pol.Gen_CPFA(data['gt'])
                else:
                    self.cpfa=data['cpfa']
                # gen init lq
                if not self.configs.data.val.params.data_source.source1.use_lq:
                    self.lq=util_pol.init_CPDM(self.cpfa)
                else:
                    self.lq=data['lq']
                self.gt=data['gt']
                gt_clone=data['gt'].clone()
                lq_clone=self.lq.clone()
                s0_gt,dolp_gt,_=util_pol.I2S012(gt_clone,save_img=True,use_loss=False,improve_contrast=False,clip=True)
                s0_lq,dolp_lq,_=util_pol.I2S012(lq_clone,save_img=True,use_loss=False,improve_contrast=False,clip=True)
            elif self.configs.data.val.type=="RealCPDM":
                # use real cpfa
                self.cpfa=data['lq'].clone()
                crop_size=self.configs.data.val.crop_szie
                tfs=transforms.CenterCrop([crop_size,crop_size])
                self.cpfa=tfs(self.cpfa)
                # gen init lq
                self.lq=util_pol.init_CPDM(self.cpfa[:,0,:,:])
                lq_clone=self.lq.clone()
                self.gt=self.lq.clone()
                gt_clone=self.gt.clone()
                s0_gt,dolp_gt,_=util_pol.I2S012(gt_clone,save_img=True,use_loss=False,improve_contrast=False,clip=True)
                s0_lq,dolp_lq,_=util_pol.I2S012(lq_clone,save_img=True,use_loss=False,improve_contrast=False,clip=True)
        batch = {'lq':self.lq.cuda(), 'gt':self.gt.cuda(), 's0_gt':s0_gt.cuda(), 's0_lq':s0_lq.cuda(),  'dolp_gt':dolp_gt.cuda(), 'dolp_lq':dolp_lq.cuda()}
        return batch
    
    @torch.no_grad()
    def log_step_train(self, losses, tt, micro_data,x0_pred,  x0_base=False, x0_sd=False,phase='train'):
        '''
        param losses: a dict recording the loss informations
        '''
        '''
        param loss: a dict recording the loss informations
        param micro_data: batch data
        param tt: 1-D tensor, time steps
        '''
        if hasattr(self.configs.train, 'timesteps'):
            if len(self.configs.train.timesteps) < 3:
                record_steps = sorted(self.configs.train.timesteps)
            else:
                record_steps = [min(self.configs.train.timesteps),
                                max(self.configs.train.timesteps)]
        else:
            max_inference_steps = self.configs.train.max_inference_steps
            record_steps = [1, max_inference_steps//2, max_inference_steps]
        if (self.current_iters %
            self.configs.train.log_freq[0]) == 1:
            self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                              for key in losses.keys()}
            self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
        for jj in range(len(record_steps)):
            for key, value in losses.items():
                index = record_steps[jj] - 1
                mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                assert value.shape == mask.shape
                current_loss = torch.sum(value.detach() * mask)
                self.loss_mean[key][jj] += current_loss.item()
            self.loss_count[jj] += mask.sum().item()

        if (self.current_iters  %
            self.configs.train.log_freq[0])  == 0:
            if torch.any(self.loss_count == 0):
                self.loss_count += 1e-4
            for key in losses.keys():
                self.loss_mean[key] /= self.loss_count
            log_str = f"Train: {self.current_iters:06d}/{self.configs.train.iterations:06d}, "
            valid_keys = sorted([key for key in losses.keys() if key not in ['loss', 'real', 'fake']])
            for ii, key in enumerate(valid_keys):
                if ii == 0:
                    log_str += f"{key}"
                else:
                    log_str += f"/{key}"
 
            log_str += ":"
            for jj, current_record in enumerate(record_steps):
                for ii, key in enumerate(valid_keys):
                    if ii == 0:
                        log_str += 't({:d}):{:.1e}'.format(
                                current_record,
                                self.loss_mean[key][jj].item(),
                                )
                    else:
                        log_str += f"/{self.loss_mean[key][jj].item():.1e}"

                log_str += f", "
            log_str += 'lr:{:.1e}'.format(self.optimizer.param_groups[0]['lr'])
            self.logger.info(log_str)
            self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
        if (self.current_iters  %
            self.configs.train.log_freq[1]) == 0:
            if x0_base is not None:
                x0_base=util_pol.chunkdim2batch(x0_base)
                self.logging_image(x0_base, tag='x0-base', phase=phase, add_global_step=False)
            if x0_sd is not None:
                x0_sd=util_pol.chunkdim2batch(x0_sd)
                self.logging_image(x0_sd, tag='x0-sd', phase=phase, add_global_step=False)
            if x0_pred is not None:
                s0_pred,dolp_pred,_ = util_pol.I2S012(x0_pred.detach(),save_img=True,use_loss=False,improve_contrast=False,clip=True)
                x0_recon = 2*x0_pred.detach()-1
                x0_recon=util_pol.chunkdim2batch(x0_recon)
                self.logging_image(x0_recon, tag='x0-final', phase=phase, add_global_step=False)
                self.logging_image(s0_pred, tag='s0-final', phase=phase, add_global_step=False)
                self.logging_image(dolp_pred, tag='dolp-final', phase=phase, add_global_step=False)
            self.logging_image(micro_data['s0_gt'], tag='s0_gt', phase=phase, add_global_step=False)
            self.logging_image(micro_data['dolp_gt'], tag='dolp_gt', phase=phase, add_global_step=True)

        if (self.current_iters%
            self.configs.train.save_freq) == 1:
            self.tic = time.time()
        if (self.current_iters %
            self.configs.train.save_freq) == 0:
            self.toc = time.time()
            elaplsed = (self.toc - self.tic)
            self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
            self.logger.info("="*100)
    def backward_step_onlylora(self, micro_data):
        loss_coef = self.configs.train.loss_coef
        losses = {}
        tt = torch.tensor(
            random.choices(self.configs.train.timesteps, k=micro_data['lq'].shape[0]),
            dtype=torch.int64,
            device=f"cuda:{self.rank}",
        ) - 1
        with torch.autocast(device_type="cuda", enabled=self.configs.train.use_amp):

            input_base=2*micro_data['lq']-1
            # base branch
            with torch.no_grad():
                eps_pred=self.psr_net(input_base,tt)
            
            if isinstance(eps_pred, list):
                x0_base=input_base-eps_pred[0]
            else:
                x0_base=input_base-eps_pred[0]
            input_sd=util_pol.chunkdim2batch(x0_base)
            # sd branch forward
            z0_pred, _, sigmas = self.sd_forward_step(
                image_lq=0.5*input_sd+0.5,
                model_pred=None,
                timesteps=tt.repeat(4),
            )
            x0_sd=self.decode_first_stage_withgrad(z0_pred)
            x0_sd=util_pol.chunkbatch2dim(x0_sd)
            x0_sd=0.5*x0_sd+0.5
            latent_gt=micro_data['gt']
            total_loss=0
            if loss_coef.get('lpix', 0) > 0:
                if self.configs.train.loss_type == 'L2':
                    lsd_loss=F.mse_loss(x0_sd, latent_gt, reduction='none')
                elif self.configs.train.loss_type == 'L1':
                    lsd_loss=F.l1_loss(x0_sd, latent_gt, reduction='none')
                else:
                    raise TypeError(f"Unsupported Loss type for Diffusion: {self.configs.train.loss_type}")
                lsd_loss=torch.mean(lsd_loss, dim=list(range(1, z0_pred.ndim)))
                losses['lpix']=lsd_loss*loss_coef['lpix']
                total_loss+=losses['lpix'].mean()
            if loss_coef.get('lpol', 0) > 0:
                gt_S0,gt_dolp,_=util_pol.I2S012(micro_data['gt'],save_img=True,use_loss=True,clip=True,improve_contrast=False)
                sd_S0,sd_dolp,_=util_pol.I2S012(x0_sd,save_img=True,use_loss=True,clip=True,improve_contrast=False)
                lsdp_loss=F.l1_loss(gt_S0, sd_S0, reduction='none')+F.l1_loss(gt_dolp, sd_dolp, reduction='none')
                lsdp_loss=torch.mean(lsdp_loss, dim=list(range(1, x0_sd.ndim)))
                losses['lpol']=+lsdp_loss*loss_coef['lpol']
                total_loss+=losses['lpol'].mean()
        if self.amp_scaler is None:
            total_loss.backward()
        else:
            self.amp_scaler.scale(total_loss).backward()
        return losses,x0_base,x0_sd,tt
    def backward_step_onlypsrnet(self, micro_data):
        loss_coef = self.configs.train.loss_coef
        losses = {}
        tt = torch.tensor(
            random.choices(self.configs.train.timesteps, k=micro_data['lq'].shape[0]),
            dtype=torch.int64,
            device=f"cuda:{self.rank}",
        ) - 1
        with torch.autocast(device_type="cuda", enabled=self.configs.train.use_amp):
            micro_data['lq']=2*micro_data['lq']-1
            # base branch
            eps_pred=self.psr_net(micro_data['lq'],tt)
            if isinstance(eps_pred, list):
                x0_pred=micro_data['lq']-eps_pred[0]
            else:
                x0_pred=micro_data['lq']-eps_pred
            x0_pred=x0_pred.clamp(-1,1)
            x0_pred=0.5*x0_pred+0.5
            total_loss=0
            if loss_coef.get('lpix', 0) > 0:
                if self.configs.train.loss_type == 'L2':
                    ldif_loss_12dim=F.mse_loss(x0_pred, micro_data['gt'], reduction='none')
                elif self.configs.train.loss_type == 'L1':
                    ldif_loss_12dim=F.l1_loss(x0_pred, micro_data['gt'], reduction='none')
                else:
                    raise TypeError(f"Unsupported Loss type for Diffusion: {self.configs.train.loss_type}")
                ldif_loss_12dim = torch.mean(ldif_loss_12dim, dim=list(range(1, x0_pred.ndim)))
                losses['lpix']=ldif_loss_12dim*loss_coef['lpix']
                total_loss+=losses['lpix'].mean()
            if loss_coef.get('lpol', 0) > 0:
                gt_S0,gt_dolp,_=util_pol.I2S012(micro_data['gt'],save_img=True,use_loss=True,clip=True,improve_contrast=False)
                pred_S0,pred_dolp,_=util_pol.I2S012(x0_pred,save_img=True,use_loss=True,clip=True,improve_contrast=False)
                lpol_loss=F.l1_loss(gt_S0, pred_S0, reduction='none')+F.l1_loss(gt_dolp, pred_dolp, reduction='none')
                lpol_loss=torch.mean(lpol_loss, dim=list(range(1, x0_pred.ndim)))
                losses['lpol']=lpol_loss*loss_coef['lpol']
                total_loss+=losses['lpol'].mean()
            if loss_coef.get('lvar', 0) > 0:
                _,dolp_pred,_=util_pol.I2S012(x0_pred,save_img=True,use_loss=True,clip=True,improve_contrast=False)
                s = torch.exp(-2*eps_pred[1])
                sr_=dolp_pred
                hr_=micro_data['dolp_gt']
                lvar_loss=0.5*s*F.mse_loss(sr_, hr_, reduction='none')+2*eps_pred[1]
                add=0.5*(torch.log(dolp_pred+1e-5)-torch.log(micro_data['dolp_gt']+1e-5))
                lvar_loss=lvar_loss+add
                lvar_loss=torch.mean(lvar_loss, dim=list(range(1, x0_pred.ndim)))
                losses['lvar']=lvar_loss*loss_coef['lvar']
                total_loss+=losses['lvar'].mean()
        if self.amp_scaler is None:
            total_loss.backward()
        else:
            self.amp_scaler.scale(total_loss).backward()
        return losses,x0_pred, tt
    def backward_step_fusion(self, micro_data):
        loss_coef = self.configs.train.loss_coef
        losses = {}
        tt = torch.tensor(
            random.choices(self.configs.train.timesteps, k=micro_data['lq'].shape[0]),
            dtype=torch.int64,
            device=f"cuda:{self.rank}",
        ) - 1
        with torch.autocast(device_type="cuda", enabled=self.configs.train.use_amp):
            input_base=2*micro_data['lq']-1
            # base branch
            with torch.no_grad():
                eps_pred=self.psr_net(input_base,tt)
            if isinstance(eps_pred, list):
                x0_base=input_base-eps_pred[0]
            else:
                x0_base=input_base-eps_pred
            input_sd=util_pol.chunkdim2batch(x0_base)
            # sd branch forward
            with torch.no_grad():
                z0_pred, _, sigmas = self.sd_forward_step(
                    image_lq=0.5*input_sd+0.5,
                    model_pred=None,
                    timesteps=tt.repeat(4),
                )
                x0_sd=self.decode_first_stage(z0_pred)
            x0_base=x0_base.clamp(-1,1)
            x0_sd=util_pol.chunkbatch2dim(x0_sd)
            # fusion module
            if self.configs.get('use_fusion',False):
                x0_pred=self.fusion_net(x0_sd=x0_sd,x0_base=x0_base)
                x0_pred=x0_pred.clamp(-1,1)
                x0_pred=0.5*x0_pred+0.5
            # cal uncertainty
            with torch.no_grad():
                eps_teacher=self.teacher(input_base,tt)
                eps_teacher[1]=util_pol.rgb_to_ycbcr(eps_teacher[1])[:,0:1,:,:]
                b, c, _, _ = eps_teacher[1].shape
                s1=eps_teacher[1].view(b, c, -1)
                pmin = torch.min(s1, dim=-1)
                pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                pmax = torch.max(s1, dim=-1)
                pmax = pmax[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                s_teacher = eps_teacher[1]
                s_teacher=(s_teacher - pmin) / (pmax - pmin + 1e-6)
            total_loss=0
            if loss_coef.get('lpix', 0) > 0:
                if self.configs.train.loss_type == 'L2':
                    ldif_loss=F.mse_loss(x0_pred, micro_data['gt'], reduction='none')
                    lsd_loss=F.mse_loss(0.5*x0_sd+0.5, micro_data['gt'], reduction='none')
                elif self.configs.train.loss_type == 'L1':
                    ldif_loss=F.l1_loss(x0_pred, micro_data['gt'], reduction='none')
                    lsd_loss=F.l1_loss(0.5*x0_sd+0.5, micro_data['gt'], reduction='none')
                else:
                    raise TypeError(f"Unsupported Loss type for Diffusion: {self.configs.train.loss_type}")
                ldif_loss = torch.mean(ldif_loss, dim=list(range(1, x0_pred.ndim)))
                lsd_loss=torch.mean(lsd_loss, dim=list(range(1, x0_pred.ndim)))
                losses['lpix']=ldif_loss*loss_coef['lpix']+lsd_loss*loss_coef['lpix']*0.2
                total_loss+=losses['lpix'].mean()
            if loss_coef.get('lpol', 0) > 0:
                gt_S0,gt_dolp,_=util_pol.I2S012(micro_data['gt'],save_img=True,use_loss=True,clip=True,improve_contrast=False)
                pred_S0,pred_dolp,_=util_pol.I2S012(x0_pred,save_img=True,use_loss=True,clip=True,improve_contrast=False)
                sd_S0,sd_dolp,_=util_pol.I2S012(0.5*x0_sd+0.5,save_img=True,use_loss=True,clip=True,improve_contrast=False)
                lpol_loss=F.mse_loss(gt_S0, pred_S0, reduction='none')+F.mse_loss(gt_dolp, pred_dolp, reduction='none')
                lsdp_loss=F.mse_loss(sd_S0, pred_S0, reduction='none')+F.mse_loss(sd_dolp, pred_dolp, reduction='none')
                lpol_loss=(1-s_teacher)*lpol_loss
                lsdp_loss=s_teacher*lsdp_loss
                lpol2_loss=lpol_loss+lsdp_loss
                lpol2_loss=torch.mean(lpol2_loss, dim=list(range(1, x0_pred.ndim)))
                losses['lpol']=lpol2_loss*loss_coef['lpol']
                total_loss+=losses['lpol'].mean()
            if loss_coef.get('lself', 0) > 0:
                lsd_loss=s_teacher*F.mse_loss(x0_pred, 0.5*x0_sd+0.5, reduction='none')
                lbase_loss=(1-s_teacher)*F.mse_loss(x0_pred, 0.5*x0_base+0.5, reduction='none')
                lself_loss=lsd_loss+lbase_loss
                lself_loss=torch.mean(lself_loss, dim=list(range(1, x0_pred.ndim)))
                losses['lself']=lself_loss*loss_coef['lself']
                total_loss+=losses['lself'].mean()
        if self.amp_scaler is None:
            total_loss.backward()
        else:
            self.amp_scaler.scale(total_loss).backward()
        return losses, x0_pred,x0_base,x0_sd,tt
    def sd_forward_step(
        self,
        image_lq: torch.Tensor = None,
        model_pred: DiagonalGaussianDistribution = None,
        timesteps: List[int] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image_lq (`torch.Tensor`): The low-quality image(s) for enhancement, range in [0, 1].
            image_hq (`torch.Tensor`): The high-quality image(s) for enhancement, range in [0, 1].
            noise_pred (`torch.Tensor`): Predicted noise by the noise prediction model
            latents_hq (`torch.Tensor`, *optional*):
                Pre-generated high-quality latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation.  If not provided, a latents tensor will be generated by sampling using vae .
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
        """
        device=torch.device(f"cuda:{self.rank}")
        prompt_embeds=torch.zeros_like(image_lq)
        if not hasattr(self, 'sigmas_cache'):
            assert self.sd_pipe.scheduler.sigmas.numel() == (self.configs.sd_pipe.num_train_steps + 1)
            self.sigmas_cache = self.sd_pipe.scheduler.sigmas.flip(0)[1:].to(device) #ascending,1000
        sigmas = self.sigmas_cache[timesteps] # (b,)

        # Prepare input for SD

        image_lq_up = image_lq
        zt_clean = self.encode_first_stage_withgrad(
            image_lq_up, center_input_sample=True,
            deterministic=True,
        )

        sigmas = append_dims(sigmas, zt_clean.ndim)
        zt_noisy = self.add_noise(zt_clean, sigmas, model_pred)

        prompt_embeds = prompt_embeds.to(device)
        # prompt_embeds=None
        zt_noisy_scale = self.scale_sd_input(zt_noisy, sigmas)
        eps_pred = self.sd_pipe.unet(
            zt_noisy_scale,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
            lq=image_lq
        )   # eps-mode for sdxl and sdxl-refiner

        if self.configs.train.noise_detach:
            z0_pred = zt_noisy.detach() - sigmas * eps_pred[0]
        else:
            z0_pred = zt_noisy - sigmas * eps_pred[0]
 
        return z0_pred, zt_noisy,sigmas
    
    @torch.no_grad()
    def test(self):
        torch.cuda.empty_cache()
        crop_size=self.configs.data.test.crop_size
        tfs=transforms.Compose([transforms.CenterCrop([crop_size,crop_size]), transforms.ToTensor()
        ])
        transforms.CenterCrop([crop_size,crop_size])
        cpfa_path=self.configs.data.test.cpfa_path
        name_list=os.listdir(cpfa_path)
        name_list.sort()
        self.logger.info(f"Testing on {len(name_list)} images")
        for name in name_list:
            xt_progressive = []
            cpfa=tfs(Image.open(os.path.join(cpfa_path,name)))
            cpfa=cpfa.unsqueeze(dim=0).cuda()
            lq=util_pol.init_CPDM(cpfa[:,0,:,:])
            tt = torch.tensor(
                random.choices(self.configs.train.timesteps, k=lq.shape[0]),
                dtype=torch.int64,
                device=f"cuda:{self.rank}",
                    ) - 1
            with torch.amp.autocast('cuda'):
                if self.configs.get('use_sd', True):
                    input_base=2*lq-1
                    eps_pred=self.psr_net(input_base,tt)
                    if isinstance(eps_pred, list):
                        x0_base=input_base-eps_pred[0]
                        eps_pred[1]=util_pol.rgb_to_ycbcr(eps_pred[1])[:,0:1,:,:]
                        b, c, _, _ = eps_pred[1].shape
                        s1=eps_pred[1].view(b, c, -1)
                        pmin = torch.min(s1, dim=-1)
                        pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                        pmax = torch.max(s1, dim=-1)
                        pmax = pmax[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                        s = eps_pred[1]
                        s=(s - pmin) / (pmax - pmin + 1e-6)
                    else:
                        x0_base=input_base-eps_pred
                    input_sd=util_pol.chunkdim2batch(x0_base)
                    z0_pred, _,sigmas = self.sd_forward_step(
                        image_lq=0.5*input_sd+0.5,
                        model_pred=None,
                        timesteps=tt.repeat(4),
                    )
                    x0_sd=self.decode_first_stage(z0_pred)
                    x0_sd=util_pol.chunkbatch2dim(x0_sd)
                    x0_base=x0_base.clamp(-1,1)
                    if self.configs.get('use_fusion'):
                        x0_pred=self.fusion_net(x0_sd=x0_sd,x0_base=x0_base)
                        x0_pred=x0_pred.clamp(-1,1)
                    else:
                        x0_pred=x0_sd
                    xt_progressive.append(x0_pred)
                else:
                    lq=2*lq-1
                    eps_pred=self.ema_model(lq,tt)
                    if isinstance(eps_pred, list):
                        x0_pred=lq-eps_pred[0]
                    else:
                        x0_pred=lq-eps_pred
                    x0_pred=x0_pred.clamp(-1,1)
                    xt_progressive.append(x0_pred)
                    if isinstance(eps_pred, list):
                        s=util_pol.rgb_to_ycbcr(eps_pred[1])[:,0:1,:,:]
                        for i in range(s.shape[0]):
                            s[i:i+1,:,:,:] = util_pol.draw_features(s[i:i+1,:,:,:])
                if self.use_teacher:
                    eps_teacher=self.teacher(input_base,tt)
                    s_teacher=util_pol.rgb_to_ycbcr(eps_teacher[1])[:,0:1,:,:]
                    for i in range(s_teacher.shape[0]):
                        s_teacher[i:i+1,:,:,:] = util_pol.draw_features(s_teacher[i:i+1,:,:,:])
            x0 = xt_progressive[-1]
            num_inference_steps = len(xt_progressive)
            x0_normalize = util_image.normalize_th(x0, mean=0.5, std=0.5, reverse=True)
            S0_pred,DOLP_pred,_=util_pol.I2S012(x0_normalize,save_img=True,use_loss=False,improve_contrast=False)
            if isinstance(eps_pred, list):
                self.logging_image(
                    s,
                    tag='var',
                    phase='test',
                    add_global_step=False,
                    nrow=num_inference_steps,
                    colormap=cv2.COLORMAP_JET
                )
                if self.use_teacher:
                        self.logging_image(
                            s_teacher,
                            tag='var_teacher',
                            phase='test',
                            add_global_step=False,
                            nrow=num_inference_steps,
                            colormap=cv2.COLORMAP_JET
                        )
            if self.configs.get('use_sd', True):
                S0_sd,DOLP_sd,_=util_pol.I2S012(0.5*x0_sd+0.5,save_img=True,use_loss=False,improve_contrast=False)
                S0_base,DOLP_base,_=util_pol.I2S012(0.5*x0_base+0.5,save_img=True,use_loss=False,improve_contrast=False)
                self.logging_image(
                    S0_sd,
                    tag='s0_sd',
                    phase='test',
                    add_global_step=False,
                    nrow=num_inference_steps,
                )
                self.logging_image(
                    S0_base,
                    tag='s0_base',
                    phase='test',
                    add_global_step=False,
                    nrow=num_inference_steps,
                )
                self.logging_image(
                DOLP_sd,
                tag='dolp_sd',
                phase='test',
                add_global_step=False,
                nrow=num_inference_steps,
                )
                self.logging_image(
                    DOLP_base,
                    tag='dolp_base',
                    phase='test',
                    add_global_step=False,
                    nrow=num_inference_steps,
                )
            self.logging_image(
                S0_pred,
                tag='s0',
                phase='test',
                add_global_step=False,
                nrow=num_inference_steps,
            )
            self.logging_image(
                DOLP_pred,
                tag='dolp',
                phase='test',
                add_global_step=True,
                nrow=num_inference_steps,
            )
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validation(self, phase='val'):
        torch.cuda.empty_cache()
        num_iters_epoch = math.ceil(len(self.datasets[phase]) / self.configs.validate.batch)
        mean_psnr = 0
        mean_psnr_S0=0
        mean_psnr_DOLP=0
        for jj, data in enumerate(self.dataloaders[phase]):
            xt_progressive = []
            data = self.prepare_data(data, phase='val')
            tt = torch.tensor(
            random.choices(self.configs.train.timesteps, k=data['lq'].shape[0]),
            dtype=torch.int64,
            device=f"cuda:{self.rank}",
                ) - 1
            with torch.amp.autocast('cuda'):
                if self.configs.get('use_sd', True):
                    input_base=2*data['lq']-1
                    eps_pred=self.psr_net(input_base,tt)
                    if isinstance(eps_pred, list):
                        x0_base=input_base-eps_pred[0]
                        eps_pred[1]=util_pol.rgb_to_ycbcr(eps_pred[1])[:,0:1,:,:]
                        b, c, _, _ = eps_pred[1].shape
                        s1=eps_pred[1].view(b, c, -1)
                        pmin = torch.min(s1, dim=-1)
                        pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                        pmax = torch.max(s1, dim=-1)
                        pmax = pmax[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                        s = eps_pred[1]
                        s=(s - pmin) / (pmax - pmin + 1e-6)
                    else:
                        x0_base=input_base-eps_pred
                    input_sd=util_pol.chunkdim2batch(x0_base)
                    z0_pred, _,sigmas = self.sd_forward_step(
                        image_lq=0.5*input_sd+0.5,
                        model_pred=None,
                        timesteps=tt.repeat(4),
                    )
                    x0_sd=self.decode_first_stage(z0_pred)
                    x0_sd=util_pol.chunkbatch2dim(x0_sd)
                    x0_base=x0_base.clamp(-1,1)
                    if self.configs.get('use_fusion'):
                        x0_pred=self.fusion_net(x0_sd=x0_sd,x0_base=x0_base)
                        x0_pred=x0_pred.clamp(-1,1)
                    else:
                        x0_pred=x0_sd
                    xt_progressive.append(x0_pred)
                else:
                    data['lq']=2*data['lq']-1
                    eps_pred=self.ema_model(data['lq'],tt)
                    if isinstance(eps_pred, list):
                        x0_pred=data['lq']-eps_pred[0]
                    else:
                        x0_pred=data['lq']-eps_pred
                    x0_pred=x0_pred.clamp(-1,1)
                    xt_progressive.append(x0_pred)
                    if isinstance(eps_pred, list):
                        s=util_pol.rgb_to_ycbcr(eps_pred[1])[:,0:1,:,:]
                        for i in range(s.shape[0]):
                            s[i:i+1,:,:,:] = util_pol.draw_features(s[i:i+1,:,:,:])
                if self.use_teacher:
                    eps_teacher=self.teacher(2*data['lq']-1,tt)
                    s_teacher=util_pol.rgb_to_ycbcr(eps_teacher[1])[:,0:1,:,:]
                    for i in range(s_teacher.shape[0]):
                        s_teacher[i:i+1,:,:,:] = util_pol.draw_features(s_teacher[i:i+1,:,:,:])
            x0 = xt_progressive[-1]
            num_inference_steps = len(xt_progressive)

            if 'gt' in data:
                if not hasattr(self, 'psnr_metric'):
                    self.psnr_metric = pyiqa.create_metric(
                            'psnr',
                            test_y_channel=self.configs.train.get('val_y_channel', True),
                            color_space='rgb',
                            device=torch.device("cuda"),
                            )
                x0_normalize = util_image.normalize_th(x0, mean=0.5, std=0.5, reverse=True)
                S0_pred,DOLP_pred,_=util_pol.I2S012(x0_normalize,save_img=True,use_loss=False,improve_contrast=False)
                S0_gt,DOLP_gt,_=util_pol.I2S012(data['gt'],save_img=True,use_loss=False,improve_contrast=False)
                x0_normalize_12dim=util_pol.chunkdim2batch(x0_normalize)
                gt_12dim=util_pol.chunkdim2batch(data['gt'])
                mean_psnr += self.psnr_metric(x0_normalize_12dim, gt_12dim).sum().item()
                mean_psnr_S0+=self.psnr_metric(S0_pred, S0_gt).sum().item()
                mean_psnr_DOLP+=self.psnr_metric(DOLP_pred, DOLP_gt).sum().item()

            if (jj + 1) % self.configs.validate.log_freq == 0:
                self.logger.info(f'Validation: {jj+1:02d}/{num_iters_epoch:02d}...')

                if isinstance(eps_pred, list):
                    self.logging_image(
                        s,
                        tag='var',
                        phase=phase,
                        add_global_step=False,
                        nrow=num_inference_steps,
                        colormap=cv2.COLORMAP_JET
                    )
                    if self.use_teacher:
                            self.logging_image(
                                s_teacher,
                                tag='var_teacher',
                                phase=phase,
                                add_global_step=False,
                                nrow=num_inference_steps,
                                colormap=cv2.COLORMAP_JET
                            )
                if self.configs.get('use_sd', True):
                    S0_sd,DOLP_sd,_=util_pol.I2S012(0.5*x0_sd+0.5,save_img=True,use_loss=False,improve_contrast=False)
                    S0_base,DOLP_base,_=util_pol.I2S012(0.5*x0_base+0.5,save_img=True,use_loss=False,improve_contrast=False)
                    self.logging_image(
                        S0_sd,
                        tag='s0_sd',
                        phase=phase,
                        add_global_step=False,
                        nrow=num_inference_steps,
                    )
                    self.logging_image(
                        S0_base,
                        tag='s0_base',
                        phase=phase,
                        add_global_step=False,
                        nrow=num_inference_steps,
                    )
                    self.logging_image(
                    DOLP_sd,
                    tag='dolp_sd',
                    phase=phase,
                    add_global_step=False,
                    nrow=num_inference_steps,
                    )
                    self.logging_image(
                        DOLP_base,
                        tag='dolp_base',
                        phase=phase,
                        add_global_step=False,
                        nrow=num_inference_steps,
                    )
                self.logging_image(
                    S0_pred,
                    tag='s0',
                    phase=phase,
                    add_global_step=False,
                    nrow=num_inference_steps,
                )
                self.logging_image(
                    DOLP_pred,
                    tag='dolp',
                    phase=phase,
                    add_global_step=True,
                    nrow=num_inference_steps,
                )
        if 'gt' in data:
            mean_psnr /= 4*len(self.datasets[phase])
            mean_psnr_S0/=len(self.datasets[phase])
            mean_psnr_DOLP/=len(self.datasets[phase])
            self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}')
            self.logger.info(f'Validation Metric: PSNR_S0={mean_psnr_S0:5.2f}')
            self.logger.info(f'Validation Metric: PSNR_DOLP={mean_psnr_DOLP:5.2f}')
            if mean_psnr> self.best_psnr:
                self.best_psnr = mean_psnr
                self.logger.info('find new best model at iter {:d}'.format(self.current_iters))
                self.best_model_name='model_{:d}'.format(self.current_iters)
            else:
                 self.logger.info('The best model is '+self.best_model_name)
            self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
            self.logging_metric(mean_psnr_S0, tag='PSNR_S0', phase=phase, add_global_step=False)
            self.logging_metric(mean_psnr_DOLP, tag='PSNR_DOLP', phase=phase, add_global_step=True)

        self.logger.info("="*100)
        torch.cuda.empty_cache()

    @torch.amp.autocast('cuda')
    def encode_first_stage_withgrad(self, x, deterministic=True, center_input_sample=True):
        if center_input_sample:
            x = x * 2.0 - 1.0
        latents_mean = latents_std = None
        if hasattr(self.sd_pipe.vae.config, "latents_mean") and self.sd_pipe.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.sd_pipe.vae.config.latents_mean).view(1, -1, 1, 1)
        if hasattr(self.sd_pipe.vae.config, "latents_std") and self.sd_pipe.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.sd_pipe.vae.config.latents_std).view(1, -1, 1, 1)

        if deterministic:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.mode()
        else:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.sample()

        trunk_size = self.configs.sd_pipe.vae_split
        if trunk_size < x.shape[0]:
            init_latents = torch.cat([partial_encode(xx) for xx in x.split(trunk_size, 0)], dim=0)
        else:
            init_latents = partial_encode(x)

        scaling_factor = self.sd_pipe.vae.config.scaling_factor
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=x.device, dtype=x.dtype)
            latents_std = latents_std.to(device=x.device, dtype=x.dtype)
            init_latents = (init_latents - latents_mean) * scaling_factor / latents_std
        else:
            init_latents = init_latents * scaling_factor

        return init_latents
    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def encode_first_stage(self, x, deterministic=True, center_input_sample=True):
        if center_input_sample:
            x = x * 2.0 - 1.0
        latents_mean = latents_std = None
        if hasattr(self.sd_pipe.vae.config, "latents_mean") and self.sd_pipe.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.sd_pipe.vae.config.latents_mean).view(1, -1, 1, 1)
        if hasattr(self.sd_pipe.vae.config, "latents_std") and self.sd_pipe.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.sd_pipe.vae.config.latents_std).view(1, -1, 1, 1)

        if deterministic:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.mode()
        else:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.sample()

        trunk_size = self.configs.sd_pipe.vae_split
        if trunk_size < x.shape[0]:
            init_latents = torch.cat([partial_encode(xx) for xx in x.split(trunk_size, 0)], dim=0)
        else:
            init_latents = partial_encode(x)

        scaling_factor = self.sd_pipe.vae.config.scaling_factor
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=x.device, dtype=x.dtype)
            latents_std = latents_std.to(device=x.device, dtype=x.dtype)
            init_latents = (init_latents - latents_mean) * scaling_factor / latents_std
        else:
            init_latents = init_latents * scaling_factor

        return init_latents

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def decode_first_stage(self, z, clamp=True):
        decode = self.sd_pipe.vae.decode
        z = z / self.sd_pipe.vae.config.scaling_factor
        trunk_size = 1
        if trunk_size < z.shape[0]:
            out = torch.cat(
                [decode(xx).sample for xx in z.split(trunk_size, 0)], dim=0,
            )
        else:
            out = decode(z).sample
        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

    @torch.amp.autocast('cuda')
    def decode_first_stage_withgrad(self, z, clamp=True):
        z = z / self.sd_pipe.vae.config.scaling_factor

        trunk_size = 1
        if trunk_size < z.shape[0]:
            out = torch.cat(
                [self.sd_pipe.vae.decode(xx).sample for xx in z.split(trunk_size, 0)], dim=0,
            )
        else:
            out = self.sd_pipe.vae.decode(z).sample
        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

    def training_step(self, data):
        current_bs = data['gt'].shape[0]
        micro_bs = self.configs.train.microbatch

        for jj in range(0, current_bs, micro_bs):
            micro_data=data
            if self.configs.get('use_unet', True):
                if not self.configs.get('use_sd', True):
                    losses,x0_pred,tt=self.backward_step_onlypsrnet(micro_data)
                    if self.configs.train.use_amp:
                        self.amp_scaler.step(self.optimizer)
                        self.amp_scaler.update()
                        self.optimizer.zero_grad()
                else:
                    losses, x0_pred, x0_base,x0_sd,tt = self.backward_step_fusion(micro_data)
                    if self.configs.train.use_amp:
                        self.amp_scaler.step(self.optimizer)
                        self.amp_scaler.update()
                        self.optimizer.zero_grad()
            else:
                losses, x0_base,x0_sd,tt = self.backward_step_onlylora(micro_data)
                if self.configs.train.use_amp:
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                    self.optimizer.zero_grad()
        # make logging
        if self.rank == 0:
            if self.configs.get('use_unet', True):
                if not self.configs.get('use_sd', True):
                    self.log_step_train(
                        losses, tt, micro_data,x0_pred=x0_pred,  x0_base=None,x0_sd=None
                    )
                else:
                    self.log_step_train(
                        losses, tt, micro_data,x0_pred=x0_pred, x0_base=x0_base,x0_sd=x0_sd
                    )
            else:
                self.log_step_train(
                    losses, tt, micro_data,x0_pred=None, x0_base=x0_base,x0_sd=x0_sd
                )
            
    def scale_sd_input(
        self,
        x:torch.Tensor,
        sigmas: torch.Tensor = None,
        timestep: torch.Tensor = None,
    ) :
        if sigmas is None:
            if not self.sd_pipe.scheduler.sigmas.numel() == (self.configs.sd_pipe.num_train_steps + 1):
                self.sd_pipe.scheduler = EulerDiscreteScheduler.from_pipe(
                    self.configs.sd_pipe.params.pretrained_model_name_or_path,
                    cache_dir=self.configs.sd_pipe.params.cache_dir,
                    subfolder='scheduler',
                )
                assert self.sd_pipe.scheduler.sigmas.numel() == (self.configs.sd_pipe.num_train_steps + 1)
            sigmas = self.sd_pipe.scheduler.sigmas.flip(0).to(x.device)[timestep] # (b,)
            sigmas = append_dims(sigmas, x.ndim)

        if sigmas.ndim < x.ndim:
            sigmas = append_dims(sigmas, x.ndim)
        out = x / ((sigmas**2 + 1) ** 0.5)
        return out
