
import math, random
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from utils.util_ops import append_dims
from utils import util_net, util_pol
from utils import util_image
from utils import util_common
from utils import util_color_fix
from peft import LoraConfig
import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from datapipe.datasets import create_dataset
from diffusers import StableDiffusionInvEnhancePipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from basicsr.arch.unet import CorrectNet

class BaseSampler:
    def __init__(self, configs):

        self.configs = configs

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def write_log(self, log_str):
        print(log_str, flush=True)

    def build_model(self):
        # Build Stable diffusion
        self.device=torch.device(f"cuda:{self.configs.device_id}")
        params = dict(self.configs.sd_pipe.params)
        torch_dtype = params.pop('torch_dtype')
        params['torch_dtype'] = get_torch_dtype(torch_dtype)
        base_pipe = util_common.get_obj_from_str(self.configs.sd_pipe.target).from_pretrained(**params)
        if self.configs.get('scheduler', None) is not None:
            pipe_id = self.configs.scheduler.target.split('.')[-1]
            self.write_log(f'Loading scheduler of {pipe_id}...')
            base_pipe.scheduler = util_common.get_obj_from_str(self.configs.scheduler.target).from_config(
                base_pipe.scheduler.config
            )
            self.write_log('Loaded Done')
        if self.configs.get('vae_fp16', None) is not None:
            params_vae = dict(self.configs.vae_fp16.params)
            torch_dtype = params_vae.pop('torch_dtype')
            params_vae['torch_dtype'] = get_torch_dtype(torch_dtype)
            pipe_id = self.configs.vae_fp16.params.pretrained_model_name_or_path
            self.write_log(f'Loading improved vae from {pipe_id}...')
            base_pipe.vae = util_common.get_obj_from_str(self.configs.vae_fp16.target).from_pretrained(
                **params_vae,
            )
            self.write_log('Loaded Done')
        if self.configs.base_model in ['sd-turbo', 'sd2base'] :
            sd_pipe = StableDiffusionInvEnhancePipeline.from_pipe(base_pipe)
        else:
            raise ValueError(f"Unsupported base model: {self.configs.base_model}!")
        if self.configs.sd_pipe.lora_ckpt_path is not None:
            lora_ckpt=torch.load(self.configs.sd_pipe.lora_ckpt_path)
            lora_conf_encoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["unet_lora_encoder_modules"])
            lora_conf_decoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["unet_lora_decoder_modules"])
            lora_conf_others = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["unet_lora_others_modules"])
            sd_pipe.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            sd_pipe.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
            sd_pipe.unet.add_adapter(lora_conf_others, adapter_name="default_others")
            for n, p in sd_pipe.unet.named_parameters():
                if "lora" in n or "conv_in" in n:
                    p.data.copy_(lora_ckpt["state_dict_unet"][n])
            sd_pipe.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

            # load vae lora
            vae_lora_conf_encoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["vae_lora_encoder_modules"])
            sd_pipe.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
            for n, p in sd_pipe.vae.named_parameters():
                if "lora" in n:
                    p.data.copy_(lora_ckpt["state_dict_vae"][n])
            sd_pipe.vae.set_adapter(['default_encoder'])
            sd_pipe.vae = sd_pipe.vae.merge_and_unload()
            sd_pipe.unet = sd_pipe.unet.merge_and_unload()
            if self.configs.use_fusion:
                self.write_log(f"Loading Fusion net from {self.configs.sd_pipe.lora_ckpt_path}...")
                self.fusion_net=CorrectNet().to(self.device)
                lora_ckpt=torch.load(self.configs.sd_pipe.lora_ckpt_path)
                util_net.reload_model(self.fusion_net, lora_ckpt['state_dict_correct'])
        sd_pipe.to(self.device)
        if self.configs.sliced_vae:
            sd_pipe.vae.enable_slicing()
        if self.configs.tiled_vae:
            sd_pipe.vae.enable_tiling()
            sd_pipe.vae.tile_latent_min_size = self.configs.latent_tiled_size
            sd_pipe.vae.tile_sample_min_size = self.configs.sample_tiled_size
        if self.configs.gradient_checkpointing_vae:
            self.write_log(f"Activating gradient checkpoing for vae...")
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.encoder, True)
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.decoder, True)
        params = dict(self.configs.unet.params)
        unet = util_common.get_obj_from_str(self.configs.unet.target)(**params)
        unet=unet.to(self.device)
        ckpt_path = self.configs.unet.get('ckpt_path')
        assert ckpt_path is not None
        self.write_log(f"Loading started model from {ckpt_path}...")
        state = torch.load(ckpt_path)
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(unet, state)
        self.write_log(f"Loading Done")
        unet.eval()
        self.unet=unet
        self.sd_pipe = sd_pipe

class SamplerPSR(BaseSampler):
    def inference(self, config, bs=1):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''

        # in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        # out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
        if config.data.input.type=="CPDM":
            in_path=Path(config.data.get('input', dict).params.data_source.source1.dataroot_gt_0)
        elif config.data.input.type=="RealCPDM":
            in_path=Path(config.data.get('input', dict).params.dataroot_gt)
        out_path_0=Path(config.get('output', dict).output_path_0)
        out_path_45=Path(config.get('output', dict).output_path_45)
        out_path_90=Path(config.get('output', dict).output_path_90)
        out_path_135=Path(config.get('output', dict).output_path_135)
        out_path_S0=Path(config.get('output', dict).output_path_S0)
        out_path_DOLP=Path(config.get('output', dict).output_path_DOLP)
        out_path_AOP=Path(config.get('output', dict).output_path_AOP)
        if not out_path_0.exists():
            out_path_0.mkdir(parents=True)
            out_path_45.mkdir(parents=True)
            out_path_90.mkdir(parents=True)
            out_path_135.mkdir(parents=True)
            out_path_S0.mkdir(parents=True)
            out_path_AOP.mkdir(parents=True)
            out_path_DOLP.mkdir(parents=True)
        if in_path.is_dir():
            dataset=create_dataset(config.data.get('input', dict))
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=False, drop_last=False,
            )
            for data in dataloader:
                batch=self.prepare_data(data)
                res_sr_0,res_sr_45,res_sr_90,res_sr_135,res_sr_S0,res_sr_DOLP,res_sr_AOP= self.sample_func(im_cond=batch)
                for jj in range(res_sr_0.shape[0]):
                    im_name = Path(data['lq_path'][jj]).stem
                    base_name=im_name.replace('_0','')
                    save_path_0 = str(out_path_0 / f"{base_name}_0.png")
                    save_path_45 = str(out_path_45 / f"{base_name}_45.png")
                    save_path_90 = str(out_path_90 / f"{base_name}_90.png")
                    save_path_135 = str(out_path_135 / f"{base_name}_135.png")
                    save_path_S0 = str(out_path_S0 / f"{base_name}_S0.png")
                    save_path_DOLP = str(out_path_DOLP / f"{base_name}_DOLP.png")
                    save_path_AOP = str(out_path_AOP / f"{base_name}_AOP.png")
                    util_image.imwrite(res_sr_0[jj], save_path_0, dtype_in='float32')
                    util_image.imwrite(res_sr_45[jj], save_path_45, dtype_in='float32')
                    util_image.imwrite(res_sr_90[jj], save_path_90, dtype_in='float32')
                    util_image.imwrite(res_sr_135[jj], save_path_135, dtype_in='float32')
                    util_image.imwrite(res_sr_S0[jj], save_path_S0, dtype_in='float32')
                    util_image.imwrite(res_sr_DOLP[jj], save_path_DOLP, dtype_in='float32')
                    util_image.imwrite(res_sr_AOP[jj], save_path_AOP, dtype_in='float32')
        self.write_log(f"Processing done, enjoy the results in {str(out_path_0)}")
    def prepare_data(self, data,):
        if self.configs.data.input.type=="CPDM":
            crop_size=self.configs.crop_size
            tfs=transforms.CenterCrop([crop_size,crop_size])
            data['gt']=tfs(data['gt'])
            if not self.configs.data.input.params.data_source.source1.use_cpfa:
                self.cpfa=util_pol.Gen_CPFA(data['gt'])
            else:
                self.cpfa=data['cpfa']
            if not self.configs.data.input.params.data_source.source1.use_lq:
                self.lq=util_pol.init_CPDM(self.cpfa)
            else:
                self.lq=data['lq']
            self.gt=data['gt']
            gt_clone=data['gt'].clone()
            lq_clone=self.lq.clone()
            self.gt_3dim = util_pol.chunkdim2batch(gt_clone)
            self.lq_3dim=util_pol.chunkdim2batch(lq_clone)
            self.lq_downsample=util_pol.CPFA_downsample(self.cpfa[:,0:1,:,:])
        elif self.configs.data.input.type=="RealCPDM":
            self.cpfa=data['lq'].clone()
            crop_size=self.configs.crop_size
            tfs=transforms.CenterCrop([crop_size,crop_size])
            self.cpfa=tfs(self.cpfa)
            self.lq=util_pol.init_CPDM(self.cpfa[:,0,:,:])
            lq_clone=self.lq.clone()
            self.gt=self.lq.clone()
            self.gt_3dim = util_pol.chunkdim2batch(self.gt)
            self.lq_3dim=util_pol.chunkdim2batch(lq_clone)
            self.lq_downsample=util_pol.CPFA_downsample(self.cpfa[:,0:1,:,:])
        batch = {'lq':self.lq.to(self.device), 'lq_3dim':self.lq_3dim.to(self.device), 'gt':self.gt.to(self.device),\
                  'gt_3dim':self.gt_3dim.to(self.device), 'lq_downsample':self.lq_downsample.to(self.device)}
        return batch
    @torch.no_grad()
    def sample_func(self, im_cond):
        '''
        Input:
            im_cond: b x c x h x w, torch tensor, [0,1], RGB
        Output:
            xt: h x w x c, numpy array, [0,1], RGB
        '''
        input=im_cond['lq'].type(torch.float32)
        tt = torch.tensor(
            random.choices([100], k=input.shape[0]),
            dtype=torch.int64,
            device=self.device,
        ) - 1
        if self.configs.get('use_sd', True):
            input_base=2*im_cond['lq']-1
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            eps_pred=self.unet(input_base,tt)
            if isinstance(eps_pred, list):
                x0_base=input_base-eps_pred[0]
            else:
                x0_base=input_base-eps_pred
            input_sd=util_pol.chunkdim2batch(x0_base)
            z0_pred, _,_ = self.sd_forward_step(
                prompt=None,
                latents_hq=None,
                image_lq=0.5*input_sd+0.5,
                image_hq=None,
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
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"模型推理耗时: {elapsed_time_ms:.3f} ms")
        res_sr=0.5*x0_pred+0.5
        res_sr_0 = res_sr[:,0:3,:,:].clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()
        res_sr_45 = res_sr[:,3:6,:,:].clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()
        res_sr_90 = res_sr[:,6:9,:,:].clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()
        res_sr_135 = res_sr[:,9:12,:,:].clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()
        res_sr_S0,res_sr_DOLP,res_sr_AOP = util_pol.I2S012(res_sr,save_img=True,use_loss=False,clip=True,improve_contrast=True)
        res_sr_S0=res_sr_S0.clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()
        res_sr_DOLP=res_sr_DOLP.clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()
        res_sr_AOP=res_sr_AOP.clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()
        return res_sr_0,res_sr_45,res_sr_90,res_sr_135,res_sr_S0,res_sr_DOLP, res_sr_AOP
    def sd_forward_step(
        self,
        prompt: Union[str, List[str]] = None,
        latents_hq: Optional[torch.Tensor] = None,
        image_lq: torch.Tensor = None,
        image_hq: torch.Tensor = None,
        model_pred: DiagonalGaussianDistribution = None,
        timesteps: List[int] = None,
        **kwargs,
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
        # self.rank=0
        device=self.device
        # Encode input prompt
        prompt_embeds=torch.zeros_like(image_lq)
        # self.prompt_embeds = None

        # select the noise level, self.scheduler.sigmas, [1001,], descending
        if not hasattr(self, 'sigmas_cache'):
            assert self.sd_pipe.scheduler.sigmas.numel() == (self.configs.sd_pipe.num_train_steps + 1)
            self.sigmas_cache = self.sd_pipe.scheduler.sigmas.flip(0)[1:].to(device) #ascending,1000
        sigmas = self.sigmas_cache[timesteps] # (b,)

        # Prepare input for SD
        image_lq_up = image_lq
        zt_clean = self.encode_first_stage(
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
        z0_pred = zt_noisy - sigmas * eps_pred[0]
 
        return z0_pred, zt_noisy,sigmas
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
    def add_noise(self, latents, sigmas, model_pred=None):
        if sigmas.ndim < latents.ndim:
            sigmas = append_dims(sigmas, latents.ndim)

        if model_pred is None:
            # noise = torch.randn_like(latents)
            # zt_noisy = latents + sigmas * noise
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
def get_torch_dtype(torch_dtype: str):
    if torch_dtype == 'torch.float16':
        return torch.float16
    elif torch_dtype == 'torch.bfloat16':
        return torch.bfloat16
    elif torch_dtype == 'torch.float32':
        return torch.float32
    else:
        raise ValueError(f'Unexpected torch dtype:{torch_dtype}')

if __name__ == '__main__':
    pass

