
import math
import torch
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import torch.nn.functional as F
from peft import LoraConfig,get_peft_model
def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

def forward_chop(net, x, net_kwargs=None, scale=1, shave=10, min_size=160000):
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if net_kwargs is None:
                sr_batch = net(lr_batch)
            else:
                sr_batch = net(lr_batch, **net_kwargs)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def measure_time(net, inputs, num_forward=100):
    '''
    Measuring the average runing time (seconds) for pytorch.
    out = net(*inputs)
    '''
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.set_grad_enabled(False):
        for _ in range(num_forward):
            out = net(*inputs)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / 1000

def reload_model(model, ckpt,load_strict=False):
    module_flag = list(ckpt.keys())[0].startswith('module.')
    compile_flag = '_orig_mod' in list(ckpt.keys())[0]

    for source_key, source_value in model.state_dict().items():
        target_key = source_key
        if compile_flag and (not '_orig_mod.' in source_key):
            target_key = '_orig_mod.' + target_key
        if module_flag and (not source_key.startswith('module')):
            target_key = 'module.' + target_key

        if target_key in ckpt:
            source_value.copy_(ckpt[target_key])
        elif not load_strict:  # If a is False, allow missing parameters
            print(f"Warning: {target_key} not found in checkpoint.")
def initialize_vae(lora_rank,sd_pipe,configs):
    sd_pipe.vae.requires_grad_(False)
    sd_pipe.vae.train()
    if configs.sd_pipe.lora_ckpt_path is not None:
        print("vae resume from lora")
        lora_ckpt=torch.load(configs.sd_pipe.lora_ckpt_path)
        if configs.sd_pipe.get('use_lora', False):
            lora_conf_encoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["vae_lora_encoder_modules"])
            sd_pipe.vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            for n, p in sd_pipe.vae.named_parameters():
                if "lora" in n:
                    p.data.copy_(lora_ckpt["state_dict_vae"][n])
            sd_pipe.lora_vae_modules_encoder=lora_ckpt["vae_lora_encoder_modules"]
            sd_pipe.vae.set_adapter(["default_encoder"])

    else:
        l_target_modules_encoder = []
        l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
        for n, p in sd_pipe.vae.named_parameters():
            if "bias" in n or "norm" in n: 
                continue
            for pattern in l_grep:
                # if pattern in n and ("encoder" in n):
                if pattern in n :
                    l_target_modules_encoder.append(n.replace(".weight",""))
                elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                    l_target_modules_encoder.append(n.replace(".weight",""))
        
        lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
        sd_pipe.vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        sd_pipe.lora_vae_modules_encoder=l_target_modules_encoder
        sd_pipe.vae.set_adapter(['default_encoder'])
def initialize_unet(lora_rank, sd_pipe,configs):
    sd_pipe.unet.requires_grad_(False)
    sd_pipe.unet.train()
    if configs.sd_pipe.lora_ckpt_path is not None:
        print("unet resume from lora")
        lora_ckpt=torch.load(configs.sd_pipe.lora_ckpt_path)
        if configs.sd_pipe.get('use_lora', False):
            lora_conf_encoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["unet_lora_encoder_modules"])
            lora_conf_decoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["unet_lora_decoder_modules"])
            lora_conf_others = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["unet_lora_others_modules"])
            sd_pipe.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            sd_pipe.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
            sd_pipe.unet.add_adapter(lora_conf_others, adapter_name="default_others")
            for n, p in sd_pipe.unet.named_parameters():
                if "lora" in n or "conv_in" in n:
                    p.data.copy_(lora_ckpt["state_dict_unet"][n])
            sd_pipe.lora_unet_modules_encoder=lora_ckpt["unet_lora_encoder_modules"]
            sd_pipe.lora_unet_modules_decoder=lora_ckpt["unet_lora_decoder_modules"]
            sd_pipe.lora_unet_others=lora_ckpt["unet_lora_others_modules"]
            sd_pipe.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    else:
        l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
        l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
        for n, p in sd_pipe.unet.named_parameters():
            if "bias" in n or "norm" in n:
                continue
            for pattern in l_grep:
                if "A_injector" in n:
                    continue
                if pattern in n and ("down_blocks" in n or "conv_in" in n):
                    l_target_modules_encoder.append(n.replace(".weight",""))
                    break
                elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                    l_target_modules_decoder.append(n.replace(".weight",""))
                    break
                elif pattern in n:
                    l_modules_others.append(n.replace(".weight",""))
                    break
        lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
        lora_conf_decoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
        lora_conf_others = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
        sd_pipe.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        sd_pipe.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        sd_pipe.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        sd_pipe.lora_unet_modules_encoder=l_target_modules_encoder
        sd_pipe.lora_unet_modules_decoder=l_target_modules_decoder
        sd_pipe.lora_unet_others=l_modules_others
        sd_pipe.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
def initialize_psrnet(lora_rank, psr_net,configs):
    psr_net.requires_grad_(False)
    psr_net.train()
    if configs.sd_pipe.lora_ckpt_path is not None:
        print("psrnet resume from lora")
        lora_ckpt=torch.load(configs.unet.lora_ckpt_path)
        if configs.unet.get('use_lora', False):
            lora_conf_encoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["psrnet_lora_encoder_modules"])
            lora_conf_decoder = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["psrnet_lora_decoder_modules"])
            lora_conf_others = LoraConfig(r=lora_ckpt["lora_rank"], init_lora_weights="gaussian", target_modules=lora_ckpt["psrnet_lora_others_modules"])
            psr_net.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            psr_net.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
            psr_net.add_adapter(lora_conf_others, adapter_name="default_others")
            for n, p in psr_net.named_parameters():
                if "lora" in n or "conv_in" in n:
                    p.data.copy_(lora_ckpt["state_dict_psrnet"][n])
            psr_net.lora_psrnet_modules_encoder=lora_ckpt["psrnet_lora_encoder_modules"]
            psr_net.lora_psrnet_modules_decoder=lora_ckpt["psrnet_lora_decoder_modules"]
            psr_net.lora_psrnet_others=lora_ckpt["psrnet_lora_others_modules"]
            psr_net.set_adapter(["default_encoder", "default_decoder", "default_others"])

    else:
        l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
        l_grep = [ "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "q","q_dwconv","kv","kv_dwconv","project_out","project_in","dwconv","project_out"]
        for n, p in psr_net.named_parameters():
            if "bias" in n or "norm" in n:
                continue
            for pattern in l_grep:
                if pattern in n and ("down_blocks" in n or "conv_in" in n):
                    l_target_modules_encoder.append(n.replace(".weight",""))
                    break
                elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                    l_target_modules_decoder.append(n.replace(".weight",""))
                    break
                elif pattern in n:
                    l_modules_others.append(n.replace(".weight",""))
                    break
        lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
        lora_conf_decoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
        lora_conf_others = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
        psr_net.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        psr_net.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        psr_net.add_adapter(lora_conf_others, adapter_name="default_others")
        psr_net.lora_psrnet_modules_encoder=l_target_modules_encoder
        psr_net.lora_psrnet_modules_decoder=l_target_modules_decoder
        psr_net.lora_psrnet_others=l_modules_others
        psr_net.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
def set_train_psrnet(psr_net):
    psr_net.train()
    for n, _p in psr_net.named_parameters():
        if "lora" in n:
            _p.requires_grad = True
    psr_net.conv_in.requires_grad_(True)
def set_train_unet(sd_pipe):
    sd_pipe.unet.train()
    for n, _p in sd_pipe.unet.named_parameters():
        if "lora" in n:
            _p.requires_grad = True
    sd_pipe.unet.conv_in.requires_grad_(True)
def set_train_vae(sd_pipe):
    sd_pipe.vae.train()
    for n, _p in sd_pipe.vae.named_parameters():
        if "lora" in n:
            _p.requires_grad = True
    
