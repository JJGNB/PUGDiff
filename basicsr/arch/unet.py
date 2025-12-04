from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from basicsr.arch.restormer import TransformerBlock

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .swin_transformer import BasicLayer

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x ch
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            # q,k, v: (b*heads) x ch x length
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards     # (b*heads) x M x M
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v)  # (b*heads) x ch x length
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x length
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            q, k, v = qkv.chunk(3, dim=1)  # b x heads*ch x length
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts",
                (q * scale).view(bs * self.n_heads, ch, length),
                (k * scale).view(bs * self.n_heads, ch, length),
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
    
NUM_CLASSES = 1000

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    model_path='',
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    if isinstance(attention_resolutions, int):
        attention_ds.append(image_size // attention_resolutions)
    elif isinstance(attention_resolutions, str):
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
    else:
        raise NotImplementedError

    model= UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

    try:
        model.load_state_dict(th.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Got exception: {e} / Randomly initialize")
    return model
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

class UNetModelSwin(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
        self,
        image_size,
        task_type,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_restormer=False,
        use_res=False,
        use_multout=False,
        swin_depth=2,
        swin_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_lq=True,
        cond_mask=False,
        lq_size=256,
        use_var=False
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks
        self.use_res=use_res
        self.use_multout=use_multout
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask
        self.use_var=use_var
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            self.feature_extractor = nn.Identity()
            base_chn = 3 if task_type=="SR" else 12
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        ch = input_ch = int(channel_mult[0] * model_channels)
        # in_channels += base_chn
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    if  use_restormer:
                        layers.append(nn.Sequential(*[TransformerBlock(dim=ch,num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(swin_depth)]))
                    else:
                        layers.append(
                            BasicLayer(
                                    in_chans=ch,
                                    embed_dim=swin_embed_dim,
                                    num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                    window_size=window_size,
                                    depth=swin_depth,
                                    img_size=ds,
                                    patch_size=1,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop=dropout,
                                    attn_drop=0.,
                                    drop_path=0.,
                                    use_checkpoint=False,
                                    norm_layer=normalization,
                                    patch_norm=patch_norm,
                                    )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2
        if use_restormer:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                nn.Sequential(*[TransformerBlock(dim=ch,num_heads=8, ffn_expansion_factor=2,
                                                 bias=False, LayerNorm_type='WithBias') for _ in range(swin_depth)]),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )
        else:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                    BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                            ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i==0:
                    if use_restormer:
                        layers.append(nn.Sequential(*[TransformerBlock(dim=ch,num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(swin_depth)]))
                    else:
                        layers.append(
                            BasicLayer(
                                    in_chans=ch,
                                    embed_dim=swin_embed_dim,
                                    num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                    window_size=window_size,
                                    depth=swin_depth,
                                    img_size=ds,
                                    patch_size=1,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop=dropout,
                                    attn_drop=0.,
                                    drop_path=0.,
                                    use_checkpoint=False,
                                    norm_layer=normalization,
                                    patch_norm=patch_norm,
                                    )
                        )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))


        if self.use_multout:
            self.out_0 = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, 3, 3, padding=1),
            )
            self.out_45 = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, 3, 3, padding=1),
            )
            self.out_90 = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, 3, 3, padding=1),
            )
            self.out_135 = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, 3, 3, padding=1),
            )
        else:
            self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
            )
        if self.use_var:
            self.var_conv = nn.Sequential(*[nn.Conv2d(input_ch, input_ch, 3,1,1), nn.ELU(),nn.Conv2d(input_ch, input_ch, 3,1,1), nn.ELU(),nn.Conv2d(input_ch, out_channels//4, 3,1,1)])
    def forward(self, input, timesteps, lq=None, mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([input, lq], dim=1)
        else:
            x=input
        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        if self.use_multout:
            out_0=self.out_0(h)
            out_45=self.out_45(h)
            out_90=self.out_90(h)
            out_135=self.out_135(h)
            out=th.concat([out_0,out_45,out_90,out_135],dim=1)
        else:
            out = self.out(h)
        if self.use_var:
            var=self.var_conv(h)
            return [out, var]
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
class MaskNetwork(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MaskNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.tanh = nn.Tanh()  # 使用Sigmoid激活确保值在0到1之间
        # self.alpha = nn.Parameter(th.tensor(0.5))  # 可学习的参数
    def forward(self, x):
        res=x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        mask = self.tanh(x) 
        mask = mask + res
        mask= th.clamp(mask, 0, 1)  # 确保mask在0到1之间
        # mask = mask * self.alpha
        return mask
class CorrectNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(CorrectNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels*2, 48, 3, padding=1)                 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Sequential(*[TransformerBlock(dim=48,num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(2)])
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Sequential(*[TransformerBlock(dim=48,num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(2)])
        self.out=nn.Conv2d(48, 12, 3, padding=1)  
    def forward(self, x0_sd,x0_base):
        residual = 0.5*x0_sd+0.5*x0_base
        x= th.cat([x0_sd, x0_base], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x= self.out(x)
        x = x+residual
        return x
class ResBlockConv(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetModelConv(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        cond_lq=True,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.dtype = th.float16 if use_fp16 else th.float32

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.cond_lq = cond_lq

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlockConv(
                        ch,
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConv(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlockConv(
                ch,
                time_embed_dim,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlockConv(
                ch,
                time_embed_dim,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockConv(
                        ch + ich,
                        time_embed_dim,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlockConv(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if lq is not None:
            assert self.cond_lq
            if lq.shape[2:] != x.shape[2:]:
                lq = F.pixel_unshuffle(lq, 2)
            x = th.cat([x, lq], dim=1)

        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

