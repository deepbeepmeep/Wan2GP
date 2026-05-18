# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
from mmgp import offload
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'WanVAE',
]

CACHE_T = 2


def _vae_float_to_cpu_uint8(frames):
    frames.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().clamp_(0.0, 255.0)
    return frames.to(device="cpu", dtype=torch.uint8)


def _blend_v_edge_(top_edge, tile, blend_extent):
    blend_extent = min(int(top_edge.shape[-2]), int(tile.shape[-2]), int(blend_extent))
    if blend_extent <= 0:
        return
    weights = torch.arange(blend_extent, device=tile.device, dtype=tile.dtype).div_(blend_extent).view(1, 1, 1, blend_extent, 1)
    edge = top_edge[:, :, :, -blend_extent:, :].to(device=tile.device, dtype=tile.dtype)
    edge.mul_(1.0 - weights)
    tile[:, :, :, :blend_extent, :].mul_(weights).add_(edge)


def _blend_h_edge_(left_edge, tile, blend_extent):
    blend_extent = min(int(left_edge.shape[-1]), int(tile.shape[-1]), int(blend_extent))
    if blend_extent <= 0:
        return
    weights = torch.arange(blend_extent, device=tile.device, dtype=tile.dtype).div_(blend_extent).view(1, 1, 1, 1, blend_extent)
    edge = left_edge[:, :, :, :, -blend_extent:].to(device=tile.device, dtype=tile.dtype)
    edge.mul_(1.0 - weights)
    tile[:, :, :, :, :blend_extent].mul_(weights).add_(edge)


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            cache_x = None
        x = F.pad(x, padding)
        try:
            out = super().forward(x)
            return out
        except RuntimeError as e:
            if "miopenStatus" in str(e):
                print("⚠️ MIOpen fallback: AMD gets upset when trying to work with large areas, and so CPU will be "
                      "used for this decoding (which is very slow). Consider using tiled VAE Decoding.")
                x_cpu = x.float().cpu()
                weight_cpu = self.weight.float().cpu()
                bias_cpu = self.bias.float().cpu() if self.bias is not None else None
                print(f"[Fallback] x shape: {x_cpu.shape}, weight shape: {weight_cpu.shape}")
                out = F.conv3d(x_cpu, weight_cpu, bias_cpu,
                               self.stride, (0, 0, 0),  # avoid double padding here
                               self.dilation, self.groups)
                out = out.to(x.device)
                if x.dtype in (torch.float16, torch.bfloat16):
                    out = out.half()
                if x.dtype != out.dtype:
                    out = out.to(x.dtype)
                return out
            raise


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        dtype = x.dtype
        x = F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias
        x = x.to(dtype)
        return x 

class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    clone = True
                    cache_x = x[:, :, -CACHE_T:, :, :]#.clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        clone = False
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        clone = False
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if clone:
                        cache_x = cache_x.clone()
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x #.to("cpu") #x.clone() yyyy
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x#.to("cpu") #yyyyy
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        dtype = x.dtype
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx]).to(dtype)
                feat_cache[idx] = cache_x#.to("cpu")
                feat_idx[0] += 1
            else:
                x = layer(x).to(dtype)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        dtype = x.dtype
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx]).to(dtype)
            feat_cache[idx] = cache_x
            del cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)


        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)


        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)


        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                del cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)


        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0,
                 upsampler_factor = 1,
                 ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3 * int(upsampler_factor*upsampler_factor), 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x#.to("cpu")
            del cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        cache_x = None

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :] .clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x#.to("cpu")
                del cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    _offload_hooks = ['encode', 'decode']

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 upsampler_factor = 1,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.upsampler_factor = upsampler_factor

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout, upsampler_factor)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale = None, any_end_frame = False):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        if any_end_frame:
            iter_ = 2 + (t - 2) // 4
        else:
            iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        out_list = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out_list.append(self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx))
            elif any_end_frame and i== iter_ -1:
                out_list.append(self.encoder(
                    x[:, :, -1:, :, :],
                    feat_cache= None,
                    feat_idx=self._enc_conv_idx))
            else:
                out_list.append(self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx))

        self.clear_cache()
        out = torch.cat(out_list, 2)
        out_list = None

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale != None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        return mu


    def decode(self, z, scale=None, any_end_frame = False):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if scale != None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        out_list = []
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out_list.append(self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx))
            elif any_end_frame and i==iter_-1:
                out_list.append(self.decoder(
                    x[:, :, -1:, :, :],
                    feat_cache=None ,
                    feat_idx=self._conv_idx))
            else:
                out_list.append(self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx))
        self.clear_cache()
        out = torch.cat(out_list, 2)

        if self.upsampler_factor > 1:
            out = F.pixel_shuffle(out.movedim(2, 1), upscale_factor=self.upsampler_factor).movedim(1, 2)  # pixel shuffle needs [..., C, H, W] format

        return out
    
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b
    
    def spatial_tiled_decode(self, z, scale, tile_size, any_end_frame= False):
        tile_sample_min_size = tile_size
        tile_latent_min_size = int(tile_sample_min_size / 8)
        tile_overlap_factor = 0.25

        # z: [b,c,t,h,w]

        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view( 1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]


        overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor)) #8 0.75
        tile_sample_min_size *=  self.upsampler_factor
        blend_extent = int(tile_sample_min_size * tile_overlap_factor) #256 0.25
        row_limit = tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i: i + tile_latent_min_size, j: j + tile_latent_min_size]
                decoded = self.decode(tile, any_end_frame= any_end_frame)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=-2)

    def decode_tile_chunks(self, z, any_end_frame=False):
        self.clear_cache()
        x = self.conv2(z)
        frame_start = 0
        try:
            for i in range(x.shape[2]):
                self._conv_idx = [0]
                if i == 0:
                    tile = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                elif any_end_frame and i == x.shape[2] - 1:
                    tile = self.decoder(x[:, :, -1:, :, :], feat_cache=None, feat_idx=self._conv_idx)
                else:
                    tile = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                if self.upsampler_factor > 1:
                    tile = F.pixel_shuffle(tile.movedim(2, 1), upscale_factor=self.upsampler_factor).movedim(1, 2)
                yield frame_start, tile
                frame_start += int(tile.shape[2])
                del tile
        finally:
            del x
            self.clear_cache()

    def decode_to_cpu_uint8(self, z, scale=None, tile_size=0, target_frames=None, target_height=None, target_width=None, any_end_frame=False, device=None):
        device = torch.device(device) if device is not None else (scale[0].device if scale is not None and isinstance(scale[0], torch.Tensor) else z.device)
        dtype = getattr(self, "_model_dtype", z.dtype)
        tile_size = int(tile_size or 0)
        latent_source = z.detach()
        if tile_size > 0 and latent_source.device.type != "cpu":
            latent_source = latent_source.to("cpu")
        decoded_frame_count = 0 if latent_source.shape[2] <= 0 else (int(latent_source.shape[2]) - 1) * 4 + 1
        target_frames = decoded_frame_count if target_frames is None else min(int(target_frames), decoded_frame_count)
        needed_latents = 0 if target_frames <= 0 else min(int(latent_source.shape[2]), (max(int(target_frames), 1) - 1 + 3) // 4 + 1)
        latent_source = latent_source[:, :, :needed_latents]
        full_height = latent_source.shape[-2] * 8 * self.upsampler_factor
        full_width = latent_source.shape[-1] * 8 * self.upsampler_factor
        target_height = full_height if target_height is None else min(int(target_height), full_height)
        target_width = full_width if target_width is None else min(int(target_width), full_width)
        if target_frames <= 0:
            return torch.empty((latent_source.shape[0], 3, 0, target_height, target_width), dtype=torch.uint8, device="cpu")
        if scale is not None and isinstance(scale[0], torch.Tensor):
            scale = [u.to(device=device) for u in scale]
        if tile_size <= 0:
            z = latent_source.to(device=device, dtype=dtype)
            frames = self.decode(z, scale, any_end_frame=any_end_frame)[:, :, :target_frames, :target_height, :target_width]
            decoded = _vae_float_to_cpu_uint8(frames)
            del z, frames
            return decoded

        tile_sample_min_size = tile_size
        tile_latent_min_size = max(1, int(tile_sample_min_size / 8))
        tile_overlap_factor = 0.25
        overlap_size = max(1, int(tile_latent_min_size * (1 - tile_overlap_factor)))
        tile_sample_min_size *= self.upsampler_factor
        blend_extent = int(tile_sample_min_size * tile_overlap_factor)
        row_limit = max(1, tile_sample_min_size - blend_extent)
        decoded = torch.empty((latent_source.shape[0], 3, target_frames, target_height, target_width), dtype=torch.uint8, device="cpu")
        previous_row_edges = []
        row_index = 0
        for latent_y in range(0, latent_source.shape[-2], overlap_size):
            current_row_edges = []
            left_edge = None
            col_index = 0
            write_y0 = row_index * row_limit
            write_y1 = min(write_y0 + row_limit, target_height)
            has_next_row = write_y1 < target_height
            if write_y1 <= write_y0:
                break
            for latent_x in range(0, latent_source.shape[-1], overlap_size):
                write_x0 = col_index * row_limit
                write_x1 = min(write_x0 + row_limit, target_width)
                has_next_col = write_x1 < target_width
                if write_x1 <= write_x0:
                    break
                tile_latents = latent_source[:, :, :, latent_y:latent_y + tile_latent_min_size, latent_x:latent_x + tile_latent_min_size].to(device=device, dtype=dtype)
                if scale is not None:
                    if isinstance(scale[0], torch.Tensor):
                        tile_latents.div_(scale[1].view(1, self.z_dim, 1, 1, 1)).add_(scale[0].view(1, self.z_dim, 1, 1, 1))
                    else:
                        tile_latents.div_(scale[1]).add_(scale[0])
                bottom_edge = None
                right_edge = None
                previous_edge = previous_row_edges[col_index] if row_index > 0 and col_index < len(previous_row_edges) else None
                for frame_start, tile in self.decode_tile_chunks(tile_latents, any_end_frame=any_end_frame):
                    if frame_start >= target_frames:
                        break
                    frame_end = min(frame_start + int(tile.shape[2]), target_frames)
                    tile = tile[:, :, :frame_end - frame_start]
                    if previous_edge is not None:
                        _blend_v_edge_(previous_edge[:, :, frame_start:frame_end], tile, blend_extent)
                    if left_edge is not None:
                        _blend_h_edge_(left_edge[:, :, frame_start:frame_end], tile, blend_extent)
                    if has_next_row:
                        edge = tile[:, :, :, -min(blend_extent, tile.shape[-2]):, :].detach().cpu()
                        if bottom_edge is None:
                            bottom_edge = torch.empty((edge.shape[0], edge.shape[1], target_frames, edge.shape[3], edge.shape[4]), dtype=edge.dtype, device="cpu")
                        bottom_edge[:, :, frame_start:frame_end].copy_(edge)
                        del edge
                    if has_next_col:
                        edge = tile[:, :, :, :, -min(blend_extent, tile.shape[-1]):].detach().cpu()
                        if right_edge is None:
                            right_edge = torch.empty((edge.shape[0], edge.shape[1], target_frames, edge.shape[3], edge.shape[4]), dtype=edge.dtype, device="cpu")
                        right_edge[:, :, frame_start:frame_end].copy_(edge)
                        del edge
                    tile = tile[:, :, :, :write_y1 - write_y0, :write_x1 - write_x0]
                    decoded[:, :, frame_start:frame_end, write_y0:write_y0 + tile.shape[-2], write_x0:write_x0 + tile.shape[-1]].copy_(_vae_float_to_cpu_uint8(tile))
                    del tile
                current_row_edges.append(bottom_edge)
                left_edge = right_edge
                del tile_latents, previous_edge
                col_index += 1
            left_edge = None
            previous_row_edges = current_row_edges
            row_index += 1
        return decoded


    def spatial_tiled_encode(self, x, scale, tile_size, any_end_frame = False) :
        tile_sample_min_size = tile_size
        tile_latent_min_size = int(tile_sample_min_size / 8)
        tile_overlap_factor = 0.25

        overlap_size = int(tile_sample_min_size * (1 - tile_overlap_factor))
        blend_extent = int(tile_latent_min_size * tile_overlap_factor)
        row_limit = tile_latent_min_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[:, :, :, i: i + tile_sample_min_size, j: j + tile_sample_min_size]
                tile = self.encode(tile, any_end_frame= any_end_frame)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        mu = torch.cat(result_rows, dim=-2)

        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]

        return mu


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', preprocess_sd=None, **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    from mmgp import offload
    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    # model.load_state_dict(
    #     torch.load(pretrained_path, map_location=device), assign=True)
    # offload.load_model_data(model, pretrained_path.replace(".pth", "_bf16.safetensors"), writable_tensors= False)    
    offload.load_model_data(model, pretrained_path, writable_tensors=False, preprocess_sd=preprocess_sd)    
    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 upsampler_factor = 1,
                 device="cuda",
                 preprocess_sd=None):
        self.dtype = dtype
        self.device = device
        self.z_dim = z_dim

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            upsampler_factor = upsampler_factor,
            z_dim=z_dim,
            preprocess_sd=preprocess_sd,
        ).to(dtype).eval() #.requires_grad_(False).to(device)
        self.model._model_dtype = dtype

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        # VAE Tiling
        if vae_config == 0:
            if mixed_precision:
                device_mem_capacity = device_mem_capacity / 2
            if device_mem_capacity >= 24000:
                use_vae_config = 1
            elif device_mem_capacity >= 16000:
                use_vae_config = 3
            elif device_mem_capacity >= 8000:
                use_vae_config = 4
            else:          
                use_vae_config = 5
        else:
            # Keep WGP's historical public presets; Wan inserts one internal tiers between presets 1 and 3.
            use_vae_config = vae_config if vae_config == 1 else vae_config + 2

        if output_height is not None and output_width is not None and int(output_height) * int(output_width) > 1920 * 1088:
            use_vae_config = min(use_vae_config + 1, 4)

        if use_vae_config == 1:
            VAE_tile_size = 0  
        elif use_vae_config == 2:
            VAE_tile_size = 1024 
        elif use_vae_config == 3:
            VAE_tile_size = 512 
        elif use_vae_config == 4:
            VAE_tile_size = 256  
        else: 
            VAE_tile_size = 128  

        return  VAE_tile_size

    def encode(self, videos, tile_size = 256, any_end_frame = False):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        scale = [u.to(device = self.device) for u in self.scale]  
        if tile_size > 0:
            return [ self.model.spatial_tiled_encode(u.to(self.dtype).unsqueeze(0), scale, tile_size, any_end_frame=any_end_frame).float().squeeze(0) for u in videos ]
        else:
            return [ self.model.encode(u.to(self.dtype).unsqueeze(0), scale, any_end_frame=any_end_frame).float().squeeze(0) for u in videos ]


    def decode(self, zs, tile_size, any_end_frame = False):
        scale = [u.to(device = self.device) for u in self.scale]  
        if tile_size > 0:
            return [ self.model.spatial_tiled_decode(u.to(self.dtype).unsqueeze(0), scale, tile_size, any_end_frame=any_end_frame).clamp_(-1, 1).float().squeeze(0) for u in zs ]
        else:
            return [ self.model.decode(u.to(self.dtype).unsqueeze(0), scale, any_end_frame=any_end_frame).clamp_(-1, 1).float().squeeze(0) for u in zs ]

    def decode_to_cpu_uint8(self, zs, tile_size, target_frames=None, target_height=None, target_width=None, any_end_frame=False):
        scale = [u.to(device=self.device) for u in self.scale]
        tile_size = int(tile_size or 0)
        return [
            self.model.decode_to_cpu_uint8(u.detach().to(device="cpu" if tile_size > 0 else u.device, dtype=self.dtype).unsqueeze(0), scale, tile_size, target_frames=target_frames, target_height=target_height, target_width=target_width, any_end_frame=any_end_frame, device=self.device).squeeze(0)
            for u in zs
        ]
