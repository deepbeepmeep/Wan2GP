#!/usr/bin/env python3
"""
Tiny AutoEncoder for Hunyuan Video (Decoder-only, pruned)
- Encoder removed
- Transplant/widening helpers removed
- Deepening (IdentityConv2d+ReLU) is now built into the decoder structure itself
"""
from shared.utils.media_control import inference_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from shared.kernels import kernel_policy
try:  # Triton 3x3 convolution with the ReLU fused
    from shared.kernels import conv3d_triton
except ImportError:
    conv3d_triton = None
from collections import namedtuple
from einops import rearrange
import torch.nn.init as init

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

# ----------------------------
# Utility / building blocks
# ----------------------------

class Conv2dCL(nn.Conv2d):
    """Conv2d with a channels-last copy of its weight per call (the checkpoint layout stays as loaded and offloaded): cuDNN then works in
    NHWC end to end, without transposing the activations to and from NCHW around every convolution. Bit-identical; 1.17x faster and
    ~1 GiB lower peak for a 1920x1088 decode. relu: the ReLU that followed it (replaced by an Identity, see fuse_relus), fused in the
    epilogue of the Triton convolution on validated GPUs in "fast" kernel precision (accumulation order of the convolution only)."""
    relu = False

    def forward(self, x):
        if conv3d_triton is not None and conv3d_triton.conv2d_eligible(x, self) and kernel_policy.allow_approximate():
            return conv3d_triton.conv2d(x, self.weight, self.bias, self.relu)
        x = self._conv_forward(x, self.weight.contiguous(memory_format=torch.channels_last), self.bias)
        return F.relu_(x) if self.relu else x


def fuse_relus(sequential):
    """Each Conv2dCL followed by a ReLU applies it itself, the ReLU becoming an Identity (same module indices, same checkpoint keys)."""
    for i in range(len(sequential) - 1):
        if isinstance(sequential[i], Conv2dCL) and isinstance(sequential[i + 1], nn.ReLU):
            sequential[i].relu, sequential[i + 1] = True, nn.Identity()
    return sequential

class IdentityConv2d(Conv2dCL):
    """Same-shape Conv2d initialized to identity (Dirac)."""
    def __init__(self, C, kernel_size=3, bias=False):
        pad = kernel_size // 2
        super().__init__(C, C, kernel_size, padding=pad, bias=bias)
        with torch.no_grad():
            init.dirac_(self.weight)
            if self.bias is not None:
                self.bias.zero_()

def conv(n_in, n_out, **kwargs):
    return Conv2dCL(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = fuse_relus(nn.Sequential(
            conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out)
        ))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride, n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww
    def forward(self, x):
        # x: (B, C, F, H, W)
        B, C, F, H, W = x.shape
        if F % self.ff != 0:
            first_frame = x[:, :, 0:1, :, :].repeat(1, 1, self.ff - F % self.ff, 1, 1)
            x = torch.cat([first_frame, x], dim=2)
        return rearrange(
            x,
            'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
            ff=self.ff, hh=self.hh, ww=self.ww
        ).transpose(1, 2)

# ----------------------------
# Generic NTCHW graph executor (kept; used by decoder)
# ----------------------------

def _halo(module):
    """Pixels a module reads around each output pixel (zero-padded 3x3 convolutions), for the tile planner."""
    if isinstance(module, nn.Conv2d):
        return module.kernel_size[0] // 2
    if isinstance(module, MemBlock):
        return sum(m.kernel_size[0] // 2 for m in module.conv if isinstance(m, nn.Conv2d))
    return 0


def tile_plan(model, size, region):
    """Exact spatial tile of the decoder `model` for latent frames of `size` (h, w): the latent region ((r0, r1), (c0, c1)) to decode and,
    per module, the crop of its output, so that the decoded `region` (output pixels) equals the untiled decode. Every module only computes
    what the following ones need: a zero-padded convolution corrupts one pixel next to a region border that is not a frame border, the
    halos absorb it (the old single halo of the whole decoder made each tile compute ~2.5x its area at 512 pixels)."""
    sizes = [size]
    for module in model:
        up = isinstance(module, nn.Upsample) and module.scale_factor not in (None, 1, 1.0)
        sizes.append(tuple(2 * s for s in sizes[-1]) if up else sizes[-1])
    crops, need = [None] * len(model), []
    for axis, (lo, hi) in enumerate(region):
        for i in reversed(range(len(model))):
            if sizes[i + 1][axis] != sizes[i][axis]:  # nearest x2
                a, b = lo // 2, (hi - 1) // 2 + 1
                crop = slice(lo - 2 * a, hi - 2 * a)
            else:
                halo = _halo(model[i])
                a, b = max(lo - halo, 0), min(hi + halo, sizes[i][axis])
                crop = slice(lo - a, hi - a)
            full = crop.start == 0 and crop.stop == (b - a) * sizes[i + 1][axis] // sizes[i][axis]
            crops[i] = (crops[i] or ()) + ((None if full else crop),)
            lo, hi = a, b
        need.append((lo, hi))
    return tuple(need), [None if crop == (None, None) else tuple(slice(None) if c is None else c for c in crop) for crop in crops]


def _crop(x, crop):
    return x if crop is None else x[..., crop[0], crop[1]].contiguous(memory_format=torch.channels_last)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar, mem=None, crops=None):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N*T, C, H, W)
        for b in tqdm(model, disable=not show_progress_bar):
            inference_checkpoint()
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out = []
        work_queue = [TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))]
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        while work_queue:
            inference_checkpoint()
            xt, i = work_queue.pop(0)
            if i == 0:
                progress_bar.update(1)
            if i == len(model):
                out.append(xt)
            else:
                b = model[i]
                crop = None if crops is None else crops[i]
                if isinstance(b, MemBlock):
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt)
                    work_queue.insert(0, TWorkItem(_crop(xt_new, crop), i+1))
                elif isinstance(b, TPool):
                    if mem[i] is None:
                        mem[i] = []
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        raise ValueError("TPool internal state invalid.")
                    elif len(mem[i]) == b.stride:
                        N_, C_, H_, W_ = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N_*b.stride, C_, H_, W_))
                        mem[i] = []
                        work_queue.insert(0, TWorkItem(xt, i+1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C_, H_, W_ = xt.shape
                    for xt_next in reversed(xt.view(N, b.stride*C_, H_, W_).chunk(b.stride, 1)):
                        work_queue.insert(0, TWorkItem(xt_next, i+1))
                else:
                    xt = _crop(b(xt), crop)
                    work_queue.insert(0, TWorkItem(xt, i+1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x, mem

# ----------------------------
# Decoder-only TAEHV
# ----------------------------

class TAEHV(nn.Module):
    image_channels = 3
    def __init__(
        self,
        checkpoint_path="taehv.pth",
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        channels = [256, 128, 64, 64],
        latent_channels = 16
    ):
        """Initialize TAEHV (decoder-only) with built-in deepening after every ReLU.
        Deepening config: how_many_each=1, k=3 (fixed as requested).
        """
        super().__init__()
        self.latent_channels = latent_channels
        n_f = channels
        self.frames_to_trim = 2**sum(decoder_time_upscale) - 1

        # Build the decoder "skeleton"
        base_decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),

            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),

            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),

            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),

            nn.ReLU(inplace=True), conv(n_f[3], TAEHV.image_channels),
        )

        # Inline deepening: insert (IdentityConv2d(k=3) + ReLU) after every ReLU
        self.decoder = fuse_relus(self._apply_identity_deepen(base_decoder, how_many_each=1, k=3))

        self.pixel_shuffle = PixelShuffle3d(4, 8, 8)

        if checkpoint_path is not None:
            missing_keys = self.load_state_dict(
                self.patch_tgrow_layers(torch.load(checkpoint_path, map_location="cpu", weights_only=True)),
                strict=False
            )
            print('missing_keys', missing_keys)

        # Initialize decoder mem state
        self.mem = [None] * len(self.decoder)

    @staticmethod
    def _apply_identity_deepen(decoder: nn.Sequential, how_many_each=1, k=3) -> nn.Sequential:
        """Return a new Sequential where every nn.ReLU is followed by how_many_each*(IdentityConv2d(k)+ReLU)."""
        new_layers = []
        for b in decoder:
            new_layers.append(b)
            if isinstance(b, nn.ReLU):
                # Deduce channel count from preceding layer
                C = None
                if len(new_layers) >= 2 and isinstance(new_layers[-2], nn.Conv2d):
                    C = new_layers[-2].out_channels
                elif len(new_layers) >= 2 and isinstance(new_layers[-2], MemBlock):
                    C = new_layers[-2].conv[-1].out_channels
                if C is not None:
                    for _ in range(how_many_each):
                        new_layers.append(IdentityConv2d(C, kernel_size=k, bias=False))
                        new_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*new_layers)

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed (decoder-only)."""
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def decode_video(self, x, parallel=True, show_progress_bar=False, cond=None, crops=None):
        """Decode a sequence of frames from latents.
        x: NTCHW latent tensor; returns NTCHW RGB in ~[0, 1]. crops: per decoder module, the crop of its output (tile_plan, sequential only).
        """
        trim_flag = self.mem[-8] is None  # keeps original relative check

        if cond is not None:
            x = torch.cat([self.pixel_shuffle(cond), x], dim=2)

        x, self.mem = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar, mem=self.mem, crops=crops)

        if trim_flag:
            return x[:, self.frames_to_trim:]
        return x

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Decoder-only model: call decode_video(...) instead.")

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class TAEW2_1DiffusersWrapper(nn.Module):
    def __init__(self, pretrained_path=None, channels = [256, 128, 64, 64]):
        super().__init__()
        self.dtype = torch.bfloat16
        self.device = "cuda"
        self.taehv = TAEHV(pretrained_path, channels = channels).to(self.dtype)
        self.temperal_downsample = [True, True, False]  # [sic]
        self.config = DotDict(scaling_factor=1.0, latents_mean=torch.zeros(16), z_dim=16, latents_std=torch.ones(16))

    def decode(self, latents, return_dict=None):
        n, c, t, h, w = latents.shape
        return (self.taehv.decode_video(latents.transpose(1, 2), parallel=False).transpose(1, 2).mul_(2).sub_(1),)

    def stream_decode_with_cond(self, latents, tiled=False, cond=None):
        n, c, t, h, w = latents.shape
        return self.taehv.decode_video(latents.transpose(1, 2), parallel=False, cond=cond).transpose(1, 2).mul_(2).sub_(1)

    def decode_video(self, latents, parallel=False, show_progress_bar=False, cond=None):
        """Decode video latents to frames. Wrapper for TAEHV.decode_video.
        
        Args:
            latents: (B, F, C, H, W) tensor - already transposed from pipeline
            cond: (B, F, C, H, W) tensor or None - conditioning frames
        
        Returns:
            (B, F, C_out, H_out, W_out) decoded frames
        """
        return self.taehv.decode_video(latents, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)

    def clean_mem(self):
        self.taehv.clean_mem()

# ----------------------------
# Simplified builder (no small, no transplant, no post-hoc deepening)
# ----------------------------

def build_tcdecoder(new_channels = [512, 256, 128, 128],
                                  device="cuda",
                                  dtype=torch.bfloat16,
                                  new_latent_channels=None):
    """
    构建"更宽"的 decoder；深度增强（IdentityConv2d+ReLU）已在 TAEHV 内部完成。
    - 不创建 small / 不做移植
    - base_ckpt_path 参数保留但不使用（接口兼容）

    返回：big （单个模型）
    """
    if new_latent_channels is not None:
        big = TAEHV(checkpoint_path=None, channels=new_channels, latent_channels=new_latent_channels).to(device).to(dtype).train()
    else:
        big = TAEHV(checkpoint_path=None, channels=new_channels).to(device).to(dtype).train()

    big.clean_mem()
    return big
