import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from shared.attention import pay_attention


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def _apply_rope_inplace(x: Tensor, freqs: Tensor) -> Tensor:
    freqs = freqs[:, None, :, :, :]
    cos = freqs[..., 0, 0].to(x.dtype)
    sin = freqs[..., 1, 0].to(x.dtype)
    x_pair = x.reshape(*x.shape[:-1], -1, 2)
    x0 = x_pair[..., 0].clone()
    x1 = x_pair[..., 1]
    x_pair[..., 0].mul_(cos).sub_(x1 * sin)
    x_pair[..., 1].mul_(cos).add_(x0 * sin)
    return x


def _apply_rope_reference(x: Tensor, freqs: Tensor) -> Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    x_ = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return x_.reshape(*x.shape).to(x.dtype)


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor, reference: bool = False) -> tuple[Tensor, Tensor]:
    if reference:
        return _apply_rope_reference(xq, freqs), _apply_rope_reference(xk, freqs)
    return _apply_rope_inplace(xq, freqs), _apply_rope_inplace(xk, freqs)


def attention(qkv_list: list[Tensor], mask: Tensor | None = None, scale: float | None = None, gqa: bool = False, experimental_patch: bool = False) -> Tensor:
    q, k, v = qkv_list
    qkv_list.clear()
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    if mask is not None:
        mask = mask.transpose(1, 2)
    if gqa:
        batch = q.shape[0]
        groups = k.shape[2]
        repeat = q.shape[2] // groups
        q = rearrange(q, "B L (G R) D -> (B G R) L 1 D", G=groups, R=repeat)
        if experimental_patch:
            q = q.contiguous()
            k = rearrange(k, "B L G D -> B G 1 L D").repeat_interleave(repeat, dim=2)
            v = rearrange(v, "B L G D -> B G 1 L D").repeat_interleave(repeat, dim=2)
        else:
            k = rearrange(k, "B L G D -> B G 1 L D").expand(batch, groups, repeat, -1, -1)
            v = rearrange(v, "B L G D -> B G 1 L D").expand(batch, groups, repeat, -1, -1)
        k = rearrange(k, "B G R L D -> (B G R) L 1 D")
        v = rearrange(v, "B G R L D -> (B G R) L 1 D")
        if experimental_patch:
            k = k.contiguous()
            v = v.contiguous()
        if mask is not None:
            mask = mask.repeat_interleave(groups * repeat, dim=0).contiguous() if experimental_patch or batch > 1 else mask.expand(groups * repeat, -1, -1, -1)
        qkv_list = [q, k, v]
        q = k = v = None
        out = pay_attention(qkv_list, attention_mask=mask, softmax_scale=scale, recycle_q=not experimental_patch)
        return rearrange(out, "(B G R) L 1 D -> B L (G R D)", B=batch, G=groups, R=repeat)
    qkv_list = [q, k, v]
    q = k = v = None
    return rearrange(pay_attention(qkv_list, attention_mask=mask, softmax_scale=scale, recycle_q=not experimental_patch), "B L H D -> B L (H D)")


def key_padding_mask(mask: Tensor) -> Tensor:
    return mask.unsqueeze(1).unsqueeze(2)


def temb(t: Tensor, dim: int, period: float = 1e4, tfactor: float = 1e3, device: torch.device = None, dtype: torch.dtype = None) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = 12
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20


def config_from_diffusers(data: dict) -> SingleMMDiTConfig:
    features = data["attention_head_dim"] * data["num_attention_heads"]
    base_mlp = int(2 * features / 3)
    return SingleMMDiTConfig(
        features=features,
        tdim=data["timestep_embed_dim"],
        txtdim=data["text_hidden_dim"],
        heads=data["num_attention_heads"],
        kvheads=data["num_key_value_heads"],
        multiplier=data["intermediate_size"] // base_mlp,
        layers=data["num_layers"],
        patch=2,
        channels=data["in_channels"] // 4,
        txtheads=data["text_num_attention_heads"],
        txtkvheads=data["text_num_key_value_heads"],
        txtlayers=data["num_text_layers"],
        theta=data["rope_theta"],
    )


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):
        out = vec + self.lin
        return out.chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat([rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)], dim=-3)


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, device=device, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        return self.forward_list([x], dtype)

    def forward_list(self, x_list: list[Tensor], dtype: torch.dtype | None = None) -> Tensor:
        x = x_list[0]
        x_list.clear()
        dtype = x.dtype if dtype is None else dtype
        x = x.float()
        out = F.rms_norm(x, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))
        del x
        return out.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, qkv_list: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        q_list = [qkv_list[0]]
        k_list = [qkv_list[1]]
        v = qkv_list[2]
        qkv_list.clear()
        return self.qnorm.forward_list(q_list), self.knorm.forward_list(k_list), v


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)
        self.features = features
        self.mlpdim = mlpdim

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        chunk_size = 0 if seq_len <= 1024 else max(128, min(seq_len, seq_len * self.features // max(2 * self.mlpdim, 1)))
        if chunk_size == 0:
            gate = F.silu(self.gate(x))
            up = self.up(x)
            gate.mul_(up)
            del up
            return self.down(gate)
        out = x.new_empty(*x.shape[:-1], self.features)
        for start in range(0, seq_len, chunk_size):
            chunk = x.narrow(-2, start, min(chunk_size, seq_len - start))
            gate = F.silu(self.gate(chunk))
            up = self.up(chunk)
            gate.mul_(up)
            del up
            chunk_out = self.down(gate)
            out.narrow(-2, start, chunk_out.shape[-2]).copy_(chunk_out)
            del chunk, gate, chunk_out
        return out


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.experimental_patch = False
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        q, k, v = self.wq(qkv), self.wk(qkv), self.wv(qkv)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        qkv_list = [q, k, v]
        q = k = v = None
        q, k, v = self.qknorm(qkv_list)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs, reference=self.experimental_patch)
        qkv_list = [q, k, v]
        q = k = v = None
        out = attention(qkv_list, mask=mask, gqa=self.gqa, experimental_patch=self.experimental_patch)
        gate = F.sigmoid(self.gate(qkv))
        out.mul_(gate)
        del gate
        return self.wo(out)


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
            if getattr(self, "_interrupt", False):
                return None
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
            if getattr(self, "_interrupt", False):
                return None
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        headdim = config.features // config.heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        self.posemb = PositionalEncoding(config.features, axes, theta=config.theta, ntk=1.0)
        self.first = nn.Linear(config.channels * config.patch**2, config.features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(config.features, config.heads, config.multiplier, config.bias, config.kvheads) for _ in range(config.layers)])
        self.tmlp = nn.Sequential(nn.Linear(config.tdim, config.features), nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features))
        self.txtfusion = TextFusionTransformer(config.txtlayers, config.txtdim, config.txtheads, config.multiplier, config.bias, config.txtkvheads)
        self.txtmlp = nn.Sequential(RMSNorm(config.txtdim), nn.Linear(config.txtdim, config.features), nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features))
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6))
        self.experimental_patch = False

    def set_experimental_patch(self, enabled: bool):
        self.experimental_patch = enabled
        for module in self.modules():
            if isinstance(module, Attention):
                module.experimental_patch = enabled

    def _embed_context(self, context: Tensor, mask: Tensor) -> Tensor | None:
        self.txtfusion._interrupt = getattr(self, "_interrupt", False)
        txtmask = key_padding_mask(mask[:, : context.shape[1]])
        context = self.txtfusion(context, mask=txtmask)
        if context is None:
            return None
        return self.txtmlp(context)

    def _build_stream(self, img: Tensor, context: Tensor, pos: Tensor, mask: Tensor, freqs: Tensor | None = None):
        txtlen, imglen = context.shape[1], img.shape[1]
        combined = img.new_empty(img.shape[0], txtlen + imglen, img.shape[-1])
        combined[:, :txtlen].copy_(context)
        combined[:, txtlen:].copy_(img)
        del context
        fulllen = combined.shape[1]
        padlen = (-fulllen) % 256
        if padlen > 0:
            combined = F.pad(combined, (0, 0, 0, padlen))
            mask = F.pad(mask, (0, padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, padlen))
        mask = key_padding_mask(mask)
        if freqs is None:
            freqs = self.posemb(pos)
            if not self.experimental_patch:
                freqs = freqs.to(combined.dtype)
        return combined, txtlen, imglen, freqs, mask

    def forward(self, img: Tensor, context: Tensor, t: Tensor, pos: Tensor, mask: Tensor | None = None) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)
        context = self._embed_context(context, mask)
        if context is None:
            return None
        combined, txtlen, imglen, freqs, mask = self._build_stream(img, context, pos, mask)
        del img, context, pos
        for block in self.blocks:
            combined = block(combined, tvec, freqs, mask)
            if getattr(self, "_interrupt", False):
                return None
            self.txtfusion._interrupt = getattr(self, "_interrupt", False)
        image_tokens = combined[:, txtlen : txtlen + imglen]
        return self.last(image_tokens, t)

    def forward_cfg(self, img: Tensor, context: Tensor, uncond_context: Tensor, t: Tensor, pos: Tensor, uncond_pos: Tensor, mask: Tensor, uncond_mask: Tensor) -> tuple[Tensor | None, Tensor | None]:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)
        context = self._embed_context(context, mask)
        if context is None:
            return None, None
        uncond_context = self._embed_context(uncond_context, uncond_mask)
        if uncond_context is None:
            return None, None
        share_freqs = pos.shape == uncond_pos.shape
        combined, txtlen, imglen, freqs, mask = self._build_stream(img, context, pos, mask)
        uncond_combined, uncond_txtlen, uncond_imglen, uncond_freqs, uncond_mask = self._build_stream(img, uncond_context, uncond_pos, uncond_mask, freqs=freqs if share_freqs else None)
        del img, context, uncond_context, pos, uncond_pos
        for block in self.blocks:
            combined = block(combined, tvec, freqs, mask)
            if getattr(self, "_interrupt", False):
                return None, None
            uncond_combined = block(uncond_combined, tvec, uncond_freqs, uncond_mask)
            if getattr(self, "_interrupt", False):
                return None, None
        return self.last(combined[:, txtlen : txtlen + imglen], t), self.last(uncond_combined[:, uncond_txtlen : uncond_txtlen + uncond_imglen], t)
