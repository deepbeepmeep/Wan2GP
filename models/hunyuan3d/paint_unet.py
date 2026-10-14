# Hunyuan3D 2.1 multiview PBR paint UNet (UNet2p5DConditionModel), ported from Tencent Hunyuan3D 2.1
# hy3dpaint/hunyuanpaintpbr/unet/{modules.py, attn_processor.py} (Copyright (C) 2025 Tencent; Tencent Hunyuan 3D 2.1 Community License,
# see models/hunyuan3d/LICENSE and Notice.txt), on a local Stable Diffusion 2.1 UNet whose blocks follow diffusers'
# UNet2DConditionModel (Apache-2.0, Copyright The HuggingFace Team) with the same parameter names.
# Modified by DeepBeepMeep for WanGP: no diffusers UNet; inference only; the self-attentions (material-specific, multiview with 3D RoPE)
# through shared.attention_kit and the cross-attentions (material tokens, DINO tokens, reference views) through pay_attention; the
# three guidance branches (unconditional, reference, full) run one after the other inside each block; the zero-scale reference
# attention of the unconditional branch is skipped (its contribution is exactly zero); the keys and values of the constant
# conditionings (material tokens, DINO tokens) are projected once per generation; the reference UNet stops once the last reference
# feature is captured; feed-forwards run by chunks of rows; the first resnet and the attentions of the first transformer block that see
# the same inputs in every branch run once; the 3D RoPE as a Triton kernel on CUDA GPUs (triton_kernels.py).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from shared import attention_kit
from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort

try:
    from . import triton_kernels as _kernels
except (ImportError, OSError, RuntimeError):
    _kernels = None

PBR_SETTING = ("albedo", "mr")
FF_CHUNK_ELEMENTS = 1 << 26  # elements of a feed-forward input chunk: its 8x GEGLU projection stays around 1 GB in FP16


def _no_norm(query, key, group):
    pass


def _attend(query, key, value, force_attention=None):
    qkv_list = [query, key, value]
    query = key = value = None
    return pay_attention(qkv_list, recycle_q=True, force_attention=force_attention)


class Attention(nn.Module):
    """diffusers Attention: heads of 64 channels, no q/k/v bias, to_out.0 with bias."""

    def __init__(self, query_dim, heads, cross_attention_dim=None):
        super().__init__()
        self.heads, self.head_dim = heads, query_dim // heads
        context_dim = cross_attention_dim or query_dim
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim)])

    def self_attention(self, x, projections=None, norm_rope=_no_norm):
        """Self-attention of x (batch, tokens, channels) through the attention kit, (batch, tokens, heads, head_dim)."""
        to_q, to_k, to_v = projections or (self.to_q, self.to_k, self.to_v)
        return attention_kit.qkv_attention([x], to_q, to_k, to_v, self.heads, self.head_dim, norm_rope)

    def keys_values(self, context):
        return self.to_k(context).unflatten(-1, (self.heads, self.head_dim)), self.to_v(context).unflatten(-1, (self.heads, self.head_dim))

    def cross_attention(self, x, key, value):
        """Attention of x to the keys/values (batch or 1, tokens, heads, head_dim) of the learned material tokens or the DINO tokens,
        followed by to_out. Always SDPA: these token sets are (nearly) constant (the albedo tokens differ by 1e-5, the unconditional DINO
        tokens are identical), which SageAttention's smoothed INT8 keys cannot represent (NaN)."""
        return self.to_out[0](_attend(self.to_q(x).unflatten(-1, (self.heads, self.head_dim)), key, value, "sdpa").flatten(2))


class MaterialProjections(nn.Module):
    """attn1.processor of the material-specific self-attention: the q/k/v/out projections of the non-albedo materials."""

    def __init__(self, dim):
        super().__init__()
        for token in PBR_SETTING[1:]:
            setattr(self, f"to_q_{token}", nn.Linear(dim, dim, bias=False))
            setattr(self, f"to_k_{token}", nn.Linear(dim, dim, bias=False))
            setattr(self, f"to_v_{token}", nn.Linear(dim, dim, bias=False))
            setattr(self, f"to_out_{token}", nn.ModuleList([nn.Linear(dim, dim)]))


class ReferenceValues(nn.Module):
    """attn_refview.processor: the value and output projections of the non-albedo materials."""

    def __init__(self, dim):
        super().__init__()
        for token in PBR_SETTING[1:]:
            setattr(self, f"to_v_{token}", nn.Linear(dim, dim, bias=False))
            setattr(self, f"to_out_{token}", nn.ModuleList([nn.Linear(dim, dim)]))


class FeedForward(nn.Module):
    """diffusers FeedForward with GEGLU (net.0.proj, net.2)."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.ModuleList([nn.Module(), nn.Identity(), nn.Linear(dim * 4, dim)])
        self.net[0].proj = nn.Linear(dim, dim * 8)

    def forward(self, x):
        """x (batch, tokens, dim) by chunks of rows (rows are independent: the values of the whole-tensor computation)."""
        out = torch.empty_like(x)
        flat, flat_out = x.view(-1, x.shape[-1]), out.view(-1, x.shape[-1])
        rows = max(1, FF_CHUNK_ELEMENTS // (x.shape[-1] * 8))
        for start in range(0, flat.shape[0], rows):
            hidden, gate = self.net[0].proj(flat[start:start + rows]).chunk(2, dim=-1)
            flat_out[start:start + rows] = self.net[2](hidden * F.gelu(gate))
            del hidden, gate
        return out


class BasicTransformerBlock(nn.Module):
    """diffusers BasicTransformerBlock (the `transformer` of Basic2p5DTransformerBlock)."""

    def __init__(self, dim, heads, cross_attention_dim, mda):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads)
        if mda:
            self.attn1.processor = MaterialProjections(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, heads, cross_attention_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)


def _rope_rows(part, first, last, cos, sin):
    """Upstream RotaryEmbedding.apply_rotary_emb on rows first:last of (batch, rows, heads, head_dim): FP32 math on interleaved pairs."""
    x = part.float()
    rotated = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return (x * cos[:, first:last] + rotated * sin[:, first:last]).to(part.dtype)


@dataclass
class Branch:
    """One classifier-free guidance branch: its reference attention scale and the DINO tokens (1, tokens, 1024) it attends to."""
    ref_scale: float
    dino_tokens: torch.Tensor
    dino_kv: dict = field(default_factory=dict)


@dataclass
class PaintContext:
    """Conditioning shared by the branches: views per material, reference features per layer, 3D RoPE tables per sequence length."""
    views: int
    references: dict
    rope: dict
    material_kv: dict = field(default_factory=dict)


class Basic2p5DTransformerBlock(nn.Module):
    def __init__(self, dim, heads, cross_attention_dim, layer_name, full):
        super().__init__()
        self.layer_name = layer_name
        self.transformer = BasicTransformerBlock(dim, heads, cross_attention_dim, mda=full)
        if full:
            self.attn_multiview = Attention(dim, heads)
            self.attn_refview = Attention(dim, heads)
            self.attn_refview.processor = ReferenceValues(dim)
            self.attn_dino = Attention(dim, heads, cross_attention_dim)

    def forward_reference(self, hidden_states, material_kv, store):
        """Reference UNet ("w" mode): the norm1 output is stored for the reference attention of the main UNet. material_kv None: the
        reference tokens are all zero, whose cross-attention output is exactly the output bias (zero keys and values)."""
        t = self.transformer
        norm_hidden_states = t.norm1(hidden_states)
        store[self.layer_name] = norm_hidden_states.reshape(1, -1, norm_hidden_states.shape[-1])
        hidden_states = t.attn1.to_out[0](t.attn1.self_attention(norm_hidden_states).flatten(2)) + hidden_states
        del norm_hidden_states
        if material_kv is None:
            hidden_states = hidden_states + t.attn2.to_out[0].bias
        else:
            hidden_states = t.attn2.cross_attention(t.norm2(hidden_states), *material_kv) + hidden_states
        return t.ff(t.norm3(hidden_states)).add_(hidden_states)

    def _self_attention(self, norm_hidden_states, hidden_states):
        """Material-specific self-attention, one material at a time over its views, added to the hidden states (a new tensor)."""
        t, views = self.transformer, norm_hidden_states.shape[0] // len(PBR_SETTING)
        attn_output = torch.empty_like(hidden_states)
        for index, token in enumerate(PBR_SETTING):
            if token == "albedo":
                projections, out_proj = (t.attn1.to_q, t.attn1.to_k, t.attn1.to_v), t.attn1.to_out[0]
            else:
                proc = t.attn1.processor
                projections, out_proj = (getattr(proc, f"to_q_{token}"), getattr(proc, f"to_k_{token}"), getattr(proc, f"to_v_{token}")), getattr(proc, f"to_out_{token}")[0]
            attn_output[index * views:(index + 1) * views] = out_proj(t.attn1.self_attention(norm_hidden_states[index * views:(index + 1) * views], projections).flatten(2))
        return attn_output.add_(hidden_states)

    def _add_reference(self, hidden_states, norm_hidden_states, ctx, ref_scale):
        """Reference attention: albedo queries of all views, keys of the reference features, one value set per material."""
        ref, views, materials = self.attn_refview, ctx.views, len(PBR_SETTING)
        heads, head_dim, tokens, dim = ref.heads, ref.head_dim, hidden_states.shape[1], hidden_states.shape[2]
        reference = ctx.references[self.layer_name]
        query = ref.to_q(norm_hidden_states[:views].reshape(1, views * tokens, dim)).unflatten(-1, (heads, head_dim))
        key = ref.to_k(reference).unflatten(-1, (heads, head_dim))
        # upstream concatenates the materials' values and reads them as heads of 2 x head_dim: each head's halves feed one material
        values = torch.cat([ref.to_v(reference), ref.processor.to_v_mr(reference)], dim=-1).unflatten(-1, (heads, 2 * head_dim))
        del reference
        for index, token in enumerate(PBR_SETTING):
            out_proj = ref.to_out[0] if token == "albedo" else getattr(ref.processor, f"to_out_{token}")[0]
            last = index == materials - 1
            out = out_proj(_attend(query if last else query.clone(), key, values[..., index * head_dim:(index + 1) * head_dim].contiguous()).flatten(2))
            if ref_scale != 1:
                out.mul_(ref_scale)
            hidden_states[index * views:(index + 1) * views].add_(out.view(views, tokens, dim))
            del out
        del query, key, values

    def _multiview(self, norm_hidden_states, ctx):
        """Multiview attention: the views of each material in one sequence, 3D RoPE from the position maps."""
        materials, views = len(PBR_SETTING), ctx.views
        tokens, dim = norm_hidden_states.shape[1], norm_hidden_states.shape[2]
        cos, sin = ctx.rope[views * tokens]
        fused = _kernels is not None and _kernels.supported(norm_hidden_states.device)

        def rope(query, key, group):
            for tensor in (query, key):
                if tensor is None:
                    continue
                if fused:
                    _kernels.rope_(tensor, cos, sin)
                else:
                    attention_kit.rows_(tensor, lambda part, first, last: _rope_rows(part, first, last, cos, sin))

        mv = self.attn_multiview
        return mv.to_out[0](mv.self_attention(norm_hidden_states.view(materials, views * tokens, dim), norm_rope=rope).flatten(2)).view(materials * views, tokens, dim)

    def _material_tokens(self, hidden_states, ctx):
        """norm2 of the hidden states, returned after the cross-attention to the learned material tokens is added to them."""
        t = self.transformer
        norm_hidden_states = t.norm2(hidden_states)
        if self.layer_name not in ctx.material_kv:
            tokens_kv = t.attn2.keys_values(ctx.material_tokens)
            ctx.material_kv[self.layer_name] = tuple(x.repeat_interleave(ctx.views, 0) for x in tokens_kv)
        hidden_states.add_(t.attn2.cross_attention(norm_hidden_states, *ctx.material_kv[self.layer_name]))
        return norm_hidden_states

    def _dino_and_ff(self, hidden_states, norm_hidden_states, branch):
        """The cross-attention to the branch DINO tokens (from the same norm2 output), then the feed-forward."""
        t = self.transformer
        if self.layer_name not in branch.dino_kv:
            branch.dino_kv[self.layer_name] = self.attn_dino.keys_values(branch.dino_tokens)
        hidden_states.add_(self.attn_dino.cross_attention(norm_hidden_states, *branch.dino_kv[self.layer_name]))
        del norm_hidden_states
        return hidden_states.add_(t.ff(t.norm3(hidden_states)))

    def forward(self, hidden_states, ctx, branch):
        """Main UNet ("r" mode) for one guidance branch: hidden_states (materials x views, tokens, dim), materials outermost."""
        norm_hidden_states = self.transformer.norm1(hidden_states)
        hidden_states = self._self_attention(norm_hidden_states, hidden_states)
        if branch.ref_scale != 0:
            self._add_reference(hidden_states, norm_hidden_states, ctx, branch.ref_scale)
        if ctx.views > 1:
            hidden_states.add_(self._multiview(norm_hidden_states, ctx))
        del norm_hidden_states
        return self._dino_and_ff(hidden_states, self._material_tokens(hidden_states, ctx), branch)

    def forward_branches(self, hidden_states, ctx, branches):
        """forward() of every branch for the same input hidden states (the first block of the UNet): the norm1 output, the self-attention
        and the multiview attention (which reads the norm1 output) are computed once, the reference attention once per reference scale,
        the material tokens cross-attention once per resulting state; only the DINO cross-attention and the feed-forward run per branch.
        Each branch state gets the operations of forward() in the same order: the same values."""
        norm_hidden_states = self.transformer.norm1(hidden_states)
        attended = self._self_attention(norm_hidden_states, hidden_states)
        multiview = self._multiview(norm_hidden_states, ctx) if ctx.views > 1 else None
        scales = list(dict.fromkeys(branch.ref_scale for branch in branches))
        states = {}
        for scale in scales:
            state = attended if scale == scales[-1] else attended.clone()
            if scale != 0:
                self._add_reference(state, norm_hidden_states, ctx, scale)
            if multiview is not None:
                state.add_(multiview)
            states[scale] = [state, self._material_tokens(state, ctx), sum(branch.ref_scale == scale for branch in branches)]
        del norm_hidden_states, attended, multiview
        outputs = []
        for branch in branches:
            shared = states[branch.ref_scale]
            shared[2] -= 1
            last = shared[2] == 0
            outputs.append(self._dino_and_ff(shared[0] if last else shared[0].clone(), shared[1], branch))
            if last:
                del states[branch.ref_scale], shared
        return outputs


class Transformer2DModel(nn.Module):
    """diffusers Transformer2DModel with linear projections and one transformer block."""

    def __init__(self, channels, heads, cross_attention_dim, layer_name, full):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList([Basic2p5DTransformerBlock(channels, heads, cross_attention_dim, layer_name, full)])
        self.proj_out = nn.Linear(channels, channels)

    def _project_in(self, hidden_states):
        batch, channels, height, width = hidden_states.shape
        return self.proj_in(self.norm(hidden_states).permute(0, 2, 3, 1).reshape(batch, height * width, channels))

    def _project_out(self, hidden_states, residual):
        batch, channels, height, width = residual.shape
        return self.proj_out(hidden_states).reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous() + residual

    def forward(self, hidden_states, run):
        return self._project_out(run(self.transformer_blocks[0], self._project_in(hidden_states)), hidden_states)

    def forward_branches(self, hidden_states, ctx, branches):
        """forward() of every guidance branch for the same input (Basic2p5DTransformerBlock.forward_branches)."""
        states = self.transformer_blocks[0].forward_branches(self._project_in(hidden_states), ctx, branches)
        return [self._project_out(states.pop(0), hidden_states) for _ in branches]


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=1280):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, input_tensor, temb):
        hidden_states = self.conv1(F.silu(self.norm1(input_tensor)))
        hidden_states = hidden_states + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())
        return (input_tensor + hidden_states) / 1.0


class Sampler(nn.Module):
    """Downsample2D (strided convolution) or Upsample2D (nearest x2 then convolution)."""

    def __init__(self, channels, down):
        super().__init__()
        self.down = down
        self.conv = nn.Conv2d(channels, channels, 3, stride=2 if down else 1, padding=1)

    def forward(self, x):
        if not self.down:
            x = F.interpolate(x.contiguous() if x.shape[0] >= 64 else x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetBlock(nn.Module):
    """Down, mid or up block of the SD 2.1 UNet: resnets interleaved with transformers, and a resampler."""

    def __init__(self, kind, index, in_channels, out_channels, prev_channels, heads, cross_attention_dim, attention, sampler, full):
        super().__init__()
        self.kind = kind
        if kind == "down":
            self.resnets = nn.ModuleList([ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels) for i in range(2)])
        elif kind == "mid":
            self.resnets = nn.ModuleList([ResnetBlock2D(in_channels, in_channels) for _ in range(2)])
        else:
            skip_channels = (out_channels, out_channels, in_channels)
            self.resnets = nn.ModuleList([ResnetBlock2D((prev_channels if i == 0 else out_channels) + skip_channels[i], out_channels) for i in range(3)])
        if attention:
            count, channels = (1, in_channels) if kind == "mid" else (len(self.resnets), out_channels)
            prefix = "mid" if kind == "mid" else f"{kind}_{index}"
            self.attentions = nn.ModuleList([Transformer2DModel(channels, heads, cross_attention_dim, f"{prefix}_{i}_0", full) for i in range(count)])
        else:
            self.attentions = None
        self.sampler = sampler
        if sampler:
            setattr(self, "downsamplers" if kind == "down" else "upsamplers", nn.ModuleList([Sampler(out_channels, kind == "down")]))

    def forward(self, hidden_states, temb, run, res_samples=None, start=0):
        """start: the first resnet to run (hidden_states is then the output of the attention before it)."""
        if self.kind == "mid":
            hidden_states = self.attentions[0](self.resnets[0](hidden_states, temb), run)
            return self.resnets[1](hidden_states, temb)
        outputs = []
        for i, resnet in enumerate(self.resnets[start:], start):
            if self.kind == "up":
                hidden_states = torch.cat([hidden_states, res_samples.pop()], dim=1)
            hidden_states = resnet(hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states, run)
            outputs.append(hidden_states)
        if self.sampler:
            hidden_states = (self.downsamplers if self.kind == "down" else self.upsamplers)[0](hidden_states)
            outputs.append(hidden_states)
        return (hidden_states, outputs) if self.kind == "down" else hidden_states


def timestep_embedding(timesteps, dim=320):
    """diffusers get_timestep_embedding with flip_sin_to_cos=True and downscale_freq_shift=0."""
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    emb = timesteps[:, None].float() * torch.exp(exponent / (half_dim - 0))[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)


class SD21UNet(nn.Module):
    """Stable Diffusion 2.1 UNet2DConditionModel layout: channels 320/640/1280/1280, heads 5/10/20/20 of 64 channels."""

    def __init__(self, in_channels, full, cross_attention_dim=1024):
        super().__init__()
        channels, heads = (320, 640, 1280, 1280), (5, 10, 20, 20)
        self.conv_in = nn.Conv2d(in_channels, 320, 3, padding=1)
        self.time_embedding = nn.Module()
        self.time_embedding.linear_1 = nn.Linear(320, 1280)
        self.time_embedding.linear_2 = nn.Linear(1280, 1280)
        self.down_blocks = nn.ModuleList([UNetBlock("down", i, channels[i - 1] if i else 320, channels[i], None, heads[i], cross_attention_dim, i < 3, i < 3, full)
                                          for i in range(4)])
        self.mid_block = UNetBlock("mid", 0, 1280, 1280, None, 20, cross_attention_dim, True, False, full)
        up_channels, up_heads = channels[::-1], heads[::-1]
        self.up_blocks = nn.ModuleList([UNetBlock("up", i, up_channels[min(i + 1, 3)], up_channels[i], up_channels[i - 1] if i else 1280, up_heads[i],
                                                  cross_attention_dim, i > 0, i < 3, full) for i in range(4)])
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-5)
        self.conv_out = nn.Conv2d(320, 4, 3, padding=1)

    def embed_time(self, timestep, batch, dtype):
        t_emb = timestep_embedding(timestep.expand(batch)).to(dtype)
        return self.time_embedding.linear_2(F.silu(self.time_embedding.linear_1(t_emb)))


class ReferenceUNet(SD21UNet):
    """unet_dual: run once on the reference image latents at timestep 0 to capture the norm1 features of every transformer block."""

    LAST_LAYER = "up_3_2_0"

    def __init__(self):
        super().__init__(4, full=False)

    def forward(self, latents, material_tokens):
        """latents (1, 4, h, w) of the reference image; material_tokens (1, 77, 1024) (learned_text_clip_ref). Returns {layer: features}."""
        store, zero_tokens = {}, not material_tokens.any()

        def run(block, hidden_states):
            material_kv = None if zero_tokens else block.transformer.attn2.keys_values(material_tokens)
            hidden_states = block.forward_reference(hidden_states, material_kv, store)
            if block.layer_name == self.LAST_LAYER:
                raise StopIteration
            return hidden_states

        temb = self.embed_time(torch.zeros(1, dtype=torch.int64, device=latents.device), latents.shape[0], latents.dtype)
        hidden_states = self.conv_in(latents)
        res_samples = [hidden_states]
        try:
            for block in self.down_blocks:
                hidden_states, outputs = block(hidden_states, temb, run)
                res_samples += outputs
                check_abort()
            hidden_states = self.mid_block(hidden_states, temb, run)
            for block in self.up_blocks:
                hidden_states = block(hidden_states, temb, run, res_samples)
                check_abort()
        except StopIteration:
            pass
        return store


class PaintUNet(SD21UNet):
    """The main UNet of UNet2p5DConditionModel (`unet`), with its learned material tokens and DINO projector."""

    def __init__(self):
        super().__init__(12, full=True)
        self.learned_text_clip_albedo = nn.Parameter(torch.empty(77, 1024))
        self.learned_text_clip_mr = nn.Parameter(torch.empty(77, 1024))
        self.learned_text_clip_ref = nn.Parameter(torch.empty(77, 1024))
        self.image_proj_model_dino = nn.Module()
        self.image_proj_model_dino.proj = nn.Linear(1536, 4 * 1024)
        self.image_proj_model_dino.norm = nn.LayerNorm(1024)

    def project_dino(self, dino_hidden_states):
        """ImageProjModel: each DINO token becomes 4 context tokens, (1, tokens * 4, 1024)."""
        proj = self.image_proj_model_dino
        return proj.norm(proj.proj(dino_hidden_states[0]).reshape(-1, 4, 1024)).reshape(1, -1, 1024)

    def forward(self, sample, timestep, ctx, branches):
        """sample (materials x views, 12, h, w): noisy latents with the normal and position latents. One prediction per branch, the
        branches run one after the other inside each block."""
        temb = self.embed_time(timestep, sample.shape[0], sample.dtype)
        ctx.material_tokens = torch.stack([self.learned_text_clip_albedo, self.learned_text_clip_mr])
        conv_in = self.conv_in(sample)
        # the branches see the same inputs until the first transformer block: its first resnet and attentions that do not depend on the
        # branch run once
        first = self.down_blocks[0]
        states = first.attentions[0].forward_branches(first.resnets[0](conv_in, temb), ctx, branches)
        res = [[conv_in, state] for state in states]
        del conv_in
        runs = [lambda block, hidden_states, branch=branch: block(hidden_states, ctx, branch) for branch in branches]
        for index, block in enumerate(self.down_blocks):
            for b in range(len(branches)):
                states[b], outputs = block(states[b], temb, runs[b], start=1 if index == 0 else 0)
                res[b] += outputs
            check_abort()
        for b in range(len(branches)):
            states[b] = self.mid_block(states[b], temb, runs[b])
        check_abort()
        for block in self.up_blocks:
            for b in range(len(branches)):
                states[b] = block(states[b], temb, runs[b], res[b])
            check_abort()
        return [self.conv_out(F.silu(self.conv_norm_out(state))) for state in states]
