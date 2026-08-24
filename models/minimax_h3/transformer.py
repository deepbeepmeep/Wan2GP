# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for WanGP: raw-checkpoint names, MMGP QKV splitting/offload, pruned
# AdaLN curves, shared attention backends, chunked FFNs, and early tensor release.

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import pay_attention

from .interrupt import GenerationInterrupted
from .sol_attention import MiniMaxH3SolAttention
from .components.packing import (
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PreparedReference,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)


VISUAL_COND_TIMESTEP = MINIMAX_H3_KEYFRAME_NOISE_AUG
AUDIO_COND_TIMESTEP = 1.0


def patchify_video(latent, patch_size=(1, 2, 2)):
    return patchify_video_latents(latent, patch_size)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    return unpatchify_video_tokens(rows, t, h * patch_size[1], w * patch_size[2], c, patch_size)


def pack_audio(latent):
    return latent[0].permute(1, 2, 0).reshape(-1, latent.shape[1]).contiguous()


def unpack_audio(rows, ch=2):
    return unpack_audio_tokens(rows, rows.shape[0] // ch).permute(1, 0, 2).unsqueeze(0)


def _split_interleaved_qkv(src, dim, split_sizes, context):
    info = context["info"]
    heads, head_dim = info["num_attention_heads"], info["attention_head_dim"]
    grouped = src.reshape(heads, 3, head_dim, *src.shape[1:])
    return [grouped[:, index].reshape(split_sizes[index], *src.shape[1:]).contiguous() for index in range(3)]


def get_linear_split_map(inner_size, num_attention_heads=56, attention_head_dim=128):
    return {"qkv_proj": {"mapped_modules": ["q_proj", "k_proj", "v_proj"], "split_sizes": [inner_size] * 3,
                         "num_attention_heads": num_attention_heads, "attention_head_dim": attention_head_dim,
                         "split_handlers": {"weight": _split_interleaved_qkv}}}


def _take(x_list):
    x = x_list[0]
    x_list.clear()
    return x


def _to_dtype(x_list, dtype):
    x = _take(x_list)
    return x if x.dtype == dtype else x.to(dtype)


def _rope_table(angles, dtype):
    return torch.stack((angles.cos(), angles.sin()), dim=-1).unsqueeze(0).unsqueeze(2).to(dtype)


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out, dtype=None, device=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True, dtype=dtype, device=device)
        self.proj_out = nn.Linear(hidden, out, bias=True, dtype=dtype, device=device)

    def forward(self, timestep):
        half = self.freq_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        angles = timestep.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.proj_out(F.silu(self.proj_in(embedding)))


class RotaryEmbedding(nn.Module):
    def __init__(self, freq_dim, theta=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32, device=device) / (2 * freq_dim)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, position_ids, device):
        positions = position_ids.to(device=device, dtype=torch.float32)
        frequencies = positions.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        return torch.cat(frequencies.unbind(dim=1), dim=-1)


class MLP(nn.Module):
    def __init__(self, hidden, ffn, chunk_size=0, dtype=None, device=None):
        super().__init__()
        self.hidden = hidden
        self.ffn = ffn
        self.chunk_size = int(chunk_size)
        self.fc1 = nn.Linear(hidden, 2 * ffn, bias=False, dtype=dtype, device=device)
        self.fc2 = nn.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)

    def _project(self, x_list):
        x = _take(x_list)
        expanded = self.fc1(x)
        del x
        gate, value = expanded.chunk(2, dim=-1)
        F.silu(gate, inplace=True).mul_(value)
        del expanded, value
        return self.fc2(gate)

    def forward(self, x_list):
        x = _take(x_list)
        chunk_size = self.chunk_size
        if chunk_size > 0:
            chunk_size = max(1, x.shape[0] * self.hidden // (2 * self.ffn))
        if chunk_size <= 0 or x.shape[0] <= chunk_size:
            return self._project([x])
        for start in range(0, x.shape[0], chunk_size):
            output = self._project([x[start:start + chunk_size]])
            x[start:start + output.shape[0]].copy_(output)
            del output
        return x


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, sol_attention=None, dtype=None, device=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.sol_attention = sol_attention
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * inner, bias=False, dtype=dtype, device=device)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x_list, rope=None, transformer_options=None):
        x = _take(x_list)
        seq_len = x.shape[0]
        use_sol = self.sol_attention is not None and self.sol_attention.use_for_layer(seq_len)
        split_qkv = hasattr(self, "q_proj")
        if split_qkv:
            query = self.q_proj(x).view(1, seq_len, self.heads, self.head_dim)
            if not use_sol:
                query = self.q_norm(query)
            key = self.k_proj(x).view(1, seq_len, self.heads, self.head_dim)
            if not use_sol:
                key = self.k_norm(key)
            value = self.v_proj(x).view(1, seq_len, self.heads, self.head_dim)
        else:
            qkv = self.qkv_proj(x)
            query, key, value = qkv.split(self.heads * self.head_dim, dim=-1)
            query = query.view(seq_len, self.heads, self.head_dim)
            key = key.view(seq_len, self.heads, self.head_dim)
            value = value.view(seq_len, self.heads, self.head_dim)
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
            if not use_sol:
                value = value.clone()
            del qkv
        del x
        if use_sol:
            from shared.sol_attn import qk_rms_norm_rope_
            qk_rms_norm_rope_(query, key, self.q_norm.weight, self.k_norm.weight, rope, self.q_norm.eps)
        elif not split_qkv:
            query, key = self.q_norm(query), self.k_norm(key)
        qkv_list = [query, key, value]
        del query, key, value
        if rope is not None and not use_sol:
            pairs = rope.shape[-2]
            cosine, sine = rope[..., 0], rope[..., 1]
            scratch = torch.empty_like(qkv_list[0][..., :pairs])
            for index in range(2):
                tensor = qkv_list[index]
                first, second = tensor[..., :pairs], tensor[..., pairs:2 * pairs]
                scratch.copy_(first)
                first.mul_(cosine).addcmul_(second, sine, value=-1)
                second.mul_(cosine).addcmul_(scratch, sine)
            del scratch, tensor, first, second
        attention = pay_attention(qkv_list, recycle_q=True) if self.sol_attention is None else self.sol_attention(qkv_list, use_sol)
        output = attention.reshape(seq_len, -1)
        return self.out_proj(output)


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, ffn_chunk_size=0, dtype=None, device=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden, ffn, ffn_chunk_size, dtype=dtype, device=device)

    def forward(self, x_list):
        hidden = _take(x_list)
        branch = self.attn([self.norm1(hidden)])
        branch.add_(hidden)
        del hidden
        residual = branch
        branch = self.mlp([self.norm2(residual)])
        branch.add_(residual)
        del residual
        return branch


class TokenRefiner(nn.Module):
    def __init__(self, layers, hidden, heads, head_dim, ffn, eps, qk_eps, out_eps, dtype=None, device=None):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps,
                                                   dtype=dtype, device=device) for _ in range(layers)])
        self.final_norm = nn.RMSNorm(hidden, eps=out_eps, dtype=dtype, device=device)
        self._interrupt = False

    def forward(self, x_list):
        hidden = _take(x_list)
        for block in self.blocks:
            if self._interrupt:
                raise GenerationInterrupted
            hidden = block([hidden])
        return self.final_norm(hidden)


class AdalnProj(nn.Module):
    def __init__(self, time_dim, hidden, expand, modalities=3, apply_silu=True, dtype=None, device=None):
        super().__init__()
        self.hidden = hidden
        self.expand = expand
        self.modalities = modalities
        self.apply_silu = apply_silu
        self.linear = nn.Linear(time_dim, expand * modalities * hidden, bias=True, dtype=dtype, device=device)

    def forward(self, temb):
        if self.apply_silu:
            temb = F.silu(temb)
        output = self.linear(temb.to(self.linear.weight.dtype)).view(-1, self.expand * self.hidden)
        return output.chunk(self.expand, dim=-1)


def _modulate(hidden, shift_scale_list, segments):
    shift, scale = shift_scale_list
    shift_scale_list.clear()
    shift, scale = shift.to(hidden.dtype), scale.to(hidden.dtype)
    for start, stop, row in segments:
        hidden[start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    return hidden


def _gated_residual(hidden_list, gate_list, branch_list, segments):
    hidden = _take(hidden_list)
    gate, branch = gate_list[0].to(hidden.dtype), _take(branch_list)
    gate_list.clear()
    for start, stop, row in segments:
        branch[start:stop].mul_(gate[row]).add_(hidden[start:stop])
        hidden[start:stop].copy_(branch[start:stop])
    return hidden


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, time_dim, eps, qk_eps, apply_silu=True,
                 adaln_dtype=None, ffn_chunk_size=2048, sol_attention=None, dtype=None, device=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, sol_attention=sol_attention, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden, ffn, ffn_chunk_size, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(time_dim, hidden, 6, apply_silu=apply_silu,
                                    dtype=adaln_dtype or dtype, device=device)

    @staticmethod
    def _gated_branch(hidden, gate, branch, segments, signature_stride=0, signature=None):
        for start, stop, row in segments:
            branch[start:stop].mul_(gate[row])
        if signature_stride:
            # Sample the existing gated branch before it is released; never materialize a full block residual copy.
            sampled = branch.reshape(-1)[::signature_stride]
            if signature is None:
                signature = sampled.clone()
            else:
                signature.add_(sampled)
            hidden.add_(branch)
        else:
            branch.add_(hidden)
            hidden.copy_(branch)
        return hidden, signature

    def forward(self, x_list, temb, segments, rope, residual_signature_elements=0):
        residual_list = [_take(x_list)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        h_list = [_modulate(self.norm1(residual_list[0]), [shift_msa, scale_msa], segments)]
        if not residual_signature_elements:
            residual_list = [_gated_residual(residual_list, [gate_msa], [self.attn(h_list, rope=rope)], segments)]
            h_list = [_modulate(self.norm2(residual_list[0]), [shift_mlp, scale_mlp], segments)]
            return _gated_residual(residual_list, [gate_mlp], [self.mlp(h_list)], segments)

        hidden = _take(residual_list)
        signature_stride = max(1, math.ceil(hidden.numel() / residual_signature_elements))
        branch = self.attn(h_list, rope=rope)
        hidden, signature = self._gated_branch(hidden, gate_msa.to(hidden.dtype), branch, segments, signature_stride)
        del branch
        h_list = [_modulate(self.norm2(hidden), [shift_mlp, scale_mlp], segments)]
        branch = self.mlp(h_list)
        hidden, signature = self._gated_branch(hidden, gate_mlp.to(hidden.dtype), branch, segments, signature_stride, signature)
        return hidden, signature


class FinalLayer(nn.Module):
    def __init__(self, hidden, time_dim, video_dim, audio_dim, eps, apply_silu=True,
                 adaln_dtype=None, dtype=None, device=None):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(time_dim, hidden, 2, modalities=1, apply_silu=apply_silu,
                                    dtype=adaln_dtype or dtype, device=device)
        self.video_out = nn.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)

    def _head(self, h_list, shift, scale, row, output):
        value = self.norm(_take(h_list)).to(scale.dtype)
        if isinstance(row, list):
            for start, stop, index in row:
                value[start:stop].mul_(1.0 + scale[index]).add_(shift[index])
        else:
            value.mul_(1.0 + scale[row]).add_(shift[row])
        value = value.to(output.weight.dtype)
        return output(value)

    def forward(self, x_list, temb, video_segment, audio_segment):
        hidden = _take(x_list)
        a0, a1, arow = audio_segment
        v0, v1, vrow = video_segment
        shift, scale = self.adaln_proj(temb)
        h_list = [hidden]
        del hidden
        audio_list = [h_list[0][a0:a1]]
        video_list = [_take(h_list)[v0:v1]]
        audio = self._head(audio_list, shift, scale, arow, self.audio_out)
        video = self._head(video_list, shift, scale, vrow, self.video_out)
        return video, audio


def _prepared_references(refs):
    prepared = []
    for ref in refs or ():
        kind = ref["kind"]
        if kind == "video_audio":
            kind = "video"
        prepared.append(MiniMaxH3PreparedReference(
            kind=kind,
            has_audio=ref.get("ref_audio_t", 0) > 0,
            num_latent_frames=ref.get("latent_t", 1),
            latent_height=ref.get("latent_h", 0),
            latent_width=ref.get("latent_w", 0),
            num_audio_latents=ref.get("ref_audio_t", 0),
        ))
    return prepared


class MiniMaxH3Model(nn.Module):
    def preprocess_loras(self, model_type, state_dict):
        diffusers_format = any(key.startswith(("transformer_blocks.", "token_refiner.refiner_blocks.",
                                               "time_embedder.linear_", "audio_proj_in.", "proj_in.",
                                               "context_embedder.", "norm_out.", "audio_proj_out.", "proj_out."))
                               for key in state_dict)
        converted = {}
        for key, value in state_dict.items():
            if key.startswith("lora_unet_"):
                path, suffix = key[len("lora_unet_"):].split(".", 1)
                key = path.replace("blocks_", "blocks.", 1).replace("_attn_", ".attn.").replace("_mlp_", ".mlp.") + "." + suffix
            if diffusers_format:
                diffusers_fc1 = ".ff.net.0.proj." in key
                for source, target in (("token_refiner.refiner_blocks.", "token_refiner.blocks."),
                                       ("transformer_blocks.", "blocks."),
                                       ("time_embedder.linear_1.", "time_embedder.proj_in."),
                                       ("time_embedder.linear_2.", "time_embedder.proj_out."),
                                       ("audio_proj_in.", "audio_patch_proj."), ("proj_in.", "video_patch_proj."),
                                       ("context_embedder.", "condition_proj."),
                                       ("norm_out.norm.", "final_layer.norm."),
                                       ("norm_out.linear.", "final_layer.adaln_proj.linear."),
                                       ("audio_proj_out.", "final_layer.audio_out."), ("proj_out.", "final_layer.video_out.")):
                    if key.startswith(source):
                        key = target + key[len(source):]
                        break
                for source, target in ((".attn.norm_q.", ".attn.q_norm."), (".attn.norm_k.", ".attn.k_norm."),
                                       (".attn.to_out.0.", ".attn.out_proj."), (".attn.to_q.", ".attn.q_proj."),
                                       (".attn.to_k.", ".attn.k_proj."), (".attn.to_v.", ".attn.v_proj."),
                                       (".ff.net.0.proj.", ".mlp.fc1."), (".ff.net.2.", ".mlp.fc2.")):
                    key = key.replace(source, target)
                if diffusers_fc1 and key.endswith((".lora_B.weight", ".lora_B.default.weight", ".lora_up.weight",
                                                    ".lora_up.default.weight", ".lora.B.weight", ".lora.B.default.weight",
                                                    ".lora.up.weight", ".lora.up.default.weight")):
                    value = torch.cat(value.chunk(2, dim=0)[::-1], dim=0).contiguous()
            converted[key] = value
        from .lora_affine import convert_adaln_loras

        start = time.perf_counter()
        count, architecture, source_width, target_width = convert_adaln_loras(
            model_type, converted, self.adaln_t_table if self.use_adaln_curves else None)
        if count:
            source = f"full AdaLN width {source_width}" if source_width == 2688 else f"{architecture.upper()} pruned AdaLN width {source_width}"
            target = f"full AdaLN width {target_width}" if target_width == 2688 else f"{architecture.upper()} pruned AdaLN width {target_width}"
            print(f"MiniMax H3 LoRA: converted {count} AdaLN adapters from {source} to {target} in {time.perf_counter() - start:.2f}s")
        if hasattr(self.blocks[0].attn, "q_proj"):
            return converted
        for down_suffix, up_suffix in (("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight"),
                                       ("lora_A.default.weight", "lora_B.default.weight"),
                                       ("lora_down.default.weight", "lora_up.default.weight"),
                                       ("lora.A.weight", "lora.B.weight"), ("lora.down.weight", "lora.up.weight"),
                                       ("lora.A.default.weight", "lora.B.default.weight"),
                                       ("lora.down.default.weight", "lora.up.default.weight")):
            marker = "q_proj." + down_suffix
            for key in [key for key in converted if key.endswith(marker)]:
                prefix = key[:-len(marker)]
                down, up, scales = [], [], []
                for projection in ("q_proj", "k_proj", "v_proj"):
                    down.append(converted.pop(prefix + projection + "." + down_suffix))
                    up.append(converted.pop(prefix + projection + "." + up_suffix))
                    alpha = converted.pop(prefix + projection + ".alpha", None)
                    scales.append(1.0 if alpha is None else float(alpha) / down[-1].shape[0])
                converted[prefix + "qkv_proj." + down_suffix] = torch.cat(down)
                converted[prefix + "qkv_proj." + up_suffix] = torch.block_diag(*(weight * scale for weight, scale in zip(up, scales)))
        return converted

    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5,
                 final_norm_eps=1e-5, sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 ffn_chunk_size=2048, adaln_curve_grid=None, image_model=None,
                 dtype=None, device=None, **kwargs):
        super().__init__()
        self._interrupt = False
        self.cache = None
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.attention_inner_size = num_attention_heads * attention_head_dim
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.use_adaln_curves = adaln_curve_grid is not None
        video_dim = latents_dim * math.prod(self.patch_size)
        self.video_patch_proj = nn.Linear(video_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32, device=device))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                              dtype=torch.float32, device=device)
        self.rope = RotaryEmbedding(rope_inv_freq_len, rope_theta, device=device)
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps, dtype=dtype, device=device)
        self.sol_attention = MiniMaxH3SolAttention()
        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                                               time_embed_dim, norm_eps, qk_norm_eps, **curve,
                                               ffn_chunk_size=ffn_chunk_size, sol_attention=self.sol_attention,
                                               dtype=dtype, device=device) for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_dim, audio_latents_dim,
                                      final_norm_eps, **curve, dtype=dtype, device=device)
        fp32_modules = [self.video_patch_proj, self.audio_patch_proj, self.final_layer.video_out, self.final_layer.audio_out]
        if self.use_adaln_curves:
            fp32_modules.extend(block.adaln_proj.linear for block in self.blocks)
            fp32_modules.append(self.final_layer.adaln_proj.linear)
        else:
            fp32_modules.extend((self.time_embedder.proj_in, self.time_embedder.proj_out))
        for module in fp32_modules:
            module._lock_dtype = torch.float32

    def preprocess_text_embeds(self, text_states):
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        self.token_refiner._interrupt = self._interrupt
        return self.token_refiner([self.condition_proj(text_states[0])]).unsqueeze(0)

    def _check_interrupt(self):
        if self._interrupt:
            raise GenerationInterrupted

    def _layout(self, text_tags, latent_t, latent_h, latent_w, audio_t, payload):
        target_spatial_context = payload.get("target_spatial_context")
        signature = (text_tags.numel(), latent_t, latent_h, latent_w, audio_t,
                     payload["fps"],
                     target_spatial_context,
                     payload.get("target_audio_condition_latents", 0),
                     payload.get("target_video_condition_frames", 0),
                     tuple((k["anchor"], k["latent_frame_count"], k.get("frame_index")) for k in payload.get("keyframes") or ()),
                     tuple((k["anchor"], k["latent_frame_count"]) for k in payload.get("audio_keyframes") or ()),
                     tuple((r["kind"], r.get("latent_t"), r.get("latent_h"), r.get("latent_w"), r.get("ref_audio_t"))
                           for r in payload.get("refs") or ()))
        if payload.get("layout_signature") == signature:
            return payload["layout"]
        with torch.device("cpu"):
            video_time_scale = 24.0 / payload["fps"]
            anchors = tuple((keyframe["anchor"], keyframe["latent_frame_count"], keyframe.get("frame_index"))
                            for keyframe in payload.get("keyframes") or ())
            audio_anchors = tuple((keyframe["anchor"], keyframe["latent_frame_count"])
                                  for keyframe in payload.get("audio_keyframes") or ())
            target_audio_condition_latents = payload.get("target_audio_condition_latents", 0)
            target_video_condition_frames = payload.get("target_video_condition_frames", 0)
            if payload.get("refs"):
                layout = build_ref2va_packed_sequence(text_tags, _prepared_references(payload["refs"]), latent_t,
                                                      latent_h, latent_w, audio_t, self.patch_size, video_time_scale,
                                                      keyframe_anchors=anchors, audio_condition_anchors=audio_anchors,
                                                      target_condition_audio_latents=target_audio_condition_latents,
                                                      target_condition_video_frames=target_video_condition_frames,
                                                      target_spatial_context=target_spatial_context)
            else:
                layout = build_packed_sequence(text_tags, latent_t, latent_h, latent_w, audio_t, self.patch_size,
                                               anchors, video_time_scale, audio_condition_anchors=audio_anchors,
                                               target_condition_audio_latents=target_audio_condition_latents,
                                               target_condition_video_frames=target_video_condition_frames,
                                               target_spatial_context=target_spatial_context)
        payload["layout_signature"], payload["layout"] = signature, layout
        return layout

    def _time_embedding(self, timesteps):
        if not self.use_adaln_curves:
            return self.time_embedder(timesteps)
        table = self.adaln_t_table
        position = timesteps.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))

    def forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload, spectrum=None, first_block_cache=None):
        device, dtype = video_x.device, self.dtype or next(self.blocks.parameters()).dtype
        video_dtype, audio_dtype = video_x.dtype, audio_x.dtype
        _, _, latent_t, latent_h, latent_w = video_x.shape
        sigma_video = sigma_video.flatten()
        if sigma_video.numel() not in (1, latent_t):
            raise ValueError(f"MiniMax H3 received {sigma_video.numel()} video sigmas for {latent_t} latent frames")
        if sigma_video.numel() > 1 and not bool((sigma_video == sigma_video[0]).all()):
            raise ValueError("MiniMax H3 requires one uniform video sigma")
        audio_t = audio_x.shape[-1]
        text_tags = payload["text_token_tags"].view(-1).cpu()
        layout = self._layout(text_tags, latent_t, latent_h, latent_w, audio_t, payload)

        if spectrum is not None and spectrum.forecasting:
            timestep, timestep_indices = build_row_timesteps(
                layout,
                float(1.0 - sigma_video.flatten()[0]),
                float(1.0 - sigma_audio.flatten()[0]),
                max(float(1.0 - sigma_video.flatten()[0]), VISUAL_COND_TIMESTEP),
                AUDIO_COND_TIMESTEP,
            )
            timestep, timestep_indices = timestep.to(device), timestep_indices.to(device)
            temb = self._time_embedding(timestep)
            target_video_rows = latent_t * (latent_h // self.patch_size[1]) * (latent_w // self.patch_size[2])
            target_audio_rows = audio_t * 2
            video_start = layout.sequence_length - target_video_rows
            audio_start = video_start - target_audio_rows
            video_row = int(timestep_indices[video_start])
            audio_row = int(timestep_indices[audio_start + min(layout.num_target_condition_audio_latents,
                                                               max(audio_t - 1, 0))])
            hidden = spectrum.predict(device, dtype, self._check_interrupt)
            video, audio = self.final_layer([hidden], temb, (target_audio_rows, target_audio_rows + target_video_rows, video_row),
                                            (0, target_audio_rows, audio_row))
            del temb, timestep_indices
            video = _to_dtype([video], video_dtype)
            audio = _to_dtype([audio], audio_dtype)
            return (unpatchify_video_tokens(video, latent_t, latent_h, latent_w, self.latents_dim, self.patch_size),
                    unpack_audio(audio))

        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video, cond_audio = payload.get("cond_video_rows"), payload.get("cond_audio_rows")
        if cond_video is not None:
            video_rows = torch.cat((cond_video.to(device), video_rows))
        if cond_audio is not None:
            audio_rows = torch.cat((cond_audio.to(device), audio_rows))
        video_embeds = self.video_patch_proj(video_rows).to(dtype)
        del video_rows
        audio_embeds = self.audio_patch_proj(audio_rows).to(dtype)
        del audio_rows

        text_embeds = context[0]
        if text_embeds.shape[-1] != self.hidden_size:
            text_embeds = self.preprocess_text_embeds(context)[0]
        hidden = torch.empty(layout.sequence_length, self.hidden_size, dtype=dtype, device=device)
        hidden.index_copy_(0, layout.text_indices.to(device), text_embeds)
        hidden.index_copy_(0, layout.video_indices.to(device), video_embeds)
        hidden.index_copy_(0, layout.audio_indices.to(device), audio_embeds)
        del text_embeds, video_embeds, audio_embeds

        timestep, timestep_indices = build_row_timesteps(
            layout,
            float(1.0 - sigma_video.flatten()[0]),
            float(1.0 - sigma_audio.flatten()[0]),
            max(float(1.0 - sigma_video.flatten()[0]), VISUAL_COND_TIMESTEP),
            AUDIO_COND_TIMESTEP,
        )
        timestep, timestep_indices = timestep.to(device), timestep_indices.to(device)
        target_video_rows = latent_t * (latent_h // self.patch_size[1]) * (latent_w // self.patch_size[2])
        video_start = layout.sequence_length - target_video_rows
        video_head_row = int(timestep_indices[video_start])
        frame_rows = None
        if sigma_video.numel() == latent_t:
            frame_timesteps = 1.0 - sigma_video.to(device=device, dtype=torch.float32)
            timestep, remap = torch.unique(torch.cat((timestep, frame_timesteps)), sorted=True, return_inverse=True)
            timestep_indices = remap[:timestep_indices.max().item() + 1][timestep_indices]
            frame_rows = remap[-latent_t:]
            video_head_row = [(index * (target_video_rows // latent_t), (index + 1) * (target_video_rows // latent_t), int(row))
                              for index, row in enumerate(frame_rows)]
        adaln_indices = timestep_indices * 3 + layout.token_tags.to(device).clamp_min(0)
        changes = torch.cat((torch.ones(1, dtype=torch.bool, device=device), adaln_indices[1:] != adaln_indices[:-1],
                             torch.ones(1, dtype=torch.bool, device=device))).nonzero().flatten()
        segments = [(int(changes[index]), int(changes[index + 1]), int(adaln_indices[changes[index]]))
                    for index in range(changes.numel() - 1)]
        if frame_rows is not None:
            segments = [segment for segment in segments if segment[0] < video_start]
            rows_per_frame = target_video_rows // latent_t
            segments.extend((video_start + index * rows_per_frame, video_start + (index + 1) * rows_per_frame,
                             int(row) * 3 + MINIMAX_H3_VIDEO_TAG) for index, row in enumerate(frame_rows))
        temb = self._time_embedding(timestep)
        rope = payload.get("rope")
        if rope is None:
            positions = layout.position_ids.to(torch.float32)
            frequencies = positions.unsqueeze(-1) * self.rope.inv_freq.detach().cpu().view(1, 1, -1)
            rope = _rope_table(torch.cat(frequencies.unbind(dim=1), dim=-1), dtype).to(device)
            payload["rope"] = rope
            del positions, frequencies
        del adaln_indices, changes
        target_audio_rows = audio_t * 2
        audio_start = video_start - target_audio_rows
        self.sol_attention.begin_forward(layout, device, dtype, payload["attention_sparsity"])

        if first_block_cache is None:
            for block in self.blocks:
                self._check_interrupt()
                h_list = [hidden]
                hidden = None
                hidden = block(h_list, temb, segments, rope)
        else:
            self._check_interrupt()
            hidden, signature = self.blocks[0]([hidden], temb, segments, rope,
                                                residual_signature_elements=first_block_cache.MAX_SIGNATURE_ELEMENTS)
            if first_block_cache.should_compute(signature):
                head_output = first_block_cache.capture_head_output(hidden[audio_start:])
                for block_index in range(1, len(self.blocks)):
                    self._check_interrupt()
                    hidden = self.blocks[block_index]([hidden], temb, segments, rope)
                first_block_cache.store_tail_residual(hidden[audio_start:], head_output)
            else:
                first_block_cache.apply_tail_residual(hidden[audio_start:])

        audio_row = int(timestep_indices[audio_start + min(layout.num_target_condition_audio_latents,
                                                           max(audio_t - 1, 0))])
        if spectrum is not None:
            spectrum.observe(hidden[audio_start:], target_audio_rows, self._check_interrupt)
        h_list = [hidden]
        hidden = None
        video, audio = self.final_layer(h_list, temb, (video_start, layout.sequence_length, video_head_row),
                                        (audio_start, video_start, audio_row))
        del temb, rope, timestep_indices
        video = _to_dtype([video], video_dtype)
        audio = _to_dtype([audio], audio_dtype)
        return (unpatchify_video_tokens(video, latent_t, latent_h, latent_w, self.latents_dim, self.patch_size),
                unpack_audio(audio))


__all__ = ["AUDIO_COND_TIMESTEP", "Attention", "DiTBlock", "MLP", "MiniMaxH3Model", "TimeEmbedder",
           "VISUAL_COND_TIMESTEP", "get_linear_split_map", "pack_audio", "patchify_video", "unpack_audio",
           "unpatchify_video"]
