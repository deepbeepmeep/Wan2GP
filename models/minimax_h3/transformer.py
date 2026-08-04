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

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import pay_attention

from .interrupt import GenerationInterrupted
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
            chunk_size = min(chunk_size, max(1, x.shape[0] * self.hidden // (2 * self.ffn)))
        if chunk_size <= 0 or x.shape[0] <= chunk_size:
            return self._project([x])
        for start in range(0, x.shape[0], chunk_size):
            output = self._project([x[start:start + chunk_size]])
            x[start:start + output.shape[0]].copy_(output)
            del output
        return x


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, dtype=None, device=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * inner, bias=False, dtype=dtype, device=device)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x_list, rope=None, transformer_options=None):
        x = _take(x_list)
        seq_len = x.shape[0]
        if hasattr(self, "q_proj"):
            query = self.q_norm(self.q_proj(x).view(1, seq_len, self.heads, self.head_dim))
            key = self.k_norm(self.k_proj(x).view(1, seq_len, self.heads, self.head_dim))
            value = self.v_proj(x).view(1, seq_len, self.heads, self.head_dim)
        else:
            qkv = self.qkv_proj(x).view(seq_len, self.heads, 3, self.head_dim)
            query, key, value = qkv.unbind(dim=2)
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0).clone()
            del qkv
        del x
        if not hasattr(self, "q_proj"):
            query, key = self.q_norm(query), self.k_norm(key)
        if rope is not None:
            pairs = rope.shape[-2]
            cosine, sine = rope[..., 0], rope[..., 1]
            for tensor in (query, key):
                first, second = tensor[..., :pairs], tensor[..., pairs:2 * pairs]
                first_out = first * cosine - second * sine
                second_out = second * cosine + first * sine
                first.copy_(first_out)
                second.copy_(second_out)
                del first_out, second_out
        qkv_list = [query, key, value]
        del query, key, value
        output = pay_attention(qkv_list, recycle_q=True).reshape(seq_len, -1)
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
                 adaln_dtype=None, ffn_chunk_size=2048, dtype=None, device=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden, ffn, ffn_chunk_size, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(time_dim, hidden, 6, apply_silu=apply_silu,
                                    dtype=adaln_dtype or dtype, device=device)

    def forward(self, x_list, temb, segments, rope):
        residual_list = [_take(x_list)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        h_list = [_modulate(self.norm1(residual_list[0]), [shift_msa, scale_msa], segments)]
        residual_list = [_gated_residual(residual_list, [gate_msa], [self.attn(h_list, rope=rope)], segments)]
        h_list = [_modulate(self.norm2(residual_list[0]), [shift_mlp, scale_mlp], segments)]
        return _gated_residual(residual_list, [gate_mlp], [self.mlp(h_list)], segments)


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
        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                                               time_embed_dim, norm_eps, qk_norm_eps, **curve,
                                               ffn_chunk_size=ffn_chunk_size, dtype=dtype, device=device) for _ in range(num_layers)])
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

    def _layout(self, text_tags, latent_t, latent_h, latent_w, audio_t, payload):
        signature = (text_tags.numel(), latent_t, latent_h, latent_w, audio_t,
                     tuple(k["resolved_frame_index"] for k in payload.get("keyframes") or ()),
                     tuple((r["kind"], r.get("latent_t"), r.get("latent_h"), r.get("latent_w"), r.get("ref_audio_t"))
                           for r in payload.get("refs") or ()))
        if payload.get("layout_signature") == signature:
            return payload["layout"]
        with torch.device("cpu"):
            if payload.get("refs"):
                layout = build_ref2va_packed_sequence(text_tags, _prepared_references(payload["refs"]), latent_t,
                                                      latent_h, latent_w, audio_t, self.patch_size)
            else:
                anchors = tuple("first" if keyframe["resolved_frame_index"] == 0 else "last"
                                for keyframe in payload.get("keyframes") or ())
                layout = build_packed_sequence(text_tags, latent_t, latent_h, latent_w, audio_t, self.patch_size, anchors)
        payload["layout_signature"], payload["layout"] = signature, layout
        return layout

    def _time_embedding(self, timesteps):
        if not self.use_adaln_curves:
            return self.time_embedder(timesteps)
        table = self.adaln_t_table
        position = timesteps.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))

    def forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload):
        device, dtype = video_x.device, self.dtype or next(self.blocks.parameters()).dtype
        video_dtype, audio_dtype = video_x.dtype, audio_x.dtype
        _, _, latent_t, latent_h, latent_w = video_x.shape
        audio_t = audio_x.shape[-1]
        text_tags = payload["text_token_tags"].view(-1).cpu()
        layout = self._layout(text_tags, latent_t, latent_h, latent_w, audio_t, payload)

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
        adaln_indices = timestep_indices * 3 + layout.token_tags.to(device).clamp_min(0)
        changes = torch.cat((torch.ones(1, dtype=torch.bool, device=device), adaln_indices[1:] != adaln_indices[:-1],
                             torch.ones(1, dtype=torch.bool, device=device))).nonzero().flatten()
        segments = [(int(changes[index]), int(changes[index + 1]), int(adaln_indices[changes[index]]))
                    for index in range(changes.numel() - 1)]
        temb = self._time_embedding(timestep)
        rope = payload.get("rope")
        if rope is None:
            positions = layout.position_ids.to(torch.float32)
            frequencies = positions.unsqueeze(-1) * self.rope.inv_freq.detach().cpu().view(1, 1, -1)
            rope = _rope_table(torch.cat(frequencies.unbind(dim=1), dim=-1), dtype).to(device)
            payload["rope"] = rope
            del positions, frequencies
        del adaln_indices, changes

        cache = getattr(self, "cache", None)
        use_fbc = cache is not None and getattr(cache, "h3_first_block", False)
        if use_fbc:
            from .step_cache import should_skip_remaining

            step_no = int(payload.get("step_no", 0) or 0)
            # Blocks mutate hidden in-place via gated residuals; clone the pre-block state.
            original = hidden.detach().clone()
            if self._interrupt:
                raise GenerationInterrupted
            hidden = self.blocks[0]([hidden], temb, segments, rope)
            first_residual = hidden - original
            if should_skip_remaining(cache, step_no, first_residual):
                # Reuse previous full-stack residual; first block already ran.
                prev = cache.previous_residual
                if prev.device != original.device or prev.dtype != original.dtype:
                    prev = prev.to(device=original.device, dtype=original.dtype)
                hidden = original.add(prev)
                cache.skipped_steps = int(getattr(cache, "skipped_steps", 0) or 0) + 1
                del first_residual, original
            else:
                for block in self.blocks[1:]:
                    if self._interrupt:
                        raise GenerationInterrupted
                    h_list = [hidden]
                    hidden = None
                    hidden = block(h_list, temb, segments, rope)
                # Store residuals for a later skip (detached; no grad graph).
                cache.previous_residual = (hidden.detach() - original).contiguous()
                cache.previous_modulated_input = first_residual.detach().contiguous()
                cache.full_steps = int(getattr(cache, "full_steps", 0) or 0) + 1
                del first_residual, original
        else:
            for block in self.blocks:
                if self._interrupt:
                    raise GenerationInterrupted
                h_list = [hidden]
                hidden = None
                hidden = block(h_list, temb, segments, rope)

        target_video_rows = latent_t * (latent_h // self.patch_size[1]) * (latent_w // self.patch_size[2])
        target_audio_rows = audio_t * 2
        video_start = layout.sequence_length - target_video_rows
        audio_start = video_start - target_audio_rows
        video_row = int(timestep_indices[video_start])
        audio_row = int(timestep_indices[audio_start])
        h_list = [hidden]
        hidden = None
        video, audio = self.final_layer(h_list, temb, (video_start, layout.sequence_length, video_row),
                                        (audio_start, video_start, audio_row))
        del temb, rope, timestep_indices
        video = _to_dtype([video], video_dtype)
        audio = _to_dtype([audio], audio_dtype)
        return (unpatchify_video_tokens(video, latent_t, latent_h, latent_w, self.latents_dim, self.patch_size),
                unpack_audio(audio))


__all__ = ["AUDIO_COND_TIMESTEP", "Attention", "DiTBlock", "MLP", "MiniMaxH3Model", "TimeEmbedder",
           "VISUAL_COND_TIMESTEP", "get_linear_split_map", "pack_audio", "patchify_video", "unpack_audio",
           "unpatchify_video"]
