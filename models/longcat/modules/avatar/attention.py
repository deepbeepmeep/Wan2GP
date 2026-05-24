from typing import List, Optional

import torch
import torch.nn as nn

from einops import rearrange

from shared.attention import pay_attention
from .rope_3d import RotaryPositionalEmbedding
from ..blocks import RMSNorm_FP32, _take_tensor
from ...audio_process.torch_utils import get_attn_map_with_target
from .rope_3d import RotaryPositionalEmbedding1D


def _run_attention(x_list, out_dtype, **attn_kwargs):
    q, k, v = x_list
    if out_dtype in (torch.float16, torch.bfloat16):
        attn_dtype = out_dtype
    else:
        attn_dtype = torch.bfloat16
    if q.dtype != attn_dtype:
        q = q.to(attn_dtype)
        k = k.to(attn_dtype)
        v = v.to(attn_dtype)
    x_list[:] = [q, k, v]
    del q, k, v
    attn_kwargs.setdefault("recycle_q", True)
    x = pay_attention(x_list, **attn_kwargs)
    if x.dtype != out_dtype:
        x = x.to(out_dtype)
    return x


def _run_sparse_attention(x_list, out_dtype, shape, bsa_params, **attn_kwargs):
    raise NotImplementedError("LongCat sparse/BSA attention is not wired to WanGP shared attention.")


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range 
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(
            self.head_dim,
            cp_split_hw=cp_split_hw
        )

    def _process_attn(self, q, k, v, shape, out_dtype):
        """
            function wrapper to do attention with q, k, v
        """
        if self.enable_bsa:
            return _run_sparse_attention([q, k, v], out_dtype, shape, self.bsa_params)
        return _run_attention([q, k, v], out_dtype)

    def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False, num_ref_latents=None, ref_img_index=None, mask_frame_range=None, ref_target_masks=None) -> torch.Tensor:
        """
        """
        x = _take_tensor(x)
        B, N, C = x.shape
        out_dtype = x.dtype
        qkv = self.qkv(x)
        x = None
        if qkv.dtype != out_dtype:
            qkv = qkv.to(out_dtype)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        v = v.contiguous()
        del qkv

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape, ref_img_index, num_ref_latents)

        N_t, N_h, N_w = shape
        # cond mode
        if num_cond_latents is not None and num_cond_latents == 1:
            # image to video
            num_cond_latents_thw = num_cond_latents * (N // N_t)
            # process the condition tokens
            q_cond = q[:, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :num_cond_latents_thw].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape, out_dtype)
            # process the noise tokens
            q_noise = q[:, num_cond_latents_thw:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape, out_dtype)
            # merge x_cond and x_noise
            x = x_cond.new_empty(B, N, self.num_heads, self.head_dim)
            x[:, :num_cond_latents_thw].copy_(x_cond)
            x[:, num_cond_latents_thw:].copy_(x_noise)
            del x_cond, x_noise
        elif num_cond_latents is not None and num_cond_latents > 1:
            # video continuation
            assert num_ref_latents is not None and ref_img_index is not None, f"No specified insertion position for reference frame"
            num_ref_latents_thw = (N // N_t)
            num_cond_latents_thw = num_cond_latents * (N // N_t)
            # process the condition tokens
            q_ref = q[:, :num_ref_latents_thw].contiguous()
            k_ref = k[:, :num_ref_latents_thw].contiguous()
            v_ref = v[:, :num_ref_latents_thw].contiguous()
            q_cond = q[:, num_ref_latents_thw:num_cond_latents_thw].contiguous()
            k_cond = k[:, num_ref_latents_thw:num_cond_latents_thw].contiguous()
            v_cond = v[:, num_ref_latents_thw:num_cond_latents_thw].contiguous()
            x_ref = self._process_attn(q_ref, k_ref, v_ref, shape, out_dtype)
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape, out_dtype)
            if num_cond_latents == N_t:
                x = x_ref.new_empty(B, num_cond_latents_thw, self.num_heads, self.head_dim)
                x[:, :num_ref_latents_thw].copy_(x_ref)
                x[:, num_ref_latents_thw:num_cond_latents_thw].copy_(x_cond)
                del x_ref, x_cond
            else:
                # process the noise tokens
                q_noise = q[:, num_cond_latents_thw:].contiguous()
                
                start_noise, end_noise, num_noisy_frames = 0, 0, N_t - num_cond_latents
                if mask_frame_range is not None and mask_frame_range > 0:
                    start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
                    end_noise   = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1

                if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
                    # remove attention with the reference image in the target range, preventing repeated actions.
                    start_pos = start_noise * (N // N_t)
                    end_pos   = end_noise * (N // N_t)
                    q_noise_front = q_noise[:, :start_pos].contiguous()
                    q_noise_maskref = q_noise[:, start_pos:end_pos].contiguous()
                    q_noise_back = q_noise[:, end_pos:].contiguous()
                    k_non_ref = k[:, num_ref_latents_thw:].contiguous()
                    v_non_ref = v[:, num_ref_latents_thw:].contiguous()
                    x_noise_front = self._process_attn(q_noise_front, k, v, shape, out_dtype) # q_front has attention with ref + cond + noisy
                    x_noise_back = self._process_attn(q_noise_back, k, v, shape, out_dtype) # q_back has attention with ref + cond + noisy
                    x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref, shape, out_dtype) # q_mask has attention with cond+noisy
                    x_noise = x_noise_front.new_empty(B, q_noise.shape[1], self.num_heads, self.head_dim)
                    x_noise[:, :start_pos].copy_(x_noise_front)
                    x_noise[:, start_pos:end_pos].copy_(x_noise_maskref)
                    x_noise[:, end_pos:].copy_(x_noise_back)
                    del x_noise_front, x_noise_maskref, x_noise_back
                else:
                    x_noise = self._process_attn(q_noise, k, v, shape, out_dtype)
                # merge x_cond and x_noise
                x = x_ref.new_empty(B, N, self.num_heads, self.head_dim)
                x[:, :num_ref_latents_thw].copy_(x_ref)
                x[:, num_ref_latents_thw:num_cond_latents_thw].copy_(x_cond)
                x[:, num_cond_latents_thw:].copy_(x_noise)
                del x_ref, x_cond, x_noise

        else:
            # text to video
            x = self._process_attn(q, k, v, shape, out_dtype)

        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)

        # calculate attention mask for the given area in reference image
        x_ref_attn_map = None
        if ref_target_masks is not None:
            assert num_cond_latents is not None and num_cond_latents > 0, f"currently, multitalk only supports image to video or video continuation"
            x_ref_attn_map = get_attn_map_with_target(
                q[:, num_cond_latents_thw:].type_as(x),
                k.type_as(x),
                shape,
                ref_target_masks=ref_target_masks,
                cp_split_hw=self.cp_split_hw,
            )
        q = k = v = None

        if return_kv:
            return x, (k_cache, v_cache), x_ref_attn_map
        else:
            return x, x_ref_attn_map

    def forward_with_kv_cache(self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None, num_ref_latents=None, ref_img_index=None, mask_frame_range=None, ref_target_masks=None) -> torch.Tensor:
        """
        """
        x = _take_tensor(x)
        B, N, C = x.shape
        out_dtype = x.dtype
        qkv = self.qkv(x)
        x = None
        if qkv.dtype != out_dtype:
            qkv = qkv.to(out_dtype)
        
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        v = v.contiguous()
        del qkv

        N_t, N_h, N_w = shape
        k_cache, v_cache = kv_cache
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)
        
        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = torch.cat([k_cache, k], dim=1).contiguous()
            v_full = torch.cat([v_cache, v], dim=1).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=1).contiguous()
            q_padding, k_full = self.rope_3d(q_padding, k_full, (N_t + num_cond_latents, N_h, N_w), ref_img_index, num_ref_latents)
            q = q_padding[:, -N:].contiguous()
            del q_padding
        else:
            k_full = k
            v_full = v
        
        start_noise, end_noise, num_noisy_frames = 0, 0, N_t
        if mask_frame_range is not None and mask_frame_range > 0:
            start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents 
            end_noise   = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1 
        
        if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
            # remove attention with the reference image in the target range, preventing repeated actions.
            num_ref_latents_thw = (N // N_t)
            start_pos = start_noise * (N // N_t)
            end_pos   = end_noise * (N // N_t)
            q_noise_front = q[:, :start_pos].contiguous()
            q_noise_maskref = q[:, start_pos:end_pos].contiguous()
            q_noise_back = q[:, end_pos:].contiguous()
            k_non_ref = k_full[:, num_ref_latents_thw:].contiguous()
            v_non_ref = v_full[:, num_ref_latents_thw:].contiguous()
            x_noise_front = self._process_attn(q_noise_front, k_full, v_full, shape, out_dtype) # q_front --> ref+cond+noisy
            x_noise_back = self._process_attn(q_noise_back, k_full, v_full, shape, out_dtype) # q_back --> ref+cond+noisy
            x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref, shape, out_dtype) # q_mask --> cond+noisy
            x = x_noise_front.new_empty(B, N, self.num_heads, self.head_dim)
            x[:, :start_pos].copy_(x_noise_front)
            x[:, start_pos:end_pos].copy_(x_noise_maskref)
            x[:, end_pos:].copy_(x_noise_back)
            del x_noise_front, x_noise_maskref, x_noise_back
        else:
            x = self._process_attn(q, k_full, v_full, shape, out_dtype)
        
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape) 
        x = self.proj(x)

        # calculate attention mask for the given area in reference image
        x_ref_attn_map = None
        if ref_target_masks is not None:
            assert num_cond_latents is not None and num_cond_latents > 0, f"currently, multitalk only supports image to video or video continuation"
            x_ref_attn_map = get_attn_map_with_target(
                q.type_as(x),
                k_full.type_as(x),
                shape,
                ref_target_masks=ref_target_masks,
                cp_split_hw=self.cp_split_hw,
            )
        q = k = v = k_full = v_full = None

        return x, x_ref_attn_map


class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
        cp_split_hw: Optional[List[int]] = None,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cp_split_hw = cp_split_hw
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        # multitalk related params
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1  = (0, self.class_interval)
        self.rope_h2  = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def _process_cross_attn(self, x, cond, frames_num=None, x_ref_attn_map=None, human_num=None, speaker_token_masks=None):
        x = _take_tensor(x)
        cond = _take_tensor(cond)

        N_t = frames_num
        out_dtype = x.dtype
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
        x = None
        if q.dtype != out_dtype:
            q = q.to(out_dtype)
        q = self.q_norm(q)

        # multitalk with rope1d pe
        if x_ref_attn_map is not None:
            max_values = x_ref_attn_map.max(1).values[:, None, None] 
            min_values = x_ref_attn_map.min(1).values[:, None, None] 
            max_min_values = torch.cat([max_values, min_values], dim=2) 
            human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
            human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

            human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
            human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
            background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
            back   = torch.full((x_ref_attn_map.size(1),), background_pos, dtype=human1.dtype).to(human1.device)
            max_indices = x_ref_attn_map.argmax(dim=0).clamp(max=2)
            normalized_map = torch.stack([human1, human2, back], dim=1)
            normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices] 

            q = rearrange(q, "(B N_t) S H C -> B (N_t S) H C", N_t=N_t)
            q = self.rope_1d(q, normalized_pos)
            q = rearrange(q, "B (N_t S) H C -> (B N_t) S H C", N_t=N_t)
        
        # get kv from encoder_hidden_states
        _, N_a, _ = cond.shape
        encoder_kv = self.kv_linear(cond).view(B, N_a, 2, self.num_heads, self.head_dim)
        cond = None
        if encoder_kv.dtype != out_dtype:
            encoder_kv = encoder_kv.to(out_dtype)
        encoder_k, encoder_v = encoder_kv.unbind(2)
        encoder_v = encoder_v.contiguous()
        del encoder_kv
        encoder_k = self.k_norm(encoder_k)


        # multitalk with rope1d pe
        if x_ref_attn_map is not None:
            per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
            human1_pos = (self.rope_h1[0] + self.rope_h1[1]) / 2
            human2_pos = (self.rope_h2[0] + self.rope_h2[1]) / 2
            if human_num is not None and human_num > 2:
                background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
                tokens_per_human = per_frame.size(0) // human_num
                per_frame[:tokens_per_human] = human1_pos
                per_frame[tokens_per_human:2*tokens_per_human] = human2_pos
                per_frame[2*tokens_per_human:] = background_pos
            else:
                per_frame[:per_frame.size(0)//2] = human1_pos
                per_frame[per_frame.size(0)//2:] = human2_pos
            encoder_pos = torch.concat([per_frame] * N_t, dim=0)
            encoder_k = rearrange(encoder_k, "(B N_t) S H C -> B (N_t S) H C", N_t=N_t)
            encoder_k = self.rope_1d(encoder_k, encoder_pos)
            encoder_k = rearrange(encoder_k, "B (N_t S) H C -> (B N_t) S H C", N_t=N_t)
        attention_mask = None
        if speaker_token_masks is not None and human_num == 2:
            token_owner = speaker_token_masks.argmax(dim=0).clamp(max=2).unsqueeze(0).expand(N_t, -1)
            if B != N_t:
                token_owner = token_owner.repeat(B // N_t, 1)
            split = N_a // human_num
            attention_mask = torch.empty((B, N, 1, N_a), device=q.device, dtype=torch.bool)
            attention_mask[..., :split] = token_owner.ne(1).unsqueeze(-1).unsqueeze(-1)
            attention_mask[..., split:] = token_owner.ne(0).unsqueeze(-1).unsqueeze(-1)

        qkv_list = [q, encoder_k, encoder_v]
        del q, encoder_k, encoder_v
        x = _run_attention(qkv_list, out_dtype, attention_mask=attention_mask)

        # linear transform
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        # reshape x to origin shape
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x.type(out_dtype)

    def forward(self, x, cond, shape=None, num_cond_latents=None, x_ref_attn_map=None, human_num=None, speaker_token_masks=None):

        x = _take_tensor(x)
        cond = _take_tensor(cond)
        B, N, C = x.shape
        if (num_cond_latents is None or num_cond_latents == 0): 
            # text to video
            x_list = [x]
            cond_list = [cond]
            x = cond = None
            output = self._process_cross_attn(x_list, cond_list, shape[0], x_ref_attn_map, human_num=human_num, speaker_token_masks=speaker_token_masks)
            return None, output
        elif num_cond_latents is not None and num_cond_latents > 0:
            # image to video or video continuation
            assert shape is not None, "SHOULD pass in the shape"
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            x_noise = x[:, num_cond_latents_thw:]
            x = None
            cond = rearrange(cond, "(B N_t) M C -> B N_t M C", B=B)
            cond = cond[:, num_cond_latents:] 
            cond = rearrange(cond, "B N_t M C -> (B N_t) M C")
            frames_num = shape[0] - num_cond_latents
            if human_num is not None and human_num == 2:
                # multitalk mode
                x_noise_list = [x_noise]
                cond_list = [cond]
                x_noise = cond = None
                output_noise = self._process_cross_attn(x_noise_list, cond_list, frames_num, x_ref_attn_map, human_num=human_num, speaker_token_masks=speaker_token_masks)
            elif human_num is not None and human_num > 2:
                # multitalk mode with background silent audio
                x_noise_list = [x_noise]
                cond_list = [cond]
                x_noise = cond = None
                output_noise = self._process_cross_attn(x_noise_list, cond_list, frames_num, x_ref_attn_map, human_num=human_num, speaker_token_masks=speaker_token_masks)
            else:
                # singletalk mode
                x_noise_list = [x_noise]
                cond_list = [cond]
                x_noise = cond = None
                output_noise = self._process_cross_attn(x_noise_list, cond_list, frames_num)
            return num_cond_latents_thw, output_noise
        else:
            raise NotImplementedError
