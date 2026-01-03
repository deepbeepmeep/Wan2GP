from typing import List, Optional

import torch
import torch.nn as nn

from shared.attention import pay_attention, pay_sparse_attention
from .rope_3d import RotaryPositionalEmbedding
from .blocks import RMSNorm_FP32


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
    x = pay_attention(x_list, **attn_kwargs)
    x_list[:] = []
    if x.dtype != out_dtype:
        x = x.to(out_dtype)
    return x


def _run_sparse_attention(x_list, out_dtype, shape, bsa_params, **attn_kwargs):
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
    x = pay_sparse_attention(x_list, shape=shape, bsa_params=bsa_params, **attn_kwargs)
    x_list[:] = []
    if x.dtype != out_dtype:
        x = x.to(out_dtype)
    return x


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

    def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        out_dtype = x.dtype
        qkv = self.qkv(x)
        if qkv.dtype != out_dtype:
            qkv = qkv.to(out_dtype)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape)

        # cond mode
        if num_cond_latents is not None and num_cond_latents > 0:
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            # process the condition tokens
            q_cond = q[:, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :num_cond_latents_thw].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape, out_dtype)
            # process the noise tokens
            q_noise = q[:, num_cond_latents_thw:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape, out_dtype)
            # merge x_cond and x_noise
            x = torch.cat([x_cond, x_noise], dim=1).contiguous()
        else:
            x = self._process_attn(q, k, v, shape, out_dtype)

        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)

        if return_kv:
            return x, (k_cache, v_cache)
        else:
            return x

    def forward_with_kv_cache(self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        out_dtype = x.dtype
        qkv = self.qkv(x)
        if qkv.dtype != out_dtype:
            qkv = qkv.to(out_dtype)
        
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        T, H, W = shape
        k_cache, v_cache = kv_cache
        if k_cache.shape[0] == 1 and B > 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)
        
        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = torch.cat([k_cache, k], dim=1).contiguous()
            v_full = torch.cat([v_cache, v], dim=1).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=1).contiguous()
            q_padding, k_full = self.rope_3d(q_padding, k_full, (T + num_cond_latents, H, W))
            q = q_padding[:, -N:].contiguous()
        else:
            k_full = k
            v_full = v
            
        x = self._process_attn(q, k_full, v_full, shape, out_dtype)
        
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)

        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
        ):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

    def _process_cross_attn(self, x, cond, kv_seqlen):
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim
        out_dtype = x.dtype

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
        if q.dtype != out_dtype:
            q = q.to(out_dtype)
        kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
        if kv.dtype != out_dtype:
            kv = kv.to(out_dtype)
        k, v = kv.unbind(2)

        q, k = self.q_norm(q), self.k_norm(k)

        k_lens = kv_seqlen
        if k_lens is not None:
            if isinstance(k_lens, torch.Tensor):
                k_lens = k_lens.tolist() if B > 1 else k_lens.to(q.device)
            elif isinstance(k_lens, list) and B == 1:
                k_lens = torch.tensor(k_lens, device=q.device)

        x = _run_attention([q, k, v], out_dtype, k_lens=k_lens, cross_attn=True)
        x = x.view(B, N, C)
        x = self.proj(x)
        return x

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        """
            x: [B, N, C]
            cond: [B, M, C]
        """
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)
        else:
            B, N, C = x.shape
            if num_cond_latents is not None and num_cond_latents > 0:
                assert shape is not None, "SHOULD pass in the shape"
                num_cond_latents_thw = num_cond_latents * (N // shape[0])
                x_noise = x[:, num_cond_latents_thw:] # [B, N_noise, C]
                output_noise = self._process_cross_attn(x_noise, cond, kv_seqlen) # [B, N_noise, C]
                output = torch.cat([
                    torch.zeros((B, num_cond_latents_thw, C), dtype=output_noise.dtype, device=output_noise.device),
                    output_noise
                ], dim=1).contiguous()
            else:
                raise NotImplementedError
                
            return output
