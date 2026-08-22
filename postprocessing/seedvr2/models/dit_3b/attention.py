# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from shared.attention import pay_attention


def _pad_packed(x_list: list[torch.Tensor], lengths: list[int], max_len: int) -> torch.Tensor:
    x = x_list.pop()
    batch = x.new_zeros(len(lengths), max_len, *x.shape[1:])
    offset = 0
    for row, length in enumerate(lengths):
        batch[row, :length].copy_(x[offset:offset + length])
        offset += length
    return batch


class FlashAttentionVarlen(nn.Module):
    """Adapt SeedVR2's packed attention to WanGP's shared attention backends."""

    def __init__(self, attention_mode: str = "sdpa", window_batch_size: int = 0, **kwargs):
        super().__init__()
        self.window_batch_size = int(window_batch_size)

    def tflops(self, args, kwargs, output) -> float:
        q_lens = kwargs["cu_seqlens_q"].diff() / 1e6
        k_lens = kwargs["cu_seqlens_k"].diff() / 1e6
        return output.shape[-2] * (4 * output.shape[-1] * (q_lens * k_lens).sum())

    def forward(self, qkv_list, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q_lens=None, k_lens=None, **kwargs):
        q, k, v = qkv_list
        qkv_list.clear()
        q_lens = cu_seqlens_q.diff().tolist() if q_lens is None else q_lens
        k_lens = cu_seqlens_k.diff().tolist() if k_lens is None else k_lens
        if len(q_lens) == 1:
            qkv_list = [q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)]
            q = k = v = None
            return pay_attention(qkv_list, recycle_q=True)[0, :q_lens[0]]

        max_q, max_k = max(q_lens), max(k_lens)
        q_list = [q]
        q = None
        q_batch = _pad_packed(q_list, q_lens, max_q)
        k_list = [k]
        k = None
        k_batch = _pad_packed(k_list, k_lens, max_k)
        v_list = [v]
        v = None
        v_batch = _pad_packed(v_list, k_lens, max_k)
        qkv_list = [q_batch, k_batch, v_batch]
        q_batch = k_batch = v_batch = None
        output_batch = pay_attention(qkv_list, k_lens=None if min(k_lens) == max_k else k_lens, recycle_q=True)
        output = output_batch.new_empty(sum(q_lens), *output_batch.shape[2:])
        output_offset = 0
        for row, length in enumerate(q_lens):
            output[output_offset:output_offset + length].copy_(output_batch[row, :length])
            output_offset += length
        return output
