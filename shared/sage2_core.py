"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F

from sageattention.triton.quant_per_block import per_block_int8 as per_block_int8_triton
from sageattention.triton.quant_per_block_varlen import per_block_int8 as per_block_int8_varlen_triton
from sageattention.triton.attn_qk_int8_per_block import forward as attn_false
from sageattention.triton.attn_qk_int8_per_block_causal import forward as attn_true
from sageattention.triton.attn_qk_int8_block_varlen import forward as attn_false_varlen
from sageattention.triton.attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen

from sageattention.triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

try:
    from sageattention import _qattn_sm80
    if not hasattr(_qattn_sm80, "qk_int8_sv_f16_accum_f32_attn"): 
        _qattn_sm80 = torch.ops.sageattention_qattn_sm80
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    from sageattention import _qattn_sm89
    if not hasattr(_qattn_sm89, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"): 
        _qattn_sm89 = torch.ops.sageattention_qattn_sm89
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

try:
    from sageattention import _qattn_sm90
    if not hasattr(_qattn_sm90, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"): 
        _qattn_sm90 = torch.ops.sageattention_qattn_sm90
    SM90_ENABLED = True
except:
    SM90_ENABLED = False

from sageattention.quant import per_block_int8 as per_block_int8_cuda
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
from sageattention.quant import sub_mean
from sageattention.quant import per_channel_fp8

from typing import Any, List, Literal, Optional, Tuple, Union
import warnings
import os

def is_sage2_supported():
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            return False
    return True

from importlib.metadata import version
sg2_version = version("sageattention")
sg2pp = sg2_version.startswith("2.2")

import subprocess
import re
def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major, minor
    except Exception as e:
        print("Failed to get CUDA version:", e)
    return None, None

def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs

def sageattn(
    qkv_list,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute capability.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """
        
    arch = get_cuda_arch_versions()[qkv_list[0].device.index]
    if arch == "sm80":
        return sageattn_qk_int8_pv_fp16_cuda(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    elif arch == "sm86":
        return sageattn_qk_int8_pv_fp16_triton(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse)
    elif arch == "sm89":
        return sageattn_qk_int8_pv_fp8_cuda(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp16" if sg2pp else "fp32+fp32")
    elif arch == "sm90":
        return sageattn_qk_int8_pv_fp8_cuda_sm90(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    elif arch == "sm120":
        return sageattn_qk_int8_pv_fp8_cuda(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype= "fp32+fp16" if sg2pp else "fp32", smooth_v= not sg2pp) # sm120 has accurate fp32 accumulator for fp8 mma and triton kernel is currently not usable on sm120.
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

@torch.compiler.disable
def sageattn_qk_int8_pv_fp16_triton(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor, 
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool =False, 
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with per-block INT8 quantization for Q and K, FP16 PV with FP16 accumulation, implemented using Triton.
    The FP16 accumulator is added to a FP32 buffer immediately after each iteration.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    quantization_backend : str
        The quantization backend, either "triton" or "cuda".
        "cuda" backend offers better performance due to kernel fusion.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q, k, v = qkv_list
    qkv_list.clear()
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if quantization_backend == "triton":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    elif quantization_backend == "cuda":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")
    del q,k, km

    if is_causal:
        o, lse = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse)
    else:
        o, lse = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_varlen(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    cu_seqlens_q: torch.Tensor, 
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int, 
    max_seqlen_k: int, 
    is_causal: bool = False,
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    cu_seqlens_q : torch.Tensor
        The cumulative sequence lengths for the query sequences in the batch, used to index into `q`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    cu_seqlens_k : torch.Tensor
        The cumulative sequence lengths for the key and value sequences in the batch, used to index into `k` and `v`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen_q : int
        The maximum sequence length for the query tensor in the batch.
    
    max_seqlen_k : int
        The maximum sequence length for the key and value tensors in the batch.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len for each sequence.
        Default: False.
    
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - The tensors `cu_seqlens_q` and `cu_seqlens_k` must have the dtype ``torch.int32`` or ``torch.int64``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if smooth_k:
        km = k.mean(dim=0, keepdim=True) # ! km is calculated on the all the batches. Calculate over each individual sequence requires dedicated kernel.
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = per_block_int8_varlen_triton(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=sm_scale)

    if is_causal:
        o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)
    else:
        o = attn_false_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)

    o = o[..., :head_dim_og]

    return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp16_cuda(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP16 PV with FP16/FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp16", "fp16+fp32" or "fp32".
        - "fp16": PV accumulation is done in fully in FP16. This is the fastest option but may lead to numerical instability. `smooth_v` option will increase the accuracy in cases when the value tensor has a large bias (like in CogVideoX-2b).
        - "fp32": PV accumulation is done in FP32. This is the most accurate option but may be slower than "fp16" due to CUDA core overhead.
        - "fp16+fp32": PV accumulation is done in FP16, but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32" or "fp16+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q,k,v = qkv_list
    qkv_list.clear() 
    dtype = q.dtype
    assert SM80_ENABLED, "SM80 kernel is not available. make sure you GPUs with compute capability 8.0 or higher."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64, WARPK=64)

    q_size = q.size()
    q_device = q.device
    del q,k, km
    o = torch.empty(q_size, dtype=dtype, device=q_device)

    if pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == 'fp32':
        v = v.to(torch.float16)
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            del v
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            v = v.to(torch.float16)
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16+fp32":
        v = v.to(torch.float16)
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_cuda(
    qkv_list,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = None,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    if pv_accum_dtype == None:
        pv_accum_dtype = "fp32+fp16" if sg2pp else "fp32+fp32"
        
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q, k, v = qkv_list
    qkv_list.clear()

    dtype = q.dtype
    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # if sg2pp:
    #     cuda_major_version, cuda_minor_version = get_cuda_version()
    #     if(cuda_major_version, cuda_minor_version) < (12, 8) and pv_accum_dtype == 'fp32+fp16':
    #         warnings.warn("cuda version < 12.8, change pv_accum_dtype to 'fp32+fp32'")
    #         pv_accum_dtype = 'fp32+fp32'

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
    q_size = q.size()
    q_device = q.device
    del q,k,km

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False
    if sg2pp:
        if pv_accum_dtype == 'fp32+fp16' and smooth_v:
            warnings.warn("pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored.")
            smooth_v = False

        quant_v_scale_max = 448.0
        if pv_accum_dtype == 'fp32+fp16':
            quant_v_scale_max = 2.25

        v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=smooth_v)
    else:
        v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=smooth_v)
    del v
    o = torch.empty(q_size, dtype=dtype, device=q_device)
    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp16":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)


    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_window_cuda(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    window = -1,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q,k,v = qkv_list
    qkv_list.clear()
    dtype = q.dtype
    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)

    q_size = q.size()
    q_device = q.device
    del q,k

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=smooth_v)
    del v
    o = torch.empty(q_size, dtype=dtype, device=q_device)

    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse, window)
        else:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse,  window)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse,  window)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_cuda_sm90(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q,k,v = qkv_list
    qkv_list.clear()
    dtype = q.dtype
    assert SM90_ENABLED, "SM90 kernel is not available. Make sure you GPUs with compute capability 9.0."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)

    q_size = q.size()
    kv_len = k.size(seq_dim)
    q_device = q.device
    del q,k


    # pad v to multiple of 128
    # TODO: modify per_channel_fp8 kernel to handle this
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)], dim=2)
        else:
            v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)
    del v
    o = torch.empty(q_size, dtype=dtype, device=q_device)

    if pv_accum_dtype == "fp32":
        raise NotImplementedError("Please use pv_accum_dtype='fp32+fp32' for sm90.")
        lse = _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o