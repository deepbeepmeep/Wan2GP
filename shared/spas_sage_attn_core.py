"""
Copyright (c) 2025 by SpargeAttn team.

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

import importlib

import torch
from spas_sage_attn.utils import hyperparameter_check, get_block_map_meansim_fuse_quant, get_vanilla_qk_quant, block_map_lut_triton
from spas_sage_attn.triton_kernel_example import spas_sage_attn_meansim as spas_sage_attn_meansim_triton
from einops import rearrange


def _load_qattn_kernel(compile_module, direct_module, attr_name, op_namespace):
    try:
        return getattr(importlib.import_module(compile_module), attr_name)
    except ModuleNotFoundError as exc:
        if exc.name != compile_module:
            raise
        importlib.import_module(direct_module)
        return getattr(torch.ops, op_namespace)


try:
    _qattn_sm80 = _load_qattn_kernel("spas_sage_attn.sm80_compile", "spas_sage_attn._qattn_sm80", "_qattn_sm80", "spas_sage_attn_qattn_sm80")
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    _qattn_sm89 = _load_qattn_kernel("spas_sage_attn.sm89_compile", "spas_sage_attn._qattn_sm89", "_qattn_sm89", "spas_sage_attn_qattn_sm89")
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

try:
    _qattn_sm90 = _load_qattn_kernel("spas_sage_attn.sm90_compile", "spas_sage_attn._qattn_sm90", "_qattn_sm90", "spas_sage_attn_qattn_sm90")
    SM90_ENABLED = True
except:
    SM90_ENABLED = False

import spas_sage_attn._fused
_fused = torch.ops.spas_sage_attn_fused


def get_cuda_version():
    version = torch.version.cuda
    major, minor = version.split('.')
    return int(major), int(minor)


def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


# Currently get_cuda_arch_versions cannot be traced by torch.compile
_cuda_archs = get_cuda_arch_versions()


def _unpack_qkv(q, k=None, v=None):
    if k is None and v is None and isinstance(q, list):
        qkv_list = q
        q, k, v = qkv_list
        qkv_list.clear()
    return q, k, v


def _unpack_tensor(x):
    if isinstance(x, list):
        x_list = x
        x = x_list[0]
        x_list.clear()
    return x


def _contiguous_to_dtype(x, dtype):
    x = _unpack_tensor(x)
    if x.is_contiguous() and x.dtype == dtype:
        return x
    if x.dtype == dtype:
        return x.contiguous()
    out = torch.empty_like(x, dtype=dtype, memory_format=torch.contiguous_format)
    out.copy_(x)
    return out


def _quantize_v_fp8(v, arch):
    v = _unpack_tensor(v)
    b, h_kv, kv_len, head_dim = v.shape
    padded_len = (kv_len + 127) // 128 * 128
    v_transposed_permuted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
    _fused.transpose_pad_permute_cuda(v, v_transposed_permuted, 1)
    del v
    v_fp8 = torch.empty(v_transposed_permuted.shape, dtype=torch.float8_e4m3fn, device=v_transposed_permuted.device)
    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v_transposed_permuted.device)
    scale_max = 2.25 if arch in {"sm89", "sm100", "sm120", "sm121"} else 448.0
    _fused.scale_fuse_quant_cuda(v_transposed_permuted, v_fp8, v_scale, kv_len, scale_max, 1)
    del v_transposed_permuted
    return v_fp8, v_scale


def spas_sage_attn_meansim(q, k=None, v=None, *args, **kwargs):
    q, k, v = _unpack_qkv(q, k, v)
    arch = _cuda_archs[q.device.index]
    if arch in {"sm70", "sm75"}:
        return spas_sage_attn_meansim_triton(q, k, v, *args, **kwargs)
    else:
        return spas_sage2_attn_meansim_cuda(q, k, v, *args, **kwargs)


def spas_sage2_attn_meansim_cuda(q, k=None, v=None, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, recycle_q=False):
    q, k, v = _unpack_qkv(q, k, v)
    assert tensor_layout in ['HND', 'NHD']
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    q_size = q.size()
    q_device = q.device
    q_dtype = q.dtype
    q_heads = q.size(-3)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    else:
        km = None
    headdim = q.size(-1)

    arch = _cuda_archs[q.device.index]
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)
    del k, km

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q_heads, q_device)

    if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
        v_ref = [v]
        v = None
        v_fp8, v_scale = _quantize_v_fp8(v_ref, arch)

    _is_causal = 1 if is_causal else 0
    if recycle_q:
        o = q
    else:
        del q
        o = torch.empty(q_size, dtype=q_dtype, device=q_device)

    if arch in {"sm80", "sm86", "sm87"}:
        assert SM80_ENABLED
        _qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
        del v
    elif arch in {"sm89", "sm100", "sm120", "sm121"}:
        assert SM89_ENABLED
        if get_cuda_version() < (12, 8):
            f = _qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        else:
            f = _qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        f(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    elif arch == "sm90":
        assert SM90_ENABLED
        _qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

    del q_int8, k_int8, q_scale, k_scale
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        del lut, valid_block_num
        return o, qk_sparsity.item()
    else:
        del lut, valid_block_num
        return o


def spas_sage2_attn_meansim_topk_cuda(q, k=None, v=None, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, recycle_q=False):
    q, k, v = _unpack_qkv(q, k, v)
    assert tensor_layout in ['HND', 'NHD']
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    q_size = q.size()
    q_device = q.device
    q_dtype = q.dtype
    q_heads = q.size(-3)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    else:
        km = None
    headdim = q.size(-1)

    arch = _cuda_archs[q.device.index]
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)
    del k, km

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q_heads, q_device)

    if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
        v_ref = [v]
        v = None
        v_fp8, v_scale = _quantize_v_fp8(v_ref, arch)

    _is_causal = 1 if is_causal else 0
    if recycle_q:
        o = q
    else:
        del q
        o = torch.empty(q_size, dtype=q_dtype, device=q_device)

    if arch in {"sm80", "sm86", "sm87"}:
        assert SM80_ENABLED
        _qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
        del v
    elif arch in {"sm89", "sm100", "sm120", "sm121"}:
        assert SM89_ENABLED
        if get_cuda_version() < (12, 8):
            f = _qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        else:
            f = _qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        f(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    elif arch == "sm90":
        assert SM90_ENABLED
        _qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

    del q_int8, k_int8, q_scale, k_scale
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        del lut, valid_block_num
        return o, qk_sparsity.item()
    else:
        del lut, valid_block_num
        return o


def block_sparse_attn_cuda(q, k=None, v=None, mask_id=None, dropout_p=0.0, scale=None, smooth_k=True, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, recycle_q=False):
    q, k, v = _unpack_qkv(q, k, v)
    mask_id = _unpack_tensor(mask_id)
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q = rearrange(q, '... L H D -> ... H L D')
        k = rearrange(k, '... L H D -> ... H L D')
        v = rearrange(v, '... L H D -> ... H L D')
    arch = _cuda_archs[q.device.index]
    v_ref = [v]
    v = None
    v = _contiguous_to_dtype(v_ref, torch.float16)
    if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
        v_ref = [v]
        v = None
        v_fp8, v_scale = _quantize_v_fp8(v_ref, arch)
    qk_dtype = torch.float16 if q.dtype in (torch.float32, torch.float16) else torch.bfloat16
    q_ref = [q]
    q = None
    q = _contiguous_to_dtype(q_ref, qk_dtype)
    k_ref = [k]
    k = None
    k = _contiguous_to_dtype(k_ref, qk_dtype)
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    q_size = q.size()
    q_device = q.device
    q_dtype = q.dtype
    q_heads = q.size(-3)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    else:
        km = None
    headdim = q.size(-1)
    
    if arch == "sm90":
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 128)
    else:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 128, 64)
    del k, km
    lut, valid_block_num = block_map_lut_triton(block_map=mask_id)
    del mask_id
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q_heads, q_device)



    if recycle_q:
        o = q
    else:
        del q
        o = torch.empty(q_size, dtype=q_dtype, device=q_device)

    if arch in {"sm80", "sm86", "sm87"}:
        assert SM80_ENABLED
        _qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0)
        del v
    elif arch in {"sm89", "sm100", "sm120", "sm121"}:
        assert SM89_ENABLED
        if get_cuda_version() < (12, 8):
            f = _qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        else:
            f = _qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        f(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        del v_fp8, v_scale
    elif arch == "sm90":
        assert SM90_ENABLED
        _qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        del v_fp8, v_scale
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

    del q_int8, k_int8, q_scale, k_scale
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        del lut, valid_block_num
        return o, qk_sparsity.item()
    else:
        del lut, valid_block_num
        return o


# For compatibility. spas_sage2_attn_meansim_cuda and spas_sage2_attn_meansim_topk_cuda already support sm80/86/89/90/120
spas_sage_attn_meansim_cuda = spas_sage2_attn_meansim_cuda
spas_sage_attn_meansim_topk_cuda = spas_sage2_attn_meansim_topk_cuda
