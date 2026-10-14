# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import sys
from contextlib import contextmanager

import torch
from importlib.metadata import version
from mmgp import cuda_graphs, offload
import torch.nn.functional as F
import warnings
from importlib.metadata import version

_is_mps = sys.platform == 'darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

major, minor = (0, 0) if _is_mps else torch.cuda.get_device_capability(None)
bfloat16_supported =  major >= 8
_MASKED_ATTENTION_SDPA_WARNED = False
_MISSING = object()

try:
    import triton
    triton_installed = True
except:
    triton_installed = False


# Model-specific attention modes exposed only by per-generation overrides.
# Keep these out of get_attention_modes() so they never appear in the main
# application attention configuration.
ATTENTION_MODE_AVAILABILITY = {
    "sol": {
        "installed": triton_installed,
        "supported": triton_installed and (major, minor) in ((8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)),
    },
    "veda": {
        "installed": triton_installed,
        "supported": triton_installed and major >= 8 and not torch.version.hip,
    },
}


try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    flash_attn = None

try:
    from sageattention import sageattn_varlen
    from . import sage1_core
    def sageattn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            is_causal=False,
            sm_scale=None,
        ):
        return sageattn_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            is_causal=is_causal,
            sm_scale=sm_scale,
        )
    
except ImportError:
    sageattn_varlen_wrapper = sage1_core = None

try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda
except ImportError:
    block_sparse_sage2_attn_cuda = None
    if not triton_installed: 
        try:
            sg2_version = version("sageattention")
            print("Sage Attention has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

try:
    from .sage2_core import sageattn as sageattn2, is_sage2_supported, sageattn_attention_mask_support_reason
    from . import sage2_core
    sage2_supported =  is_sage2_supported()
except ImportError:
    sageattn2 = sage2_core = None
    sage2_supported = False
    sageattn_attention_mask_support_reason = lambda *args, **kwargs: "SageAttention 2 is unavailable"
    if not triton_installed: 
        try:
            sg2_version = version("sageattention")
            if not triton_installed: print("Sage Attention 2 has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

def sage_attention_mask(attention_mask, q_dtype, causal_mask=None):
    """pay_attention's mask ((batch, q, heads, k), 3D or 2D) as SageAttention takes it: (batch, heads, q, k), in q's dtype when it is
    additive; a boolean mask lets the queries that see no key attend to all keys."""
    if attention_mask.ndim == 4:
        attention_mask = attention_mask.transpose(1, 2)
    elif attention_mask.ndim == 3:
        attention_mask = attention_mask.unsqueeze(1)
    elif attention_mask.ndim == 2:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    if torch.is_floating_point(attention_mask):
        attention_mask = attention_mask.to(dtype=q_dtype)
        if causal_mask is not None:
            attention_mask = attention_mask.masked_fill(~causal_mask, torch.finfo(attention_mask.dtype).min)
    elif attention_mask.dtype == torch.bool:
        if causal_mask is not None:
            attention_mask = attention_mask & causal_mask
        has_keys = attention_mask.any(dim=-1, keepdim=True)
        attention_mask = torch.where(has_keys, attention_mask, torch.ones_like(attention_mask))
    return attention_mask


@torch.compiler.disable()
def sageattn2_wrapper(
        qkv_list,
        attention_length,
        recycle_q = False,
        attention_mask = None,
        causal = False,
    ):
    # no reference is kept here: the kernel takes q, k and v out of the list and frees k and v once quantized
    if attention_mask is not None:
        causal_mask = None
        if causal:
            lq, lk, device = qkv_list[0].shape[1], qkv_list[1].shape[1], qkv_list[0].device
            row = torch.arange(lq, device=device)[:, None]
            col = torch.arange(lk, device=device)[None, :]
            causal_mask = (col <= row).view(1, 1, lq, lk)
            causal = False
        attention_mask = sage_attention_mask(attention_mask, qkv_list[0].dtype, causal_mask)
    o = sageattn2(qkv_list, tensor_layout="NHD", is_causal=causal, recycle_q=recycle_q, attn_mask=attention_mask)
    qkv_list.clear()

    return o

try:
    from sageattn import sageattn_blackwell as sageattn3
    if not triton_installed:
        print("Sage Attention 3 is installed but it won't be supported until Triton is installed.")
except ImportError:
    sageattn3 = None
    if not triton_installed: 
        try:
            sg3_version = version("sageattn_blackwell")
            print("Sage Attention 3 has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

if sageattn3 is None:
    try:
        from sageattn3 import sageattn3_blackwell as sageattn3 #word0 windows version
    except ImportError:
        sageattn3 = None
        if not triton_installed: 
            try:
                sg3_version = version("sageattn3_blackwell")
                print("Sage Attention 3 has been detected but it won't work until Triton is installed.")
            except ImportError:
                pass

@torch.compiler.disable()
def sageattn3_wrapper(
        qkv_list,
        attention_length
    ):
    q,k, v = qkv_list
    # qkv_list = [q,k,v]
    # del q, k ,v
    # o = sageattn3(qkv_list, tensor_layout="NHD")
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    o = sageattn3(q, k, v)
    o = o.transpose(1,2)
    qkv_list.clear()

    return o

     


# try:
# if True:
    # from .sage2_core import sageattn_qk_int8_pv_fp8_window_cuda
    # @torch.compiler.disable()
    # def sageattn_window_wrapper(
    #         qkv_list,
    #         attention_length,
    #         window
    #     ):
    #     q,k, v = qkv_list
    #     padding_length = q.shape[0] -attention_length
    #     q = q[:attention_length, :, : ].unsqueeze(0)
    #     k = k[:attention_length, :, : ].unsqueeze(0)
    #     v = v[:attention_length, :, : ].unsqueeze(0)
    #     qkvl_list = [q, k , v]
    #     del q, k ,v
    #     o = sageattn_qk_int8_pv_fp8_window_cuda(qkvl_list, tensor_layout="NHD", window = window).squeeze(0)
    #     qkv_list.clear()

    #     if padding_length > 0:
    #         o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

    #     return o
# except ImportError:
#     sageattn2 = sageattn_qk_int8_pv_fp8_window_cuda

@torch.compiler.disable()
def sdpa_wrapper(
        qkv_list,
        attention_length,
        attention_mask = None,
        causal = False,
    ):
    q, k, v = qkv_list

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    if attention_mask != None:
        attention_mask = attention_mask.transpose(1,2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=causal).transpose(1,2)
    del q, k ,v
    qkv_list.clear()

    return o


def get_attention_modes():
    ret = ["sdpa", "auto"]
    if flash_attn != None:
        ret.append("flash")
    if memory_efficient_attention != None:
        ret.append("xformers")
    if sageattn_varlen_wrapper != None:
        ret.append("sage")
    if sageattn2 != None and version("sageattention").startswith("2") :
        ret.append("sage2")
    if block_sparse_sage2_attn_cuda != None and version("sageattention").startswith("2") :
        ret.append("radial")

    if sageattn3 != None: # and version("sageattention").startswith("3") :
        ret.append("sage3")
        
    return ret

def get_supported_attention_modes():
    # MPS compatibility: only SDPA is supported on Apple Silicon
    if _is_mps:
        return ["sdpa", "auto"]
    ret = get_attention_modes()
    major, minor = torch.cuda.get_device_capability()
    if  major < 10 or not triton_installed:
        if "sage3" in ret:
            ret.remove("sage3")

    if not sage2_supported or not triton_installed:
        if "sage2" in ret:
            ret.remove("sage2")
        if "radial" in ret:
            ret.remove("radial")

    if major < 7 or not triton_installed:
        if "sage" in ret:
            ret.remove("sage")

    return ret


def get_override_attention_modes():
    return get_attention_modes() + [mode for mode, status in ATTENTION_MODE_AVAILABILITY.items() if status["installed"]]


def get_supported_override_attention_modes():
    return get_supported_attention_modes() + [mode for mode, status in ATTENTION_MODE_AVAILABILITY.items() if status["supported"]]


def get_default_attention_mode():
    for attn in ("sage2", "sage", "sdpa"):
        if attn in get_supported_attention_modes():
            return attn
    return "sdpa"


def get_current_cuda_architecture(device=None):
    if _is_mps or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


_SAGE_ATTENTION_MODES = {"sage", "sage2", "sage3", "radial"}


def resolve_attention_mode(attention_mode=None, disable_sage_pre_ada=False):
    attn = str(attention_mode or "auto").strip().lower()
    attn = get_default_attention_mode() if attn == "auto" else attn
    architecture = get_current_cuda_architecture() if disable_sage_pre_ada else None
    if architecture is not None and architecture < 89 and attn in _SAGE_ATTENTION_MODES:
        attn = "sdpa"
    if attn not in get_supported_attention_modes():
        raise ValueError(f"Attention mode '{attn}' is not installed or supported on this system.")
    return attn


@contextmanager
def attention_shared_state(default_attention=None):
    previous = offload.shared_state.get("_attention", _MISSING)
    if previous is _MISSING or previous == "auto":
        offload.shared_state["_attention"] = default_attention or get_default_attention_mode()
    else:
        offload.shared_state["_attention"] = previous
    try:
        yield
    finally:
        if previous is _MISSING:
            offload.shared_state.pop("_attention", None)
        else:
            offload.shared_state["_attention"] = previous


@contextmanager
def attention_config_shared_state(attention_mode=None, resolver=resolve_attention_mode, **resolver_kwargs):
    previous = offload.shared_state.get("_attention", _MISSING)
    offload.shared_state["_attention"] = resolver(attention_mode, **resolver_kwargs)
    try:
        yield offload.shared_state["_attention"]
    finally:
        if previous is _MISSING:
            offload.shared_state.pop("_attention", None)
        else:
            offload.shared_state["_attention"] = previous

def sage_staged(device, head_dim, force_attention=None, masked=False):
    """(core, settings) of the staged_* functions (q, k and v quantized as soon as each is computed) of the SageAttention that
    pay_attention would run without sequence lengths on this device (with this force_attention, and a mask when masked):
    shared.sage2_core for SageAttention 2, shared.sage1_core for SageAttention without a mask; None otherwise."""
    attn = force_attention or offload.shared_state["_attention"]
    if attn == "veda":
        attn = offload.shared_state.get("_attention_fallback") or get_default_attention_mode()
    attn = get_default_attention_mode() if attn in ("sol", "vdn") else attn
    if attn == "sage":
        return None if masked or sage1_core is None or head_dim not in sage1_core.HEAD_DIMS else (sage1_core, sage1_core.SETTINGS)
    if attn not in ("sage2", "radial") or sageattn2 is None or masked and sageattn_attention_mask_support_reason(device=device) is not None:
        return None
    settings = sage2_core.staged_settings(device, masked)
    return None if settings is None else (sage2_core, settings)

__all__ = [
    'ATTENTION_MODE_AVAILABILITY',
    'attention_config_shared_state',
    'attention_shared_state',
    'get_current_cuda_architecture',
    'get_override_attention_modes',
    'get_supported_override_attention_modes',
    'resolve_attention_mode',
    'pay_attention',
    'sage_staged',
    'sage_attention_mask',
    'attention',
]

def get_cu_seqlens(batch_size, lens, max_len):
    # MPS compatibility: use dynamic device detection
    _cu_device = "mps" if _is_mps else "cuda"
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=_cu_device)

    for i in range(batch_size):
        s = lens[i] 
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens

@torch.compiler.disable()
def pay_attention(
    qkv_list,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    version=None,
    force_attention= None,
    attention_mask = None,
    recycle_q = False,
    q_lens = None,
    k_lens = None,
):
    global _MASKED_ATTENTION_SDPA_WARNED
    # format : torch.Size([batches, tokens, heads, head_features])
    # assume if q_lens is non null, each q is padded up to lq (one q out of two will need to be discarded or ignored)
    # assume if k_lens is non null, each k is padded up to lk (one k out of two will need to be discarded or ignored)
    if attention_mask is not None and causal:
        raise ValueError("pay_attention received both attention_mask and causal=True; build a combined mask once and pass causal=False.")
    if attention_mask != None:
        requested_attn = offload.shared_state["_attention"] if force_attention == None else force_attention
        if requested_attn == "veda":
            requested_attn = offload.shared_state.get("_attention_fallback") or get_default_attention_mode()
        requested_attn = get_default_attention_mode() if requested_attn in ("sol", "vdn") else requested_attn
        requested_attn = "sage2" if requested_attn == "radial" else requested_attn
        support_reason = None
        if _is_mps:
            support_reason = "MPS uses SDPA for masked attention"
        elif requested_attn == "sage2" and sageattn2 != None and q_lens == None and k_lens == None:
            support_reason = sageattn_attention_mask_support_reason(qkv_list, attention_mask, tensor_layout="NHD")
        if requested_attn == "sage2" and support_reason is None and sageattn2 != None and q_lens == None and k_lens == None:
            force_attention = "sage2"
        else:
            force_attention = "sdpa"
            if requested_attn != "sdpa" and not _MASKED_ATTENTION_SDPA_WARNED:
                detail = f" ({support_reason})" if support_reason else ""
                print(f"[WAN2GP] Attention mask is unsupported by selected attention '{requested_attn}'{detail}. Masked attention will use SDPA.")
                _MASKED_ATTENTION_SDPA_WARNED = True
        if  attention_mask.dtype == torch.bfloat16 and not bfloat16_supported:
            attention_mask = attention_mask.to(torch.float16)
    attn = offload.shared_state["_attention"] if force_attention== None else force_attention
    if attn == "veda":
        attn = offload.shared_state.get("_attention_fallback") or get_default_attention_mode()
    attn = get_default_attention_mode() if attn in ("sol", "vdn") else attn

    q,k,v = qkv_list
    qkv_list.clear()
    out_dtype = q.dtype
    if q.dtype == torch.bfloat16 and not bfloat16_supported:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    final_padding = 0
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    batch = len(q)
    if len(k) != batch: k = k.expand(batch, -1, -1, -1)
    if len(v) != batch: v = v.expand(batch, -1, -1, -1)
    if q.device.type == "mps": q, k, v = q.contiguous(), k.contiguous(),v.contiguous()
    if attn == "chipmunk":
        from src.chipmunk.modules import SparseDiffMlp, SparseDiffAttn
        from src.chipmunk.util import LayerCounter, GLOBAL_CONFIG
    if attn == "radial": attn ="sage2"
    # a block recorded for a CUDA graph: Sage 3 kernels, which are not known to run on PyTorch's current stream, give way to Sage 2
    if attn == "sage3" and sageattn2 is not None and cuda_graphs.recording(): attn = "sage2"

    if b > 1 and k_lens != None and attn in ("sage2", "sage3", "sdpa"):
        assert attention_mask == None
        # Poor's man var k len attention
        assert q_lens == None
        chunk_sizes = []
        k_sizes = []
        current_size = k_lens[0]
        current_count= 1
        for k_len in k_lens[1:]:
            if k_len == current_size:
                current_count += 1
            else:
                chunk_sizes.append(current_count)
                k_sizes.append(current_size)
                current_count = 1
                current_size = k_len
        chunk_sizes.append(current_count)
        k_sizes.append(k_len)
        if len(chunk_sizes) > 1 or k_lens[0] != k.shape[1]:
            q_chunks =torch.split(q, chunk_sizes)
            k_chunks =torch.split(k, chunk_sizes)
            v_chunks =torch.split(v, chunk_sizes)
            q, k, v = None, None, None
            k_chunks = [ u[:, :sz] for u, sz in zip(k_chunks, k_sizes)]
            v_chunks = [ u[:, :sz] for u, sz in zip(v_chunks, k_sizes)]
            o = []
            for sub_q, sub_k, sub_v in zip(q_chunks, k_chunks, v_chunks): 
                qkv_list = [sub_q, sub_k, sub_v]
                sub_q, sub_k, sub_v = None, None, None
                o.append( pay_attention(qkv_list) )
            q_chunks, k_chunks, v_chunks = None, None, None
            o = torch.cat(o, dim = 0)
            return o
    elif (q_lens != None or k_lens != None) and attn in ("sage2", "sage3", "sdpa"):
        assert b == 1
        szq = q_lens[0].item() if q_lens != None else lq
        szk = k_lens[0].item() if k_lens != None else lk
        final_padding = lq - szq
        q = q[:, :szq]
        k = k[:, :szk]
        v = v[:, :szk]

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # SageAttention on every batch item with all its tokens: its kernels without its device sync (CUDA graphs can record them)
    sage_rows = attn == "sage" and q_lens == None and k_lens == None and not causal and softmax_scale is None and q.shape[-1] in sage1_core.HEAD_DIMS
    if attn=="sage" and not sage_rows or attn=="flash":
        if b != 1 :
            if k_lens == None:
                k_lens = torch.full((b,), lk, dtype=torch.int32, device=q.device)  # made on the GPU: no host copy (not allowed in CUDA graphs)
            if q_lens == None:
                q_lens = torch.full((b,), lq, dtype=torch.int32, device=q.device)
            k = k.reshape(-1, *k.shape[-2:])
            v = v.reshape(-1, *v.shape[-2:])
            q = q.reshape(-1, *q.shape[-2:])
            cu_seqlens_q=get_cu_seqlens(b, q_lens, lq) 
            cu_seqlens_k=get_cu_seqlens(b, k_lens, lk) 
        else:
            szq = q_lens[0].item() if q_lens != None else lq
            szk = k_lens[0].item() if k_lens != None else lk
            if szq != lq or szk != lk:
                cu_seqlens_q = torch.tensor([0, szq, lq], dtype=torch.int32, device=q.device)
                cu_seqlens_k = torch.tensor([0, szk, lk], dtype=torch.int32, device=q.device)
            else:
                cu_seqlens_q = torch.arange(0, 2 * lq, lq, dtype=torch.int32, device=q.device)  # [0, lq] made on the GPU: no host copy (not allowed in CUDA graphs)
                cu_seqlens_k = torch.arange(0, 2 * lk, lk, dtype=torch.int32, device=q.device)
            q = q.squeeze(0)
            k = k.squeeze(0)
            v = v.squeeze(0)


    # apply attention
    if sage_rows:
        qkv_list = [t.reshape(-1, *t.shape[2:]).view(t.shape) for t in (q, k, v)]  # batch items one after another (copies only otherwise)
        del q, k, v
        x = sage1_core.attention(qkv_list, recycle_q)
    elif attn=="sage":
        x = sageattn_varlen_wrapper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_kv= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_kv=lk,
            is_causal=causal,
            sm_scale=softmax_scale,
        ).unflatten(0, (b, lq))
    elif attn=="sage3":
        qkv_list = [q,k,v]
        del q,k,v
        x = sageattn3_wrapper(qkv_list, lq)
    elif attn=="sage2":
        qkv_list = [q,k,v]
        del q,k,v
        x = sageattn2_wrapper(qkv_list, lq, recycle_q=recycle_q, attention_mask=attention_mask, causal=causal)
    elif attn=="sdpa":
        qkv_list = [q, k, v]
        del q ,k ,v
        x = sdpa_wrapper(qkv_list, lq, attention_mask=attention_mask, causal=causal)
    elif attn=="flash" and version == 3:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif attn=="flash":
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output

    elif attn=="xformers":
        from xformers.ops.fmha.attn_bias import BlockDiagonalPaddedKeysMask
        if k_lens == None and q_lens == None:
            x = memory_efficient_attention(q, k, v )
        elif k_lens != None and q_lens == None:
            attn_mask = BlockDiagonalPaddedKeysMask.from_seqlens([lq] * b , lk , list(k_lens) ) 
            x = memory_efficient_attention(q, k, v, attn_bias= attn_mask )
        elif b == 1:
            szq = q_lens[0].item() if q_lens != None else lq
            szk = k_lens[0].item() if k_lens != None else lk
            attn_mask = BlockDiagonalPaddedKeysMask.from_seqlens([szq, lq - szq ] , lk , [szk, 0] ) 
            x = memory_efficient_attention(q, k, v, attn_bias= attn_mask )
        else:
            assert False
    x = x.type(out_dtype)
    if final_padding > 0:
        x = torch.cat([x, torch.empty( (x.shape[0], final_padding, *x.shape[-2:]), dtype= x.dtype, device=x.device  ) ], 1)


    return x 
