# SPDX-License-Identifier: Apache-2.0
"""Public interface for the bundled Sol-Attn Triton kernels."""

from __future__ import annotations

import torch


BLOCK_SIZE = 64
SUPPORTED_CUDA_CAPABILITIES = ((8, 9), (9, 0), (10, 0), (12, 0))


def validate_runtime(device, dtype):
    if device.type != "cuda":
        raise RuntimeError("Sol-Attn requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability not in SUPPORTED_CUDA_CAPABILITIES:
        supported = ", ".join(f"SM{major}{minor}" for major, minor in SUPPORTED_CUDA_CAPABILITIES)
        raise RuntimeError(f"Sol-Attn supports {supported}; got SM{capability[0]}{capability[1]}")
    if dtype != torch.bfloat16:
        raise RuntimeError(f"Sol-Attn requires bfloat16 Q/K/V, got {dtype}")
    try:
        import triton
    except ImportError as error:
        raise RuntimeError("Sol-Attn requires Triton >= 3.6") from error
    version = tuple(int(part) for part in triton.__version__.split(".")[:2])
    if version < (3, 6):
        raise RuntimeError(f"Sol-Attn requires Triton >= 3.6, got {triton.__version__}")
    return capability


def _validate_inputs(q, k, v, thresh_type, sink_tokens=0, sink_start=None, query_start=0):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, T, H, 128]")
    if q.shape[1] == 0 or q.shape[3] != 128:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise ValueError("q, k, and v must have a contiguous head dimension")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    if not isinstance(sink_tokens, int):
        raise TypeError("sink_tokens must be an integer")
    if not 0 <= sink_tokens <= q.shape[1]:
        raise ValueError("sink_tokens must be in [0, T]")
    if sink_start is not None:
        if not isinstance(sink_start, int):
            raise TypeError("sink_start must be an integer or None")
        if not 0 <= sink_start <= q.shape[1]:
            raise ValueError("sink_start must be in [0, T]")
        if sink_start + sink_tokens > q.shape[1]:
            raise ValueError("sink_start + sink_tokens must be <= T")
    if not isinstance(query_start, int) or not 0 <= query_start < q.shape[1]:
        raise ValueError("query_start must be an integer in [0, T)")
    return tuple(torch.cuda.get_device_capability(q.device))


def _sink_block_range(tokens, sink_start, sink_tokens):
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if not sink_tokens:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else sink_start
    return start // BLOCK_SIZE, (start + sink_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def sol_attn(q, k, v, *, scale=None, tau=1.0, thresh_type="diag", sink_tokens=0, sink_start=None, query_start=0,
             recycle_q=False, int8_qk=False):
    if int8_qk:
        capability = _validate_inputs(q, k, v, thresh_type, sink_tokens, sink_start, query_start)
        if query_start:
            raise ValueError("The optimized INT8 Sol-Attn path requires query_start=0")
        if capability < (8, 9):
            raise RuntimeError(f"Triton Sol-Attn requires compute capability >= 8.9; got SM{capability[0]}{capability[1]}")
        from .saganaki import sol_attn as optimized_sol_attn

        sink_blocks = _sink_block_range(q.shape[1], sink_start, sink_tokens)
        return optimized_sol_attn(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type,
                                  int8_qk=True, sink_blocks=sink_blocks)

    from .triton_kernels import sol_attn as triton_sol_attn

    return triton_sol_attn(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type,
                           sink_tokens=sink_tokens, sink_start=sink_start, query_start=query_start, recycle_q=recycle_q)


__all__ = ["sol_attn", "validate_runtime"]
