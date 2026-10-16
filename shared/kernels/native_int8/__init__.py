# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Experimental native W8A16 kernels. Load an explicitly built library before use."""
import ctypes
import os
from pathlib import Path

import torch

_library = None
_capabilities = {}
_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_SCRATCH_BYTES = 16 * 1024 * 1024


def library_path():
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "WanGP" / "kernels" / "native_int8_v3"
    return root / ("wangp_int8.dll" if os.name == "nt" else "libwangp_int8.so")


def load_library(path=None):
    """Load outside CUDA graph capture; compilation is never triggered by inference."""
    global _library
    lib = ctypes.CDLL(str(Path(path or library_path()).resolve()))
    lib.wangp_int8_abi.restype = ctypes.c_int
    if lib.wangp_int8_abi() != 3:
        raise RuntimeError("Native INT8 library ABI mismatch; rebuild the library")
    ptr, integer, stride = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    lib.wangp_int8_linear.argtypes = [ptr] * 5 + [integer] * 3 + [stride, integer, integer, ptr]
    lib.wangp_int8_linear.restype = integer
    lib.wangp_int8_rotate.argtypes = [ptr, ptr, integer, integer, stride, integer, integer, ptr]
    lib.wangp_int8_rotate.restype = integer
    _library = lib
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        _capabilities[index] = 10 * major + minor


def _check_error(code):
    if code:
        raise RuntimeError(f"Native INT8 CUDA launch failed with error {code}")


@torch.library.custom_op("wangp_native_int8::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None, group: int = 0) -> torch.Tensor:
    """Compute X @ (INT8 weights * FP32 channel scales).T, optionally rotating X.

    Accepts 2-D FP16/BF16/FP32 input, contiguous INT8 (N,K) weights and N scales.
    ConvRot groups are 16, 64 or 256. No activation quantization or weight cache.
    """
    if _library is None:
        raise RuntimeError("Load the native INT8 library before calling linear")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Native INT8 expects X=(M,K) and weights=(N,K)")
    m, k = x.shape
    n = weight.shape[0]
    if k <= 0:
        raise ValueError("Native INT8 requires a positive input feature count")
    if not x.is_cuda or torch.version.hip or x.dtype not in _DTYPES or weight.dtype != torch.int8 or scale.dtype != torch.float32:
        raise ValueError("Native INT8 requires NVIDIA CUDA, floating activations, INT8 weights and FP32 scales")
    if not weight.is_contiguous() or weight.data_ptr() % 4 or not scale.is_contiguous() or scale.numel() != n:
        raise ValueError("Native INT8 requires contiguous, aligned weights and channel scales")
    if weight.device != x.device or scale.device != x.device or (bias is not None and (bias.device != x.device or bias.dtype != x.dtype or bias.numel() != n or not bias.is_contiguous())):
        raise ValueError("Native INT8 tensors must share a device; bias must match the input dtype and output channels")
    if group not in (0, 16, 64, 256) or (group and k % group):
        raise ValueError("ConvRot group must be 16, 64 or 256 and divide K")
    with torch.cuda.device(x.device):
        output = x.new_empty((m, n))
        # Only rotation / non-contiguous inputs require a bounded activation tile.
        rows = max(1, min(65535, _SCRATCH_BYTES // max(1, 2 * k * x.element_size())))
        stream = torch.cuda.current_stream().cuda_stream
        dtype = _DTYPES[x.dtype]
        for start in range(0, m, rows):
            tile = x[start:start + rows]
            if tile.stride(1) != 1:
                tile = tile.contiguous()
            if group:
                rotated = torch.empty_like(tile, memory_format=torch.contiguous_format)
                _check_error(_library.wangp_int8_rotate(tile.data_ptr(), rotated.data_ptr(), len(tile), k, tile.stride(0), dtype, group, stream))
                tile = rotated
            _check_error(_library.wangp_int8_linear(tile.data_ptr(), weight.data_ptr(), scale.data_ptr(), bias.data_ptr() if bias is not None else None, output[start:].data_ptr(), len(tile), n, k, tile.stride(0), dtype, _capabilities[x.device.index], stream))
            del tile
            if group:
                del rotated
        return output


@linear.register_fake
def _linear_fake(x, weight, scale, bias=None, group=0):
    return x.new_empty((x.shape[0], weight.shape[0]))
