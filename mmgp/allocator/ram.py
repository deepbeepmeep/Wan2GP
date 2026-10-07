# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MMGP RAM allocator for PyTorch's CPU tensors (Windows and Linux, PyTorch 2.6 to 2.15).

    from mmgp.allocator import ram
    ram.install()     # after importing torch: the CPU tensors allocated from then on use it
    ram.release()     # the freed blocks it keeps go back to the system now (the end of a generation, a model release)

The CPU tensors of min_mb and more get their own pages, committed at their size, instead of a block of PyTorch's CPU allocator (mimalloc on
Windows, glibc's arenas on Linux), which keeps the memory of freed tensors for the rest of the process and on Windows commits up to twice
what a tensor of 1 to 64 MiB needs. The freed blocks are kept for reuse (a loop that allocates the same sizes again reuses pages already in
memory, without page faults), at most cache_gb of them. They go back to the system on demand, with release() (call it at the end of a
workload) or when the MMGP VRAM allocator would otherwise refuse to spill into RAM, and when the workload needs it: beyond cache_gb, or when
new pages are wanted while the system's available RAM is under pressure_gb, in the thread that frees or allocates: no background thread,
no timer, nothing runs while the GPU works. Smaller tensors keep PyTorch's allocator. The library also carries the hooks of the RAM debug mode (ram_debug.py)."""
import ctypes
import os
import platform
import sys

import torch

GB, MB = 2 ** 30, 2 ** 20
_LIBRARIES = {("win32", "amd64"): "ram_alloc_win_amd64.dll", ("linux", "x86_64"): "ram_alloc_linux_x86_64.so"}
TORCH_VERSIONS = ((2, 6), (2, 15))  # c10::Allocator, DataPtr and SetAllocator checked identical from 2.6 to the 2.15 development branch
STATS = ("enabled", "live", "cached", "peak", "allocations", "hits", "fresh", "trims", "fallbacks", "releases", "released_bytes", "reserved")
_lib, _reason = None, None
active = False
settings = {}


def library():
    """(the library, None), or (None, the reason it cannot be used): loaded once."""
    global _lib, _reason
    if _lib is not None or _reason is not None:
        return _lib, _reason
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    name = _LIBRARIES.get((sys.platform, platform.machine().lower()))
    path = None if name is None else os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if name is None:
        _reason = f"no library for {sys.platform} {platform.machine()}"
    elif not TORCH_VERSIONS[0] <= version <= TORCH_VERSIONS[1]:
        _reason = f"PyTorch {torch.__version__} is not among the versions checked for it (2.6 to 2.15)"
    elif sys.platform.startswith("linux") and not torch.compiled_with_cxx11_abi():
        _reason = "this PyTorch build uses the old C++ ABI"
    elif not os.path.isfile(path):
        _reason = f"{name} is missing (python -m mmgp.allocator.build)"
    else:
        try:
            lib = ctypes.CDLL(path)
        except OSError as error:
            _reason = f"{name} could not be loaded ({error})"
        else:
            lib.ra_install.argtypes = (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
            lib.ra_install.restype = ctypes.c_int
            lib.ra_set_enabled.argtypes = (ctypes.c_int,)
            lib.ra_stats.argtypes = (ctypes.POINTER(ctypes.c_int64),)
            lib.ra_ranges.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64)
            lib.ra_ranges.restype = ctypes.c_int64
            _lib = lib
    return _lib, _reason


def install(min_mb=1, cache_gb=None, pressure_gb=None):
    """Makes the allocator PyTorch's CPU allocator for the tensors of min_mb and more. cache_gb: the most freed blocks kept for reuse (default:
    5% of the RAM, at most 2 GiB); pressure_gb: under this much available RAM, the freed blocks go back at once (default: the margin the VRAM
    allocator leaves before spilling into RAM, 10% of the RAM and at least 4 GiB, plus cache_gb). Raises RuntimeError when the allocator cannot
    be used here."""
    global active
    lib, reason = library()
    if lib is None:
        raise RuntimeError(f"The MMGP RAM allocator cannot be used: {reason}.")
    import psutil
    physical = psutil.virtual_memory().total
    cache = int(cache_gb * GB) if cache_gb is not None else min(2 * GB, physical // 20)
    pressure = int(pressure_gb * GB) if pressure_gb is not None else max(4 * GB, physical // 10) + cache
    if lib.ra_install(int(min_mb * MB), cache, pressure) != 0:
        raise RuntimeError("The MMGP RAM allocator cannot be used: PyTorch's CPU allocator could not be replaced.")
    active = True
    _link_vram_allocator()
    settings.update(min_mb=min_mb, cache_gb=round(cache / GB, 2), pressure_gb=round(pressure / GB, 2))


def _link_vram_allocator():
    # the MMGP VRAM allocator empties this cache before refusing to spill into RAM for lack of it (whichever is installed first)
    from . import _lib as vram_library
    if active and vram_library is not None:
        vram_library.vmm_set_ram_release(ctypes.cast(_lib.ra_release, ctypes.c_void_p))


def set_enabled(enabled):
    """New CPU tensors use the allocator (True) or PyTorch's (False); the blocks it handed out are freed by it either way."""
    _lib.ra_set_enabled(1 if enabled else 0)


def release():
    """The freed blocks kept for reuse go back to the system now."""
    if active:
        _lib.ra_release()


def reset_peak():
    if active:
        _lib.ra_reset_peak()


def stats():
    """Bytes in use (live), kept for reuse (cached), their peak, and counts: allocations, cache hits, fresh blocks, trimmed blocks, requests
    left to PyTorch's allocator (fallbacks), releases, released bytes, reserved address space."""
    values = (ctypes.c_int64 * len(STATS))()
    if _lib is not None:
        _lib.ra_stats(values)
    return dict(zip(STATS, values))


def ranges():
    """The address ranges the allocator reserved, as (base, size)."""
    if _lib is None:
        return []
    out = (ctypes.c_uint64 * 2048)()
    count = min(1024, _lib.ra_ranges(out, 1024))
    return [(out[2 * index], out[2 * index + 1]) for index in range(count)]
