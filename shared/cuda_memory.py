"""CUDA memory settings applied when WanGP starts, before anything initializes CUDA: the VRAM allocator (Configuration > RAM/VRAM
Management or --vram-allocator), its debug mode (--vram-debug) and the per-thread stack reserve of the CUDA context."""
import ctypes
import json
import os
import sys
import time

VRAM_ALLOCATOR_KEY = "vram_allocator"
VRAM_ALLOCATOR_CHOICES = ("default", "vmm", "vmm_spill")
_vram_debug = False
# The CUDA context reserves this much local memory for the call stack of every thread the GPU can run at once: with the driver's 1 KiB,
# about 0.25 GiB of VRAM on a large GPU. WanGP's kernels need less; the driver raises the reserve when a kernel needs more.
# WANGP_CUDA_STACK_BYTES overrides it (0 keeps the driver's default).
CUDA_STACK_BYTES = 256


def _argv_value(argv, option):
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return None


def requested_vram_allocator(argv, config_filename):
    # wgp.py reads its configuration after CUDA has started: the allocator choice is read from the same file first
    value = _argv_value(argv, "--vram-allocator")
    if value is None:
        config_dir = _argv_value(argv, "--config")
        for path in ([os.path.join(os.path.abspath(config_dir), config_filename)] if config_dir else []) + [config_filename]:
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as reader:
                    value = json.load(reader).get(VRAM_ALLOCATOR_KEY)
                break
    return value or "vmm_spill"


def apply_startup_settings(argv, config_filename):
    import torch
    if torch.version.hip is not None or not torch.cuda.is_available():
        return
    allocator = requested_vram_allocator(argv, config_filename)
    if allocator not in VRAM_ALLOCATOR_CHOICES:
        raise ValueError(f"Unknown VRAM allocator {allocator!r}: expected one of {VRAM_ALLOCATOR_CHOICES}")
    if allocator in ("vmm", "vmm_spill"):
        from mmgp import allocator as vram_allocator
        try:
            vram_allocator.install(spill=allocator == "vmm_spill")
        except RuntimeError as error:  # a GPU, driver, platform or PyTorch it does not support: the default must not stop WanGP
            print(f"[VRAM] MMGP Optimized VRAM Allocator not available ({error}): PyTorch's allocator is used")
            allocator = "default"
        else:
            print("[VRAM] MMGP Optimized VRAM Allocator" + (", spilling into RAM when VRAM is full" if allocator == "vmm_spill" else ""))
    debug_mb = float(_argv_value(argv, "--vram-debug") or 0)
    if debug_mb > 0:
        if allocator not in ("vmm", "vmm_spill"):
            raise ValueError("--vram-debug needs the MMGP allocator: --vram-allocator vmm or vmm_spill")
        from mmgp.allocator import debug
        debug.start(min_mb=debug_mb)
        global _vram_debug
        _vram_debug = True
        print(f"[VRAM] Debug mode: allocations of {debug_mb:g} MB and more recorded, a report is written after each generation in <outputs>/vram_debug")
    stack_bytes = int(os.environ.get("WANGP_CUDA_STACK_BYTES", CUDA_STACK_BYTES))
    if stack_bytes > 0:
        torch.zeros(1, device="cuda")  # creates the context, made current in this thread
        driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        result = driver.cuCtxSetLimit(0, ctypes.c_size_t(stack_bytes))  # CU_LIMIT_STACK_SIZE
        if result != 0:
            raise RuntimeError(f"cuCtxSetLimit(CU_LIMIT_STACK_SIZE, {stack_bytes}) failed with CUDA error {result}")


def write_vram_debug_report(output_dir, label):
    """With --vram-debug: the report of the generation that just ended, in output_dir/vram_debug; then new peaks for the next one."""
    if not _vram_debug:
        return
    from mmgp.allocator import debug
    path = os.path.join(output_dir, "vram_debug", f"{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}_{label or 'generation'}.json")
    debug.report(path)
    debug.reset()
    print(f"[VRAM debug] Report: {path} (summary: {os.path.splitext(path)[0]}.md)")
