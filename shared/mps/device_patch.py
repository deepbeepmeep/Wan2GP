"""Apple Silicon MPS compatibility patch for Wan2GP.

Import EARLY at startup — BEFORE any mmgp import. Patches torch.cuda functions
to redirect to MPS, disables torch.compile (CUDA-less PyTorch build), and adds
stub attributes for CUDA-only code paths.
"""
import os
import sys
import types

# CRITICAL: Disable torch.compile / dynamo before torch is imported.
# PyTorch 2.11 on macOS is built with USE_CUDA=OFF. torch.compile traces into
# functions like torch.manual_seed, follows the CUDA call chain, and hits
# C++-level "not linked with cuda" errors that Python patches cannot intercept.
os.environ.setdefault('TORCH_COMPILE', '0')
os.environ.setdefault('TORCHINDUCTOR', '0')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

_MPS_PATCH_LOGGED = False


def apply_mps_patch():
    """Patch torch.cuda functions for MPS compatibility."""
    global _MPS_PATCH_LOGGED
    log_this_call = not _MPS_PATCH_LOGGED
    _MPS_PATCH_LOGGED = True

    import torch as _torch

    chip_name = _get_chip_name()
    system_ram_gb = _get_system_memory_gb()
    total_memory_bytes = int(system_ram_gb * 1024 ** 3)

    if 'M1' in chip_name or 'M2' in chip_name:
        dev_cap = (7, 0)
        bfloat16_supported = False
    else:
        dev_cap = (11, 0)
        bfloat16_supported = True

    if log_this_call:
        print(f"[MPS] Detected: {chip_name}, {system_ram_gb:.0f}GB RAM")
        print(f"[MPS] Device capability: {dev_cap}, BF16: {bfloat16_supported}")

    # Dummy objects
    _dummy_stream = types.SimpleNamespace(
        synchronize=_torch.mps.synchronize,
        wait_stream=lambda *a, **kw: _torch.mps.synchronize(),
        query=lambda: True,
        priority=0,
    )

    class _CudaDeviceProperties:
        def __init__(self):
            self.total_memory = total_memory_bytes
            self.name = chip_name
            self.major = dev_cap[0]
            self.minor = dev_cap[1]
            _torch.cuda._device_props_cache = self
        multi_processor_count = 0
        warp_size = 32

    class _DummyEvent:
        def __init__(self, *a, **kw): pass
        def record(self, *a, **kw): _torch.mps.synchronize()
        def elapsed_time(self, *a, **kw): return 0.0
        def synchronize(self, *a, **kw): _torch.mps.synchronize()
        def query(self): return True

    class _DummyDeviceContext:
        def __init__(self, device=None): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class _DummyStreamContext:
        def __init__(self, s): pass
        def __enter__(self):
            _torch.mps.synchronize()
            return self
        def __exit__(self, *a):
            _torch.mps.synchronize()

    class _DummyGraph:
        replay = lambda s, *a, **kw: None
        capture_begin = lambda s, *a, **kw: None
        capture_end = lambda s, *a, **kw: None

    # AMP stub
    class _MpsAutocast:
        def __init__(self, enabled=True, dtype=None, device_type='mps', cache_enabled=None):
            self._autocast = _torch.autocast('mps', enabled=enabled, dtype=dtype)
        def __enter__(self): return self._autocast.__enter__()
        def __exit__(self, *a): return self._autocast.__exit__(*a)

    class _autocast_mode_mod:
        autocast = _MpsAutocast

    class _amp_common:
        @staticmethod
        def amp_definitely_not_available():
            return True

    class _PatchedAMP:
        autocast = _MpsAutocast
        autocast_mode = _autocast_mode_mod
        common = _amp_common
        class GradScaler:
            def __init__(self, *a, **kw): pass
            def step(self, *a, **kw): return a[0] if a else None
            def update(self, *a, **kw): pass
            def unscale_(self, *a, **kw): pass
            def get_scale(self): return 1.0
            def state_dict(self): return {}
            def load_state_dict(self, *a): pass

    _cuda = _torch.cuda

    # Core function patches
    _orig_mps_empty_cache = _torch.mps.empty_cache
    def _safe_mps_empty_cache(*args, **kwargs):
        _torch.mps.synchronize()
        result = _orig_mps_empty_cache(*args, **kwargs)
        _torch.mps.synchronize()
        return result

    _cuda.is_available = lambda: False
    _cuda._is_compiled = lambda: False
    _cuda.empty_cache = _safe_mps_empty_cache
    _cuda.synchronize = _torch.mps.synchronize
    _cuda.get_device_capability = lambda device=None: dev_cap
    _cuda.manual_seed_all = lambda seed: None
    _cuda.manual_seed = lambda device_or_seed, seed=None: None
    _cuda.current_stream = lambda device=None: _dummy_stream
    _cuda.get_device_properties = lambda device=None: _CudaDeviceProperties()
    _cuda._CudaDeviceProperties = _CudaDeviceProperties
    _cuda.default_stream = lambda device=None: _dummy_stream
    _cuda.set_device = lambda device: None
    _cuda.current_device = lambda: 0
    _cuda.device_count = lambda: 1
    _cuda.ipc_collect = lambda: None
    _cuda.device = _DummyDeviceContext

    class _PatchedStream:
        priority = 0
        def __init__(self, *a, **kw): pass
        def synchronize(self, *a, **kw): _torch.mps.synchronize()
        def wait_stream(self, *a, **kw): _torch.mps.synchronize()
        def query(self): return True

    _cuda.Stream = _PatchedStream
    _cuda.stream = lambda s: _DummyStreamContext(s)
    _cuda.Event = _DummyEvent
    _cuda.is_bf16_supported = lambda device=None: bfloat16_supported
    _cuda.bfloat16_supported = lambda device=None: bfloat16_supported
    _cuda.is_current_stream_capturing = lambda: False
    _cuda.graph = lambda *a, **kw: _DummyGraph()
    _cuda.CUDAGraph = _DummyGraph
    _cuda.graph_pool_handle = lambda: None
    _cuda.mem_get_info = lambda device=None: (total_memory_bytes, total_memory_bytes)
    _cuda.memory_allocated = lambda device=None: 0
    _cuda.memory_reserved = lambda device=None: 0
    _cuda.max_memory_allocated = lambda device=None: 0
    _cuda.max_memory_reserved = lambda device=None: 0
    _cuda.reset_peak_memory_stats = lambda device=None: None
    _cuda.memory_stats = lambda device=None: {}
    try:
        import torch.cuda.amp as _cuda_amp
        _cuda_amp.autocast = _MpsAutocast
        _cuda_amp.GradScaler = _PatchedAMP.GradScaler
        _cuda.amp = _cuda_amp
    except Exception:
        _cuda.amp = _PatchedAMP
    _cuda.is_initialized = lambda: True
    _cuda._lazy_init = lambda: None

    # CRITICAL: Patch torch.manual_seed to avoid internal CUDA calls.
    # torch.manual_seed calls torch.cuda.manual_seed_all internally. Even with
    # cuda.manual_seed_all patched to no-op, torch.compile tracing into manual_seed
    # can trigger C++-level CUDA failures. Replace it with MPS-only version.
    _orig_manual_seed = _torch.manual_seed
    def _mps_manual_seed(seed):
        seed = int(seed)
        _orig_manual_seed(seed)  # CPU seed
        _torch.mps.manual_seed(seed)  # MPS seed
        return _torch._C.Generator()
    _torch.manual_seed = _mps_manual_seed

    # CRITICAL: Replace torch.compile with a true no-op.
    # PyTorch 2.11 on macOS is built with USE_CUDA=OFF. Even the 'eager' backend
    # involves dynamo tracing which can trigger C++-level CUDA failures.
    # Simply return the original function unchanged.
    def _patched_compile(fn=None, *args, **kwargs):
        if fn is not None:
            return fn
        def decorator(f):
            return f
        return decorator
    _torch.compile = _patched_compile

    # CRITICAL: Patch torch.autocast to redirect 'cuda' -> 'mps'
    # Code uses torch.autocast('cuda', ...) or torch.autocast(device_type='cuda', ...)
    _orig_autocast = _torch.autocast
    def _patched_autocast(device_type=None, *args, **kwargs):
        if device_type == 'cuda':
            device_type = 'mps'
        if kwargs.get('device_type') == 'cuda':
            kwargs['device_type'] = 'mps'
        # Handle torch.cuda.amp.autocast which calls with device_type=None initially
        if device_type is None and 'device_type' not in kwargs:
            device_type = 'mps'
        # MPS only supports fp16/bf16 autocast
        dtype = kwargs.get("dtype", None)
        if device_type == "mps":
            if dtype not in (_torch.float16, _torch.bfloat16, None):
                # safest choice for Apple Silicon
                kwargs["dtype"] = _torch.float16
        return _orig_autocast(device_type, *args, **kwargs)
    _torch.autocast = _patched_autocast
    # Also patch torch.amp.autocast
    _torch.amp.autocast = _patched_autocast

    # Disable torch._dynamo entirely to avoid any traced CUDA calls
    try:
        _torch._dynamo.config.suppress_errors = True
        _torch._dynamo.config.cache_size_limit = 128
    except Exception:
        pass

    # Tensor and Module .cuda() redirects
    def _patched_tensor_cuda(self, device=None, *args, **kwargs):
        return self.to("mps")
    _torch.Tensor.cuda = _patched_tensor_cuda

    def _patched_module_cuda(self, device=None):
        return self.to("mps")
    _torch.nn.Module.cuda = _patched_module_cuda

    def _replace_cuda_device(val):
        if isinstance(val, str) and val.startswith("cuda"):
            return "mps"
        if isinstance(val, _torch.device) and val.type == "cuda":
            return _torch.device("mps")
        return val

    def _replace_map_location(map_location):
        if isinstance(map_location, dict):
            return {key: _replace_cuda_device(value) for key, value in map_location.items()}
        return _replace_cuda_device(map_location)

    # CRITICAL: Patch Tensor.to and Module.to to intercept cuda device strings.
    # This catches code that passes device="cuda" as a string to .to() calls.
    _orig_tensor_to = _torch.Tensor.to
    def _patched_tensor_to(self, *args, **kwargs):
        # Handle positional args: .to("cuda"), .to(device), .to(dtype, device)
        new_args = [_replace_cuda_device(a) for a in args]
        # Handle keyword device arg
        if "device" in kwargs:
            kwargs["device"] = _replace_cuda_device(kwargs["device"])
        target_device = kwargs.get("device", None)
        if target_device is None:
            for arg in new_args:
                if isinstance(arg, str) and arg.startswith("mps"):
                    target_device = arg
                    break
                if isinstance(arg, _torch.device) and arg.type == "mps":
                    target_device = arg
                    break
        if target_device == "mps" or (isinstance(target_device, _torch.device) and target_device.type == "mps"):
            kwargs["non_blocking"] = False
        return _orig_tensor_to(self, *new_args, **kwargs)
    _torch.Tensor.to = _patched_tensor_to

    _orig_module_to = _torch.nn.Module.to
    def _patched_module_to(self, *args, **kwargs):
        new_args = [_replace_cuda_device(a) for a in args]
        if "device" in kwargs:
            kwargs["device"] = _replace_cuda_device(kwargs["device"])
        return _orig_module_to(self, *new_args, **kwargs)
    _torch.nn.Module.to = _patched_module_to

    def _patched_pin_memory(self, *args, **kwargs):
        return self
    _torch.Tensor.pin_memory = _patched_pin_memory

    _orig_load = _torch.load
    def _patched_load(*args, **kwargs):
        if "map_location" in kwargs:
            kwargs["map_location"] = _replace_map_location(kwargs["map_location"])
        elif len(args) >= 2:
            args = (args[0], _replace_map_location(args[1]), *args[2:])
        return _orig_load(*args, **kwargs)
    _torch.load = _patched_load

    # Generator patch
    _Gen = _torch.Generator
    class _PatchedGen(_Gen):
        def __new__(cls, device=None):
            device = _replace_cuda_device(device)
            if device:
                return super().__new__(cls, device=device)
            return super().__new__(cls)
    _torch.Generator = _PatchedGen

    # Tensor creation patch — redirect cuda->mps, fix pin_memory bug
    for fn_name in ['zeros', 'ones', 'randn', 'rand', 'tensor', 'arange',
                     'linspace', 'empty', 'full', 'eye', 'zeros_like', 'ones_like',
                     'randn_like', 'rand_like', 'empty_like', 'full_like',
                     'as_tensor', 'from_numpy']:
        if hasattr(_torch, fn_name):
            orig = getattr(_torch, fn_name)
            def make_patcher(o):
                def patched(*args, **kwargs):
                    dev = kwargs.get('device')
                    new_dev = _replace_cuda_device(dev)
                    if new_dev is not dev:
                        kwargs['device'] = new_dev
                    if 'pin_memory' in kwargs:
                        kwargs.pop('pin_memory', None)
                    return o(*args, **kwargs)
                return patched
            setattr(_torch, fn_name, make_patcher(orig))

    if log_this_call:
        print(f"[MPS] Applied successfully")
        print(f"[MPS] BF16 supported: {bfloat16_supported}")
        print(f"[MPS] Available system RAM: {system_ram_gb:.0f}GB")

    # Fix: Some Wan model loading paths call .weight on an nn.Parameter,
    # which is a Tensor subclass, not a Module. On MPS this fails because
    # nn.Parameter doesn't have a .weight attribute. Duck-type it to return self.
    # Reference: https://github.com/deepbeepmeep/Wan2GP/pull/1750#issuecomment-4387455446
    if not hasattr(_torch.nn.Parameter, "weight"):
        _torch.nn.Parameter.weight = property(lambda self: self)

    return True  # signal success

def _get_chip_name():
    try:
        import subprocess
        out = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'], encoding='utf-8', stderr=subprocess.DEVNULL)
        for line in out.split('\n'):
            if 'Chip' in line:
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return "Unknown Apple Silicon"

def _get_system_memory_gb():
    try:
        import subprocess
        out = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], encoding='utf-8').strip()
        return int(out) / (1024 ** 3)
    except Exception:
        return 16.0

# Auto-apply on import if on macOS with MPS
import torch as _torch
_is_mps = sys.platform == 'darwin' and hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available()

# Patch torch._C missing C extension functions
_C = _torch._C
if not hasattr(_C, '_cuda_getDefaultStream'):
    def _cuda_getDefaultStream_stub(device_index=0):
        return (0, device_index, 0)
    _C._cuda_getDefaultStream = _cuda_getDefaultStream_stub

# Add missing torch.mps attributes
if not hasattr(_torch.mps, 'current_device'):
    _torch.mps.current_device = lambda: 0
if not hasattr(_torch.mps, 'device_count'):
    _torch.mps.device_count = lambda: 1
if not hasattr(_torch.mps, 'set_device'):
    _torch.mps.set_device = lambda device: None

# Native MPS SDPA can still intermittently trip Metal command-buffer assertions
# on WAN video paths. Default to a synchronized matmul fallback for stability.
# Set WAN2GP_MPS_NATIVE_SDPA=1 to opt into native SDPA for diagnostics.
if _is_mps:
    _orig_sdpa = _torch.nn.functional.scaled_dot_product_attention
    _sdpa_mode_announced = [False]

    def _expand_attn_bias(attn_bias, ndim):
        while attn_bias.dim() < ndim:
            attn_bias = attn_bias.unsqueeze(0)
        return attn_bias

    def _manual_sdpa_fallback(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        """Manual attention fallback: matmul + softmax, no Metal SDPA."""
        if enable_gqa:
            repeat = query.size(-3) // key.size(-3)
            key = key.repeat_interleave(repeat, -3)
            value = value.repeat_interleave(repeat, -3)

        L = query.size(-2)
        S = key.size(-2)
        D = query.size(-1)
        scale_factor = D ** -0.5 if scale is None else scale

        attn_bias = _torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            causal_mask = _torch.ones(L, S, dtype=_torch.bool, device=query.device).tril()
            attn_bias = attn_bias.masked_fill(causal_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == _torch.bool:
                attn_bias = _expand_attn_bias(attn_bias, attn_mask.dim())
                attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_bias + attn_mask

        attn_weight = _torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        attn_weight = attn_weight + _expand_attn_bias(attn_bias, query.dim())
        attn_weight = _torch.nn.functional.softmax(attn_weight, dim=-1)

        if dropout_p:
            attn_weight = _torch.dropout(attn_weight, dropout_p, train=True)

        return _torch.matmul(attn_weight, value)

    def _patched_sdpa(*args, **kwargs):
        if os.environ.get("WAN2GP_MPS_NATIVE_SDPA") != "1":
            if not _sdpa_mode_announced[0]:
                print("[MPS] SDPA mode: synchronized manual MPS fallback", flush=True)
                _sdpa_mode_announced[0] = True
            _torch.mps.synchronize()
            out = _manual_sdpa_fallback(*args, **kwargs)
            _torch.mps.synchronize()
            return out
        if not _sdpa_mode_announced[0]:
            print("[MPS] SDPA mode: native MPS diagnostic", flush=True)
            _sdpa_mode_announced[0] = True
        _torch.mps.synchronize()
        out = _orig_sdpa(*args, **kwargs)
        _torch.mps.synchronize()
        return out

    _torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

if _is_mps:
    try:
        apply_mps_patch()
    except Exception as e:
        print(f"[MPS] Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()
