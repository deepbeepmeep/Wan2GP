_PROBE_CACHE = None
_WARNED_REQUESTED_VLLM_UNAVAILABLE = False


def _check_triton():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception as exc:
        return False, f"Triton import failed: {exc}"
    return True, "ok"


def _check_flash_attention_2():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        version = str(getattr(flash_attn, "__version__", ""))
    except Exception as exc:
        return False, f"FlashAttention import failed: {exc}"

    major = None
    if len(version) > 0:
        try:
            major = int(version.split(".", 1)[0])
        except Exception:
            major = None
    if major is not None and major < 2:
        return False, f"FlashAttention major version is {major}, expected >= 2"
    return True, "ok"


def _load_linear_module():
    import importlib.util
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    linear_path = os.path.join(base_dir, "layers", "linear.py")
    if not os.path.isfile(linear_path):
        raise RuntimeError(f"Missing nanovllm linear kernel file: {linear_path}")

    spec = importlib.util.spec_from_file_location("nanovllm_linear_probe", linear_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to build import spec for nanovllm linear kernel probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_triton_int8_kernel():
    import inspect
    import torch

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        linear_module = _load_linear_module()
        if not getattr(linear_module, "_TRITON_AVAILABLE", False):
            return False, "nanovllm Triton path is disabled"

        run_kernel = getattr(linear_module, "_run_triton_fused_int8_mm", None)
        if run_kernel is None:
            return False, "nanovllm Triton int8 kernel entrypoint is missing"

        device = torch.device("cuda")
        x = torch.randn((2, 64), device=device, dtype=torch.bfloat16)
        qweight_t = torch.randint(-127, 128, (64, 32), device=device, dtype=torch.int8)
        qweight_scale = torch.ones((32,), device=device, dtype=torch.float32)
        # Support both legacy (4-arg) and current (3-arg) probe signatures.
        param_count = len(inspect.signature(run_kernel).parameters)
        if param_count >= 4:
            out = run_kernel(x, qweight_t, qweight_scale, 0.01)
        else:
            out = run_kernel(x, qweight_t, qweight_scale)
        torch.cuda.synchronize()

        if tuple(out.shape) != (2, 32):
            return False, f"Unexpected kernel output shape: {tuple(out.shape)}"
        if not torch.isfinite(out).all().item():
            return False, "Kernel output contains non-finite values"
    except Exception as exc:
        return False, f"Triton int8 kernel smoke test failed: {exc}"

    return True, "ok"


def probe_vllm_runtime(force=False):
    global _PROBE_CACHE
    if _PROBE_CACHE is not None and not force:
        return _PROBE_CACHE.copy()

    checks = {}

    triton_ok, triton_msg = _check_triton()
    checks["triton"] = {"ok": triton_ok, "message": triton_msg}

    flash_ok, flash_msg = _check_flash_attention_2()
    checks["flash_attention_2"] = {"ok": flash_ok, "message": flash_msg}

    kernel_ok, kernel_msg = _check_triton_int8_kernel()
    checks["triton_int8_kernel"] = {"ok": kernel_ok, "message": kernel_msg}

    supported = triton_ok and flash_ok and kernel_ok
    result = {
        "supported": supported,
        "preferred_engine": "vllm" if supported else "legacy",
        "checks": checks,
    }

    _PROBE_CACHE = result.copy()
    return result


def resolve_lm_decoder_engine(requested_engine):
    probe_result = probe_vllm_runtime()
    supported = bool(probe_result.get("supported", False))
    if requested_engine == "vllm":
        if supported:
            return "vllm"
        global _WARNED_REQUESTED_VLLM_UNAVAILABLE
        if not _WARNED_REQUESTED_VLLM_UNAVAILABLE:
            checks = probe_result.get("checks", {})
            reasons = []
            if isinstance(checks, dict):
                for check_name, check_data in checks.items():
                    if isinstance(check_data, dict) and not check_data.get("ok", False):
                        msg = str(check_data.get("message", "failed")).replace("\n", " ").strip()
                        if len(msg) > 220:
                            msg = msg[:220] + "..."
                        reasons.append(f"{check_name}={msg}")
            reason_text = "; ".join(reasons) if len(reasons) > 0 else "unknown reason"
            print(f"[LM] Requested decoder engine 'vllm' is unavailable at startup ({reason_text}).")
            _WARNED_REQUESTED_VLLM_UNAVAILABLE = True
        return "legacy"
    if requested_engine == "":
        return "vllm" if supported else "legacy"
    return requested_engine
