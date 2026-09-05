from __future__ import annotations

from typing import Any


_LOGGER_MARKER = "_wangp_triton_compilation_logger"


def _compile_duration_ms(times: Any) -> float:
    lowering_stages = getattr(times, "lowering_stages", ())
    total_us = int(getattr(times, "ir_initialization", 0)) + int(getattr(times, "store_results", 0))
    total_us += sum(int(duration) for _stage, duration in lowering_stages)
    return total_us / 1000.0


def install_triton_compilation_logger() -> bool:
    try:
        import triton
    except Exception:
        return False

    knobs = getattr(triton, "knobs", None)
    compilation = getattr(knobs, "compilation", None)
    runtime = getattr(knobs, "runtime", None)
    if compilation is None or runtime is None or not hasattr(compilation, "listener") or not hasattr(runtime, "jit_cache_hook"):
        return False

    previous_listener = compilation.listener
    if not getattr(previous_listener, _LOGGER_MARKER, False):
        def listener(**event):
            if previous_listener is not None:
                previous_listener(**event)
            source = event.get("src")
            kernel_name = str(getattr(source, "name", "") or type(source).__name__)
            if event.get("cache_hit", True):
                print(f"[WanGP][Triton] Loaded cached {kernel_name}.", flush=True)
                return
            duration_ms = _compile_duration_ms(event.get("times"))
            print(f"[WanGP][Triton] Compiled {kernel_name} in {duration_ms:.0f} ms.", flush=True)

        setattr(listener, _LOGGER_MARKER, True)
        compilation.listener = listener

    previous_jit_cache_hook = runtime.jit_cache_hook
    if not getattr(previous_jit_cache_hook, _LOGGER_MARKER, False):
        def jit_cache_hook(**event):
            handled = previous_jit_cache_hook(**event) if previous_jit_cache_hook is not None else None
            if handled:
                return handled
            function = event.get("fn")
            kernel_name = str(getattr(function, "name", "") or event.get("repr") or "Triton kernel")
            print(f"[WanGP][Triton] Preparing {kernel_name} (loading cache or compiling, please wait)...", flush=True)
            return handled

        setattr(jit_cache_hook, _LOGGER_MARKER, True)
        runtime.jit_cache_hook = jit_cache_hook
    return True
