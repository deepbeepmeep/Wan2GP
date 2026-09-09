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
    compiler = getattr(getattr(triton, "compiler", None), "compiler", None)
    if compilation is None or compiler is None or not hasattr(compilation, "listener") or not hasattr(compiler, "get_cache_manager"):
        return False

    previous_listener = compilation.listener
    if not getattr(previous_listener, _LOGGER_MARKER, False):
        def listener(**event):
            if previous_listener is not None:
                previous_listener(**event)
            source = event.get("src")
            kernel_name = str(getattr(source, "name", "") or type(source).__name__)
            if event.get("cache_hit", True):
                return
            duration_ms = _compile_duration_ms(event.get("times"))
            print(f"[WanGP][Triton] Compiled {kernel_name} in {duration_ms:.0f} ms.", flush=True)

        setattr(listener, _LOGGER_MARKER, True)
        compilation.listener = listener

    previous_get_cache_manager = compiler.get_cache_manager
    if not getattr(previous_get_cache_manager, _LOGGER_MARKER, False):
        def get_cache_manager(key):
            cache_manager = previous_get_cache_manager(key)

            class CompilationCacheManager:
                def __getattr__(self, name):
                    return getattr(cache_manager, name)

                def get_group(self, filename):
                    group = cache_manager.get_group(filename)
                    if filename.endswith(".json") and (compilation.always_compile or not group or group.get(filename) is None):
                        print(f"[WanGP][Triton] Preparing {filename[:-5]} (compiling, please wait)...", flush=True)
                    return group

            return CompilationCacheManager()

        setattr(get_cache_manager, _LOGGER_MARKER, True)
        compiler.get_cache_manager = get_cache_manager
    return True
