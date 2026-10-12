# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Autotuning identities describe model geometry and operations, never lengths."""
import inspect
import triton
import torch
from triton.runtime.autotuner import Autotuner

from .triton_compilation_log import install_triton_compilation_logger


class InferenceAutotuner(Autotuner):
    """Never benchmark during graph recording, nor cache its static choice."""
    def run(self, *args, **kwargs):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            named = {**dict(zip(self.arg_names, args)), **kwargs}
            inputs = {name: value for name, value in named.items() if name in self.arg_names}
            key = tuple([inputs[name] for name in self.keys if name in inputs] +
                        [str(value.dtype) for value in inputs.values() if hasattr(value, "dtype")])
            config = self.cache.get(key, self.configs[0])
            if config.pre_hook is not None:
                config.pre_hook({**named, **config.all_kwargs()})
            return self.fn.run(*args, **kwargs, **config.all_kwargs())
        return super().run(*args, **kwargs)


def model_autotune(*, configs, key=None, **options):
    if "cache_results" in inspect.signature(triton.autotune).parameters:
        options.setdefault("cache_results", True)
    decorate = triton.autotune(configs=configs, key=[], **options)
    configured = {name for config in configs for name in config.kwargs}

    def apply(kernel):
        tuned = decorate(kernel)
        tuned.__class__ = InferenceAutotuner
        leaf = kernel
        while hasattr(leaf, "fn") and not hasattr(leaf, "params"):
            leaf = leaf.fn
        tuned.keys = [param.name for param in leaf.params if param.is_constexpr and param.name not in configured]
        return tuned
    return apply


install_triton_compilation_logger()
