# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Common eligibility and family ordering for activation kernels.

Dispatch never caches tensors, modules, bound methods or shape-specific choices.
An operation supplies only the implementations that satisfy its arithmetic and
layout contract; the existing user policy orders those implementations.
"""
import functools
import torch
from mmgp import cuda_graphs
from . import kernel_policy

_announced = set()


@functools.cache
def _sm120(device_index):
    """Device capabilities are immutable; cache metadata, never activations."""
    return torch.version.hip is None and torch.cuda.get_device_capability(device_index) == (12, 0)


def take(x_list):
    """Consume an owned activation. Callers must drop their other references.

    Residuals, parameters, rotary tables and caches are borrowed explicitly, not
    put in this holder. In-place APIs ending in '_' borrow their destination.
    """
    if not isinstance(x_list, list) or len(x_list) != 1:
        raise ValueError("Expected a one-element activation holder")
    return x_list.pop()


def choose(operation=None, *, wgpkernels=False, kitchen=False, triton=False, owned_triton=False):
    # WanGP-owned Triton implementations occupy the WanGP slot unless a native
    # implementation exists. Disabled concerns INT8 linears, not these fusions.
    family = kernel_policy.first({"wgpkernels": wgpkernels or owned_triton,
                                  "kitchen": kitchen, "triton": triton or owned_triton})
    if operation is not None and not torch.compiler.is_compiling():
        implementation = "WanGP Triton" if family == "wgpkernels" and not wgpkernels else family or "pytorch"
        key = operation, implementation, kernel_policy.precision
        if key not in _announced:
            _announced.add(key)
            print(f"[Kernels] {operation}: {implementation} ({kernel_policy.precision}).")
    return family


def cuda_inference(x, *, dtypes=(torch.bfloat16,), sm120=True, approximate=False, inference=True):
    if (torch.compiler.is_compiling() or (not torch.is_inference_mode_enabled() if inference else torch.is_grad_enabled())
            or type(x) is not torch.Tensor or not x.is_cuda or x.dtype not in dtypes
            or (approximate and not kernel_policy.allow_approximate())):
        return False
    return not sm120 or _sm120(x.device.index)


# A preferred family can change while the resolved INT8 backend stays the same.
# Include the preference itself, including for standalone managed components.
cuda_graphs.register_key(lambda: (kernel_policy.precision, kernel_policy.families))
