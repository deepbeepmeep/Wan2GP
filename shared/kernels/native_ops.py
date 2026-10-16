# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional ahead-of-time WGPKernels CUDA operations, separate from INT8 linears."""
import functools
import torch

try:
    from wgpkernels import ops as implementation
    implementation.load_library()
except (ImportError, OSError):
    implementation = None


@functools.cache
def available(device):
    # Eligibility before launch; never hide kernel execution/ABI errors.
    return implementation is not None and device.type == 'cuda' and torch.version.hip is None and torch.cuda.get_device_capability(device) == (12, 0)
