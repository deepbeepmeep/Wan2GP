# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Policy for optional kernel optimizations.
Precision: "strict" keeps kernels whose error against a high-precision reference is no larger than PyTorch's (bit identity not
required); "fast" adds kernels that trade precision for speed.
Families: the INT8 Math Kernels choice also orders the kernel families of the other fused operations. The selected family comes first;
when it has no kernel for an operation, or that kernel is unavailable or excluded by the precision, the operation behaves as with Auto
(RANKING). Auto and Disabled (PyTorch INT8 linears) use RANKING. WanGP's own Triton kernels take the WGPKernels slot of an operation
that has no WGPKernels kernel."""


CHOICES = [("Preserve Precision", "strict"), ("Allow Faster Approximate Kernels", "fast")]
precision = "fast"
RANKING = ("wgpkernels", "kitchen", "triton", "pytorch")
families = RANKING


def configure_families(selection):
    global families
    families = (selection,) + tuple(family for family in RANKING if family != selection) if selection in RANKING[:3] else RANKING


def first(candidates):
    """The preferred family among candidates ({family: usable}), or None (the PyTorch path)."""
    for family in families:
        if candidates.get(family):
            return family
    return None


def configure(value):
    global precision
    if value not in ("strict", "fast"):
        raise ValueError(f"Unknown non-INT8 kernel precision: {value!r}")
    precision = value


def allow_approximate():
    return precision == "fast"
