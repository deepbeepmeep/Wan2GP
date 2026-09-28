"""Precision policy for optional non-INT8 kernel optimizations."""

CHOICES = [("Preserve Precision", "strict"), ("Allow Faster Approximate Kernels", "fast")]
precision = "fast"


def configure(value):
    global precision
    if value not in ("strict", "fast"):
        raise ValueError(f"Unknown non-INT8 kernel precision: {value!r}")
    precision = value


def allow_approximate():
    return precision == "fast"
