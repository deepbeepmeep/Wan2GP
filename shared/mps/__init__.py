"""MPS compatibility helpers and local test scripts."""


def is_mps_available():
    import torch
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def mps_device():
    import torch
    return torch.device("mps") if is_mps_available() else None


def mps_device_or(default):
    return mps_device() or default
