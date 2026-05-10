from contextlib import nullcontext

import torch


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_accelerator_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_accelerator_device(device) -> bool:
    if device is None:
        return False
    return torch.device(device).type in {"cuda", "mps"}


def accelerator_autocast(dtype=torch.bfloat16):
    device_type = get_accelerator_device().type
    if device_type in {"cuda", "mps"}:
        return torch.autocast(device_type=device_type, dtype=dtype)
    return nullcontext()


def empty_accelerator_cache():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    elif mps_is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
