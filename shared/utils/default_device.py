"""Torch default device policy for a model's generate() call.

mmgp installs torch.set_default_device("cuda") when it loads a pipeline, so tensors created without an explicit device
land on the GPU. That default is a TorchFunctionMode: every torch call made by the thread goes through Python first,
which costs a measurable share of host time in host-bound phases. Models whose model definition declares
"device_explicit": True create every tensor with an explicit device and run their generate() without any default device.

WANGP_AUDIT_DEFAULT_DEVICE=<log path> keeps the current default and appends, for each generate() call, every
factory call that relied on it (file:line, calling function, torch function, count).
"""
import collections
import contextlib
import os
import sys
import time

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._device import _device_constructors

_TORCH_DIR = os.path.dirname(torch.__file__)


def device_explicit(model_def):
    return model_def.get("device_explicit", False)


class _DefaultDeviceAudit(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.sites = collections.Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get("device") is None and func in _device_constructors():
            frame = sys._getframe(1)
            while frame is not None and (frame.f_code.co_filename.startswith(_TORCH_DIR) or frame.f_code.co_filename == __file__):
                frame = frame.f_back
            if frame is not None:
                self.sites[(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name, getattr(func, "__name__", str(func)))] += 1
        return func(*args, **kwargs)


def _write_audit(sites, label, path):
    lines = [f"== {time.strftime('%Y-%m-%d %H:%M:%S')} {label}: {len(sites)} call sites relied on the default device"]
    lines += [f"{count:8d}  {file}:{line} {function} [{func}]" for (file, line, function, func), count in sites.most_common()]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(lines[0] + f" (details in {path})")


@contextlib.contextmanager
def keep_default_device():
    """Restores the thread's default device on exit, e.g. around an mmgp offload.profile() call made during generate(): it
    leaves the CUDA default device installed."""
    context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
    try:
        yield
    finally:
        torch.set_default_device(None if context is None else context.device)


@contextlib.contextmanager
def generation_default_device(model_def, label=""):
    audit_path = os.environ.get("WANGP_AUDIT_DEFAULT_DEVICE", "")
    if audit_path:
        mode = _DefaultDeviceAudit()
        try:
            with mode:
                yield
        finally:
            _write_audit(mode.sites, label, audit_path)
        return
    if not device_explicit(model_def):
        yield
        return
    with keep_default_device():
        torch.set_default_device(None)
        yield


def call_with_default_device(model_def, label, fn, *args, **kwargs):
    with generation_default_device(model_def, label):
        return fn(*args, **kwargs)
