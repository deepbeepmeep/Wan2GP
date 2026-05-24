from __future__ import annotations

import os
import sys

from collections.abc import MutableSequence


def _arg_name_to_option(arg_name: str) -> str:
    arg_name = str(arg_name or "").strip()
    if not arg_name:
        return ""
    return arg_name if arg_name.startswith("--") else f"--{arg_name}"


def _cuda_visible_device(device: str) -> str:
    device = str(device or "").strip().lower()
    if device.startswith("cuda:"):
        device = device.split(":", 1)[1]
    return device if device.isdigit() else ""


def _rewrite_arg_value(argv: MutableSequence[str], option: str, value: str) -> None:
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            argv[index + 1] = value
            return
        if str(arg).startswith(f"{option}="):
            argv[index] = f"{option}={value}"
            return


def set_default_cuda_device_from_arg(arg_name: str, default_device: str = "cuda:0") -> bool:
    option = _arg_name_to_option(arg_name)
    if not option:
        return False
    argv = sys.argv
    for index, arg in enumerate(argv[1:], start=1):
        if arg == option and index + 1 < len(argv):
            visible_device = _cuda_visible_device(argv[index + 1])
            break
        if str(arg).startswith(f"{option}="):
            visible_device = _cuda_visible_device(str(arg).split("=", 1)[1])
            break
    else:
        return False

    if not visible_device:
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    _rewrite_arg_value(argv, option, default_device)
    return True
