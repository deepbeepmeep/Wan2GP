"""Windows power throttling opt-out for the WanGP process.

Windows may run the processes of a minimized or background application at a reduced CPU speed (EcoQoS power throttling).
WanGP's main thread feeds the GPU, so a throttled CPU slows down the steps where the GPU waits for it, sometimes several
times. The opt-out is a per-process setting: it changes no system configuration and ends with the process.
"""
from __future__ import annotations

import ctypes
import os

_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS ProcessPowerThrottling
_THROTTLING_VERSION = 1
_EXECUTION_SPEED = 0x1
_IGNORE_TIMER_RESOLUTION = 0x4


def prevent_power_throttling() -> bool:
    """Asks Windows not to throttle the CPU speed and timer resolution of this process; returns True if applied."""
    if os.name != "nt":
        return False
    from ctypes import wintypes

    class _ThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    state = _ThrottlingState(_THROTTLING_VERSION, _EXECUTION_SPEED | _IGNORE_TIMER_RESOLUTION, 0)  # controlled, and never throttled
    return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)))
