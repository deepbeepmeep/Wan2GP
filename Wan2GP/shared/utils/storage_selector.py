from __future__ import annotations

import sys

try:
    from .drive_detector import DriveInfo
except ImportError:
    from drive_detector import DriveInfo


_HEADER = "Select model storage location (\u2191/\u2193 to navigate, Enter to select, q to use default):"


def select_storage_drive(drives: list[DriveInfo]) -> str | None:
    if not sys.stdin.isatty() or not drives:
        return None

    if sys.platform == "win32":
        return _select_storage_drive_windows(drives)
    return _select_storage_drive_unix(drives)


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    value = float(max(num_bytes, 0))
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.1f} {units[unit_index]}"


def _select_storage_drive_windows(drives: list[DriveInfo]) -> str | None:
    try:
        return _selection_loop(drives, _read_windows_key)
    finally:
        _finish_render()


def _read_windows_key() -> str:
    import msvcrt

    char = msvcrt.getwch()
    if char in ("\x00", "\xe0"):
        second = msvcrt.getwch()
        return char + second
    return char


def _select_storage_drive_unix(drives: list[DriveInfo]) -> str | None:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return _selection_loop(drives, _read_unix_key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        _finish_render()


def _read_unix_key() -> str:
    char = sys.stdin.read(1)
    if char == "\x1b":
        next_char = sys.stdin.read(1)
        if next_char == "[":
            return char + next_char + sys.stdin.read(1)
        return char + next_char
    return char


def _selection_loop(drives: list[DriveInfo], read_key) -> str | None:
    selected_index = 0
    sys.stdout.write("\0337")  # save cursor position
    _render_selector(drives, selected_index)
    while True:
        key = read_key()
        if key in ("\x03", "q", "Q"):
            return None
        if key in ("\r", "\n"):
            return drives[selected_index].mount_point
        moved = False
        if key in ("\x1b[A", "\xe0H", "\x00H"):
            selected_index = (selected_index - 1) % len(drives)
            moved = True
        elif key in ("\x1b[B", "\xe0P", "\x00P"):
            selected_index = (selected_index + 1) % len(drives)
            moved = True
        if moved:
            sys.stdout.write("\0338")  # restore cursor position
            _render_selector(drives, selected_index)


def _render_selector(drives: list[DriveInfo], selected_index: int) -> None:
    sys.stdout.write("\033[J")  # clear from cursor down

    sys.stdout.write(_HEADER + "\r\n\r\n")
    for index, drive in enumerate(drives):
        free = _format_bytes(drive.free_bytes)
        total = _format_bytes(drive.total_bytes)
        line = f"{drive.mount_point}  ({drive.label})  {free} free / {total}"
        if index == selected_index:
            sys.stdout.write(f"  > \033[1;36m{line}\033[0m\r\n")
        else:
            sys.stdout.write(f"    {line}\r\n")
    sys.stdout.flush()


def _finish_render() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


__all__ = ["select_storage_drive"]
