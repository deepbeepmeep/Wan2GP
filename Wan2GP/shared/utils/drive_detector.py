from __future__ import annotations

import os
import sys
from collections import namedtuple


DriveInfo = namedtuple("DriveInfo", ["mount_point", "label", "total_bytes", "free_bytes"])


_LINUX_VIRTUAL_FS_TYPES = {
    "autofs",
    "binfmt_misc",
    "bpf",
    "cgroup",
    "cgroup2",
    "configfs",
    "debugfs",
    "devpts",
    "devtmpfs",
    "fuse.portal",
    "fusectl",
    "hugetlbfs",
    "mqueue",
    "nsfs",
    "overlay",
    "proc",
    "pstore",
    "ramfs",
    "securityfs",
    "selinuxfs",
    "squashfs",
    "sysfs",
    "tmpfs",
    "tracefs",
}
_LINUX_VIRTUAL_FS_PREFIXES = ("fuse.",)
_LINUX_VIRTUAL_MOUNT_PREFIXES = (
    "/dev",
    "/proc",
    "/snap",
    "/sys",
    "/var/lib/docker",
    "/var/lib/containers",
)


def detect_drives() -> list[DriveInfo]:
    if sys.platform == "darwin":
        drives = _detect_macos_drives()
    elif sys.platform == "win32":
        drives = _detect_windows_drives()
    else:
        drives = _detect_linux_drives()
    return sorted(drives, key=lambda drive: drive.mount_point)


def is_single_drive(drives: list[DriveInfo]) -> bool:
    return len(drives) <= 1


def suggest_checkpoint_path(mount_point: str, cwd: str) -> str:
    try:
        if os.stat(mount_point).st_dev == os.stat(cwd).st_dev:
            return "ckpts"
    except OSError:
        pass
    return os.path.join(mount_point, "Reigh", "ckpts")


def _detect_macos_drives() -> list[DriveInfo]:
    mount_points = ["/"]
    volumes_root = "/Volumes"
    if os.path.isdir(volumes_root):
        for entry in sorted(os.scandir(volumes_root), key=lambda item: item.path):
            if entry.is_dir(follow_symlinks=False):
                mount_points.append(entry.path)

    drives = []
    seen = set()
    for mount_point in mount_points:
        if mount_point in seen:
            continue
        seen.add(mount_point)
        label = "Macintosh HD" if mount_point == "/" else os.path.basename(mount_point.rstrip(os.sep))
        drive = _build_drive_info(mount_point, label)
        if drive is not None:
            drives.append(drive)
    return drives


def _detect_linux_drives() -> list[DriveInfo]:
    drives = []
    seen = set()

    for mount_point, label in _iter_linux_mount_candidates():
        if mount_point in seen:
            continue
        seen.add(mount_point)
        drive = _build_drive_info(mount_point, label)
        if drive is not None:
            drives.append(drive)
    return drives


def _iter_linux_mount_candidates():
    mounts_path = "/proc/mounts"
    if os.path.isfile(mounts_path):
        try:
            with open(mounts_path, "r", encoding="utf-8") as reader:
                for raw_line in reader:
                    fields = raw_line.split()
                    if len(fields) < 3:
                        continue
                    source = _decode_linux_mount_value(fields[0])
                    mount_point = _decode_linux_mount_value(fields[1])
                    fs_type = fields[2]
                    if _is_linux_virtual_fs(source, mount_point, fs_type):
                        continue
                    if not os.path.isdir(mount_point):
                        continue
                    yield mount_point, _linux_mount_label(mount_point, source)
            return
        except OSError:
            pass

    for mount_point in _iter_linux_fallback_mounts():
        yield mount_point, _linux_mount_label(mount_point, mount_point)


def _iter_linux_fallback_mounts():
    seen = {"/"}
    if os.path.ismount("/"):
        yield "/"

    for base_path in ("/mnt", "/media"):
        if not os.path.isdir(base_path):
            continue
        try:
            entries = [entry.path for entry in os.scandir(base_path) if entry.is_dir(follow_symlinks=False)]
        except OSError:
            continue

        candidates = list(entries)
        for entry_path in entries:
            try:
                candidates.extend(
                    child.path
                    for child in os.scandir(entry_path)
                    if child.is_dir(follow_symlinks=False)
                )
            except OSError:
                continue

        for candidate in candidates:
            if candidate in seen or not os.path.ismount(candidate):
                continue
            seen.add(candidate)
            yield candidate


def _decode_linux_mount_value(value: str) -> str:
    result = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 3 < len(value):
            octal = value[index + 1:index + 4]
            if octal.isdigit():
                try:
                    result.append(chr(int(octal, 8)))
                    index += 4
                    continue
                except ValueError:
                    pass
        result.append(value[index])
        index += 1
    return "".join(result)


def _is_linux_virtual_fs(source: str, mount_point: str, fs_type: str) -> bool:
    if fs_type in _LINUX_VIRTUAL_FS_TYPES:
        return True
    if any(fs_type.startswith(prefix) for prefix in _LINUX_VIRTUAL_FS_PREFIXES):
        return True
    if mount_point != "/" and any(
        mount_point == prefix or mount_point.startswith(prefix + "/")
        for prefix in _LINUX_VIRTUAL_MOUNT_PREFIXES
    ):
        return True
    if source.startswith("/dev/loop"):
        return True
    return False


def _linux_mount_label(mount_point: str, source: str) -> str:
    if mount_point == "/":
        try:
            return os.uname().nodename
        except AttributeError:
            return "/"

    base_name = os.path.basename(mount_point.rstrip(os.sep))
    if base_name:
        return base_name

    source_name = os.path.basename(source.rstrip(os.sep))
    return source_name or mount_point


def _detect_windows_drives() -> list[DriveInfo]:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    buffer_length = kernel32.GetLogicalDriveStringsW(0, None)
    if buffer_length <= 0:
        return []

    buffer = ctypes.create_unicode_buffer(buffer_length + 1)
    written = kernel32.GetLogicalDriveStringsW(buffer_length + 1, buffer)
    if written <= 0:
        return []

    mount_points = [value for value in buffer[:written].split("\x00") if value]
    drives = []
    for mount_point in mount_points:
        drive = _build_windows_drive_info(kernel32, mount_point)
        if drive is not None:
            drives.append(drive)
    return drives


def _build_windows_drive_info(kernel32, mount_point: str):
    import ctypes

    free_bytes = ctypes.c_ulonglong(0)
    total_bytes = ctypes.c_ulonglong(0)
    total_free_bytes = ctypes.c_ulonglong(0)
    if not kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(mount_point),
        ctypes.byref(free_bytes),
        ctypes.byref(total_bytes),
        ctypes.byref(total_free_bytes),
    ):
        return None

    label_buffer = ctypes.create_unicode_buffer(261)
    fs_name_buffer = ctypes.create_unicode_buffer(261)
    serial_number = ctypes.c_ulong(0)
    max_component_len = ctypes.c_ulong(0)
    flags = ctypes.c_ulong(0)
    label = mount_point.rstrip("\\")
    if kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(mount_point),
        label_buffer,
        len(label_buffer),
        ctypes.byref(serial_number),
        ctypes.byref(max_component_len),
        ctypes.byref(flags),
        fs_name_buffer,
        len(fs_name_buffer),
    ):
        label = label_buffer.value or label

    return DriveInfo(mount_point, label, total_bytes.value, free_bytes.value)


def _build_drive_info(mount_point: str, label: str):
    try:
        stat = os.statvfs(mount_point)
    except OSError:
        return None

    block_size = stat.f_frsize or stat.f_bsize
    total_bytes = stat.f_blocks * block_size
    free_bytes = stat.f_bavail * block_size
    return DriveInfo(mount_point, label, total_bytes, free_bytes)


__all__ = ["DriveInfo", "detect_drives", "is_single_drive", "suggest_checkpoint_path"]
