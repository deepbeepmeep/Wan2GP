# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""RAM debug mode: where the RAM of the process goes, including the memory that no Python object holds any more (Windows and Linux).

    from mmgp.allocator import ram_debug
    ram_debug.start(min_mb=16)                 # after importing torch
    ...                                        # run the workload
    ram_debug.snapshot("after generation")     # a full census now
    ram_debug.report("ram.json")               # JSON for an agent, and ram.md, a summary

A census walks the address space of the process and splits its private memory (Windows: committed; Linux: anonymous, resident or
swapped) and its resident memory into owners: PyTorch's CPU allocator (on Windows mimalloc arenas, with what its live tensors use and what
it keeps for reuse), the C heap (in use / free), MMGP pinned weights and staging ring, CUDA pinned memory (PyTorch's pin_memory cache),
CUDA driver buffers, Python objects, OpenBLAS thread buffers, thread stacks, library images and memory-mapped files. It then lists the
Python objects alive that hold RAM: CPU tensors, NumPy arrays, bytes, PIL images, the weights of the models MMGP manages, the reference
cycles the garbage collector frees, and for the largest objects the chains of references that keep them alive (module globals, thread
frames, library caches, or no Python referrer: held from C/C++). A sampler thread follows the peaks of each phase (one phase per model that
MMGP loads to the GPU, or mark("name")) and takes a light census near them.

With the library of the MMGP RAM allocator (ram.py, PyTorch 2.6 to 2.15), PyTorch's CPU allocator and NumPy's default allocator also go through pass-through hooks
that record the allocations of min_mb or more with their origin (module, tag, Python stack): the report lists them at the peak of each
phase, the totals of each origin, and the live allocations that no Python object references (held by C++ code or libraries).

Command line: python -m mmgp.allocator.ram_debug summary report.json | diff before.json after.json | scan PID (another process, from outside)"""
import atexit
import bisect
import collections
import ctypes
import gc
import json
import os
import platform
import re
import sys
import threading
import time
import types

import torch

from . import debug as _vram_debug
from . import ram as _ram

GB, MB, KB = 2 ** 30, 2 ** 20, 2 ** 10
PAGE = 4096
WINDOWS = sys.platform == "win32"
SMALL_BYTES = 64 * KB  # objects and hooked allocations below this are only totalled
CLASSES = {  # key: label
    "mmgp_ram": "MMGP RAM allocator (CPU tensors of 1 MiB and more)",
    "torch_cpu": "PyTorch CPU allocator (tensors)",
    "malloc": "C heap (malloc: NumPy, PIL, C libraries)",
    "pinned_mmgp": "MMGP pinned weights (locked RAM)",
    "pinned_ring": "MMGP staging ring (locked RAM)",
    "pinned_cuda": "CUDA pinned memory (PyTorch pin_memory and its cache)",
    "cuda_driver": "CUDA driver buffers",
    "python": "Python objects (pymalloc arenas)",
    "openblas": "OpenBLAS thread buffers (NumPy/SciPy), untouched",
    "stacks": "Thread stacks",
    "images": "Program and library images",
    "mapped_files": "Memory-mapped files",
    "shared": "Shared memory sections",
    "other": "Other private memory",
}
if WINDOWS:  # WDDM charges every VRAM allocation to the commit of the process, as write-combined memory that is never resident
    CLASSES["cuda_driver"] = "CUDA: commit charge of the VRAM (WDDM) and driver buffers"
else:
    CLASSES["malloc"] = "C heap (glibc malloc: PyTorch CPU tensors, NumPy, C libraries)"
    del CLASSES["torch_cpu"]


# ---------------------------------------------------------------------------------------------------------------------------- platform
if WINDOWS:
    import ctypes.wintypes as wt
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _MBI(ctypes.Structure):
        _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p), ("AllocationProtect", wt.DWORD), ("PartitionId", wt.WORD),
                    ("RegionSize", ctypes.c_size_t), ("State", wt.DWORD), ("Protect", wt.DWORD), ("Type", wt.DWORD)]

    class _WSX(ctypes.Structure):
        _fields_ = [("VirtualAddress", ctypes.c_void_p), ("VirtualAttributes", ctypes.c_size_t)]

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [(name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

    class _PERF(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
            "KernelNonpaged", "PageSize")] + [("HandleCount", wt.DWORD), ("ProcessCount", wt.DWORD), ("ThreadCount", wt.DWORD)]

    class _HEAP_SUMMARY(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("cbAllocated", ctypes.c_size_t), ("cbCommitted", ctypes.c_size_t), ("cbReserved", ctypes.c_size_t),
                    ("cbMaxReserve", ctypes.c_size_t)]

    _k32.OpenProcess.restype = wt.HANDLE
    _k32.GetCurrentProcess.restype = wt.HANDLE
    _k32.VirtualQueryEx.argtypes = (wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(_MBI), ctypes.c_size_t)
    _k32.K32GetMappedFileNameW.argtypes = (wt.HANDLE, ctypes.c_void_p, wt.LPWSTR, wt.DWORD)
    _k32.K32QueryWorkingSetEx.argtypes = (wt.HANDLE, ctypes.c_void_p, wt.DWORD)
    _k32.K32GetProcessMemoryInfo.argtypes = (wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD)
    _k32.K32GetPerformanceInfo.argtypes = (ctypes.POINTER(_PERF), wt.DWORD)
    _k32.GetProcessHeaps.argtypes = (wt.DWORD, ctypes.POINTER(wt.HANDLE))
    _k32.HeapSummary.argtypes = (wt.HANDLE, wt.DWORD, ctypes.POINTER(_HEAP_SUMMARY))
    _k32.CloseHandle.argtypes = (wt.HANDLE,)
    _MEM_COMMIT, _MEM_FREE, _MEM_PRIVATE, _MEM_MAPPED, _MEM_IMAGE = 0x1000, 0x10000, 0x20000, 0x40000, 0x1000000
    _PAGE_GUARD, _PAGE_WRITECOMBINE, _WRITECOPY = 0x100, 0x400, (0x08, 0x80)


def _open(pid):
    if pid is None:
        return _k32.GetCurrentProcess(), False
    handle = _k32.OpenProcess(0x0400 | 0x0010, False, pid)  # QUERY_INFORMATION | VM_READ (QueryWorkingSetEx); no memory content is read
    if not handle:
        raise OSError(f"Cannot open process {pid}: error {ctypes.get_last_error()}")
    return handle, True


def _status(pid):
    with open(f"/proc/{pid or 'self'}/status") as reader:
        return {key: int(value.split()[0]) * KB for key, value in (line.split(":", 1) for line in reader) if value.strip().endswith("kB")}


def counters(pid=None):
    """Bytes of the process: private (Windows: committed, Task Manager's commit size; Linux: anonymous memory, resident or swapped) and
    resident (working set / RSS), with their peaks (Linux: peak RSS only)."""
    if WINDOWS:
        handle, owned = _open(pid)
        try:
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)
            if not _k32.K32GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
        finally:
            if owned:
                _k32.CloseHandle(handle)
        return {"private": pmc.PrivateUsage, "resident": pmc.WorkingSetSize, "peak_private": pmc.PeakPagefileUsage, "peak_resident": pmc.PeakWorkingSetSize}
    status = _status(pid)
    private = status.get("RssAnon", 0) + status.get("VmSwap", 0)
    return {"private": private, "resident": status.get("VmRSS", 0), "peak_private": None, "peak_resident": status.get("VmHWM", 0)}


def system():
    """System-wide memory: physical RAM and the commit charge against its limit (Windows), or available memory and swap (Linux)."""
    if WINDOWS:
        perf = _PERF()
        perf.cb = ctypes.sizeof(perf)
        _k32.K32GetPerformanceInfo(ctypes.byref(perf), perf.cb)
        page = perf.PageSize
        return {"physical_gb": round(perf.PhysicalTotal * page / GB, 2), "available_gb": round(perf.PhysicalAvailable * page / GB, 2),
                "commit_gb": round(perf.CommitTotal * page / GB, 2), "commit_limit_gb": round(perf.CommitLimit * page / GB, 2)}
    with open("/proc/meminfo") as reader:
        info = {key: int(value.split()[0]) * KB for key, value in (line.split(":", 1) for line in reader)}
    return {"physical_gb": round(info["MemTotal"] / GB, 2), "available_gb": round(info["MemAvailable"] / GB, 2),
            "swap_used_gb": round((info["SwapTotal"] - info["SwapFree"]) / GB, 2), "commit_gb": round(info["Committed_AS"] / GB, 2),
            "commit_limit_gb": round(info["CommitLimit"] / GB, 2)}


def _residency(handle, runs, max_samples=128):
    # resident bytes of the committed runs, from up to max_samples pages of each (QueryWorkingSetEx)
    total = 0
    for base, size in runs:
        pages = max(1, size // PAGE)
        stride = max(1, pages // max_samples)
        indices = range(0, pages, stride)
        query = (_WSX * len(indices))()
        for slot, page in enumerate(indices):
            query[slot].VirtualAddress = base + page * PAGE
        if _k32.K32QueryWorkingSetEx(handle, query, ctypes.sizeof(query)):
            total += size * sum(entry.VirtualAttributes & 1 for entry in query) // len(indices)
    return total


def _windows_reservations(pid, residency):
    handle, owned = _open(pid)
    try:
        regions, address, info = collections.OrderedDict(), 0, _MBI()
        while address < 0x7FFFFFFFFFFF and _k32.VirtualQueryEx(handle, address, ctypes.byref(info), ctypes.sizeof(info)):
            base, size = info.BaseAddress or 0, info.RegionSize
            if info.State != _MEM_FREE:
                group = regions.get(info.AllocationBase or 0)
                if group is None:
                    group = regions[info.AllocationBase or 0] = {"base": info.AllocationBase or 0, "reserved": 0, "committed": 0, "cow": 0, "type": info.Type,
                                                                "runs": [], "guard": False, "wc": False, "protect": collections.Counter()}
                group["reserved"] += size
                if info.State == _MEM_COMMIT:
                    group["committed"] += size
                    group["runs"].append((base, size))
                    group["protect"][info.Protect & 0xFF] += size
                    group["guard"] |= bool(info.Protect & _PAGE_GUARD)
                    group["wc"] |= bool(info.Protect & _PAGE_WRITECOMBINE)
                    if info.Protect & 0xFF in _WRITECOPY:
                        group["cow"] += size
            address = base + size
        name = ctypes.create_unicode_buffer(1024)
        result = []
        for group in regions.values():
            if not group["committed"]:
                continue
            kind = {_MEM_PRIVATE: "private", _MEM_MAPPED: "mapped", _MEM_IMAGE: "image"}.get(group["type"], "private")
            path = ""
            if kind != "private":
                path = name.value if _k32.K32GetMappedFileNameW(handle, group["base"], name, 1024) else ""
                if kind == "mapped" and not path:
                    kind = "shared"  # a section backed by the paging file
            group.update(kind=kind, path=path, resident=_residency(handle, group["runs"]) if residency else 0)
            group["private"] = group["committed"] if kind == "private" else group["cow"]  # the commit charge of a view is its copy-on-write part
            del group["type"]
            result.append(group)
        return result
    finally:
        if owned:
            _k32.CloseHandle(handle)


_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \d+\s*(.*)$")


def _linux_reservations(pid, residency=True):
    result, current = [], None
    with open(f"/proc/{pid or 'self'}/smaps") as reader:
        for line in reader:
            match = _SMAPS_HEADER.match(line)
            if match:
                start, end = int(match.group(1), 16), int(match.group(2), 16)
                path = match.group(4).strip()
                current = {"base": start, "reserved": end - start, "perms": match.group(3), "path": path, "runs": [(start, end - start)], "guard": False, "wc": False,
                           "fields": {}}
                result.append(current)
            elif current is not None and ":" in line:
                key, value = line.split(":", 1)
                if value.strip().endswith("kB"):
                    current["fields"][key] = int(value.split()[0]) * KB
    reservations = []
    for item in result:
        fields, path, perms = item.pop("fields"), item["path"], item.pop("perms")
        anonymous = fields.get("Anonymous", 0) + fields.get("Swap", 0)
        if path and not path.startswith("[") and not path.startswith("/dev/") and not path.startswith("/memfd:") and "SYSV" not in path and not path.startswith("/dev/shm"):
            kind = "image" if (".so" in os.path.basename(path) or "x" in perms) else "mapped"
            private = fields.get("Private_Dirty", 0) if "w" in perms else 0
        elif path.startswith("/dev/shm") or path.startswith("/memfd:") or "SYSV" in path or "(deleted)" in path and path.startswith("/"):
            kind, private = "shared", 0
        elif path.startswith("/dev/"):
            kind, private = "driver", anonymous
        elif path in ("[vvar]", "[vdso]", "[vsyscall]"):
            kind, private = "image", 0
        else:
            kind, private = "private", anonymous
        if perms.startswith("---"):
            if reservations and reservations[-1]["base"] + reservations[-1]["reserved"] == item["base"]:
                reservations[-1]["reserved"] += item["reserved"]  # the inaccessible tail of a glibc arena
            continue
        resident = fields.get("Rss", 0)
        if not (resident or private):
            continue
        item.update(kind=kind, private=private, committed=private if kind == "private" else resident, resident=resident, cow=private if kind != "private" else 0,
                    stack=path.startswith("[stack"), heap=path == "[heap]", locked=fields.get("Locked", 0), protect=collections.Counter())
        reservations.append(item)
    return reservations


def reservations(pid=None, residency=True):
    """The committed allocations of the address space (Windows: by reservation; Linux: by mapping), sorted by address. Each: base, reserved,
    committed, private (its share of the process's private bytes), resident, kind (private, mapped, shared, image, driver), path, runs."""
    found = _windows_reservations(pid, residency) if WINDOWS else _linux_reservations(pid, residency)
    found.sort(key=lambda item: item["base"])
    return found


def _heap_summary():
    # the C heaps of this process (Windows): committed and allocated bytes, from each heap's summary (fast, no walk)
    handles = (wt.HANDLE * 1024)()
    count = min(1024, _k32.GetProcessHeaps(1024, handles))
    committed = allocated = 0
    for handle in handles[:count]:
        summary = _HEAP_SUMMARY()
        summary.cb = ctypes.sizeof(summary)
        if _k32.HeapSummary(handle, 0, ctypes.byref(summary)):
            committed += summary.cbCommitted
            allocated += summary.cbAllocated
    return {"heaps": count, "committed": committed, "in_use": allocated}


class _MallInfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def _mallinfo():
    # glibc's arenas (all of them) and its separately mapped large blocks
    libc = ctypes.CDLL(None)
    if not hasattr(libc, "mallinfo2"):
        return None
    libc.mallinfo2.restype = _MallInfo2
    info = libc.mallinfo2()
    return {"committed": info.arena + info.hblkhd, "in_use": info.uordblks + info.hblkhd, "free": info.fordblks, "mapped_blocks": info.hblkhd}


# ---------------------------------------------------------------------------------------------------------------------------- CUDA
_cuda = None


def _cuda_host_bases(candidates):
    """The bases among candidates (addresses) that CUDA knows as host memory: pinned by cudaHostAlloc or registered (cudaHostRegister)."""
    global _cuda
    if not candidates or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return set()
    if _cuda is None:
        _cuda = ctypes.WinDLL("nvcuda.dll") if WINDOWS else ctypes.CDLL("libcuda.so.1")
        _cuda.cuPointerGetAttribute.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64)
    value, found = ctypes.c_uint(0), set()
    for base in candidates:
        if _cuda.cuPointerGetAttribute(ctypes.byref(value), 2, base) == 0 and value.value == 1:  # CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_MEMORYTYPE_HOST
            found.add(base)
    return found


def _torch_pinned_cache():
    try:
        stats = torch.cuda.host_memory_stats() if torch.cuda.is_available() and torch.cuda.is_initialized() else {}
    except RuntimeError:
        stats = {}
    return stats.get("allocated_bytes.current", 0)


def _vmm_stats():
    from . import active, stats
    if active != "vmm":
        return None
    values = stats()
    return {"spilled_gb": round(values["spilled"] / GB, 3), "driver_spilled_gb": round(values["driver_spilled"] / GB, 3)}


# ---------------------------------------------------------------------------------------------------------------------------- classification
def _arena_like(item):
    # mimalloc (PyTorch's CPU allocator on Windows) reserves arenas of 1 GiB and more, aligned on its 32 MiB blocks
    return WINDOWS and item["reserved"] >= 512 * MB and item["base"] % (32 * MB) == 0 and item["reserved"] % (32 * MB) == 0


def classify(found, evidence):
    """Assigns a class to each reservation (item["cls"]) and returns {class: totals}. evidence: torch_blocks (32 MiB granules that served
    PyTorch CPU allocations), registrations [(pointer, size, class)], torch_ptrs and array_ptrs (sorted addresses of live buffers),
    cuda_host (bases known to CUDA), numpy (imported)."""
    registrations = sorted(evidence.get("registrations", ()))
    starts = [entry[0] for entry in registrations]
    torch_ptrs, array_ptrs = evidence.get("torch_ptrs", []), evidence.get("array_ptrs", [])
    blocks, cuda_host = evidence.get("torch_blocks", set()), evidence.get("cuda_host", set())

    def contains(sorted_ptrs, item):
        index = bisect.bisect_left(sorted_ptrs, item["base"])
        return index < len(sorted_ptrs) and sorted_ptrs[index] < item["base"] + item["reserved"]

    def registered(item):  # (class, bytes) of the registrations inside the reservation
        index, end, cls, overlap = bisect.bisect_left(starts, item["base"] + item["reserved"]) - 1, item["base"] + item["reserved"], None, 0
        while index >= 0 and registrations[index][0] + registrations[index][1] > item["base"]:
            cls = registrations[index][2]
            overlap += min(end, registrations[index][0] + registrations[index][1]) - max(item["base"], registrations[index][0])
            index -= 1
        return cls, overlap

    ram_ranges = evidence.get("ram_ranges", [])

    def in_ranges(item):
        return any(base <= item["base"] < base + size for base, size in ram_ranges)

    def torch_owned(item):
        if blocks and _arena_like(item) and any(block in blocks for block in range(item["base"] >> 25, (item["base"] + item["reserved"] - 1 >> 25) + 1)):
            return True  # an arena that served hooked PyTorch allocations
        return contains(torch_ptrs, item) and not contains(array_ptrs, item) or not blocks and _arena_like(item)

    for item in found:
        kind = item["kind"]
        if kind == "image":
            cls = "images"
        elif kind == "mapped":
            cls = "mapped_files"
        elif kind == "shared":
            cls = "pinned_cuda" if item["base"] in cuda_host else "shared"
        elif item["wc"]:
            cls = "cuda_driver"  # write-combined: the driver's own staging buffers
        elif item.get("heap"):
            cls = "malloc"
        elif item.get("stack") or item["guard"]:
            cls = "stacks"
        else:
            cls, locked = registered(item)
            if cls is not None and locked < 0.9 * item["committed"]:  # a registration inside a larger reservation: split below
                item["registered_part"], cls = (cls, locked), None
            if cls is None and item["base"] in cuda_host:
                cls = "pinned_cuda"
            if cls is None and kind == "driver":
                cls = "cuda_driver"
            if cls is None and (in_ranges(item) if ram_ranges else evidence.get("external") and item["reserved"] % (64 * GB) == 0):
                cls = "mmgp_ram"  # from outside: the MMGP RAM allocator reserves its address space by 64 GiB
            if cls is None and WINDOWS and torch_owned(item):
                cls = "torch_cpu"
            if cls is None and WINDOWS and contains(array_ptrs, item):
                cls = "malloc"  # a large block of the C heap (NumPy data)
            if cls is None and evidence.get("numpy") and item["reserved"] == item["committed"] == 32 * MB and item["resident"] < MB:
                cls = "openblas"
            if cls is None and item["reserved"] == item["committed"] == MB:
                cls = "python"
            if cls is None:
                cls = "malloc" if not WINDOWS else "other"  # Windows: the heaps' share is taken from "other" with their summaries
        item["cls"] = cls
    totals = {key: {"private": 0, "resident": 0, "count": 0} for key in CLASSES}  # shared mappings have no private bytes (Linux)
    for item in found:
        private, resident = item["private"], item["resident"]
        if "registered_part" in item:
            cls, locked = item["registered_part"]
            totals[cls]["private"] += locked
            totals[cls]["resident"] += min(locked, resident)
            totals[cls]["count"] += 1
            private, resident = private - locked, max(0, resident - locked)
        total = totals[item["cls"]]
        total["private"] += private
        total["resident"] += resident
        total["count"] += 1
    return totals


# ---------------------------------------------------------------------------------------------------------------------------- live objects
def _numpy():
    return sys.modules.get("numpy")


def _array_owner(array, ndarray):
    while isinstance(array.base, ndarray):
        array = array.base
    return array


def _describe_tensor(tensor):
    return f"tensor {tuple(tensor.shape)} {str(tensor.dtype).replace('torch.', '')}"


def _subclasses(base):
    found, pending = {base}, [base]
    while pending:
        for subclass in type.__subclasses__(pending.pop()):
            if subclass not in found:
                found.add(subclass)
                pending.append(subclass)
    return found


def _find_objects():
    """Python objects that hold RAM: CPU tensors and storages, NumPy arrays, bytes, PIL images, MMGP registrations and offload objects.
    Objects are recognized by their type only: isinstance would run __class__ properties (lazy modules import on access)."""
    numpy = _numpy()
    pil, offload = sys.modules.get("PIL.Image"), sys.modules.get("mmgp.offload")  # looked up, never imported by a census
    slots = [("tensors", torch.Tensor), ("storages", torch.UntypedStorage), ("storages", torch.TypedStorage)]
    slots += [("registrations", offload._HostRegistration), ("offloads", offload.offload)] if offload is not None else []
    slots += [("images", pil.Image)] if pil is not None else []
    found = {"tensors": [], "storages": [], "arrays": {}, "bytes": {}, "images": [], "registrations": [], "offloads": []}
    slot_of = found["types"] = {}  # type -> slot name, or "" when the type holds nothing of interest

    def slot(kind):
        for name, base in slots:
            if issubclass(kind, base):
                return name
        return ""
    objects = gc.get_objects()
    try:
        for obj in objects:
            kind = type(obj)
            name = slot_of.get(kind)
            if name is None:
                name = slot_of[kind] = slot(kind)
            if name:
                found[name].append(obj)
        # arrays and bytes are not tracked by the garbage collector: they are found among the references of the objects it tracks,
        # or in the untracked tuples among them (a tuple of untracked objects is untracked)
        array_types = _subclasses(numpy.ndarray) if numpy is not None else set()
        wanted = array_types | {bytes, bytearray, tuple}
        arrays, blobs = found["arrays"], found["bytes"]
        for start in range(0, len(objects), 20000):
            refs = gc.get_referents(*objects[start:start + 20000])
            for ref in [ref for ref in refs if type(ref) in wanted]:
                kind = type(ref)
                if kind is tuple:
                    if not gc.is_tracked(ref):
                        for inner in ref:
                            if type(inner) in array_types:
                                arrays[id(inner)] = inner
                elif kind in array_types:
                    arrays[id(ref)] = ref
                elif len(ref) >= SMALL_BYTES:
                    blobs[id(ref)] = ref
            del refs
    finally:
        del objects
    return found


def _model_names(offloads):
    names = {}
    for offloader in offloads:
        for model_id, model in (offloader.models or {}).items():
            if isinstance(model, torch.nn.Module):
                for path, module in model.named_modules():  # memoized: module graphs with cycles (a child refers to its parent) are walked once
                    for name, tensor in list(module._parameters.items()) + list(module._buffers.items()):
                        pending = [(tensor, f"{path}.{name}" if path else name)]
                        while pending:  # the tensor, and the inner tensors of a wrapper subclass (quantized weights)
                            tensor, tensor_path = pending.pop()
                            if isinstance(tensor, torch.Tensor) and id(tensor) not in names:
                                names[id(tensor)] = (model_id, tensor_path)
                                if hasattr(type(tensor), "__tensor_flatten__"):
                                    pending += [(getattr(tensor, attribute, None), f"{tensor_path}.{attribute}") for attribute in tensor.__tensor_flatten__()[0]]
    return names


def _buffers(found):
    """Live buffers by address: {pointer: {nbytes, kind, what, objects, model}}; storages shared by several tensors are counted once."""
    buffers, totals = {}, collections.Counter()
    numpy = _numpy()
    names = _model_names(found["offloads"])

    def add(pointer, nbytes, kind, what, obj, model=None):
        if not pointer or nbytes <= 0:
            return
        entry = buffers.get(pointer)
        if entry is None:
            entry = buffers[pointer] = {"nbytes": nbytes, "kind": kind, "what": what, "objects": [], "models": collections.Counter(), "array": False}
        elif nbytes > entry["nbytes"]:
            entry.update(nbytes=nbytes, what=what)
        entry["array"] |= kind.startswith("array")
        if len(entry["objects"]) < 8:
            entry["objects"].append(obj)
        if model is not None:
            entry["models"][model] += 1

    for tensor in found["tensors"]:
        try:
            if tensor.device.type != "cpu" or tensor.is_sparse:
                continue
            storage = tensor.untyped_storage()
            pointer, nbytes = storage.data_ptr(), storage.nbytes()
        except (RuntimeError, NotImplementedError, TypeError, AttributeError):
            continue  # a wrapper subclass (quantized weights): its inner tensors are counted themselves
        name = names.get(id(tensor))
        add(pointer, nbytes, "tensor", _describe_tensor(tensor), tensor, name[0] if name else None)
    for storage in found["storages"]:
        try:
            untyped = storage.untyped() if isinstance(storage, torch.TypedStorage) else storage
            if untyped.device.type == "cpu":
                add(untyped.data_ptr(), untyped.nbytes(), "storage", f"storage {untyped.nbytes()} bytes", storage)
        except (RuntimeError, NotImplementedError, TypeError):
            pass
    if numpy is not None:
        for array in found["arrays"].values():
            owner = _array_owner(array, numpy.ndarray)
            try:
                pointer = owner.__array_interface__["data"][0]
            except (AttributeError, TypeError, KeyError):
                continue
            kind = "array" if owner.base is None or isinstance(owner.base, numpy.ndarray) else f"array on {type(owner.base).__name__}"
            add(pointer, owner.nbytes, kind, f"array {owner.shape} {owner.dtype}" + (f" ({type(owner).__name__})" if type(owner) is not numpy.ndarray else ""), owner)
    for blob in found["bytes"].values():
        add(id(blob), len(blob), type(blob).__name__, f"{type(blob).__name__} {len(blob)}", blob)
    host_start, host_end = 0, 0  # buffers inside an array's memory belong to it (tensors made from an array, views of an aligned slice)
    for pointer in sorted(buffers):
        entry = buffers[pointer]
        if pointer < host_end:
            host = buffers[host_start]
            host["objects"] = (host["objects"] + entry["objects"])[:8]
            host["models"].update(entry["models"])
            del buffers[pointer]
        elif entry["array"]:
            host_start, host_end = pointer, pointer + entry["nbytes"]
    images = []
    for image in found["images"]:
        if image.__dict__.get("im") is not None:
            width, height = image.size
            images.append((width * height * max(1, len(image.getbands())), image))
    for buffer in buffers.values():
        totals[buffer["kind"].split(" ")[0]] += buffer["nbytes"]
    return buffers, images, totals


# ---------------------------------------------------------------------------------------------------------------------------- holders
def _short(value, limit=60):
    # repr only for plain values: a key can be any object (a module's repr walks the whole model and can recurse without end)
    text = repr(value) if value is None or isinstance(value, (str, bytes, int, float, bool)) else f"<{type(value).__name__} at {id(value):#x}>"
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _edge(parent, child):
    """How parent refers to child, for a reference path."""
    if isinstance(parent, dict):
        for key, value in parent.items():
            if value is child:
                return f"[{_short(key)}]"
        return "[?]"
    if isinstance(parent, (list, tuple, collections.deque)):
        for index, value in enumerate(parent):
            if value is child:
                return f"[{index}]"
        return "[?]"
    if isinstance(parent, (set, frozenset)):
        return "{...}"
    if isinstance(parent, types.FrameType):
        try:
            for key, value in parent.f_locals.items():
                if value is child:
                    return f" local {key}"
        except Exception:
            pass
        return " (frame)"
    if isinstance(parent, types.CellType):
        return " (closure cell)"
    if isinstance(parent, types.FunctionType):
        for name in ("__closure__", "__defaults__", "__kwdefaults__", "__globals__", "__dict__"):
            if getattr(parent, name, None) is child:
                return f".{name}"
    try:  # attributes, read without running the class's __getattribute__ (Python 3.11+ keeps them inline: the object refers to the value)
        attributes = object.__getattribute__(parent, "__dict__")
    except Exception:
        attributes = None
    if attributes is child:
        return ".__dict__"
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            if value is child:
                return f".{key}"
    for kind in type(parent).__mro__:
        for key, slot in vars(kind).items():
            if isinstance(slot, types.MemberDescriptorType):
                try:
                    if slot.__get__(parent) is child:
                        return f".{key}"
                except Exception:
                    pass
    return ""


def _object_text(obj):
    if isinstance(obj, types.ModuleType):
        return f"module {obj.__name__}"
    if isinstance(obj, type):
        return f"class {obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, types.FunctionType):
        return f"function {obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, types.FrameType):
        return f"frame {obj.f_code.co_name} ({_vram_debug._frame_text(obj.f_code, obj.f_lineno)})"
    return type(obj).__name__


def _holder_paths(targets, budget_s, max_depth=14, level_cap=1000, paths_per_target=2):
    """For each key of targets ({key: [objects]}), up to paths_per_target chains of references from a root (module global, running frame,
    class attribute) to one of its objects, found by walking the referrers backwards. Returns ({key: [text]}, complete)."""
    deadline = time.perf_counter() + budget_s
    module_dicts = {id(module.__dict__): module for module in list(sys.modules.values()) if isinstance(module, types.ModuleType)}
    own_code = {code for code in (_holder_paths.__code__, _census_objects.__code__, census.__code__, snapshot.__code__, report.__code__)}
    keep, parents, roots = {}, {}, {}  # id -> object; id(child) -> [id(parent)]; id(root) -> text
    level = []
    own = {id(keep), id(parents), id(roots), id(level), id(targets)}
    running, local_roots = {}, {}  # id(frame) -> thread name; id(value of a local variable of a running frame) -> text
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    for thread_id, frame in sys._current_frames().items():
        while frame is not None:
            running[id(frame)] = names.get(thread_id, str(thread_id))
            if frame.f_code not in own_code and frame.f_code.co_filename != __file__ and frame.f_locals is not frame.f_globals:  # module code: found by its module
                try:
                    frame_locals = frame.f_locals
                except Exception:
                    frame_locals = {}
                own.add(id(frame_locals))  # the snapshot of the locals, a dict that refers to them
                for key, value in frame_locals.items():
                    local_roots.setdefault(id(value), f"thread {running[id(frame)]}: {_object_text(frame)} local {key}")
            frame = frame.f_back
    for objects in targets.values():
        own.add(id(objects))
        for obj in objects:
            if id(obj) not in keep:
                keep[id(obj)] = obj
                if id(obj) in local_roots:
                    roots[id(obj)] = local_roots[id(obj)]
                level.append(obj)
    complete, explored = True, set()

    def rooted(objects, need):  # number of distinct chains (up to need) from one of objects to a root found so far
        found, stack, seen = 0, [id(obj) for obj in objects], set()
        while stack and found < need:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in roots:
                found += 1
            else:
                stack += parents.get(node, ())
        return found
    for _ in range(max_depth):
        if not level or all(rooted(objects, paths_per_target) >= paths_per_target for objects in targets.values()):
            break
        next_level = []
        own.add(id(next_level))
        for start in range(0, len(level), 48):
            if time.perf_counter() > deadline:
                complete = False
                break
            chunk = level[start:start + 48]
            own.add(id(chunk))
            chunk_ids = {id(obj) for obj in chunk}
            explored |= chunk_ids
            referrers = gc.get_referrers(*chunk)
            own.add(id(referrers))
            for parent in referrers:
                parent_id = id(parent)
                if parent_id in own or (type(parent) is tuple and len(parent) == len(chunk) and parent[0] is chunk[0]):
                    continue
                if isinstance(parent, types.FrameType) and (parent.f_code in own_code or parent_id not in running and parent.f_code.co_filename == __file__):
                    continue
                linked = [id(child) for child in gc.get_referents(parent) if id(child) in chunk_ids]
                for child_id in linked:
                    edges = parents.setdefault(child_id, [])
                    if len(edges) < 3 and parent_id not in edges:
                        edges.append(parent_id)
                if not linked or parent_id in keep:
                    continue
                keep[parent_id] = parent
                if parent_id in local_roots:
                    roots[parent_id] = local_roots[parent_id]
                elif parent_id in module_dicts:
                    roots[parent_id] = f"module {module_dicts[parent_id].__name__}"
                elif isinstance(parent, types.FrameType) and parent_id in running:
                    roots[parent_id] = f"thread {running[parent_id]}: {_object_text(parent)}"
                elif isinstance(parent, type):
                    roots[parent_id] = _object_text(parent)
                elif len(next_level) < level_cap:
                    next_level.append(parent)
            del referrers, chunk
        level = next_level
        if not complete:
            break
    own.add(id(level))

    def render(path):  # path: [object ids], root first
        top = keep[path[0]]
        parts = [roots.get(path[0]) or (f"{_object_text(top)} (no further Python referrer: held from C/C++ code or an untracked object)" if path[0] in explored
                                        else f"{_object_text(top)} (search stopped here)")]
        index = 1
        while index < len(path):
            parent, child = keep[path[index - 1]], keep[path[index]]
            label = _edge(parent, child)
            if path[index - 1] in module_dicts and label.startswith("['"):
                label = "." + label[2:-2]
            if label == ".__dict__" and index + 1 < len(path):  # obj -> obj.__dict__ -> value reads as obj.attribute
                key = _edge(child, keep[path[index + 1]])
                parts.append(f".{key[2:-2]}" if key.startswith("['") else key)
                index += 2
                continue
            parts.append(label or f" -> {_object_text(child)}")
            index += 1
        return "".join(parts)

    result = {}
    for key, objects in targets.items():
        rooted_texts, other_texts = [], []  # chains that reach a root come first; chains ending without a referrer or unexplored after
        for obj in objects:
            stack, seen = [(id(obj), [id(obj)])], set()
            while stack and len(rooted_texts) < paths_per_target:
                node, path = stack.pop()
                if node in seen or len(path) > max_depth + 1:
                    continue
                seen.add(node)
                above = parents.get(node)
                if not above or node in roots:
                    if node in roots and len(path) == 1:
                        text = f"{roots[node]} -> {_object_text(obj)}"
                    elif not above and len(path) == 1:
                        text = f"{_object_text(obj)}: no Python referrer (held from C/C++ code, an untracked object, or only by this view's base)" if node in explored else f"{_object_text(obj)} (search stopped)"
                    else:
                        text = render(path[::-1]) + f" -> {_object_text(obj)}"
                    texts = rooted_texts if node in roots else other_texts
                    if text not in texts:
                        texts.append(text)
                    continue
                for parent_id in above:
                    stack.append((parent_id, path + [parent_id]))
            if len(rooted_texts) >= paths_per_target:
                break
        result[key] = (rooted_texts + other_texts)[:paths_per_target]
    keep.clear()
    return result, complete


# ---------------------------------------------------------------------------------------------------------------------------- census
def _evidence(buffers=None, registrations=None, hook_blocks=None):
    numpy = _numpy()
    evidence = {"numpy": numpy is not None, "registrations": registrations or [], "torch_blocks": hook_blocks or set()}
    if buffers is not None:
        evidence["torch_ptrs"] = sorted(pointer for pointer, buffer in buffers.items() if buffer["kind"] in ("tensor", "storage") and not buffer["array"])
        evidence["array_ptrs"] = sorted(pointer for pointer, buffer in buffers.items() if buffer["array"])
    return evidence


def _registrations(found):
    result = []
    for registration in found["registrations"]:
        if getattr(registration, "registered", False):
            result.append((registration.pointer, registration.size, "pinned_ring" if isinstance(registration.buffer, torch.Tensor) else "pinned_mmgp"))
    return result


def _class_table(found_regions, totals, extra):
    """The classes with their private/resident bytes, what is in use and what their allocator keeps free."""
    rows = []
    heap = extra.get("heap")
    mallinfo = extra.get("mallinfo")
    if mallinfo is not None:  # Linux: the anonymous memory is in "malloc"; what glibc did not allocate moves to "other"
        malloc = totals["malloc"]
        in_heap = min(malloc["private"], mallinfo["committed"])
        resident = int(malloc["resident"] * in_heap / malloc["private"]) if malloc["private"] else 0
        totals["other"] = {"private": malloc["private"] - in_heap, "resident": malloc["resident"] - resident, "count": 0}
        totals["malloc"] = {"private": in_heap, "resident": resident, "count": malloc["count"]}
    if WINDOWS and heap is not None:  # the rest of the heaps is in "other": their committed bytes (less the registered blocks and the large blocks
        # already found) move to "malloc", with a proportional share of the resident bytes
        malloc, other = totals["malloc"], totals["other"]
        in_heap = min(other["private"], max(0, heap["committed"] - extra.get("registered_numpy", 0) - malloc["private"]))
        resident = int(other["resident"] * in_heap / other["private"]) if other["private"] else 0
        totals["malloc"] = {"private": malloc["private"] + in_heap, "resident": malloc["resident"] + resident, "count": heap["heaps"]}
        totals["other"] = {"private": other["private"] - in_heap, "resident": other["resident"] - resident, "count": other["count"]}
    for key, label in CLASSES.items():
        total = totals.get(key)
        if not total or max(total["private"], total["resident"]) < MB:
            continue
        row = {"key": key, "label": label, "private_gb": round(total["private"] / GB, 3), "resident_gb": round(total["resident"] / GB, 3), "count": total["count"]}
        in_use = extra.get("in_use", {}).get(key)
        if in_use is not None:
            row["in_use_gb"] = round(in_use / GB, 3)
            row["free_gb"] = round(max(0, total["private"] - in_use) / GB, 3)
        note = extra.get("notes", {}).get(key)
        if note:
            row["note"] = note
        rows.append(row)
    return rows


def _largest(found_regions, top=25):
    rows = []
    for item in sorted(found_regions, key=lambda item: -max(item["private"], item["resident"]))[:top]:
        rows.append({"address": hex(item["base"]), "class": item["cls"], "private_mb": round(item["private"] / MB, 1), "resident_mb": round(item["resident"] / MB, 1),
                     "reserved_mb": round(item["reserved"] / MB, 1), "path": os.path.basename(item["path"]) if item.get("path") else ""})
    return rows


def _files(found_regions, top=20):
    files = collections.defaultdict(lambda: [0, 0, ""])
    for item in found_regions:
        if item["cls"] in ("mapped_files", "images") and item.get("path"):
            entry = files[item["path"]]
            entry[0] += item["resident"]
            entry[1] += item["private"]
            entry[2] = "library" if item["cls"] == "images" else "data"
    return [{"file": path, "kind": kind, "resident_gb": round(resident / GB, 3), "private_gb": round(private / GB, 3)}
            for path, (resident, private, kind) in sorted(files.items(), key=lambda entry: -(entry[1][0] + entry[1][1]))[:top] if resident or private]


def _garbage_chain(target, garbage_ids, max_depth=10):
    """The objects of the cyclic garbage that refer to target, outermost first: the reference cycle that kept it until a collection."""
    path, seen = [target], {id(target)}
    while len(path) <= max_depth:
        parents = [parent for parent in gc.get_referrers(path[-1]) if id(parent) in garbage_ids and id(parent) not in seen]
        if not parents:
            break
        path.append(min(parents, key=lambda obj: 0 if isinstance(obj, types.FrameType) else 2 if isinstance(obj, (dict, list, tuple, types.CellType)) else 1))
        seen.add(id(path[-1]))
    closer = next((parent for parent in gc.get_referrers(path[-1]) if id(parent) in seen), None)  # the reference that closes the cycle
    parts = [_object_text(path[-1])]
    for parent, child in zip(path[:0:-1], path[-2::-1]):
        label = _edge(parent, child)
        parts.append(f"{label} ({_object_text(child)})" if label and isinstance(child, types.FrameType) else label or f" -> {_object_text(child)}")
    if closer is not None:
        parts.append(f"; cycle closed by {_object_text(closer)}{_edge(closer, path[-1]) or ''} -> {_object_text(path[-1])}")
    return "".join(parts)


def _cyclic_garbage(min_bytes, top=5):
    """Collects the reference cycles now, after measuring the large buffers they held (memory waiting for the garbage collector) and
    the chain of references that kept the largest ones."""
    flags = gc.get_debug()
    gc.set_debug(flags | gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        garbage = list(gc.garbage)
        gc.garbage.clear()
    finally:
        gc.set_debug(flags)
    held, numpy, seen, large = 0, _numpy(), set(), []
    for obj in garbage:
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
                storage = obj.untyped_storage()
                if storage.data_ptr() not in seen:
                    seen.add(storage.data_ptr())
                    held += storage.nbytes()
                    if storage.nbytes() >= min_bytes:
                        large.append((storage.nbytes(), _describe_tensor(obj), obj))
            elif numpy is not None and isinstance(obj, numpy.ndarray) and obj.base is None and obj.nbytes >= min_bytes:
                held += obj.nbytes
                large.append((obj.nbytes, f"ndarray {obj.shape} {obj.dtype}", obj))
        except (RuntimeError, NotImplementedError, TypeError, AttributeError):
            pass
    garbage_ids = {id(obj) for obj in garbage}
    cycles = [{"mb": round(nbytes / MB, 1), "what": what, "chain": _garbage_chain(obj, garbage_ids)} for nbytes, what, obj in sorted(large, key=lambda item: -item[0])[:top]]
    count = len(garbage)
    del garbage, large
    gc.collect()
    return count, held, cycles


def _census_objects(min_bytes, holders, budget_s):
    started = time.perf_counter()
    garbage = _cyclic_garbage(min_bytes)
    found = _find_objects()
    buffers, images, totals = _buffers(found)
    registrations = _registrations(found)
    found_time = time.perf_counter() - started
    found_types = list(found["types"])
    del found
    top = sorted((item for item in buffers.items() if item[1]["nbytes"] >= min_bytes and not item[1]["models"]), key=lambda item: -item[1]["nbytes"])[:30]
    paths, complete, holder_time = {}, True, 0.0
    targets = {pointer: buffer["objects"] for pointer, buffer in top} if holders else {}
    for buffer in buffers.values():
        buffer["objects"] = None  # no reference left but the targets' (the holder search skips them), none in the report
    if targets:
        started_holders = time.perf_counter()
        paths, complete = _holder_paths(targets, budget_s)
        holder_time = time.perf_counter() - started_holders
    del targets
    types_by_address = {id(kind): kind for kind in found_types}
    types_by_address.update({id(kind): kind for kind in (bytes, bytearray, str, int, float, tuple)})
    return {"buffers": buffers, "images": images, "totals": totals, "registrations": registrations, "top": top, "paths": paths, "complete": complete, "types": types_by_address,
            "garbage": garbage, "times": (round(found_time, 2), round(holder_time, 2))}


def _region_lookup(found_regions):
    bases = [item["base"] for item in found_regions]

    def lookup(pointer):
        index = bisect.bisect_right(bases, pointer) - 1
        if index >= 0 and pointer < found_regions[index]["base"] + found_regions[index]["reserved"]:
            return found_regions[index]
        return None
    return lookup


def census(min_mb=16, objects=True, holders=True, budget_s=20, label="census", _hook=None):
    """A census of the process's RAM now (a dict, see the module's description): classes of memory, live objects, the holders of the largest."""
    started = time.perf_counter()
    process = counters()
    min_bytes = int(min_mb * MB)
    objs = _census_objects(min_bytes, holders, budget_s) if objects else None
    hook_blocks = _hook.blocks() if _hook is not None else None
    registrations = set(objs["registrations"]) if objs else set()
    if _rec is not None:  # the registrations made while recording are followed as they happen (see _watch_mmgp)
        registrations |= {(pointer, size, cls) for pointer, (size, cls) in list(_rec.pinned.items())}
    registrations = sorted(registrations)
    found_regions = reservations()
    evidence = _evidence(objs["buffers"] if objs else None, registrations, hook_blocks)
    hooked = _hook.live() if _hook is not None else []
    if hooked:  # the hooked allocations alive tell PyTorch's memory from NumPy's (the C heap) without a census of the objects
        evidence["torch_ptrs"] = sorted(set(evidence.get("torch_ptrs", [])) | {entry.ptr for entry in hooked if entry.kind == 0})
        evidence["array_ptrs"] = sorted(set(evidence.get("array_ptrs", [])) | {entry.ptr for entry in hooked if entry.kind == 1})
    evidence["ram_ranges"] = _ram.ranges() if _ram.active else []
    candidates = [item["base"] for item in found_regions if item["kind"] in ("private", "driver", "shared") and not item["wc"]]
    evidence["cuda_host"] = _cuda_host_bases(candidates)
    totals = classify(found_regions, evidence)
    extra = {"in_use": {}, "notes": {}}
    if WINDOWS:
        extra["heap"] = heap = _heap_summary()
        extra["registered_numpy"] = sum(size for _, size, cls in registrations if cls == "pinned_mmgp")
        extra["in_use"]["malloc"] = max(0, heap["in_use"] - extra["registered_numpy"])
    else:
        mallinfo = _mallinfo()
        if mallinfo:
            extra["mallinfo"] = mallinfo
            extra["in_use"]["malloc"] = mallinfo["in_use"]
            extra["notes"]["malloc"] = f"glibc: {mallinfo['free'] / GB:.2f} GiB free in its arenas, {mallinfo['mapped_blocks'] / GB:.2f} GiB in separately mapped blocks"
    region_of = _region_lookup(found_regions)
    locked_starts = [pointer for pointer, _, _ in registrations]

    def lookup_class(pointer):  # the class of the memory at pointer: a registration's, or its reservation's
        index = bisect.bisect_right(locked_starts, pointer) - 1
        if index >= 0 and pointer < registrations[index][0] + registrations[index][1]:
            return registrations[index][2]
        region = region_of(pointer)
        return region["cls"] if region else "?"
    live_by_class = collections.Counter()
    if objs:
        for pointer, buffer in objs["buffers"].items():
            buffer["cls"] = lookup_class(pointer)
            live_by_class[buffer["cls"]] += buffer["nbytes"]
    if _hook is not None:
        if WINDOWS:
            in_arenas = [entry.size for entry in hooked if entry.kind == 0 and lookup_class(entry.ptr) == "torch_cpu"]
            extra["in_use"]["torch_cpu"] = sum(in_arenas)
            extra["notes"]["torch_cpu"] = f"in use: {len(in_arenas)} hooked allocations of {SMALL_BYTES // KB} KiB and more; the rest is kept by mimalloc for reuse"
    elif objs and WINDOWS:
        extra["in_use"]["torch_cpu"] = live_by_class["torch_cpu"]
        extra["notes"]["torch_cpu"] = "in use: the CPU tensors Python sees; the rest is free memory mimalloc keeps, or tensors only C++ code holds"
    if objs:
        extra["in_use"]["pinned_cuda"] = live_by_class["pinned_cuda"]
    if _ram.active:  # the allocator's own counts; the registered (locked) blocks it handed out are in their own classes
        allocator = _ram.stats()
        registered_inside = sum(size for pointer, size, _ in registrations if any(base <= pointer < base + length for base, length in evidence["ram_ranges"]))
        extra["in_use"]["mmgp_ram"] = allocator["live"] - registered_inside
        extra["notes"]["mmgp_ram"] = (f"{allocator['cached'] / GB:.2f} GiB of freed blocks kept for reuse until the end of the workload; "
                                      f"{allocator['hits']} of {allocator['allocations']} allocations reused a block")
    pinned_cache = _torch_pinned_cache()
    if pinned_cache:
        extra["notes"]["pinned_cuda"] = f"PyTorch's pinned allocator holds {pinned_cache / GB:.2f} GiB (sizes rounded to powers of two; freed blocks stay cached until torch._C._host_emptyCache())"
    if WINDOWS and torch.cuda.is_available() and torch.cuda.is_initialized():
        extra["notes"]["cuda_driver"] = f"the process holds {torch.cuda.memory_reserved() / GB:.2f} GiB of VRAM: Windows charges it to the commit, it is not resident"
    if totals.get("openblas", {}).get("count"):
        extra["notes"]["openblas"] = "32 MiB per thread for each OpenBLAS copy (NumPy, SciPy): commit charge only until a BLAS call uses them"
    if totals.get("python", {}).get("count"):
        extra["notes"]["python"] = f"{totals['python']['count']} arenas of 1 MiB"
    result = {"label": label, "time_s": round(time.perf_counter() - (_rec.t0 if _rec is not None else started), 2), "platform": sys.platform, "process": _gb(process), "system": system(),
              "classes": _class_table(found_regions, totals, extra), "largest": _largest(found_regions), "files": _files(found_regions)}
    vmm = _vmm_stats()
    if vmm:
        result["vram_spilled_to_ram"] = vmm
    if objs:
        result["objects"] = _objects_report(objs, min_bytes)
        if WINDOWS and _hook is not None:
            result["c_heap"] = _hook.heap_report(objs["buffers"], registrations, objs["types"])
    if _hook is not None:
        result["hook"] = _hook.live_report(objs["buffers"] if objs else None)
    result["census_s"] = round(time.perf_counter() - started, 2)
    return result


def _gb(values):
    return {f"{key}_gb": None if value is None else round(value / GB, 3) for key, value in values.items()}


def _objects_report(objs, min_bytes):
    buffers = objs["buffers"]
    by_kind = collections.defaultdict(lambda: collections.Counter())
    for buffer in buffers.values():
        by_kind[buffer["kind"].split(" ")[0]][buffer["cls"]] += buffer["nbytes"]
    models = collections.defaultdict(lambda: collections.Counter())
    model_counts = collections.Counter()
    for buffer in buffers.values():
        if buffer["models"]:
            model = buffer["models"].most_common(1)[0][0]
            models[model][buffer["cls"]] += buffer["nbytes"]
            model_counts[model] += 1
    top = []
    for pointer, buffer in objs["top"]:
        top.append({"mb": round(buffer["nbytes"] / MB, 1), "kind": buffer["kind"], "what": buffer["what"], "memory": buffer["cls"],
                    "holders": objs["paths"].get(pointer, [])})
    images = sorted(objs["images"], key=lambda item: -item[0])
    garbage_count, garbage_bytes, cycles = objs["garbage"]
    return {"by_kind_gb": {kind: {cls: round(value / GB, 3) for cls, value in counter.most_common()} for kind, counter in by_kind.items()},
            "models": [{"model": model, "buffers": model_counts[model], "gb": {cls: round(value / GB, 3) for cls, value in counter.most_common()}}
                       for model, counter in sorted(models.items(), key=lambda item: -sum(item[1].values()))],
            "top": top, "holders_complete": objs["complete"],
            "pil_images": {"count": len(images), "gb": round(sum(size for size, _ in images) / GB, 3)},
            "cyclic_garbage": {"objects": garbage_count, "large_buffers_gb": round(garbage_bytes / GB, 3), "largest": cycles},
            "times_s": {"objects": objs["times"][0], "holders": objs["times"][1]}}


# ---------------------------------------------------------------------------------------------------------------------------- hook library


class _HookEntry(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64), ("size", ctypes.c_int64), ("origin", ctypes.c_int64), ("seq", ctypes.c_int64), ("kind", ctypes.c_int64), ("time", ctypes.c_double)]


class _HookHeader(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("phase", "torch_live", "numpy_live", "entries")] + [("time", ctypes.c_double)]


_HOOK_ORIGIN = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_size_t, ctypes.c_int)
_HOOK_KINDS = ("torch", "numpy")


def _hook_origin(size, kind):  # called by the hook library for each allocation of the threshold and more, from a Python thread
    rec = _rec
    if rec is None:
        return -1
    try:
        local = _vram_debug._local
        modules, tags = getattr(local, "modules", None), getattr(local, "tags", None)
        stack_id = -1
        if rec.stacks:
            frame, frames = sys._getframe().f_back, []
            while frame is not None:
                frames.append((frame.f_code, frame.f_lineno))
                frame = frame.f_back
            frames = tuple(frames)
            stack_id = rec.stack_ids.get(frames)
            if stack_id is None:
                stack_id = rec.stack_ids[frames] = len(rec.stack_frames)
                rec.stack_frames.append(frames)
        key = (modules[-1] if modules else "", tags[-1] if tags else "", stack_id)
        origin = rec.origin_ids.get(key)
        if origin is None:
            origin = rec.origin_ids[key] = len(rec.origin_keys)
            rec.origin_keys.append(key)
        return origin
    except Exception:
        return -1


_hook_origin_callback = _HOOK_ORIGIN(_hook_origin)  # kept for the life of the process: the library may call it while a recording stops


class _Hook:
    """The companion library: pass-through hooks of PyTorch's CPU allocator and NumPy's default allocator."""

    def __init__(self, lib, rec):
        self.lib, self.rec = lib, rec

    @staticmethod
    def load():
        """(library, None) or (None, the reason it cannot be used): the library of the RAM allocator (ram.py), which carries the hooks."""
        lib, reason = _ram.library()
        if lib is None:
            return None, reason
        lib.rh_start.argtypes = (ctypes.c_int64, ctypes.c_int64, _HOOK_ORIGIN, ctypes.c_void_p)
        lib.rh_install_torch.restype = lib.rh_patch_numpy.restype = ctypes.c_int
        lib.rh_patch_numpy.argtypes = (ctypes.c_void_p,)
        lib.rh_mark.argtypes = (ctypes.c_int64,)
        lib.rh_stats.argtypes = (ctypes.POINTER(ctypes.c_int64),)
        lib.rh_live.argtypes = (ctypes.POINTER(_HookEntry), ctypes.c_int64)
        lib.rh_phases.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
        lib.rh_phase.argtypes = (ctypes.c_int64, ctypes.POINTER(_HookHeader), ctypes.POINTER(_HookEntry), ctypes.c_int64)
        lib.rh_origins.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
        lib.rh_blocks.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64)
        for function in ("rh_live", "rh_phases", "rh_phase", "rh_origins", "rh_blocks"):
            getattr(lib, function).restype = ctypes.c_int64
        if WINDOWS:
            lib.rh_heap_walk.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.c_int64)
            lib.rh_heap_walk.restype = ctypes.c_int64
        return lib, None

    def start(self, threshold, numpy_hook=True):
        lib = self.lib
        self.offset = time.perf_counter() - self.rec.t0  # the library's clock starts now: its times + offset are the recording's
        lib.rh_start(threshold, SMALL_BYTES, _hook_origin_callback, ctypes.cast(ctypes.pythonapi.PyGILState_GetThisThreadState, ctypes.c_void_p))
        if lib.rh_install_torch() != 0:
            raise RuntimeError("PyTorch's CPU allocator could not be hooked")
        numpy = _numpy()
        if numpy_hook and numpy is not None:
            handler = _numpy_default_handler(numpy)
            if handler is None or lib.rh_patch_numpy(handler) != 0:
                print("[RAM debug] NumPy's allocator could not be hooked: NumPy arrays are only seen by the census")

    def stop(self):
        self.lib.rh_stop()  # the hooks stay: they forward, so memory they handed out is freed normally

    def mark(self, phase):
        self.lib.rh_mark(phase)

    def reset(self):
        self.lib.rh_reset()

    def stats(self):
        values = (ctypes.c_int64 * 8)()
        self.lib.rh_stats(values)
        return dict(zip(("torch_live", "numpy_live", "torch_count", "numpy_count", "unwrapped", "allocations", "numpy_patched", "torch_installed"), values))

    def _read(self, function, *args):
        count = function(*args, None, 0)
        buffer = (_HookEntry * count)()
        return list(buffer[:min(count, function(*args, buffer, count))])

    def live(self):
        return self._read(self.lib.rh_live)

    def blocks(self):
        count = self.lib.rh_blocks(None, 0)
        buffer = (ctypes.c_uint64 * count)()
        return set(buffer[:min(count, self.lib.rh_blocks(buffer, count))])

    def phase(self, phase):
        header = _HookHeader()
        entries = self._read(lambda out, cap: self.lib.rh_phase(phase, ctypes.byref(header), out, cap))
        return header, entries

    def phases(self):
        ids = (ctypes.c_int64 * 1024)()
        return list(ids[:self.lib.rh_phases(ids, 1024)])

    def origins(self):
        rows = (ctypes.c_int64 * (7 * 65536))()
        count = min(65536, self.lib.rh_origins(rows, 65536))
        return [tuple(rows[7 * row: 7 * row + 7]) for row in range(count)]

    def heap_report(self, buffers, registrations, types_by_address, top=40, sniffed=4096):
        """The blocks in use in the process heap (malloc's) by size; the blocks of 1 MiB and more (the sniffed largest) by owner, recognized from
        the Python objects found or from their first bytes; the largest of them (blocks of MMGP pinned weights are counted, not listed)."""
        started = time.perf_counter()
        walked = sniffed + len(registrations)
        classes, largest, heads = (ctypes.c_int64 * 10)(), (ctypes.c_uint64 * (2 * walked))(), (ctypes.c_char * (HEAD_BYTES * walked))()
        entries = self.lib.rh_heap_walk(classes, largest, heads, walked)
        if entries < 0:
            return None
        pointers = sorted(buffers) if buffers else []
        locked = sorted((pointer, pointer + size) for pointer, size, _ in registrations)
        rows, pinned, owners = [], [0, 0], collections.defaultdict(lambda: [0, 0])
        for index in range(walked):
            address, size = largest[2 * index], largest[2 * index + 1]
            if not size:
                break
            inside = bisect.bisect_right(locked, (address + size,)) - 1
            if inside >= 0 and locked[inside][1] > address:
                pinned[0] += 1
                pinned[1] += size
                continue
            first = bisect.bisect_left(pointers, address)
            held = first < len(pointers) and pointers[first] < address + size
            owner = buffers[pointers[first]]["what"] if held else _sniff(heads.raw[HEAD_BYTES * index: HEAD_BYTES * (index + 1)], types_by_address)
            kind = re.sub(r" \(.*|\(.*", "", owner.split(" ")[0]) if held else (owner if not owner.startswith("text data") else "text data")
            owners[kind][0] += 1
            owners[kind][1] += size
            if len(rows) < top:
                rows.append({"mb": round(size / MB, 1), "owner": owner})
        labels = ("under 1 KiB", "1 to 64 KiB", "64 KiB to 1 MiB", "1 to 16 MiB", "16 MiB and more")
        return {"entries": entries, "walk_s": round(time.perf_counter() - started, 2),
                "by_size": [{"blocks": labels[index], "count": classes[2 * index], "gb": round(classes[2 * index + 1] / GB, 3)} for index in range(5)],
                "mmgp_pinned_blocks": {"count": pinned[0], "gb": round(pinned[1] / GB, 3)}, "largest": rows,
                "large_blocks_by_owner": [{"owner": owner, "count": count, "gb": round(size / GB, 3)} for owner, (count, size) in sorted(owners.items(), key=lambda item: -item[1][1])]}

    def live_report(self, buffers):
        stats, entries = self.stats(), self.live()
        large = [entry for entry in entries if entry.size >= self.rec.threshold]
        report = {"torch_live_gb": round(stats["torch_live"] / GB, 3), "numpy_live_gb": round(stats["numpy_live"] / GB, 3),
                  "large_gb": round(sum(entry.size for entry in large) / GB, 3), "groups": _hook_groups(self.rec, large, time.perf_counter() - self.rec.t0 - self.offset)}
        if buffers is not None:  # live allocations that no Python object references: kept by C++ code or a library
            visible = buffers.keys()
            unheld = [entry for entry in large if entry.ptr not in visible]
            report["unheld_gb"] = round(sum(entry.size for entry in unheld) / GB, 3)
            report["unheld"] = _hook_groups(self.rec, unheld, time.perf_counter() - self.rec.t0 - self.offset)
        return report


HEAD_BYTES = 32  # first bytes of the largest C heap blocks, copied while the heap is locked
_SIGNATURES = ((b"\x7fELF", "ELF image (CUDA kernels or library data)"), (b"\x50\xed\x55\xba", "CUDA fat binary"), (b"\xb1\x43\x62\x46", "CUDA fat binary"),
               (b"PK\x03\x04", "zip data"), (b"\x89PNG", "PNG image"), (b"\xff\xd8\xff", "JPEG image"))


def _sniff(head, types_by_address):
    """What a C heap block holds, from its first bytes: a known format, or a Python object (its type pointer, after an optional GC header)."""
    for signature, text in _SIGNATURES:
        if head.startswith(signature):
            return text
    for offset in (8, 24):  # PyObject: refcount then type; objects tracked by the garbage collector start with a 16-byte header
        kind = types_by_address.get(int.from_bytes(head[offset:offset + 8], "little"))
        if kind is not None:
            return f"Python {kind.__module__}.{kind.__qualname__} object" if kind.__module__ != "builtins" else f"Python {kind.__qualname__} object"
    if head[:1] in (b"{", b"[") or head[:8].isascii() and head[:8].strip().isalnum():
        return f"text data {head[:24]!r}"
    return "no Python object found (C/C++ code, or a Python container's storage)"


def _numpy_default_handler(numpy):
    """The address of NumPy's default PyDataMem_Handler (shared by every thread whose context sets no other), or None."""
    core = getattr(numpy, "_core", None) or getattr(numpy, "core", None)
    api = getattr(getattr(core, "_multiarray_umath", None), "_ARRAY_API", None)
    if api is None or tuple(int(part) for part in numpy.__version__.split(".")[:2]) < (1, 22):
        return None
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype, get_pointer.argtypes = ctypes.c_void_p, (ctypes.py_object, ctypes.c_char_p)
    table = ctypes.cast(get_pointer(api, None), ctypes.POINTER(ctypes.c_void_p))
    default = ctypes.cast(table[306], ctypes.POINTER(ctypes.py_object)).contents.value  # PyDataMem_DefaultHandler (C-API index 306, NumPy 1.22+)
    return get_pointer(default, b"mem_handler")


def _ram_where(rec, stack_id):
    """The allocation site: the innermost frame of the application's code, with the mmgp function that allocated for it (pinning, staging)."""
    where = _vram_debug._where(rec, stack_id)
    for code, line in rec.stack_frames[stack_id]:
        if code.co_filename.endswith(os.path.join("mmgp", "offload.py")):
            name = getattr(code, "co_qualname", code.co_name)
            return f"{where} (mmgp {name})" if code.co_name not in where else where
    return where


def _hook_groups(rec, entries, now):
    groups = {}
    for entry in entries:
        module, region, stack_id = rec.origin_keys[entry.origin] if 0 <= entry.origin < len(rec.origin_keys) else ("", "", -1)
        key = (_vram_debug._normalized(module), region, stack_id, entry.kind)
        group = groups.get(key)
        if group is None:
            group = groups[key] = {"mb": 0.0, "count": 0, "sizes_mb": [], "kind": _HOOK_KINDS[entry.kind] if 0 <= entry.kind < 2 else str(entry.kind), "module": key[0], "tag": region,
                                   "stack": stack_id, "where": _ram_where(rec, stack_id) if stack_id >= 0 else ("(no Python thread)" if entry.origin == -2 else ""), "age_s": []}
        group["mb"] += entry.size / MB
        group["count"] += 1
        group["sizes_mb"].append(round(entry.size / MB, 1))
        group["age_s"].append(round(now - entry.time, 1))
    result = sorted(groups.values(), key=lambda group: -group["mb"])
    for group in result:
        group["mb"] = round(group["mb"], 1)
        group["sizes_mb"] = sorted(group["sizes_mb"], reverse=True)[:12]
        ages = group.pop("age_s")
        group["age_s"] = [min(ages), max(ages)]
    return result


# ---------------------------------------------------------------------------------------------------------------------------- recording
_rec = None


class _Recording:
    def __init__(self, min_mb, stacks, sample_s, holders_budget_s):
        self.min_mb, self.threshold, self.stacks, self.sample_s, self.holders_budget_s = min_mb, int(min_mb * MB), stacks, sample_s, holders_budget_s
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.phase, self.phase_names, self.phases = None, [], {}
        self.snapshots, self.baseline, self.pinned = [], None, {}  # pinned: pointer -> (size, class) of the RAM mmgp registered while recording
        self.stack_ids, self.stack_frames, self.origin_ids, self.origin_keys = {}, [], {}, []
        self.hook, self.hook_note, self.patched = None, None, []
        self.stop_event, self.sampler, self.last_light = threading.Event(), None, 0.0
        self.peak = {"private": 0, "resident": 0}
        self.history = []  # what each census of a report keeps for the following reports: the trend of the whole recording


def _new_phase(rec, name):
    if name not in rec.phases:
        rec.phase_names.append(name)
        rec.phases[name] = {"name": name, "peak_private": 0, "peak_resident": 0, "peak_time": 0.0, "first_time": round(time.perf_counter() - rec.t0, 2),
                            "census": None, "census_private": 0}
    return rec.phases[name]


def _sample(rec):
    while not rec.stop_event.wait(rec.sample_s):
        try:
            values = counters()
            now = time.perf_counter() - rec.t0
            with rec.lock:
                phase = rec.phases.get(rec.phase)
                if phase is None:
                    continue
                if values["private"] > phase["peak_private"]:
                    phase["peak_private"], phase["peak_time"], phase["resident_at_peak"] = values["private"], round(now, 2), values["resident"]
                phase["peak_resident"] = max(phase["peak_resident"], values["resident"])
                rec.peak["private"] = max(rec.peak["private"], values["private"])
                rec.peak["resident"] = max(rec.peak["resident"], values["resident"])
                due = values["private"] >= phase["census_private"] + max(256 * MB, values["private"] // 20) and now - rec.last_light >= 1.0
            if due:  # a light census (address space and hooked allocations) near the peak of the phase
                light = census(objects=False, label=f"peak of {phase['name']}", _hook=rec.hook)
                with rec.lock:
                    phase["census"], phase["census_private"] = light, values["private"]
                    rec.last_light = time.perf_counter() - rec.t0
        except Exception as error:  # the sampler must never stop the workload
            print(f"[RAM debug] sampler error: {error}")


def _watch_mmgp(rec):  # a phase per model that mmgp's offload loads to the GPU, as in the VRAM debug mode; the RAM it registers (pins)
    from mmgp import offload
    for attribute in ("gpu_load", "gpu_load_blocks"):
        original = getattr(offload.offload, attribute)

        def wrapped(self, model_id, *args, _original=original, **kwargs):
            if _rec is rec and model_id != rec.phase:
                mark(model_id)
            return _original(self, model_id, *args, **kwargs)
        setattr(offload.offload, attribute, wrapped)
        rec.patched.append((offload.offload, attribute, original))
    registration = offload._HostRegistration
    original_register, original_close = registration.register, registration.close

    def register(self):
        error = original_register(self)
        if int(error) == 0:
            rec.pinned[self.pointer] = (self.size, "pinned_ring" if isinstance(self.buffer, torch.Tensor) else "pinned_mmgp")
        return error

    def close(self):
        closed = original_close(self)
        if closed:
            rec.pinned.pop(self.pointer, None)
        return closed
    registration.register, registration.close = register, close
    rec.patched += [(registration, "register", original_register), (registration, "close", original_close)]


def start(min_mb=16, stacks=True, hooks=True, sample_ms=50, holders_budget_s=20, watch_mmgp=True):
    """Starts the RAM debug mode: a sampler of the process's memory, phases, and with hooks the companion library's allocation records of
    min_mb or more. Takes a first census (the baseline of the reports)."""
    global _rec
    if _rec is not None:
        stop()
    rec = _Recording(min_mb, stacks, sample_ms / 1000, holders_budget_s)
    if hooks:
        lib, reason = _Hook.load()
        if lib is None:
            rec.hook_note = f"allocation hooks unavailable: {reason}"
            print(f"[RAM debug] {rec.hook_note}; the census still runs")
        else:
            rec.hook = _Hook(lib, rec)
    _vram_debug._use_module_hooks()
    if watch_mmgp:
        _watch_mmgp(rec)
    _rec = rec
    if rec.hook is not None:
        rec.hook.start(rec.threshold)
        atexit.register(rec.hook.stop)  # no origin callback while Python shuts down
    mark("start")
    rec.baseline = census(min_mb, objects=True, holders=False, label="start of the recording", _hook=rec.hook)
    rec.history.append(_trend_entry(None, rec.baseline))
    rec.sampler = threading.Thread(target=_sample, args=(rec,), name="mmgp_ram_debug", daemon=True)
    rec.sampler.start()


def stop():
    global _rec
    rec = _rec
    if rec is None:
        return
    rec.stop_event.set()
    rec.sampler.join()
    if rec.hook is not None:
        rec.hook.stop()
    _rec = None
    _vram_debug._release_module_hooks()
    for owner, attribute, original in rec.patched:
        setattr(owner, attribute, original)


def active():
    return _rec is not None


def mark(name):
    """Starts or continues the phase name: the sampler follows its peak, the hooks copy their live allocations at its peak."""
    rec = _rec
    if rec is None:
        return
    with rec.lock:
        _new_phase(rec, name)
        rec.phase = name
    if rec.hook is not None:
        rec.hook.mark(rec.phase_names.index(name))


def snapshot(label, objects=True, holders=True):
    """A full census now, kept for the next report."""
    rec = _rec
    if rec is None:
        raise RuntimeError("No RAM debug recording in progress: ram_debug.start() first.")
    result = census(rec.min_mb, objects=objects, holders=holders, budget_s=rec.holders_budget_s, label=label, _hook=rec.hook)
    rec.snapshots.append(result)
    return result


def reset():
    """New peaks, phases and snapshots (between two reports); the hooked allocations alive stay recorded."""
    rec = _rec
    if rec is None:
        return
    with rec.lock:
        rec.phase, rec.phase_names, rec.phases, rec.snapshots = None, [], {}, []
        rec.peak = {"private": 0, "resident": 0}
    if rec.hook is not None:
        rec.hook.reset()
    mark("start")


def _trend_entry(label, snap):
    """What a report keeps of its census for the following ones: memory by class, the largest objects and the hooked allocations alive."""
    objects, hook = snap.get("objects") or {"top": []}, snap.get("hook") or {}
    return {"label": f"{label}, {snap['label']}" if label else snap["label"], "census": snap["label"], "time_s": snap["time_s"],
            "private_gb": snap["process"]["private_gb"], "resident_gb": snap["process"]["resident_gb"], "classes": {row["label"]: row["private_gb"] for row in snap["classes"]},
            "objects": [{"mb": item["mb"], "what": item["what"], "memory": item["memory"], "held_by": item["holders"][0] if item["holders"] else ""} for item in objects["top"]],
            "allocations": [{"mb": round(group["mb"], 1), "count": group["count"], "kind": group["kind"], "where": group["where"], "stack": group["stack"]} for group in hook.get("groups", [])]}


def _phase_report(rec, phase, hook_phase):
    result = {"name": phase["name"], "first_s": phase["first_time"], "peak_private_gb": round(phase["peak_private"] / GB, 3),
              "peak_resident_gb": round(phase["peak_resident"] / GB, 3), "peak_time_s": phase["peak_time"],
              "resident_at_peak_gb": round(phase.get("resident_at_peak", 0) / GB, 3)}
    if phase["census"] is not None:
        light = phase["census"]
        result["census"] = {"at_private_gb": light["process"]["private_gb"], "time_s": light["time_s"], "classes": light["classes"],
                            "largest": light["largest"][:10], "files": [item for item in light["files"] if item["kind"] == "data"][:8]}
    if hook_phase is not None:
        header, entries = hook_phase
        large = [entry for entry in entries]
        result["hooked_at_peak"] = {"torch_gb": round(header.torch_live / GB, 3), "numpy_gb": round(header.numpy_live / GB, 3), "time_s": round(header.time + rec.hook.offset, 2),
                                    "groups": _hook_groups(rec, large, header.time)}
    return result


def report(path=None, label=None, snapshot_label="report"):
    """Takes a full census (snapshot_label, unless None), then returns the report (a dict), also written to path (.json) with a summary (.md)."""
    rec = _rec
    if rec is None:
        raise RuntimeError("No RAM debug recording in progress: ram_debug.start() first.")
    if snapshot_label:
        snapshot(snapshot_label)
    if rec.snapshots:
        rec.history.append(_trend_entry(label, rec.snapshots[-1]))
    hook_phases = {}
    if rec.hook is not None:
        for phase_id in rec.hook.phases():
            if phase_id < len(rec.phase_names):
                hook_phases[rec.phase_names[phase_id]] = rec.hook.phase(phase_id)
    with rec.lock:
        phases = [_phase_report(rec, rec.phases[name], hook_phases.get(name)) for name in rec.phase_names
                  if rec.phases[name]["peak_private"] or name in hook_phases and hook_phases[name][1]]
        peak = dict(rec.peak)
    origins = []
    if rec.hook is not None:
        merged = {}
        for origin, kind, count, total, largest, freed, life_us in rec.hook.origins():
            module, region, stack_id = rec.origin_keys[origin] if 0 <= origin < len(rec.origin_keys) else ("", "", -1)
            where = _ram_where(rec, stack_id) if stack_id >= 0 else ""
            entry = merged.setdefault((_vram_debug._normalized(module), region, where, kind), {"kind": _HOOK_KINDS[kind] if 0 <= kind < 2 else str(kind), "module": _vram_debug._normalized(module),
                                                                                            "tag": region, "stack": stack_id, "where": where, "count": 0, "total": 0, "max": 0, "freed": 0, "life_us": 0})
            entry["count"] += count
            entry["total"] += total
            entry["max"] = max(entry["max"], largest)
            entry["freed"] += freed
            entry["life_us"] += life_us
        origins = [{"kind": entry["kind"], "module": entry["module"], "tag": entry["tag"], "stack": entry["stack"], "where": entry["where"], "count": entry["count"],
                    "total_gb": round(entry["total"] / GB, 3), "max_mb": round(entry["max"] / MB, 1), "freed": entry["freed"],
                    "mean_life_ms": round(entry["life_us"] / entry["freed"] / 1000, 2) if entry["freed"] else None} for entry in merged.values()]
        origins.sort(key=lambda origin: -origin["total_gb"])
    used = set()
    for phase in phases:
        used |= {group["stack"] for group in phase.get("hooked_at_peak", {}).get("groups", [])}
    for snap in rec.snapshots:
        used |= {group["stack"] for key in ("groups", "unheld") for group in snap.get("hook", {}).get(key, [])}
    used |= {origin["stack"] for origin in origins[:200]}
    used |= {group["stack"] for entry in rec.history[-2:] for group in entry["allocations"]}
    result = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, "platform": f"{sys.platform} {platform.machine()}", "threshold_mb": rec.min_mb,
              "hooks": "PyTorch CPU allocator and NumPy" if rec.hook is not None else rec.hook_note, "peak": _gb(peak), "baseline": rec.baseline, "phases": phases,
              "snapshots": rec.snapshots, "history": rec.history, "origins": origins[:200],
              "stacks": {str(stack_id): [_vram_debug._frame_text(code, line) for code, line in rec.stack_frames[stack_id][:40]] for stack_id in sorted(used) if stack_id >= 0}}
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as writer:
            json.dump(result, writer, indent=1)
        with open(os.path.splitext(path)[0] + ".md", "w", encoding="utf-8") as writer:
            writer.write(summary(result))
    return result


# ---------------------------------------------------------------------------------------------------------------------------- external scan
def wangp_processes():
    """The running WanGP processes (wgp.py among their arguments), the largest private memory first: [(pid, private bytes, command line)]."""
    import psutil
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            args = process.info["cmdline"] or []
            if process.info["pid"] != os.getpid() and any(os.path.basename(arg) == "wgp.py" for arg in args):
                found.append((process.info["pid"], counters(process.info["pid"])["private"], " ".join(args)))
        except (OSError, psutil.Error):
            pass
    return sorted(found, key=lambda item: -item[1])


def scan(pid):
    """A census of another process from outside (no object census, no heap summaries, no CUDA queries): its classes are inferred from the
    shape of its allocations (mimalloc arenas, 32 MiB OpenBLAS buffers, 1 MiB Python arenas, write-combined CUDA buffers)."""
    found = reservations(pid)
    totals = classify(found, {"numpy": True, "external": True})
    import psutil
    return {"label": f"process {pid} (scanned from outside)", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "command": " ".join(psutil.Process(pid).cmdline()),
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__}, "time_s": 0, "platform": sys.platform, "process": _gb(counters(pid)), "system": system(),
            "classes": _class_table(found, totals, {"notes": {"torch_cpu": "inferred from the arena shape: free and live memory together",
                                                               "other": "includes the C heaps (malloc), not separable from outside",
                                                               "mmgp_ram": "live tensors and freed blocks kept for reuse together"}}),
            "largest": _largest(found, top=60), "files": _files(found, top=40)}


# ---------------------------------------------------------------------------------------------------------------------------- summaries
def _class_rows(classes):
    lines = ["| memory | private GiB | resident GiB | in use GiB | free, kept by its allocator GiB | notes |", "|---|---:|---:|---:|---:|---|"]
    for row in classes:
        in_use, free = row.get("in_use_gb"), row.get("free_gb")
        lines.append(f"| {row['label']} | {row['private_gb']:.2f} | {row['resident_gb']:.2f} | {'' if in_use is None else f'{in_use:.2f}'} | "
                     f"{'' if free is None else f'{free:.2f}'} | {row.get('note', '')} |")
    return "\n".join(lines)


def _group_rows(groups, top=20):
    lines = ["| MB | count | kind | module | allocated at | tag | age (s) |", "|---:|---:|---|---|---|---|---|"]
    for group in groups[:top]:
        age = group["age_s"]
        lines.append(f"| {group['mb']:.0f} | {group['count']} | {group['kind']} | {group['module'] or '-'} | {group['where'] or '-'} | {group['tag'] or '-'} | "
                     f"{age[0]:.0f}{'' if age[0] == age[1] else f'-{age[1]:.0f}'} |")
    if len(groups) > top:
        lines.append(f"| {sum(group['mb'] for group in groups[top:]):.0f} | {sum(group['count'] for group in groups[top:])} | | {len(groups) - top} more origins | | | |")
    return "\n".join(lines)


def _census_summary(snap, title):
    process, system_info = snap["process"], snap["system"]
    meaning = "commit size" if snap.get("platform") == "win32" else "anonymous memory, resident or swapped"
    text = [f"## {title}\n\nProcess: private {process['private_gb']:.2f} GiB ({meaning}), resident {process['resident_gb']:.2f} GiB. "
            f"System: {system_info['available_gb']:.1f} of {system_info['physical_gb']:.1f} GiB available, commit {system_info['commit_gb']:.1f} / {system_info['commit_limit_gb']:.1f} GiB.\n\n",
            _class_rows(snap["classes"]), "\n\n"]
    if snap.get("vram_spilled_to_ram"):
        spill = snap["vram_spilled_to_ram"]
        text.append(f"VRAM allocations spilled to RAM by the VRAM allocator: pinned {spill['spilled_gb']:.2f} GiB, by the driver {spill['driver_spilled_gb']:.2f} GiB.\n\n")
    objects = snap.get("objects")
    if objects:
        kinds = ", ".join(f"{kind} {sum(values.values()):.2f}" for kind, values in objects["by_kind_gb"].items())
        text.append(f"Python objects alive (GiB): {kinds or 'none'}; PIL images {objects['pil_images']['count']} ({objects['pil_images']['gb']:.2f} GiB); "
                    f"reference cycles collected now: {objects['cyclic_garbage']['objects']} objects, {objects['cyclic_garbage']['large_buffers_gb']:.2f} GiB of buffers. "
                    f"Census {objects['times_s']['objects']:.1f} s, holders {objects['times_s']['holders']:.1f} s"
                    f"{'' if objects['holders_complete'] else ' (time budget reached: some chains are incomplete)'}.\n\n")
        if objects["cyclic_garbage"].get("largest"):
            text.append("| MB | buffer of a reference cycle (alive until a garbage collection) | kept by |\n|---:|---|---|\n")
            for cycle in objects["cyclic_garbage"]["largest"]:
                text.append(f"| {cycle['mb']:.0f} | {cycle['what']} | {cycle['chain']} |\n")
            text.append("\n")
        if objects["models"]:
            text.append("| model (weights in RAM) | buffers | GiB by memory |\n|---|---:|---|\n")
            for model in objects["models"]:
                text.append(f"| {model['model']} | {model['buffers']} | {', '.join(f'{cls} {gb:.2f}' for cls, gb in model['gb'].items())} |\n")
            text.append("\n")
        if objects["top"]:
            text.append("| MB | object | memory | held by |\n|---:|---|---|---|\n")
            for item in objects["top"]:
                holders = "<br>".join(item["holders"]) or "-"
                text.append(f"| {item['mb']:.0f} | {item['what']} | {item['memory']} | {holders} |\n")
            text.append("\n")
    heap = snap.get("c_heap")
    if heap:
        sizes = ", ".join(f"{row['blocks']} {row['gb']:.2f} GiB ({row['count']})" for row in heap["by_size"] if row["count"])
        pinned = heap["mmgp_pinned_blocks"]
        text.append(f"C heap blocks in use by size: {sizes}" + (f"; of which MMGP pinned weights {pinned['gb']:.2f} GiB ({pinned['count']} blocks, not listed below)" if pinned["count"] else "") + ".\n\n")
        if heap.get("large_blocks_by_owner"):
            text.append("| C heap blocks of 1 MiB and more, by owner | blocks | GiB |\n|---|---:|---:|\n" + "".join(f"| {row['owner']} | {row['count']} | {row['gb']:.2f} |\n" for row in heap["large_blocks_by_owner"][:12]) + "\n")
        if heap["largest"]:
            text.append("| largest C heap blocks in use, MB | owner |\n|---:|---|\n" + "".join(f"| {row['mb']:.0f} | {row['owner']} |\n" for row in heap["largest"][:10]) + "\n")
    hook = snap.get("hook")
    if hook:
        text.append(f"Hooked allocations alive: PyTorch {hook['torch_live_gb']:.2f} GiB, NumPy {hook['numpy_live_gb']:.2f} GiB; "
                    f"{hook['large_gb']:.2f} GiB in allocations of the threshold and more.\n\n")
        if hook.get("unheld"):
            text.append(f"Of these, {hook['unheld_gb']:.2f} GiB are referenced by no Python object (held by C++ code or a library), by origin:\n\n{_group_rows(hook['unheld'])}\n\n")
    data_files = [item for item in snap.get("files", []) if item["kind"] == "data"]
    if data_files:
        text.append("| memory-mapped file | resident GiB | private GiB |\n|---|---:|---:|\n")
        for item in data_files[:12]:
            text.append(f"| {item['file']} | {item['resident_gb']:.2f} | {item['private_gb']:.2f} |\n")
        text.append("\n")
    return "".join(text)


def summary(result):
    """Markdown summary of a report."""
    text = [f"# RAM debug report, {result['created']}{' (' + result['label'] + ')' if result.get('label') else ''}\n\n"
            f"Peak since the previous report: private {result['peak']['private_gb']:.2f} GiB, resident {result['peak']['resident_gb']:.2f} GiB. "
            f"Allocation hooks: {result['hooks']}; threshold {result['threshold_mb']} MB.\n\n"]
    if len(result.get("history", [])) > 1:
        text.append(_trend_summary(result["history"]))
    if result["phases"]:
        text.append("## Peaks by phase\n\n| phase | peak private GiB | resident at it | peak resident GiB | at (s) | largest memory near the peak |\n|---|---:|---:|---:|---:|---|\n")
        for phase in result["phases"]:
            largest = ", ".join(f"{row['label']} {row['private_gb']:.1f}" for row in sorted(phase.get("census", {}).get("classes", []), key=lambda row: -row["private_gb"])[:4])
            if phase["peak_private_gb"]:
                text.append(f"| {phase['name']} | {phase['peak_private_gb']:.2f} | {phase['resident_at_peak_gb']:.2f} | {phase['peak_resident_gb']:.2f} | {phase['peak_time_s']:.1f} | {largest} |\n")
            else:
                text.append(f"| {phase['name']} | - | - | - | - | too short for the sampler |\n")
        text.append("\n")
        for phase in result["phases"]:
            hooked = phase.get("hooked_at_peak")
            if hooked and hooked["groups"]:
                pinning = [group for group in hooked["groups"] if "(mmgp _pinned_block" in group["where"] or "(mmgp _StagingRing" in group["where"]]
                others = [group for group in hooked["groups"] if group not in pinning]
                note = f"; MMGP pinned weights and staging ring {sum(group['mb'] for group in pinning) / 1024:.2f} GiB, not listed" if pinning else ""
                text.append(f"### Hooked allocations alive at the peak of {phase['name']} ({hooked['time_s']:.1f} s): PyTorch {hooked['torch_gb']:.2f} GiB, NumPy {hooked['numpy_gb']:.2f} GiB{note}\n\n"
                            + (f"{_group_rows(others)}\n\n" if others else ""))
    for snap in result["snapshots"]:
        text.append(_census_summary(snap, f"{snap['label']} ({snap['time_s']:.1f} s)"))
    if result.get("baseline"):
        text.append(_census_summary(result["baseline"], f"Baseline: {result['baseline']['label']}"))
    if result["origins"]:
        text.append("## Hooked allocations by origin (totals)\n\n| kind | allocations | total GiB | max MB | mean life (ms) | module | allocated at |\n|---|---:|---:|---:|---:|---|---|\n")
        for origin in result["origins"][:25]:
            life = origin["mean_life_ms"]
            text.append(f"| {origin['kind']} | {origin['count']} | {origin['total_gb']:.2f} | {origin['max_mb']:.0f} | {'-' if life is None else f'{life:.1f}'} | "
                        f"{origin['module'] or '-'} | {origin['where'] or '-'} |\n")
        text.append("\n")
    text.append("Stacks (innermost frame first) are in the JSON report under `stacks`, by the `stack` id of each group.\n")
    return "".join(text)


def _trend_summary(history):
    """Markdown: the RAM after each report since the start of the recording, and what is alive now that was not at the previous census of
    the same kind (after a generation, after a model release): with the same generation repeated, what keeps appearing there is the leak."""
    first, last = history[0], history[-1]
    labels = list(dict.fromkeys(label for entry in history for label in entry["classes"]))
    spans = {label: [entry["classes"].get(label, 0.0) for entry in history] for label in labels}
    changing = sorted((label for label in labels if max(spans[label]) - min(spans[label]) >= 0.1), key=lambda label: -abs(spans[label][-1] - spans[label][0]))[:6]
    text = ["## RAM after each report since the start of the recording (GiB)\n\n| census | time (s) | private | resident | " + " | ".join(changing) + " |\n|---|---:|---:|---:|" + "---:|" * len(changing) + "\n"]
    for entry in history:
        text.append(f"| {entry['label']} | {entry['time_s']:.0f} | {entry['private_gb']:.2f} | {entry['resident_gb']:.2f} | " + " | ".join(f"{entry['classes'].get(label, 0.0):.2f}" for label in changing) + " |\n")
    text.append(f"\nChange since the start: private {last['private_gb'] - first['private_gb']:+.2f} GiB, resident {last['resident_gb'] - first['resident_gb']:+.2f} GiB.\n\n")
    previous = next((entry for entry in reversed(history[:-1]) if entry["census"] == last["census"]), None)
    if previous is None:
        return "".join(text)
    seen = {(item["what"], item["held_by"]) for item in previous["objects"]}
    new_objects = [item for item in last["objects"] if (item["what"], item["held_by"]) not in seen]
    def by_origin(entry):
        totals = {}
        for group in entry["allocations"]:
            total = totals.setdefault((group["kind"], group["where"]), {"mb": 0.0, "count": 0, "stack": group["stack"]})
            total["mb"] += group["mb"]
            total["count"] += group["count"]
        return totals
    before, now = by_origin(previous), by_origin(last)
    grown = sorted(((key, total, total["mb"] - before.get(key, {"mb": 0.0})["mb"], total["count"] - before.get(key, {"count": 0})["count"]) for key, total in now.items()), key=lambda row: -row[2])
    grown = [row for row in grown if row[2] >= 1]
    text.append(f"## Alive now and not at the previous census of this kind ({previous['label']})\n\n")
    if not new_objects and not grown:
        text.append("Nothing new among the largest objects and the hooked allocations.\n\n")
    if new_objects:
        text.append("| MB | new object | memory | held by |\n|---:|---|---|---|\n" + "".join(f"| {item['mb']:.0f} | {item['what']} | {item['memory']} | {item['held_by'] or '-'} |\n" for item in new_objects) + "\n")
    if grown:
        text.append("| MB now | change | allocations (change) | kind | allocated at | stack |\n|---:|---:|---:|---|---|---:|\n"
                    + "".join(f"| {total['mb']:.0f} | {change:+.0f} | {total['count']} ({count_change:+d}) | {kind} | {where or '-'} | {total['stack']} |\n" for (kind, where), total, change, count_change in grown[:20]) + "\n")
    return "".join(text)


def diff(before, after):
    """Markdown comparison of the last snapshots of two reports (or two scans) by class of memory."""
    def last(result):
        return result["snapshots"][-1] if "snapshots" in result and result["snapshots"] else result
    old, new = last(before), last(after)
    rows = {}
    for index, snap in enumerate((old, new)):
        for row in snap["classes"]:
            rows.setdefault(row["label"], [[0.0, 0.0], [0.0, 0.0]])[index] = [row["private_gb"], row["resident_gb"]]
    lines = [f"# RAM debug comparison: {old.get('label')} -> {new.get('label')}\n",
             f"Process private {old['process']['private_gb']:.2f} -> {new['process']['private_gb']:.2f} GiB, resident {old['process']['resident_gb']:.2f} -> {new['process']['resident_gb']:.2f} GiB\n",
             "| memory | private before | after | change | resident before | after | change |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, ((p0, r0), (p1, r1)) in sorted(rows.items(), key=lambda item: -abs(item[1][1][0] - item[1][0][0])):
        lines.append(f"| {label} | {p0:.2f} | {p1:.2f} | {p1 - p0:+.2f} | {r0:.2f} | {r1:.2f} | {r1 - r0:+.2f} |")
    return "\n".join(lines) + "\n"


def _main(argv):
    if len(argv) == 2 and argv[0] == "summary":
        with open(argv[1], encoding="utf-8") as reader:
            result = json.load(reader)
        print(summary(result) if "snapshots" in result else _census_summary(result, result["label"]))
    elif len(argv) == 3 and argv[0] == "diff":
        with open(argv[1], encoding="utf-8") as first, open(argv[2], encoding="utf-8") as second:
            print(diff(json.load(first), json.load(second)))
    elif argv and argv[0] == "scan" and len(argv) <= 3:
        rest = argv[1:]
        pid = int(rest.pop(0)) if rest and rest[0].isdigit() else None
        if pid is None:  # the running WanGP with the most memory
            processes = wangp_processes()
            if not processes:
                print("No running WanGP (wgp.py) was found: give its process id, python -m mmgp.allocator.ram_debug scan PID")
                return
            pid = processes[0][0]
            for other, private, command in processes[1:]:
                print(f"Also running: WanGP process {other} ({private / GB:.1f} GiB), not scanned: {command[:120]}")
        path = rest[0] if rest else f"ram_scan_{pid}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        try:
            result = scan(pid)
        except OSError as error:
            print(f"Process {pid} cannot be read ({error}): run this command as the same user as WanGP.")
            return
        with open(path, "w", encoding="utf-8") as writer:
            json.dump(result, writer, indent=1)
        print(_census_summary(result, f"{result['label']}: {result['command'][:120]}"))
        print(f"Saved to {os.path.abspath(path)}")
    else:
        print("usage: python -m mmgp.allocator.ram_debug summary report.json | diff before.json after.json | scan [PID] [out.json]\n"
              "scan without PID: the running WanGP with the most memory; the result is saved to ram_scan_<pid>_<time>.json unless out.json is given")


if __name__ == "__main__":
    _main(sys.argv[1:])
