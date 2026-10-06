# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""VRAM debug mode of the mmgp allocator: which tensors fill the VRAM at its peak, and where in the code they come from.

    from mmgp.allocator import debug
    debug.start(min_mb=16)          # the allocator must be installed (mode "vmm")
    ...                             # run the workload
    debug.report("vram.json")       # JSON for an agent, and vram.md, a summary

Every allocation of min_mb or more is recorded with its origin: the innermost module running (its path in the models that mmgp's
offload loads, or in those given to name_models), a tag, and the Python stack (raw frames, formatted in the report: about 8 us per
recorded allocation; max_stacks_per_s limits them). Tags: `x._allocator_tag = "kv cache"` names the allocation of a tensor (views share
it, the first tag is kept), `with debug.tag("decode tile"):` names the allocations of a block of code. The allocator copies the records
at the peak of each phase (one phase per model that mmgp loads to the GPU, or debug.mark("name")) and when an allocation fails, and totals
the allocations of each origin (count, bytes, lifetime of the freed ones: scratch buffers). The report lists, for each phase, the
tensors alive at its peak grouped by origin, with their sizes, ages at the peak, tags and stacks; the recorded allocations still alive
when it is written (leaks); the totals of each origin. Without debug mode the tags cost nothing (a plain attribute, an empty context).
The origin is asked from inside the allocation, which takes the GIL: a thread holding the GIL while waiting for a lock that the
allocating thread holds would deadlock (PyTorch's own memory history has the same constraint).

Command line: python -m mmgp.allocator.debug summary report.json | diff before.json after.json"""
import contextlib
import ctypes
import json
import os
import re
import sys
import threading
import time

import torch

KINDS = ("chunk range", "small pool", "mid-size pool", "spilled", "graph pool")
GB, MB = 2 ** 30, 2 ** 20


class _Entry(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64), ("size", ctypes.c_int64), ("origin", ctypes.c_int64), ("seq", ctypes.c_int64), ("kind", ctypes.c_int64), ("time", ctypes.c_double)]


class _Header(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("phase", "allocated", "reserved", "spilled", "chunks", "small_pool", "mid_pool", "graph_pools",
                                                    "cached_ranges", "spare", "small_cached", "request", "entries")] + [("time", ctypes.c_double)]


_ORIGIN = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_size_t, ctypes.c_int)
_local = threading.local()
_rec = None  # the recording in progress


class _Recording:
    def __init__(self, min_mb, stacks, max_stacks_per_s):
        self.threshold = int(min_mb * MB)
        self.min_mb = min_mb
        self.stacks = stacks
        self.stack_interval = 1 / max_stacks_per_s if max_stacks_per_s else 0
        self.last_stack = 0.0
        self.origin_ids, self.origin_keys = {}, []  # (module, tag, stack id) <-> origin id
        self.stack_ids, self.stack_frames = {}, []  # raw frames <-> stack id
        self.phase_ids, self.phase_names = {}, []
        self.phase = None
        self.tags_by_seq = {}
        self.module_names = {}  # id(module) -> path
        self.named_models = set()
        self.hooks, self.patched = [], []
        self.t0 = 0.0  # the library's recording clock starts with start()


def _library():
    from mmgp import allocator
    if allocator.active != "vmm":
        raise RuntimeError("The VRAM debug mode needs the mmgp allocator: allocator.install('vmm') before CUDA starts.")
    lib = allocator._lib
    if not getattr(lib, "_debug_ready", False):
        lib.vmm_debug_start.argtypes = (ctypes.c_size_t, _ORIGIN)
        lib.vmm_debug_mark.argtypes = (ctypes.c_int64,)
        lib.vmm_debug_seq.argtypes = (ctypes.c_void_p,)
        lib.vmm_debug_seq.restype = ctypes.c_int64
        lib.vmm_debug_phases.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
        lib.vmm_debug_phases.restype = ctypes.c_int64
        lib.vmm_debug_snapshot.argtypes = (ctypes.c_int, ctypes.c_int64, ctypes.POINTER(_Header), ctypes.POINTER(_Entry), ctypes.c_int64)
        lib.vmm_debug_snapshot.restype = ctypes.c_int64
        lib.vmm_debug_live.argtypes = (ctypes.c_int, ctypes.POINTER(_Entry), ctypes.c_int64)
        lib.vmm_debug_live.restype = ctypes.c_int64
        lib.vmm_debug_origins.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
        lib.vmm_debug_origins.restype = ctypes.c_int64
        lib._debug_ready = True
    return lib


def _origin(size, device):  # called by the allocator for each allocation of the threshold and more
    rec = _rec
    if rec is None:
        return -1
    try:
        modules, tags = getattr(_local, "modules", None), getattr(_local, "tags", None)
        stack_id = -1
        if rec.stacks:
            now = time.perf_counter()
            if now - rec.last_stack >= rec.stack_interval:
                rec.last_stack = now
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


_origin_callback = _ORIGIN(_origin)  # kept for the life of the process: the library may call it while a recording stops


def _module_entered(module, args):
    modules = getattr(_local, "modules", None)
    if modules is None:
        modules = _local.modules = []
    rec = _rec
    name = rec.module_names.get(id(module)) if rec is not None else None
    modules.append(name or (modules[-1] + "/" if modules else "") + type(module).__name__)


def _module_left(module, args, output):
    modules = getattr(_local, "modules", None)
    if modules:
        modules.pop()


def _tag_get(tensor):
    return tensor.__dict__.get("_allocator_tag")


def _tag_set(tensor, name):
    tensor.__dict__["_allocator_tag"] = name
    rec = _rec
    if rec is not None and tensor.is_cuda:
        seq = _library().vmm_debug_seq(tensor.untyped_storage().data_ptr())
        if seq >= 0:
            rec.tags_by_seq.setdefault(seq, name)


def _watch_mmgp(rec):  # a phase per model that mmgp's offload loads to the GPU, whose modules get their paths
    from mmgp import offload
    for attribute in ("gpu_load", "gpu_load_blocks"):
        original = getattr(offload.offload, attribute)

        def wrapped(self, model_id, *args, _original=original, **kwargs):
            if model_id != rec.phase:
                if model_id not in rec.named_models and model_id in self.models:
                    name_models({model_id: self.models[model_id]})
                mark(model_id)
            return _original(self, model_id, *args, **kwargs)
        setattr(offload.offload, attribute, wrapped)
        rec.patched.append((offload.offload, attribute, original))


def start(min_mb=16, stacks=True, max_stacks_per_s=0, watch_mmgp=True):
    """Starts recording the allocations of min_mb or more (a new recording if one is in progress)."""
    global _rec
    if _rec is not None:
        stop()
    lib = _library()
    rec = _Recording(min_mb, stacks, max_stacks_per_s)
    rec.hooks = [torch.nn.modules.module.register_module_forward_pre_hook(_module_entered),
                 torch.nn.modules.module.register_module_forward_hook(_module_left, always_call=True)]
    torch.Tensor._allocator_tag = property(_tag_get, _tag_set)
    if watch_mmgp:
        _watch_mmgp(rec)
    _rec = rec
    rec.t0 = time.perf_counter()
    lib.vmm_debug_start(rec.threshold, _origin_callback)
    mark("start")


def stop():
    global _rec
    rec = _rec
    if rec is None:
        return
    _library().vmm_debug_stop()
    _rec = None
    for hook in rec.hooks:
        hook.remove()
    for owner, attribute, original in rec.patched:
        setattr(owner, attribute, original)
    del torch.Tensor._allocator_tag


def active():
    return _rec is not None


def mark(name):
    """The allocations that follow count for the peak of phase `name` (a phase seen before continues its peak)."""
    rec = _rec
    if rec is None:
        return
    if name not in rec.phase_ids:
        rec.phase_ids[name] = len(rec.phase_names)
        rec.phase_names.append(name)
    rec.phase = name
    _library().vmm_debug_mark(rec.phase_ids[name])


@contextlib.contextmanager
def tag(name):
    """Tags the allocations made in the block (the innermost tag wins)."""
    tags = getattr(_local, "tags", None)
    if tags is None:
        tags = _local.tags = []
    tags.append(name)
    try:
        yield
    finally:
        tags.pop()


def name_models(models):
    """Names the modules of {name: model} by their paths, e.g. transformer.blocks.12.attn.to_q."""
    rec = _rec
    if rec is None:
        return
    for model_name, model in models.items():
        if isinstance(model, torch.nn.Module):
            rec.named_models.add(model_name)
            for path, module in model.named_modules():
                rec.module_names[id(module)] = f"{model_name}.{path}" if path else model_name


def reset():
    """New peaks and totals (between two generations); the allocations alive stay recorded."""
    rec = _rec
    if rec is None:
        return
    _library().vmm_debug_reset()
    rec.phase = None
    mark("start")


def _device():
    return torch.cuda.current_device()


def _snapshot(phase):
    lib, header = _library(), _Header()
    count = lib.vmm_debug_snapshot(_device(), phase, ctypes.byref(header), None, 0)
    buffer = (_Entry * count)()
    count = min(count, lib.vmm_debug_snapshot(_device(), phase, ctypes.byref(header), buffer, count))
    return header, list(buffer[:count])


_INTERNAL = (os.sep + "site-packages" + os.sep, os.path.join("mmgp", "offload.py"), os.path.join("mmgp", "allocator"), "<frozen")


def _frame_text(code, line):
    path = code.co_filename
    try:
        path = os.path.relpath(path)
    except ValueError:
        pass
    if path.startswith(".."):
        index = path.find("site-packages")
        path = path[index:] if index >= 0 else path
    return f"{path.replace(os.sep, '/')}:{line} in {code.co_name}"


_WEIGHT_LOADERS = {"to_gpu", "_take_slot", "_move_block", "copy_block", "cpu_to_gpu"}  # mmgp offload functions that allocate model weights


def _where(rec, stack_id):  # the innermost frame of the application's own code, or mmgp's when it loads model weights
    if stack_id < 0:
        return ""
    code, line = rec.stack_frames[stack_id][0]
    if code.co_name in _WEIGHT_LOADERS and code.co_filename.endswith(os.path.join("mmgp", "offload.py")):
        return _frame_text(code, line) + " (model weights loaded by mmgp)"
    for code, line in rec.stack_frames[stack_id]:
        if not any(part in code.co_filename for part in _INTERNAL):
            return _frame_text(code, line)
    return _frame_text(*rec.stack_frames[stack_id][0])


def _normalized(module):
    return re.sub(r"\.\d+(?=\.|/|$)", ".*", module)


def _groups(rec, entries, now):  # entries grouped by origin, the largest first
    groups = {}
    for entry in entries:
        module, region, stack_id = rec.origin_keys[entry.origin] if 0 <= entry.origin < len(rec.origin_keys) else ("", "", -1)
        name = rec.tags_by_seq.get(entry.seq, "")
        key = (_normalized(module), region, name, stack_id)
        group = groups.get(key)
        if group is None:
            group = groups[key] = {"mb": 0.0, "count": 0, "sizes_mb": [], "module": key[0], "tag": region, "tensor_tag": name, "stack": stack_id,
                                   "where": _where(rec, stack_id), "kinds": set(), "age_ms": [], "modules": set()}
        group["mb"] += entry.size / MB
        group["count"] += 1
        group["sizes_mb"].append(round(entry.size / MB, 1))
        group["kinds"].add(KINDS[entry.kind])
        group["age_ms"].append(round((now - entry.time) * 1000, 1))
        if len(group["modules"]) < 4:
            group["modules"].add(module)
    result = sorted(groups.values(), key=lambda group: -group["mb"])
    for group in result:
        group["mb"] = round(group["mb"], 1)
        group["sizes_mb"] = sorted(group["sizes_mb"], reverse=True)[:12]
        group["kinds"], group["modules"] = sorted(group["kinds"]), sorted(group["modules"])
        ages = group.pop("age_ms")
        group["age_ms"] = [min(ages), max(ages)]
    return result


def _phase_report(rec, header, entries, name):
    large = sum(entry.size for entry in entries)
    return {"name": name, "allocated_gb": round(header.allocated / GB, 3), "reserved_gb": round(header.reserved / GB, 3), "spilled_gb": round(header.spilled / GB, 3),
            "large_gb": round(large / GB, 3), "smaller_gb": round((header.allocated - large) / GB, 3), "time_s": round(header.time, 3),
            "request_mb": round(header.request / MB, 1) if header.request else None,
            "reserved_detail_gb": {key: round(getattr(header, key) / GB, 3) for key in ("chunks", "small_pool", "mid_pool", "graph_pools", "cached_ranges", "spare", "small_cached")},
            "groups": _groups(rec, entries, header.time)}


def report(path=None):
    """The report (a dict), also written to path (.json) with a summary (.md) when path is given."""
    rec = _rec
    if rec is None:
        raise RuntimeError("No VRAM debug recording in progress: debug.start() first.")
    lib = _library()
    phase_ids = (ctypes.c_int64 * 1024)()
    phases = []
    for phase in phase_ids[:lib.vmm_debug_phases(_device(), phase_ids, 1024)]:
        header, entries = _snapshot(phase)
        phases.append(_phase_report(rec, header, entries, rec.phase_names[phase] if phase < len(rec.phase_names) else str(phase)))
    header, entries = _snapshot(-1)
    oom = _phase_report(rec, header, entries, rec.phase_names[header.phase] if header.phase < len(rec.phase_names) else "") if header.time > 0 else None
    count = lib.vmm_debug_live(_device(), None, 0)
    buffer = (_Entry * count)()
    live_entries = list(buffer[:min(count, lib.vmm_debug_live(_device(), buffer, count))])
    totals = (ctypes.c_int64 * (6 * 65536))()
    merged = {}  # the origins of the same module path (all blocks) and allocation site together
    for row in range(min(65536, lib.vmm_debug_origins(_device(), totals, 65536))):
        origin, count, total, largest, freed, life_us = totals[6 * row: 6 * row + 6]
        module, region, stack_id = rec.origin_keys[origin] if 0 <= origin < len(rec.origin_keys) else ("", "", -1)
        where = _where(rec, stack_id)
        entry = merged.setdefault((_normalized(module), region, where), {"module": _normalized(module), "tag": region, "stack": stack_id, "where": where,
                                                                        "count": 0, "total": 0, "max": 0, "freed": 0, "life_us": 0})
        entry["count"] += count
        entry["total"] += total
        entry["max"] = max(entry["max"], largest)
        entry["freed"] += freed
        entry["life_us"] += life_us
    origins = [{"module": entry["module"], "tag": entry["tag"], "stack": entry["stack"], "where": entry["where"], "count": entry["count"],
                "total_gb": round(entry["total"] / GB, 3), "max_mb": round(entry["max"] / MB, 1), "freed": entry["freed"],
                "mean_life_ms": round(entry["life_us"] / entry["freed"] / 1000, 2) if entry["freed"] else None} for entry in merged.values()]
    origins.sort(key=lambda origin: -origin["total_gb"])
    live = _groups(rec, live_entries, time.perf_counter() - rec.t0)
    used = {group["stack"] for phase in phases + ([oom] if oom else []) for group in phase["groups"]} | {group["stack"] for group in live} | {origin["stack"] for origin in origins[:200]}
    result = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "threshold_mb": rec.min_mb, "phases": phases, "out_of_memory": oom, "live_now": live, "origins": origins[:200],
              "stacks": {str(stack_id): [_frame_text(code, line) for code, line in rec.stack_frames[stack_id][:40]] for stack_id in sorted(used) if stack_id >= 0}}
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as writer:
            json.dump(result, writer, indent=1)
        with open(os.path.splitext(path)[0] + ".md", "w", encoding="utf-8") as writer:
            writer.write(summary(result))
    return result


def _table(groups, top):
    lines = ["| MB | count | module | allocated at | tag | age at peak (ms) |", "|---:|---:|---|---|---|---|"]
    for group in groups[:top]:
        tag_text = " / ".join(filter(None, (group["tag"], group["tensor_tag"])))
        age = group["age_ms"]
        lines.append(f"| {group['mb']:.0f} | {group['count']} | {group['module'] or '-'} | {group['where'] or '-'} | {tag_text or '-'} | "
                     f"{age[0]:.0f}{'' if age[0] == age[1] else f'-{age[1]:.0f}'} |")
    rest = groups[top:]
    if rest:
        lines.append(f"| {sum(group['mb'] for group in rest):.0f} | {sum(group['count'] for group in rest)} | {len(rest)} more origins | | | |")
    return "\n".join(lines)


def _phase_summary(phase, title):
    detail = phase["reserved_detail_gb"]
    return (f"## {title}\n\nIn use {phase['allocated_gb']:.2f} GiB: {phase['large_gb']:.2f} in the tensors below, {phase['smaller_gb']:.2f} in smaller ones. "
            f"Reserved {phase['reserved_gb']:.2f} GiB (chunks {detail['chunks']:.2f}, of which cached ranges {detail['cached_ranges']:.2f} and spare "
            f"{detail['spare']:.2f}; small pool {detail['small_pool']:.2f}; mid-size pool {detail['mid_pool']:.2f}; graph pools {detail['graph_pools']:.2f}), "
            f"spilled {phase['spilled_gb']:.2f} GiB.\n\n{_table(phase['groups'], 25)}\n\n")


def summary(result):
    """Markdown summary of a report."""
    text = [f"# VRAM debug report, {result['created']}: allocations of {result['threshold_mb']} MB and more\n\n"]
    for phase in result["phases"]:
        text.append(_phase_summary(phase, f"Peak of phase {phase['name']} ({phase['time_s']:.1f} s)"))
    if result["out_of_memory"]:
        oom = result["out_of_memory"]
        text.append(_phase_summary(oom, f"Out of memory in phase {oom['name']} ({oom['time_s']:.1f} s), {oom['request_mb']} MB requested"))
    text.append("## Totals by origin (scratch buffers: many allocations, short lives)\n\n| allocations | total GiB | max MB | mean life (ms) | module | allocated at | tag |\n|---:|---:|---:|---:|---|---|---|\n")
    for origin in result["origins"][:25]:
        life = origin["mean_life_ms"]
        text.append(f"| {origin['count']} | {origin['total_gb']:.2f} | {origin['max_mb']:.0f} | {'-' if life is None else f'{life:.1f}'} | "
                    f"{origin['module'] or '-'} | {origin['where'] or '-'} | {origin['tag'] or '-'} |\n")
    text.append(f"\n## Recorded allocations alive when the report was written\n\n{_table(result['live_now'], 25)}\n\n")
    text.append("Stacks (innermost frame first) are in the JSON report under `stacks`, by the `stack` id of each group.\n")
    return "".join(text)


def diff(before, after):
    """Markdown comparison of two reports: the peaks of the phases, and the origins whose memory at the peak changed."""
    lines = ["# VRAM debug comparison\n"]
    phases_before = {phase["name"]: phase for phase in before["phases"]}
    for phase in after["phases"]:
        old = phases_before.get(phase["name"])
        if old is None:
            lines.append(f"## {phase['name']}: only in the second report, {phase['allocated_gb']:.2f} GiB\n")
            continue
        lines.append(f"## {phase['name']}: peak in use {old['allocated_gb']:.2f} -> {phase['allocated_gb']:.2f} GiB "
                     f"({phase['allocated_gb'] - old['allocated_gb']:+.2f}), reserved {old['reserved_gb']:.2f} -> {phase['reserved_gb']:.2f}\n")
        key = lambda group: (group["module"], re.sub(r":\d+ in ", " in ", group["where"]), group["tag"], group["tensor_tag"])  # line numbers move with edits
        sizes = {}
        for index, report_phase in enumerate((old, phase)):
            for group in report_phase["groups"]:
                sizes.setdefault(key(group), [0.0, 0.0])[index] += group["mb"]
        changes = sorted(sizes.items(), key=lambda item: -abs(item[1][1] - item[1][0]))
        lines.append("| MB before | MB after | change | module | allocated at |\n|---:|---:|---:|---|---|")
        for (module, where, region, name), (old_mb, new_mb) in changes[:20]:
            if abs(new_mb - old_mb) >= 1:
                lines.append(f"| {old_mb:.0f} | {new_mb:.0f} | {new_mb - old_mb:+.0f} | {module or '-'} | {where or '-'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _main(argv):
    if len(argv) == 2 and argv[0] == "summary":
        with open(argv[1], encoding="utf-8") as reader:
            print(summary(json.load(reader)))
    elif len(argv) == 3 and argv[0] == "diff":
        with open(argv[1], encoding="utf-8") as first, open(argv[2], encoding="utf-8") as second:
            print(diff(json.load(first), json.load(second)))
    else:
        print("usage: python -m mmgp.allocator.debug summary report.json | diff before.json after.json")


if __name__ == "__main__":
    _main(sys.argv[1:])
