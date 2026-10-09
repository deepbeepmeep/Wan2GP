# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Timeline of MMGP's memory events for the debug modes (VRAM: debug.py, RAM: ram_debug.py), recorded while one of them runs: the models
loaded to the GPU, unloaded and released (with those still alive after a release), the RAM pinned and unpinned by model, the staging ring,
the releases of the MMGP RAM allocator's cache (explicit ones with their caller; the others: trims, RAM pressure, its cap or the VRAM
allocator), its fallbacks, and the RAM spilled by the VRAM allocator and its recoveries. Each event carries the process's RAM (private,
resident) and VRAM (allocated, reserved) at that time. A report shows the events since its previous one (`since`)."""
import os
import sys
import threading
import time
import weakref

GB, MB = 2 ** 30, 2 ** 20
SAMPLE_S = 0.1  # the allocators' counters are read at this interval
events = []  # oldest first: {"t" (perf_counter), "kind", "text", "ram_private", "ram_resident", "vram_allocated", "vram_reserved"}
registrations = {}  # pointer -> {"size", "cls" (pinned_mmgp / pinned_ring), "owner" (model id or "staging ring"), "t"}: RAM pinned while recording
releases = []  # MMGP releases: {"t", "models": {model id: {"ref": weak reference, "class", "files", "pinned": bytes pinned at the release}}}
_sources = weakref.WeakKeyDictionary()  # module -> the names of the checkpoint files that its model was loaded from (all its submodules)
owner_files = {}  # model id -> the checkpoint files of the model whose weights it pinned
_lock = threading.RLock()
_owner = None  # the model whose weights are being pinned (their blocks are registered by copy threads)
_users, _patched, _sampler, _stop = 0, [], None, None
_unpins = {}  # owner -> [blocks, bytes] unregistered and not reported yet: an event per owner and sampling interval
_explained = 0  # bytes given back by the explicit releases of the RAM allocator's cache since the counters were read


def _memory():
    from .ram_debug import counters
    import torch
    process = counters()
    values = {"ram_private": process["private"], "ram_resident": process["resident"]}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        values["vram_allocated"], values["vram_reserved"] = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    return values


def _flush_unpins():
    for owner, (count, size) in _unpins.items():
        events.append({"t": time.perf_counter(), "kind": "unpin", "text": f"{owner}: {count} pinned blocks unregistered ({size / GB:.2f} GiB)", **_memory(),
                       "owner": owner, "blocks": count, "bytes": size})
    _unpins.clear()


def record(kind, text, **values):
    """An event now (nothing when no debug mode records)."""
    if not _users:
        return
    entry = {"t": time.perf_counter(), "kind": kind, "text": text, **_memory(), **values}
    with _lock:
        _flush_unpins()
        events.append(entry)


def sources(model):
    """The names of the checkpoint files that the model (or the model it belongs to) was loaded from, while recording."""
    return list(_sources.get(model, ()))


def pinned_by_owner(owners=None):
    """{owner: [blocks, bytes]} of the RAM registered now (by the models in owners, or all)."""
    result = {}
    with _lock:
        for info in registrations.values():
            if owners is None or info["owner"] in owners:
                total = result.setdefault(info["owner"], [0, 0])
                total[0] += 1
                total[1] += info["size"]
    return result


def since(cursor):
    """The events after cursor (a position in events) and the new cursor."""
    with _lock:
        _flush_unpins()
        return list(events[cursor:]), len(events)


def _caller(depth=2):
    frame = sys._getframe(depth)
    return f"{frame.f_code.co_name} ({frame.f_code.co_filename.replace(chr(92), '/').rsplit('/', 1)[-1]}:{frame.f_lineno})"


def _install():
    import torch
    from mmgp import offload
    from . import debug, ram

    def patch(owner, name, make):
        original = getattr(owner, name)
        setattr(owner, name, make(original))
        _patched.append((owner, name, original))

    def load_model_data(original):
        def wrapped(model, file_path, *args, **kwargs):
            result = original(model, file_path, *args, **kwargs)
            names = [os.path.basename(str(path)) for path in (file_path if isinstance(file_path, (list, tuple)) else [file_path]) if path]
            files = list(dict.fromkeys(sources(model) + names))  # one list for the model and its submodules (MMGP may manage a submodule)
            for module in model.modules():
                _sources[module] = files
            return result
        return wrapped

    def gpu_load(original):
        def wrapped(self, model_id, *args, **kwargs):
            loading = model_id not in self.active_models_ids
            result = original(self, model_id, *args, **kwargs)
            if loading:
                record("load", f"{model_id} loaded to the GPU")
            return result
        return wrapped

    def unload_all(original):
        def wrapped(self, *args, keep=(), **kwargs):
            leaving = [model_id for model_id in self.active_models_ids if model_id not in keep]
            result = original(self, *args, keep=keep, **kwargs)
            if leaving:
                record("unload", f"{', '.join(leaving)} unloaded from the GPU")
            return result
        return wrapped

    def release(original):
        def wrapped(self):
            if self.models is None:
                return original(self)
            pinned = pinned_by_owner(set(self.models))
            models = {model_id: {"ref": weakref.ref(model), "class": type(model).__name__, "files": sources(model), "pinned": pinned.get(model_id, [0, 0])[1]}
                      for model_id, model in self.models.items()}
            record("release", f"MMGP release of {', '.join(models)} ({sum(info['pinned'] for info in models.values()) / GB:.2f} GiB pinned)")
            result = original(self)
            alive = [model_id for model_id, info in models.items() if info["ref"]() is not None]
            left = sum(size for _, size in pinned_by_owner(set(models)).values())
            record("released", f"MMGP release done: {('still alive: ' + ', '.join(alive)) if alive else 'no model alive'}; {left / GB:.2f} GiB still pinned")
            with _lock:
                releases.append({"t": time.perf_counter(), "models": models})
            return result
        return wrapped

    def pin_to_memory(original):
        def wrapped(model, model_id, *args, **kwargs):
            global _owner
            previous, _owner = _owner, model_id
            before = pinned_by_owner({model_id}).get(model_id, [0, 0])
            try:
                with debug.tag(f"pinned {model_id}"):
                    return original(model, model_id, *args, **kwargs)
            finally:
                _owner = previous
                after = pinned_by_owner({model_id}).get(model_id, [0, 0])
                owner_files[model_id] = files = sources(model)
                record("pin", f"{model_id}{' (' + ', '.join(files) + ')' if files else ''}: {(after[1] - before[1]) / GB:.2f} GiB pinned in {after[0] - before[0]} blocks")
        return wrapped

    def register(original):
        def wrapped(self):
            error = original(self)
            if int(error) == 0:
                ring = isinstance(self.buffer, torch.Tensor)  # the staging ring; the pinned weights are NumPy arrays
                with _lock:
                    registrations[self.pointer] = {"size": self.size, "cls": "pinned_ring" if ring else "pinned_mmgp",
                                                   "owner": "staging ring" if ring else _owner or "?", "t": time.perf_counter()}
                if ring:
                    record("ring", f"staging ring of {self.size / MB:.0f} MiB pinned")
            return error
        return wrapped

    def close(original):
        def wrapped(self):
            pointer = self.pointer
            closed = original(self)
            if closed:
                with _lock:
                    info = registrations.pop(pointer, None)
                    if info is not None and info["cls"] == "pinned_mmgp":
                        total = _unpins.setdefault(info["owner"], [0, 0])
                        total[0] += 1
                        total[1] += info["size"]
                if info is not None and info["cls"] == "pinned_ring":
                    record("ring", f"staging ring of {info['size'] / MB:.0f} MiB unpinned")
            return closed
        return wrapped

    def ram_release(original):
        def wrapped():
            global _explained
            before = ram.stats()["released_bytes"]
            original()
            released = ram.stats()["released_bytes"] - before
            with _lock:
                _explained += released
            if released:
                record("ram release", f"RAM allocator cache given back ({released / GB:.2f} GiB), by {_caller()}")
        return wrapped

    patch(offload, "load_model_data", load_model_data)
    patch(offload.offload, "gpu_load", gpu_load)
    patch(offload.offload, "unload_all", unload_all)
    patch(offload.offload, "release", release)
    patch(offload, "_pin_to_memory", pin_to_memory)
    patch(offload._HostRegistration, "register", register)
    patch(offload._HostRegistration, "close", close)
    patch(ram, "release", ram_release)


def _vram_counters():
    from . import _lib, stats
    return stats() if _lib is not None else None


def _sample(stop):
    """Events from the changes of the allocators' counters."""
    global _explained
    from . import ram
    ram_before, vram_before = ram.stats() if ram.active else None, _vram_counters()
    while not stop.wait(SAMPLE_S):
        try:
            if ram_before is not None:
                now = ram.stats()
                released, trims, fallbacks = (now[key] - ram_before[key] for key in ("released_bytes", "trims", "fallbacks"))
                with _lock:
                    released, _explained = released - _explained, 0
                if released >= MB:
                    record("ram release", f"RAM allocator gave back {released / GB:.2f} GiB of its cache ({trims} trims, RAM pressure, its cap or the VRAM allocator); "
                                          f"cached now {now['cached'] / GB:.2f} GiB")
                if fallbacks:
                    record("ram fallback", f"{fallbacks} CPU tensors left to PyTorch's allocator (the RAM allocator could not commit them)")
                ram_before = now
            if vram_before is not None:
                now = _vram_counters()
                for key, text in (("spilled", "spilled to pinned RAM"), ("driver_spilled", "spilled through the driver")):
                    if abs(now[key] - vram_before[key]) >= 64 * MB:
                        record("vram spill", f"VRAM allocator: {now[key] / GB:.2f} GiB {text} ({(now[key] - vram_before[key]) / GB:+.2f})")
                if now["recoveries"] > vram_before["recoveries"]:
                    record("vram recovery", f"VRAM allocator short of VRAM: {now['recoveries'] - vram_before['recoveries']} recoveries (waits for other streams or synchronizations)")
                vram_before = now
            with _lock:
                if _unpins:
                    _flush_unpins()
        except Exception as error:  # the sampler must never stop the workload
            print(f"[mmgp events] sampler error: {error}")


def start():
    """A debug mode starts: the events are recorded while at least one runs."""
    global _users, _sampler, _stop
    with _lock:
        _users += 1
        if _users > 1:
            return
        _install()
        _stop = threading.Event()
        _sampler = threading.Thread(target=_sample, args=(_stop,), name="mmgp_events", daemon=True)
        _sampler.start()


def stop():
    global _users
    with _lock:
        _users -= 1
        if _users:
            return
        _stop.set()
    _sampler.join()
    for owner, name, original in reversed(_patched):
        setattr(owner, name, original)
    _patched.clear()


def timeline_json(entries, top=150):
    """Markdown table of the events of a report (as_json)."""
    if not entries:
        return "No MMGP memory event.\n\n"
    merged = []  # the unregistrations of a model's blocks that follow each other: one row, from its first to its last time
    for event in entries:
        last = merged[-1] if merged else None
        if last is not None and event["kind"] == "unpin" and last["kind"] == "unpin" and event.get("owner") == last.get("owner"):
            blocks, gb = last["blocks"] + event["blocks"], last["gb"] + event["gb"]
            merged[-1] = {**event, "first_s": last.get("first_s", last["time_s"]), "blocks": blocks, "gb": gb,
                          "text": f"{event['owner']}: {blocks} pinned blocks unregistered ({gb:.2f} GiB) since {last.get('first_s', last['time_s']):.1f} s"}
        else:
            merged.append(event)
    shown = merged[-top:]
    rows = ["| time (s) | event | RAM private GiB | RAM resident GiB | VRAM allocated GiB | VRAM reserved GiB |", "|---:|---|---:|---:|---:|---:|"]
    for event in shown:
        rows.append(f"| {event['time_s']:.1f} | {event['kind']}: {event['text']} | " + " | ".join("-" if event.get(key) is None else f"{event[key]:.2f}"
                                                                                           for key in ("ram_private", "ram_resident", "vram_allocated", "vram_reserved")) + " |")
    entries = merged
    more = f"{len(entries) - len(shown)} earlier events in the JSON report.\n\n" if len(entries) > len(shown) else ""
    return "\n".join(rows) + "\n\n" + more


def as_json(entries, t0):
    """The events with times from t0 and sizes in GiB."""
    return [{"time_s": round(event["t"] - t0, 2), "kind": event["kind"], "text": event["text"],
             **{key: round(event[key] / GB, 3) for key in ("ram_private", "ram_resident", "vram_allocated", "vram_reserved") if key in event},
             **({"owner": event["owner"], "blocks": event["blocks"], "gb": round(event["bytes"] / GB, 3)} if "owner" in event else {})} for event in entries]
