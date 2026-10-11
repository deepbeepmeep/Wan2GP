# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional Windows hints for live read-only mappings: ordered pinning batches and asynchronous GPU look-ahead."""
import bisect
import collections
import ctypes
import mmap
import sys
import threading
import time
import warnings

from . import safetensors2


WINDOW = 256 * 2**20
BATCH = 64 * 2**20
MAX_RANGES = 128
MAX_JOBS = 8
FAST_RATE = 4 * 2**30
PROBE_INTERVAL = 2**30


class _Range(ctypes.Structure):
    _fields_ = [('address', ctypes.c_void_p), ('size', ctypes.c_size_t)]


def _windows_prefetch():
    if sys.platform != 'win32':
        return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    try:
        hint = kernel.PrefetchVirtualMemory
    except AttributeError:
        return None
    hint.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_Range), ctypes.c_ulong]
    hint.restype = ctypes.c_int
    def prefetch(ranges):
        array = (_Range * len(ranges))(*(_Range(address, size) for address, size in ranges))
        if not hint(ctypes.c_void_p(-1), len(array), array, 0):
            raise ctypes.WinError(ctypes.get_last_error())
    return prefetch


class MappedSources:
    """Mapping identities, not tensor identities: views work and replaced storage stops matching."""
    def __init__(self):
        self.maps = sorted((record['address'], record['address'] + record['size'], record['mmap'])
                           for tracker in list(safetensors2.mmm.values())
                           for record in tracker.get_active_maps().values() if record['read_only'])
        self.starts = [start for start, _, _ in self.maps]

    def plan(self, sources):
        # Sources are (address, byte count) in actual copy order. Metadata holds only weak references.
        spans, position = [], 0
        for address, size in sources:
            index = bisect.bisect_right(self.starts, address) - 1
            if size and index >= 0:
                _, end, ref = self.maps[index]
                if address + size <= end and ref() is not None:
                    spans.append((position, address, size, ref))
            position += size
        return (tuple(spans), position) if spans else None


class _Job:
    def __init__(self, owner, plan):
        self.owner = owner
        self.spans, self.size = plan
        self.cursor = 0
        self.completed = {}
        self.finished = False

    def complete(self, start, size):
        if not size:
            return
        with self.owner.condition:
            if self.finished or start + size <= self.cursor:
                return
            self.completed[start] = start + size
            while self.cursor in self.completed:
                self.cursor = self.completed.pop(self.cursor)
            self.owner.condition.notify()

    def finish(self):
        with self.owner.condition:
            self.finished = True
            self.cursor = self.size
            self.completed.clear()
            self.owner.condition.notify()


class ReadAhead:
    """Bounded hints without tensor buffers; GPU copies never wait on the background hint worker."""
    def __init__(self, prefetch):
        self.prefetch = prefetch
        self.mappings = MappedSources()
        self.condition = threading.Condition()
        self.jobs = collections.deque()
        self.pending = collections.deque()
        self.outstanding = self.peak_outstanding = 0
        self.closed = False
        self.thread = None
        self.fast_remaining = 0
        self.last_request = 0.0
        self.calls = self.requested_bytes = self.max_ranges = 0
        self.error = None

    @classmethod
    def create(cls, enabled):
        if not enabled:
            return None
        prefetch = _windows_prefetch()
        if prefetch is None:
            return None
        worker = cls(prefetch)
        return worker if worker.mappings.maps else None

    def schedule(self, plan):
        if plan is None or self.closed or self.error is not None:
            return None
        with self.condition:
            if self._skip_cached(plan[1]):
                return None
            if len(self.jobs) >= MAX_JOBS:
                return None
            job = _Job(self, plan)
            self.jobs.append(job)
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='mmgp_read_ahead', daemon=True)
                self.thread.start()
            self.condition.notify()
            return job

    def _skip_cached(self, size):
        now = time.monotonic()
        if now - self.last_request > 5:
            self.fast_remaining = 0
        if self.fast_remaining <= 0:
            return False
        self.fast_remaining -= size
        self.last_request = now
        return True

    def cancel(self, plan):
        """Stop queued hints for this plan, including jobs handed to a staging copy. An OS call already in progress can finish."""
        if plan is None:
            return
        with self.condition:
            jobs = set(self.jobs) | {job for job, _, _ in self.pending}
            for job in jobs:
                if job.spans is plan[0]:
                    job.finish()
            self._reap()

    def for_copy(self, job):
        # A queued job may have been submitted before an earlier probe found warm pages.
        # In that case use the original copier too, avoiding per-tensor progress bookkeeping.
        if job is not None:
            with self.condition:
                skip = self._skip_cached(job.size)
            if skip:
                job.finish()
                return None
        return job

    def hint_copy(self, sources):
        """Pinning submits a bounded batch only after its hint, before copy workers can fault it in."""
        if self.error is not None:
            return
        plan = self.mappings.plan(sources)
        if plan is None:
            return
        # The pinning copier already bounded the payload and descriptor count. Preserve that batch:
        # individually rounding adjacent tensors to pages would split a full batch into smaller I/O requests.
        ranges = [(address, address + size, id(ref)) for _, address, size, ref in plan[0]]
        refs = {id(ref): ref for _, _, _, ref in plan[0]}
        try:
            self._hint(ranges, refs)
        except OSError as error:
            self.error = str(error)
            warnings.warn(f'MMGP Read Ahead disabled after a Windows prefetch error: {error}', RuntimeWarning)

    def _reap(self):
        remaining = collections.deque()
        for job, end, size in self.pending:
            if job.finished or job.cursor >= end:
                self.outstanding -= size
            else:
                remaining.append((job, end, size))
        self.pending = remaining

    def _batches(self, job):
        ranges, owners, size, end = [], {}, 0, 0
        page = mmap.PAGESIZE
        for position, address, length, ref in job.spans:
            offset = 0
            while offset < length:
                # Page-aligned bounded pieces, including the final partial page in the budget.
                start = (address + offset) // page * page
                count = min(length - offset, BATCH - (address + offset - start))
                stop = (address + offset + count + page - 1) // page * page
                if ranges and (size + stop - start > BATCH or len(ranges) >= MAX_RANGES):
                    yield ranges, owners, size, end
                    ranges, owners, size = [], {}, 0
                ranges.append((start, stop, id(ref)))
                owners[id(ref)] = ref
                size += stop - start
                end = position + offset + count
                offset += count
        if ranges:
            yield ranges, owners, size, end

    def _hint(self, ranges, refs):
        # Hold mappings only during the native call. Never keep old tensors alive through metadata.
        owners = {key: ref() for key, ref in refs.items()}
        merged = []
        for start, stop, key in sorted(ranges):
            if owners[key] is None:
                continue
            if merged and merged[-1][2] == key and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop, key])
        if not merged:
            return
        size = sum(stop - start for start, stop, _ in merged)
        if len(merged) > 16 and size < len(merged) * 65536:
            return  # avoid hint overhead for batches dominated by scattered tiny scales
        now = time.monotonic()
        if now - self.last_request > 5:
            self.fast_remaining = 0  # reconsider after an idle period
        self.last_request = now
        if self.fast_remaining > 0:
            self.fast_remaining -= size
            return
        start = time.perf_counter()
        self.prefetch([(start, stop - start) for start, stop, _ in merged])
        elapsed = time.perf_counter() - start
        self.calls += 1
        self.requested_bytes += size
        self.max_ranges = max(self.max_ranges, len(merged))
        if size >= 8 * 2**20 and size / max(elapsed, 1e-9) >= FAST_RATE:
            self.fast_remaining = PROBE_INTERVAL

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.jobs)
                    if self.closed:
                        return
                    job = self.jobs[0]
                for ranges, refs, size, end in self._batches(job):
                    with self.condition:
                        while not self.closed and not job.finished:
                            self._reap()
                            if self.outstanding + size <= WINDOW:
                                break
                            self.condition.wait()
                        if self.closed or job.finished:
                            break
                        if job.cursor >= end:
                            continue  # already copied; a late hint is unnecessary
                        self.pending.append((job, end, size))
                        self.outstanding += size
                        self.peak_outstanding = max(self.peak_outstanding, self.outstanding)
                    self._hint(ranges, refs)
                with self.condition:
                    if self.jobs and self.jobs[0] is job:
                        self.jobs.popleft()
                    self._reap()
        except OSError as error:
            # Hints are optional OS work. A failed hint must never interrupt valid tensor copies.
            self.error = str(error)
            warnings.warn(f'MMGP Read Ahead disabled after a Windows prefetch error: {error}', RuntimeWarning)
        finally:
            with self.condition:
                self.jobs.clear()
                self.pending.clear()
                self.outstanding = 0

    def reset(self):
        with self.condition:
            for job in self.jobs:
                job.finished = True
            for job, _, _ in self.pending:
                job.finished = True
            self.jobs.clear()
            self.pending.clear()
            self.outstanding = 0
            self.fast_remaining = 0
            self.last_request = 0.0
            self.condition.notify()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        if self.thread is not None:
            self.thread.join()
        self.reset()
