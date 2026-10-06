# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Automatic VRAM preload of the models processed block by block (offload.all / offload.profile(..., autoPreload=True or a dict)):
each such model keeps in VRAM as many of its blocks as the VRAM left by its current workload allows, instead of a fixed preload. Needs the
MMGP Optimized VRAM Allocator (mmgp.allocator)."""
import dataclasses
import threading
import weakref

import torch

from . import allocator

ONE_MB = 1 << 20
MARGIN = 512 * ONE_MB  # left free beyond the measured peak: the allocator's headroom (256 MB) and its fragmentation
MARGIN_SHARE = 0.05  # and this share of the peak, for variations between the cycles of a workload
GROW_MIN = 512 * ONE_MB  # a plan grows by at least this much (or 2 blocks), not to change for small variations of the free VRAM
_process_measures = {}  # with autoPreload=True: (model class, model id, structure, workload) -> peak of the activations measured
_live = weakref.WeakSet()  # the instances that the allocator's callback serves


def _van_der_corput(n):
    x, f = 0.0, 0.5
    while n:
        if n & 1:
            x += f
        n >>= 1
        f /= 2
    return x


def even_order(count):
    """0..count-1 in an order whose every prefix is spread evenly over the range (van der Corput positions, nearest free index)."""
    order, free, n = [], list(range(count)), 0
    while free:
        target = _van_der_corput(n) * count
        best = min(free, key=lambda i: (abs(i - target), i))
        free.remove(best)
        order.append(best)
        n += 1
    return order


def _interleave(towers):
    """The blocks of the towers ([(name, size), ...] in the order they run), each tower in its even order, the towers sharing the ranks in
    proportion to their sizes."""
    orders = [[floors[i] for i in even_order(len(floors))] for floors in towers]
    totals = [sum(size for _, size in floors) or 1 for floors in towers]
    taken, positions, ranking = [0] * len(towers), [0] * len(towers), []
    while True:
        candidates = [t for t in range(len(towers)) if positions[t] < len(orders[t])]
        if not candidates:
            return ranking
        t = min(candidates, key=lambda t: ((taken[t] + orders[t][positions[t]][1]) / totals[t], t))
        name, size = orders[t][positions[t]]
        positions[t] += 1
        taken[t] += size
        ranking.append(name)


def ranked_blocks(towers, slow=()):
    """towers: [[(block name, size), ...] in the order they run, ...]. The block names in the order they are kept in VRAM: the slow ones first
    (not pinned), then the others, each group spread evenly across the towers in proportion to their sizes."""
    return [name for in_slow in (True, False) for name in _interleave([[(name, size) for name, size in floors if (name in slow) == in_slow] for floors in towers])]


def workload(args, kwargs):
    """Number of elements of the largest input tensor of a block (one level of lists, tuples, dicts and dataclasses): the workload of a
    cycle."""
    largest = 0
    for value in (*args, *kwargs.values()):
        if isinstance(value, (list, tuple)):
            items = value
        elif isinstance(value, dict):
            items = value.values()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):  # e.g. the arguments of each modality of LTX-2's blocks
            items = [getattr(value, field.name) for field in dataclasses.fields(value)]
        else:
            items = (value,)
        for item in items:
            if torch.is_tensor(item):
                largest = max(largest, item.numel())
    return largest


def _on_pressure(size, device):
    freed = 0
    for instance in list(_live):
        freed += instance.on_pressure(size - freed)
    return freed


class _Plan:
    """The blocks of a model and what it keeps in VRAM."""
    def __init__(self, model_id, model_class, towers, base):
        self.model_id, self.towers, self.base = model_id, towers, base
        self.sizes = {name: size for _, floors in towers for name, size in floors}
        self.ranking = ranked_blocks([floors for _, floors in towers])
        self.largest = max(self.sizes.values())
        self.structure = (model_class, model_id, len(self.sizes), sum(self.sizes.values()))  # another quantization or model is measured again
        self.first = towers[0][1][0][0]  # entering the first block of the first tower starts a cycle
        self.floor = 0  # number of ranked blocks preloaded for an unknown workload (the model's budget)
        self.pending = set()  # blocks of the plan that become preloaded when they have been transferred
        self.last = None  # block that ran last
        self.workload = None  # of the cycle running, None between cycles
        self.min_weights = 0  # least VRAM held by the model's weights during the cycle
        self.grew = False  # blocks were kept during the cycle: its peak is not a measure of the activations
        self.relink = False  # blocks were released by the pressure callback: to chain again with the streamed blocks
        self.slow = set()  # blocks not pinned: staged through RAM at each transfer, the first to keep in VRAM
        self.recoveries = None  # the allocator's recoveries when a cycle with blocks kept in VRAM started (None: no block kept)


class AutoPreload:
    def __init__(self, manager, model_ids, verbose_level=1, measures=None):
        """manager: the offload object, once its models are hooked and their budgets planned (tune_preloading). The preload of each model
        becomes the same number of ranked blocks, kept for the workloads not measured yet. measures: the dictionary that keeps the measures
        (None: kept for the process)."""
        self.manager, self.verbose = manager, verbose_level
        self.measures = _process_measures if measures is None else measures
        self.plans = {}
        self.thread = None  # the thread running the blocks, the only one that may change the plans
        self.dropping = False
        self.window = manager.prefetch_window if manager.prefetch_window is not None else 2  # fixed: the slots do not change with the plans
        for model_id in model_ids:
            plan = self._new_plan(model_id)
            if plan is not None:
                self.plans[model_id] = plan
                plan.floor = len(manager.preloaded_blocks_per_model.get(model_id, {}))
                self._set(plan, plan.ranking[:plan.floor])
        if self.plans:
            if not _live:
                allocator.set_pressure_callback(_on_pressure)
            _live.add(self)
            if self.verbose >= 1:
                print(f"Automatic VRAM preload of {', '.join(repr(model_id) for model_id in self.plans)}: the blocks kept in VRAM follow the VRAM left by each workload")

    def close(self):
        _live.discard(self)
        if not _live:
            allocator.set_pressure_callback(None)

    def after_pinning(self):
        """The RAM is pinned: the blocks that are not pinned are the first to keep in VRAM. Nothing is in VRAM yet."""
        m = self.manager
        for plan in self.plans.values():
            plan.slow = {name for name in plan.sizes if m._block_items(plan.model_id + "/" + name)[2] is None}
            plan.ranking = ranked_blocks([floors for _, floors in plan.towers], plan.slow)
            self._set(plan, plan.ranking[:plan.floor])

    def _new_plan(self, model_id):
        m = self.manager
        towers = []
        for tower_name in m.towers_per_model.get(model_id) or ():
            prefix = model_id + "/" + tower_name
            floors = sorted((int(name[len(prefix):]), name[len(model_id) + 1:], size) for name, size in m.blocks_of_modules_sizes.items() if name.startswith(prefix) and name[len(prefix):].isdigit())
            if floors:
                towers.append((tower_name, [(name, size) for _, name, size in floors]))
        if not towers:
            return None
        return _Plan(model_id, type(m.models[model_id]).__name__, towers, m.blocks_of_modules_sizes.get(model_id, 0))

    def _weights(self, plan, preloaded):
        """VRAM of the model's weights with these blocks preloaded: its base, the blocks and the slots of the blocks streamed."""
        streamed = len(preloaded) < len(plan.sizes)
        return plan.base + sum(plan.sizes[name] for name in preloaded) + ((self.window + 1) * plan.largest if streamed else 0)

    def _fit(self, plan, peak):
        """The longest prefix of the ranking that fits beside a peak of activations, with the base and the slots of the streamed blocks."""
        stats = allocator.stats()
        target = stats["reserved"] + allocator.room() - peak - MARGIN - int(MARGIN_SHARE * peak)
        if plan.base + sum(plan.sizes.values()) <= target:
            return list(plan.ranking)
        shuttle, used, count = (self.window + 1) * plan.largest, plan.base, 0
        for name in plan.ranking:
            if used + plan.sizes[name] + shuttle > target:
                break
            used += plan.sizes[name]
            count += 1
        return plan.ranking[:count]

    def own_buffer(self, model_id, block_name):
        """A block transferred into a buffer of its own rather than a shared slot: a block to keep in VRAM once transferred, or any block while
        no block of the towers is streamed (the slots, sized for the blocks of the towers, would only hold the small ones outside them)."""
        plan = self.plans.get(model_id)
        return plan is not None and (block_name in plan.pending or len(self.manager.preloaded_blocks_per_model[model_id]) + len(plan.pending) == len(plan.sizes))

    def keep(self, model_id, block_name):
        """Called when the block that ran last is about to be unloaded: True keeps it in VRAM, preloaded from now on (it was transferred
        into a buffer of its own; one that got a shared slot before it was planned is kept when it is transferred again)."""
        plan = self.plans.get(model_id)
        if plan is None or block_name not in plan.pending:
            return False
        m, entry = self.manager, model_id + "/" + block_name
        if any(slot[1] == entry for slot in m.block_slots.get(model_id, ())):
            return False
        plan.pending.discard(block_name)
        m.preloaded_blocks_per_model[model_id][block_name] = plan.sizes[block_name]
        if m.read_ahead is not None:
            m._cancel_read_ahead(entry)
        self._link(plan)
        self._complete_loras(model_id, entry)
        plan.grew = True
        return True

    def on_block(self, model_id, block_name, args, kwargs):
        """Called before each module of a block of the model runs: entering the first block starts a cycle."""
        plan = self.plans.get(model_id)
        if plan is None:
            return
        last, plan.last = plan.last, block_name
        if block_name != plan.first or last == block_name:
            return
        self.thread = threading.get_ident()
        self._end_cycle(plan)
        loaded = self.manager.loaded_blocks[model_id]
        if loaded is not None and self.keep(model_id, loaded):  # the last block streamed to grow a plan: no streamed block retires it any more
            self.manager.loaded_blocks[model_id] = None
        work = workload(args, kwargs)
        peak = self.measures.get((*plan.structure, work))
        wanted = plan.ranking[:plan.floor] if peak is None else self._fit(plan, peak)
        current = self.manager.preloaded_blocks_per_model[model_id]
        missing = [name for name in wanted if name not in current]
        dropped = [name for name in current if name not in wanted]
        if dropped or plan.relink:
            self._drop(plan, dropped)
        # small growths are skipped (variations of the free VRAM), not that of blocks not pinned (left out when the pressure callback stopped a
        # growth): each one streamed costs a copy through RAM at every step
        if plan.pending or sum(plan.sizes[name] for name in missing) >= max(GROW_MIN, 2 * plan.largest) or not plan.slow.isdisjoint(missing):
            plan.pending = set(missing)
        if self.verbose >= 1 and (dropped or plan.pending):
            measured = "the budget's preload until measured" if peak is None else f"{peak / ONE_MB:0.0f} MB of activations measured"
            print(f"Automatic VRAM preload of '{model_id}': {len(current) + len(plan.pending)} of its {len(plan.sizes)} blocks to keep in VRAM ({measured})")
        if len(current) + len(plan.pending) == len(plan.sizes):
            self._free_slots(plan)
        plan.workload, plan.grew = work, False
        plan.min_weights = self._weights(plan, current)
        plan.recoveries = allocator.stats()["recoveries"] if current or plan.pending else None
        allocator.mark_reset()

    def on_unload(self, model_ids):
        """The models left the VRAM: their measures are kept, their preload is back to their budget's for the next load."""
        for model_id in model_ids:
            plan = self.plans.get(model_id)
            if plan is not None:
                self._end_cycle(plan)
                plan.last, plan.pending = None, set()
                self._set(plan, plan.ranking[:plan.floor])

    def on_pressure(self, need):
        """The allocator found the VRAM short (without its lock): the plans stop growing, and the preloaded blocks of the models loaded leave
        the VRAM, the lowest ranked first, except the block running. They are streamed again when they run (chained again at the next
        cycle), and the measured peak of the workload grows by what was released: its next plans leave that memory free instead of growing
        back into the same shortage at every step."""
        m, freed = self.manager, 0
        if self.dropping or need <= 0 or threading.get_ident() != self.thread:
            return 0
        need += MARGIN
        for model_id in reversed(m.active_models_ids):
            plan = self.plans.get(model_id)
            if plan is None:
                continue
            plan.pending = set()
            preloaded = m.preloaded_blocks_per_model[model_id]
            victims = [name for name in reversed(plan.ranking) if name in preloaded and name != plan.last]
            if not victims:
                continue
            m.transfer_stream.wait_stream(torch.cuda.current_stream())  # their memory may still be read by the work queued
            released = 0
            for name in victims:
                if freed + released >= need:
                    break
                m.gpu_unload_blocks(model_id, name)
                del preloaded[name]
                if m.read_ahead is not None:
                    m._cancel_read_ahead(model_id + "/" + name)
                released += plan.sizes[name]
            freed += released
            plan.min_weights = min(plan.min_weights, self._weights(plan, preloaded))
            plan.relink = True  # back in the chain of the streamed blocks at the next cycle, not inside an allocation
            if plan.workload is not None:  # the plan was too large for this workload: its next plans leave the memory released free
                key = (*plan.structure, plan.workload)
                self.measures[key] = self.measures.get(key, 0) + released
            if freed >= need:
                break
        if freed:
            print(f"Automatic VRAM preload: VRAM short, {freed / ONE_MB:0.0f} MB of preloaded blocks released")
        return freed

    def _end_cycle(self, plan):
        if plan.workload is not None:
            key = (*plan.structure, plan.workload)
            if not plan.grew:
                self.measures[key] = max(self.measures.get(key, 0), allocator.mark_peak() - plan.min_weights)
            if plan.recoveries is not None and allocator.stats()["recoveries"] > plan.recoveries:
                # the allocator found the VRAM short and waited for other streams or synchronized the device to reuse memory, which stalls the
                # transfers (H3 at 720p under 11 GB: 14 instead of 12.6 s per step): the next plans leave a block more free
                self.measures[key] = self.measures.get(key, 0) + plan.largest
        plan.workload = None

    def _free_slots(self, plan):
        """No block of the towers is streamed any more: the VRAM slots of the model are freed, with the block loaded in one of them (a block
        outside the towers, loaded again into a buffer of its own: the slots are sized again for the blocks streamed when there are). They
        would otherwise hold VRAM, measured as activations. A slot still used by a block in flight is freed at the next cycle."""
        m, model_id = self.manager, plan.model_id
        slots = m.block_slots.get(model_id)
        if not slots:
            return
        m.transfer_stream.wait_stream(torch.cuda.current_stream())  # their memory may still be read by the work queued
        loaded = m.loaded_blocks[model_id]
        if loaded is not None and any(slot[1] == model_id + "/" + loaded for slot in slots):
            m.gpu_unload_blocks(model_id, loaded)
        for slot_no, slot in enumerate(slots):
            if slot[1] is None and slot[0] is not None:
                slot[0] = None
                m.slot_views = {key: views for key, views in m.slot_views.items() if key[1] != slot_no or not key[0].startswith(model_id + "/")}
        if all(slot[1] is None for slot in slots):
            del m.block_slots[model_id]

    def _set(self, plan, names):
        """Records the blocks preloaded and links the others (no VRAM change)."""
        m = self.manager
        m.preloaded_blocks_per_model[plan.model_id] = {name: plan.sizes[name] for name in names}
        m.planned_windows[plan.model_id] = self.window
        m.prefetch_windows.pop(plan.model_id, None)
        m.shuttle_sizes[plan.model_id] = plan.largest
        self._link(plan)

    def _link(self, plan):
        """Chains the streamed blocks as tune_preloading does: in the order they run, a single tower in a cycle (its last streamed block
        prefetches the first one), the preloaded blocks out of the chain."""
        m, model_id = self.manager, plan.model_id
        preloaded = m.preloaded_blocks_per_model[model_id]
        for entry, target in m.next_blocks_names.items():
            # a link learned while the block was streamed (e.g. from a block outside the towers that runs before it) would transfer it again
            # once it is preloaded: its parameters would then point to a buffer still being copied while it runs without waiting for it
            if target is not None and target.startswith(model_id + "/") and target[len(model_id) + 1:] in preloaded:
                m.next_blocks_names[entry] = None
        for _, floors in plan.towers:
            first = prev = None
            for name, _ in floors:
                entry = model_id + "/" + name
                m.learned_links.discard(entry)
                if name in preloaded:
                    m.next_blocks_names[entry] = m.prev_blocks_names[entry] = None
                    continue
                if prev is None:
                    first = entry
                else:
                    m.next_blocks_names[prev] = entry
                m.prev_blocks_names[entry] = prev
                prev = entry
            if prev is not None:
                m.next_blocks_names[prev] = None
                if len(plan.towers) == 1:
                    m.next_blocks_names[prev] = first
                    m.prev_blocks_names[first] = prev
                    m.learned_links.add(prev)

    def _complete_loras(self, model_id, entry):
        """A streamed block got the LoRA copies of the adapters active at this step: a preloaded block needs those of every step."""
        m = self.manager
        model = m.models[model_id]
        adapters, data = getattr(model, "_loras_active_adapters", None), getattr(model, "_loras_model_data", None)
        if adapters and data is not None:
            modules = {parent: data[parent] for parent, _, _, _, _ in m.blocks_of_modules[entry] if parent in data}
            if modules:
                m._move_loras(adapters, modules, True, model)

    def _drop(self, plan, names):
        """Preloaded blocks leave the VRAM at the start of a cycle and are streamed again (none: the blocks the pressure callback released
        are chained again)."""
        m, model_id = self.manager, plan.model_id
        preloaded = m.preloaded_blocks_per_model[model_id]
        self.dropping = True
        try:
            if m.read_ahead is not None:
                # Eviction inserts blocks into the copy order. Discard old predictions before rebuilding it.
                for entry in m.read_plans:
                    if entry.startswith(model_id + "/"):
                        m._cancel_read_ahead(entry)
            if model_id in m.active_models_ids:
                if m.ring is not None:  # the staging ring follows the chain of the streamed blocks: emptied before the chain changes
                    m.wait_prefetch()
                    torch.cuda.synchronize()
                    m.ring.reset()
                    for entry in [entry for entry in m.inflight if entry.startswith(model_id + "/")]:
                        m.gpu_unload_blocks(model_id, entry[len(model_id) + 1:])
                else:
                    m.transfer_stream.wait_stream(torch.cuda.current_stream())  # their memory may still be read by the work queued
                for name in names:
                    m.gpu_unload_blocks(model_id, name)
            for name in names:
                del preloaded[name]
            self._link(plan)
            plan.relink = False
        finally:
            self.dropping = False
