"""Automatic CUDA graphs of the blocks of the towers (the repeated layers of a transformer): the host work of a block call (Python,
kernel launches) is recorded once and replayed, for models whose steps are short enough for the host to be the bottleneck.

Inside `scope()` (the denoising loop of a model; elsewhere blocks run as usual), each call of a block is keyed by the tensor objects of
its weights, its active LoRAs and their scalings, the shapes of its tensor arguments and extra inputs, its other arguments and the
registered global keys: the first call of a key runs eagerly (it compiles and tunes the kernels, makes the caches of the weights), then
is captured into a CUDA graph that the next calls replay. Blocks run eagerly, without captures, while the placement of their weights is not
settled (`ready()` False: e.g. an automatic preload measuring a new workload, which keeps other blocks in VRAM at the next step). A block thus has a graph per location of its weights (kept in VRAM, or each
slot of the swapped blocks), and a model loaded again captures its graphs again. The tensor arguments, extra inputs (tensors that come
and go with a block, e.g. its LoRA copies) and outputs go through static buffers shared by the graphs, the arguments a block modifies in
place are copied back when needed, and an output that is an argument is that argument. A block keeps its `MAX_KEYS` most recently used graphs (a new
resolution replaces them); all the graphs go with the model or `invalidate()`. With MMGP's VMM allocator, eager preparation and capture
share a pool. Suitable live arguments and first-call outputs supply the static storage; retained caller outputs are protected before
another call can overwrite them. A large argument that only its caller holds (a tower's hidden state, new at each step) moves into its
static buffer instead of being copied there: the caller's tensor takes the buffer's storage and frees its own, so the graphs keep no
second copy of it and the next blocks copy nothing; while the caller holds it, other calls use another buffer. Eager stages beside the
blocks (input preparation, output head) can run on the pool's idle memory with `pooled()`. Graph warmup is scoped to its manager's pool.
A consumed single-tensor list can transfer its input's execution lifetime: after its last execution view dies, later temporaries may use
its physical pages while every graph address stays fixed. Caller references, storage aliases and recorded stream consumers prevent this
transfer. Large outputs can reuse their warmup storage at the corresponding captured allocation; a changed allocation trace is recaptured
without that output plan. These plans add no replay-time mapping or synchronization, and do not participate in ordinary, non-graph
allocation.

Contract of a block call in a scope: the block reads only its weights, its tensor arguments (in lists, tuples, dicts and dataclasses)
and tensors it creates; its effects are its outputs and in-place updates of its tensor arguments (Python side effects are not replayed,
and the capture runs the Python of the block once more); no synchronization or CPU round trip. The capture checks the reads and raises
an error naming any other tensor.
"""
import contextlib
import dataclasses
import enum
import sys
import traceback
import weakref

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from . import runtime

MAX_KEYS = 8  # graphs kept per block, the least recently used dropped first: weight locations, CFG branches of other lengths, resolutions
MOVE_BYTES = 32 << 20  # arguments from this size move into their static buffer when only their caller holds them
_SCALARS = (bool, int, float, str, type(None), torch.dtype, torch.device)
_depth = 0
_recording = 0
_key_providers = []
_gates = []
_blocked = None  # why the blocks of the scope running are not graphed (a gate's reason), or None
_reported = set()
_capture_setups = []
_constants = []  # tensors that blocks may read besides their weights and arguments: kept for the life of the process
_capture_streams = {}
_blas_workspaces = set()
_registered = set()
_managers = []
_capturing = False  # a block's capture is running (its forward is recorded, nothing runs on the GPU)
_autotune_guarded = False
_static_storages = weakref.WeakValueDictionary()  # storage address -> a static buffer of some graph manager (`_mm_graphs`)


def _guard_triton_autotune():
    """Triton's autotuners (@triton.autotune) benchmark their configurations on a cache miss: inside a capture, the benchmark launches
    would be recorded in the graph and CUDA refuses its timing synchronizations. The eager call that precedes each capture tunes the keys
    a block uses (sizes and dtypes); a miss that still happens in a capture (a key that differs there) raises an error naming the kernel."""
    global _autotune_guarded
    _autotune_guarded = True
    try:
        from triton.runtime import autotuner
    except ImportError:
        return
    bench = autotuner.Autotuner._bench

    def guarded_bench(self, *args, **kwargs):
        if _capturing:
            fn = getattr(self, "base_fn", self.fn)
            raise RuntimeError(f"the Triton kernel {getattr(fn, '__name__', fn)} would tune itself (@triton.autotune) during a CUDA graph capture, "
                               f"for a key the eager call before the capture did not use")
        return bench(self, *args, **kwargs)

    autotuner.Autotuner._bench = guarded_bench


@contextlib.contextmanager
def scope(enabled=True):
    """Block calls in this scope use CUDA graphs when the mode is on for their model (a no-op otherwise)."""
    global _depth, _blocked
    if enabled and _depth == 0:
        _blocked = next((reason for reason in (gate() for gate in _gates) if reason), None)
        if _blocked is not None and _blocked not in _reported:
            _reported.add(_blocked)
            print(f"[MMGP] CUDA graphs are off: {_blocked}.")
    _depth += bool(enabled)
    try:
        yield
    finally:
        _depth -= bool(enabled)


def recording():
    """True while a block of a graphed model runs to be captured (its first, eager call and its capture): a kernel launched outside
    PyTorch's current stream (e.g. on the legacy default stream) would be missing from its graph, so code with such kernels uses others."""
    return _recording > 0


def _graphs(module):
    owner = runtime._owner(module)
    if owner is None:
        return None
    manager, model_id = owner
    return manager.cuda_graphs.get(model_id)


def is_enabled(module):
    """Whether a managed module can use graphs in the current scope.

    Includes mode selection, registered gates and weight-residency readiness.
    Use only to skip preparation specific to a graph stage; run() also enforces
    these conditions. Unmanaged/released modules return False.
    """
    graphs = _graphs(module) if _depth else None
    return graphs is not None and _blocked is None and graphs.disabled is None and graphs.ready()


def run(module, *args, owner=None, method="forward", enabled=True, **kwargs):
    """Execute an extra stage using its owner's existing graph cache and lifetime.

    Call inside scope(). An unregistered stage may supply a managed owner whose
    weights it uses; its parameters need not be registered a second time. The
    selected method must satisfy the same graph contract as a tower block.
    Outside a scope, with enabled=False, or without an active graph manager,
    execute the method normally. Gates/readiness and cache invalidation/release
    remain MMGP-owned. PyTorch compilation does not enable these CUDA graphs.
    A stage may span several blocks of its owner (e.g. a whole decoding step) only while
    runtime.weights_resident(owner): graphed blocks it calls are then captured inline.
    """
    forward = getattr(module, method)
    graphs = _graphs(module if owner is None else owner) if enabled and _depth else None
    return forward(*args, **kwargs) if graphs is None else graphs.call(module, forward, args, kwargs)


def pooled(owner, fn, *args, **kwargs):
    """Run an eager stage beside the graphed blocks of `owner`'s model (its input preparation, its output head) on the idle memory of the
    model's graph pool, where its temporaries would otherwise add to the pool's memory.

    Call inside scope(); fn runs normally while the model's graphs are off. The stage keeps nothing but its
    outputs (no cache filled, no tensor stored elsewhere: checked). Its tensor outputs leave the pool before any graph replays: a
    large one (a hidden state) moves into the free static buffer of its signature, which the next block then reads without a copy,
    the others move to ordinary memory.
    """
    from . import allocator
    graphs = _graphs(owner) if _depth and not _recording and allocator.active == "vmm" else None
    if graphs is None or _blocked is not None or graphs.disabled is not None or not graphs.ready():
        return fn(*args, **kwargs)
    return graphs.pooled(fn, args, kwargs)


def register_key(provider):
    """provider() -> hashable: a global state that changes what blocks compute (attention mode, kernel options), part of every key."""
    if provider not in _key_providers:
        _key_providers.append(provider)


def register_gate(gate):
    """gate() -> None, or the reason why the blocks must run eagerly (e.g. a kernel that cannot be captured): checked when a scope starts."""
    if gate not in _gates:
        _gates.append(gate)


def register_constant(tensor):
    """A tensor that blocks may read and that is never freed nor replaced (a process-wide cache, e.g. a rotation matrix): graphs read it
    at its address."""
    from . import allocator
    if allocator.in_graph_warmup():
        if _capturing:
            raise RuntimeError("A persistent CUDA graph constant must be prepared during warmup, before capture")
        with allocator.normal_allocations():
            tensor = tensor.clone()
    _constants.append(tensor)
    return tensor


def register_capture_setup(setup):
    """setup(stream) is called before each capture on the capture stream, the one stream of the captures of a device: it prepares what a
    library would allocate there on first use, which a capture cannot do (per-stream workspaces), and returns at once when it is ready."""
    if setup not in _capture_setups:
        _capture_setups.append(setup)


def _prepare_blas_stream(stream):
    # PyTorch owns these workspaces per handle/stream, beyond the life of a model's graphs. Initialize both cuBLAS and cuBLASLt
    # before warmup/capture can put them in graph scratch (which would also pin that entire pool after the model is released).
    from . import allocator
    with allocator.normal_allocations(), torch.cuda.stream(stream), torch.inference_mode():
        key = (torch.cuda.current_blas_handle(), stream.cuda_stream)
        if key not in _blas_workspaces:
            matrix = torch.zeros((16, 16), dtype=torch.float16, device=stream.device)
            torch.addmm(matrix[0], matrix, matrix)
            _blas_workspaces.add(key)


def _capture_stream(device):
    stream = _capture_streams.get(device, None)
    if stream is None:
        stream = _capture_streams[device] = torch.cuda.Stream(device)
    _prepare_blas_stream(stream)
    for setup in _capture_setups:
        setup(stream)
    return stream


def invalidate():
    """Drops every graph, e.g. after a change of the settings that the graphs depend on."""
    for manager in _managers:
        manager.clear()


def release(managers):
    """Drops the graphs of these managers for good (their offload object is released)."""
    for manager in list(managers):
        manager.clear()
        if manager in _managers:
            _managers.remove(manager)


def _register_dataclass(cls):
    fields = [field.name for field in dataclasses.fields(cls)]
    pytree.register_pytree_node(cls, lambda obj: ([getattr(obj, name) for name in fields], None),
                                lambda values, _: cls(**dict(zip(fields, values))), serialized_type_name=f"{cls.__module__}.{cls.__qualname__}")
    _registered.add(cls)


class _OverlappingArgument(Exception):
    def __init__(self, index):
        self.index = index


def _argument_path(tree, target, path):
    """Where `target` sits in a block call's arguments (lists, tuples, dicts, dataclasses), for error messages."""
    if tree is target:
        return path
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        items = [(f".{field.name}", getattr(tree, field.name)) for field in dataclasses.fields(tree)]
    elif isinstance(tree, dict):
        items = [(f"[{key!r}]", value) for key, value in tree.items()]
    elif isinstance(tree, (list, tuple)):
        items = [(f"[{index}]", value) for index, value in enumerate(tree)]
    else:
        return None
    return next((found for suffix, value in items if (found := _argument_path(value, target, path + suffix)) is not None), None)


def _flatten(tree):
    while True:
        leaves, spec = pytree.tree_flatten(tree)
        new = {type(leaf) for leaf in leaves if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type) and type(leaf) not in _registered}
        if not new:
            return leaves, spec
        for cls in new:
            _register_dataclass(cls)


def _signature(leaves):
    signature = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            signature.append((tuple(leaf.shape), leaf.stride(), leaf.dtype, leaf.device))
        elif isinstance(leaf, _SCALARS) or isinstance(leaf, enum.Enum):
            signature.append(leaf)
        elif isinstance(leaf, slice):
            signature.append(("slice", leaf.start, leaf.stop, leaf.step))
        else:
            raise TypeError(f"CUDA graph block call: an argument of type {type(leaf).__name__} is not supported, pass its tensors as arguments")
    return tuple(signature)


def _storage(t):
    return t.untyped_storage().data_ptr()


def _same_buffer(a, b):
    return b is not None and a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()


def _unshared(tensor):
    # The static TensorImpl and this temporary Storage wrapper are the only owners. Protect detached views and raw Storage holders too.
    storage = tensor.untyped_storage()
    return torch._C._storage_Use_Count(storage._cdata) == 2 and sys.getrefcount(storage) == 3


def _adoptable(tensor, owner):
    # Another manager's static buffer (an output it returned) is never adopted: sharing its storage would keep that output retained,
    # and its graphs would be evicted and captured again at each of their calls.
    from . import allocator
    static = _static_storages.get(_storage(tensor)) if tensor.is_cuda else None
    return (type(tensor) is torch.Tensor and tensor.is_cuda and not tensor.requires_grad and tensor.storage_offset() == 0 and
            torch._debug_has_internal_overlap(tensor) == 0 and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size() and
            allocator.owns_tensor(tensor) and (static is None or static._mm_graphs() is owner))


def _movable(leaf, static):
    # The caller's tensor alone holds its storage (no view, no other holder), and the static buffer is an allocation of the VMM allocator,
    # which orders its reuse after the streams that used it.
    from . import allocator
    return (allocator.active == "vmm" and type(leaf) is torch.Tensor and not leaf.requires_grad and leaf.numel() * leaf.element_size() >= MOVE_BYTES
            and (torch.is_inference_mode_enabled() or not leaf.is_inference()) and _unshared(leaf) and allocator.owns_tensor(static))


def _buffer_alias(tensor):
    # A non-inference TensorImpl tracks in-place updates during capture without allocating another activation.
    with torch.inference_mode(False):
        return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride())


class _ReadCheck(TorchDispatchMode):
    """Records, during a capture, the CUDA tensors read by the operations that are neither allowed nor created by the capture."""

    def __init__(self, allowed):
        super().__init__()
        self.allowed, self.unknown = allowed, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for leaf in pytree.tree_leaves((args, kwargs)):
            if type(leaf) is torch.Tensor and leaf.is_cuda and _storage(leaf) not in self.allowed and len(self.unknown) < 8:
                self.unknown.append(f"{func} reads a tensor {tuple(leaf.shape)} {leaf.dtype}\n" + "".join(traceback.format_stack(limit=12)[:-2]))
        out = func(*args, **kwargs)
        for leaf in pytree.tree_leaves(out):
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
                self.allowed.add(_storage(leaf))
        return out


class _Graph:
    __slots__ = ("graph", "positions", "static", "extra_static", "static_key", "mutated", "out_leaves", "out_spec", "outputs", "lifetimes", "emptied")


def _handoffs(args, kwargs):
    """Single-tensor lists can transfer ownership to a block. Other argument containers retain their normal lifetime."""
    found, seen = {}, set()
    pending = [args, kwargs]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], torch.Tensor):
            found[id(value[0])] = value
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in dataclasses.fields(value))
    return found


def _execute_lifetimes(forward, args, kwargs, positions, leaves):
    """Plan consumed input storage during warmup/capture; no operation dispatch interception or replay-time remapping."""
    if not positions:
        return forward(*args, **kwargs), ()
    from . import allocator
    containers = _handoffs(args, kwargs)
    bindings = []
    released = []
    try:
        for index in positions:
            source = leaves[index]
            container = containers.get(id(source))
            if container is None:
                continue
            view, token = allocator.lifetime_view(source)
            if token is not None:
                bindings.append((index, container, source, weakref.ref(view), token, view._version))
                container[0] = view
            del view
        out = forward(*args, **kwargs)
        # Rebind returned input views to the ordinary owner. Escaped results must own their storage beyond this execution.
        flat, spec = _flatten(out)
        by_storage = {_storage(source): source for _, _, source, _, _, _ in bindings}
        for index, value in enumerate(flat):
            if isinstance(value, torch.Tensor) and (source := by_storage.get(_storage(value))) is not None:
                if _same_buffer(source, value) and source.dtype == value.dtype:
                    flat[index] = source
                else:
                    with torch.inference_mode(False):
                        flat[index] = torch.empty(0, dtype=value.dtype, device=value.device).set_(
                            source.untyped_storage(), value.storage_offset(), value.shape, value.stride())
        out = pytree.tree_unflatten(flat, spec)
    finally:
        for index, container, source, ref, token, version in bindings:
            if allocator.end_lifetime(token):
                released.append(index)
            view = ref()
            if view is not None:
                if view._version != version:
                    torch.autograd.graph.increment_version(source)
                if len(container) == 1 and container[0] is view:
                    container[0] = source
    return out, tuple(released)


class BlockGraphs:
    """CUDA graphs of the tower blocks of one model."""

    def __init__(self, name, weights_key, weight_tensors, extra_inputs, ready):
        # extra_inputs(block) -> [(list, index)]: tensors read by the block that are neither weights nor arguments, read by its graphs from
        # static buffers (their list items are these buffers during a capture); ready() -> False while the weights may move
        self.name, self.weights_key, self.weight_tensors, self.extra_inputs, self.ready = name, weights_key, weight_tensors, extra_inputs, ready
        self.disabled = None  # the reason why this model's blocks run eagerly (e.g. weights whose kernels cannot be captured), or None
        self.reported = False
        self.checked = set()  # (block, method, signature) whose capture checked the reads
        self.clear()
        _managers.append(self)

    def clear(self):
        # entries: key -> graph; keys: block -> its keys, least recently used first; static: buffers shared by the graphs
        self.entries, self.keys, self.static, self.pool = {}, {}, {}, None
        self.static_by_buffer, self.static_by_base = {}, {}
        self.borrowed = set()
        self._tensor_refs = {}
        _blas_workspaces.clear()  # library caches may be released between models too; prepare them again before a new capture

    def wrap(self, module, forward):
        def graphed_forward(*args, **kwargs):
            if _depth == 0:
                return forward(*args, **kwargs)
            return self.call(module, forward, args, kwargs)
        return graphed_forward

    def call(self, module, forward, args, kwargs):
        global _recording
        # A block called by a stage being prepared or captured (a stage spanning several blocks) runs inside that stage's graph.
        if _recording or _blocked is not None or self.disabled is not None or not self.ready():
            return forward(*args, **kwargs)
        leaves, spec = _flatten((args, kwargs))
        signature = _signature(leaves)
        extras = self.extra_inputs(module)
        extra_signature = _signature([holder[index] for holder, index in extras])
        positions = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        input_signatures = [signature[index] for index in positions] + list(extra_signature)
        try:
            static_key, buffers = self._buffers(input_signatures, [leaves[index] for index in positions] + [None] * len(extras), arguments=True)
        except _OverlappingArgument as overlap:
            leaf = leaves[positions[overlap.index]]
            name = _argument_path(args, leaf, "args") or _argument_path(kwargs, leaf, "kwargs")
            raise ValueError(f"CUDA graph block call of '{self.name}' ({type(module).__name__}): argument {name} is a view whose elements share memory "
                             f"(shape {tuple(leaf.shape)}, stride {leaf.stride()}, e.g. from expand), which a graph's static buffer cannot hold. Pass the "
                             f"unexpanded tensor and expand it inside the block's forward (no cost), or make it contiguous before the call.") from None
        method = getattr(forward, "__func__", forward)
        key = (id(module), spec, signature, extra_signature, self.weights_key(module), tuple(provider() for provider in _key_providers), tuple(static_key), method)
        keys = self.keys.setdefault(id(module), {})
        entry = self.entries.get(key, None)
        lifetimes = None
        if entry is not None and entry.lifetimes:
            lifetimes = self._consumable(args, kwargs, leaves, positions, buffers)
            if not set(entry.lifetimes).issubset(lifetimes):
                key = (*key, "retained-inputs")
                entry = self.entries.get(key)
                lifetimes = ()
        retained_output = entry is not None and any(k in self.borrowed and not _unshared(self.static[k]) for k in entry.static_key[len(buffers):])
        if retained_output:
            self._evict(key, static_key)
            entry = None
        if entry is None:  # first call of this key: eager (it compiles and tunes the kernels, makes the caches of the weights), then captured
            if lifetimes is None:
                lifetimes = self._consumable(args, kwargs, leaves, positions, buffers)
            if len(keys) >= MAX_KEYS:
                self._evict(next(iter(keys)), static_key)
            for ordinal, index in enumerate(positions):  # before the warmup, which then runs on the moved arguments
                if not _same_buffer(buffers[ordinal], leaves[index]) and _movable(leaves[index], buffers[ordinal]):
                    self._move(static_key[ordinal], leaves[index])
            _recording += 1
            try:
                from . import allocator
                if self.pool is None:
                    self.pool = torch.cuda.graph_pool_handle()
                _prepare_blas_stream(torch.cuda.current_stream())
                with allocator.graph_warmup(self.pool):
                    plan = allocator.OutputPlan(self.pool)
                    try:
                        out, consumed = _execute_lifetimes(forward, args, kwargs, lifetimes, leaves)
                        plan.pause()
                        if consumed:
                            # An escaped warmup result must not retain pages loaned by a static input: the preceding graph may
                            # write that input again before this graph can protect its retained outputs. Stabilize only overlapping
                            # results, after input claims were restored, so capture copies to independent output storage as usual.
                            flat, out_spec = _flatten(out)
                            for index, value in enumerate(flat):
                                if isinstance(value, torch.Tensor) and value.is_cuda and any(
                                        allocator.graph_storage_overlaps(value, leaves[position]) for position in consumed):
                                    flat[index] = value.clone()
                            out = pytree.tree_unflatten(flat, out_spec)
                            value = None
                            del flat
                        # Returned views keep execution storage alive: capture them with the ordinary owner's version counter.
                        self._capture_lifetimes, self._capture_output_plan = consumed, plan
                        entry = self._capture(module, forward, leaves, spec, signature, extras, extra_signature, out, static_key, buffers, not retained_output)
                        if entry is None:  # a different allocation trace must not replay the proposed aliases
                            self._capture_output_plan = None
                            entry = self._capture(module, forward, leaves, spec, signature, extras, extra_signature, out, static_key, buffers, not retained_output)
                        self.entries[key] = entry
                    finally:
                        self.__dict__.pop('_capture_lifetimes', None)
                        self.__dict__.pop('_capture_output_plan', None)
                        plan.close()
            finally:
                _recording -= 1
                self._prune()
            keys[key] = None
            return out
        keys[key] = keys.pop(key)  # most recently used
        result = self._replay(entry, leaves, extras)
        if entry.emptied:
            handoffs = _handoffs(args, kwargs)
            for index in entry.emptied:
                container = handoffs.get(id(leaves[index]))
                if container is not None:
                    container.clear()
        return result

    def _move(self, key, leaf):
        # The caller's tensor takes the storage of its static buffer, which then serves no other call while the caller holds it (borrowed).
        static = self.static[key]
        static.copy_(leaf)
        leaf.record_stream(torch.cuda.current_stream())
        leaf.set_(static.untyped_storage(), static.storage_offset(), static.shape, static.stride())
        self.borrowed.add(key)

    def _free_static(self, leaf):
        # the static buffer that a block call would give this tensor (first of its signature), when no caller holds it
        for key in self.static_by_base.get((_signature((leaf,))[0], 0), ()):
            if (key not in self.borrowed or _unshared(self.static[key])) and _movable(leaf, self.static[key]):
                return key
        return None

    def pooled(self, fn, args, kwargs):
        from . import allocator
        if self.pool is None:  # a stage before the first captures: its temporaries give the pool its first memory, which they would keep from it
            self.pool = torch.cuda.graph_pool_handle()
        live = allocator.pool_live(self.pool)
        with allocator.graph_warmup(self.pool):
            out = fn(*args, **kwargs)
        holders = {}
        for leaf in _flatten(out)[0]:
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda and allocator.in_pool(leaf, self.pool):
                holders.setdefault(_storage(leaf), {})[id(leaf)] = leaf
        for tensors in holders.values():
            tensors = list(tensors.values())
            leaf = tensors[0]
            key = self._free_static(leaf) if len(tensors) == 1 else None
            if key is not None:
                if key in self.borrowed:
                    allocator.prepare_reuse(self.static[key])
                self._move(key, leaf)
                continue
            storage = leaf.untyped_storage()
            if torch._C._storage_Use_Count(storage._cdata) != len(tensors) + 1:
                raise RuntimeError(f"A pooled stage of '{self.name}' returned a tensor whose graph-pool storage is also held elsewhere (a view kept by the "
                                   f"stage, a cache): the next graph replay would overwrite it.")
            copy = torch.UntypedStorage(storage.nbytes(), device=leaf.device)
            copy.copy_(storage)
            for tensor in tensors:
                tensor.set_(copy, tensor.storage_offset(), tensor.shape, tensor.stride())
            del storage
        left = allocator.pool_live(self.pool) - live
        if left:
            raise RuntimeError(f"A pooled stage of '{self.name}' kept {left} allocation(s) in the graph pool beyond its outputs (a cache filled or a tensor "
                               f"stored elsewhere), which the next graph replay would overwrite: fill such caches outside the stage.")
        return out

    def _consumable(self, args, kwargs, leaves, positions, buffers):
        from . import allocator
        if allocator.active != "vmm":
            return ()
        handoffs = _handoffs(args, kwargs)
        eligible = []
        for ordinal, index in enumerate(positions):
            source, buffer = leaves[index], buffers[ordinal]
            if (id(source) not in handoffs or not source.is_cuda or source.numel() * source.element_size() < 32 * 1024 * 1024 or
                    not _same_buffer(source, buffer) or sum(value is source for value in leaves) != 1):
                continue
            # Account only for MMGP's references, this call's flattened arguments/buffers, and the transferring list.
            # Any additional Python owner or storage/view owner keeps the input protected.
            expected = self._tensor_refs.get(id(source), 0) + 4 + sum(value is source for value in buffers) + int(source is buffer)
            if sys.getrefcount(source) != expected or source._use_count() != 1:
                continue
            storage = source.untyped_storage()
            if (source.storage_offset() != 0 or storage.nbytes() != source.numel() * source.element_size() or
                    torch._C._storage_Use_Count(storage._cdata) != 2 + int(source is not buffer) or sys.getrefcount(storage) != 3):
                continue
            if not allocator.lifetime_available(source):
                continue
            eligible.append(index)
        return tuple(eligible)

    def _outputs(self, leaves, out):
        # per output leaf: ("argument", index of the argument it is), its signature, or None if it is not a tensor
        by_pointer = {leaf.data_ptr(): (index, leaf) for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)}
        outputs = []
        for leaf in _flatten(out)[0]:  # an output that is an argument (its storage, shape and strides) is returned as the caller's tensor
            if isinstance(leaf, torch.Tensor):
                index, argument = by_pointer.get(leaf.data_ptr(), (None, None))
                same = argument is not None and leaf.dtype == argument.dtype and leaf.shape == argument.shape and leaf.stride() == argument.stride()
                outputs.append(("argument", index) if same else _signature((leaf,))[0])
            else:
                outputs.append(None)
        return outputs

    def _evict(self, key, keep=()):
        del self.keys[key[0]][key]
        del self.entries[key]
        self._prune(keep)

    def _prune(self, keep=()):
        used = set(keep) | {k for entry in self.entries.values() for k in entry.static_key}
        if any(k not in used for k in self.static):
            self.static = {k: buffer for k, buffer in self.static.items() if k in used}
            self.static_by_buffer, self.static_by_base = {}, {}
            for key, buffer in self.static.items():
                self._index_buffer(key, buffer)
        self.borrowed.intersection_update(used)
        refs = {}
        for buffer in self.static.values():
            refs[id(buffer)] = refs.get(id(buffer), 0) + 1
        for entry in self.entries.values():
            for buffer in (*entry.static, *entry.extra_static, *entry.outputs):
                if isinstance(buffer, torch.Tensor):
                    refs[id(buffer)] = refs.get(id(buffer), 0) + 1
        self._tensor_refs = refs

    def _index_buffer(self, key, buffer):
        self.static_by_buffer.setdefault((key[0], buffer.data_ptr()), []).append(key)
        self.static_by_base.setdefault(key[:2], []).append(key)

    def _buffers(self, signatures, sources, counts=None, arguments=False):
        # static buffers, shared by the graphs (they run one after the other) for each signature and occurrence in a call; not inference
        # tensors, whose version counters show the in-place updates of the arguments. An argument whose elements share memory (an expanded
        # view) cannot be copied into a buffer of its layout: arguments=True reports it (checked only when a buffer is created).
        counts, keys, buffers = {} if counts is None else counts, [], []
        for signature, source in zip(signatures, sources):
            base = (signature, counts.setdefault(signature, 0))
            counts[signature] += 1
            # Prefix tensors can keep hundreds of adopted buffers alive. A replay usually names an existing buffer exactly;
            # look it up directly instead of scanning every other layer's buffers for every argument.
            matching = None if source is None else next((k for k in self.static_by_buffer.get((signature, source.data_ptr()), ())
                                                        if _same_buffer(self.static[k], source)), None)
            if matching is not None:
                key = matching
            else:
                candidates = [k for k in self.static_by_base.get(base, ()) if k not in keys]
                key = next((k for k in candidates if k not in self.borrowed or _unshared(self.static[k])), None)
            if key in self.borrowed and matching is None:
                from . import allocator
                allocator.prepare_reuse(self.static[key])
            if key is None:
                key = (*base, 1 + max((k[2] for k in candidates), default=-1))
                shape, stride, dtype, device = signature
                if arguments and source is not None and torch._debug_has_internal_overlap(source) == 1:
                    raise _OverlappingArgument(len(keys))
                if source is not None and _adoptable(source, self):
                    self.static[key] = _buffer_alias(source)
                    self.borrowed.add(key)
                else:
                    with torch.inference_mode(False):
                        self.static[key] = torch.empty_strided(shape, stride, dtype=dtype, device=device)
                self.static[key]._mm_graphs = weakref.ref(self)
                _static_storages[_storage(self.static[key])] = self.static[key]
                self._index_buffer(key, self.static[key])
                buffer_id = id(self.static[key])
                self._tensor_refs[buffer_id] = self._tensor_refs.get(buffer_id, 0) + 1
            keys.append(key)
            buffers.append(self.static[key])
        return keys, buffers

    def _capture(self, module, forward, leaves, spec, signature, extras, extra_signature, warm_output, static_key, buffers, adopt_outputs):
        from . import allocator
        entry = _Graph()
        outputs = self._outputs(leaves, warm_output)
        entry.positions = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        out_signatures = [output for output in outputs if output is not None and output[0] != "argument"]
        counts = {}
        for sig in [signature[index] for index in entry.positions] + list(extra_signature):
            counts[sig] = counts.get(sig, 0) + 1
        out_sources = [leaf if adopt_outputs else None for leaf, output in zip(_flatten(warm_output)[0], outputs) if output is not None and output[0] != "argument"]
        output_keys, output_buffers = self._buffers(out_signatures, out_sources, counts)
        plan = self._capture_output_plan
        if plan is not None:
            for source, buffer in zip(out_sources, output_buffers):
                if source is not None and _same_buffer(source, buffer) and _adoptable(source, self):
                    plan.bind(buffer)
        entry.static_key = static_key + output_keys
        buffers = buffers + output_buffers
        inputs = len(entry.positions)
        entry.static, entry.extra_static = buffers[:inputs], buffers[inputs:inputs + len(extras)]
        out_buffers = iter(buffers[inputs + len(extras):])
        static_leaves = list(leaves)
        for index, static in zip(entry.positions, entry.static):  # a capture runs nothing: the buffers need no values
            static_leaves[index] = static
        args, kwargs = pytree.tree_unflatten(static_leaves, spec)
        versions = [static._version for static in entry.static]
        checked_key = (id(module), getattr(forward, "__func__", forward), signature)
        if checked_key in self.checked:
            check = contextlib.nullcontext()
        else:  # costly (each operation goes through Python): once per code path, which other weights capture again
            allowed = {_storage(t) for t in buffers} | {_storage(t) for t in _constants}
            allowed.update(_storage(t) for t in self.weight_tensors(module) if t.is_cuda)
            check = _ReadCheck(allowed)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        entry.graph = torch.cuda.CUDAGraph()
        failed = f"CUDA graph capture of a block of '{self.name}' ({type(module).__name__}) failed"
        # torch.cuda.graph would synchronize the device and empty the allocator cache before each capture, which no capture needs
        originals = [holder[index] for holder, index in extras]
        for (holder, index), static in zip(extras, entry.extra_static):
            holder[index] = static
        if not _autotune_guarded:
            _guard_triton_autotune()
        global _capturing
        with torch.cuda.stream(_capture_stream(torch.cuda.current_device())):
            entry.graph.capture_begin(self.pool, capture_error_mode="thread_local")
            _capturing = True
            if plan is not None:
                plan.capture()
            plan_valid = True
            try:
                with check:
                    handoffs = _handoffs(args, kwargs) if self._capture_lifetimes else {}
                    output, entry.lifetimes = _execute_lifetimes(forward, args, kwargs, self._capture_lifetimes, static_leaves)
                    entry.emptied = tuple(index for index in entry.lifetimes if not handoffs[id(static_leaves[index])])
                    out_leaves, entry.out_spec = _flatten(output)
                    del output
                    entry.outputs = []
                    for leaf, output in zip(out_leaves, outputs):  # the other outputs are copied to static buffers, which leaves the memory of
                        if output is None:  # the tensors of the block to the next captures
                            entry.outputs.append(None)
                        elif output[0] == "argument":
                            entry.outputs.append(entry.positions.index(output[1]))
                        else:
                            static = next(out_buffers)
                            if not allocator.same_graph_storage(leaf, static):
                                static.copy_(leaf)
                            entry.outputs.append(static)
                    entry.out_leaves = [None if isinstance(leaf, torch.Tensor) else leaf for leaf in out_leaves]
                    del out_leaves
                    leaf = None
            except BaseException:
                with contextlib.suppress(Exception):  # the stream leaves the capture, which the error has invalidated
                    entry.graph.capture_end()
                raise
            finally:
                if plan is not None:
                    plan_valid = plan.finish()
                _capturing = False
                for (holder, index), original in zip(extras, originals):
                    holder[index] = original
            try:
                entry.graph.capture_end()
            except Exception as error:
                raise RuntimeError(f"{failed}: {error}") from error
        if not plan_valid:
            return None
        if isinstance(check, _ReadCheck):
            if check.unknown:
                raise RuntimeError(f"CUDA graph capture of a block of '{self.name}' ({type(module).__name__}): it reads tensors that are neither "
                                   f"its weights nor its arguments, whose addresses may change between replays:\n" + "\n".join(check.unknown))
            self.checked.add(checked_key)
        entry.mutated = [index for index, (static, version) in enumerate(zip(entry.static, versions)) if static._version != version]
        if not self.reported:
            print(f"[MMGP] CUDA graphs of the blocks of '{self.name}' are being captured.")
            self.reported = True
        return entry

    def _replay(self, entry, leaves, extras):
        from . import allocator
        for key in entry.static_key[len(entry.static) + len(entry.extra_static):]:
            if key in self.borrowed:
                allocator.prepare_reuse(self.static[key])
        for ordinal, (index, static) in enumerate(zip(entry.positions, entry.static)):
            if _same_buffer(static, leaves[index]):
                continue
            if _movable(leaves[index], static):
                self._move(entry.static_key[ordinal], leaves[index])
            else:
                static.copy_(leaves[index])
        if extras:
            torch._foreach_copy_(entry.extra_static, [holder[index] for holder, index in extras])
        entry.graph.replay()
        stream = torch.cuda.current_stream()
        for key in set(entry.static_key):
            if key in self.borrowed:
                self.static[key].record_stream(stream)
        for index in entry.mutated:
            if not _same_buffer(leaves[entry.positions[index]], entry.static[index]):
                leaves[entry.positions[index]].copy_(entry.static[index])
            else:
                torch.autograd.graph.increment_version(leaves[entry.positions[index]])
        output_keys = iter(entry.static_key[len(entry.static) + len(entry.extra_static):])
        out = [leaf if output is None else leaves[entry.positions[output]] if isinstance(output, int) else
               _buffer_alias(output) if next(output_keys) in self.borrowed else output.clone()
               for leaf, output in zip(entry.out_leaves, entry.outputs)]
        return pytree.tree_unflatten(out, entry.out_spec)
