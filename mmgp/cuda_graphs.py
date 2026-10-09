"""Automatic CUDA graphs of the blocks of the towers (the repeated layers of a transformer): the host work of a block call (Python,
kernel launches) is recorded once and replayed, for models whose steps are short enough for the host to be the bottleneck.

Inside `scope()` (the denoising loop of a model; elsewhere blocks run as usual), each call of a block is keyed by the tensor objects of
its weights, its active LoRAs and their scalings, the shapes of its tensor arguments and extra inputs, its other arguments and the
registered global keys: the first call of a key runs eagerly (it compiles and tunes the kernels, makes the caches of the weights), then
is captured into a CUDA graph that the next calls replay. Blocks run eagerly, without captures, while the placement of their weights is not
settled (`ready()` False: e.g. an automatic preload measuring a new workload, which keeps other blocks in VRAM at the next step). A block thus has a graph per location of its weights (kept in VRAM, or each
slot of the swapped blocks), and a model loaded again captures its graphs again. The tensor arguments, extra inputs (tensors that come
and go with a block, e.g. its LoRA copies) and outputs go through static buffers shared by the graphs, the arguments a block modifies in
place are copied back, and an output that is an argument is that argument. A block keeps its `MAX_KEYS` most recently used graphs (a new
resolution replaces them); all the graphs go with the model or `invalidate()`.

Contract of a block call in a scope: the block reads only its weights, its tensor arguments (in lists, tuples, dicts and dataclasses)
and tensors it creates; its effects are its outputs and in-place updates of its tensor arguments (Python side effects are not replayed,
and the capture runs the Python of the block once more); no synchronization or CPU round trip. The capture checks the reads and raises
an error naming any other tensor.
"""
import contextlib
import dataclasses
import enum
import traceback

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

MAX_KEYS = 8  # graphs kept per block, the least recently used dropped first: weight locations, CFG branches of other lengths, resolutions
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
_registered = set()
_managers = []
_capturing = False  # a block's capture is running (its forward is recorded, nothing runs on the GPU)
_autotune_guarded = False


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
    _constants.append(tensor)
    return tensor


def register_capture_setup(setup):
    """setup(stream) is called before each capture on the capture stream, the one stream of the captures of a device: it prepares what a
    library would allocate there on first use, which a capture cannot do (per-stream workspaces), and returns at once when it is ready."""
    if setup not in _capture_setups:
        _capture_setups.append(setup)


def _capture_stream(device):
    stream = _capture_streams.get(device, None)
    if stream is None:
        stream = _capture_streams[device] = torch.cuda.Stream(device)
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
    __slots__ = ("graph", "positions", "static", "extra_static", "static_key", "mutated", "out_leaves", "out_spec", "outputs")


class BlockGraphs:
    """CUDA graphs of the tower blocks of one model."""

    def __init__(self, name, weights_key, weight_tensors, extra_inputs, ready):
        # extra_inputs(block) -> [(list, index)]: tensors read by the block that are neither weights nor arguments, read by its graphs from
        # static buffers (their list items are these buffers during a capture); ready() -> False while the weights may move
        self.name, self.weights_key, self.weight_tensors, self.extra_inputs, self.ready = name, weights_key, weight_tensors, extra_inputs, ready
        self.disabled = None  # the reason why this model's blocks run eagerly (e.g. weights whose kernels cannot be captured), or None
        self.reported = False
        self.checked = set()  # (block, signature) whose capture checked the reads
        self.clear()
        _managers.append(self)

    def clear(self):
        # entries: key -> graph; keys: block -> its keys, least recently used first; static: buffers shared by the graphs
        self.entries, self.keys, self.static, self.pool = {}, {}, {}, None

    def wrap(self, module, forward):
        def graphed_forward(*args, **kwargs):
            if _depth == 0:
                return forward(*args, **kwargs)
            return self.call(module, forward, args, kwargs)
        return graphed_forward

    def call(self, module, forward, args, kwargs):
        global _recording
        if _blocked is not None or self.disabled is not None or not self.ready():
            return forward(*args, **kwargs)
        leaves, spec = _flatten((args, kwargs))
        signature = _signature(leaves)
        extras = self.extra_inputs(module)
        extra_signature = _signature([holder[index] for holder, index in extras])
        key = (id(module), spec, signature, extra_signature, self.weights_key(module), tuple(provider() for provider in _key_providers))
        keys = self.keys.setdefault(id(module), {})
        entry = self.entries.get(key, None)
        if entry is None:  # first call of this key: eager (it compiles and tunes the kernels, makes the caches of the weights), then captured
            if len(keys) >= MAX_KEYS:
                self._evict(next(iter(keys)))
            _recording += 1
            try:
                out = forward(*args, **kwargs)
                self.entries[key] = self._capture(module, forward, leaves, spec, signature, extras, extra_signature, self._outputs(leaves, out))
            finally:
                _recording -= 1
            keys[key] = None
            return out
        keys[key] = keys.pop(key)  # most recently used
        return self._replay(entry, leaves, extras)

    def _outputs(self, leaves, out):
        # per output leaf: ("argument", index of the argument it is), its signature, or None if it is not a tensor
        by_pointer = {leaf.data_ptr(): (index, leaf) for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)}
        outputs = []
        for leaf in _flatten(out)[0]:  # an output that is an argument (its storage, shape and strides) is returned as the caller's tensor
            if isinstance(leaf, torch.Tensor):
                index, argument = by_pointer.get(leaf.data_ptr(), (None, None))
                same = argument is not None and leaf.shape == argument.shape and leaf.stride() == argument.stride()
                outputs.append(("argument", index) if same else _signature((leaf,))[0])
            else:
                outputs.append(None)
        return outputs

    def _evict(self, key):
        del self.keys[key[0]][key]
        del self.entries[key]
        used = {k for entry in self.entries.values() for k in entry.static_key}
        self.static = {k: buffer for k, buffer in self.static.items() if k in used}

    def _buffers(self, signatures):
        # static buffers, shared by the graphs (they run one after the other) for each signature and occurrence in a call; not inference
        # tensors, whose version counters show the in-place updates of the arguments
        counts, keys, buffers = {}, [], []
        for signature in signatures:
            key = (signature, counts.setdefault(signature, 0))
            counts[signature] += 1
            buffer = self.static.get(key, None)
            if buffer is None:
                shape, stride, dtype, device = signature
                with torch.inference_mode(False):
                    buffer = self.static[key] = torch.empty_strided(shape, stride, dtype=dtype, device=device)
            keys.append(key)
            buffers.append(buffer)
        return keys, buffers

    def _capture(self, module, forward, leaves, spec, signature, extras, extra_signature, outputs):
        entry = _Graph()
        entry.positions = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        out_signatures = [output for output in outputs if output is not None and output[0] != "argument"]
        entry.static_key, buffers = self._buffers([signature[index] for index in entry.positions] + list(extra_signature) + out_signatures)
        inputs = len(entry.positions)
        entry.static, entry.extra_static = buffers[:inputs], buffers[inputs:inputs + len(extras)]
        out_buffers = iter(buffers[inputs + len(extras):])
        static_leaves = list(leaves)
        for index, static in zip(entry.positions, entry.static):  # a capture runs nothing: the buffers need no values
            static_leaves[index] = static
        args, kwargs = pytree.tree_unflatten(static_leaves, spec)
        versions = [static._version for static in entry.static]
        if (id(module), signature) in self.checked:
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
            try:
                with check:
                    out_leaves, entry.out_spec = _flatten(forward(*args, **kwargs))
                    entry.outputs = []
                    for leaf, output in zip(out_leaves, outputs):  # the other outputs are copied to static buffers, which leaves the memory of
                        if output is None:  # the tensors of the block to the next captures
                            entry.outputs.append(None)
                        elif output[0] == "argument":
                            entry.outputs.append(entry.positions.index(output[1]))
                        else:
                            static = next(out_buffers)
                            static.copy_(leaf)
                            entry.outputs.append(static)
                    entry.out_leaves = [None if isinstance(leaf, torch.Tensor) else leaf for leaf in out_leaves]
                    del out_leaves
            except Exception as error:
                with contextlib.suppress(Exception):  # the stream leaves the capture, which the error has invalidated
                    entry.graph.capture_end()
                raise RuntimeError(f"{failed}: {error}") from error
            finally:
                _capturing = False
                for (holder, index), original in zip(extras, originals):
                    holder[index] = original
            try:
                entry.graph.capture_end()
            except Exception as error:
                raise RuntimeError(f"{failed}: {error}") from error
        if isinstance(check, _ReadCheck):
            if check.unknown:
                raise RuntimeError(f"CUDA graph capture of a block of '{self.name}' ({type(module).__name__}): it reads tensors that are neither "
                                   f"its weights nor its arguments, whose addresses may change between replays:\n" + "\n".join(check.unknown))
            self.checked.add((id(module), signature))
        entry.mutated = [index for index, (static, version) in enumerate(zip(entry.static, versions)) if static._version != version]
        if not self.reported:
            print(f"[MMGP] CUDA graphs of the blocks of '{self.name}' are being captured.")
            self.reported = True
        return entry

    def _replay(self, entry, leaves, extras):
        for index, static in zip(entry.positions, entry.static):
            static.copy_(leaves[index])
        if extras:
            torch._foreach_copy_(entry.extra_static, [holder[index] for holder, index in extras])
        entry.graph.replay()
        for index in entry.mutated:
            leaves[entry.positions[index]].copy_(entry.static[index])
        out = [leaf if output is None else leaves[entry.positions[output]] if isinstance(output, int) else output.clone()
               for leaf, output in zip(entry.out_leaves, entry.outputs)]
        return pytree.tree_unflatten(out, entry.out_spec)
