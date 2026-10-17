# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Model-facing execution services, independent of MMGP's offload/compilation hooks."""
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_owners = weakref.WeakKeyDictionary()
_resources = weakref.WeakKeyDictionary()
_resource_users = {}
_execution_checkpoint = ContextVar("mmgp_execution_checkpoint", default=None)


@contextmanager
def execution_scope(checkpoint):
    """Supply a local cancellation check for long MMGP residency operations."""
    token = _execution_checkpoint.set(checkpoint)
    try:
        yield
    finally:
        _execution_checkpoint.reset(token)


def checkpoint():
    callback = _execution_checkpoint.get()
    if callback is not None:
        callback()


def register_resource(module, prepare, release):
    """Tie a shared execution resource to a model's GPU residency.

    prepare(device) runs before its first owner loads, outside graph warmup;
    release() runs after its last owner unloads and its graphs are destroyed.
    Register before profiling, with the same callback pair for models sharing the resource. Neither
    callback may retain the model or allocate resources at registration time.
    """
    _resources.setdefault(module, set()).add((prepare, release))


def _load_resources(manager, model_id):
    model = manager.models[model_id]
    for resource in manager.runtime_resources[model_id]:
        users = _resource_users.setdefault(resource, weakref.WeakSet())
        first = not users
        users.add(model)  # an interrupted preparation is cleaned up by unload_all too
        if first:
            from . import allocator
            with allocator.normal_allocations():
                resource[0](torch.device("cuda", torch.cuda.current_device()))


def _unload_resources(manager, model_ids):
    for model_id in model_ids:
        model = manager.models[model_id]
        for resource in manager.runtime_resources[model_id]:
            users = _resource_users.get(resource)
            if users is None:
                continue
            users.discard(model)
            if not users:
                resource[1]()
                del _resource_users[resource]


def _register(manager, models):
    owners = {model_id: (weakref.ref(manager), model_id) for model_id in models}
    manager.runtime_resources = {}
    for model_id, model in models.items():
        resources = manager.runtime_resources[model_id] = set()
        for module in model.modules():
            _owners[module] = owners[model_id]
            resources.update(_resources.get(module, ()))
    # A separately managed root owns itself even when another model contains it.
    for model_id, model in models.items():
        _owners[model] = owners[model_id]


def _unregister(manager):
    for module, (reference, _) in list(_owners.items()):
        if reference() is manager:
            del _owners[module]


def _owner(module):
    entry = _owners.get(module)
    if entry is None:
        return None
    manager = entry[0]()
    return None if manager is None else (manager, entry[1])


def execution_device(module):
    """Return the execution device, even while managed weights are on CPU.

    An unmanaged module uses its first parameter/buffer's device, or CPU if empty.
    Registration is removed when its offload manager is released.
    """
    from .experts import ExpertBank
    if isinstance(module, ExpertBank):
        return torch.device("cpu")
    if _owner(module) is not None:
        return torch.device("cuda")
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return torch.device("cpu") if tensor is None else tensor.device


def ensure_loaded(module):
    """Load a managed model's base/preloaded weights; leave block swapping to MMGP.

    Use before custom entry points that read weights without calling a hooked
    forward. Unmanaged modules are unchanged. This does not pin a model in VRAM.
    """
    owner = _owner(module)
    if owner is not None:
        manager, model_id = owner
        manager.ensure_model_loaded(model_id)


def weights_resident(module):
    """Whether a managed model's weights are all in VRAM now (loaded, no block streamed; expert banks aside): an MMGP graph stage
    may then span several of its blocks (see cuda_graphs.run). Check it at each call: a dynamic preload can release blocks between
    calls. Unmanaged modules: True."""
    owner = _owner(module)
    if owner is None:
        return True
    manager, model_id = owner
    if model_id not in manager.active_models_ids:
        return False
    preloaded = manager.preloaded_blocks_per_model[model_id]
    prefix = model_id + "/"
    return all(name[len(prefix):] in preloaded for name, size in manager.blocks_of_modules_sizes.items() if size and name.startswith(prefix))


def place_experts(module):
    """Place a managed model's optional GPU expert copies now, outside graph recording (e.g. at the first step of a decoding loop replayed
    as one graph, which MMGP's hooks never see): as many as its budget or the dynamic preload allows, ordinary blocks first; never
    shrinks them. Returns False while its ordinary blocks are not all resident yet (call again at a later step). Models without
    experts: True."""
    owner = _owner(module)
    if owner is None:
        return True
    manager, model_id = owner
    residency = manager.expert_residency.get(model_id)
    return residency is None or residency.place()


def requires_layer_offload(module):
    """Whether this owner's weights may move during execution or include CPU experts.

    Whole-model external CUDA graphs need stable weight addresses. A dynamic
    preload remains reclaimable even when it currently happens to fit entirely.
    MMGP's own block graphs support these models and share their release lifecycle.
    """
    owner = _owner(module)
    if owner is None:
        return False
    manager, model_id = owner
    if model_id in manager.expert_residency:
        return True
    if manager.auto_preload is not None and model_id in manager.auto_preload.plans:
        return True
    return any(name.startswith(model_id + "/") and size for name, size in manager.blocks_of_modules_sizes.items())
