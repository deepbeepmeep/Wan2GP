# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Model-facing execution services, independent of MMGP's offload/compilation hooks."""
import weakref

import torch


_owners = weakref.WeakKeyDictionary()


def _register(manager, models):
    owners = {model_id: (weakref.ref(manager), model_id) for model_id in models}
    for model_id, model in models.items():
        for module in model.modules():
            _owners[module] = owners[model_id]
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
