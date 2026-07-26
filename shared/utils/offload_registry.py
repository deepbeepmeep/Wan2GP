"""Registry of mmgp offload objects owned by WanGP extensions (postprocessing
upsamplers, voice converters, ...).

Every extension that creates its own ``offload.profile(...)`` object should
register it here so WanGP can track extension VRAM/RAM usage and release it
centrally (for instance from the toolbar "Unload Models" tool).
"""

from __future__ import annotations

import gc

# name -> (offloadobj, release_fn). `release_fn` is the owner's teardown (it must
# release the offload object AND clear the owner's model references); when None the
# registry falls back to releasing the offload object directly.
_extension_offloadobjs: dict[str, tuple[object, object]] = {}


def register_offloadobj(name: str, offloadobj, release_fn=None) -> None:
    """Track the offload object owned by extension `name` (replaces any previous one)."""
    _extension_offloadobjs[name] = (offloadobj, release_fn)


def unregister_offloadobj(name: str, offloadobj=None) -> None:
    """Stop tracking the offload object of extension `name`.

    If `offloadobj` is provided, only unregister when it is still the tracked one.
    """
    entry = _extension_offloadobjs.get(name)
    if entry is not None and (offloadobj is None or entry[0] is offloadobj):
        _extension_offloadobjs.pop(name, None)


def registered_names() -> list[str]:
    return list(_extension_offloadobjs)


def query_offloadobjs() -> dict[str, object]:
    return {name: entry[0] for name, entry in _extension_offloadobjs.items()}


def unload_vram(names=None) -> list[str]:
    unloaded = []
    for name in list(_extension_offloadobjs) if names is None else list(names):
        entry = _extension_offloadobjs.get(name)
        if entry is None:
            continue
        offloadobj, _ = entry
        offloadobj.unload_all()
        unloaded.append(name)
    if unloaded:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return unloaded


def release_all(names=None) -> list[str]:
    """Release the offload objects of the given extensions (all when `names` is None).

    Returns the names of the extensions that were released.
    """
    released = []
    for name in list(_extension_offloadobjs) if names is None else list(names):
        entry = _extension_offloadobjs.pop(name, None)
        if entry is None:
            continue
        offloadobj, release_fn = entry
        if release_fn is not None:
            release_fn()
        else:
            offloadobj.release()
        released.append(name)
    if released:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return released
