# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Coordinate reclaimable VRAM: optional expert copies precede streamed blocks."""
import weakref

from . import allocator

_clients = weakref.WeakKeyDictionary()


def _reclaim(size, device):
    freed = 0
    for client, _ in sorted(list(_clients.items()), key=lambda item: item[1]):
        freed += client.on_pressure(max(0, size - freed))
        if freed >= size:
            break
    return freed


def register(client, priority=1):
    _clients[client] = priority
    if allocator.active == "vmm":
        allocator.set_pressure_callback(_reclaim)


def unregister(client):
    _clients.pop(client, None)
    if not _clients and allocator.active == "vmm":
        allocator.set_pressure_callback(None)
