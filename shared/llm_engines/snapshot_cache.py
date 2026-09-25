"""Immutable CPU cache segments: share a stable prefix, copy only the changed tail."""

import torch


def snapshot_cache(tensor, ranges, *, axis, previous=None, reuse=0):
    segments = []
    remaining = reuse
    if previous is not None:
        for start, saved in previous["segments"]:
            count = min(remaining, saved.shape[axis])
            if count == 0:
                break
            segments.append((start, saved.narrow(axis, 0, count)))
            remaining -= count
    assert remaining == 0, "Saved cache does not cover its reusable prefix."
    remaining = reuse
    for start, end in ranges:
        skipped = min(remaining, end - start)
        start += skipped
        remaining -= skipped
        if start < end:
            saved = tensor.narrow(axis, start, end - start).detach().as_subclass(torch.Tensor).to("cpu", copy=True)
            segments.append((start, saved))
    return {"shape": tuple(tensor.shape), "axis": axis, "segments": segments}


def restore_cache(tensor, snapshot):
    axis = snapshot["axis"]
    for start, saved in snapshot["segments"]:
        tensor.narrow(axis, start, saved.shape[axis]).copy_(saved)
