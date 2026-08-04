"""First-Block residual cache for MiniMax H3 (Sol-Engine style, single-GPU).

Runs DiT block 0 every step. If the first-block residual is close enough to the
previous computed step, reuses the previous full-stack residual and skips blocks
1..N-1. Always fully computes the warmup head and the final step.

This is an approximation. Prefer conservative thresholds for audio+video quality.
Inspired by Sol-Engine's FirstBlockCache line and cache-dit FBC; clean-room code
for WanGP (no Sol-Engine / Spectrum source copied).
"""

from __future__ import annotations

import time
from typing import Any, Optional


# Multiplier (UI "around xN speed up") -> residual relative-L1 threshold.
# Sol-Engine's published H3 fullopt used 0.08 on multi-GPU; single-GPU starts
# more conservative because offload and shorter step counts change the curve.
_MULTIPLIER_THRESHOLDS = (
    (1.5, 0.05),
    (1.75, 0.065),
    (2.0, 0.08),
    (2.25, 0.10),
    (2.5, 0.12),
)


def threshold_from_multiplier(multiplier: float) -> float:
    mult = float(multiplier)
    if mult <= _MULTIPLIER_THRESHOLDS[0][0]:
        return _MULTIPLIER_THRESHOLDS[0][1]
    if mult >= _MULTIPLIER_THRESHOLDS[-1][0]:
        return _MULTIPLIER_THRESHOLDS[-1][1]
    for (m0, t0), (m1, t1) in zip(_MULTIPLIER_THRESHOLDS, _MULTIPLIER_THRESHOLDS[1:]):
        if m0 <= mult <= m1:
            span = m1 - m0
            alpha = 0.0 if span <= 0 else (mult - m0) / span
            return t0 + alpha * (t1 - t0)
    return 0.08


def relative_l1(current, previous) -> float:
    """Mean |a-b| / mean |b|, matching FirstBlockCache skip metric."""
    import torch

    prev = previous.detach().float()
    cur = current.detach().float()
    denom = prev.abs().mean().clamp_min(1e-8)
    return float((cur - prev).abs().mean().div(denom).item())


def configure_h3_cache(skip_steps_cache: Any) -> dict:
    """Attach H3 First-Block settings onto WanGP's skip_steps_cache object."""
    threshold = threshold_from_multiplier(getattr(skip_steps_cache, "multiplier", 2.0))
    skip_steps_cache.update({
        "h3_first_block": True,
        "fbc_threshold": threshold,
        "cache_type": "fbc",
    })
    return {"method": "h3_first_block_cache", "threshold": threshold}


def reset_h3_cache(cache: Any, num_steps: int) -> None:
    if cache is None or not getattr(cache, "h3_first_block", False):
        return
    cache.num_steps = int(num_steps)
    cache.skipped_steps = 0
    cache.full_steps = 0
    cache.previous_residual = None
    cache.previous_modulated_input = None  # stores previous first-block residual
    cache._denoise_t0 = time.perf_counter()
    cache._denoise_seconds = None


def finish_h3_cache(cache: Any) -> Optional[str]:
    if cache is None or not getattr(cache, "h3_first_block", False):
        return None
    if getattr(cache, "_denoise_t0", None) is not None:
        cache._denoise_seconds = time.perf_counter() - cache._denoise_t0
    full = int(getattr(cache, "full_steps", 0) or 0)
    skipped = int(getattr(cache, "skipped_steps", 0) or 0)
    total = full + skipped
    thr = float(getattr(cache, "fbc_threshold", 0.0) or 0.0)
    seconds = getattr(cache, "_denoise_seconds", None)
    parts = [
        f"[H3 FirstBlockCache] steps full={full} skipped_tail={skipped}/{total or 0}",
        f"threshold={thr:.3f}",
    ]
    if seconds is not None:
        parts.append(f"denoise={seconds:.2f}s")
    if total > 0 and skipped > 0:
        parts.append(f"block-stack cut~{100.0 * skipped / total:.0f}% of steps")
    return " ".join(parts)


def should_skip_remaining(
    cache: Any,
    step_no: int,
    first_residual,
) -> bool:
    """Return True if blocks after the first may reuse the previous full residual."""
    if cache is None or not getattr(cache, "h3_first_block", False):
        return False
    num_steps = int(getattr(cache, "num_steps", 0) or 0)
    start_step = int(getattr(cache, "start_step", 0) or 0)
    if step_no <= start_step:
        return False
    if num_steps > 0 and step_no >= num_steps - 1:
        return False
    prev_first = getattr(cache, "previous_modulated_input", None)
    prev_full = getattr(cache, "previous_residual", None)
    if prev_first is None or prev_full is None:
        return False
    if prev_first.shape != first_residual.shape:
        return False
    if prev_full.shape != first_residual.shape:
        return False
    delta = relative_l1(first_residual, prev_first)
    cache.last_fbc_delta = delta
    return delta < float(getattr(cache, "fbc_threshold", 0.08))


__all__ = [
    "configure_h3_cache",
    "finish_h3_cache",
    "relative_l1",
    "reset_h3_cache",
    "should_skip_remaining",
    "threshold_from_multiplier",
]
