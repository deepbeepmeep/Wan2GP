"""Normalized generation/continuation policy derived from task payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ContinuationStrategy = str


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_overlap_from_payload(payload: dict[str, Any]) -> int:
    frame_overlap = payload.get("frame_overlap")
    if isinstance(frame_overlap, list) and frame_overlap:
        return max(0, _coerce_int(frame_overlap[0], 10))
    return max(0, _coerce_int(frame_overlap, 10))


@dataclass(frozen=True)
class ContinuationPolicy:
    enabled: bool
    strategy: ContinuationStrategy
    overlap_frames: int
    uses_guide_for_overlap: bool
    uses_mask_video: bool

    @property
    def requires_video_source(self) -> bool:
        """True for strategies that inject a predecessor clip as video_source (SVI or LTX prefix)."""
        return self.strategy in ("prefix_video_source", "svi_latent_chaining")

    @property
    def uses_svi_latent_chaining(self) -> bool:
        return self.strategy == "svi_latent_chaining"


@dataclass(frozen=True)
class GenerationPolicy:
    travel_mode: str
    continuation: ContinuationPolicy

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GenerationPolicy":
        travel_mode = str(payload.get("model_type") or "i2v")
        continuation_payload = payload.get("continuation_config")

        strategy: ContinuationStrategy = "none"
        overlap_frames = 0

        if isinstance(continuation_payload, dict):
            strategy = str(continuation_payload.get("strategy") or "none")
            overlap_frames = max(0, _coerce_int(continuation_payload.get("overlap_frames"), 0))
        elif travel_mode == "vace" and bool(payload.get("chain_segments", True)):
            strategy = "guide_overlap_masked"
            overlap_frames = _resolve_overlap_from_payload(payload)

        enabled = strategy != "none"
        uses_overlap_guide = strategy == "guide_overlap_masked"

        return cls(
            travel_mode=travel_mode,
            continuation=ContinuationPolicy(
                enabled=enabled,
                strategy=strategy,
                overlap_frames=overlap_frames,
                uses_guide_for_overlap=uses_overlap_guide,
                uses_mask_video=uses_overlap_guide,
            ),
        )


__all__ = [
    "ContinuationPolicy",
    "GenerationPolicy",
]
