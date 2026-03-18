"""Debug artifact helpers for transition-prompt generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from source.media.vlm.image_prep import create_labeled_debug_image


@dataclass(frozen=True)
class TransitionDebugArtifacts:
    combined: Path | None
    left: Path | None
    right: Path | None


def save_transition_debug_artifacts(*, start_img, end_img, pair_index: int, start_image_path: str):
    base_dir = Path(start_image_path).resolve().parent / "vlm_debug"
    base_dir.mkdir(parents=True, exist_ok=True)
    combined = base_dir / f"vlm_combined_pair{pair_index}.jpg"
    left = base_dir / f"vlm_pair{pair_index}_LEFT_start.jpg"
    right = base_dir / f"vlm_pair{pair_index}_RIGHT_end.jpg"
    create_labeled_debug_image(start_img, end_img, pair_index=pair_index).save(combined, quality=95)
    start_img.save(left, quality=95)
    end_img.save(right, quality=95)
    return TransitionDebugArtifacts(combined=combined, left=left, right=right)


__all__ = ["TransitionDebugArtifacts", "save_transition_debug_artifacts"]
