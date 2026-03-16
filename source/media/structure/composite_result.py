"""Result helpers for composite guidance video creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class GuidanceCompositeResult:
    """Outcome of composite-guidance-video creation."""

    status: str
    output_path: Path | None = None


def create_composite_guidance_video_result(
    *,
    create_fn: Callable[..., Path],
    structure_configs: list[dict],
    total_frames: int,
    structure_type: str,
    target_resolution: tuple[int, int],
    target_fps: int,
    output_path: Path,
    **kwargs,
) -> GuidanceCompositeResult:
    """Create a result object around a composite-guidance-video builder."""
    if not structure_configs:
        return GuidanceCompositeResult(status="skipped", output_path=None)

    created_path = create_fn(
        structure_configs=structure_configs,
        total_frames=total_frames,
        structure_type=structure_type,
        target_resolution=target_resolution,
        target_fps=target_fps,
        output_path=output_path,
        **kwargs,
    )
    return GuidanceCompositeResult(status="created", output_path=Path(created_path))
