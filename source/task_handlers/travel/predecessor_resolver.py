"""Shared predecessor-resolution helpers for travel segment continuation flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

from source.core.db.dependencies.task_dependencies_queries import get_segment_predecessor_output
from source.utils.download_utils import download_file


@dataclass(frozen=True)
class PredecessorResult:
    task_id: str | None
    output_url: str | None

    @property
    def found(self) -> bool:
        return bool(self.task_id and self.output_url)


def resolve_generation_id(key: str, *sources: dict[str, Any] | None) -> str | None:
    """Resolve a generation id from multiple parameter sources (first truthy wins)."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        value = source.get(key)
        if value:
            return value
    return None


def resolve_segment_predecessor(
    task_id: str,
    parent_generation_id: str | None,
    child_generation_id: str | None,
    child_order: int | None,
    segment_index: int | None,
) -> PredecessorResult:
    predecessor_task_id, predecessor_output_url = get_segment_predecessor_output(
        task_id=task_id,
        parent_generation_id=parent_generation_id,
        child_generation_id=child_generation_id,
        child_order=child_order,
        segment_index=segment_index,
    )
    return PredecessorResult(
        task_id=predecessor_task_id,
        output_url=predecessor_output_url,
    )


def download_predecessor_video(
    predecessor_output_url: str,
    output_dir: Path,
    *,
    prefix: str,
) -> str | None:
    """Download remote predecessor videos locally, rejecting empty cached downloads."""
    if not predecessor_output_url.startswith("http"):
        return predecessor_output_url

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        local_filename = Path(predecessor_output_url).name or "predecessor.mp4"
        local_download_path = output_dir / f"{prefix}_{local_filename}"

        if local_download_path.exists() and local_download_path.stat().st_size <= 0:
            local_download_path.unlink()

        if not local_download_path.exists():
            download_ok = download_file(
                predecessor_output_url,
                output_dir,
                local_download_path.name,
            )
            if download_ok is False:
                return None

        if not local_download_path.exists() or local_download_path.stat().st_size <= 0:
            local_download_path.unlink(missing_ok=True)
            return None

        return str(local_download_path.resolve())
    except (OSError, ValueError, RuntimeError):
        return None


def extract_prefix_video(
    predecessor_video_path: str,
    output_dir: Path,
    *,
    segment_idx: int,
    frames_needed: int,
    prefix: str,
) -> str:
    """Extract the last N frames from a predecessor video into a new prefix clip."""
    from source.media.video import (
        extract_frame_range_to_video,
        get_video_frame_count_and_fps,
    )

    pred_frames, pred_fps = get_video_frame_count_and_fps(predecessor_video_path)
    if not pred_frames or pred_frames <= 0:
        return predecessor_video_path

    start_frame = max(0, int(pred_frames) - int(frames_needed))
    trimmed_prefix_path = output_dir / (
        f"{prefix}_{segment_idx:02d}_last{frames_needed}frames_{uuid.uuid4().hex[:6]}.mp4"
    )
    trimmed_result = extract_frame_range_to_video(
        input_video_path=predecessor_video_path,
        output_video_path=str(trimmed_prefix_path),
        start_frame=start_frame,
        end_frame=None,
        fps=float(pred_fps) if pred_fps and pred_fps > 0 else 16.0,
    )
    return str(trimmed_result)


__all__ = [
    "PredecessorResult",
    "download_predecessor_video",
    "extract_prefix_video",
    "resolve_generation_id",
    "resolve_segment_predecessor",
]
