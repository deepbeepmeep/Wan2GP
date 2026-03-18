"""Join-clip preprocessing helpers used before transition generation."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from source.media.video.video_info import VideoMetadataError
from source.task_handlers.join.video_preprocess_utils import (
    require_video_frame_count_and_fps,
)
from source.utils.output_paths import upload_intermediate_file_to_storage


def _detect_resolution(video_path: Path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        return None
    return width, height


def preprocess_clips_for_join(
    *,
    clip_list: list[dict],
    join_settings: dict,
    temp_dir: Path,
    orchestrator_task_id: str,
    skip_frame_validation: bool = False,
):
    """Normalize join-clip inputs into local/uploaded assets plus frame metadata."""
    del skip_frame_validation  # Metadata is always required even when stricter checks are skipped.

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    processed_clips: list[dict] = []
    frame_counts: list[int] = []
    detected_resolution: tuple[int, int] | None = None

    for idx, clip in enumerate(clip_list):
        url = clip.get("url")
        if not url:
            raise ValueError(f"clip {idx} missing 'url'")

        source_path = Path(url)
        if not source_path.exists():
            raise ValueError(f"clip {idx} not found: {url}")

        try:
            frame_count, _fps = require_video_frame_count_and_fps(
                source_path,
                context=f"preprocess clip {idx}",
            )
        except VideoMetadataError as exc:
            raise ValueError(str(exc)) from exc
        frame_counts.append(frame_count)

        if detected_resolution is None and join_settings.get("use_input_video_resolution"):
            detected_resolution = _detect_resolution(source_path)

        local_name = f"join_clip_{idx}{source_path.suffix or '.mp4'}"
        local_path = temp_dir / local_name
        shutil.copy2(source_path, local_path)

        uploaded_url = upload_intermediate_file_to_storage(
            local_file_path=local_path,
            task_id=orchestrator_task_id,
            filename=local_path.name,
            runtime_config=None,
        )

        processed_clip = dict(clip)
        processed_clip["url"] = uploaded_url or str(local_path)
        processed_clips.append(processed_clip)

    return processed_clips, frame_counts, detected_resolution
