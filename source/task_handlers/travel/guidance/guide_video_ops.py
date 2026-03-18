"""Typed guide-video request adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.media.video.travel_guide import (
    create_guide_video_for_travel_segment,
    prepare_vace_ref_for_segment,
    rife_interpolate_images_to_video,
)


@dataclass(frozen=True)
class GuideVideoRequest:
    segment_idx_for_logging: int
    end_anchor_image_index: int
    is_first_segment_from_scratch: bool
    total_frames_for_segment: int
    parsed_res_wh: tuple[int, int]
    fps_helpers: int
    input_images_resolved_for_guide: list[str]
    path_to_previous_segment_video_output_for_guide: str | None
    output_target_dir: Path
    guide_video_base_name: str
    segment_image_download_dir: Path
    task_id_for_logging: str
    orchestrator_details: dict[str, Any]
    segment_params: dict[str, Any]
    structure_config: Any = None


def create_guide_video(request: GuideVideoRequest):
    return create_guide_video_for_travel_segment(
        segment_idx_for_logging=request.segment_idx_for_logging,
        end_anchor_image_index=request.end_anchor_image_index,
        is_first_segment_from_scratch=request.is_first_segment_from_scratch,
        total_frames_for_segment=request.total_frames_for_segment,
        parsed_res_wh=request.parsed_res_wh,
        fps_helpers=request.fps_helpers,
        input_images_resolved_for_guide=request.input_images_resolved_for_guide,
        path_to_previous_segment_video_output_for_guide=request.path_to_previous_segment_video_output_for_guide,
        output_target_dir=request.output_target_dir,
        guide_video_base_name=request.guide_video_base_name,
        segment_image_download_dir=request.segment_image_download_dir,
        task_id_for_logging=request.task_id_for_logging,
        orchestrator_details=request.orchestrator_details,
        segment_params=request.segment_params,
        structure_config=request.structure_config,
    )


__all__ = [
    "GuideVideoRequest",
    "create_guide_video",
    "create_guide_video_for_travel_segment",
    "prepare_vace_ref_for_segment",
    "rife_interpolate_images_to_video",
]
