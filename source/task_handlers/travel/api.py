"""Travel wrapper facade for guide-video helpers."""

from source.task_handlers.travel.guidance import guide_video_ops as _guide_video_ops

GuideVideoRequest = _guide_video_ops.GuideVideoRequest
create_guide_video = _guide_video_ops.create_guide_video
create_guide_video_for_travel_segment = _guide_video_ops.create_guide_video_for_travel_segment
prepare_vace_ref_for_segment = _guide_video_ops.prepare_vace_ref_for_segment
rife_interpolate_images_to_video = _guide_video_ops.rife_interpolate_images_to_video

__all__ = [
    "_guide_video_ops",
    "GuideVideoRequest",
    "create_guide_video",
    "create_guide_video_for_travel_segment",
    "prepare_vace_ref_for_segment",
    "rife_interpolate_images_to_video",
]
