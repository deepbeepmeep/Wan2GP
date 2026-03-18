"""Compatibility shim for mask generation helpers."""

from source.utils.frame_utils import create_video_from_frames_list
from source.utils.mask_utils import create_mask_video_from_inactive_indices as _compat_create_mask_video

__all__ = ["create_video_from_frames_list", "create_mask_video_from_inactive_indices"]


def create_mask_video_from_inactive_indices(*args, **kwargs):
    return _compat_create_mask_video(
        *args,
        create_video_from_frames_list_fn=create_video_from_frames_list,
        **kwargs,
    )
