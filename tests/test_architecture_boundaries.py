"""Architecture boundary policy tests used by CI.

These checks keep broad compatibility package surfaces from becoming the
recommended import path again. Behavioral smoke tests cover the underlying
helpers; this file enforces the package-level public surface.
"""

from __future__ import annotations

import pytest

import source.media.video as video
import source.utils as utils


def test_video_namespace_exports_only_module_boundaries() -> None:
    assert list(video.__all__) == ["api", "hires_utils", "vace_frame_utils"]

    for legacy_name in (
        "add_audio_to_video",
        "create_video_from_frames_list",
        "extract_frames_from_video",
        "stitch_videos_with_crossfade",
    ):
        with pytest.raises(AttributeError):
            getattr(video, legacy_name)


def test_utils_namespace_exports_only_module_boundaries() -> None:
    assert utils.__all__ == ["output_paths"]

    for legacy_name in (
        "download_image_if_url",
        "ensure_valid_prompt",
        "parse_resolution",
        "upload_and_get_final_output_location",
    ):
        assert legacy_name not in dir(utils)
