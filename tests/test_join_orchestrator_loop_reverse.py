"""Test that the join orchestrator loop_first_clip path calls reverse_video with kwargs.

Regression test for: reverse_video() called with positional args but the
transform_api wrapper only accepts **kwargs, causing TypeError at runtime.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy GPU/media deps so the orchestrator module can be imported
# without a full worker environment.
# ---------------------------------------------------------------------------
_HEAVY_DEPS = [
    "cv2", "torch", "torch.nn", "torch.nn.functional", "torch.cuda",
    "numpy", "numpy.typing", "PIL", "PIL.Image",
    "imageio", "imageio_ffmpeg", "mmgp", "diffusers", "transformers",
    "accelerate", "tqdm", "rife", "scipy", "scipy.ndimage",
]
for _mod in _HEAVY_DEPS:
    sys.modules.setdefault(_mod, MagicMock())

# Make source.media.video resolve any attribute lazily to a MagicMock
# so that imports like `from source.media.video import extract_frames_from_video`
# succeed without needing cv2/torch/etc.
import source.media.video as _video_pkg  # noqa: E402

_original_getattr = _video_pkg.__getattr__ if hasattr(_video_pkg, "__getattr__") else None


def _permissive_getattr(name: str):
    try:
        if _original_getattr:
            return _original_getattr(name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass
    mock = MagicMock(name=f"source.media.video.{name}")
    setattr(_video_pkg, name, mock)
    return mock


_video_pkg.__getattr__ = _permissive_getattr  # type: ignore[attr-defined]


@pytest.fixture()
def _mock_loop_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Mock external deps so the loop_first_clip code path can execute."""
    fake_clip = tmp_path / "first_clip.mp4"
    fake_clip.write_bytes(b"fake")

    fake_reversed = tmp_path / "clip_0_reversed.mp4"
    fake_reversed.write_bytes(b"fake-reversed")

    monkeypatch.setattr(
        "source.task_handlers.join.orchestrator.download_video_if_url",
        lambda *_args, **_kwargs: str(fake_clip),
    )
    monkeypatch.setattr(
        "source.task_handlers.join.orchestrator.upload_intermediate_file_to_storage",
        lambda **_kwargs: "https://storage/reversed.mp4",
    )

    return fake_reversed


def test_reverse_video_called_with_kwargs(
    _mock_loop_deps: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """reverse_video must be called with keyword args (transform_api wrapper uses **kwargs)."""
    from source.task_handlers.join.orchestrator import (
        _handle_join_clips_orchestrator_task,
    )

    fake_reversed = _mock_loop_deps
    captured_calls: list[dict] = []

    def spy_reverse_video(**kwargs):
        captured_calls.append(kwargs)
        return fake_reversed

    monkeypatch.setattr(
        "source.task_handlers.join.orchestrator.reverse_video",
        spy_reverse_video,
    )

    task_params = {
        "orchestrator_details": {
            "orchestrator_task_id_ref": "orch-loop-1",
            "run_id": "run-loop-test",
            "segment_task_ids": [],
            "clip_list": [
                {"url": "https://example.com/clip_a.mp4", "name": "clip_0"},
                {"url": "https://example.com/clip_b.mp4", "name": "clip_1"},
            ],
            "loop_first_clip": True,
        },
    }

    _handle_join_clips_orchestrator_task(
        task_params,
        main_output_dir_base=tmp_path,
        orchestrator_task_id_str="orch-loop-1",
        orchestrator_project_id="proj-1",
    )

    assert len(captured_calls) == 1, "reverse_video should be called exactly once"
    call = captured_calls[0]
    assert "input_video_path" in call, "must pass input_video_path as keyword arg"
    assert "output_video_path" in call, "must pass output_video_path as keyword arg"
