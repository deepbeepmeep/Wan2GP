import sys
import types

import numpy as np

from source.media.structure import preprocessors


class _FakePoseAnnotator:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, frames):
        return [np.zeros_like(frame) for frame in frames]


def _install_fake_pose_module(monkeypatch):
    fake_pose_module = types.ModuleType("Wan2GP.preprocessing.dwpose.pose")
    fake_pose_module.PoseBodyFaceVideoAnnotator = _FakePoseAnnotator
    monkeypatch.setitem(sys.modules, "Wan2GP.preprocessing.dwpose.pose", fake_pose_module)
    monkeypatch.setattr(preprocessors.Path, "exists", lambda self: True)


def test_get_structure_preprocessor_pose_returns_callable(monkeypatch):
    _install_fake_pose_module(monkeypatch)

    pose_preprocessor = preprocessors.get_structure_preprocessor("pose")
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    processed = pose_preprocessor(frames)

    assert callable(pose_preprocessor)
    assert len(processed) == 1
    assert processed[0].shape == frames[0].shape


def test_process_structure_frames_accepts_pose(monkeypatch):
    _install_fake_pose_module(monkeypatch)

    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]

    processed = preprocessors.process_structure_frames(
        frames=frames,
        structure_type="pose",
        motion_strength=1.0,
        canny_intensity=1.0,
        depth_contrast=1.0,
    )

    assert len(processed) == len(frames)
