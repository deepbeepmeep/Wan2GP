from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from source.media import video_contract
from source.media.video_contract import (
    VideoArtifactContract,
    VideoContractError,
    validate_video_artifact,
)


def _ffprobe_payload(*, frames="49", fps="24/1", duration="2.041667", audio=True, width=1280, height=720):
    streams = [
        {
            "codec_type": "video",
            "width": width,
            "height": height,
            "nb_frames": frames,
            "avg_frame_rate": fps,
            "duration": duration,
        }
    ]
    if audio:
        streams.append({"codec_type": "audio", "duration": duration})
    return {"streams": streams, "format": {"duration": duration}}


def _patch_ffprobe(monkeypatch, payload, *, returncode=0, stderr=""):
    def _run(command, text, capture_output, check):
        assert command[0] == "ffprobe"
        return SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps(payload),
            stderr=stderr,
        )

    monkeypatch.setattr(video_contract.subprocess, "run", _run)


def test_validate_video_artifact_accepts_matching_ffprobe_contract(monkeypatch, tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"fake")
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"fake")
    _patch_ffprobe(monkeypatch, _ffprobe_payload())

    metadata = validate_video_artifact(
        video,
        VideoArtifactContract(
            expected_frame_count=49,
            expected_fps=24,
            require_audio=True,
            expected_width=1280,
            expected_height=720,
            require_thumbnail=True,
            thumbnail_path=str(thumbnail),
        ),
    )

    assert metadata.content_type == "video/mp4"
    assert metadata.frame_count == 49
    assert metadata.fps == 24
    assert metadata.has_audio is True


@pytest.mark.parametrize(
    ("contract", "payload", "message"),
    [
        (VideoArtifactContract(expected_frame_count=50), _ffprobe_payload(frames="49"), "frame count mismatch"),
        (VideoArtifactContract(expected_fps=30), _ffprobe_payload(fps="24/1"), "fps mismatch"),
        (VideoArtifactContract(require_audio=True), _ffprobe_payload(audio=False), "audio stream is required"),
        (VideoArtifactContract(expected_width=1024), _ffprobe_payload(width=1280), "width mismatch"),
        (VideoArtifactContract(expected_height=576), _ffprobe_payload(height=720), "height mismatch"),
    ],
)
def test_validate_video_artifact_rejects_contract_violations(monkeypatch, tmp_path, contract, payload, message):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"fake")
    _patch_ffprobe(monkeypatch, payload)

    with pytest.raises(VideoContractError, match=message):
        validate_video_artifact(video, contract)


def test_validate_video_artifact_rejects_malformed_or_non_video_artifacts(monkeypatch, tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"fake")
    _patch_ffprobe(monkeypatch, {"streams": [{"codec_type": "audio"}], "format": {}})

    with pytest.raises(VideoContractError, match="no video stream"):
        validate_video_artifact(video, VideoArtifactContract())


def test_validate_video_artifact_rejects_missing_thumbnail(tmp_path, monkeypatch):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"fake")
    _patch_ffprobe(monkeypatch, _ffprobe_payload())

    with pytest.raises(VideoContractError, match="thumbnail is required"):
        validate_video_artifact(video, VideoArtifactContract(require_thumbnail=True))
