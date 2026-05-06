from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts.dual_run_compare.compare import compare_route_observations
from scripts.dual_run_compare.media import compare_image_artifacts, compare_video_artifacts
from scripts.dual_run_compare.thresholds import Thresholds


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def test_image_adapter_emits_similarity_observations(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    _write_png(reference, (20, 80, 140))
    _write_png(candidate, (20, 80, 140))

    result = compare_image_artifacts(reference, candidate)

    assert result["status"] == "observed"
    assert result["missing_evidence"] == []
    assert result["observations"]["image_phash_normalized_hamming"] == 0.0
    assert result["observations"]["image_ssim"] == pytest.approx(1.0)
    assert result["observations"]["image_pixel_dimensions"] == {
        "observed": "32x32",
        "expected": "32x32",
    }
    assert result["observations"]["image_format_container"] == {
        "observed": "png",
        "expected": "png",
    }


def test_missing_image_artifact_feeds_strict_missing_evidence(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    candidate = tmp_path / "candidate.png"
    _write_png(candidate, (0, 0, 0))

    media_result = compare_image_artifacts(missing, candidate)
    comparison = compare_route_observations(
        Thresholds.load(strict=True),
        "z_image_turbo",
        media_result["observations"],
        required_metric_keys=media_result["required_metric_keys"],
        landed=True,
    )

    assert media_result["status"] == "missing_evidence"
    assert {item["key"] for item in media_result["missing_evidence"]} == set(media_result["required_metric_keys"])
    assert comparison["status"] == "red"
    assert set(comparison["missing_required_observations"]) == set(media_result["required_metric_keys"])


def _write_test_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (24, 24))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer unavailable")
    try:
        for index in range(4):
            frame = np.full((24, 24, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_adapter_emits_runtime_media_observations(tmp_path: Path) -> None:
    reference = tmp_path / "reference.avi"
    candidate = tmp_path / "candidate.avi"
    _write_test_video(reference)
    _write_test_video(candidate)

    result = compare_video_artifacts(reference, candidate)

    assert result["status"] == "observed"
    assert result["observations"]["video_frame_count"] == {
        "observed": 4,
        "expected": 4,
    }
    assert result["observations"]["video_phash_mean"] == pytest.approx(0.0)
    assert result["observations"]["video_phash_p95"] == pytest.approx(0.0)
    assert result["observations"]["video_duration_ms"]["observed"] == pytest.approx(800.0)
    assert result["observations"]["video_fps"]["observed"] == pytest.approx(5.0)
    assert result["optional_metric_results"]["video_audio_duration_ms"]["status"] == "not_applicable"


def test_missing_video_artifact_is_explicit_missing_evidence(tmp_path: Path) -> None:
    result = compare_video_artifacts(tmp_path / "missing-reference.avi", tmp_path / "missing-candidate.avi")

    assert result["status"] == "missing_evidence"
    assert {item["key"] for item in result["missing_evidence"]} == set(result["required_metric_keys"])
