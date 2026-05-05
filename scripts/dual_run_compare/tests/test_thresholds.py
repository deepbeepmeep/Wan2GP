from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.dual_run_compare.thresholds import (
    DEFAULT_PATH,
    EXPECTED_METRIC_KEYS,
    ThresholdValidationError,
    Thresholds,
)


def _write_thresholds(tmp_path: Path, mutator=None) -> Path:
    data = yaml.safe_load(DEFAULT_PATH.read_text())
    if mutator is not None:
        mutator(data)
    path = tmp_path / "migration-thresholds.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_loads_complete_threshold_yaml() -> None:
    thresholds = Thresholds.load(strict=True)
    assert thresholds.raw["version"] == "0B-2026-05-05"
    assert tuple(thresholds.raw["metric_keys"]) == EXPECTED_METRIC_KEYS
    assert tuple(thresholds.raw["defaults"]) == EXPECTED_METRIC_KEYS
    assert tuple(thresholds.metrics) == EXPECTED_METRIC_KEYS
    assert "z_image_turbo" in thresholds.routes


def test_route_override_merge(tmp_path: Path) -> None:
    def mutate(data):
        data["routes"]["z_image_turbo"]["thresholds_override"] = {
            "image_ssim": {"threshold": 0.95, "calibration_note": "tightened for deterministic fixture"}
        }

    path = _write_thresholds(tmp_path, mutate)
    route = Thresholds.load(path).for_route("z_image_turbo")
    assert route.metrics["image_ssim"]["threshold"] == 0.95
    assert route.metrics["image_ssim"]["calibration_note"] == "tightened for deterministic fixture"
    assert route.metrics["image_phash_normalized_hamming"]["threshold"] == 0.05


def test_invalid_status_rejected(tmp_path: Path) -> None:
    def mutate(data):
        data["routes"]["z_image_turbo"]["calibration_status"] = "almost_green"

    path = _write_thresholds(tmp_path, mutate)
    with pytest.raises(ThresholdValidationError, match="invalid calibration_status"):
        Thresholds.load(path)


def test_missing_metric_rejected(tmp_path: Path) -> None:
    def mutate(data):
        data["defaults"].pop("video_phash_p95")

    path = _write_thresholds(tmp_path, mutate)
    with pytest.raises(ThresholdValidationError, match="defaults missing metric rows: video_phash_p95"):
        Thresholds.load(path)


def test_metric_key_list_drift_rejected(tmp_path: Path) -> None:
    def mutate(data):
        data["metric_keys"].remove("video_phash_p95")

    path = _write_thresholds(tmp_path, mutate)
    with pytest.raises(ThresholdValidationError, match="metric_keys missing rows: video_phash_p95"):
        Thresholds.load(path)


def test_unknown_route_override_rejected_by_strict_cli(tmp_path: Path) -> None:
    def mutate(data):
        data["routes"]["z_image_turbo"]["thresholds_override"] = {
            "not_a_metric": {"threshold": 123}
        }

    path = _write_thresholds(tmp_path, mutate)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dual_run_compare.check_thresholds",
            "--strict",
            "--path",
            str(path),
        ],
        cwd=Path(__file__).parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "unknown threshold overrides: not_a_metric" in result.stderr


def test_manifest_yaml_drift_rejected_when_manifests_exist(tmp_path: Path) -> None:
    path = _write_thresholds(tmp_path)
    manifest_dir = tmp_path / "golden" / "orphan_route"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text("{}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dual_run_compare.check_thresholds",
            "--strict",
            "--path",
            str(path),
        ],
        cwd=Path(__file__).parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "golden manifests without YAML routes: orphan_route" in result.stderr
    assert "routes missing golden manifests:" in result.stderr
