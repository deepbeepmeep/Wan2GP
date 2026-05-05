from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dual_run_compare.thresholds import Thresholds
from scripts.dual_run_compare.wgp_self_repeat import (
    INVALID_LIVE_TEST_ARGS,
    LiveRunConfig,
    build_live_test_command,
    compare_route_observations,
)


def test_live_test_command_uses_real_cli_timeout_flags() -> None:
    command = build_live_test_command(
        LiveRunConfig(
            variant="fresh",
            wgp_profile=3,
            timeout_image=11,
            timeout_travel_segment=22,
            timeout_travel_orchestrator=33,
        )
    )

    assert command[:2] == ["python", "scripts/live_test/main.py"]
    assert command[command.index("--variant") + 1] == "fresh"
    assert command[command.index("--wgp-profile") + 1] == "3"
    assert command[command.index("--timeout-image") + 1] == "11"
    assert command[command.index("--timeout-travel-segment") + 1] == "22"
    assert command[command.index("--timeout-travel-orchestrator") + 1] == "33"
    assert INVALID_LIVE_TEST_ARGS.isdisjoint(command)


def test_update_variant_is_not_positional_and_excludes_invalid_flags() -> None:
    command = build_live_test_command(
        LiveRunConfig(
            variant="update",
            wgp_profile=4,
            timeout_image=11,
            timeout_travel_segment=22,
            timeout_travel_orchestrator=33,
            pod_id="pod-123",
            no_terminate=True,
        )
    )

    assert "update" in command
    assert command[command.index("update") - 1] == "--variant"
    assert "--pod-id" in command
    assert "--no-terminate" in command
    assert INVALID_LIVE_TEST_ARGS.isdisjoint(command)


def test_command_builder_rejects_update_without_takeover_source() -> None:
    with pytest.raises(ValueError, match="update variant requires exactly one"):
        build_live_test_command(
            LiveRunConfig(
                variant="update",
                wgp_profile=3,
                timeout_image=11,
                timeout_travel_segment=22,
                timeout_travel_orchestrator=33,
            )
        )


def test_threshold_comparison_uses_thresholds_for_route() -> None:
    thresholds = Thresholds.load(strict=True)
    comparison = compare_route_observations(
        thresholds,
        "z_image_turbo",
        {
            "image_phash_normalized_hamming": 0.04,
            "image_ssim": 0.95,
            "image_pixel_dimensions": {"observed": "1024x1024", "expected": "1024x1024"},
        },
    )

    assert comparison["status"] == "green"
    assert comparison["calibration_status"] == thresholds.for_route("z_image_turbo").calibration_status
    assert [metric["passed"] for metric in comparison["metrics"]] == [True, True, True]


def test_record_deferral_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    report_id = "pytest-wgp-self-repeat"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dual_run_compare.wgp_self_repeat",
            "--record-deferral",
            "--route-key",
            "z_image_turbo",
            "--report-id",
            report_id,
            "--repo-root",
            str(Path(__file__).parents[3]),
            "--python-executable",
            "python",
        ],
        cwd=Path(__file__).parents[3],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report_dir = Path(__file__).parents[1] / "reports"
    json_path = report_dir / f"{report_id}.json"
    markdown_path = report_dir / f"{report_id}.md"
    try:
        report = json.loads(json_path.read_text())
        assert report["mode"] == "deferral"
        assert report["threshold_version"] == "0B-2026-05-05"
        assert report["routes"][0]["route_key"] == "z_image_turbo"
        command = report["live_runs"][0]["command"]
        assert "--timeout-image" in command
        assert "--timeout-travel-segment" in command
        assert "--timeout-travel-orchestrator" in command
        assert INVALID_LIVE_TEST_ARGS.isdisjoint(command)
        assert "WGP Self Repeatability Report" in markdown_path.read_text()
    finally:
        json_path.unlink(missing_ok=True)
        markdown_path.unlink(missing_ok=True)
