from __future__ import annotations

import pytest

from scripts.dual_run_compare.compare import compare_route_observations
from scripts.dual_run_compare.status import (
    PENDING,
    RED,
    REPORT_STATUSES,
    SECTION_PENDING_NOT_IMPLEMENTED,
    WGP_ONLY,
    map_calibration_status_to_report_status,
    pending_not_implemented_section,
)
from scripts.dual_run_compare.thresholds import APPROVED_CALIBRATION_STATUSES, Thresholds


def test_report_statuses_are_separate_from_calibration_statuses() -> None:
    assert "pending_not_implemented" not in APPROVED_CALIBRATION_STATUSES
    assert "fallback" not in APPROVED_CALIBRATION_STATUSES
    assert "pending" not in APPROVED_CALIBRATION_STATUSES
    assert "pending_not_implemented" not in REPORT_STATUSES


def test_calibration_status_maps_explicitly_to_initial_report_status() -> None:
    assert map_calibration_status_to_report_status("green") == PENDING
    assert map_calibration_status_to_report_status("pending_calibration") == PENDING
    assert map_calibration_status_to_report_status("deferred_pending_sprint_0c_disk") == PENDING
    assert map_calibration_status_to_report_status("owner_deferred") == PENDING
    assert map_calibration_status_to_report_status("wgp_only") == WGP_ONLY

    with pytest.raises(ValueError, match="unknown calibration status"):
        map_calibration_status_to_report_status("pending_not_implemented")


def test_sparse_required_observations_are_red_for_landed_routes() -> None:
    thresholds = Thresholds.load(strict=True)
    comparison = compare_route_observations(
        thresholds,
        "z_image_turbo",
        {"image_ssim": 0.95},
        required_metric_keys=["image_ssim", "image_pixel_dimensions"],
        landed=True,
    )

    assert comparison["status"] == RED
    assert comparison["report_status"] == RED
    assert comparison["missing_required_observations"] == ["image_pixel_dimensions"]
    missing_section = next(section for section in comparison["sections"] if section["key"] == "image_pixel_dimensions")
    assert missing_section["status"] == "missing_evidence"


def test_sparse_required_observations_are_pending_for_unclear_routes() -> None:
    thresholds = Thresholds.load(strict=True)
    comparison = compare_route_observations(
        thresholds,
        "z_image_turbo",
        {"image_ssim": 0.95},
        required_metric_keys=["image_ssim", "image_pixel_dimensions"],
        landed=False,
    )

    assert comparison["status"] == PENDING
    assert comparison["initial_report_status"] == PENDING
    assert comparison["missing_required_observations"] == ["image_pixel_dimensions"]


def test_complete_required_observations_can_be_green() -> None:
    thresholds = Thresholds.load(strict=True)
    comparison = compare_route_observations(
        thresholds,
        "z_image_turbo",
        {
            "image_ssim": 0.95,
            "image_pixel_dimensions": {"observed": "1024x1024", "expected": "1024x1024"},
        },
        required_metric_keys=["image_ssim", "image_pixel_dimensions"],
        landed=True,
    )

    assert comparison["status"] == "green"
    assert comparison["missing_required_observations"] == []
    assert {section["status"] for section in comparison["sections"]} == {"green"}


def test_pending_not_implemented_is_section_only() -> None:
    section = pending_not_implemented_section("refund", "no refund ledger path discovered")

    assert section == {
        "key": "refund",
        "status": SECTION_PENDING_NOT_IMPLEMENTED,
        "reason": "no refund ledger path discovered",
    }
