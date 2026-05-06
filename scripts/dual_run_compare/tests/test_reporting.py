from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from scripts.dual_run_compare.dual_run_compare import _finalize_args, build_parser, build_report, main
from scripts.dual_run_compare.oracles import REQUIRED_FULL_CANARY_SECTIONS
from scripts.dual_run_compare.reporting import evaluate_exit_policy, markdown_report


def _dry_run_report(report_id: str = "pytest-t11-report") -> dict:
    args = _finalize_args(build_parser().parse_args(["--dry-run", "--report-id", report_id]))
    return build_report(args)


def test_json_report_includes_route_sections_oracles_shadow_and_raw_refs() -> None:
    report = _dry_run_report()

    assert report["registry_coverage"]["active_api_owned_route_count"] == 14
    assert set(report["oracle_reports"]) == {
        "shared_path_oracles",
        "queue_contract",
        "runtime_metrics",
        "product_effect_oracles",
        "billing_idempotency_oracles",
    }
    assert report["route_status_counts"]["red"] == 6
    assert report["route_status_counts"]["wgp_only"] == 3

    video = next(route for route in report["routes"] if route["route_key"] == "video_enhance")
    assert video["status"] == "red"
    assert video["calibration_status"] == "not_in_thresholds"
    assert set(video["required_sections"]) == REQUIRED_FULL_CANARY_SECTIONS
    assert video["shadow_isolation"]["skipped_side_effects"]
    assert {
        "shared_path_oracles",
        "queue_contract",
        "runtime_metrics",
        "product_effect_oracles",
        "billing_idempotency_oracles",
        "shadow_isolation",
    }.issubset({section["source"] for section in video["sections"]})
    assert {
        "registry_snapshot",
        "full_canary_fixture",
        "shared_path_oracles",
        "queue_contract",
        "runtime_metrics",
        "product_effect_oracles",
        "billing_idempotency_oracles",
        "shadow_isolation",
    }.issubset({ref["kind"] for ref in video["raw_observation_refs"]})

    qwen = next(route for route in report["routes"] if route["route_key"] == "qwen_image")
    assert qwen["calibration_status"] == "deferred_pending_sprint_0c_disk"
    assert qwen["comparison"]["required_metric_keys"]
    assert qwen["metric_results"] == []

    sprint2_landed = {
        route["route_key"]: route
        for route in report["routes"]
        if route["route_key"] in {"z_image_turbo", "qwen_image_2512"}
    }
    assert set(sprint2_landed) == {"z_image_turbo", "qwen_image_2512"}
    for route in sprint2_landed.values():
        assert route["status"] == "red"
        assert route["landed_status"] == "sprint2_vibecomfy_direct_default_resolution_landed"
        assert route["report_status_policy"] == "red_or_green_required"

    wan = next(route for route in report["routes"] if route["route_key"] == "wan_2_2_t2i")
    assert wan["status"] == "wgp_only"
    assert wan["landed_status"] == "sprint2_wgp_only"

    wan_vace_fallbacks = {
        route["route_key"]: route
        for route in report["routes"]
        if route["route_key"]
        in {
            "travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default",
            "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default",
        }
    }
    assert set(wan_vace_fallbacks) == {
        "travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default",
        "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default",
    }
    for route in wan_vace_fallbacks.values():
        assert route["status"] == "wgp_only"
        assert route["calibration_status"] == "wgp_only"


def test_markdown_orders_red_routes_before_pending_and_fallback_routes() -> None:
    markdown = markdown_report(_dry_run_report())

    first_red = markdown.index("### `animate_character`")
    pending_route = markdown.index("### `qwen_image`")
    fallback_route = markdown.index("### `banodoco_render_timeline`")
    assert first_red < pending_route
    assert first_red < fallback_route
    assert "- Status: `red`" in markdown
    assert "## Registry And Oracles" in markdown


def test_exit_policy_rejects_landed_red_sparse_green_unclassified_and_oracle_failure() -> None:
    report = _dry_run_report()
    reasons = {reason["kind"] for reason in report["exit_policy"]["nonzero_reasons"]}
    assert report["exit_policy"]["exit_code"] == 1
    assert "landed_red_route" in reasons
    assert "required_oracle_failure" in reasons

    sparse_green = deepcopy(report)
    route = next(route for route in sparse_green["routes"] if route["route_key"] == "qwen_image")
    route["status"] = "green"
    route["comparison"]["missing_required_observations"] = ["image_ssim"]
    reasons = {reason["kind"] for reason in evaluate_exit_policy(sparse_green)["nonzero_reasons"]}
    assert "sparse_green_attempt" in reasons

    unclassified = deepcopy(report)
    unclassified["registry_coverage"]["unclassified_active_api_routes"] = ["new_active_route"]
    reasons = {reason["kind"] for reason in evaluate_exit_policy(unclassified)["nonzero_reasons"]}
    assert "unclassified_active_api_route" in reasons

    oracle_failure = deepcopy(report)
    oracle_failure["oracle_reports"]["billing_idempotency_oracles"]["status"] = "red"
    reasons = {reason["kind"] for reason in evaluate_exit_policy(oracle_failure)["nonzero_reasons"]}
    assert "required_oracle_failure" in reasons


def test_cli_writes_reports_and_returns_nonzero_for_landed_red_routes() -> None:
    report_id = "pytest-t11-cli"
    report_dir = Path(__file__).parents[1] / "reports"
    json_path = report_dir / f"{report_id}.json"
    markdown_path = report_dir / f"{report_id}.md"
    try:
        assert main(["--dry-run", "--report-id", report_id]) == 1
        assert json_path.exists()
        assert markdown_path.exists()
        assert '"exit_code": 1' in json_path.read_text()
        assert "### `animate_character`" in markdown_path.read_text()
    finally:
        json_path.unlink(missing_ok=True)
        markdown_path.unlink(missing_ok=True)


def test_workspace_root_wrapper_imports_worker_package_and_resolves_worker_paths() -> None:
    workspace_root = Path(__file__).parents[4]
    worker_root = workspace_root / "reigh-worker"
    report_id = "pytest-t12-root-wrapper"
    report_dir = worker_root / "scripts" / "dual_run_compare" / "reports"
    json_path = report_dir / f"{report_id}.json"
    markdown_path = report_dir / f"{report_id}.md"

    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/dual_run_compare.py",
                "--dry-run",
                "--report-id",
                report_id,
            ],
            cwd=workspace_root,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 1
        assert str(json_path) in result.stdout
        assert json_path.exists()
        report = json.loads(json_path.read_text())
        assert report["worker_root"] == str(worker_root)
        assert Path(report["threshold_path"]).resolve() == (
            worker_root / "scripts" / "dual_run_compare" / "migration-thresholds.yaml"
        )
        assert all(run["command"][1] == "scripts/live_test/main.py" for run in report["live_runs"])
        assert report["exit_policy"]["exit_code"] == 1
    finally:
        json_path.unlink(missing_ok=True)
        markdown_path.unlink(missing_ok=True)
