from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.canary_readiness.non_rayworker import BILLING_HANDLER, COMPLETION_HANDLER, REQUIRED_ROUTES
from scripts.canary_readiness.package import build_package, load_source_reports, write_package
from scripts.canary_readiness.soak import REQUIRED_SOAK_SCENARIOS


def _write_report(path: Path, *, report_id: str, route_status: str = "green", exit_code: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "report_id": report_id,
                "created_at": "2026-05-06T00:00:00+00:00",
                "mode": "dry_run",
                "routes": [
                    {"route_key": "video_enhance", "status": route_status},
                    {"route_key": "image-upscale", "status": route_status},
                    {"route_key": "animate_character", "status": route_status},
                    {"route_key": "flux_klein_edit", "status": route_status},
                ],
                "exit_policy": {"exit_code": exit_code, "nonzero_reasons": []},
            }
        ),
        encoding="utf-8",
    )


def _complete_manifest() -> dict:
    observed_at = datetime.now(timezone.utc).isoformat()
    return {
        "non_rayworker_observations": [
            {
                "environment": "staging",
                "observed_at": observed_at,
                "task_id": f"task-{route_key}",
                "task_type": route_key,
                "route_key": route_key,
                "runtime": {"backend": "api_orchestrator", "pool": "non-rayworker"},
                "selector_namespace": "canary",
                "selector_version": "2026-05-06",
                "status": "completed",
                "completion_evidence": {"handler": COMPLETION_HANDLER, "status": "completed"},
                "billing_evidence": {"handler": BILLING_HANDLER, "status": "charged"},
                "source_ref": {"kind": "system_logs", "id": f"log-{route_key}"},
                "redaction": {"status": "redacted", "secret_scan": "passed"},
            }
            for route_key in REQUIRED_ROUTES
        ],
        "soak_scenarios": [
            {
                "scenario": scenario,
                "status": "pass",
                "observed_at": observed_at,
                "route_labels": {"route_key": "travel_segment", "selector_namespace": "canary"},
                "pool_labels": {"pool": "vibecomfy-canary", "backend": "rayworker"},
                "evidence_refs": [{"kind": "system_logs", "id": f"log-{scenario}"}],
            }
            for scenario in REQUIRED_SOAK_SCENARIOS
        ],
        "dashboards": {
            "status": "green",
            "evidence_refs": ["dashboard-export://canary-panels"],
        },
        "alerts": {
            "status": "green",
            "evidence_refs": ["config/alerts/section11-canary.yaml"],
        },
        "rollback_exercise": {
            "observed_at": observed_at,
            "environment": "staging",
            "operator": "pytest",
            "source_ref": {"kind": "system_logs", "id": "rollback-exercise"},
            "status": "pass",
        },
        "go_no_go": {
            "decision": "go",
            "evidence_refs": ["canary-readiness-sample"],
        },
    }


def test_load_source_reports_excludes_generated_canary_readiness_outputs(tmp_path: Path) -> None:
    source_dir = tmp_path / "dual_run_reports"
    source_dir.mkdir()
    _write_report(source_dir / "dual-run-source.json", report_id="dual-run-source")
    _write_report(source_dir / "canary-readiness-old.json", report_id="canary-readiness-old")

    reports = load_source_reports(source_dir)

    assert [report.path.name for report in reports] == ["dual-run-source.json"]


def test_build_package_writes_stable_json_and_markdown_to_non_input_output_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "dual_run_reports"
    output_dir = tmp_path / "canary_readiness_reports"
    source_dir.mkdir()
    _write_report(source_dir / "dual-run-source.json", report_id="dual-run-source")

    package = build_package(
        package_id="canary-readiness-pytest",
        source_report_dir=source_dir,
        output_dir=output_dir,
        created_at="2026-05-06T00:00:00+00:00",
    )
    json_path, markdown_path = write_package(package, output_dir=output_dir)

    assert json_path.parent == output_dir
    assert markdown_path.parent == output_dir
    assert output_dir != source_dir
    assert json.loads(json_path.read_text())["source_reports"][0]["report_id"] == "dual-run-source"
    assert "# Canary Readiness Package: canary-readiness-pytest" in markdown_path.read_text()
    assert package["sections"]["non_rayworker_smoke"]["status"] == "red"
    assert package["exit_policy"]["exit_code"] == 1
    assert {
        reason["section"] for reason in package["exit_policy"]["nonzero_reasons"]
    }.issuperset({"non_rayworker_smoke", "soak", "dashboards", "alerts", "rollback_exercise", "go_no_go"})


def test_package_marks_missing_source_reports_and_red_non_rayworker_as_hard_gates(tmp_path: Path) -> None:
    empty = build_package(
        package_id="canary-readiness-empty",
        source_report_dir=tmp_path / "missing",
        output_dir=tmp_path / "out",
        created_at="2026-05-06T00:00:00+00:00",
    )
    assert empty["sections"]["prerequisite_evidence"]["status"] == "red"
    assert empty["sections"]["non_rayworker_smoke"]["status"] == "red"
    assert empty["exit_policy"]["exit_code"] == 1

    source_dir = tmp_path / "dual_run_reports"
    source_dir.mkdir()
    _write_report(source_dir / "dual-run-source.json", report_id="dual-run-source", route_status="red")
    package = build_package(
        package_id="canary-readiness-red-routes",
        source_report_dir=source_dir,
        output_dir=tmp_path / "out",
        created_at="2026-05-06T00:00:00+00:00",
    )
    assert package["sections"]["non_rayworker_smoke"]["status"] == "red"
    assert "live/staging" in package["sections"]["non_rayworker_smoke"]["reasons"][0]


def test_cli_uses_dual_run_reports_as_inputs_and_returns_nonzero_for_missing_hard_gates(tmp_path: Path) -> None:
    worker_root = Path(__file__).parents[3]
    source_dir = tmp_path / "dual_run_reports"
    output_dir = tmp_path / "canary_readiness_reports"
    source_dir.mkdir()
    _write_report(source_dir / "dual-run-source.json", report_id="dual-run-source")
    _write_report(source_dir / "canary-readiness-old.json", report_id="canary-readiness-old")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.canary_readiness",
            "--source-report-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--package-id",
            "canary-readiness-cli",
        ],
        cwd=worker_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert (output_dir / "canary-readiness-cli.json").exists()
    package = json.loads((output_dir / "canary-readiness-cli.json").read_text())
    assert [source["report_id"] for source in package["source_reports"]] == ["dual-run-source"]
    assert package["source_report_dir"] != package["output_dir"]


def test_complete_redacted_manifest_is_required_for_zero_exit(tmp_path: Path) -> None:
    source_dir = tmp_path / "dual_run_reports"
    source_dir.mkdir()
    _write_report(source_dir / "dual-run-source.json", report_id="dual-run-source")

    package = build_package(
        package_id="canary-readiness-complete",
        source_report_dir=source_dir,
        output_dir=tmp_path / "out",
        evidence_manifest=_complete_manifest(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    assert package["sections"]["non_rayworker_smoke"]["status"] == "green"
    assert package["sections"]["soak"]["status"] == "green"
    assert package["sections"]["rollback_exercise"]["status"] == "green"
    assert package["exit_policy"]["exit_code"] == 0


def test_cli_fixture_only_non_rayworker_evidence_exits_nonzero(tmp_path: Path) -> None:
    worker_root = Path(__file__).parents[3]
    source_dir = tmp_path / "dual_run_reports"
    output_dir = tmp_path / "canary_readiness_reports"
    manifest_path = tmp_path / "fixture-only.json"
    source_dir.mkdir()
    _write_report(source_dir / "dual-run-source.json", report_id="dual-run-source")
    manifest = _complete_manifest()
    manifest["non_rayworker_observations"][0]["fixture_only"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.canary_readiness",
            "--source-report-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--evidence-manifest",
            str(manifest_path),
            "--package-id",
            "canary-readiness-fixture-only",
        ],
        cwd=worker_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    package = json.loads((output_dir / "canary-readiness-fixture-only.json").read_text())
    assert package["sections"]["non_rayworker_smoke"]["status"] == "red"
    assert "fixture-only" in json.dumps(package["sections"]["non_rayworker_smoke"])
