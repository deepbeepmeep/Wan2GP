from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.canary_readiness.package import build_package, load_source_reports, write_package


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
    assert package["sections"]["non_rayworker_smoke"]["status"] == "green"
    assert package["exit_policy"]["exit_code"] == 1
    assert {
        reason["section"] for reason in package["exit_policy"]["nonzero_reasons"]
    }.issuperset({"soak", "dashboards", "alerts", "rollback_exercise", "go_no_go"})


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
    assert "not green" in package["sections"]["non_rayworker_smoke"]["reasons"][0]


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

