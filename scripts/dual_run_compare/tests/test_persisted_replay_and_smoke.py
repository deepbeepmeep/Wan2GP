from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.persisted_row_replay import (
    NESTED_SNAPSHOT_KEYS,
    SNAPSHOT_KEYS,
    ReplayError,
    replay_persisted_rows,
)
from scripts.section3a_matrix_smoke import REPORT_COLUMNS, SmokeError, run_section3a_smoke


def test_persisted_row_replay_defaults_to_dry_run_and_compares_full_snapshot_shape(tmp_path: Path) -> None:
    report_path = tmp_path / "persisted-row-replay.json"

    report = replay_persisted_rows(report_path=report_path)

    assert report["dry_run"] is True
    assert report["summary"] == {"total": 3, "passed": 3, "failed": 0}
    assert report_path.exists()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["summary"] == report["summary"]

    for row in report["rows"]:
        assert row["replay_result"] == "pass"
        assert row["media_contract_result"] == "not_applicable_dry_run"
        assert row["snapshot_shape_result"] == "pass"
        assert set(SNAPSHOT_KEYS).issubset(row["normalized_snapshot"])
        nested = row["normalized_snapshot"]["route_selection_snapshot"]
        assert set(NESTED_SNAPSHOT_KEYS).issubset(nested)
        assert row["route_key"] == row["current_snapshot"]["route_key"]


def test_persisted_row_replay_reports_route_mismatch_failures(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "bad.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "fixture_id": "bad-route",
                        "task_type": "travel_segment",
                        "backend": "wgp",
                        "params": {
                            "model_name": "ltx2_22B",
                            "continuity_case": "first_last",
                        },
                        "route_contract": {
                            "route_key": "wrong",
                            "selected_backend": "wgp",
                        },
                        "expected_route_key": "also_wrong",
                        "expected_backend": "wgp",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = replay_persisted_rows(fixtures, report_path=tmp_path / "report.json")

    assert report["summary"] == {"total": 1, "passed": 0, "failed": 1}
    assert report["rows"][0]["replay_result"] == "fail"
    assert "does not match" in report["rows"][0]["wgp_only_or_blocked_reason"]


def test_persisted_row_replay_rejects_live_without_explicit_enqueue_flag() -> None:
    with pytest.raises(ReplayError, match="requires --allow-live-enqueue"):
        replay_persisted_rows(report_path=None, live=True)


def test_section3a_smoke_writes_markdown_and_json_with_required_columns(tmp_path: Path) -> None:
    markdown_report = tmp_path / "section3a-smoke.md"
    json_report = tmp_path / "section3a-smoke.json"

    report = run_section3a_smoke(report_path=markdown_report, json_report_path=json_report)

    assert report["dry_run"] is True
    assert report["columns"] == list(REPORT_COLUMNS)
    assert report["summary"]["total"] == 13
    assert report["summary"]["failed"] == 0
    assert report["summary"]["blocked"] == 7
    assert report["summary"]["reasoned_rows"] == 13
    assert {row["row_id"] for row in report["rows"]} == set(range(1, 14))
    assert all(row["replay_result"] == "pass" for row in report["rows"])
    assert all(row["media_contract_result"] == "not_applicable_dry_run" for row in report["rows"])

    blocked_rows = [row for row in report["rows"] if row["disposition"] == "BLOCKED"]
    assert {row["row_id"] for row in blocked_rows} == {7, 8, 9, 10, 11, 12, 13}
    assert all(row["wgp_only_or_blocked_reason"] for row in blocked_rows)

    markdown = markdown_report.read_text(encoding="utf-8")
    assert "| " + " | ".join(REPORT_COLUMNS) + " |" in markdown
    assert "travel_segment__model-ltx2_distilled__guidance-ltx_control_pose" in markdown
    assert json.loads(json_report.read_text(encoding="utf-8"))["summary"] == report["summary"]


def test_section3a_smoke_rejects_live_without_explicit_enqueue_flag() -> None:
    with pytest.raises(SmokeError, match="requires --allow-live-enqueue"):
        run_section3a_smoke(report_path=None, json_report_path=None, live=True)
