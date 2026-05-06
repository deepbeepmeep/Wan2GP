from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.section3a_matrix import (
    DEFAULT_DOC_PATH,
    Section3ADocsSyncError,
    assert_docs_synced,
    fixture_as_dicts,
    load_fixture,
    parse_docs_matrix,
)


REQUIRED_FIELDS = {
    "row_id",
    "task_type",
    "ui_model_id",
    "worker_model_name",
    "guidance_kind",
    "guidance_mode",
    "continuity_case",
    "profile",
    "disposition",
    "expected_backend",
    "route_key_expectation",
    "support_state_expectation",
    "blocking_reason",
}


def test_section3a_fixture_contains_rows_1_through_13_with_required_fields() -> None:
    rows = fixture_as_dicts()

    assert [row["row_id"] for row in rows] == list(range(1, 14))
    for row in rows:
        assert set(row) == REQUIRED_FIELDS
        assert row["task_type"] == "travel_segment"
        assert row["continuity_case"] == "first_last"
        assert row["profile"] == "default"


def test_section3a_fixture_keeps_disposition_separate_from_runtime_support_state() -> None:
    rows = load_fixture()
    rows_by_id = {row.row_id: row for row in rows}

    assert {rows_by_id[row_id].disposition for row_id in range(1, 7)} == {"NEW"}
    assert {rows_by_id[row_id].disposition for row_id in range(9, 14)} == {"BLOCKED"}
    assert {
        rows_by_id[row_id].support_state_expectation for row_id in [*range(1, 7), *range(9, 14)]
    } == {"vibecomfy_unsupported"}
    assert all(rows_by_id[row_id].blocking_reason for row_id in [*range(1, 7), *range(9, 14)])


def test_section3a_fixture_has_mode_aware_route_key_expectations() -> None:
    rows = load_fixture()
    rows_by_id = {row.row_id: row for row in rows}

    assert rows_by_id[2].route_key_expectation != rows_by_id[3].route_key_expectation
    assert rows_by_id[2].route_key_expectation.endswith("__guidance-vace_flow__continuity-first_last__profile-default")
    assert rows_by_id[3].route_key_expectation.endswith("__guidance-vace_canny__continuity-first_last__profile-default")
    assert rows_by_id[9].route_key_expectation.endswith(
        "__guidance-ltx_control_video__continuity-first_last__profile-default"
    )
    assert rows_by_id[13].route_key_expectation.endswith(
        "__guidance-ltx_control_cameraman__continuity-first_last__profile-default"
    )


def test_section3a_docs_matrix_is_synced_with_fixture() -> None:
    assert_docs_synced()
    docs_rows = parse_docs_matrix()
    assert sorted(docs_rows) == list(range(1, 14))


def test_section3a_docs_sync_fails_when_docs_disagree(tmp_path: Path) -> None:
    altered_doc = tmp_path / "migration-vibecomfy.md"
    altered_doc.write_text(
        DEFAULT_DOC_PATH.read_text(encoding="utf-8").replace(
            "| 8 | ltx / `ltx-2.3-fast` | i2v | none |",
            "| 8 | ltx / `ltx-2.3-fast` | i2v | ltx_control:video |",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(Section3ADocsSyncError, match="row 8 guidance_kind"):
        assert_docs_synced(doc_path=altered_doc)


def test_section3a_docs_sync_fails_when_docs_drop_a_row(tmp_path: Path) -> None:
    rows = load_fixture()
    dropped = [row for row in rows if row.row_id != 13]

    with pytest.raises(Section3ADocsSyncError, match="row ids differ"):
        assert_docs_synced(rows=dropped)


def test_section3a_fixture_validation_rejects_blocked_runtime_support() -> None:
    rows = load_fixture()
    bad_rows = [
        dataclasses.replace(row, support_state_expectation="vibecomfy_supported")
        if row.row_id == 9
        else row
        for row in rows
    ]

    with pytest.raises(ValueError, match="BLOCKED fixture rows must not claim runtime support"):
        from scripts.section3a_matrix import validate_fixture_rows

        validate_fixture_rows(bad_rows)
