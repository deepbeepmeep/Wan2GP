from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_FIXTURE_PATH = SCRIPT_DIR / "dual_run_compare" / "fixtures" / "section3a_matrix.fixture"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "migration-vibecomfy.md"

DISPOSITIONS = {"ADAPT", "BLOCKED", "FALL-BACK", "NEW", "NATIVE", "WGP-only"}
SUPPORT_STATES = {"vibecomfy_supported", "vibecomfy_unsupported", "wgp_only"}
EXPECTED_BACKENDS = {"vibecomfy", "wgp"}


@dataclass(frozen=True)
class Section3ARow:
    row_id: int
    task_type: str
    ui_model_id: str
    worker_model_name: str
    guidance_kind: str
    guidance_mode: str
    continuity_case: str
    profile: str
    disposition: str
    expected_backend: str
    route_key_expectation: str
    support_state_expectation: str
    blocking_reason: str | None


class Section3AMatrixError(ValueError):
    pass


class Section3ADocsSyncError(Section3AMatrixError):
    pass


def load_fixture(path: Path = DEFAULT_FIXTURE_PATH) -> list[Section3ARow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_fields = tuple(payload["required_fields"])
    rows: list[Section3ARow] = []
    for raw in payload["rows"]:
        missing = [field for field in required_fields if field not in raw]
        if missing:
            raise Section3AMatrixError(f"row {raw.get('row_id', '<unknown>')} missing fields: {missing}")
        rows.append(Section3ARow(**{field: raw[field] for field in required_fields}))
    validate_fixture_rows(rows)
    return rows


def validate_fixture_rows(rows: Iterable[Section3ARow]) -> None:
    rows = list(rows)
    row_ids = [row.row_id for row in rows]
    if row_ids != list(range(1, 14)):
        raise Section3AMatrixError(f"expected Section 3A rows 1-13, got {row_ids}")
    route_keys = [row.route_key_expectation for row in rows]
    if len(set(route_keys)) != len(route_keys):
        raise Section3AMatrixError("route_key_expectation values must be unique")
    for row in rows:
        if row.disposition not in DISPOSITIONS:
            raise Section3AMatrixError(f"row {row.row_id} has invalid disposition {row.disposition!r}")
        if row.expected_backend not in EXPECTED_BACKENDS:
            raise Section3AMatrixError(f"row {row.row_id} has invalid expected_backend {row.expected_backend!r}")
        if row.support_state_expectation not in SUPPORT_STATES:
            raise Section3AMatrixError(
                f"row {row.row_id} has invalid support_state_expectation "
                f"{row.support_state_expectation!r}"
            )
        if row.disposition in {"BLOCKED", "NEW", "WGP-only"} and not row.blocking_reason:
            raise Section3AMatrixError(f"row {row.row_id} requires a blocking_reason")
        if row.disposition == "BLOCKED" and row.support_state_expectation not in {
            "vibecomfy_unsupported",
            "wgp_only",
        }:
            raise Section3AMatrixError("BLOCKED fixture rows must not claim runtime support")


def parse_docs_matrix(doc_path: Path = DEFAULT_DOC_PATH) -> dict[int, dict[str, str]]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == "### Travel-segment configuration matrix"),
        None,
    )
    if start is None:
        raise Section3ADocsSyncError("could not find Section 3A travel matrix heading")

    header_index = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index].startswith("| # | model_family / id | exec mode | guidance kind:mode |")
        ),
        None,
    )
    if header_index is None:
        raise Section3ADocsSyncError("could not find Section 3A travel matrix table header")

    headers = [_clean_markdown_cell(cell) for cell in _split_markdown_row(lines[header_index])]
    parsed: dict[int, dict[str, str]] = {}
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [_clean_markdown_cell(cell) for cell in _split_markdown_row(line)]
        if len(cells) != len(headers):
            raise Section3ADocsSyncError(f"malformed matrix row: {line}")
        record = dict(zip(headers, cells, strict=True))
        try:
            row_id = int(record["#"])
        except ValueError as exc:
            raise Section3ADocsSyncError(f"invalid matrix row id: {record['#']!r}") from exc
        parsed[row_id] = {
            "ui_model_id": _extract_backticked(record["model_family / id"]),
            "guidance_kind": _normalize_guidance(record["guidance kind:mode"])[0],
            "guidance_mode": _normalize_guidance(record["guidance kind:mode"])[1],
            "disposition": _extract_disposition(record["disposition"]),
        }
    return parsed


def assert_docs_synced(
    rows: Iterable[Section3ARow] | None = None,
    *,
    doc_path: Path = DEFAULT_DOC_PATH,
) -> None:
    rows = list(rows) if rows is not None else load_fixture()
    docs_rows = parse_docs_matrix(doc_path)
    fixture_ids = {row.row_id for row in rows}
    docs_ids = set(docs_rows)
    if fixture_ids != docs_ids:
        raise Section3ADocsSyncError(
            f"Section 3A row ids differ: fixture={sorted(fixture_ids)}, docs={sorted(docs_ids)}"
        )
    mismatches: list[str] = []
    for row in rows:
        doc = docs_rows[row.row_id]
        expected = {
            "ui_model_id": row.ui_model_id,
            "guidance_kind": row.guidance_kind,
            "guidance_mode": row.guidance_mode,
            "disposition": row.disposition,
        }
        for field, value in expected.items():
            if doc[field] != value:
                mismatches.append(
                    f"row {row.row_id} {field}: fixture={value!r} docs={doc[field]!r}"
                )
    if mismatches:
        raise Section3ADocsSyncError("; ".join(mismatches))


def fixture_as_dicts(rows: Iterable[Section3ARow] | None = None) -> list[dict[str, Any]]:
    source_rows = list(rows) if rows is not None else load_fixture()
    return [row.__dict__.copy() for row in source_rows]


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _clean_markdown_cell(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = value.replace("\\|", "|")
    value = value.strip()
    return value


def _extract_backticked(value: str) -> str:
    matches = re.findall(r"`([^`]+)`", value)
    if not matches:
        raise Section3ADocsSyncError(f"expected backticked model id in {value!r}")
    return matches[-1]


def _extract_disposition(value: str) -> str:
    for disposition in sorted(DISPOSITIONS, key=len, reverse=True):
        if value.startswith(disposition):
            return disposition
    raise Section3ADocsSyncError(f"unknown Section 3A disposition {value!r}")


def _normalize_guidance(value: str) -> tuple[str, str]:
    value = re.sub(r"\s*\([^)]*\)", "", value).strip()
    if ":" in value:
        kind, mode = value.split(":", 1)
        return kind.strip(), mode.strip()
    if value == "uni3c":
        return "uni3c", "uni3c"
    return value, "none"


if __name__ == "__main__":
    assert_docs_synced()
    print(f"Section 3A matrix fixture is synced: {DEFAULT_FIXTURE_PATH}")
