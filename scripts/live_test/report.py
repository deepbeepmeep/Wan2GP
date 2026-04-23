"""Report writers for live-test matrix runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.live_test.completion_poller import TaskResult


def _is_passing_result(result: TaskResult) -> bool:
    return result.final_status == "Complete" and bool(result.generation_ids) and not result.error_summary


def _render_markdown(results: list[TaskResult], *, variant: str, pod_id: str | None, passed: int) -> str:
    total = len(results)
    lines = [
        "# Live Test Report",
        "",
        f"Variant: `{variant}`",
        f"Pod ID: `{pod_id or 'n/a'}`",
        f"Summary: `{passed}/{total} passed`",
        "",
        "| Case | Task Type | Final Status | Duration (s) | Output | Generation IDs | Error |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for result in results:
        generation_ids = ", ".join(result.generation_ids) if result.generation_ids else "-"
        output = result.output_location or "-"
        error = (result.error_summary or "-").replace("\n", " ")
        lines.append(
            f"| {result.case_name} | {result.task_type} | {result.final_status} | "
            f"{result.elapsed_sec:.3f} | {output} | {generation_ids} | {error} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(
    results: list[TaskResult],
    variant: str,
    pod_id: str | None,
    out_dir,
) -> Path:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for result in results if _is_passing_result(result))
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "pod_id": pod_id,
        "passed": passed,
        "total": len(results),
        "results": [asdict(result) for result in results],
    }

    (output_dir / "report.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_markdown(results, variant=variant, pod_id=pod_id, passed=passed),
        encoding="utf-8",
    )
    return output_dir


__all__ = ["write_report"]
