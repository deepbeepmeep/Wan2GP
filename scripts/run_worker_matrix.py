#!/usr/bin/env python3
"""Manifest-driven worker smoke runner with per-case artifacts."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import shlex
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "scripts" / "worker_matrix_cases.json"
DEFAULT_DB_SNAPSHOTS_PATH = REPO_ROOT / "scripts" / "worker_matrix_db_snapshots.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "worker-matrix"
IMAGE_OUTPUT_TASK_TYPES = {
    "annotated_image_edit",
    "extract_frame",
    "image_inpaint",
    "qwen_image",
    "qwen_image_2512",
    "qwen_image_edit",
    "qwen_image_hires",
    "qwen_image_style",
    "z_image_turbo",
    "z_image_turbo_i2i",
}
_DB_SNAPSHOTS_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    description: str
    task_type: str
    family: str
    builder: str
    expected_outcome: str = "success"
    tags: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    default_enabled: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaseDefinition":
        return cls(
            case_id=payload["case_id"],
            description=payload["description"],
            task_type=payload["task_type"],
            family=payload["family"],
            builder=payload["builder"],
            expected_outcome=payload.get("expected_outcome", "success"),
            tags=tuple(payload.get("tags", [])),
            requires=tuple(payload.get("requires", [])),
            default_enabled=bool(payload.get("default_enabled", True)),
        )


@dataclass
class PreparedCase:
    definition: CaseDefinition
    params: dict[str, Any]
    project_id: str | None
    main_output_dir: Path
    task_queue: Any = None
    colour_match_videos: bool = False
    mask_active_frames: bool = True
    patchers: list[Any] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    task_type: str
    family: str
    status: str
    expected_outcome: str
    actual_outcome: str
    duration_seconds: float
    output_path: str | None = None
    error_message: str | None = None
    log_path: str | None = None
    case_dir: str | None = None
    traceback_path: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class TeeWriter(io.TextIOBase):
    """Mirror writes to the console and a case-local file."""

    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    @property
    def encoding(self) -> str:
        for stream in self._streams:
            enc = getattr(stream, "encoding", None)
            if enc:
                return enc
        return "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class FakeImmediateTaskQueue:
    """Capture generation tasks and mark them complete or failed immediately."""

    def __init__(
        self,
        output_dir: Path,
        task_type: str,
        case_dir: Path,
        *,
        expected_model: str | None = None,
        expected_parameters: dict[str, Any] | None = None,
        fail_message: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.task_type = task_type
        self.case_dir = case_dir
        self.expected_model = expected_model
        self.expected_parameters = expected_parameters or {}
        self.fail_message = fail_message
        self.status_map: dict[str, SimpleNamespace] = {}
        self.submitted_tasks: list[dict[str, Any]] = []

    def submit_task(self, generation_task: Any) -> str:
        snapshot = {
            "id": generation_task.id,
            "model": generation_task.model,
            "prompt": generation_task.prompt,
            "parameters": _json_safe(generation_task.parameters),
            "priority": generation_task.priority,
        }
        self.submitted_tasks.append(snapshot)
        _write_json(self.case_dir / "captured_generation_task.json", snapshot)

        validation_errors: list[str] = []
        if self.expected_model and generation_task.model != self.expected_model:
            validation_errors.append(
                f"Expected model '{self.expected_model}' but saw '{generation_task.model}'"
            )
        for key, expected_value in self.expected_parameters.items():
            actual_value = generation_task.parameters.get(key)
            if actual_value != expected_value:
                validation_errors.append(
                    f"Expected parameter '{key}' == {expected_value!r} but saw {actual_value!r}"
                )

        output_path: str | None = None
        if not validation_errors and not self.fail_message:
            output_path = str(_create_synthetic_output(
                output_dir=self.output_dir,
                task_id=generation_task.id,
                task_type=self.task_type,
            ))

        if validation_errors:
            message = "; ".join(validation_errors)
            status = SimpleNamespace(
                status="failed",
                result_path=None,
                error_message=message,
                processing_time=0.0,
            )
        elif self.fail_message:
            status = SimpleNamespace(
                status="failed",
                result_path=None,
                error_message=self.fail_message,
                processing_time=0.0,
            )
        else:
            status = SimpleNamespace(
                status="completed",
                result_path=output_path,
                error_message=None,
                processing_time=0.0,
            )

        self.status_map[generation_task.id] = status
        return generation_task.id

    def get_task_status(self, task_id: str) -> Any:
        return self.status_map.get(task_id)


def load_case_definitions(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> list[CaseDefinition]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [CaseDefinition.from_dict(entry) for entry in payload["cases"]]
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"Duplicate case_id in manifest: {case.case_id}")
        seen.add(case.case_id)
    return cases


def load_db_snapshots(snapshot_path: Path = DEFAULT_DB_SNAPSHOTS_PATH) -> dict[str, Any]:
    global _DB_SNAPSHOTS_CACHE
    if _DB_SNAPSHOTS_CACHE is None:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        _DB_SNAPSHOTS_CACHE = payload["snapshots"]
    return _DB_SNAPSHOTS_CACHE


def filter_case_definitions(
    cases: list[CaseDefinition],
    *,
    case_ids: set[str] | None = None,
    task_types: set[str] | None = None,
    tags: set[str] | None = None,
    families: set[str] | None = None,
    include_non_default: bool = False,
) -> list[CaseDefinition]:
    selected: list[CaseDefinition] = []
    for case in cases:
        if not include_non_default and not case.default_enabled:
            continue
        if case_ids and case.case_id not in case_ids:
            continue
        if task_types and case.task_type not in task_types:
            continue
        if families and case.family not in families:
            continue
        if tags and not tags.intersection(case.tags):
            continue
        selected.append(case)
    return selected


def requirements_status(case: CaseDefinition) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for requirement in case.requires:
        checker = REQUIREMENT_CHECKERS.get(requirement)
        if checker is None:
            missing.append(f"unknown requirement:{requirement}")
            continue
        if not checker():
            missing.append(requirement)
    return (len(missing) == 0, missing)


def format_summary_markdown(
    results: list[CaseResult],
    *,
    started_at: str,
    run_dir: Path,
) -> str:
    lines = [
        "# Worker Matrix Summary",
        "",
        f"- Started: {started_at}",
        f"- Run dir: `{run_dir}`",
        f"- Total cases: {len(results)}",
        "",
        "| Case | Task Type | Status | Actual | Duration (s) | Output |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in results:
        output = result.output_path or ""
        lines.append(
            f"| `{result.case_id}` | `{result.task_type}` | `{result.status}` | "
            f"`{result.actual_outcome}` | {result.duration_seconds:.2f} | `{output}` |"
        )
    failed_ids = [result.case_id for result in results if result.status == "failed"]
    if failed_ids:
        lines.extend([
            "",
            "## Failed Cases",
            "",
            "```text",
            "\n".join(failed_ids),
            "```",
        ])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifest-driven worker smoke cases")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--task-type", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--failed-from", type=Path, default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_repo_on_path()
    args = parse_args(argv)
    cases = load_case_definitions(args.manifest)

    failed_case_ids: set[str] = set()
    if args.failed_from:
        failed_payload = json.loads(args.failed_from.read_text(encoding="utf-8"))
        failed_case_ids = {
            item["case_id"]
            for item in failed_payload.get("results", [])
            if item.get("status") == "failed"
        }

    combined_case_ids = set(args.case_id) | failed_case_ids
    selected = filter_case_definitions(
        cases,
        case_ids=combined_case_ids or None,
        task_types=set(args.task_type) or None,
        tags=set(args.tag) or None,
        families=set(args.family) or None,
        include_non_default=args.all_cases or bool(args.case_id) or bool(failed_case_ids),
    )

    if args.list_cases:
        for case in selected:
            tags = ",".join(case.tags)
            print(f"{case.case_id:28} {case.task_type:20} {case.family:18} {tags}")
        return 0

    if not selected:
        print("No worker matrix cases selected.", file=sys.stderr)
        return 1

    started_at = _utc_timestamp()
    run_dir = args.output_dir / started_at
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_snapshot = [asdict(case) for case in selected]
    _write_json(run_dir / "selected_cases.json", manifest_snapshot)

    print(f"Running {len(selected)} worker matrix case(s) into {run_dir}")

    results: list[CaseResult] = []
    for case in selected:
        result = run_case(case, run_dir)
        results.append(result)
        print(f"[{result.status.upper():7}] {case.case_id} -> {result.actual_outcome}")
        if args.stop_on_failure and result.status == "failed":
            break

    summary_payload = {
        "started_at": started_at,
        "run_dir": str(run_dir),
        "results": [asdict(result) for result in results],
    }
    _write_json(run_dir / "summary.json", summary_payload)
    (run_dir / "summary.md").write_text(
        format_summary_markdown(results, started_at=started_at, run_dir=run_dir),
        encoding="utf-8",
    )

    failed_ids = [result.case_id for result in results if result.status == "failed"]
    (run_dir / "failures.txt").write_text("\n".join(failed_ids) + ("\n" if failed_ids else ""), encoding="utf-8")
    rerun_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "python scripts/run_worker_matrix.py \\",
    ]
    if failed_ids:
        for case_id in failed_ids[:-1]:
            rerun_lines.append(f"  --case-id {shlex.quote(case_id)} \\")
        rerun_lines.append(f"  --case-id {shlex.quote(failed_ids[-1])}")
    else:
        rerun_lines.append("  --list-cases")
    rerun_path = run_dir / "rerun_failed.sh"
    rerun_path.write_text("\n".join(rerun_lines) + "\n", encoding="utf-8")
    rerun_path.chmod(0o755)

    return 1 if failed_ids else 0


def run_case(case: CaseDefinition, run_dir: Path) -> CaseResult:
    ensure_repo_on_path()
    case_dir = run_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "worker.log"
    _write_json(case_dir / "case_definition.json", asdict(case))

    requirements_ok, missing_requirements = requirements_status(case)
    if not requirements_ok:
        result = CaseResult(
            case_id=case.case_id,
            task_type=case.task_type,
            family=case.family,
            status="skipped",
            expected_outcome=case.expected_outcome,
            actual_outcome="skipped",
            duration_seconds=0.0,
            error_message=f"Missing requirements: {', '.join(missing_requirements)}",
            log_path=str(log_path),
            case_dir=str(case_dir),
        )
        _write_json(case_dir / "result.json", asdict(result))
        return result

    try:
        prepared = CASE_BUILDERS[case.builder](case, case_dir)
    except Exception as exc:  # noqa: BLE001
        tb_path = case_dir / "traceback.txt"
        tb_path.write_text(traceback.format_exc(), encoding="utf-8")
        result = CaseResult(
            case_id=case.case_id,
            task_type=case.task_type,
            family=case.family,
            status="failed",
            expected_outcome=case.expected_outcome,
            actual_outcome="failure",
            duration_seconds=0.0,
            error_message=f"Builder {case.builder!r} crashed: {exc}",
            log_path=str(log_path),
            case_dir=str(case_dir),
            traceback_path=str(tb_path),
        )
        _write_json(case_dir / "result.json", asdict(result))
        return result

    _write_json(case_dir / "prepared_input.json", {
        "task_type": prepared.definition.task_type,
        "project_id": prepared.project_id,
        "params": _json_safe(prepared.params),
        "notes": _json_safe(prepared.notes),
    })

    started = datetime.now(timezone.utc)
    actual_outcome = "failure"
    output_path: str | None = None
    error_message: str | None = None
    traceback_path: str | None = None

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = TeeWriter(sys.__stdout__, log_file)
        tee_stderr = TeeWriter(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            started_marker = datetime.now(timezone.utc).isoformat()
            print(
                f"[WORKER_MATRIX][CASE_START] case_id={case.case_id} "
                f"task_type={case.task_type} started_at={started_marker}"
            )
            try:
                with contextlib.ExitStack() as stack:
                    for patcher in prepared.patchers:
                        stack.enter_context(patcher)

                    worker_mod = __import__("worker")
                    from source.core.log import enable_debug_mode

                    enable_debug_mode()
                    raw_result = worker_mod.process_single_task(
                        task_params_dict=copy.deepcopy(prepared.params),
                        main_output_dir_base=prepared.main_output_dir,
                        task_type=prepared.definition.task_type,
                        project_id_for_task=prepared.project_id,
                        task_queue=prepared.task_queue,
                        colour_match_videos=prepared.colour_match_videos,
                        mask_active_frames=prepared.mask_active_frames,
                    )
                success, output_path, error_message = _normalize_worker_result(raw_result)
                actual_outcome = "success" if success else "failure"
                print(
                    f"[WORKER_MATRIX][CASE_END] case_id={case.case_id} "
                    f"outcome={actual_outcome} output_path={output_path or ''} "
                    f"error={error_message or ''}"
                )
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                actual_outcome = "failure"
                traceback_path = str(case_dir / "traceback.txt")
                (case_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(
                    f"[WORKER_MATRIX][CASE_EXCEPTION] case_id={case.case_id} "
                    f"error={error_message}"
                )

    duration_seconds = (datetime.now(timezone.utc) - started).total_seconds()
    expectation_matched = actual_outcome == case.expected_outcome
    status = "passed" if expectation_matched else "failed"

    result = CaseResult(
        case_id=case.case_id,
        task_type=case.task_type,
        family=case.family,
        status=status,
        expected_outcome=case.expected_outcome,
        actual_outcome=actual_outcome,
        duration_seconds=duration_seconds,
        output_path=output_path,
        error_message=error_message,
        log_path=str(log_path),
        case_dir=str(case_dir),
        traceback_path=traceback_path,
        notes=_json_safe(prepared.notes),
    )
    _write_json(case_dir / "result.json", asdict(result))
    return result


def _normalize_worker_result(raw_result: Any) -> tuple[bool, str | None, str | None]:
    if hasattr(raw_result, "outcome"):
        from source.core.params.task_result import TaskOutcome

        outcome = raw_result.outcome
        if outcome == TaskOutcome.FAILED:
            return False, None, getattr(raw_result, "error_message", None) or getattr(raw_result, "output_path", None)
        if outcome in (TaskOutcome.ORCHESTRATING, TaskOutcome.ORCHESTRATOR_COMPLETE):
            return False, None, f"Unexpected orchestrator outcome: {outcome.value}"
        return True, getattr(raw_result, "output_path", None), None

    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        success, value = raw_result
        return bool(success), value if success else None, None if success else value

    raise TypeError(f"Unsupported worker result shape: {type(raw_result).__name__}")


def _build_direct_queue_case(
    case: CaseDefinition,
    case_dir: Path,
    *,
    params: dict[str, Any],
    expected_model: str | None = None,
    expected_parameters: dict[str, Any] | None = None,
    patchers: list[Any] | None = None,
    notes: dict[str, Any] | None = None,
    ) -> PreparedCase:
    output_dir = case_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    case_params = copy.deepcopy(params)
    case_params.setdefault("task_id", f"{case.case_id}-task")
    queue = FakeImmediateTaskQueue(
        output_dir=output_dir,
        task_type=case.task_type,
        case_dir=case_dir,
        expected_model=expected_model,
        expected_parameters=expected_parameters,
    )
    return PreparedCase(
        definition=case,
        params=case_params,
        project_id=None,
        main_output_dir=output_dir,
        task_queue=queue,
        patchers=patchers or [],
        notes=notes or {},
    )


def build_vace_basic(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    snapshot, params = _load_snapshot_params("vace_basic")
    return _build_direct_queue_case(
        case,
        case_dir,
        params=params,
        expected_model="vace_14B_cocktail_2_2",
        expected_parameters={"guidance_phases": 2, "guidance_scale": 1},
        notes=_snapshot_notes(snapshot),
    )


def build_ltx2_basic(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    return _build_direct_queue_case(
        case,
        case_dir,
        params={
            "model": "ltx2_22B",
            "prompt": "worker matrix ltx2 dev smoke",
            "resolution": "512x288",
            "video_length": 9,
            "num_inference_steps": 4,
            "guidance_scale": 3.0,
            "audio_guidance_scale": 7.0,
            "seed": 17,
        },
        expected_model="ltx2_22B",
        expected_parameters={"guidance_scale": 3.0, "audio_guidance_scale": 7.0},
    )


def build_ltx2_distilled(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    return _build_direct_queue_case(
        case,
        case_dir,
        params={
            "model": "ltx2_22B_distilled",
            "prompt": "worker matrix ltx2 distilled smoke",
            "resolution": "512x288",
            "video_length": 9,
            "num_inference_steps": 8,
            "seed": 23,
        },
        expected_model="ltx2_22B_distilled",
        expected_parameters={"num_inference_steps": 8},
    )


def build_z_image_turbo_basic(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    snapshot, params = _load_snapshot_params("z_image_turbo")
    return _build_direct_queue_case(
        case,
        case_dir,
        params=params,
        expected_model="z_image",
        expected_parameters={"video_length": 1, "guidance_scale": 0},
        notes=_snapshot_notes(snapshot),
    )


def build_z_image_turbo_i2i_basic(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    image_path = _create_test_image(case_dir / "fixtures" / "z_i2i_input.png", text="Z")
    snapshot, params = _load_snapshot_params(
        "z_image_turbo_i2i",
        fixture_urls={"z_image_i2i_input": "https://worker-matrix.invalid/z-image-i2i-input.png"},
    )

    class _Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    patchers = [
        patch(
            "requests.get",
            side_effect=lambda url, timeout=30: _Response(image_path.read_bytes()),
        )
    ]
    return _build_direct_queue_case(
        case,
        case_dir,
        params=params,
        expected_model="z_image_img2img",
        expected_parameters={"video_length": 1, "guidance_scale": 0, "denoising_strength": 0.4},
        patchers=patchers,
        notes=_snapshot_notes(snapshot),
    )


def build_qwen_image_basic(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    reference_image = _create_test_image(case_dir / "fixtures" / "qwen_style_ref.png", text="QS")
    snapshot, params = _load_snapshot_params(
        "qwen_image_style",
        fixture_paths={"qwen_style_reference": reference_image},
    )
    patchers = [patch("source.models.model_handlers.qwen_handler.hf_hub_download", side_effect=_fake_hf_hub_download)]
    return _build_direct_queue_case(
        case,
        case_dir,
        params=params,
        expected_model="qwen_image_edit_20B",
        expected_parameters={"video_length": 1},
        patchers=patchers,
        notes=_snapshot_notes(snapshot),
    )


def build_qwen_image_edit_basic(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    image_path = _create_test_image(case_dir / "fixtures" / "qwen_edit_input.png", text="QE")
    snapshot, params = _load_snapshot_params(
        "qwen_image_edit",
        fixture_paths={"qwen_edit_input": image_path},
    )
    patchers = [patch("source.models.model_handlers.qwen_handler.hf_hub_download", side_effect=_fake_hf_hub_download)]
    return _build_direct_queue_case(
        case,
        case_dir,
        params=params,
        expected_model="qwen_image_edit_20B",
        expected_parameters={"video_length": 1},
        patchers=patchers,
        notes=_snapshot_notes(snapshot),
    )


def build_wan22_individual_segment_db_like(case: CaseDefinition, case_dir: Path) -> PreparedCase:
    start_image = _create_test_image(case_dir / "fixtures" / "wan22_start.png", text="W1")
    end_image = _create_test_image(case_dir / "fixtures" / "wan22_end.png", text="W2")
    snapshot, params = _load_snapshot_params(
        "individual_travel_segment_wan22_i2v",
        fixture_paths={
            "wan22_start_image": start_image,
            "wan22_end_image": end_image,
        },
    )
    output_dir = case_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    queue = FakeImmediateTaskQueue(
        output_dir=output_dir,
        task_type="travel_segment",
        case_dir=case_dir,
        expected_model="wan_2_2_i2v_lightning_baseline_2_2_2",
        expected_parameters={"video_length": 50},
    )
    return PreparedCase(
        definition=case,
        params=params,
        project_id=None,
        main_output_dir=output_dir,
        task_queue=queue,
        notes=_snapshot_notes(snapshot),
    )


CASE_BUILDERS: dict[str, Callable[[CaseDefinition, Path], PreparedCase]] = {
    "vace_basic": build_vace_basic,
    "ltx2_basic": build_ltx2_basic,
    "ltx2_distilled": build_ltx2_distilled,
    "z_image_turbo_basic": build_z_image_turbo_basic,
    "z_image_turbo_i2i_basic": build_z_image_turbo_i2i_basic,
    "qwen_image_basic": build_qwen_image_basic,
    "qwen_image_edit_basic": build_qwen_image_edit_basic,
    "wan22_individual_segment_db_like": build_wan22_individual_segment_db_like,
}


def _create_test_image(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), color=(24, 64, 96))
    pixels = image.load()
    for x in range(32, 224):
        for y in range(96, 160):
            pixels[x, y] = (180, 80, 32)
    image.save(path)
    return path


def _create_test_video(path: Path, *, width: int = 320, height: int = 180, frames: int = 6, fps: int = 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for frame_idx in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = (frame_idx * 25) % 255
        image[:, :, 1] = 80
        image[:, :, 2] = 200 - (frame_idx * 10)
        writer.write(image)
    writer.release()
    return path


def _create_synthetic_output(output_dir: Path, task_id: str, task_type: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if task_type in IMAGE_OUTPUT_TASK_TYPES:
        return _create_test_image(output_dir / f"{task_id}.png", text=task_type[:2].upper())
    return _create_test_video(output_dir / f"{task_id}.mp4")


def _fake_hf_hub_download(*, filename: str, local_dir: str, **_: Any) -> str:
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake-lora")
    return str(target)


def _load_snapshot_params(
    snapshot_name: str,
    *,
    fixture_paths: dict[str, Path] | None = None,
    fixture_urls: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = copy.deepcopy(load_db_snapshots()[snapshot_name])
    replacements: dict[str, str] = {}
    for key, value in (fixture_paths or {}).items():
        replacements[f"__FIXTURE_PATH__:{key}"] = str(value)
    for key, value in (fixture_urls or {}).items():
        replacements[f"__FIXTURE_URL__:{key}"] = value
    params = _replace_snapshot_tokens(snapshot["params"], replacements)
    return snapshot, params


def _replace_snapshot_tokens(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, dict):
        return {key: _replace_snapshot_tokens(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_snapshot_tokens(item, replacements) for item in value]
    return value


def _snapshot_notes(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_name": snapshot["snapshot_name"],
        "source_task_id": snapshot["source_task_id"],
        "source_created_at": snapshot["source_created_at"],
        "db_task_type": snapshot["task_type"],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


REQUIREMENT_CHECKERS: dict[str, Callable[[], bool]] = {
    "ffmpeg": lambda: shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
    "moviepy": lambda: _module_available("moviepy.editor"),
}


def ensure_repo_on_path() -> None:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


if __name__ == "__main__":
    raise SystemExit(main())
