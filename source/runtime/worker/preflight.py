"""Worker preflight checks and readiness metadata publishing."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from source.core.log import headless_logger


PREFLIGHT_STATUS_PENDING = "pending"
PREFLIGHT_STATUS_RUNNING = "running"
PREFLIGHT_STATUS_PASSED = "passed"
PREFLIGHT_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class WorkerPreflightResult:
    status: str
    checks: list[PreflightCheck]
    started_at: float
    completed_at: float

    @property
    def ok(self) -> bool:
        return self.status == PREFLIGHT_STATUS_PASSED

    @property
    def failed_checks(self) -> list[str]:
        return [check.name for check in self.checks if check.required and not check.ok]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "preflight_status": self.status,
            "preflight_ok": self.ok,
            "preflight_failed_checks": self.failed_checks,
            "preflight_checks": [asdict(check) for check in self.checks],
            "preflight_started_at": self.started_at,
            "preflight_completed_at": self.completed_at,
        }


def run_worker_preflight(
    *,
    repo_root: Path,
    wan2gp_path: Path,
    main_output_dir: Path,
    backend: str,
) -> WorkerPreflightResult:
    started_at = time.time()
    checks: list[PreflightCheck] = []

    _append_path_check(checks, "repo_root", repo_root, expected_type="dir")
    _append_path_check(checks, "wan2gp_path", wan2gp_path, expected_type="dir")
    _append_path_check(checks, "wan2gp_submodule_marker", wan2gp_path / ".git", expected_type="any")
    _append_path_check(checks, "wgp_entrypoint", wan2gp_path / "wgp.py", expected_type="file")
    _append_import_check(checks, "torch")
    _append_import_check(checks, "dotenv")
    _append_import_check(checks, "fastapi")

    vibecomfy_required = _vibecomfy_preflight_required(backend)
    _append_vibecomfy_check(checks, repo_root=repo_root, required=vibecomfy_required)

    _append_path_check(
        checks,
        "task_dispatch_manifest",
        repo_root / "source" / "task_handlers" / "tasks" / "dispatch_manifest.py",
        expected_type="file",
    )
    _append_path_check(
        checks,
        "lora_module_manifest",
        repo_root / "source" / "models" / "lora" / "module_manifest.py",
        expected_type="file",
    )
    _append_path_check(checks, "wan2gp_models_dir", wan2gp_path / "models", expected_type="dir")
    _append_path_check(checks, "wan2gp_plugins_dir", wan2gp_path / "plugins", expected_type="dir")
    _append_writable_dir_check(checks, "main_output_dir", main_output_dir)
    _append_writable_dir_check(checks, "uv_cache_dir", Path(os.environ.get("UV_CACHE_DIR", str(repo_root / ".uv-cache"))))

    status = PREFLIGHT_STATUS_PASSED if all(check.ok or not check.required for check in checks) else PREFLIGHT_STATUS_FAILED
    result = WorkerPreflightResult(
        status=status,
        checks=checks,
        started_at=started_at,
        completed_at=time.time(),
    )
    headless_logger.essential(
        f"[PREFLIGHT] status={result.status} backend={backend} failed={','.join(result.failed_checks) or 'none'}"
    )
    return result


def result_from_failure(name: str, detail: str, *, started_at: float | None = None) -> WorkerPreflightResult:
    started = started_at or time.time()
    return WorkerPreflightResult(
        status=PREFLIGHT_STATUS_FAILED,
        checks=[PreflightCheck(name=name, ok=False, detail=detail, required=True)],
        started_at=started,
        completed_at=time.time(),
    )


def finalize_preflight_result(
    base: WorkerPreflightResult,
    *,
    extra_checks: list[PreflightCheck],
) -> WorkerPreflightResult:
    checks = [*base.checks, *extra_checks]
    status = PREFLIGHT_STATUS_PASSED if all(check.ok or not check.required for check in checks) else PREFLIGHT_STATUS_FAILED
    return WorkerPreflightResult(
        status=status,
        checks=checks,
        started_at=base.started_at,
        completed_at=time.time(),
    )


def publish_preflight_metadata(
    *,
    supabase_client: Any,
    worker_id: str,
    result: WorkerPreflightResult,
    ready_for_tasks: bool,
) -> bool:
    metadata_update = {
        **result.to_metadata(),
        "ready_for_tasks": bool(ready_for_tasks and result.ok),
    }
    write_preflight_state(worker_id, metadata_update)
    try:
        current_result = (
            supabase_client.table("workers")
            .select("metadata")
            .eq("id", worker_id)
            .limit(1)
            .execute()
        )
        rows = getattr(current_result, "data", None) or []
        metadata = dict(rows[0].get("metadata") or {}) if rows else {}
        metadata.update(metadata_update)
        (
            supabase_client.table("workers")
            .update({"metadata": metadata})
            .eq("id", worker_id)
            .execute()
        )
        return True
    except Exception as exc:
        headless_logger.warning(f"[PREFLIGHT] Failed to publish worker metadata for {worker_id}: {exc}")
        return False


def write_preflight_state(worker_id: str, metadata: dict[str, Any]) -> Path:
    path = preflight_state_path(worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return path


def read_preflight_state(worker_id: str | None) -> dict[str, Any] | None:
    if not worker_id:
        return None
    path = preflight_state_path(worker_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def preflight_state_path(worker_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in worker_id)
    return Path(os.environ.get("REIGH_PREFLIGHT_STATE_DIR", "/tmp")) / f"reigh_worker_preflight_{safe}.json"


def _append_path_check(
    checks: list[PreflightCheck],
    name: str,
    path: Path,
    *,
    expected_type: str,
    required: bool = True,
) -> None:
    if expected_type == "dir":
        ok = path.is_dir()
    elif expected_type == "file":
        ok = path.is_file()
    else:
        ok = path.exists()
    checks.append(PreflightCheck(name=name, ok=ok, detail=str(path), required=required))


def _append_import_check(checks: list[PreflightCheck], module_name: str, *, required: bool = True) -> None:
    spec = importlib.util.find_spec(module_name)
    checks.append(
        PreflightCheck(
            name=f"import:{module_name}",
            ok=spec is not None,
            detail=getattr(spec, "origin", None) or "not found",
            required=required,
        )
    )


def _vibecomfy_preflight_required(backend: str) -> bool:
    configured = os.environ.get("REIGH_PREFLIGHT_REQUIRE_VIBECOMFY")
    if configured is not None:
        return configured.strip().lower() not in {"0", "false", "no"}
    return str(backend).strip().lower() == "vibecomfy"


def _append_vibecomfy_check(checks: list[PreflightCheck], *, repo_root: Path, required: bool) -> None:
    spec = importlib.util.find_spec("vibecomfy")
    candidates = [
        Path(os.environ["VIBECOMFY_PATH"]) if os.environ.get("VIBECOMFY_PATH") else None,
        repo_root.parent / "vibecomfy",
        Path("/workspace/vibecomfy"),
    ]
    existing = [path for path in candidates if path and path.exists()]
    checks.append(
        PreflightCheck(
            name="vibecomfy_available",
            ok=spec is not None or bool(existing),
            detail=getattr(spec, "origin", None) or ",".join(str(path) for path in existing) or "not found",
            required=required,
        )
    )
    if existing:
        vibecomfy_root = existing[0]
        _append_path_check(
            checks,
            "vibecomfy_template_index",
            vibecomfy_root / "template_index.json",
            expected_type="file",
            required=required,
        )
        _append_path_check(
            checks,
            "vibecomfy_custom_nodes_manifest",
            vibecomfy_root / "workflow_corpus" / "manifests" / "coverage.json",
            expected_type="file",
            required=required,
        )


def _append_writable_dir_check(checks: list[PreflightCheck], name: str, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".reigh-preflight-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append(PreflightCheck(name=name, ok=True, detail=str(path), required=True))
    except OSError as exc:
        checks.append(PreflightCheck(name=name, ok=False, detail=f"{path}: {exc}", required=True))
