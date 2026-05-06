"""Local resource pressure checks for claims and large local writes."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.core.log import headless_logger
from source.models.lora.lora_utils import sweep_lora_cache_from_env

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class DiskVolume:
    path: str
    used_pct: float
    free_bytes: int


@dataclass(frozen=True)
class ResourcePressureResult:
    status: str
    action: str
    allow_work: bool
    quota_alert: bool
    required_free_bytes: int
    recovered_bytes: int
    volumes: tuple[DiskVolume, ...]
    cleanup: dict[str, Any]
    reason: str = ""

    def to_state(self) -> dict[str, Any]:
        return {
            "resource_pressure_status": self.status,
            "resource_pressure_action": self.action,
            "resource_pressure_allow_work": self.allow_work,
            "resource_pressure_quota_alert": self.quota_alert,
            "resource_pressure_required_free_mb": int(self.required_free_bytes / _BYTES_PER_MB),
            "resource_pressure_recovered_bytes": self.recovered_bytes,
            "resource_pressure_reason": self.reason,
            "volumes": [
                {
                    "path": volume.path,
                    "used_pct": volume.used_pct,
                    "free_mb": int(volume.free_bytes / _BYTES_PER_MB),
                }
                for volume in self.volumes
            ],
            "cleanup": self.cleanup,
            "checked_at": int(time.time()),
        }


def ensure_resources_for_claim(worker_id: str) -> ResourcePressureResult:
    """Suppress claims when local disk pressure remains high after cleanup."""
    required = _mb_env("REIGH_DISK_CLAIM_MIN_FREE_MB", _mb_env("REIGH_DISK_MIN_FREE_MB", 0))
    return _ensure_resources(
        worker_id=worker_id,
        action="claim",
        required_free_bytes=required * _BYTES_PER_MB,
    )


def ensure_resources_for_write(
    *,
    worker_id: str,
    target_path: Path,
    required_bytes: int | None = None,
) -> ResourcePressureResult:
    """Require enough local space for a large write, after cleanup has run."""
    reserve = _mb_env("REIGH_DISK_WRITE_RESERVE_MB", 0) * _BYTES_PER_MB
    required = max(int(required_bytes or 0) + reserve, _mb_env("REIGH_DISK_WRITE_MIN_FREE_MB", 0) * _BYTES_PER_MB)
    return _ensure_resources(
        worker_id=worker_id,
        action="write",
        required_free_bytes=required,
        extra_paths=[target_path.parent],
    )


def resource_pressure_state_path(worker_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in worker_id)
    state_dir = Path(os.environ.get("REIGH_RESOURCE_PRESSURE_STATE_DIR", os.environ.get("REIGH_PREFLIGHT_STATE_DIR", "/tmp")))
    return state_dir / f"reigh_worker_resource_pressure_{safe}.json"


def write_resource_pressure_state(worker_id: str, result: ResourcePressureResult) -> Path:
    path = resource_pressure_state_path(worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_state(), sort_keys=True), encoding="utf-8")
    return path


def read_resource_pressure_state(worker_id: str | None) -> dict[str, Any] | None:
    if not worker_id:
        return None
    try:
        return json.loads(resource_pressure_state_path(worker_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ensure_resources(
    *,
    worker_id: str,
    action: str,
    required_free_bytes: int,
    extra_paths: list[Path] | None = None,
) -> ResourcePressureResult:
    threshold_pct = _float_env("REIGH_DISK_NEAR_FULL_PCT", 100.0)
    paths = _disk_paths_from_env(extra_paths)
    before = _disk_status(paths, threshold_pct, required_free_bytes)
    if before["ok"]:
        result = ResourcePressureResult(
            status="ok",
            action=action,
            allow_work=True,
            quota_alert=False,
            required_free_bytes=required_free_bytes,
            recovered_bytes=0,
            volumes=before["volumes"],
            cleanup={"lora": {}, "artifacts": {}},
        )
        write_resource_pressure_state(worker_id, result)
        return result

    cleanup = _run_cleanup()
    after = _disk_status(paths, threshold_pct, required_free_bytes)
    recovered = max(0, int(after["free_bytes"] - before["free_bytes"]))
    if after["ok"]:
        result = ResourcePressureResult(
            status="recovered",
            action=action,
            allow_work=True,
            quota_alert=True,
            required_free_bytes=required_free_bytes,
            recovered_bytes=recovered,
            volumes=after["volumes"],
            cleanup=cleanup,
            reason="disk_pressure_recovered_after_cleanup",
        )
        headless_logger.warning(
            f"[QUOTA_ALERT] disk pressure recovered action={action} recovered_bytes={recovered} worker_id={worker_id}"
        )
        write_resource_pressure_state(worker_id, result)
        return result

    result = ResourcePressureResult(
        status="near_full",
        action="claim_suppressed" if action == "claim" else "write_blocked",
        allow_work=False,
        quota_alert=True,
        required_free_bytes=required_free_bytes,
        recovered_bytes=recovered,
        volumes=after["volumes"],
        cleanup=cleanup,
        reason="disk_pressure_unrecoverable",
    )
    headless_logger.error(
        "[QUOTA_ALERT] disk pressure unrecovered "
        f"action={action} required_free_mb={int(required_free_bytes / _BYTES_PER_MB)} "
        f"recovered_bytes={recovered} worker_id={worker_id}"
    )
    write_resource_pressure_state(worker_id, result)
    return result


def _run_cleanup() -> dict[str, Any]:
    lora = sweep_lora_cache_from_env().to_dict()
    artifact = sweep_artifact_orphans_from_env()
    return {"lora": lora, "artifacts": artifact}


def sweep_artifact_orphans_from_env() -> dict[str, Any]:
    max_age_seconds = _int_env("REIGH_ARTIFACT_ORPHAN_MAX_AGE_SECONDS", 21600)
    paths = _artifact_cleanup_paths_from_env()
    return sweep_artifact_orphans(paths=paths, max_age_seconds=max_age_seconds)


def sweep_artifact_orphans(*, paths: list[Path], max_age_seconds: int, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    removed_files = 0
    removed_bytes = 0
    scanned_files = 0
    for root in paths:
        try:
            resolved = root.expanduser().resolve(strict=False)
        except OSError:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        for path in resolved.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if not _artifact_file_is_eligible(path):
                continue
            scanned_files += 1
            try:
                stat = path.stat()
            except OSError:
                continue
            if now - stat.st_mtime < max_age_seconds:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                headless_logger.debug(f"[RESOURCE_PRESSURE] artifact sweep failed for {path}: {exc}")
                continue
            removed_files += 1
            removed_bytes += stat.st_size
    return {
        "scanned_files": scanned_files,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
    }


def _disk_status(paths: list[Path], threshold_pct: float, required_free_bytes: int) -> dict[str, Any]:
    volumes: list[DiskVolume] = []
    worst_used_pct = 0.0
    lowest_free_bytes: int | None = None
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        used_pct = round(((usage.total - usage.free) / usage.total) * 100, 2) if usage.total else 0.0
        worst_used_pct = max(worst_used_pct, used_pct)
        lowest_free_bytes = usage.free if lowest_free_bytes is None else min(lowest_free_bytes, usage.free)
        volumes.append(DiskVolume(path=str(path), used_pct=used_pct, free_bytes=usage.free))

    free_bytes = lowest_free_bytes if lowest_free_bytes is not None else 0
    return {
        "ok": bool(volumes) and worst_used_pct < threshold_pct and free_bytes >= required_free_bytes,
        "free_bytes": free_bytes,
        "volumes": tuple(volumes),
    }


def _artifact_file_is_eligible(path: Path) -> bool:
    if path.suffix.lower() in {".tmp", ".part", ".download"}:
        return True
    name = path.name.lower()
    return name.startswith("debug_") or ".debug." in name or "/debug/" in path.as_posix().lower()


def _disk_paths_from_env(extra_paths: list[Path] | None = None) -> list[Path]:
    configured = os.environ.get("REIGH_DISK_HEALTH_PATHS")
    paths = [Path(part) for part in configured.split(":") if part] if configured else [Path("/")]
    if extra_paths:
        paths.extend(extra_paths)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _artifact_cleanup_paths_from_env() -> list[Path]:
    configured = os.environ.get("REIGH_ARTIFACT_CLEANUP_PATHS")
    if configured:
        return [Path(part) for part in configured.split(":") if part]
    paths: list[Path] = []
    for name in ("REIGH_LOCAL_MATERIALIZATION_DIR", "REIGH_LOCAL_WORKER_DIR", "REIGH_OUTPUT_DIR"):
        value = os.environ.get(name)
        if value:
            paths.append(Path(value))
    return paths


def _mb_env(name: str, default: int) -> int:
    return _int_env(name, default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
