from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from scripts.dual_run_compare.status import SECTION_GREEN, SECTION_RED
from scripts.dual_run_compare.thresholds import DEFAULT_PATH


DUAL_RUN_DIR = DEFAULT_PATH.parent
ARTIFACTS_DIR = DUAL_RUN_DIR / "artifacts"
SKIPPED_SIDE_EFFECTS = ("completion", "billing", "upload", "user_visible")
REMOTE_STORAGE_HOST_MARKERS = ("supabase", "storage.googleapis.com", "s3.amazonaws.com", "r2.cloudflarestorage.com")
REMOTE_STORAGE_PATH_MARKERS = ("/storage/v1/object/", "/image_uploads/", "/tasks/")
PRODUCTION_PATH_MARKERS = ("/image_uploads/", "/storage/v1/object/", "/generations/", "/tasks/")


class ShadowIsolationError(ValueError):
    pass


@dataclass(frozen=True)
class ShadowEnvelope:
    report_id: str
    route_key: str
    artifact_root: Path
    shadow_root: Path

    def as_report_data(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "route_key": self.route_key,
            "artifact_root": str(self.artifact_root),
            "shadow_root": str(self.shadow_root),
            "side_effects": skipped_side_effect_report(),
        }


def _validate_component(name: str, value: str) -> None:
    if not value or value in {".", ".."}:
        raise ShadowIsolationError(f"{name} must be non-empty")
    if "/" in value or "\\" in value:
        raise ShadowIsolationError(f"{name} must not contain path separators")


def create_shadow_envelope(
    report_id: str,
    route_key: str,
    *,
    artifacts_dir: Path = ARTIFACTS_DIR,
    create: bool = True,
) -> ShadowEnvelope:
    _validate_component("report_id", report_id)
    _validate_component("route_key", route_key)
    artifact_root = (artifacts_dir / report_id / route_key).resolve()
    shadow_root = (artifact_root / "shadow").resolve()
    if create:
        shadow_root.mkdir(parents=True, exist_ok=True)
    return ShadowEnvelope(
        report_id=report_id,
        route_key=route_key,
        artifact_root=artifact_root,
        shadow_root=shadow_root,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_shadow_artifact_path(envelope: ShadowEnvelope, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ShadowIsolationError("shadow artifact path must be relative")
    candidate = (envelope.shadow_root / path).resolve()
    if not _is_relative_to(candidate, envelope.shadow_root):
        raise ShadowIsolationError("shadow artifact path escapes the shadow root")
    return candidate


def validate_shadow_artifact_path(envelope: ShadowEnvelope, path: str | Path) -> Path:
    candidate = Path(path).resolve()
    if not _is_relative_to(candidate, envelope.shadow_root):
        raise ShadowIsolationError("artifact writes must remain under the shadow root")
    return candidate


def _url_is_remote_storage(parsed) -> bool:
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return any(marker in host for marker in REMOTE_STORAGE_HOST_MARKERS) or any(
        marker in path for marker in REMOTE_STORAGE_PATH_MARKERS
    )


def _url_is_explicitly_disposable(parsed) -> bool:
    path = parsed.path.lower()
    return "shadow" in path and "disposable" in path


def validate_shadow_target(
    envelope: ShadowEnvelope,
    target: str | Path,
    *,
    allow_disposable_remote: bool = False,
) -> dict[str, Any]:
    target_text = str(target)
    parsed = urlparse(target_text)
    if parsed.scheme in {"http", "https", "s3", "gs"}:
        if _url_is_remote_storage(parsed):
            if allow_disposable_remote and _url_is_explicitly_disposable(parsed):
                return {
                    "status": SECTION_GREEN,
                    "target": target_text,
                    "target_kind": "disposable_remote_storage",
                    "reason": "explicit disposable shadow storage target",
                }
            raise ShadowIsolationError("remote storage targets are blocked unless explicitly disposable shadow storage")
        raise ShadowIsolationError("remote URLs are blocked in shadow mode")

    path = Path(target_text)
    if path.is_absolute():
        validate_shadow_artifact_path(envelope, path)
    else:
        lowered = target_text.replace("\\", "/").lower()
        if any(marker.strip("/") in lowered.split("/") for marker in PRODUCTION_PATH_MARKERS):
            raise ShadowIsolationError("production or task-upload path is blocked in shadow mode")
        path = resolve_shadow_artifact_path(envelope, path)
    return {
        "status": SECTION_GREEN,
        "target": str(path),
        "target_kind": "local_shadow_artifact",
        "shadow_root": str(envelope.shadow_root),
    }


def skipped_side_effect_report(effects: Iterable[str] = SKIPPED_SIDE_EFFECTS) -> list[dict[str, str]]:
    return [
        {
            "effect": effect,
            "status": "skipped",
            "reason": "shadow mode records intended side effect without executing it",
        }
        for effect in effects
    ]


def shadow_isolation_report(
    envelope: ShadowEnvelope,
    *,
    attempted_targets: Iterable[str | Path] = (),
) -> dict[str, Any]:
    target_results: list[dict[str, Any]] = []
    status = SECTION_GREEN
    for target in attempted_targets:
        try:
            target_results.append(validate_shadow_target(envelope, target))
        except ShadowIsolationError as exc:
            status = SECTION_RED
            target_results.append(
                {
                    "status": SECTION_RED,
                    "target": str(target),
                    "reason": str(exc),
                }
            )

    return {
        "status": status,
        "artifact_root": str(envelope.artifact_root),
        "shadow_root": str(envelope.shadow_root),
        "targets": target_results,
        "skipped_side_effects": skipped_side_effect_report(),
    }
