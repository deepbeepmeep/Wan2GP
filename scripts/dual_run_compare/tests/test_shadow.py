from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dual_run_compare.shadow import (
    ShadowIsolationError,
    create_shadow_envelope,
    resolve_shadow_artifact_path,
    shadow_isolation_report,
    validate_shadow_artifact_path,
    validate_shadow_target,
)


def test_shadow_envelope_places_artifacts_under_report_route_shadow_root(tmp_path: Path) -> None:
    envelope = create_shadow_envelope("report-1", "video_enhance", artifacts_dir=tmp_path)

    assert envelope.artifact_root == tmp_path / "report-1" / "video_enhance"
    assert envelope.shadow_root == tmp_path / "report-1" / "video_enhance" / "shadow"
    assert envelope.shadow_root.is_dir()

    output = resolve_shadow_artifact_path(envelope, "media/output.mp4")
    assert output == envelope.shadow_root / "media" / "output.mp4"


def test_shadow_artifact_policy_rejects_escape_and_non_shadow_absolute_paths(tmp_path: Path) -> None:
    envelope = create_shadow_envelope("report-1", "video_enhance", artifacts_dir=tmp_path)

    with pytest.raises(ShadowIsolationError, match="escapes"):
        resolve_shadow_artifact_path(envelope, "../visible/output.mp4")

    with pytest.raises(ShadowIsolationError, match="shadow root"):
        validate_shadow_artifact_path(envelope, tmp_path / "report-1" / "video_enhance" / "output.mp4")


def test_shadow_target_rejects_production_task_paths_and_remote_storage(tmp_path: Path) -> None:
    envelope = create_shadow_envelope("report-1", "video_enhance", artifacts_dir=tmp_path)

    with pytest.raises(ShadowIsolationError, match="task-upload"):
        validate_shadow_target(envelope, "image_uploads/user-42/tasks/task-1/output.mp4")

    with pytest.raises(ShadowIsolationError, match="remote storage"):
        validate_shadow_target(
            envelope,
            "https://example.supabase.co/storage/v1/object/public/image_uploads/user-42/tasks/task-1/output.mp4",
        )

    with pytest.raises(ShadowIsolationError, match="Remote URLs|remote URLs"):
        validate_shadow_target(envelope, "https://example.com/result.png")


def test_shadow_target_allows_only_explicit_disposable_remote_storage(tmp_path: Path) -> None:
    envelope = create_shadow_envelope("report-1", "video_enhance", artifacts_dir=tmp_path)

    with pytest.raises(ShadowIsolationError):
        validate_shadow_target(
            envelope,
            "https://disposable.example.supabase.co/storage/v1/object/public/image_uploads/shadow/output.mp4",
            allow_disposable_remote=True,
        )

    result = validate_shadow_target(
        envelope,
        "https://disposable.example.supabase.co/storage/v1/object/public/image_uploads/disposable-shadow/output.mp4",
        allow_disposable_remote=True,
    )
    assert result["status"] == "green"
    assert result["target_kind"] == "disposable_remote_storage"


def test_shadow_report_records_skipped_side_effects(tmp_path: Path) -> None:
    envelope = create_shadow_envelope("report-1", "video_enhance", artifacts_dir=tmp_path)
    report = shadow_isolation_report(
        envelope,
        attempted_targets=[
            "local/result.mp4",
            "https://example.supabase.co/storage/v1/object/public/image_uploads/user/tasks/task/result.mp4",
        ],
    )

    assert report["status"] == "red"
    assert report["targets"][0]["status"] == "green"
    assert report["targets"][1]["status"] == "red"
    assert {
        effect["effect"] for effect in report["skipped_side_effects"]
    } == {"completion", "billing", "upload", "user_visible"}
    assert {effect["status"] for effect in report["skipped_side_effects"]} == {"skipped"}
