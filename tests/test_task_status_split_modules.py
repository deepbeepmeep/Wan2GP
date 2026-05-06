"""Direct tests for split task-status helper modules."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

sys.modules.setdefault(
    "httpx",
    SimpleNamespace(
        HTTPError=Exception,
        RequestError=Exception,
        TimeoutException=Exception,
    ),
)

import source.core.db.lifecycle.task_status_complete as complete_router
import source.core.db.lifecycle.task_status_complete_local as complete_local
import source.core.db.lifecycle.task_status_complete_remote as complete_remote
import source.core.db.lifecycle.task_status_retry as retry_mod
import source.core.db.lifecycle.task_status_runtime as runtime_mod
import source.core.db.lifecycle.task_status_update_edge as update_edge_mod
from source.core.db.config import DBRuntimeConfig


class _Resp:
    def __init__(self, status_code: int, text: str = "", payload: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


def _runtime() -> DBRuntimeConfig:
    return DBRuntimeConfig(
        db_type="supabase",
        pg_table_name="tasks",
        supabase_url="https://example.supabase.co",
        supabase_service_key="svc",
        supabase_video_bucket="image_uploads",
        supabase_client=object(),
        supabase_access_token="tok",
        supabase_edge_complete_task_url="https://edge.example/complete",
        supabase_edge_create_task_url=None,
        supabase_edge_claim_task_url=None,
        debug_mode=False,
    )


def test_runtime_resolve_prefers_explicit_runtime():
    runtime = _runtime()
    assert runtime_mod.resolve_runtime_config(runtime) is runtime


def test_runtime_resolve_update_status_request_delegates(monkeypatch):
    captured = {}

    def _fake_resolve_edge_request(function_name, **kwargs):
        captured["function_name"] = function_name
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(runtime_mod, "resolve_edge_request", _fake_resolve_edge_request)
    runtime = _runtime()
    result = runtime_mod.resolve_update_status_request(runtime)
    assert result == "ok"
    assert captured["function_name"] == "update-task-status"


def test_update_status_via_edge_success(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(
        update_edge_mod,
        "resolve_update_status_request",
        lambda _runtime: SimpleNamespace(url="https://edge.example/update", headers={"Authorization": "Bearer x"}),
    )

    captured = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _Resp(200), None

    monkeypatch.setattr(update_edge_mod, "call_edge_function_with_retry", _fake_call)
    ok = update_edge_mod.update_status_via_edge(
        "task-1",
        "In Progress",
        output_location_val="out",
        runtime_config=runtime,
    )
    assert ok is True
    assert captured["function_name"] == "update-task-status"


def test_requeue_retry_falls_back_to_direct_db(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(retry_mod, "resolve_runtime_config", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(
        retry_mod,
        "resolve_update_status_request",
        lambda _runtime: SimpleNamespace(url=None, headers={}),
    )

    called = {}
    monkeypatch.setattr(
        retry_mod,
        "requeue_task_direct_db",
        lambda task_id, attempts, details, runtime_config=None: called.update(
            {"task_id": task_id, "attempts": attempts, "details": details}
        )
        or True,
    )

    ok = retry_mod.requeue_task_for_retry("task-2", "temporary", 1, "network")
    assert ok is True
    assert called["task_id"] == "task-2"
    assert called["attempts"] == 2


def test_complete_local_file_base64_path(monkeypatch, tmp_path: Path):
    output_file = tmp_path / "artifact.txt"
    output_file.write_text("hello-world")
    runtime = _runtime()
    complete_request = SimpleNamespace(url="https://edge.example/complete", headers={})

    calls = []

    def _fake_call(**kwargs):
        calls.append(kwargs["function_name"])
        return _Resp(200, payload={"public_url": "https://cdn.example/file.txt"}), None

    monkeypatch.setattr(complete_local, "call_edge_function_with_retry", _fake_call)
    result = complete_local.complete_task_with_local_file(
        "task-3",
        output_file,
        complete_request=complete_request,
        runtime=runtime,
    )
    assert result["public_url"] == "https://cdn.example/file.txt"
    assert "complete_task" in calls


def test_complete_local_small_base64_video_includes_thumbnail(monkeypatch, tmp_path: Path):
    output_file = tmp_path / "artifact.mp4"
    output_file.write_bytes(b"video-with-audio")
    runtime = _runtime()
    complete_request = SimpleNamespace(url="https://edge.example/complete", headers={})
    captured = {}

    def _fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"thumb")
        return SimpleNamespace(returncode=0)

    def _fake_call(**kwargs):
        captured.update(kwargs["payload"])
        return _Resp(200, payload={"public_url": "https://cdn.example/file.mp4"}), None

    monkeypatch.setattr(complete_local.subprocess, "run", _fake_run)
    monkeypatch.setattr(complete_local, "call_edge_function_with_retry", _fake_call)

    result = complete_local.complete_task_with_local_file(
        "task-video-small",
        output_file,
        complete_request=complete_request,
        runtime=runtime,
    )

    assert result["public_url"] == "https://cdn.example/file.mp4"
    assert captured["filename"] == "artifact.mp4"
    assert captured["first_frame_filename"] == "thumb_task-vid.jpg"
    assert captured["first_frame_data"]


def test_complete_remote_storage_path_payload(monkeypatch):
    runtime = _runtime()
    complete_request = SimpleNamespace(url="https://edge.example/complete", headers={})
    captured = {}

    def _fake_complete_with_payload(task_id_str, payload, **_kwargs):
        captured["task_id"] = task_id_str
        captured["payload"] = payload
        return _Resp(200, payload={"public_url": "https://cdn.example/remote.mp4"})

    monkeypatch.setattr(complete_remote, "_complete_with_payload", _fake_complete_with_payload)
    result = complete_remote.complete_task_with_remote_output(
        "task-4",
        "https://example.supabase.co/storage/v1/object/public/image_uploads/user/file.mp4",
        thumbnail_url_val=None,
        complete_request=complete_request,
        runtime=runtime,
    )
    assert captured["payload"]["storage_path"] == "user/file.mp4"
    assert result["public_url"] == "https://cdn.example/remote.mp4"


def test_complete_remote_preserves_thumbnail_storage_path(monkeypatch):
    runtime = _runtime()
    complete_request = SimpleNamespace(url="https://edge.example/complete", headers={})
    captured = {}

    def _fake_complete_with_payload(task_id_str, payload, **_kwargs):
        captured["task_id"] = task_id_str
        captured["payload"] = payload
        return _Resp(200, payload={"public_url": "https://cdn.example/remote.mp4"})

    monkeypatch.setattr(complete_remote, "_complete_with_payload", _fake_complete_with_payload)
    result = complete_remote.complete_task_with_remote_output(
        "task-remote-thumb",
        "https://example.supabase.co/storage/v1/object/public/image_uploads/user/tasks/task-remote-thumb/file.mp4",
        thumbnail_url_val="https://example.supabase.co/storage/v1/object/public/image_uploads/user/tasks/task-remote-thumb/thumbnails/thumb.jpg",
        complete_request=complete_request,
        runtime=runtime,
    )

    assert captured["payload"] == {
        "task_id": "task-remote-thumb",
        "storage_path": "user/tasks/task-remote-thumb/file.mp4",
        "thumbnail_storage_path": "user/tasks/task-remote-thumb/thumbnails/thumb.jpg",
    }
    assert result["public_url"] == "https://cdn.example/remote.mp4"


def test_complete_router_uses_update_edge_for_non_complete(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(complete_router, "resolve_runtime_config", lambda *_args, **_kwargs: runtime)

    captured = {}
    monkeypatch.setattr(
        complete_router,
        "update_status_via_edge",
        lambda task_id, status, **kwargs: captured.update(
            {"task_id": task_id, "status": status, "kwargs": kwargs}
        )
        or True,
    )

    result = complete_router.update_task_status_supabase_legacy(
        "task-5",
        "In Progress",
        "out",
    )
    assert result is True
    assert captured["task_id"] == "task-5"


def test_complete_remote_marks_failed_when_complete_task_call_fails(monkeypatch):
    runtime = _runtime()
    complete_request = SimpleNamespace(url="https://edge.example/complete", headers={})
    marked = {}

    monkeypatch.setattr(
        complete_remote,
        "call_edge_function_with_retry",
        lambda **_kwargs: (None, "edge unavailable"),
    )
    monkeypatch.setattr(
        complete_remote,
        "mark_task_failed_via_edge_function",
        lambda task_id, message, runtime_config=None: marked.update(
            {"task_id": task_id, "message": message, "runtime": runtime_config}
        ),
    )

    result = complete_remote.complete_task_with_remote_output(
        "task-remote-fail",
        "https://example.supabase.co/storage/v1/object/public/image_uploads/user/file.mp4",
        thumbnail_url_val=None,
        complete_request=complete_request,
        runtime=runtime,
    )

    assert result is False
    assert marked["task_id"] == "task-remote-fail"
    assert marked["runtime"] is runtime
    assert marked["message"].startswith("Completion failed:")


def test_complete_local_marks_failed_when_presigned_upload_put_fails(monkeypatch, tmp_path: Path):
    runtime = _runtime()
    complete_request = SimpleNamespace(url="https://edge.example/complete", headers={})
    marked = {}

    output_file = tmp_path / "artifact.txt"
    output_file.write_text("payload")

    monkeypatch.setattr(complete_local, "FILE_SIZE_THRESHOLD_MB", 0.0)
    monkeypatch.setattr(
        complete_local,
        "resolve_generate_upload_url_request",
        lambda _runtime: SimpleNamespace(url="https://edge.example/generate-upload", headers={}),
    )

    call_count = {"n": 0}

    def _fake_edge_call(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                _Resp(
                    200,
                    payload={
                        "upload_url": "https://upload.example/file",
                        "storage_path": "user/file.txt",
                    },
                ),
                None,
            )
        return None, "put failed"

    monkeypatch.setattr(complete_local, "call_edge_function_with_retry", _fake_edge_call)
    monkeypatch.setattr(
        complete_local,
        "mark_task_failed_via_edge_function",
        lambda task_id, message, runtime_config=None: marked.update(
            {"task_id": task_id, "message": message, "runtime": runtime_config}
        ),
    )

    result = complete_local.complete_task_with_local_file(
        "task-local-fail",
        output_file,
        complete_request=complete_request,
        runtime=runtime,
    )

    assert result is False
    assert marked["task_id"] == "task-local-fail"
    assert marked["runtime"] is runtime
    assert marked["message"].startswith("Upload failed:")


def test_complete_local_large_presigned_video_uploads_thumbnail_and_completes_with_thumbnail_storage_path(monkeypatch, tmp_path: Path):
    runtime = _runtime()
    complete_request = SimpleNamespace(url="https://edge.example/complete", headers={})
    output_file = tmp_path / "large.mp4"
    output_file.write_bytes(b"video-with-audio")
    calls: list[dict] = []

    monkeypatch.setattr(complete_local, "FILE_SIZE_THRESHOLD_MB", 0.0)
    monkeypatch.setattr(
        complete_local,
        "resolve_generate_upload_url_request",
        lambda _runtime: SimpleNamespace(url="https://edge.example/generate-upload", headers={}),
    )

    def _fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"thumb")
        return SimpleNamespace(returncode=0)

    def _fake_edge_call(**kwargs):
        calls.append(kwargs)
        function_name = kwargs["function_name"]
        if function_name == "generate-upload-url":
            return (
                _Resp(
                    200,
                    payload={
                        "upload_url": "https://upload.example/file",
                        "storage_path": "user/tasks/task-large-video/large.mp4",
                        "thumbnail_upload_url": "https://upload.example/thumb",
                        "thumbnail_storage_path": "user/tasks/task-large-video/thumbnails/thumb.jpg",
                    },
                ),
                None,
            )
        if function_name in {"storage-upload-thumbnail", "storage-upload-file"}:
            return _Resp(201), None
        if function_name == "complete_task":
            return _Resp(200, payload={"public_url": "https://cdn.example/large.mp4"}), None
        raise AssertionError(function_name)

    monkeypatch.setattr(complete_local.subprocess, "run", _fake_run)
    monkeypatch.setattr(complete_local, "call_edge_function_with_retry", _fake_edge_call)

    result = complete_local.complete_task_with_local_file(
        "task-large-video",
        output_file,
        complete_request=complete_request,
        runtime=runtime,
    )

    assert result["public_url"] == "https://cdn.example/large.mp4"
    assert [call["function_name"] for call in calls] == [
        "generate-upload-url",
        "storage-upload-thumbnail",
        "storage-upload-file",
        "complete_task",
    ]
    assert calls[0]["payload"]["generate_thumbnail_url"] is True
    assert calls[1]["edge_url"] == "https://upload.example/thumb"
    assert calls[1]["headers"]["Content-Type"] == "image/jpeg"
    assert calls[2]["payload"] == output_file
    assert calls[3]["payload"] == {
        "task_id": "task-large-video",
        "storage_path": "user/tasks/task-large-video/large.mp4",
        "thumbnail_storage_path": "user/tasks/task-large-video/thumbnails/thumb.jpg",
    }
