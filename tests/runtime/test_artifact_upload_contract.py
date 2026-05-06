from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace


def test_output_paths_requests_intermediate_artifact_upload_url(monkeypatch, tmp_path: Path) -> None:
    from source.core.db import config as db_config
    from source.utils import output_paths

    local_file = tmp_path / "latent.pt"
    local_file.write_bytes(b"latent")
    captured_generate_payload = {}

    class _GenerateResponse:
        status_code = 200
        text = "ok"

        @staticmethod
        def json():
            return {
                "upload_url": "https://upload.example",
                "storage_path": "user/tasks/task-1/intermediates/latent.pt",
            }

    class _UploadResponse:
        status_code = 201
        text = "created"

    def fake_post(_url, *, json, **_kwargs):
        captured_generate_payload.update(json)
        return _GenerateResponse()

    monkeypatch.setattr(db_config, "SUPABASE_URL", "https://example.supabase.co", raising=False)
    monkeypatch.setattr(db_config, "SUPABASE_ACCESS_TOKEN", "token", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            post=fake_post,
            put=lambda *_args, **_kwargs: _UploadResponse(),
            HTTPError=Exception,
        ),
    )

    result = output_paths.upload_intermediate_file_to_storage(local_file, "task-1", "latent.pt")

    assert result == "https://example.supabase.co/storage/v1/object/public/image_uploads/user/tasks/task-1/intermediates/latent.pt"
    assert captured_generate_payload["task_id"] == "task-1"
    assert captured_generate_payload["filename"] == "latent.pt"
    assert captured_generate_payload["content_type"]
    assert captured_generate_payload["artifact_class"] == "intermediate"


def test_video_storage_requests_intermediate_artifact_upload_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(HTTPError=Exception, TimeoutException=Exception, RequestError=Exception),
    )
    from source.media.video import storage

    local_file = tmp_path / "preview.mp4"
    local_file.write_bytes(b"video")
    captured_payload = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "upload_url": "https://upload.example",
                "storage_path": "user/tasks/task-1/intermediates/preview.mp4",
            }

    class _UploadResponse:
        status_code = 201

    monkeypatch.setattr(
        storage,
        "resolve_edge_request",
        lambda *_args, **_kwargs: SimpleNamespace(
            url="https://edge.example/generate-upload-url",
            headers={"Authorization": "Bearer token"},
        ),
    )
    monkeypatch.setattr(storage, "has_required_edge_credentials", lambda _headers: True)

    def fake_call_edge_function_with_retry(**kwargs):
        if kwargs["function_name"] == "generate-upload-url":
            captured_payload.update(kwargs["payload"])
            return _Response(), None
        return _UploadResponse(), None

    monkeypatch.setattr(storage, "call_edge_function_with_retry", fake_call_edge_function_with_retry)

    result = storage.upload_intermediate_file_to_storage(
        local_file,
        "task-1",
        "preview.mp4",
        runtime_config=SimpleNamespace(supabase_url="https://example.supabase.co"),
    )

    assert result == "https://example.supabase.co/storage/v1/object/public/image_uploads/user/tasks/task-1/intermediates/preview.mp4"
    assert captured_payload == {
        "task_id": "task-1",
        "filename": "preview.mp4",
        "content_type": "video/mp4",
        "artifact_class": "intermediate",
    }
