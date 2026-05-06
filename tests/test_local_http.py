import os
import socket
from pathlib import Path
from unittest import mock

import pytest
import requests

from source.runtime.worker.local_http import start_local_http_server
from source.runtime.worker.preflight import write_preflight_state
from source.runtime.worker.resource_pressure import ResourcePressureResult


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def local_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path / "preflight"))
    monkeypatch.setenv("REIGH_WARM_CACHE_STATE_DIR", str(tmp_path / "warm-cache"))
    for name in (
        "REIGH_BACKEND",
        "WORKER_BACKEND",
        "REIGH_WORKER_PROFILE",
        "WGP_PROFILE",
        "REIGH_WORKER_POOL",
        "WORKER_POOL",
        "REIGH_SELECTOR_NAMESPACE",
        "ROUTE_SELECTOR_NAMESPACE",
        "REIGH_SELECTOR_VERSION",
        "ROUTE_SELECTOR_VERSION",
        "REIGH_WORKER_RUN_ID",
        "WORKER_RUN_ID",
        "REIGH_WORKER_CONTRACT_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    server = start_local_http_server(
        materialization_dir=tmp_path / "mat",
        port=_free_port(),
        worker_id="worker-test",
        version="test-version",
        file_ttl_seconds=3600,
        janitor_interval_seconds=99999,
    )
    try:
        yield server, tmp_path / "mat"
    finally:
        server.shutdown()


def _base_url(server) -> str:
    return f"http://127.0.0.1:{server.server.server_address[1]}"


def _auth_headers(server) -> dict[str, str]:
    return {"Authorization": f"Bearer {server.token}", "Connection": "close"}


def test_health_no_auth_returns_ok(local_server):
    server, _ = local_server

    response = requests.get(f"{_base_url(server)}/health", headers={"Connection": "close"}, timeout=5)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["worker_id"] == "worker-test"
    assert payload["version"] == "test-version"
    assert payload["labels"]["route"]["backend"] == "wgp"
    assert payload["labels"]["disk"]["status"] in {"ok", "near_full"}
    assert payload["labels"]["warm_cache"]["status"] == "unknown"


def test_ingest_missing_auth_returns_401(local_server):
    server, _ = local_server

    response = requests.post(
        f"{_base_url(server)}/ingest",
        files={"file": ("input.txt", b"data")},
        headers={"Connection": "close"},
        timeout=5,
    )

    assert response.status_code == 401


def test_ingest_wrong_token_returns_401(local_server):
    server, _ = local_server

    response = requests.post(
        f"{_base_url(server)}/ingest",
        files={"file": ("input.txt", b"data")},
        headers={"Authorization": "Bearer wrong", "Connection": "close"},
        timeout=5,
    )

    assert response.status_code == 401


def test_ingest_valid_upload_returns_path(local_server):
    server, materialization_dir = local_server

    response = requests.post(
        f"{_base_url(server)}/ingest",
        files={"file": ("input.png", b"image-bytes")},
        headers=_auth_headers(server),
        timeout=5,
    )

    assert response.status_code == 200
    payload = response.json()
    output_path = Path(payload["path"])
    assert payload["size"] == len(b"image-bytes")
    assert output_path.exists()
    assert output_path.read_bytes() == b"image-bytes"
    assert output_path.resolve().is_relative_to(materialization_dir.resolve())
    if os.name == "posix":
        assert output_path.stat().st_mode & 0o777 == 0o600


def test_ingest_returns_507_when_disk_cleanup_cannot_recover(local_server, monkeypatch):
    server, _ = local_server
    import source.runtime.worker.local_http as local_http

    blocked = ResourcePressureResult(
        status="near_full",
        action="write_blocked",
        allow_work=False,
        quota_alert=True,
        required_free_bytes=1024,
        recovered_bytes=0,
        volumes=(),
        cleanup={"lora": {}, "artifacts": {}},
        reason="disk_pressure_unrecoverable",
    )

    monkeypatch.setattr(local_http, "ensure_resources_for_write", lambda **_kwargs: blocked)

    response = requests.post(
        f"{_base_url(server)}/ingest",
        files={"file": ("input.png", b"image-bytes")},
        headers=_auth_headers(server),
        timeout=5,
    )

    assert response.status_code == 507
    payload = response.json()
    assert payload["error"] == "insufficient local disk space"
    assert payload["resource_pressure"]["resource_pressure_action"] == "write_blocked"


def test_ingest_accepts_no_auth_when_auth_optional_true(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = start_local_http_server(
        materialization_dir=tmp_path / "mat",
        port=_free_port(),
        worker_id="worker-test",
        version="test-version",
        auth_optional=True,
        file_ttl_seconds=3600,
        janitor_interval_seconds=99999,
    )
    try:
        response = requests.post(
            f"{_base_url(server)}/ingest",
            files={"file": ("input.txt", b"data")},
            headers={"Connection": "close"},
            timeout=5,
        )
    finally:
        server.shutdown()

    assert response.status_code == 200


def test_cleanup_rejects_dotdot_component_literal(local_server):
    server, materialization_dir = local_server

    response = requests.post(
        f"{_base_url(server)}/cleanup",
        json={"path": str(materialization_dir / ".." / "outside.txt")},
        headers=_auth_headers(server),
        timeout=5,
    )

    assert response.status_code == 400


def test_cleanup_rejects_absolute_path_outside_dir(local_server, tmp_path):
    server, _ = local_server
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("outside", encoding="utf-8")

    with mock.patch.object(Path, "unlink") as unlink_mock:
        response = requests.post(
            f"{_base_url(server)}/cleanup",
            json={"path": str(outside_path)},
            headers=_auth_headers(server),
            timeout=5,
        )
        unlink_mock.assert_not_called()

    assert response.status_code == 400


def test_cleanup_rejects_traversal_escape(local_server, tmp_path):
    server, materialization_dir = local_server
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("outside", encoding="utf-8")

    response = requests.post(
        f"{_base_url(server)}/cleanup",
        json={"path": str(materialization_dir / ".." / outside_path.name)},
        headers=_auth_headers(server),
        timeout=5,
    )

    assert response.status_code == 400
    assert outside_path.exists()


def test_cleanup_deletes_valid_file_returns_204(local_server):
    server, materialization_dir = local_server
    target = materialization_dir / "delete-me.txt"
    target.write_text("delete", encoding="utf-8")

    response = requests.post(
        f"{_base_url(server)}/cleanup",
        json={"path": str(target)},
        headers=_auth_headers(server),
        timeout=5,
    )

    assert response.status_code == 204
    assert not target.exists()


def test_cleanup_idempotent_on_missing_file(local_server):
    server, materialization_dir = local_server
    target = materialization_dir / "already-gone.txt"

    first_response = requests.post(
        f"{_base_url(server)}/cleanup",
        json={"path": str(target)},
        headers=_auth_headers(server),
        timeout=5,
    )
    second_response = requests.post(
        f"{_base_url(server)}/cleanup",
        json={"path": str(target)},
        headers=_auth_headers(server),
        timeout=5,
    )

    assert first_response.status_code == 204
    assert second_response.status_code == 204


def test_options_preflight_returns_cors_headers(local_server):
    server, _ = local_server

    response = requests.options(f"{_base_url(server)}/anything", headers={"Connection": "close"}, timeout=5)

    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert response.headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"
    assert response.headers["Access-Control-Allow-Headers"] == "Authorization, Content-Type"
    assert response.headers["Access-Control-Max-Age"] == "600"


def test_get_health_includes_cors_headers(local_server):
    server, _ = local_server

    response = requests.get(f"{_base_url(server)}/health", headers={"Connection": "close"}, timeout=5)

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert response.headers["Access-Control-Allow-Headers"] == "Authorization, Content-Type"


def test_get_health_includes_preflight_state_when_available(local_server, tmp_path, monkeypatch):
    server, _ = local_server
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    write_preflight_state("worker-test", {"preflight_status": "passed", "ready_for_tasks": True})

    response = requests.get(f"{_base_url(server)}/health", headers={"Connection": "close"}, timeout=5)

    assert response.status_code == 200
    assert response.json()["preflight"]["preflight_status"] == "passed"
    assert response.json()["labels"]["preflight"]["status"] == "passed"


def test_get_health_includes_safe_route_disk_and_warm_cache_labels(local_server, tmp_path, monkeypatch):
    server, _ = local_server
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    monkeypatch.setenv("REIGH_WORKER_RUN_ID", "run-abc")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-leak")
    monkeypatch.setenv("REIGH_WARM_CACHE_STATE_DIR", str(tmp_path))

    from source.runtime.worker.health_labels import write_warm_cache_state

    write_warm_cache_state(
        "worker-test",
        {
            "warm_cache_status": "hit",
            "warm_cache_model": "vibe-model",
            "warm_cache_source": "manifest",
            "api_token": "secret",
        },
    )

    response = requests.get(f"{_base_url(server)}/health", headers={"Connection": "close"}, timeout=5)
    payload = response.json()

    assert payload["labels"]["route"] == {
        "backend": "vibecomfy",
        "profile": "3",
        "pool": "gpu-wgp-production",
        "selector_namespace": "canary",
        "selector_version": "42",
        "worker_contract_version": "1",
        "run_id": "run-abc",
        "route_key": "",
        "template_id": "",
        "current_task_id": "",
        "current_task_type": "",
    }
    assert payload["labels"]["warm_cache"]["status"] == "hit"
    assert payload["labels"]["warm_cache"]["model"] == "vibe-model"
    assert "must-not-leak" not in response.text
    assert payload["warm_cache"]["api_token"] == "[redacted]"


def test_token_file_mode_is_0600_on_posix(local_server):
    server, _ = local_server

    if os.name != "posix":
        pytest.skip("mode checks are POSIX-specific")

    assert server.token_path.stat().st_mode & 0o777 == 0o600


def test_port_file_contains_bound_port(local_server):
    server, _ = local_server

    assert server.port_path.read_text(encoding="utf-8").strip() == str(server.server.server_address[1])
