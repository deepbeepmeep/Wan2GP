import os
import socket
from pathlib import Path
from unittest import mock

import pytest
import requests

from source.runtime.worker.local_http import start_local_http_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def local_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
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
    assert response.json() == {"ok": True, "worker_id": "worker-test", "version": "test-version"}


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


def test_token_file_mode_is_0600_on_posix(local_server):
    server, _ = local_server

    if os.name != "posix":
        pytest.skip("mode checks are POSIX-specific")

    assert server.token_path.stat().st_mode & 0o777 == 0o600


def test_port_file_contains_bound_port(local_server):
    server, _ = local_server

    assert server.port_path.read_text(encoding="utf-8").strip() == str(server.server.server_address[1])
