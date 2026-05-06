"""Local-worker HTTP surface for browser-to-worker file materialization."""

from __future__ import annotations

import cgi
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from source.core.log import headless_logger
from source.runtime.worker.health_labels import build_health_payload
from source.runtime.worker.resource_pressure import ensure_resources_for_write


_STATE_DIR_NAME = ".reigh-local-worker"
_TOKEN_FILE_NAME = "auth-token"
_PORT_FILE_NAME = "port"
_UPLOAD_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]+$")
_JANITOR_FILENAME_RE = re.compile(r"^[a-f0-9]{32}\.[A-Za-z0-9]+$")


@dataclass
class LocalHttpServer:
    server: ThreadingHTTPServer
    thread: threading.Thread
    token: str
    state_dir: Path
    token_path: Path
    port_path: Path
    stop_event: threading.Event
    janitor_thread: threading.Thread | None = None

    def shutdown(self) -> None:
        self.stop_event.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.janitor_thread is not None:
            self.janitor_thread.join(timeout=5)
        for path in (self.token_path, self.port_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                headless_logger.debug(f"Failed to remove local HTTP state file {path}: {exc}")


def start_local_http_server(
    *,
    materialization_dir: Path,
    port: int,
    worker_id: str,
    version: str,
    auth_optional: bool = False,
    file_ttl_seconds: int = 21600,
    janitor_interval_seconds: int = 1800,
) -> LocalHttpServer:
    materialization_dir = materialization_dir.expanduser()
    materialization_dir.mkdir(mode=0o700, exist_ok=True, parents=True)
    try:
        os.chmod(materialization_dir, 0o700)
    except OSError as exc:
        headless_logger.debug(f"Failed to chmod local materialization dir {materialization_dir}: {exc}")

    state_dir = Path.home() / _STATE_DIR_NAME
    state_dir.mkdir(mode=0o700, exist_ok=True, parents=True)
    try:
        os.chmod(state_dir, 0o700)
    except OSError as exc:
        headless_logger.debug(f"Failed to chmod local HTTP state dir {state_dir}: {exc}")

    token = secrets.token_urlsafe(32)
    token_path = state_dir / _TOKEN_FILE_NAME
    port_path = state_dir / _PORT_FILE_NAME

    handler_class = _make_handler(
        materialization_dir=materialization_dir,
        worker_id=worker_id,
        version=version,
        token=token,
        auth_optional=auth_optional,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_class)
    actual_port = int(server.server_address[1])

    _write_private_file(token_path, token)
    _write_private_file(port_path, f"{actual_port}\n")

    stop_event = threading.Event()
    janitor = _FileJanitor(
        materialization_dir=materialization_dir,
        file_ttl_seconds=file_ttl_seconds,
        interval_seconds=janitor_interval_seconds,
        stop_event=stop_event,
    )
    janitor_thread = threading.Thread(target=janitor.run, name="reigh-local-file-janitor", daemon=True)
    janitor_thread.start()

    thread = threading.Thread(target=server.serve_forever, name="reigh-local-http", daemon=True)
    thread.start()

    return LocalHttpServer(
        server=server,
        thread=thread,
        token=token,
        state_dir=state_dir,
        token_path=token_path,
        port_path=port_path,
        stop_event=stop_event,
        janitor_thread=janitor_thread,
    )


def _write_private_file(path: Path, content: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)


class _FileJanitor:
    def __init__(
        self,
        *,
        materialization_dir: Path,
        file_ttl_seconds: int,
        interval_seconds: int,
        stop_event: threading.Event,
    ) -> None:
        self.materialization_dir = materialization_dir
        self.file_ttl_seconds = file_ttl_seconds
        self.interval_seconds = interval_seconds
        self.stop_event = stop_event

    def run(self) -> None:
        self._sweep()
        while not self.stop_event.wait(self.interval_seconds):
            self._sweep()

    def _sweep(self) -> None:
        try:
            candidates = list(self.materialization_dir.iterdir())
        except OSError as exc:
            headless_logger.debug(f"Local file janitor could not scan {self.materialization_dir}: {exc}")
            return

        now = time.time()
        for candidate in candidates:
            if not _JANITOR_FILENAME_RE.match(candidate.name):
                continue
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                age = now - candidate.stat().st_mtime
            except OSError as exc:
                headless_logger.debug(f"Local file janitor could not stat {candidate}: {exc}")
                continue
            if age <= self.file_ttl_seconds:
                continue
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                headless_logger.debug(f"Local file janitor could not remove {candidate}: {exc}")


def _make_handler(
    *,
    materialization_dir: Path,
    worker_id: str,
    version: str,
    token: str,
    auth_optional: bool,
) -> type[BaseHTTPRequestHandler]:
    resolved_materialization_dir = materialization_dir.resolve()

    class LocalWorkerRequestHandler(BaseHTTPRequestHandler):
        server_version = "ReighLocalWorkerHTTP/1.0"

        def do_OPTIONS(self) -> None:
            self._send_empty(HTTPStatus.NO_CONTENT)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            payload = build_health_payload(
                worker_id=worker_id,
                version=version,
                disk_paths=[materialization_dir],
            )
            self._send_json(HTTPStatus.OK, payload)

        def do_POST(self) -> None:
            if self.path == "/ingest":
                self._handle_ingest()
                return
            if self.path == "/cleanup":
                self._handle_cleanup()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_HEAD(self) -> None:
            self._send_empty(HTTPStatus.NOT_FOUND)

        def _send_not_found(self) -> None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        do_PUT = _send_not_found
        do_PATCH = _send_not_found
        do_DELETE = _send_not_found

        def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
            del explain
            status = HTTPStatus.NOT_FOUND if code == HTTPStatus.NOT_IMPLEMENTED else HTTPStatus(code)
            self._send_json(status, {"error": message or status.phrase})

        def log_message(self, format: str, *args: Any) -> None:
            headless_logger.debug(f"Local HTTP {self.address_string()} - {format % args}")

        def _handle_ingest(self) -> None:
            if not self._is_authorized():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "expected multipart/form-data"})
                return

            try:
                required_bytes = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            file_field = form["file"] if "file" in form else None
            if isinstance(file_field, list) or file_field is None or not getattr(file_field, "file", None):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "expected exactly one file part named file"})
                return

            filename = getattr(file_field, "filename", "") or ""
            suffix_match = _UPLOAD_SUFFIX_RE.search(filename)
            suffix = suffix_match.group(0) if suffix_match else ".bin"
            output_path = materialization_dir / f"{uuid.uuid4().hex}{suffix}"

            resource_check = ensure_resources_for_write(
                worker_id=worker_id,
                target_path=output_path,
                required_bytes=required_bytes,
            )
            if not resource_check.allow_work:
                self._send_json(
                    HTTPStatus.INSUFFICIENT_STORAGE,
                    {
                        "error": "insufficient local disk space",
                        "resource_pressure": resource_check.to_state(),
                    },
                )
                return

            fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            size = 0
            try:
                with os.fdopen(fd, "wb") as handle:
                    while True:
                        chunk = file_field.file.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        size += len(chunk)
            except Exception:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError as exc:
                    headless_logger.debug(f"Failed to clean partial ingest file {output_path}: {exc}")
                raise

            self._send_json(HTTPStatus.OK, {"path": str(output_path.resolve()), "size": size})

        def _handle_cleanup(self) -> None:
            if not self._is_authorized():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                return

            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return

            payload_path = payload.get("path") if isinstance(payload, dict) else None
            if not isinstance(payload_path, str):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "path is required"})
                return

            candidate = Path(payload_path)
            if any(part == ".." for part in candidate.parts):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "path not allowed"})
                return

            resolved_candidate = candidate.resolve(strict=False)
            if not resolved_candidate.is_relative_to(resolved_materialization_dir):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "path not allowed"})
                return

            if not resolved_candidate.exists():
                self._send_empty(HTTPStatus.NO_CONTENT)
                return

            try:
                resolved_candidate.unlink()
            except FileNotFoundError:
                pass
            self._send_empty(HTTPStatus.NO_CONTENT)

        def _is_authorized(self) -> bool:
            if auth_optional:
                return True
            auth_header = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not auth_header.startswith(prefix):
                return False
            return hmac.compare_digest(auth_header[len(prefix) :], token)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_empty(self, status: HTTPStatus) -> None:
            self.send_response(status)
            self._send_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Max-Age", "600")

    return LocalWorkerRequestHandler
