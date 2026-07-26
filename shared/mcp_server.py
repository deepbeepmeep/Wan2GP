"""MCP server adapter for WanGP's in-process API."""

import argparse
import contextlib
import copy
import dataclasses
import io
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.api import SessionJob


_MAX_STORED_EVENTS = 500
_TRANSPORT_ALIASES = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable-http": "streamable-http",
    "streamable_http": "streamable-http",
}


def _normalize_transport(value: str | None) -> str:
    transport = str(value or "stdio").strip().lower()
    try:
        return _TRANSPORT_ALIASES[transport]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported MCP transport: {value}. Use stdio, sse, or streamable-http.") from exc


@contextlib.contextmanager
def _stdio_safe_startup_output(transport: str):
    if transport != "stdio":
        yield
        return
    target = sys.stderr if sys.stderr is not None else io.StringIO()
    with contextlib.redirect_stdout(target):
        yield


def _artifact_to_dict(artifact: Any) -> dict[str, Any]:
    return {
        "path": artifact.path,
        "media_type": artifact.media_type,
        "client_id": artifact.client_id,
        "hdr": artifact.hdr,
        "audio_sampling_rate": artifact.audio_sampling_rate,
        "fps": artifact.fps,
        "has_video_tensor_uint8": artifact.video_tensor_uint8 is not None,
        "has_video_tensor_hdr": artifact.video_tensor_hdr is not None,
        "has_audio_tensor": artifact.audio_tensor is not None,
        "has_flashvsr_continue_cache": artifact.flashvsr_continue_cache is not None,
    }


def _error_to_dict(error: Any) -> dict[str, Any]:
    return {
        "message": error.message,
        "task_index": error.task_index,
        "task_id": error.task_id,
        "stage": error.stage,
        "cancelled": error.cancelled,
    }


def _result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "success": result.success,
        "cancelled": result.cancelled,
        "generated_files": list(result.generated_files),
        "errors": [_error_to_dict(error) for error in result.errors],
        "total_tasks": result.total_tasks,
        "successful_tasks": result.successful_tasks,
        "failed_tasks": result.failed_tasks,
        "artifacts": [_artifact_to_dict(artifact) for artifact in result.artifacts],
    }


def _event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "timestamp": event.timestamp,
        "data": _json_safe(event.data),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    class_name = type(value).__name__
    if class_name == "GeneratedArtifact":
        return _artifact_to_dict(value)
    if class_name == "GenerationError":
        return _error_to_dict(value)
    if class_name == "GenerationResult":
        return _result_to_dict(value)
    if class_name == "SessionEvent":
        return _event_to_dict(value)
    if class_name == "StreamMessage":
        return {"stream": value.stream, "text": value.text}
    if class_name == "PreviewUpdate":
        return {
            "has_image_preview": value.image is not None,
            "phase": value.phase,
            "status": value.status,
            "progress": value.progress,
            "current_step": value.current_step,
            "total_steps": value.total_steps,
        }
    if dataclasses.is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return {"type": type(value).__name__, "shape": list(shape) if shape is not None else None, "dtype": str(dtype) if dtype is not None else None}
    return str(value)


class _JobRecord:
    def __init__(self, job_id: str, job: "SessionJob") -> None:
        self.job_id = job_id
        self.job = job
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name=f"wangp-mcp-job-{job_id}")

    def start(self) -> None:
        self._watcher.start()

    def _watch(self) -> None:
        for event in self.job.events.iter(timeout=0.2):
            event_dict = _event_to_dict(event)
            with self._lock:
                self.events.append(event_dict)
                if len(self.events) > _MAX_STORED_EVENTS:
                    del self.events[: len(self.events) - _MAX_STORED_EVENTS]
                self.updated_at = time.time()
                if event.kind == "completed" and type(event.data).__name__ == "GenerationResult":
                    self.result = _result_to_dict(event.data)
        self._capture_result_if_done()

    def _capture_result_if_done(self) -> None:
        if not self.job.done:
            return
        try:
            result = _result_to_dict(self.job.result(timeout=0))
        except Exception:
            return
        with self._lock:
            self.result = result
            self.updated_at = time.time()

    def snapshot(self, event_limit: int = 20) -> dict[str, Any]:
        self._capture_result_if_done()
        event_limit = 20 if event_limit is None else event_limit
        event_limit = max(0, min(int(event_limit), _MAX_STORED_EVENTS))
        with self._lock:
            events = copy.deepcopy(self.events[-event_limit:] if event_limit else [])
            result = copy.deepcopy(self.result)
            updated_at = self.updated_at
        return {
            "job_id": self.job_id,
            "done": self.job.done,
            "cancel_requested": self.job.cancel_requested,
            "created_at": self.created_at,
            "updated_at": updated_at,
            "events": events,
            "result": result,
        }


class _JobStore:
    def __init__(self, session) -> None:
        self._session = session
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()

    def submit(self, source: dict[str, Any] | list[dict[str, Any]]) -> _JobRecord:
        job = self._session.submit(source)
        record = _JobRecord(uuid.uuid4().hex, job)
        with self._lock:
            self._jobs[record.job_id] = record
        record.start()
        return record

    def get(self, job_id: str) -> _JobRecord:
        with self._lock:
            record = self._jobs.get(str(job_id or "").strip())
        if record is None:
            raise KeyError(f"Unknown WanGP job_id: {job_id}")
        return record


def _config_file_from_arg(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        return str(path / "wgp_config.json")
    return str(path)


def build_server(args: argparse.Namespace):
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError("WanGP MCP server requires the 'mcp' Python package. Install project requirements or run `pip install mcp`.") from exc

    from shared.api import init

    args.transport = _normalize_transport(getattr(args, "transport", "stdio"))
    console_output = bool(args.console_output) and args.transport != "stdio"
    with _stdio_safe_startup_output(args.transport):
        session = init(
            root=args.root,
            config_path=_config_file_from_arg(args.config),
            output_dir=args.output_dir,
            cli_args=tuple(args.cli_arg or ()),
            console_output=console_output,
            console_isatty=False,
        )
    jobs = _JobStore(session)
    settings: dict[str, Any] = {}
    if args.host is not None:
        settings["host"] = args.host
    if args.port is not None:
        settings["port"] = args.port
    if args.transport == "streamable-http":
        settings["json_response"] = True
        settings["stateless_http"] = True
    mcp = FastMCP("WanGP", **settings)

    @mcp.tool()
    def wangp_list_models(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None, include_availability: bool = False) -> list[dict[str, Any]]:
        """List compact model metadata records, optionally filtered by metadata fields."""

        return session.list_model_metadata(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs, include_availability=include_availability)

    @mcp.tool()
    def wangp_list_model_defs(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None) -> list[dict[str, Any]]:
        """List full WanGP model definitions, optionally filtered by metadata fields."""

        return session.list_model_defs(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs)

    @mcp.tool()
    def wangp_get_model(model_type: str) -> dict[str, Any] | None:
        """Return one full WanGP model definition."""

        return session.get_model_def(model_type)

    @mcp.tool()
    def wangp_get_model_metadata(model_type: str, include_availability: bool = False) -> dict[str, Any] | None:
        """Return one compact model metadata record."""

        return session.get_model_metadata(model_type, include_availability=include_availability)

    @mcp.tool()
    def wangp_get_model_availability(model_type: str) -> dict[str, Any]:
        """Return local file availability for one model using the same status as the UI model selector."""

        return session.get_model_availability(model_type)

    @mcp.tool()
    def wangp_list_model_availability(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None) -> list[dict[str, Any]]:
        """List local file availability for models, optionally filtered by metadata fields."""

        return session.list_model_availability(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs)

    @mcp.tool()
    def wangp_get_default_settings(model_type: str) -> dict[str, Any]:
        """Return generated default settings for a model."""

        return session.get_default_settings(model_type)

    @mcp.tool()
    def wangp_get_model_schema(model_type: str) -> dict[str, Any] | None:
        """Return model definition, inferred metadata, setting values, and default settings."""

        return session.get_model_schema(model_type)

    @mcp.tool()
    def wangp_generate(source: dict[str, Any] | list[dict[str, Any]], wait: bool = False, timeout_s: float | None = None, event_limit: int = 20) -> dict[str, Any]:
        """Start a WanGP generation from a settings dict, task dict, or task list."""

        if not isinstance(source, (dict, list)):
            raise TypeError("source must be a settings dict, task dict, manifest dict, or task list")
        record = jobs.submit(source)
        if wait:
            record.job.result(timeout=timeout_s)
        return record.snapshot(event_limit=event_limit)

    @mcp.tool()
    def wangp_get_job(job_id: str, event_limit: int = 20) -> dict[str, Any]:
        """Poll a WanGP generation job."""

        return jobs.get(job_id).snapshot(event_limit=event_limit)

    @mcp.tool()
    def wangp_cancel_job(job_id: str) -> dict[str, Any]:
        """Request cancellation of a WanGP generation job."""

        record = jobs.get(job_id)
        record.job.cancel()
        return record.snapshot(event_limit=20)

    return mcp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WanGP as an MCP server.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="WanGP repository root.")
    parser.add_argument("--config", default=None, help="Path to wgp_config.json or a directory containing it.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated media.")
    parser.add_argument("--cli-arg", action="append", default=[], help="Extra argument passed to wgp.py during runtime initialization. Repeat for multiple args.")
    parser.add_argument("--console-output", action="store_true", help="Mirror WanGP stdout/stderr to the MCP server console.")
    parser.add_argument("--transport", default="stdio", help="MCP transport: stdio, sse, or streamable-http.")
    parser.add_argument("--host", default=None, help="Optional host for non-stdio transports.")
    parser.add_argument("--port", type=int, default=None, help="Optional port for non-stdio transports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        server = build_server(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return run_server(server, args)


def run_server(server, args: argparse.Namespace) -> int:
    transport = _normalize_transport(getattr(args, "transport", "stdio"))
    if transport == "stdio":
        server.run()
    else:
        server.run(transport=transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
