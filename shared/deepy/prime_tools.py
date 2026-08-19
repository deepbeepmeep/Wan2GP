from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from concurrent.futures import Future
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Callable

import anyio
from anyio.from_thread import start_blocking_portal

from shared.api import WanGPSession
from shared.deepy.config import DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT, DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT, DEEPY_CONTEXT_TOKENS_KEY, DEEPY_PRIME_MCP_SERVERS_KEY, get_deepy_config_value, normalize_deepy_allow_read_file_system, normalize_deepy_context_tokens, normalize_deepy_prime_mcp_servers
from shared.mcp_server import build_inprocess_server


class DeepyPrimeTools:
    _POLL_INTERVAL_SECONDS = 0.2

    def __init__(self, state: dict[str, Any], send_cmd: Callable[..., None], assistant_session, zero_tools=None) -> None:
        self.state = state
        self.send_cmd = send_cmd
        self.assistant_session = assistant_session
        self._zero_tools = zero_tools
        self.allow_read_file_system = normalize_deepy_allow_read_file_system(get_deepy_config_value(DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT))
        self._tool_progress_callback: Callable[..., None] | None = None
        self._api_session = WanGPSession(webui_state=state, console_output=False, console_isatty=False)
        self._api_session._gradio_webui_context = {"defer_load_queue_trigger": True}
        self._server = build_inprocess_server(self._api_session, toolbox=zero_tools, default_job_event_limit=0, allow_read_file_system=self.allow_read_file_system)
        self._external_servers = normalize_deepy_prime_mcp_servers(get_deepy_config_value(DEEPY_PRIME_MCP_SERVERS_KEY, {}))
        self._request_queue: queue.Queue[Any] = queue.Queue()
        self._ready_event = threading.Event()
        self._worker_error: BaseException | None = None
        self._portal_context = start_blocking_portal(name="deepy-prime-mcp")
        self._portal = self._portal_context.__enter__()
        self._worker_future = self._portal.start_task_soon(self._connection_worker)
        if not self._ready_event.wait(timeout=60):
            self._request_queue.put(None)
            self._portal_context.__exit__(None, None, None)
            raise TimeoutError("Deepy Prime MCP server connections did not initialize in time.")
        if self._worker_error is not None:
            self._portal_context.__exit__(None, None, None)
            raise RuntimeError(f"Deepy Prime MCP server initialization failed: {self._worker_error}") from self._worker_error
        self._tool_defs_by_name = {tool["function"]["name"]: tool for tool in self._tool_defs}
        self._active_job_id = ""
        self._active_job_cancel_requested = False

    def bind_turn(self, state: dict[str, Any], send_cmd: Callable[..., None]) -> None:
        if self.state is not state:
            raise RuntimeError("Deepy Prime cannot move between WanGP browser sessions.")
        self.send_cmd = send_cmd
        if self._zero_tools is not None:
            self._zero_tools.send_cmd = send_cmd

    def close(self) -> None:
        if self._active_job_id:
            try:
                self._call_mcp_tool("wangp_cancel_job", {"job_id": self._active_job_id})
            except Exception:
                pass
            self._active_job_id = ""
        self._request_queue.put(None)
        self._worker_future.result(timeout=30)
        self._portal_context.__exit__(None, None, None)
        self._api_session.close()

    def cancel_active_job(self) -> str:
        job_id = str(self._active_job_id or "").strip()
        if not job_id or self._active_job_cancel_requested:
            return ""
        self._active_job_cancel_requested = True
        try:
            self._call_mcp_tool("wangp_cancel_job", {"job_id": job_id})
        except Exception:
            self._active_job_cancel_requested = False
            raise
        return job_id

    def bind_runtime_tools(self, vision_query_callback=None, tool_progress_callback=None) -> None:
        self._tool_progress_callback = tool_progress_callback
        if self._zero_tools is not None:
            self._zero_tools.bind_runtime_tools(vision_query_callback=vision_query_callback, tool_progress_callback=tool_progress_callback)

    @staticmethod
    def _safe_tool_name(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_")
        return normalized or "tool"

    async def _connection_worker(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        from mcp.shared.memory import create_connected_server_and_client_session

        logger = logging.getLogger("mcp.server.lowlevel.server")
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            async with AsyncExitStack() as stack:
                clients = {"wangp": await stack.enter_async_context(create_connected_server_and_client_session(self._server._mcp_server))}
                for server_name, config in self._external_servers.items():
                    transport = config["transport"]
                    if transport == "stdio":
                        env = None
                        if isinstance(config.get("env"), dict):
                            env = {**os.environ, **{str(key): str(value) for key, value in config["env"].items()}}
                        streams = await stack.enter_async_context(stdio_client(StdioServerParameters(command=str(config["command"]), args=[str(arg) for arg in list(config.get("args", []) or [])], env=env, cwd=config.get("cwd"))))
                    elif transport == "sse":
                        streams = await stack.enter_async_context(sse_client(str(config["url"]), headers=dict(config.get("headers", {}) or {}), timeout=float(config.get("timeout_seconds", 30)), sse_read_timeout=float(config.get("read_timeout_seconds", 300))))
                    else:
                        streams = await stack.enter_async_context(streamablehttp_client(str(config["url"]), headers={str(key): str(value) for key, value in dict(config.get("headers", {}) or {}).items()}, timeout=float(config.get("timeout_seconds", 30)), sse_read_timeout=float(config.get("read_timeout_seconds", 300))))
                    client = await stack.enter_async_context(ClientSession(streams[0], streams[1], read_timeout_seconds=timedelta(seconds=float(config.get("read_timeout_seconds", 300)))))
                    await client.initialize()
                    clients[server_name] = client

                self._tool_defs = []
                self._tool_routes = {}
                self._resource_defs = []
                self._server_instructions = ""
                prompts = await clients["wangp"].list_prompts()
                if any(str(prompt.name) == "wangp_agent" for prompt in prompts.prompts):
                    prompt_result = await clients["wangp"].get_prompt("wangp_agent")
                    self._server_instructions = "\n\n".join(str(getattr(message.content, "text", "") or "").strip() for message in prompt_result.messages if getattr(message.content, "type", "") == "text").strip()
                for server_name, client in clients.items():
                    cursor = None
                    try:
                        while True:
                            listed_resources = await client.list_resources(cursor=cursor)
                            for resource in listed_resources.resources:
                                self._resource_defs.append({
                                    "server": server_name,
                                    "uri": str(resource.uri),
                                    "name": str(getattr(resource, "name", "") or ""),
                                    "title": str(getattr(resource, "title", "") or ""),
                                    "description": str(getattr(resource, "description", "") or ""),
                                    "mime_type": str(getattr(resource, "mimeType", "") or ""),
                                })
                            cursor = getattr(listed_resources, "nextCursor", None)
                            if not cursor:
                                break
                    except Exception:
                        pass
                    listed_tools = await client.list_tools()
                    for tool in listed_tools.tools:
                        exposed_name = str(tool.name) if server_name == "wangp" else f"mcp_{self._safe_tool_name(server_name)}_{self._safe_tool_name(tool.name)}"
                        if exposed_name in self._tool_routes:
                            raise RuntimeError(f"Duplicate Deepy Prime MCP tool name: {exposed_name}")
                        self._tool_routes[exposed_name] = (server_name, str(tool.name))
                        description = str(tool.description or "")
                        if server_name != "wangp":
                            description = f"External MCP server '{server_name}'. {description}".strip()
                        self._tool_defs.append({"type": "function", "function": {"name": exposed_name, "description": description, "parameters": dict(tool.inputSchema or {"type": "object", "properties": {}})}})
                self._tool_defs.extend([
                    {"type": "function", "function": {"name": "mcp_list_resources", "description": "List documentation and other resources exposed by connected MCP servers.", "parameters": {"type": "object", "properties": {"server": {"type": "string", "description": "Optional MCP server name such as wangp."}}}}},
                    {"type": "function", "function": {"name": "mcp_read_resource", "description": "Read one MCP resource by the exact server and URI returned by mcp_list_resources or documented by the trusted WanGP agent guide.", "parameters": {"type": "object", "properties": {"server": {"type": "string", "description": "MCP server name, normally wangp."}, "uri": {"type": "string", "description": "Exact resource URI."}}, "required": ["server", "uri"]}}},
                ])
                self._ready_event.set()

                while True:
                    request = await anyio.to_thread.run_sync(self._request_queue.get)
                    if request is None:
                        break
                    tool_name, arguments, future = request
                    try:
                        if tool_name == "mcp_list_resources":
                            requested_server = str(arguments.get("server", "") or "").strip()
                            resources = [resource for resource in self._resource_defs if not requested_server or resource["server"] == requested_server]
                            future.set_result({"status": "done", "resources": resources, "count": len(resources)})
                            continue
                        if tool_name == "mcp_read_resource":
                            server_name = str(arguments.get("server", "") or "").strip()
                            uri = str(arguments.get("uri", "") or "").strip()
                            if server_name not in clients:
                                raise ValueError(f"Unknown MCP server: {server_name}")
                            if not any(resource["server"] == server_name and resource["uri"] == uri for resource in self._resource_defs):
                                raise ValueError(f"Unknown MCP resource for server '{server_name}': {uri}")
                            resource_result = await clients[server_name].read_resource(uri)
                            contents = []
                            for content in resource_result.contents:
                                item = {"uri": str(content.uri), "mime_type": str(content.mimeType or "")}
                                if hasattr(content, "text"):
                                    item["text"] = str(content.text)
                                elif hasattr(content, "blob"):
                                    item["blob"] = str(content.blob)
                                contents.append(item)
                            future.set_result({"status": "done", "server": server_name, "uri": uri, "contents": contents})
                            continue
                        server_name, original_name = self._tool_routes[tool_name]
                        future.set_result(await clients[server_name].call_tool(original_name, arguments))
                    except BaseException as exc:
                        future.set_exception(exc)
        except BaseException as exc:
            self._worker_error = exc
            self._ready_event.set()
            raise
        finally:
            logger.setLevel(previous_level)

    def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        future = Future()
        self._request_queue.put((tool_name, dict(arguments or {}), future))
        result = future.result()
        if isinstance(result, dict):
            return self._enforce_output_budget(tool_name, result)
        content_text = "\n".join(str(getattr(item, "text", "") or "") for item in result.content if getattr(item, "type", "") == "text").strip()
        if result.isError:
            return {"status": "error", "tool": tool_name, "error": content_text or f"MCP tool '{tool_name}' failed."}
        payload = result.structuredContent
        if payload is None and content_text:
            try:
                payload = json.loads(content_text)
            except json.JSONDecodeError:
                payload = content_text
        if isinstance(payload, dict):
            normalized = dict(payload)
            normalized.setdefault("status", "done")
        else:
            normalized = {"status": "done", "content": payload}
        return self._enforce_output_budget(tool_name, normalized)

    @staticmethod
    def _enforce_output_budget(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
        max_chars = max(8000, min(normalize_deepy_context_tokens(get_deepy_config_value(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT)), 100000))
        if len(serialized) <= max_chars:
            return result
        return {
            "status": "error",
            "tool": tool_name,
            "error": f"MCP tool result exceeded Deepy's context-safe output budget ({len(serialized):,} > {max_chars:,} characters).",
            "hint": "Use query, wildcard filters, pagination, or a more specific item lookup before retrying.",
        }

    def _is_interrupted(self) -> bool:
        return bool(self.assistant_session.interrupt_requested)

    def _update_tool_progress(self, status: str, status_text: str, result: dict[str, Any]) -> None:
        if callable(self._tool_progress_callback):
            self._tool_progress_callback(status=status, status_text=status_text, result=result)

    def _wait_for_generation(self, initial: dict[str, Any]) -> dict[str, Any]:
        job_id = str(initial.get("job_id", "") or "").strip()
        if not job_id:
            return initial
        self._active_job_id = job_id
        self._active_job_cancel_requested = False
        queue_triggered = False
        cancel_requested = False
        snapshot = initial
        try:
            while True:
                if snapshot.get("webui_submission_ready") and not queue_triggered:
                    self.send_cmd("load_queue_trigger", {"job_id": job_id, "token": snapshot.get("webui_load_queue_token", "")})
                    queue_triggered = True
                    self._update_tool_progress("running", "Running", {"status": "running", "job_id": job_id})
                if snapshot.get("done"):
                    result = snapshot.get("result")
                    if isinstance(result, dict):
                        snapshot["status"] = "done" if result.get("success") else ("interrupted" if result.get("cancelled") else "error")
                        generated_files = list(result.get("generated_files", []) or [])
                        if generated_files:
                            snapshot["output_file"] = str(generated_files[-1])
                    return snapshot
                if self._is_interrupted() and not cancel_requested:
                    if not self._active_job_cancel_requested:
                        snapshot = self._call_mcp_tool("wangp_cancel_job", {"job_id": job_id})
                        self._active_job_cancel_requested = True
                    cancel_requested = True
                    self._update_tool_progress("running", "Stopping generation", {"status": "running", "job_id": job_id})
                    continue
                time.sleep(self._POLL_INTERVAL_SECONDS)
                snapshot = self._call_mcp_tool("wangp_get_job", {"job_id": job_id})
        finally:
            self._active_job_id = ""
            self._active_job_cancel_requested = False

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._tool_defs)

    def get_system_instructions(self) -> str:
        return str(self._server_instructions or "").strip()

    def get_system_context(self) -> str:
        from shared.deepy import ui_settings as deepy_ui_settings

        live_settings = self.assistant_session.tool_ui_settings
        settings = deepy_ui_settings.normalize_assistant_tool_ui_settings(**live_settings) if isinstance(live_settings, dict) and live_settings else deepy_ui_settings.get_persisted_assistant_tool_ui_settings()
        lines = [
            "Deepy Settings / General Properties determine whether dimensions, frame count or audio duration, and seed come from templates or from standing defaults.",
            f"Use properties defined in template settings files: {bool(settings['use_template_properties'])}.",
        ]
        if settings["use_template_properties"]:
            lines.append("When using a Deepy template, retain its width, height, frame count, audio duration, and seed unless the user asks for an override.")
        else:
            lines.extend([
                f"Default width: {int(settings['width'])}.",
                f"Default height: {int(settings['height'])}.",
                f"Default video frame count: {int(settings['num_frames'])}.",
                f"Default audio duration: {int(settings['audio_duration'])} seconds.",
                f"Default seed: {int(settings['seed'])}.",
            ])
            lines.append("Override compatible template/model defaults with the width, height, frame count, audio duration, and seed above when the user does not specify them.")
        return "\n".join(lines)

    def get_tool_display_name(self, tool_name: str) -> str:
        normalized_name = str(tool_name or "").strip()
        if normalized_name.startswith("wangp_"):
            normalized_name = normalized_name[len("wangp_"):]
        return normalized_name.replace("_", " ").strip().title()

    def get_tool_transcript_label(self, tool_name: str) -> str:
        return self.get_tool_display_name(tool_name)

    def get_tool_template_filename(self, tool_name: str) -> str:
        return ""

    def get_tool_variant(self, tool_name: str) -> str:
        return ""

    def get_tool_policy(self, tool_name: str) -> dict[str, Any]:
        return {"pause_runtime": tool_name in {"wangp_generate", "wangp_postprocess", "wangp_toolbox"}, "pause_reason": "tool"}

    def validate_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        schema = self._tool_defs_by_name.get(str(tool_name or "").strip())
        if schema is None:
            return f"Unknown MCP tool: {tool_name}"
        parameters = schema["function"].get("parameters", {})
        for parameter_name in parameters.get("required", []) or []:
            if parameter_name not in arguments or arguments[parameter_name] is None:
                return f"{parameter_name} is required."
        return ""

    def infer_tool_calls(self, raw_text: str) -> list[dict[str, Any]]:
        from shared.deepy.engine import DeepyZeroTools

        return DeepyZeroTools.infer_tool_calls(self, raw_text)

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name in {"wangp_generate", "wangp_postprocess"}:
            call_arguments = dict(arguments or {})
            if tool_name == "wangp_generate":
                call_arguments["wait"] = False
            initial = self._call_mcp_tool(tool_name, call_arguments)
            return self._wait_for_generation(initial) if initial.get("job_id") else initial
        return self._call_mcp_tool(tool_name, arguments)

    def _get_selected_media_record_from_source(self, source: str, requested_media_type: str = "all"):
        return None, None
