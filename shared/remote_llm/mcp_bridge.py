from __future__ import annotations

import socket
import inspect
import threading
from typing import Any, Callable, Sequence


def build_tool_proxy(tools: Sequence[dict[str, Any]], call_tool: Callable[[str, dict[str, Any]], dict[str, Any]]):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("WanGP Deepy Prime")
    for definition in tools:
        function = dict(definition.get("function", {}) or {})
        name = str(function.get("name", "") or "").strip()
        if not name:
            continue
        schema = dict(function.get("parameters", {}) or {"type": "object", "properties": {}})
        properties = dict(schema.get("properties", {}) or {})
        required = set(schema.get("required", []) or [])

        def proxy(tool_name=name, **kwargs):
            return call_tool(tool_name, kwargs)

        proxy.__name__ = name
        proxy.__signature__ = inspect.Signature([
            inspect.Parameter(str(parameter_name), inspect.Parameter.KEYWORD_ONLY, default=inspect.Parameter.empty if parameter_name in required else None, annotation=Any)
            for parameter_name in properties
            if str(parameter_name).isidentifier()
        ])
        server.add_tool(proxy, name=name, description=str(function.get("description", "") or ""))
        server._tool_manager.get_tool(name).parameters = schema
    return server


class MCPHttpBridge:
    def __init__(self, fastmcp_server: Any) -> None:
        self.fastmcp_server = fastmcp_server
        self._socket: socket.socket | None = None
        self._server = None
        self._thread: threading.Thread | None = None
        self.url = ""

    def start(self) -> str:
        if self.url:
            return self.url
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("OpenCode MCP integration requires uvicorn.") from exc
        app = self.fastmcp_server.streamable_http_app()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = int(sock.getsockname()[1])
        self._socket = sock
        self._server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
        self._thread = threading.Thread(target=lambda: self._server.run(sockets=[sock]), name="wangp-opencode-mcp", daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self.url

    def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._thread = None
        self._socket = None
        self._server = None
        self.url = ""
