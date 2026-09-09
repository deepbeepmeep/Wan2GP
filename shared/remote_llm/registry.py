from __future__ import annotations

from typing import Any

from .claude_backend import ClaudeBackend
from .codex_backend import CodexBackend
from .config import ENGINE_CLAUDE, ENGINE_CODEX, ENGINE_OPENCODE, normalize_llm_config
from .opencode_backend import OpenCodeBackend


def create_backend(engine: str, server_config: dict[str, Any], *, toolbox: Any | None = None):
    config = normalize_llm_config(server_config)
    profile = config["profiles"].get(engine, {})
    if engine == ENGINE_CODEX:
        return CodexBackend(profile)
    if engine == ENGINE_CLAUDE:
        return ClaudeBackend(profile)
    if engine == ENGINE_OPENCODE:
        return OpenCodeBackend(profile, toolbox=toolbox)
    raise ValueError(f"Unsupported external LLM engine: {engine}")

