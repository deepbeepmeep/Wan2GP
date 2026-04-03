from __future__ import annotations

from pathlib import Path

from . import debug_bootstrap as _debug_bootstrap


_debug_bootstrap.bootstrap_deepy_debug()
DEBUG_DEEPY_ENABLED = _debug_bootstrap.DEBUG_DEEPY_ENABLED
DEBUG_DEEPY_LOG_PATH = _debug_bootstrap.DEBUG_DEEPY_LOG_PATH


_DEEPY_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = _DEEPY_DIR / "default_system_prompt.txt"


def load_default_system_prompt() -> str:
    try:
        return DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Deepy default system prompt file not found: {DEFAULT_SYSTEM_PROMPT_PATH}") from exc


DEFAULT_SYSTEM_PROMPT = load_default_system_prompt()


__all__ = [
    "DEBUG_DEEPY_ENABLED",
    "DEBUG_DEEPY_LOG_PATH",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "load_default_system_prompt",
]
