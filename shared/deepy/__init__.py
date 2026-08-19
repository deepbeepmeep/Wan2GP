from __future__ import annotations

from pathlib import Path

from . import debug_bootstrap as _debug_bootstrap


_debug_bootstrap.bootstrap_deepy_debug()


_DEEPY_DIR = Path(__file__).resolve().parent
ZERO_SYSTEM_PROMPT_PATH = _DEEPY_DIR / "zero_system_prompt.txt"
PRIME_SYSTEM_PROMPT_PATH = _DEEPY_DIR / "prime_system_prompt.txt"
DEFAULT_COMPACTION_PROMPT_PATH = _DEEPY_DIR / "default_compaction_prompt.txt"


def load_zero_system_prompt() -> str:
    try:
        return ZERO_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Deepy Zero system prompt file not found: {ZERO_SYSTEM_PROMPT_PATH}") from exc


ZERO_SYSTEM_PROMPT = load_zero_system_prompt()


def load_prime_system_prompt() -> str:
    try:
        return PRIME_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Deepy Prime system prompt file not found: {PRIME_SYSTEM_PROMPT_PATH}") from exc


PRIME_SYSTEM_PROMPT = load_prime_system_prompt()


def load_default_compaction_prompt() -> str:
    try:
        return DEFAULT_COMPACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Deepy default compaction prompt file not found: {DEFAULT_COMPACTION_PROMPT_PATH}") from exc


DEFAULT_COMPACTION_PROMPT = load_default_compaction_prompt()


def __getattr__(name: str):
    if name in {"DEBUG_DEEPY_ENABLED", "DEBUG_DEEPY_LOG_PATH"}:
        return getattr(_debug_bootstrap, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEBUG_DEEPY_ENABLED",
    "DEBUG_DEEPY_LOG_PATH",
    "DEFAULT_COMPACTION_PROMPT",
    "DEFAULT_COMPACTION_PROMPT_PATH",
    "ZERO_SYSTEM_PROMPT",
    "ZERO_SYSTEM_PROMPT_PATH",
    "PRIME_SYSTEM_PROMPT",
    "PRIME_SYSTEM_PROMPT_PATH",
    "load_default_compaction_prompt",
    "load_zero_system_prompt",
    "load_prime_system_prompt",
]
