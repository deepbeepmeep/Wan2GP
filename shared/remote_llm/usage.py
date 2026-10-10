from __future__ import annotations

from typing import Any, Mapping


_COUNT_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens")
SHOW_REMOTE_LLM_CUMULATIVE_USAGE = False


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def codex_token_breakdown(value: Mapping[str, Any] | None) -> dict[str, int]:
    value = value if isinstance(value, Mapping) else {}
    return {
        "input_tokens": _count(value.get("inputTokens")),
        "cached_input_tokens": _count(value.get("cachedInputTokens")),
        "cache_write_input_tokens": _count(value.get("cacheWriteInputTokens")),
        "output_tokens": _count(value.get("outputTokens")),
        "reasoning_output_tokens": _count(value.get("reasoningOutputTokens")),
        "total_tokens": _count(value.get("totalTokens")),
    }


def codex_usage_data(token_usage: Mapping[str, Any] | None, turn_start: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    token_usage = token_usage if isinstance(token_usage, Mapping) else {}
    total = codex_token_breakdown(token_usage.get("total"))
    start = turn_start if isinstance(turn_start, Mapping) else {}
    turn = {key: max(0, total[key] - _count(start.get(key))) for key in _COUNT_KEYS}
    last = codex_token_breakdown(token_usage.get("last"))
    data: dict[str, Any] = dict(turn)
    data["context_tokens"] = last["total_tokens"]
    data["context_window"] = _count(token_usage.get("modelContextWindow"))
    return data, total


def claude_usage_data(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    usage = usage if isinstance(usage, Mapping) else {}
    output_details = usage.get("output_tokens_details", {})
    output_details = output_details if isinstance(output_details, Mapping) else {}
    uncached_input = _count(usage.get("input_tokens"))
    cached_input = _count(usage.get("cache_read_input_tokens"))
    cache_write_input = _count(usage.get("cache_creation_input_tokens"))
    output = _count(usage.get("output_tokens"))
    input_total = uncached_input + cached_input + cache_write_input
    if input_total <= 0 and output <= 0:
        return {}
    return {
        "input_tokens": input_total,
        "cached_input_tokens": cached_input,
        "cache_write_input_tokens": cache_write_input,
        "output_tokens": output,
        "reasoning_output_tokens": _count(usage.get("reasoning_output_tokens", output_details.get("thinking_tokens"))),
        "total_tokens": input_total + output,
        "context_tokens": 0,
        "context_window": 0,
    }


def claude_context_window(model: str) -> int:
    model = str(model or "").strip().lower().replace(".", "-")
    if not model or model == "default":
        return 0
    if "[1m]" in model or any(marker in model for marker in ("fable-5", "mythos-5", "mythos-preview", "opus-5", "opus-4-8", "opus-4-7", "opus-4-6", "sonnet-5", "sonnet-4-6")):
        return 1_000_000
    return 200_000


def aggregate_usage_data(usages: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    return {key: sum(_count(usage.get(key)) for usage in usages.values()) for key in _COUNT_KEYS}


def opencode_usage_data(response: Mapping[str, Any] | None) -> dict[str, Any]:
    response = response if isinstance(response, Mapping) else {}
    info = response.get("info", response)
    info = info if isinstance(info, Mapping) else {}
    tokens = info.get("tokens", {})
    tokens = tokens if isinstance(tokens, Mapping) else {}
    cache = tokens.get("cache", {})
    cache = cache if isinstance(cache, Mapping) else {}
    uncached_input = _count(tokens.get("input"))
    cached_input = _count(cache.get("read"))
    cache_write_input = _count(cache.get("write"))
    output = _count(tokens.get("output"))
    input_total = uncached_input + cached_input + cache_write_input
    if input_total <= 0 and output <= 0:
        return {}
    return {
        "input_tokens": input_total,
        "cached_input_tokens": cached_input,
        "cache_write_input_tokens": cache_write_input,
        "output_tokens": output,
        "reasoning_output_tokens": _count(tokens.get("reasoning")),
        "total_tokens": input_total + output,
        "context_tokens": 0,
        "context_window": 0,
    }


def build_remote_usage_stats(data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    data = data if isinstance(data, Mapping) else {}
    input_tokens = _count(data.get("input_tokens"))
    cached_input = _count(data.get("cached_input_tokens"))
    cache_write_input = _count(data.get("cache_write_input_tokens"))
    output_tokens = _count(data.get("output_tokens"))
    reasoning_output = _count(data.get("reasoning_output_tokens"))
    total_tokens = _count(data.get("total_tokens")) or input_tokens + output_tokens
    context_tokens = _count(data.get("context_tokens"))
    context_window = _count(data.get("context_window"))
    if input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return None
    parts = []
    if SHOW_REMOTE_LLM_CUMULATIVE_USAGE:
        input_detail = []
        if cached_input:
            input_detail.append(f"{cached_input:,} cached")
        if cache_write_input:
            input_detail.append(f"{cache_write_input:,} cache write")
        input_label = f"in {input_tokens:,}"
        if input_detail:
            input_label += f" ({', '.join(input_detail)})"
        output_label = f"out {output_tokens:,}"
        if reasoning_output:
            output_label += f" ({reasoning_output:,} reasoning)"
        parts.extend((input_label, output_label, f"turn {total_tokens:,} tk"))
    if context_window:
        parts.append(f"context {context_tokens:,} / {context_window:,} tk")
    if not parts:
        return None
    return {
        "visible": True,
        "text": " | ".join(parts),
        **{key: _count(data.get(key)) for key in (*_COUNT_KEYS, "context_tokens", "context_window")},
    }
