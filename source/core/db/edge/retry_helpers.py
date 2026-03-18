"""Shared retry primitives for edge-function calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeRetryOptions:
    fallback_url: str | None = None


def _build_retry_options(existing, legacy_options: dict | None):
    if legacy_options:
        unknown = sorted(legacy_options)
        raise TypeError(f"Unexpected retry option(s): {unknown}")
    return existing or EdgeRetryOptions()


def _request_with_fallback(
    *,
    edge_url: str,
    payload,
    headers,
    method: str = "POST",
    timeout_seconds: int,
    options: EdgeRetryOptions,
    function_name: str,
    send_request,
):
    response = send_request(
        edge_url=edge_url,
        payload=payload,
        headers=headers,
        method=method,
        timeout_seconds=timeout_seconds,
    )
    if response.status_code == 404 and options.fallback_url:
        return send_request(
            edge_url=options.fallback_url,
            payload=payload,
            headers=headers,
            method=method,
            timeout_seconds=timeout_seconds,
        )
    return response


def _log_and_sleep_retry(
    *,
    function_name: str,
    context_suffix: str,
    attempt: int,
    max_retries: int,
    reason: str,
    sleep_fn,
) -> None:
    _ = function_name, context_suffix, max_retries, reason
    sleep_fn(2 ** attempt)


__all__ = [
    "EdgeRetryOptions",
    "_build_retry_options",
    "_log_and_sleep_retry",
    "_request_with_fallback",
]
