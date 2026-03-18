"""Edge request retry helpers."""

from __future__ import annotations

import time

import httpx

from source.core.db.config import RETRYABLE_STATUS_CODES
from source.core.db.edge.retry_helpers import EdgeRetryOptions, _request_with_fallback


def _send_edge_request(
    *,
    edge_url: str,
    payload,
    headers,
    method: str,
    timeout_seconds: int,
):
    normalized_method = method.upper()
    if normalized_method not in {"POST", "PUT"}:
        raise ValueError("Only 'POST' and 'PUT' methods are supported")
    sender = httpx.post if normalized_method == "POST" else httpx.put
    return sender(edge_url, json=payload, headers=headers, timeout=timeout_seconds)


def call_edge_function_with_retry(
    *,
    edge_url: str,
    payload,
    headers,
    function_name: str,
    max_retries: int = 3,
    method: str = "POST",
    timeout_seconds: int = 30,
    retry_on_404_patterns=(),
    fallback_url: str | None = None,
):
    options = EdgeRetryOptions(fallback_url=fallback_url)
    for attempt in range(1, max_retries + 1):
        try:
            response = _request_with_fallback(
                edge_url=edge_url,
                payload=payload,
                headers=headers,
                method=method,
                timeout_seconds=timeout_seconds,
                options=options,
                function_name=function_name,
                send_request=_send_edge_request,
            )
        except httpx.TimeoutException:
            if attempt >= max_retries:
                return None, f"[EDGE_FAIL:{function_name}:TIMEOUT] request timed out"
            time.sleep(1)
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            if attempt >= max_retries:
                return response, f"[EDGE_FAIL:{function_name}:HTTP_{response.status_code}] {response.text}"
            time.sleep(1)
            continue

        if response.status_code == 404 and any(pattern in response.text for pattern in retry_on_404_patterns):
            if attempt >= max_retries:
                return response, f"[EDGE_FAIL:{function_name}:HTTP_404] {response.text}"
            time.sleep(1)
            continue

        if response.status_code >= 400:
            return response, f"[EDGE_FAIL:{function_name}:HTTP_{response.status_code}] {response.text}"
        return response, None

    return None, f"[EDGE_FAIL:{function_name}:UNKNOWN] exhausted retries"
