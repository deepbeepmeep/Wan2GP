"""
Edge function call helpers: retry wrapper and JWT utilities.
"""
import json
import time
import base64
from pathlib import Path

import httpx

from source.core.log import headless_logger

__all__ = [
    "_call_edge_function_with_retry",
    "_get_user_id_from_jwt",
    "_is_jwt_token",
]

from .config import (
    EDGE_FAIL_PREFIX,
    RETRYABLE_STATUS_CODES)

def _call_edge_function_with_retry(
    edge_url: str,
    payload,
    headers: dict,
    function_name: str,
    *,
    context_id: str = "",  # task_id or worker_id for logging
    timeout: int = 30,
    max_retries: int = 3,
    method: str = "POST",
    fallback_url: str | None = None,  # Optional 404 fallback URL
    retry_on_404_patterns: list[str] | None = None,  # Patterns in 404 response body to retry
) -> tuple[httpx.Response | None, str | None]:
    """
    Call a Supabase edge function with retry logic for transient errors.

    Retries on:
    - 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout
    - Network errors (TimeoutException, RequestError)
    - 404 errors if response body matches retry_on_404_patterns (e.g. "Task not found")

    Args:
        edge_url: The edge function URL
        payload: JSON payload (for POST) or None
        headers: HTTP headers
        function_name: Name of function for logging (e.g. "complete_task", "update-task-status")
        context_id: Task/worker ID for logging context
        timeout: Base timeout in seconds (increases on retries)
        max_retries: Maximum retry attempts
        method: HTTP method ("POST" or "PUT")
        fallback_url: Optional fallback URL to try on 404
        retry_on_404_patterns: List of substrings - if 404 response contains any, retry

    Returns:
        Tuple of (response, error_message)
        - On success: (response, None)
        - On failure: (response_or_None, standardized_error_message)

        Error message format for detection:
        "[EDGE_FAIL:{function_name}:{error_type}] {details}"
    """
    ctx = f" for {context_id}" if context_id else ""

    for attempt in range(max_retries):
        try:
            current_timeout = timeout + (attempt * 15)  # Increase timeout on retries

            if method == "PUT":
                # For large uploads, avoid buffering entire files in memory:
                # - If payload is a Path/str to an existing file, open+stream per attempt.
                # - If payload is bytes, send directly.
                # - Otherwise, pass through to httpx (caller responsibility).
                if isinstance(payload, (str, Path)) and Path(payload).exists():
                    with open(Path(payload), "rb") as f:
                        resp = httpx.put(edge_url, content=f, headers=headers, timeout=current_timeout)
                else:
                    resp = httpx.put(edge_url, content=payload, headers=headers, timeout=current_timeout)
            else:
                resp = httpx.post(edge_url, json=payload, headers=headers, timeout=current_timeout)

            # Handle 404 fallback (e.g. hyphen vs underscore naming)
            if resp.status_code == 404 and fallback_url:
                headless_logger.debug_anomaly("RETRY", f"{function_name} returned 404; trying fallback URL: {fallback_url}")
                if method == "PUT":
                    if isinstance(payload, (str, Path)) and Path(payload).exists():
                        with open(Path(payload), "rb") as f:
                            resp = httpx.put(fallback_url, content=f, headers=headers, timeout=current_timeout)
                    else:
                        resp = httpx.put(fallback_url, content=payload, headers=headers, timeout=current_timeout)
                else:
                    resp = httpx.post(fallback_url, json=payload, headers=headers, timeout=current_timeout)

            # Success
            if resp.status_code in [200, 201]:
                return resp, None

            # Retry on specific 404 patterns (e.g. "Task not found" due to replication lag)
            if resp.status_code == 404 and retry_on_404_patterns and attempt < max_retries - 1:
                resp_text = resp.text
                should_retry = any(pattern.lower() in resp_text.lower() for pattern in retry_on_404_patterns)
                if should_retry:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    headless_logger.essential(f"{function_name} got 404 with retryable pattern{ctx} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    headless_logger.debug_anomaly("RETRY", f"404 response: {resp_text[:200]}")
                    time.sleep(wait_time)
                    continue

            # Retryable error (5xx)
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                headless_logger.essential(f"{function_name} got {resp.status_code}{ctx} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Non-retryable error or final attempt
            error_type = "5XX_TRANSIENT" if resp.status_code in RETRYABLE_STATUS_CODES else f"HTTP_{resp.status_code}"
            error_msg = f"{EDGE_FAIL_PREFIX}:{function_name}:{error_type}] {resp.status_code} after {attempt + 1} attempts{ctx}: {resp.text[:200]}"

            if resp.status_code in RETRYABLE_STATUS_CODES:
                headless_logger.error(f"{function_name} failed with {resp.status_code} after {max_retries} attempts{ctx}")

            return resp, error_msg

        except httpx.TimeoutException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                headless_logger.essential(f"{function_name} timeout{ctx} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"{EDGE_FAIL_PREFIX}:{function_name}:TIMEOUT] Timed out after {max_retries} attempts{ctx}: {e}"
                headless_logger.error(error_msg)
                return None, error_msg

        except httpx.RequestError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                headless_logger.essential(f"{function_name} network error{ctx} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"{EDGE_FAIL_PREFIX}:{function_name}:NETWORK] Request failed after {max_retries} attempts{ctx}: {e}"
                headless_logger.error(error_msg)
                return None, error_msg

    # Should not reach here, but safety fallback
    error_msg = f"{EDGE_FAIL_PREFIX}:{function_name}:UNKNOWN] All retries exhausted{ctx}"
    return None, error_msg

# -----------------------------------------------------------------------------
# Internal Helpers for Supabase
# -----------------------------------------------------------------------------

def _get_user_id_from_jwt(jwt_str: str) -> str | None:
    """Decodes a JWT and extracts the 'sub' (user ID) claim without validation."""
    if not jwt_str:
        return None
    try:
        # JWT is composed of header.payload.signature
        _, payload_b64, _ = jwt_str.split('.')
        # The payload is base64 encoded. It needs to be padded to be decoded correctly.
        payload_b64 += '=' * (-len(payload_b64) % 4)
        payload_json = base64.b64decode(payload_b64).decode('utf-8')
        payload = json.loads(payload_json)
        user_id = payload.get('sub')
        headless_logger.debug(f"JWT Decode: Extracted user ID (sub): {user_id}")
        return user_id
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        headless_logger.debug_anomaly("ERROR", f"Could not decode JWT to get user ID: {e}")
        return None

def _is_jwt_token(token_str: str) -> bool:
    """
    Checks if a token string looks like a JWT (has 3 parts separated by dots).
    """
    if not token_str:
        return False
    parts = token_str.split('.')
    return len(parts) == 3
