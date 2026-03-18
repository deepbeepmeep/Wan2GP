"""
Database configuration: globals, constants, and debug helpers.

All module-level state that other db submodules depend on lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

__all__ = [
    "PG_TABLE_NAME",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_VIDEO_BUCKET",
    "SUPABASE_CLIENT",
    "SUPABASE_EDGE_COMPLETE_TASK_URL",
    "SUPABASE_ACCESS_TOKEN",
    "SUPABASE_EDGE_CREATE_TASK_URL",
    "SUPABASE_EDGE_CLAIM_TASK_URL",
    "STATUS_QUEUED",
    "STATUS_IN_PROGRESS",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "debug_mode",
    "EDGE_FAIL_PREFIX",
    "RETRYABLE_STATUS_CODES",
    "DBRuntimeConfig",
    "DBRuntimeContractError",
    "initialize_db_runtime",
    "get_db_runtime_registry",
    "get_db_runtime_config",
    "resolve_edge_function_url",
    "resolve_edge_auth_token",
    "build_edge_headers",
    "has_required_edge_credentials",
    "allow_direct_query_fallback",
    "resolve_edge_request",
    "validate_config",
]

# Import centralized logger for system_logs visibility
try:
    from ...core.log import headless_logger
except ImportError:
    # Fallback if core.log not available
    headless_logger = None

try:
    from supabase import Client as SupabaseClient
except ImportError:
    SupabaseClient = None


# -----------------------------------------------------------------------------
# Global DB Configuration (will be set by worker.py)
# -----------------------------------------------------------------------------
PG_TABLE_NAME = "tasks"
SUPABASE_URL = None
SUPABASE_SERVICE_KEY = None
SUPABASE_VIDEO_BUCKET = "image_uploads"
SUPABASE_CLIENT: SupabaseClient | None = None
SUPABASE_EDGE_COMPLETE_TASK_URL: str | None = None  # Optional override for edge function
SUPABASE_ACCESS_TOKEN: str | None = None # Will be set by worker.py
SUPABASE_EDGE_CREATE_TASK_URL: str | None = None # Will be set by worker.py
SUPABASE_EDGE_CLAIM_TASK_URL: str | None = None # Will be set by worker.py

# -----------------------------------------------------------------------------
# Status Constants
# -----------------------------------------------------------------------------
STATUS_QUEUED = "Queued"
STATUS_IN_PROGRESS = "In Progress"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"
# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers
# -----------------------------------------------------------------------------
debug_mode = False


class DBRuntimeContractError(RuntimeError):
    """Raised when a required DB runtime contract is unavailable."""


@dataclass(frozen=True)
class DBRuntimeConfig:
    """Immutable snapshot of the configured DB runtime."""

    db_type: str
    pg_table_name: str
    supabase_url: str | None
    supabase_service_key: str | None
    supabase_video_bucket: str | None
    supabase_client: object | None
    supabase_access_token: str | None
    supabase_edge_complete_task_url: str | None
    supabase_edge_create_task_url: str | None
    supabase_edge_claim_task_url: str | None
    debug_mode: bool = False


class _DBRuntimeRegistry:
    def __init__(self) -> None:
        self._runtime: DBRuntimeConfig | None = None

    def set_runtime(self, runtime: DBRuntimeConfig) -> DBRuntimeConfig:
        self._runtime = runtime
        return runtime

    def get_runtime(self) -> DBRuntimeConfig | None:
        return self._runtime


_RUNTIME_REGISTRY = _DBRuntimeRegistry()

def _log_thumbnail(msg: str, level: str = "debug", task_id: str = None):
    """Log thumbnail-related messages via the centralized logger."""
    full_msg = f"[THUMBNAIL] {msg}"
    if headless_logger:
        if level == "info":
            headless_logger.info(full_msg, task_id=task_id)
        elif level == "warning":
            headless_logger.warning(full_msg, task_id=task_id)
        else:
            headless_logger.debug(full_msg, task_id=task_id)

# -----------------------------------------------------------------------------
# Edge function error prefix (used by debug.py to detect edge failures)
# -----------------------------------------------------------------------------
EDGE_FAIL_PREFIX = "[EDGE_FAIL"  # Used by debug.py to detect edge failures

RETRYABLE_STATUS_CODES = {500, 502, 503, 504}  # 500 included for transient edge function crashes (CDN issues, cold starts)


def initialize_db_runtime(
    *,
    db_type: str,
    pg_table_name: str,
    supabase_url: str | None,
    supabase_service_key: str | None,
    supabase_video_bucket: str | None,
    supabase_client: object | None,
    supabase_access_token: str | None,
    supabase_edge_complete_task_url: str | None = None,
    supabase_edge_create_task_url: str | None = None,
    supabase_edge_claim_task_url: str | None = None,
    debug: bool = False,
) -> DBRuntimeConfig:
    """Persist runtime config both as globals and as an immutable snapshot."""
    global PG_TABLE_NAME
    global SUPABASE_URL
    global SUPABASE_SERVICE_KEY
    global SUPABASE_VIDEO_BUCKET
    global SUPABASE_CLIENT
    global SUPABASE_EDGE_COMPLETE_TASK_URL
    global SUPABASE_ACCESS_TOKEN
    global SUPABASE_EDGE_CREATE_TASK_URL
    global SUPABASE_EDGE_CLAIM_TASK_URL
    global debug_mode

    PG_TABLE_NAME = pg_table_name
    SUPABASE_URL = supabase_url
    SUPABASE_SERVICE_KEY = supabase_service_key
    SUPABASE_VIDEO_BUCKET = supabase_video_bucket or "image_uploads"
    SUPABASE_CLIENT = supabase_client
    SUPABASE_EDGE_COMPLETE_TASK_URL = supabase_edge_complete_task_url
    SUPABASE_ACCESS_TOKEN = supabase_access_token
    SUPABASE_EDGE_CREATE_TASK_URL = supabase_edge_create_task_url
    SUPABASE_EDGE_CLAIM_TASK_URL = supabase_edge_claim_task_url
    debug_mode = debug

    runtime = DBRuntimeConfig(
        db_type=db_type,
        pg_table_name=pg_table_name,
        supabase_url=supabase_url,
        supabase_service_key=supabase_service_key,
        supabase_video_bucket=SUPABASE_VIDEO_BUCKET,
        supabase_client=supabase_client,
        supabase_access_token=supabase_access_token,
        supabase_edge_complete_task_url=supabase_edge_complete_task_url,
        supabase_edge_create_task_url=supabase_edge_create_task_url,
        supabase_edge_claim_task_url=supabase_edge_claim_task_url,
        debug_mode=debug,
    )
    return _RUNTIME_REGISTRY.set_runtime(runtime)


def get_db_runtime_registry() -> _DBRuntimeRegistry:
    return _RUNTIME_REGISTRY


def get_db_runtime_config(*, require_initialized: bool = False) -> DBRuntimeConfig | None:
    runtime = _RUNTIME_REGISTRY.get_runtime()
    if runtime is not None:
        return runtime

    inferred = DBRuntimeConfig(
        db_type="supabase",
        pg_table_name=PG_TABLE_NAME,
        supabase_url=SUPABASE_URL,
        supabase_service_key=SUPABASE_SERVICE_KEY,
        supabase_video_bucket=SUPABASE_VIDEO_BUCKET,
        supabase_client=SUPABASE_CLIENT,
        supabase_access_token=SUPABASE_ACCESS_TOKEN,
        supabase_edge_complete_task_url=SUPABASE_EDGE_COMPLETE_TASK_URL,
        supabase_edge_create_task_url=SUPABASE_EDGE_CREATE_TASK_URL,
        supabase_edge_claim_task_url=SUPABASE_EDGE_CLAIM_TASK_URL,
        debug_mode=debug_mode,
    )
    if require_initialized and inferred.supabase_url is None and inferred.supabase_client is None:
        raise DBRuntimeContractError("DB runtime has not been initialized")
    return inferred


def resolve_edge_function_url(
    function_name: str,
    *,
    runtime_config: DBRuntimeConfig | None = None,
) -> str | None:
    runtime = runtime_config or get_db_runtime_config()
    if runtime is None or not runtime.supabase_url:
        return None

    explicit = {
        "complete_task": runtime.supabase_edge_complete_task_url,
        "complete-task": runtime.supabase_edge_complete_task_url,
        "create-task": runtime.supabase_edge_create_task_url,
        "claim-next-task": runtime.supabase_edge_claim_task_url,
    }.get(function_name)
    if explicit:
        return explicit
    return f"{runtime.supabase_url.rstrip('/')}/functions/v1/{function_name}"


def resolve_edge_auth_token(
    *,
    scope: str = "worker",
    runtime_config: DBRuntimeConfig | None = None,
    require_token: bool = False,
    allow_worker_service_fallback: bool = False,
) -> str | None:
    runtime = runtime_config or get_db_runtime_config()
    if runtime is None:
        if require_token:
            raise DBRuntimeContractError("DB runtime is unavailable")
        return None

    if scope == "service":
        token = runtime.supabase_service_key
    else:
        token = runtime.supabase_access_token
        if token is None and allow_worker_service_fallback:
            token = runtime.supabase_service_key

    if require_token and not token:
        raise DBRuntimeContractError(f"No edge auth token available for scope={scope}")
    return token


def build_edge_headers(token: str | None, *, include_apikey: bool = True) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        if include_apikey:
            headers["apikey"] = token
    return headers


def has_required_edge_credentials(headers: dict[str, str] | None) -> bool:
    if not headers:
        return False
    lowered = {str(key).lower(): value for key, value in headers.items()}
    return bool(lowered.get("authorization"))


def allow_direct_query_fallback() -> bool:
    import os

    value = (os.getenv("SUPABASE_ALLOW_DIRECT_QUERY_FALLBACK") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolve_edge_request(
    function_name: str,
    *,
    runtime_config: DBRuntimeConfig | None = None,
    scope: str = "worker",
    include_apikey: bool = True,
    require_token: bool = False,
):
    token = resolve_edge_auth_token(
        scope=scope,
        runtime_config=runtime_config,
        require_token=require_token,
    )
    return SimpleNamespace(
        url=resolve_edge_function_url(function_name, runtime_config=runtime_config),
        headers=build_edge_headers(token, include_apikey=include_apikey),
    )


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------

def validate_config(*, runtime_config: DBRuntimeConfig | None = None) -> list[str]:
    """Validate that required config fields are set after worker initialization.

    Returns a list of error messages (empty if valid).
    """
    runtime = runtime_config or get_db_runtime_config()
    errors: list[str] = []

    if runtime is None:
        return ["DB runtime is not initialized"]

    if not runtime.supabase_url:
        errors.append("SUPABASE_URL is not set")
    elif not runtime.supabase_url.startswith("http"):
        errors.append(f"SUPABASE_URL does not look like a URL: {runtime.supabase_url!r}")

    if not runtime.supabase_service_key:
        errors.append("SUPABASE_SERVICE_KEY is not set")

    if runtime.supabase_client is None:
        errors.append("SUPABASE_CLIENT is not initialized")

    if not runtime.supabase_access_token:
        errors.append("SUPABASE_ACCESS_TOKEN is not set")

    return errors
