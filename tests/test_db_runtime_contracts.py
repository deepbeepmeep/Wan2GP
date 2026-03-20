"""Contract tests for DB runtime registry and error semantics."""

from __future__ import annotations

from typing import Any

from postgrest.exceptions import APIError
import pytest

from debug.runtime_queries_system import query_system_health
from source.core.db import config as db_config
from source.core.db import task_claim, task_polling


def _runtime_config(*, supabase_client: Any) -> db_config.DBRuntimeConfig:
    return db_config.DBRuntimeConfig(
        db_type="supabase",
        pg_table_name="tasks",
        supabase_url="https://example.supabase.co",
        supabase_service_key="service-key",
        supabase_video_bucket="image_uploads",
        supabase_client=supabase_client,
        supabase_access_token="access-token",
        supabase_edge_complete_task_url="https://edge/complete",
        supabase_edge_create_task_url="https://edge/create",
        supabase_edge_claim_task_url="https://edge/claim",
        debug_mode=False,
    )


def test_initialize_db_runtime_registers_runtime_snapshot() -> None:
    sentinel_client = object()
    initialized = db_config.initialize_db_runtime(
        db_type="supabase",
        pg_table_name="tasks",
        supabase_url="https://example.supabase.co",
        supabase_service_key="service-key",
        supabase_video_bucket="image_uploads",
        supabase_client=sentinel_client,
        supabase_access_token="access-token",
        supabase_edge_complete_task_url="https://edge/complete",
        supabase_edge_create_task_url="https://edge/create",
        supabase_edge_claim_task_url="https://edge/claim",
        debug=True,
    )

    registry_runtime = db_config.get_db_runtime_registry().get_runtime()
    assert registry_runtime == initialized
    assert db_config.get_db_runtime_config(require_initialized=True) == initialized


def test_check_task_counts_returns_none_without_access_token() -> None:
    """task counts returns None gracefully when access_token is missing (no exception)."""
    runtime = db_config.DBRuntimeConfig(
        db_type="supabase",
        pg_table_name="tasks",
        supabase_url="https://example.supabase.co",
        supabase_service_key=None,
        supabase_video_bucket="image_uploads",
        supabase_client=None,
        supabase_access_token=None,  # No access token
        supabase_edge_complete_task_url=None,
        supabase_edge_create_task_url=None,
        supabase_edge_claim_task_url=None,
    )
    result = task_claim.check_task_counts_supabase(runtime_config=runtime)
    assert result is None


def test_get_task_params_raises_when_no_supported_query_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_config(supabase_client=None)
    monkeypatch.setattr(task_polling, "resolve_edge_function_url", lambda *_a, **_k: None)
    monkeypatch.setattr(task_polling, "resolve_edge_auth_token", lambda *_a, **_k: None)
    monkeypatch.setattr(task_polling, "allow_direct_query_fallback", lambda: False)

    with pytest.raises(db_config.DBRuntimeContractError):
        task_polling.get_task_params("task-123", runtime_config=runtime)


class _FailingSupabaseQuery:
    def select(self, *_args, **_kwargs):
        return self

    def neq(self, *_args, **_kwargs):
        return self

    def execute(self):
        raise APIError({"message": "boom"})


class _FailingSupabaseClient:
    def table(self, _name: str) -> _FailingSupabaseQuery:
        return _FailingSupabaseQuery()


class _NoopLogClient:
    def get_logs(self, _query: object) -> list[dict[str, Any]]:
        return []


class _LogQuery:
    def __init__(self, **_kwargs: Any) -> None:
        pass


class _DebugContext:
    def __init__(self) -> None:
        self.supabase = _FailingSupabaseClient()
        self.log_client = _NoopLogClient()
        self.log_query_class = _LogQuery

    def _debug(self, _message: str) -> None:
        pass


def test_query_system_health_raises_typed_db_contract_error() -> None:
    with pytest.raises(db_config.DBRuntimeContractError):
        query_system_health(_DebugContext())
