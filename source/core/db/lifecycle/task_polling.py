"""Compatibility shim for the split lifecycle task-polling module."""

from source.core.db import task_polling as _task_polling
from source.core.db.task_polling import *  # noqa: F401,F403

resolve_edge_function_url = _task_polling.resolve_edge_function_url
resolve_edge_auth_token = _task_polling.resolve_edge_auth_token
allow_direct_query_fallback = _task_polling.allow_direct_query_fallback
_call_edge_function_with_retry = _task_polling._call_edge_function_with_retry
query_task_status = _task_polling.query_task_status


def _sync_root_contracts():
    _task_polling.resolve_edge_function_url = resolve_edge_function_url
    _task_polling.resolve_edge_auth_token = resolve_edge_auth_token
    _task_polling.allow_direct_query_fallback = allow_direct_query_fallback
    _task_polling._call_edge_function_with_retry = _call_edge_function_with_retry
    _task_polling.query_task_status = query_task_status


def get_task_params(*args, **kwargs):
    originals = (
        _task_polling.resolve_edge_function_url,
        _task_polling.resolve_edge_auth_token,
        _task_polling.allow_direct_query_fallback,
        _task_polling._call_edge_function_with_retry,
    )
    try:
        _sync_root_contracts()
        return _task_polling.get_task_params(*args, **kwargs)
    finally:
        (
            _task_polling.resolve_edge_function_url,
            _task_polling.resolve_edge_auth_token,
            _task_polling.allow_direct_query_fallback,
            _task_polling._call_edge_function_with_retry,
        ) = originals


def get_task_params_result(*args, **kwargs):
    originals = (
        _task_polling.resolve_edge_function_url,
        _task_polling.resolve_edge_auth_token,
        _task_polling.allow_direct_query_fallback,
        _task_polling._call_edge_function_with_retry,
    )
    try:
        _sync_root_contracts()
        return _task_polling.get_task_params_result(*args, **kwargs)
    finally:
        (
            _task_polling.resolve_edge_function_url,
            _task_polling.resolve_edge_auth_token,
            _task_polling.allow_direct_query_fallback,
            _task_polling._call_edge_function_with_retry,
        ) = originals


def get_task_output_location_from_db_result(*args, **kwargs):
    original = _task_polling.resolve_edge_function_url
    try:
        _task_polling.resolve_edge_function_url = resolve_edge_function_url
        return _task_polling.get_task_output_location_from_db_result(*args, **kwargs)
    finally:
        _task_polling.resolve_edge_function_url = original


def poll_task_status_result(*args, **kwargs):
    original = _task_polling.query_task_status
    try:
        _task_polling.query_task_status = query_task_status
        return _task_polling.poll_task_status_result(*args, **kwargs)
    finally:
        _task_polling.query_task_status = original
