#!/usr/bin/env python3
"""Stage 1 read-only probes for the live worker harness plan."""

from __future__ import annotations

import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class ProbeError(RuntimeError):
    """Raised when a stage-1 probe fails."""


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_environment() -> None:
    _load_env_file(REPO_ROOT / ".env")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    raise ProbeError(f"Missing required environment variable: {name}")


def get_live_test_token() -> str:
    token = os.environ.get("REIGH_LIVE_TEST_TOKEN", "").strip()
    if token:
        return token
    if sys.stdin.isatty():
        token = getpass.getpass("REIGH_LIVE_TEST_TOKEN: ").strip()
        if token:
            return token
    raise ProbeError("REIGH_LIVE_TEST_TOKEN is not set")


def _request_json(path: str, params: dict[str, str]) -> Any:
    supabase_url = require_env("SUPABASE_URL").rstrip("/")
    service_key = require_env("SUPABASE_SERVICE_ROLE_KEY")
    query = urllib.parse.urlencode(params)
    url = f"{supabase_url}{path}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {service_key}",
            "apikey": service_key,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        message = payload or str(exc)
        raise ProbeError(f"{exc.code} for {path}: {message}") from exc
    except urllib.error.URLError as exc:
        raise ProbeError(f"Network error for {path}: {exc}") from exc


def _request_json_allow_http_error(path: str, params: dict[str, str]) -> tuple[int, Any]:
    supabase_url = require_env("SUPABASE_URL").rstrip("/")
    service_key = require_env("SUPABASE_SERVICE_ROLE_KEY")
    query = urllib.parse.urlencode(params)
    url = f"{supabase_url}{path}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {service_key}",
            "apikey": service_key,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
            return response.status, json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        parsed: Any
        try:
            parsed = json.loads(payload) if payload else None
        except json.JSONDecodeError:
            parsed = payload
        return exc.code, parsed


def assert_user_api_token_shape() -> dict[str, Any]:
    token_rows = _request_json(
        "/rest/v1/user_api_tokens",
        {
            "select": "token,user_id",
            "limit": "1",
        },
    )
    if not isinstance(token_rows, list):
        raise ProbeError("Expected list response when checking user_api_tokens.token")

    jti_status, jti_payload = _request_json_allow_http_error(
        "/rest/v1/user_api_tokens",
        {
            "select": "jti_hash",
            "limit": "1",
        },
    )
    if jti_status == 200:
        raise ProbeError("user_api_tokens.jti_hash still exists in the live schema")

    payload_text = json.dumps(jti_payload).lower()
    if jti_status not in {400, 404} or "jti_hash" not in payload_text:
        raise ProbeError(
            f"Unexpected response when checking jti_hash absence: status={jti_status} payload={jti_payload!r}"
        )

    return {
        "token_select_status": "ok",
        "token_probe_rows": len(token_rows),
        "jti_hash_status": jti_status,
        "jti_hash_payload": jti_payload,
    }


def resolve_token_to_user_id(token: str) -> str:
    rows = _request_json(
        "/rest/v1/user_api_tokens",
        {
            "select": "user_id",
            "token": f"eq.{token}",
        },
    )
    if not isinstance(rows, list):
        raise ProbeError("Expected list response when resolving REIGH_LIVE_TEST_TOKEN")
    if len(rows) != 1:
        raise ProbeError(f"Expected exactly one user_api_tokens row for the live token, got {len(rows)}")
    user_id = rows[0].get("user_id")
    if not user_id:
        raise ProbeError("Resolved token row is missing user_id")
    return str(user_id)


def load_projects_for_user(user_id: str) -> list[dict[str, Any]]:
    rows = _request_json(
        "/rest/v1/projects",
        {
            "select": "id,name,user_id,created_at",
            "user_id": f"eq.{user_id}",
            "order": "created_at.asc",
        },
    )
    if not isinstance(rows, list):
        raise ProbeError("Expected list response when loading user projects")
    return rows


def _load_tasks_for_project(project_id: str, status: str) -> list[dict[str, Any]]:
    rows = _request_json(
        "/rest/v1/tasks",
        {
            "select": "id,project_id,status,created_at,generation_started_at,params",
            "project_id": f"eq.{project_id}",
            "status": f"eq.{status}",
            "order": "created_at.asc",
        },
    )
    if not isinstance(rows, list):
        raise ProbeError(f"Expected list response when loading tasks for project {project_id}")
    return rows


def load_non_live_test_tasks(user_id: str) -> list[dict[str, Any]]:
    projects = load_projects_for_user(user_id)
    by_project = {project["id"]: project for project in projects if project.get("id")}
    unexpected: list[dict[str, Any]] = []
    for project_id in by_project:
        for status in ("Queued", "In Progress"):
            rows = _load_tasks_for_project(project_id, status)
            for row in rows:
                params = row.get("params") or {}
                live_test_flag = str(params.get("live_test", "false")).lower() == "true"
                if live_test_flag:
                    continue
                unexpected.append(
                    {
                        "id": row.get("id"),
                        "project_id": project_id,
                        "project_name": by_project[project_id].get("name"),
                        "status": row.get("status"),
                        "created_at": row.get("created_at"),
                        "generation_started_at": row.get("generation_started_at"),
                        "live_test": params.get("live_test", False),
                    }
                )
    return unexpected


def mask_identifier(value: str) -> str:
    if len(value) <= 12:
        return value
    return f"{value[:8]}...{value[-4:]}"


def main() -> int:
    load_environment()
    token = get_live_test_token()

    schema_probe = assert_user_api_token_shape()
    user_id = resolve_token_to_user_id(token)
    projects = load_projects_for_user(user_id)
    unexpected_tasks = load_non_live_test_tasks(user_id)

    output = {
        "schema_probe": schema_probe,
        "resolved_user_id": user_id,
        "resolved_user_id_masked": mask_identifier(user_id),
        "project_count": len(projects),
        "projects": [
            {
                "id": project.get("id"),
                "name": project.get("name"),
                "created_at": project.get("created_at"),
            }
            for project in projects
        ],
        "unexpected_non_live_tasks": unexpected_tasks,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
