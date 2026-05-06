"""Worker preflight and readiness metadata contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

from source.runtime.worker import guardian
from source.runtime.worker.health_labels import write_worker_route_state
from source.runtime.worker.preflight import (
    PREFLIGHT_STATUS_PASSED,
    PreflightCheck,
    WorkerPreflightResult,
    preflight_state_path,
    publish_preflight_metadata,
    run_worker_preflight,
    write_preflight_state,
)
from source.runtime.worker.resource_pressure import ResourcePressureResult, write_resource_pressure_state


def _make_worker_repo(tmp_path):
    repo_root = tmp_path / "Reigh-Worker"
    wan2gp = repo_root / "Wan2GP"
    (wan2gp / "models").mkdir(parents=True)
    (wan2gp / "plugins").mkdir()
    (wan2gp / ".git").write_text("gitdir: ../.git/modules/Wan2GP\n", encoding="utf-8")
    (wan2gp / "wgp.py").write_text("# wgp\n", encoding="utf-8")
    (repo_root / "source" / "task_handlers" / "tasks").mkdir(parents=True)
    (repo_root / "source" / "task_handlers" / "tasks" / "dispatch_manifest.py").write_text("# manifest\n", encoding="utf-8")
    (repo_root / "source" / "models" / "lora").mkdir(parents=True)
    (repo_root / "source" / "models" / "lora" / "module_manifest.py").write_text("# manifest\n", encoding="utf-8")

    vibecomfy = tmp_path / "vibecomfy"
    (vibecomfy / "workflow_corpus" / "manifests").mkdir(parents=True)
    (vibecomfy / "template_index.json").write_text("{}", encoding="utf-8")
    (vibecomfy / "workflow_corpus" / "manifests" / "coverage.json").write_text("{}", encoding="utf-8")
    return repo_root, wan2gp


def test_worker_preflight_passes_when_required_paths_and_manifests_exist(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))

    def _fake_find_spec(module_name):
        if module_name in {"torch", "dotenv", "fastapi"}:
            return SimpleNamespace(origin=f"/fake/{module_name}.py")
        return None

    monkeypatch.setattr("source.runtime.worker.preflight.importlib.util.find_spec", _fake_find_spec)

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
    )

    assert result.status == PREFLIGHT_STATUS_PASSED
    assert result.failed_checks == []
    assert {check.name for check in result.checks} >= {
        "wan2gp_path",
        "wgp_entrypoint",
        "vibecomfy_available",
        "vibecomfy_template_index",
        "vibecomfy_custom_nodes_manifest",
        "task_dispatch_manifest",
        "lora_module_manifest",
        "wan2gp_models_dir",
        "wan2gp_plugins_dir",
        "main_output_dir",
        "uv_cache_dir",
    }


def test_worker_preflight_fails_when_wgp_path_is_missing(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))
    monkeypatch.setattr("source.runtime.worker.preflight.importlib.util.find_spec", lambda _name: SimpleNamespace(origin="/fake"))

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp / "missing",
        main_output_dir=tmp_path / "outputs",
        backend="wgp",
    )

    assert result.status == "failed"
    assert "wan2gp_path" in result.failed_checks
    assert "wgp_entrypoint" in result.failed_checks


class _FakeQuery:
    def __init__(self, client):
        self.client = client

    def select(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self.client.updated_payload = payload
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.client.updated_payload is None:
            return SimpleNamespace(data=[{"metadata": {"existing": "kept"}}])
        return SimpleNamespace(data=[self.client.updated_payload])


class _FakeSupabase:
    def __init__(self):
        self.updated_payload = None

    def table(self, table_name):
        assert table_name == "workers"
        return _FakeQuery(self)


def test_publish_preflight_metadata_merges_existing_metadata_and_ready_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    client = _FakeSupabase()
    result = WorkerPreflightResult(
        status="passed",
        checks=[PreflightCheck("wgp_import", True, "ok")],
        started_at=1.0,
        completed_at=2.0,
    )

    assert publish_preflight_metadata(
        supabase_client=client,
        worker_id="worker-1",
        result=result,
        ready_for_tasks=True,
    )

    metadata = client.updated_payload["metadata"]
    assert metadata["existing"] == "kept"
    assert metadata["preflight_status"] == "passed"
    assert metadata["ready_for_tasks"] is True
    assert json.loads(preflight_state_path("worker-1").read_text(encoding="utf-8"))["preflight_status"] == "passed"


def test_guardian_heartbeat_includes_preflight_status_log(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    write_preflight_state(
        "worker-1",
        {
            "preflight_status": "failed",
            "preflight_ok": False,
            "preflight_failed_checks": ["wgp_entrypoint"],
        },
    )
    captured = {}

    def _fake_run_subprocess(args, **_kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout=b'{"success": true}')

    monkeypatch.setattr(guardian, "run_subprocess", _fake_run_subprocess)

    assert guardian.send_heartbeat_with_logs(
        worker_id="worker-1",
        vram_total=1024,
        vram_used=512,
        logs=[],
        config={"db_url": "https://example.test", "api_key": "key"},
    )
    payload = json.loads(captured["args"][captured["args"].index("-d") + 1])
    preflight_logs = [log for log in payload["logs_param"] if log["message"] == "worker_preflight_status"]
    assert preflight_logs[-1]["metadata"]["preflight_status"] == "failed"
    health_logs = [log for log in payload["logs_param"] if log["message"] == "worker_health_labels"]
    assert health_logs[-1]["metadata"]["preflight"]["status"] == "failed"


def test_guardian_heartbeat_exposes_queryable_safe_telemetry_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_WARM_CACHE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_RESOURCE_PRESSURE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_ROUTE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    monkeypatch.setenv("REIGH_WORKER_RUN_ID", "run-abc")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-leak")
    write_preflight_state(
        "worker-telemetry",
        {
            "preflight_status": "passed",
            "preflight_ok": True,
            "preflight_failed_checks": [],
        },
    )
    write_worker_route_state(
        "worker-telemetry",
        {
            "task_id": "11111111-1111-1111-1111-111111111111",
            "task_type": "z_image_turbo",
            "params": {
                "route_contract": {
                    "selector_namespace": "canary",
                    "selector_version": 42,
                    "route_key": "z_image_turbo",
                    "selected_backend": "vibecomfy",
                    "selected_profile": "3",
                    "selected_template_id": "image/z_image",
                    "route_run_id": "run-abc",
                    "worker_contract_version": 1,
                }
            },
        },
    )
    write_resource_pressure_state(
        "worker-telemetry",
        ResourcePressureResult(
            status="near_full",
            action="claim_suppressed",
            allow_work=False,
            quota_alert=True,
            required_free_bytes=1024,
            recovered_bytes=0,
            volumes=(),
            cleanup={"lora": {}, "artifacts": {}},
            reason="disk_pressure_unrecoverable",
        ),
    )
    captured = {}

    def _fake_run_subprocess(args, **_kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout=b'{"success": true}')

    monkeypatch.setattr(guardian, "run_subprocess", _fake_run_subprocess)

    assert guardian.send_heartbeat_with_logs(
        worker_id="worker-telemetry",
        vram_total=1024,
        vram_used=512,
        logs=[{"level": "info", "message": "task log", "metadata": {"api_token": "secret"}}],
        config={"db_url": "https://example.test", "api_key": "key"},
    )
    payload = json.loads(captured["args"][captured["args"].index("-d") + 1])
    health_logs = [log for log in payload["logs_param"] if log["message"] == "worker_health_labels"]
    metadata = health_logs[-1]["metadata"]

    assert metadata["backend"] == "vibecomfy"
    assert metadata["profile"] == "3"
    assert metadata["route_key"] == "z_image_turbo"
    assert metadata["template_id"] == "image/z_image"
    assert metadata["run_id"] == "run-abc"
    assert metadata["selector_namespace"] == "canary"
    assert metadata["selector_version"] == "42"
    assert metadata["preflight_status"] == "passed"
    assert metadata["disk_status"] in {"ok", "near_full"}
    assert metadata["resource_pressure_status"] == "near_full"
    assert metadata["quota_alert"] is True
    assert metadata["route"]["current_task_type"] == "z_image_turbo"
    assert "must-not-leak" not in json.dumps(payload)
