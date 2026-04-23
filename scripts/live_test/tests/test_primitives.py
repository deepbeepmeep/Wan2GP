from __future__ import annotations

import copy
import json
import sys
import types
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_test.completion_poller import TaskResult, poll_until_complete
from scripts.live_test.heartbeat_waiter import WorkerReadyTimeoutError, wait_until_ready
from scripts.live_test.launch_command import build_direct_worker_command, build_run_worker_command
from scripts.live_test.matrix import MATRIX, MatrixCase, run_matrix
from scripts.live_test import main as live_test_main
from scripts.live_test.preflight import (
    LIVE_TEST_PROJECT_NAME,
    UnexpectedUserWorkError,
    assert_user_queue_clean,
    get_or_create_live_test_project,
)
from scripts.live_test.report import write_report
from scripts.live_test.safety_gate import UnsafeTakeoverError, assert_safe_to_take_over
from scripts.live_test.ssh_bootstrap import (
    KILL_COMMAND,
    WorkerProcessInfo,
    capture_current_worker_cmdline,
    kill_supervisor_and_worker,
)
from scripts.live_test.task_spoofer import insert_spoof_task
from scripts.live_test.terminate_guard import guarded_terminate
from scripts.live_test.token_resolver import TokenResolutionError, resolve_token_to_user_id
from scripts.live_test.variant_fresh import run as run_variant_fresh
from scripts.live_test.variant_update import _spawn_takeover_pod, run as run_variant_update


def _iso_now(offset_seconds: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def _lookup(row: dict, key: str):
    current = row
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


class FakeResult:
    def __init__(self, data):
        self.data = data


class SequenceResponder:
    def __init__(self, responses):
        self._responses = deque(copy.deepcopy(list(responses)))

    def __call__(self, _query):
        if len(self._responses) > 1:
            return self._responses.popleft()
        return copy.deepcopy(self._responses[0]) if self._responses else []


class FakeQuery:
    def __init__(self, supabase: "FakeSupabase", table_name: str):
        self.supabase = supabase
        self.table_name = table_name
        self.filters = []
        self.order_key = None
        self.order_desc = False
        self.insert_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append(lambda row: _lookup(row, key) == value)
        return self

    def in_(self, key, values):
        allowed = set(values)
        self.filters.append(lambda row: _lookup(row, key) in allowed)
        return self

    def gte(self, key, value):
        self.filters.append(lambda row: (_lookup(row, key) or "") >= value)
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = desc
        return self

    def insert(self, payload):
        self.insert_payload = copy.deepcopy(payload)
        return self

    def single(self):
        return self

    def execute(self):
        if self.insert_payload is not None:
            row = copy.deepcopy(self.insert_payload)
            row.setdefault("id", f"{self.table_name}-row-{len(self.supabase.tables.setdefault(self.table_name, [])) + 1}")
            self.supabase.tables.setdefault(self.table_name, []).append(copy.deepcopy(row))
            self.supabase.inserted.setdefault(self.table_name, []).append(copy.deepcopy(row))
            return FakeResult([row])

        source = self.supabase.sources.get(self.table_name, self.supabase.tables.get(self.table_name, []))
        if callable(source):
            rows = source(self)
        else:
            rows = copy.deepcopy(source)

        filtered = []
        for row in rows:
            if all(predicate(row) for predicate in self.filters):
                filtered.append(copy.deepcopy(row))

        if self.order_key is not None:
            filtered.sort(key=lambda row: _lookup(row, self.order_key) or "", reverse=self.order_desc)

        return FakeResult(filtered)


class FakeSupabase:
    def __init__(self, *, tables=None, sources=None):
        self.tables = copy.deepcopy(tables or {})
        self.sources = dict(sources or {})
        self.inserted = {}

    def table(self, table_name: str) -> FakeQuery:
        return FakeQuery(self, table_name)


class FakeDB:
    def __init__(self, *, tables=None, sources=None):
        self.supabase = FakeSupabase(tables=tables, sources=sources)


class ScriptedSSH:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def execute_command(self, command, timeout=600):
        self.commands.append((command, timeout))
        if not self.responses:
            return 0, "", ""
        matcher, response = self.responses.pop(0)
        if matcher is not None:
            assert matcher in command
        return response


def test_token_resolver_returns_user_id():
    db = FakeDB(tables={"user_api_tokens": [{"user_id": "user-123", "token": "secret"}]})
    assert resolve_token_to_user_id(db, "secret") == "user-123"


def test_token_resolver_raises_on_missing():
    db = FakeDB(tables={"user_api_tokens": []})
    with pytest.raises(TokenResolutionError):
        resolve_token_to_user_id(db, "missing")


def test_preflight_raises_on_stray_user_work():
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "status": "Queued",
                    "params": {"live_test": False},
                    "projects": {"user_id": "user-1"},
                }
            ]
        }
    )
    with pytest.raises(UnexpectedUserWorkError):
        assert_user_queue_clean(db, "user-1")


def test_preflight_passes_when_clean():
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "status": "Queued",
                    "params": {"live_test": True},
                    "projects": {"user_id": "user-1"},
                }
            ],
            "projects": [{"id": "project-1", "user_id": "user-1", "name": LIVE_TEST_PROJECT_NAME}],
        }
    )
    assert_user_queue_clean(db, "user-1")
    assert get_or_create_live_test_project(db, "user-1") == "project-1"


def test_task_spoofer_stamps_live_test_and_queued_status():
    db = FakeDB(tables={"tasks": []})
    task_id = insert_spoof_task(
        db,
        "project-1",
        "qwen_image",
        {"prompt": "overridden"},
        fixture_payload={"notes": "strip me", "params": {"prompt": "base prompt"}},
    )
    inserted = db.supabase.inserted["tasks"][0]
    assert task_id == inserted["id"]
    assert inserted["status"] == "Queued"
    assert inserted["params"]["prompt"] == "overridden"
    assert inserted["params"]["live_test"] is True
    assert "notes" not in inserted


def test_completion_poller_returns_on_complete(monkeypatch: pytest.MonkeyPatch):
    task_rows = SequenceResponder(
        [
            [{"id": "task-1", "project_id": "project-1", "task_type": "qwen_image", "status": "Queued", "created_at": _iso_now(-10)}],
            [{"id": "task-1", "project_id": "project-1", "task_type": "qwen_image", "status": "Complete", "created_at": _iso_now(-10), "output_location": "https://out.example/result.png"}],
        ]
    )
    generations = [
        {
            "id": "gen-1",
            "project_id": "project-1",
            "created_at": _iso_now(-5),
            "tasks": ["task-1"],
            "params": {},
            "location": "https://out.example/result.png",
        }
    ]
    db = FakeDB(sources={"tasks": task_rows}, tables={"generations": generations})
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(db, "task-1", "project-1", timeout_sec=2, interval_sec=0, case_name="case", task_type="qwen_image")
    assert result.final_status == "Complete"
    assert result.generation_ids == ["gen-1"]
    assert result.output_location == "https://out.example/result.png"
    assert result.error_summary is None


def test_completion_poller_times_out(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [
                {"id": "task-1", "project_id": "project-1", "task_type": "qwen_image", "status": "Queued", "created_at": _iso_now(-10)}
            ],
            "generations": [],
        }
    )
    monotonic_values = iter([0.0, 0.4, 1.2, 1.4, 1.6])
    monkeypatch.setattr("scripts.live_test.completion_poller.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(db, "task-1", "project-1", timeout_sec=1, interval_sec=0, case_name="case", task_type="qwen_image")
    assert result.final_status == "Queued"
    assert "Timed out waiting for task task-1" in (result.error_summary or "")


def test_completion_poller_records_failure(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "project_id": "project-1",
                    "task_type": "qwen_image",
                    "status": "Failed",
                    "error_message": "backend exploded",
                    "created_at": _iso_now(-10),
                }
            ],
            "generations": [],
        }
    )
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(db, "task-1", "project-1", timeout_sec=1, interval_sec=0)
    assert result.final_status == "Failed"
    assert result.error_summary == "backend exploded"


def test_heartbeat_waiter_requires_dwell_not_startup_phase(monkeypatch: pytest.MonkeyPatch):
    workers = SequenceResponder(
        [
            [{"id": "worker-1", "last_heartbeat": _iso_now(), "startup_phase": None}],
            [{"id": "worker-1", "last_heartbeat": _iso_now(), "startup_phase": None}],
        ]
    )
    db = FakeDB(sources={"workers": workers})
    monkeypatch.setattr("scripts.live_test.heartbeat_waiter.time.sleep", lambda _interval: None)
    worker = wait_until_ready(db, "worker-1", timeout_sec=1, interval_sec=0, dwell_polls=2)
    assert worker["id"] == "worker-1"


def test_safety_gate_rejects_fresh_in_progress_for_user():
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "status": "In Progress",
                    "generation_started_at": _iso_now(-10),
                    "projects": {"user_id": "user-1"},
                }
            ]
        }
    )
    with pytest.raises(UnsafeTakeoverError):
        assert_safe_to_take_over(db, "pod-1", "user-1")


def test_safety_gate_rejects_fresh_heartbeat_when_not_allowed():
    db = FakeDB(
        tables={
            "tasks": [],
            "workers": [{"id": "pod-1", "last_heartbeat": _iso_now()}],
        }
    )
    with pytest.raises(UnsafeTakeoverError):
        assert_safe_to_take_over(db, "pod-1", "user-1", allow_fresh_heartbeat=False)


def test_safety_gate_permits_fresh_heartbeat_when_allowed_but_still_rejects_live_pat_work():
    clean_db = FakeDB(
        tables={
            "tasks": [],
            "workers": [{"id": "pod-1", "last_heartbeat": _iso_now()}],
        }
    )
    assert_safe_to_take_over(clean_db, "pod-1", "user-1", allow_fresh_heartbeat=True)

    busy_db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-2",
                    "status": "In Progress",
                    "generation_started_at": _iso_now(-15),
                    "projects": {"user_id": "user-1"},
                }
            ],
            "workers": [{"id": "pod-1", "last_heartbeat": _iso_now()}],
        }
    )
    with pytest.raises(UnsafeTakeoverError):
        assert_safe_to_take_over(busy_db, "pod-1", "user-1", allow_fresh_heartbeat=True)


def test_terminate_guard_respects_env_var(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setenv("REIGH_LIVE_TEST_NO_TERMINATE", "1")
    monkeypatch.setattr("scripts.live_test.terminate_guard.live_test_pkg.terminate_pod", lambda pod_id, api_key: calls.append((pod_id, api_key)))
    assert guarded_terminate("pod-1", "api-key", no_terminate=False) is False
    assert calls == []


def test_terminate_guard_respects_cli_flag(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.delenv("REIGH_LIVE_TEST_NO_TERMINATE", raising=False)
    monkeypatch.setattr("scripts.live_test.terminate_guard.live_test_pkg.terminate_pod", lambda pod_id, api_key: calls.append((pod_id, api_key)))
    assert guarded_terminate("pod-1", "api-key", no_terminate=True) is False
    assert calls == []


def test_terminate_guard_skips_on_exception_path(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.delenv("REIGH_LIVE_TEST_NO_TERMINATE", raising=False)
    monkeypatch.setattr("scripts.live_test.terminate_guard.live_test_pkg.terminate_pod", lambda pod_id, api_key: calls.append((pod_id, api_key)))
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("boom")
        finally:
            assert guarded_terminate(None, "api-key", no_terminate=False) is False
    assert calls == []


def test_build_run_worker_command_uses_run_worker_py_and_idle_zero():
    command = build_run_worker_command(
        "/workspace/Reigh-Worker-LiveTest",
        reigh_token="token-1",
        supabase_url="https://supabase.example",
        worker_id="worker-1",
        wgp_profile=3,
        idle_release_minutes=0,
    )
    assert "python run_worker.py" in command
    assert "--idle-release-minutes 0" in command
    assert "--save-logging logs/worker.log" in command


def test_build_direct_worker_command_roundtrips_template_cmdline():
    command = build_direct_worker_command(
        "/workspace/Reigh-Worker",
        cli_args=["python", "worker.py", "--task-id", "task-1", "--gpu-id", "0"],
    )
    assert command.startswith("cd /workspace/Reigh-Worker && nohup python worker.py")
    assert "> logs/startup.log 2>&1 &" in command


def test_capture_cmdline_detects_supervisor_family_from_run_worker_ps():
    ssh = ScriptedSSH(
        [
            (
                "ps -eo pid=,args=",
                (
                    0,
                    "123 python run_worker.py --worker worker-1\n124 python worker.py --task-id task-1\n",
                    "",
                ),
            )
        ]
    )
    info = capture_current_worker_cmdline(ssh)
    assert info == WorkerProcessInfo(
        family="supervisor",
        cmdline=["python", "run_worker.py", "--worker", "worker-1"],
        pid=123,
    )


def test_capture_cmdline_detects_direct_family_from_template_ps():
    ssh = ScriptedSSH(
        [
            (
                "ps -eo pid=,args=",
                (0, "222 python worker.py --task-id task-1 --gpu-id 0\n", ""),
            )
        ]
    )
    info = capture_current_worker_cmdline(ssh)
    assert info == WorkerProcessInfo(
        family="direct",
        cmdline=["python", "worker.py", "--task-id", "task-1", "--gpu-id", "0"],
        pid=222,
    )


def test_kill_supervisor_and_worker_patterns_cover_both_families(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("scripts.live_test.ssh_bootstrap.time.sleep", lambda _interval: None)
    ssh = ScriptedSSH(
        [
            (None, (0, "", "")),
            ("pgrep -af run_worker.py", (0, "123 python run_worker.py\n", "")),
            ("pgrep -af run_worker.py", (0, "", "")),
        ]
    )
    kill_supervisor_and_worker(ssh)
    assert ssh.commands[0][0] == KILL_COMMAND
    assert "pgrep -af run_worker.py" in ssh.commands[1][0]
    assert "pgrep -af 'python worker.py'" in ssh.commands[1][0]
    assert "pgrep -af source.runtime.worker" in ssh.commands[1][0]


def test_matrix_contains_exactly_eight_cases():
    assert len(MATRIX) == 8
    assert [case.name for case in MATRIX].count("z_image_turbo_i2i") == 1
    assert any(case.task_type == "z_image_turbo_i2i" for case in MATRIX)


def test_run_matrix_continues_after_individual_case_failures(monkeypatch: pytest.MonkeyPatch):
    cases = [
        MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=5),
        MatrixCase(name="case-b", task_type="qwen_image_style", fixture_key="qwen_image_style_db_task", timeout_sec=5),
    ]

    inserted = []
    polled = []

    def fake_insert(_db, _project_id, task_type, _params_overrides, **_kwargs):
        task_id = f"{task_type}-task-{len(inserted) + 1}"
        inserted.append(task_id)
        return task_id

    def fake_poll(_db, task_id, _project_id, **kwargs):
        polled.append(task_id)
        if task_id.startswith("qwen_image-task"):
            return TaskResult(
                task_id=task_id,
                case_name=kwargs["case_name"],
                task_type=kwargs["task_type"],
                final_status="Failed",
                output_location=None,
                generation_ids=[],
                elapsed_sec=1.0,
                error_summary="backend exploded",
            )
        return TaskResult(
            task_id=task_id,
            case_name=kwargs["case_name"],
            task_type=kwargs["task_type"],
            final_status="Complete",
            output_location="https://out.example/image.png",
            generation_ids=["gen-2"],
            elapsed_sec=1.0,
            error_summary=None,
        )

    monkeypatch.setattr("scripts.live_test.matrix.insert_spoof_task", fake_insert)
    monkeypatch.setattr("scripts.live_test.matrix.poll_until_complete", fake_poll)

    results = run_matrix(object(), "project-1", cases)
    assert [result.case_name for result in results] == ["case-a", "case-b"]
    assert results[0].final_status == "Failed"
    assert results[1].final_status == "Complete"
    assert inserted == ["qwen_image-task-1", "qwen_image_style-task-2"]
    assert polled == inserted


def test_write_report_outputs_json_and_markdown(tmp_path: Path):
    results = [
        TaskResult(
            task_id="task-1",
            case_name="case-a",
            task_type="qwen_image",
            final_status="Complete",
            output_location="https://out.example/a.png",
            generation_ids=["gen-1"],
            elapsed_sec=1.234,
            error_summary=None,
        ),
        TaskResult(
            task_id="task-2",
            case_name="case-b",
            task_type="qwen_image_style",
            final_status="Failed",
            output_location=None,
            generation_ids=[],
            elapsed_sec=2.345,
            error_summary="backend exploded",
        ),
    ]
    out_dir = write_report(results, "fresh", "pod-1", tmp_path / "runs" / "case-1")
    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert report_json["passed"] == 1
    assert report_json["total"] == 2
    assert "Summary: `1/2 passed`" in report_md


def test_variant_fresh_dry_run_uses_livetest_workspace_and_env_exports(capsys, monkeypatch: pytest.MonkeyPatch):
    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh._prepare_context",
        lambda _args: {
            "db": object(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_fresh._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh.config.require_env",
        lambda name: {
            "SUPABASE_URL": "https://supabase.example",
        }[name],
    )
    args = SimpleNamespace(
        dry_run=True,
        no_terminate=False,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
    )
    assert run_variant_fresh(args) == 0
    output = capsys.readouterr().out
    assert "/workspace/Reigh-Worker-LiveTest" in output
    assert "Terminate after run: True" in output
    assert "REIGH_ACCESS_TOKEN" in output
    assert "SUPABASE_SERVICE_ROLE_KEY" in output
    assert "SUPABASE_URL" in output
    assert "WORKER_DB_CLIENT_AUTH_MODE" in output


def test_spawn_takeover_pod_calls_create_record_before_spawn_and_start(monkeypatch: pytest.MonkeyPatch):
    events = []

    class FakeDB:
        async def create_worker_record(self, worker_id, instance_type):
            events.append(("create_worker_record", worker_id, instance_type))
            return True

        async def update_worker_status(self, worker_id, status, metadata):
            events.append(("update_worker_status", worker_id, status, metadata["runpod_id"]))
            return True

    class FakeSpawner:
        def __init__(self):
            events.append(("init",))
            self.gpu_type = "NVIDIA GeForce RTX 4090"

        def generate_worker_id(self):
            events.append(("generate_worker_id",))
            return "worker-123"

        async def spawn_worker(self, worker_id):
            events.append(("spawn_worker", worker_id))
            return {"runpod_id": "pod-456", "pod_details": {"id": "pod-456"}}

        async def start_worker_process(self, pod_id, worker_id, has_pending_tasks=False):
            events.append(("start_worker_process", pod_id, worker_id, has_pending_tasks))
            return True

    def _fake_factory(config, db):
        return FakeSpawner()

    monkeypatch.setitem(
        sys.modules,
        "gpu_orchestrator.worker_spawner",
        types.SimpleNamespace(create_worker_spawner=_fake_factory),
    )

    worker_id, pod_id = _spawn_takeover_pod(FakeDB(), "api-key")
    assert (worker_id, pod_id) == ("worker-123", "pod-456")
    assert events == [
        ("init",),
        ("generate_worker_id",),
        ("create_worker_record", "worker-123", "NVIDIA GeForce RTX 4090"),
        ("spawn_worker", "worker-123"),
        ("update_worker_status", "worker-123", "spawning", "pod-456"),
        ("start_worker_process", "pod-456", "worker-123", False),
    ]


def test_variant_update_spawn_takeover_threads_worker_id_not_pod_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    safety_calls = []
    wait_calls = []
    launched = []
    cleanup_calls = []
    restore_calls = []

    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    monkeypatch.setattr(
        "scripts.live_test.variant_update._prepare_context",
        lambda _args: {
            "db": object(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_update._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.config.require_env",
        lambda name: {
            "RUNPOD_API_KEY": "api-key",
            "SUPABASE_URL": "https://supabase.example",
            "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        }[name],
    )
    monkeypatch.setattr("scripts.live_test.variant_update._runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "scripts.live_test.variant_update._spawn_takeover_pod",
        lambda _db, _api_key: ("worker-123", "pod-456"),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.assert_safe_to_take_over",
        lambda _db, pod_id, user_id, allow_fresh_heartbeat=False: safety_calls.append(
            (pod_id, user_id, allow_fresh_heartbeat)
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.snapshot_local_state", lambda _path: "snapshot")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.push_working_copy_to_temp_branch",
        lambda _path, _snapshot: ("live-test/branch", "sha-1"),
    )

    class DummySSH:
        def execute_command(self, _command, timeout=600):
            return 0, "", ""

        def disconnect(self):
            return None

    monkeypatch.setattr("scripts.live_test.variant_update.open_session", lambda _pod_id, _api_key: DummySSH())
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_branch", lambda _ssh: "main")
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_sha", lambda _ssh: "sha-prev")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.capture_current_worker_cmdline",
        lambda _ssh: WorkerProcessInfo(
            family="supervisor",
            cmdline=["python", "run_worker.py", "--worker", "old-worker"],
            pid=123,
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_update._remote_checkout_and_sync", lambda _ssh, _branch: None)
    monkeypatch.setattr("scripts.live_test.variant_update.kill_supervisor_and_worker", lambda _ssh: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.launch_worker_detached",
        lambda _ssh, command: launched.append(command),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.wait_until_ready",
        lambda _db, worker_id, timeout_sec=900: wait_calls.append((worker_id, timeout_sec)),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.run_matrix", lambda _db, _project_id, _cases: [])
    monkeypatch.setattr(
        "scripts.live_test.variant_update.write_report",
        lambda _results, _variant, _pod_id, _out_dir: tmp_path,
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update._restore_remote_state",
        lambda _ssh, **kwargs: restore_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.cleanup_temp_branch",
        lambda branch, preserve, submodule_path="reigh-worker": cleanup_calls.append((branch, preserve, submodule_path))
        or branch,
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.restore_local_state",
        lambda _path, _snapshot: restore_calls.append({"local_restore": True}),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.fetch_worker_logs", lambda _ssh, _workdir: "logs")
    monkeypatch.setattr("scripts.live_test.variant_update.guarded_terminate", lambda *_args, **_kwargs: False)

    args = SimpleNamespace(
        dry_run=False,
        spawn_takeover=True,
        pod_id=None,
        no_terminate=True,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
    )
    assert run_variant_update(args) == 0
    assert safety_calls == [("pod-456", "user-1", True)]
    assert wait_calls == [("worker-123", 900)]
    assert any("--worker worker-123" in command for command in launched)
    assert all("--worker pod-456" not in command for command in launched)
    assert cleanup_calls == [("live-test/branch", False, "reigh-worker")]


def test_variant_update_existing_mode_uses_stale_heartbeat_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    safety_calls = []
    wait_calls = []

    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    monkeypatch.setattr(
        "scripts.live_test.variant_update._prepare_context",
        lambda _args: {
            "db": object(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_update._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.config.require_env",
        lambda name: {
            "RUNPOD_API_KEY": "api-key",
            "SUPABASE_URL": "https://supabase.example",
            "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        }[name],
    )
    monkeypatch.setattr("scripts.live_test.variant_update._runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.assert_safe_to_take_over",
        lambda _db, pod_id, user_id, allow_fresh_heartbeat=False: safety_calls.append(
            (pod_id, user_id, allow_fresh_heartbeat)
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.snapshot_local_state", lambda _path: "snapshot")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.push_working_copy_to_temp_branch",
        lambda _path, _snapshot: ("live-test/branch", "sha-1"),
    )

    class DummySSH:
        def execute_command(self, _command, timeout=600):
            return 0, "", ""

        def disconnect(self):
            return None

    monkeypatch.setattr("scripts.live_test.variant_update.open_session", lambda _pod_id, _api_key: DummySSH())
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_branch", lambda _ssh: "main")
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_sha", lambda _ssh: "sha-prev")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.capture_current_worker_cmdline",
        lambda _ssh: WorkerProcessInfo(
            family="supervisor",
            cmdline=["python", "run_worker.py", "--worker", "worker-prev"],
            pid=123,
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_update._resolve_existing_worker_id", lambda _db, _pod_id, _prev_proc: "worker-prev")
    monkeypatch.setattr("scripts.live_test.variant_update._remote_checkout_and_sync", lambda _ssh, _branch: None)
    monkeypatch.setattr("scripts.live_test.variant_update.kill_supervisor_and_worker", lambda _ssh: None)
    monkeypatch.setattr("scripts.live_test.variant_update.launch_worker_detached", lambda _ssh, _command: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.wait_until_ready",
        lambda _db, worker_id, timeout_sec=900: wait_calls.append((worker_id, timeout_sec)),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.run_matrix", lambda _db, _project_id, _cases: [])
    monkeypatch.setattr("scripts.live_test.variant_update.write_report", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr("scripts.live_test.variant_update._restore_remote_state", lambda _ssh, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_update.cleanup_temp_branch", lambda branch, preserve, submodule_path='reigh-worker': branch)
    monkeypatch.setattr("scripts.live_test.variant_update.restore_local_state", lambda _path, _snapshot: None)
    monkeypatch.setattr("scripts.live_test.variant_update.fetch_worker_logs", lambda _ssh, _workdir: "logs")
    monkeypatch.setattr("scripts.live_test.variant_update.guarded_terminate", lambda *_args, **_kwargs: False)

    args = SimpleNamespace(
        dry_run=False,
        spawn_takeover=False,
        pod_id="pod-existing",
        no_terminate=True,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
    )
    assert run_variant_update(args) == 0
    assert safety_calls == [("pod-existing", "user-1", False)]
    assert wait_calls == [("worker-prev", 900)]


def test_main_defaults_terminate_for_fresh_and_no_terminate_for_update(monkeypatch: pytest.MonkeyPatch):
    seen = []

    monkeypatch.setattr("scripts.live_test.main.run_variant_fresh", lambda args: seen.append(("fresh", args.no_terminate)) or 0)
    monkeypatch.setattr("scripts.live_test.main.run_variant_update", lambda args: seen.append(("update", args.no_terminate)) or 0)

    assert live_test_main.main(["--variant", "fresh", "--dry-run"]) == 0
    assert live_test_main.main(["--variant", "update", "--pod-id", "pod-1", "--dry-run"]) == 0
    assert seen == [("fresh", False), ("update", True)]
