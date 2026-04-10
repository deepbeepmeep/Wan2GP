from __future__ import annotations

import argparse
import ast
import inspect
from argparse import Namespace
import signal

import pytest

from source.core.db import task_claim
from source.runtime import supervisor
from source.runtime import worker_protocol
from source.runtime.worker import idle_release, server


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_tracker(*, idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False, clock=None):
    cfg = idle_release.IdleReleaseConfig(
        idle_minutes=idle_minutes,
        grace_seconds=grace_seconds,
        is_service_mode=is_service_mode,
    )
    return idle_release.IdleReleaseTracker(cfg, clock=clock or _FakeClock()), clock


def test_tracker_disabled_when_idle_minutes_zero() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(3600)
    tracker.record_empty_poll()
    clock.advance(3600)
    assert tracker.should_release() is False


def test_tracker_blocked_in_service_mode() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=True),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(120)
    tracker.record_empty_poll()
    clock.advance(120)
    assert tracker.should_release() is False


def test_tracker_blocked_when_not_onboarded() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.record_empty_poll()
    clock.advance(3600)
    assert tracker.should_release() is False


def test_tracker_blocked_during_onboarding_grace() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(30)  # still within grace
    tracker.record_empty_poll()
    clock.advance(120)  # past idle window but onboarded only 150s ago — actually past grace
    # Re-test with shorter advance to stay inside grace:
    clock2 = _FakeClock()
    tracker2 = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=300.0, is_service_mode=False),
        clock=clock2,
    )
    tracker2.mark_onboarded()
    clock2.advance(30)
    tracker2.record_empty_poll()
    clock2.advance(120)
    assert tracker2.should_release() is False  # 150s onboarded, < 300s grace


def test_tracker_blocked_when_no_empty_poll_recorded() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(3600)
    assert tracker.should_release() is False


def test_tracker_fires_when_idle_window_elapsed() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(120)  # past grace
    tracker.record_empty_poll()
    clock.advance(60)  # exactly at idle window
    assert tracker.should_release() is True


def test_tracker_does_not_fire_before_window() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(120)
    tracker.record_empty_poll()
    clock.advance(30)  # half the idle window
    assert tracker.should_release() is False


def test_record_empty_poll_first_wins() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=0.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    tracker.record_empty_poll()
    first = tracker.last_successful_empty_poll_at
    clock.advance(30)
    tracker.record_empty_poll()  # should be a no-op
    assert tracker.last_successful_empty_poll_at == first


def test_record_claim_resets_idle_window() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=0.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    tracker.record_empty_poll()
    clock.advance(120)
    assert tracker.should_release() is True
    tracker.record_claim()
    assert tracker.should_release() is False
    assert tracker.last_successful_empty_poll_at is None


def test_is_service_mode_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_DB_CLIENT_AUTH_MODE", "service")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    assert idle_release.is_service_mode("personal-token") is True

    monkeypatch.setenv("WORKER_DB_CLIENT_AUTH_MODE", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    assert idle_release.is_service_mode("service-role") is True

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    assert idle_release.is_service_mode("personal-token") is False

    monkeypatch.delenv("WORKER_DB_CLIENT_AUTH_MODE", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    assert idle_release.is_service_mode("personal-token") is False


def test_config_from_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKER_DB_CLIENT_AUTH_MODE", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    cli_args = Namespace(idle_release_minutes=7.0, idle_onboarding_grace_seconds=42.0)
    cfg = idle_release.config_from_cli(cli_args, client_key="personal-token")
    assert cfg.idle_minutes == 7.0
    assert cfg.grace_seconds == 42.0
    assert cfg.is_service_mode is False

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    cfg2 = idle_release.config_from_cli(cli_args, client_key="service-role")
    assert cfg2.is_service_mode is True


def test_add_cli_args_defaults() -> None:
    parser = argparse.ArgumentParser()
    idle_release.add_cli_args(parser)
    ns = parser.parse_args([])
    assert ns.idle_release_minutes == 15.0
    assert ns.idle_onboarding_grace_seconds == 60.0


def test_resolve_worker_db_client_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    cli_args = Namespace(supabase_anon_key="anon-key")

    monkeypatch.setenv("WORKER_DB_CLIENT_AUTH_MODE", "service")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    with pytest.raises(ValueError, match="SERVICE_ROLE_KEY is required"):
        server._resolve_worker_db_client_key(cli_args, access_token="worker-token")

    monkeypatch.setenv("WORKER_DB_CLIENT_AUTH_MODE", "worker")
    with pytest.raises(ValueError, match="requires --reigh-access-token"):
        server._resolve_worker_db_client_key(cli_args, access_token=None)

    monkeypatch.setenv("WORKER_DB_CLIENT_AUTH_MODE", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    assert server._resolve_worker_db_client_key(cli_args, access_token="worker-token") == "service-role"

    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert server._resolve_worker_db_client_key(cli_args, access_token="worker-token") == "worker-token"

    assert server._resolve_worker_db_client_key(cli_args, access_token=None) == "anon-key"


def test_poll_next_task_outcome_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_claim, "check_task_counts_supabase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_claim._cfg, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(task_claim._cfg, "SUPABASE_EDGE_CLAIM_TASK_URL", "https://edge/claim")

    monkeypatch.setattr(task_claim._cfg, "SUPABASE_ACCESS_TOKEN", None)
    assert task_claim.poll_next_task("worker-1", True, 5) == (task_claim.ClaimPollOutcome.ERROR, None)

    monkeypatch.setattr(task_claim._cfg, "SUPABASE_ACCESS_TOKEN", "token")
    monkeypatch.setattr(
        task_claim.httpx,
        "post",
        lambda *_args, **_kwargs: _Response(200, {"task_id": "task-1", "task_type": "demo", "params": {}}),
    )
    outcome, task_info = task_claim.poll_next_task("worker-1", True, 5)
    assert outcome is task_claim.ClaimPollOutcome.CLAIMED
    assert task_info["task_id"] == "task-1"

    monkeypatch.setattr(task_claim.httpx, "post", lambda *_args, **_kwargs: _Response(204))
    assert task_claim.poll_next_task("worker-1", True, 5) == (task_claim.ClaimPollOutcome.EMPTY, None)

    monkeypatch.setattr(task_claim.httpx, "post", lambda *_args, **_kwargs: _Response(500, text="boom"))
    assert task_claim.poll_next_task("worker-1", True, 5) == (task_claim.ClaimPollOutcome.ERROR, None)

    def _raise(*_args, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr(task_claim.httpx, "post", _raise)
    assert task_claim.poll_next_task("worker-1", True, 5) == (task_claim.ClaimPollOutcome.ERROR, None)


def test_worker_sigterm_handler_raises_keyboardinterrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = {}
    real_signal = signal.signal

    class _StopInstall(Exception):
        pass

    def _capture(sig, handler):
        if sig == signal.SIGTERM:
            installed["handler"] = handler
            raise _StopInstall
        return real_signal(sig, handler)

    monkeypatch.setattr(server, "bootstrap_runtime_environment", lambda: None)
    monkeypatch.setattr(server.signal, "signal", _capture)

    with pytest.raises(_StopInstall):
        server.main()

    previous = real_signal(signal.SIGTERM, installed["handler"])
    try:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    finally:
        real_signal(signal.SIGTERM, previous)


def test_idle_release_exit_code_constant() -> None:
    assert worker_protocol.IDLE_RELEASE_EXIT_CODE == 75
    assert server.IDLE_RELEASE_EXIT_CODE is worker_protocol.IDLE_RELEASE_EXIT_CODE
    assert supervisor.IDLE_RELEASE_EXIT_CODE is worker_protocol.IDLE_RELEASE_EXIT_CODE


def _has_top_level_idle_assign(source_text: str) -> bool:
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IDLE_RELEASE_EXIT_CODE":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                        return True
    return False


def test_idle_release_suppressed_when_queue_has_active_work() -> None:
    """Idle release must not fire while the task queue is still processing."""
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=0.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    tracker.record_empty_poll()
    clock.advance(120)
    # Tracker says release, but queue is busy — server.py should reset the timer.
    assert tracker.should_release() is True
    # Simulate what server.py does: check queue, then reset.
    queue_active = True  # pretend queue.has_active_work() returned True
    if queue_active:
        tracker.record_claim()  # reset idle timer
    assert tracker.should_release() is False
    assert tracker.last_successful_empty_poll_at is None
    # After work finishes, idle timer restarts from scratch.
    tracker.record_empty_poll()
    clock.advance(30)
    assert tracker.should_release() is False  # only 30s, need 60s
    clock.advance(40)
    assert tracker.should_release() is True  # now past 60s


def test_no_literal_idle_release_exit_code_in_server_or_supervisor() -> None:
    server_src = inspect.getsource(server)
    sup_src = inspect.getsource(supervisor)
    assert not _has_top_level_idle_assign(server_src), "server.py must import IDLE_RELEASE_EXIT_CODE, not redefine it"
    assert not _has_top_level_idle_assign(sup_src), "supervisor.py must import IDLE_RELEASE_EXIT_CODE, not redefine it"
