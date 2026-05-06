"""Warm-cache route/profile planning contracts."""

from __future__ import annotations

import json

from source.runtime.worker.health_labels import read_warm_cache_state
from source.runtime.worker.warm_cache import publish_warm_cache_state, resolve_warm_cache_plan


def test_warm_cache_plan_uses_backend_profile_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "warm-cache.json"
    manifest.write_text(
        json.dumps(
            {
                "routes": [
                    {"backend": "vibecomfy", "profile": "3", "preload_model": "vibe-template-cache"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REIGH_WARM_CACHE_MANIFEST", str(manifest))

    plan = resolve_warm_cache_plan(backend="vibecomfy", profile="3")

    assert plan.preload_model == "vibe-template-cache"
    assert plan.source == "manifest"
    assert plan.skip_reason == ""


def test_warm_cache_plan_preserves_pending_task_skip(monkeypatch):
    monkeypatch.setenv("REIGH_WARM_CACHE_PRELOAD_MODEL", "would-be-warm")

    plan = resolve_warm_cache_plan(backend="wgp", profile="1", pending_tasks=True)

    assert plan.preload_model is None
    assert plan.source == "pending_task_guard"
    assert plan.skip_reason == "pending_tasks"


def test_warm_cache_state_is_redacted_and_queryable(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_WARM_CACHE_STATE_DIR", str(tmp_path))
    plan = resolve_warm_cache_plan(backend="wgp", profile="1", cli_preload_model="cli-model")

    publish_warm_cache_state("worker-1", plan, status="warmup")

    state = read_warm_cache_state("worker-1")
    assert state["warm_cache_status"] == "warmup"
    assert state["warm_cache_model"] == "cli-model"
    assert state["backend"] == "wgp"
