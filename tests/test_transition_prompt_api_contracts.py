"""Contract tests for single-transition API behavior."""

from __future__ import annotations

import importlib
import sys
import types


def _import_transition_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules.setdefault("torch", fake_torch)
    return importlib.import_module("source.media.vlm.transition_prompts")


def test_generate_transition_prompt_delegates_to_batch_pipeline(monkeypatch):
    transition_prompts = _import_transition_module()
    called = {}

    def _fake_batch(**kwargs):
        called.update(kwargs)
        return ["delegated prompt"]

    monkeypatch.setattr(transition_prompts, "generate_transition_prompts_batch", _fake_batch)

    result = transition_prompts.generate_transition_prompt(
        "start.png",
        "end.png",
        base_prompt="cinematic move",
        device="cpu",
    )

    assert result == "delegated prompt"
    assert called["image_pairs"] == [("start.png", "end.png")]
    assert called["base_prompts"] == ["cinematic move"]
    assert called["device"] == "cpu"
    assert called["task_id"] is None
    assert called["upload_debug_images"] is False


def test_generate_transition_prompt_falls_back_when_batch_returns_empty(monkeypatch):
    transition_prompts = _import_transition_module()
    monkeypatch.setattr(
        transition_prompts,
        "generate_transition_prompts_batch",
        lambda **_kwargs: [],
    )
    result = transition_prompts.generate_transition_prompt(
        "start.png",
        "end.png",
        base_prompt="",
        device="cpu",
    )
    assert result == "cinematic transition"


def test_generate_transition_prompts_batch_delegates_to_shared_service(monkeypatch):
    transition_prompts = _import_transition_module()
    called = {}

    def _fake_service(**kwargs):
        called.update(kwargs)
        return ["service prompt"]

    monkeypatch.setattr(transition_prompts, "generate_transition_pair_prompts", _fake_service)

    result = transition_prompts.generate_transition_prompts_batch(
        image_pairs=[("a.png", "b.png")],
        base_prompts=["base"],
        device="cpu",
        task_id="task-ignored",
        upload_debug_images=True,
    )

    assert result == ["service prompt"]
    assert called == {
        "image_pairs": [("a.png", "b.png")],
        "base_prompts": ["base"],
        "device": "cpu",
    }
