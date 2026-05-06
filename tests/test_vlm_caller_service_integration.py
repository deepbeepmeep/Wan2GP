"""Caller integration tests for shared VLM prompt service wrappers."""

from __future__ import annotations

import importlib
import sys
import types


def _install_fake_torch() -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules.setdefault("torch", fake_torch)


def test_single_image_prompt_batch_delegates_to_shared_service(monkeypatch):
    _install_fake_torch()
    single_prompts = importlib.import_module("source.media.vlm.single_image_prompts")
    called = {}

    def _fake_service(**kwargs):
        called.update(kwargs)
        return ["single service prompt"]

    monkeypatch.setattr(single_prompts, "generate_single_image_prompts", _fake_service)

    result = single_prompts.generate_single_image_prompts_batch(
        image_paths=["a.png"],
        base_prompts=["base"],
        device="cpu",
    )

    assert result == ["single service prompt"]
    assert called == {
        "image_paths": ["a.png"],
        "base_prompts": ["base"],
        "device": "cpu",
    }


def test_single_image_prompt_preserves_empty_service_fallback(monkeypatch):
    _install_fake_torch()
    single_prompts = importlib.import_module("source.media.vlm.single_image_prompts")
    monkeypatch.setattr(single_prompts, "generate_single_image_prompts", lambda **_kwargs: [])

    assert single_prompts.generate_single_image_prompt("a.png", base_prompt="base", device="cpu") == "base"
    assert single_prompts.generate_single_image_prompt("a.png", base_prompt="", device="cpu") == "cinematic video"
