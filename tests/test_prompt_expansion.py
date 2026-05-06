"""Fixture tests for worker-owned prompt expansion preprocessing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from source.media.prompt_expansion import expand_qwen_prompt, qwen_prompt_expansion_requested


class _FakeExpander:
    def __init__(self, calls):
        self.calls = calls

    def extend(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            prompt=f"expanded: {kwargs['prompt']}",
            status=True,
            seed=kwargs.get("seed"),
            message="fake",
        )


def test_qwen_prompt_expansion_uses_worker_owned_expander_factory():
    factory_calls = []
    extend_calls = []

    def _factory(**kwargs):
        factory_calls.append(kwargs)
        return _FakeExpander(extend_calls)

    result = expand_qwen_prompt(
        "small red boat",
        task_type="qwen_image_2512",
        params={"qwen_prompt_expansion": True},
        expander_factory=_factory,
        device="cpu",
        seed=12,
    )

    assert result.prompt == "expanded: small red boat"
    assert result.metadata.applied is True
    assert result.metadata.original_prompt == "small red boat"
    assert result.metadata.expanded_prompt == "expanded: small red boat"
    assert result.metadata.raw["seed"] == 12
    assert factory_calls == [{"model_name": "Qwen2.5_14B", "device": "cpu", "is_vl": False}]
    assert extend_calls[0]["prompt"] == "small red boat"
    assert "faithful prompts" in extend_calls[0]["system_prompt"]


def test_qwen_prompt_expansion_requires_narrow_flag():
    result = expand_qwen_prompt(
        "small red boat",
        task_type="qwen_image_2512",
        params={"prompt_enhancer": True},
        expander_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )

    assert result.prompt == "small red boat"
    assert result.metadata.applied is False
    assert result.metadata.reason == "not_requested"


def test_qwen_prompt_expansion_skips_non_qwen_routes():
    result = expand_qwen_prompt(
        "small red boat",
        task_type="z_image_turbo",
        params={"qwen_prompt_expansion": True},
        expander_factory=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )

    assert result.prompt == "small red boat"
    assert result.metadata.applied is False
    assert result.metadata.reason == "unsupported_task_type"


def test_qwen_prompt_expansion_requested_accepts_bool_and_string_values():
    assert qwen_prompt_expansion_requested({"qwen_prompt_expansion": True}) is True
    assert qwen_prompt_expansion_requested({"qwen_prompt_expansion": "yes"}) is True
    assert qwen_prompt_expansion_requested({"qwen_prompt_expansion": "false"}) is False


def test_vibecomfy_ready_templates_contain_no_qwen_prompt_expansion_logic():
    ready_templates = Path(__file__).resolve().parents[2] / "vibecomfy" / "ready_templates"

    matches = [
        path
        for path in ready_templates.rglob("*.py")
        if "qwen_prompt_expansion" in path.read_text(encoding="utf-8")
        or "create_qwen_prompt_expander" in path.read_text(encoding="utf-8")
    ]

    assert matches == []
