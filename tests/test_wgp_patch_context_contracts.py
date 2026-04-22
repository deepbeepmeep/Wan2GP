"""Contracts for context-scoped WGP patch registries."""

from __future__ import annotations

from types import SimpleNamespace

from source.models.wgp import wgp_patches
from source.runtime.wgp_ports import runtime_registry


class _FakeRuntime:
    def __init__(self, models_def: dict[str, dict[str, object]], *, transformer_type: str | None = None) -> None:
        self.models_def = models_def
        self.transformer_type = transformer_type
        self.wan_model = None
        self.offloadobj = None
        self.reload_needed = False


def _make_dummy_wgp_module(models_def: dict[str, dict[str, object]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        load_wan_model=lambda *args, **kwargs: (None, None),
        get_base_model_type=lambda model_type: model_type,
        models_def=models_def or {},
    )


def test_patch_state_is_context_scoped(monkeypatch) -> None:
    monkeypatch.setattr(wgp_patches, "apply_qwen_model_routing_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_qwen_lora_directory_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_lora_multiplier_parser_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_qwen_inpainting_lora_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_lora_key_tolerance_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_lora_caching_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_headless_app_stub", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_ltx2_runtime_fork_markers_patch", lambda *_a, **_k: True)

    context_a = "test.patch.ctx.a"
    context_b = "test.patch.ctx.b"

    try:
        wgp_patches.apply_all_wgp_patches(
            _make_dummy_wgp_module(),
            ".",
            context_id=context_a,
        )
        wgp_patches.apply_all_wgp_patches(
            _make_dummy_wgp_module(),
            ".",
            context_id=context_b,
        )

        state_a = wgp_patches.get_wgp_patch_state(context_id=context_a)
        state_b = wgp_patches.get_wgp_patch_state(context_id=context_b)

        assert "qwen_model_routing" in state_a
        assert "qwen_model_routing" in state_b

        wgp_patches.clear_wgp_patch_context(context_a)
        assert wgp_patches.get_wgp_patch_state(context_id=context_a) == {}
        assert wgp_patches.get_wgp_patch_state(context_id=context_b)
    finally:
        wgp_patches.clear_wgp_patch_context(context_a)
        wgp_patches.clear_wgp_patch_context(context_b)


def test_apply_all_wgp_patches_records_ltx2_runtime_fork_markers(monkeypatch) -> None:
    context_id = "test.bootstrap"
    models_def = {
        "ltx2_22B": {},
        "ltx2_22B_distilled": {},
        "t2v": {},
    }
    runtime = _FakeRuntime(models_def, transformer_type=None)
    monkeypatch.setattr(wgp_patches, "apply_qwen_model_routing_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_qwen_lora_directory_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_lora_multiplier_parser_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_qwen_inpainting_lora_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_lora_key_tolerance_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_lora_caching_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(wgp_patches, "apply_headless_app_stub", lambda *_a, **_k: True)
    monkeypatch.setattr(runtime_registry, "get_wgp_runtime_module", lambda **_kwargs: runtime)

    try:
        wgp_patches.apply_all_wgp_patches(
            _make_dummy_wgp_module(models_def=models_def),
            ".",
            context_id=context_id,
        )

        marker_state = wgp_patches.get_wgp_patch_state(context_id=context_id)["ltx2_runtime_fork_markers"]
        assert "model_def:ltx2_22B" in marker_state
        assert "model_def:ltx2_22B_distilled" in marker_state
        assert "model_def:t2v" not in marker_state
    finally:
        wgp_patches.clear_wgp_patch_context(context_id)
