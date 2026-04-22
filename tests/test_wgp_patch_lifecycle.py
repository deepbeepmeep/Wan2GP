"""Contracts for WGP patch idempotence metadata and rollback behavior."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import source.models.wgp.wgp_patches as wgp_patches
from source.runtime.wgp_ports import runtime_registry


class _FakeRuntime:
    def __init__(self, model_name: str, *, active: bool) -> None:
        self.models_def = {model_name: {"existing": 1}}
        self.transformer_type = model_name if active else None
        self.wan_model = SimpleNamespace(model_def={"existing": 1}) if active else None
        self.offloadobj = None
        self.reload_needed = False


def test_runtime_model_definition_patch_registers_in_single_registry(monkeypatch) -> None:
    context_id = "test.runtime.single_registry"
    patch_name = "ltx2_runtime_fork_markers"
    model_name = "ltx2_22B"
    marker_key = "_banodoco_fork_runtime_patch"
    target_key = f"model_def:{model_name}"
    runtime = _FakeRuntime(model_name, active=False)
    monkeypatch.setattr(runtime_registry, "get_wgp_runtime_module", lambda **_kwargs: runtime)

    try:
        assert wgp_patches.apply_runtime_model_definition_patch(
            model_name,
            {marker_key: True},
            patch_name=patch_name,
            context_id=context_id,
        ) is True

        assert runtime.models_def[model_name][marker_key] is True
        assert wgp_patches.get_wgp_patch_state(context_id=context_id)[patch_name][target_key]["applied"] is True

        restored = wgp_patches.rollback_wgp_patches(
            patch_name=patch_name,
            target_key=target_key,
            context_id=context_id,
        )

        assert restored == 1
        assert marker_key not in runtime.models_def[model_name]
        assert wgp_patches.get_wgp_patch_state(context_id=context_id) == {}
    finally:
        wgp_patches.clear_wgp_patch_context(context_id)


def test_runtime_model_definition_patch_is_dual_target_live_guard_idempotent(monkeypatch) -> None:
    context_id = "test.runtime.dual_target"
    patch_name = "ltx2_runtime_fork_markers"
    model_name = "ltx2_22B"
    marker_key = "_banodoco_fork_runtime_patch"
    target_key = f"model_def:{model_name}"
    runtime = _FakeRuntime(model_name, active=True)
    monkeypatch.setattr(runtime_registry, "get_wgp_runtime_module", lambda **_kwargs: runtime)

    try:
        assert wgp_patches.apply_runtime_model_definition_patch(
            model_name,
            {marker_key: True},
            patch_name=patch_name,
            context_id=context_id,
        ) is True

        assert runtime.models_def[model_name][marker_key] is True
        assert runtime.wan_model.model_def[marker_key] is True
        first_models_def = dict(runtime.models_def[model_name])
        first_loaded = dict(runtime.wan_model.model_def)
        first_state = wgp_patches.get_wgp_patch_state(context_id=context_id)[patch_name]

        assert wgp_patches.apply_runtime_model_definition_patch(
            model_name,
            {marker_key: True},
            patch_name=patch_name,
            context_id=context_id,
        ) is True

        assert runtime.models_def[model_name] == first_models_def
        assert runtime.wan_model.model_def == first_loaded
        assert wgp_patches.get_wgp_patch_state(context_id=context_id)[patch_name] == first_state

        restored = wgp_patches.rollback_wgp_patches(
            patch_name=patch_name,
            target_key=target_key,
            context_id=context_id,
        )

        assert restored == 1
        assert runtime.models_def[model_name] == {"existing": 1}
        assert runtime.wan_model.model_def == {"existing": 1}
    finally:
        wgp_patches.clear_wgp_patch_context(context_id)


def test_runtime_model_definition_patch_reapplies_when_models_def_drifts(monkeypatch) -> None:
    context_id = "test.runtime.models_def_drift"
    patch_name = "ltx2_runtime_fork_markers"
    model_name = "ltx2_22B"
    marker_key = "_banodoco_fork_runtime_patch"
    target_key = f"model_def:{model_name}"
    runtime = _FakeRuntime(model_name, active=True)
    monkeypatch.setattr(runtime_registry, "get_wgp_runtime_module", lambda **_kwargs: runtime)

    try:
        assert wgp_patches.apply_runtime_model_definition_patch(
            model_name,
            {marker_key: True},
            patch_name=patch_name,
            context_id=context_id,
        ) is True

        runtime.models_def[model_name][marker_key] = "drifted_mdef"

        assert wgp_patches.apply_runtime_model_definition_patch(
            model_name,
            {marker_key: True},
            patch_name=patch_name,
            context_id=context_id,
        ) is True

        assert runtime.models_def[model_name][marker_key] is True
        assert runtime.wan_model.model_def[marker_key] is True

        restored = wgp_patches.rollback_wgp_patches(
            patch_name=patch_name,
            target_key=target_key,
            context_id=context_id,
        )

        assert restored == 1
        assert runtime.models_def[model_name] == {"existing": 1}
        assert runtime.wan_model.model_def == {"existing": 1}
    finally:
        wgp_patches.clear_wgp_patch_context(context_id)


def test_runtime_model_definition_patch_reapplies_when_wan_model_drifts(monkeypatch) -> None:
    context_id = "test.runtime.loaded_model_drift"
    patch_name = "ltx2_runtime_fork_markers"
    model_name = "ltx2_22B"
    marker_key = "_banodoco_fork_runtime_patch"
    target_key = f"model_def:{model_name}"
    runtime = _FakeRuntime(model_name, active=True)
    monkeypatch.setattr(runtime_registry, "get_wgp_runtime_module", lambda **_kwargs: runtime)

    try:
        assert wgp_patches.apply_runtime_model_definition_patch(
            model_name,
            {marker_key: True},
            patch_name=patch_name,
            context_id=context_id,
        ) is True

        runtime.wan_model.model_def[marker_key] = "drifted_loaded"
        assert runtime.models_def[model_name][marker_key] is True

        assert wgp_patches.apply_runtime_model_definition_patch(
            model_name,
            {marker_key: True},
            patch_name=patch_name,
            context_id=context_id,
        ) is True

        assert runtime.models_def[model_name][marker_key] is True
        assert runtime.wan_model.model_def[marker_key] is True

        restored = wgp_patches.rollback_wgp_patches(
            patch_name=patch_name,
            target_key=target_key,
            context_id=context_id,
        )

        assert restored == 1
        assert runtime.models_def[model_name] == {"existing": 1}
        assert runtime.wan_model.model_def == {"existing": 1}
    finally:
        wgp_patches.clear_wgp_patch_context(context_id)


def test_qwen_lora_directory_patch_is_idempotent_and_rollbackable(tmp_path):
    wgp_patches.rollback_wgp_patches()

    qwen_dir = tmp_path / "loras_qwen"
    qwen_dir.mkdir()
    original_get_lora_dir = lambda _model_type: "default_lora_dir"
    fake_wgp_module = types.SimpleNamespace(get_lora_dir=original_get_lora_dir)

    assert wgp_patches.apply_qwen_lora_directory_patch(fake_wgp_module, str(tmp_path)) is True
    first_wrapper = fake_wgp_module.get_lora_dir
    assert fake_wgp_module.get_lora_dir("qwen_image_edit") == str(qwen_dir)

    assert wgp_patches.apply_qwen_lora_directory_patch(fake_wgp_module, str(tmp_path)) is True
    assert fake_wgp_module.get_lora_dir is first_wrapper

    patch_state = wgp_patches.get_wgp_patch_state()["qwen_lora_directory"]
    target_key = next(iter(patch_state.keys()))
    assert patch_state[target_key]["applied"] is True

    restored = wgp_patches.rollback_wgp_patches(
        patch_name="qwen_lora_directory",
        target_key=target_key,
    )
    assert restored == 1
    assert fake_wgp_module.get_lora_dir is original_get_lora_dir


def test_lora_caching_patch_is_idempotent_and_restorable(monkeypatch):
    wgp_patches.rollback_wgp_patches()

    load_calls: list[tuple[object, tuple[str, ...]]] = []
    unload_calls: list[object] = []

    def _original_load(model, lora_path, lora_multi=None, **kwargs):
        _ = (lora_multi, kwargs)
        paths = tuple(lora_path if isinstance(lora_path, list) else [lora_path])
        load_calls.append((model, paths))
        return "loaded"

    def _original_unload(model):
        unload_calls.append(model)
        return None

    fake_offload = types.SimpleNamespace(
        load_loras_into_model=_original_load,
        unload_loras_from_model=_original_unload,
    )
    fake_mmgp = types.ModuleType("mmgp")
    fake_mmgp.offload = fake_offload
    monkeypatch.setitem(sys.modules, "mmgp", fake_mmgp)

    assert wgp_patches.apply_lora_caching_patch() is True
    patched_load = fake_offload.load_loras_into_model
    assert patched_load is not _original_load

    model = types.SimpleNamespace(_loras_adapters=object())
    fake_offload.load_loras_into_model(model, ["a.safetensors"])
    fake_offload.load_loras_into_model(model, ["a.safetensors"])

    assert len(load_calls) == 1
    assert len(unload_calls) == 1

    assert wgp_patches.apply_lora_caching_patch() is True
    assert fake_offload.load_loras_into_model is patched_load

    patch_state = wgp_patches.get_wgp_patch_state()["lora_caching"]
    target_key = next(iter(patch_state.keys()))
    restored = wgp_patches.rollback_wgp_patches(
        patch_name="lora_caching",
        target_key=target_key,
    )
    assert restored == 2
    assert fake_offload.load_loras_into_model is _original_load
    assert fake_offload.unload_loras_from_model is _original_unload
