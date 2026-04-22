"""Contracts for runtime_registry patch transactions."""

from __future__ import annotations

from types import SimpleNamespace

from source.runtime.wgp_ports import runtime_registry


class _FakeRuntime:
    def __init__(self) -> None:
        self.models_def = {"model-a": {"existing": 1}}
        self.transformer_type = "model-a"
        self.wan_model = SimpleNamespace(model_def={"existing": 1})
        self.offloadobj = None
        self.reload_needed = False

    def init_model_def(self, _model_name: str, model_def: dict[str, object]) -> dict[str, object]:
        initialized = dict(model_def)
        initialized["initialized"] = True
        return initialized


def test_runtime_model_patch_transaction_apply_and_rollback(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_registry, "get_wgp_runtime_module", lambda **_kwargs: runtime)

    transaction = runtime_registry.begin_runtime_model_patch(
        "model-a",
        keys=("svi2pro", "sliding_window"),
    )

    runtime_registry.apply_runtime_model_patch(
        transaction,
        {
            "svi2pro": True,
            "sliding_window": True,
        },
    )

    assert runtime.models_def["model-a"]["svi2pro"] is True
    assert runtime.wan_model.model_def["sliding_window"] is True

    runtime_registry.rollback_runtime_model_patch(transaction)

    assert "svi2pro" not in runtime.models_def["model-a"]
    assert "sliding_window" not in runtime.wan_model.model_def


def test_upsert_model_definition_updates_loaded_model_for_active_transformer(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(runtime_registry, "get_wgp_runtime_module", lambda **_kwargs: runtime)

    result = runtime_registry.upsert_wgp_model_definition(
        "model-a",
        {"foo": "bar"},
        initialize=True,
    )

    assert result["foo"] == "bar"
    assert result["initialized"] is True
    assert runtime.models_def["model-a"]["foo"] == "bar"
    assert runtime.wan_model.model_def["foo"] == "bar"
