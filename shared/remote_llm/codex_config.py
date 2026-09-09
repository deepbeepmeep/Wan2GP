from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .codex_backend import CodexAuthenticationRequired, CodexBackend
from .config import ENGINE_CODEX, codex_model_choices, codex_reasoning_effort_choices, normalize_codex_model_selection, normalize_reasoning_effort_selection
from .config_ui_common import cached_model_catalog, save_model_catalog


@dataclass(slots=True)
class CodexConfigUI:
    group: Any
    executable: Any
    model: Any
    reasoning_effort: Any
    refresh: Any
    status: Any
    catalog_backend: CodexBackend | None = field(default=None, init=False)

    @property
    def save_components(self) -> list[Any]:
        return [self.executable, self.model, self.reasoning_effort]


def create_codex_config_ui(gr, profile: dict[str, Any], *, visible: bool, lock_config: bool) -> CodexConfigUI:
    with gr.Group(visible=visible, elem_classes=["wangp-transparent-group"]) as group:
        gr.Markdown("**Codex setup** — WanGP automatically detects a standalone/npm Codex CLI or the compatible CLI bundled with the Codex VS Code extension. [Install the Codex CLI](https://learn.chatgpt.com/docs/codex/cli) only if neither is available. If sign-in is needed, Deepy displays Codex's secure browser sign-in link in the chat.")
        with gr.Row(elem_classes=["wangp-bottom-aligned-row"]):
            executable = gr.Textbox(value=profile["executable"], label="Codex executable")
            model = gr.Dropdown(choices=codex_model_choices(profile["model_catalog"], profile["model"]), value=profile["model"], allow_custom_value=True, label="Codex model")
            reasoning_effort = gr.Dropdown(choices=codex_reasoning_effort_choices(profile["model_catalog"], profile["model"], profile["reasoning_effort"]), value=profile["reasoning_effort"], allow_custom_value=True, label="Reasoning effort")
            refresh = gr.Button("Refresh", min_width=80, scale=0, interactive=not lock_config)
        status = gr.Markdown()
    return CodexConfigUI(group, executable, model, reasoning_effort, refresh, status)


def bind_codex_config_ui(gr, ui: CodexConfigUI, server_config: dict[str, Any], server_config_filename: str) -> None:
    def refresh_catalog(executable, selected_model, selected_effort):
        executable = str(executable or "codex").strip() or "codex"
        selected_model = normalize_codex_model_selection(selected_model)
        selected_effort = normalize_reasoning_effort_selection(selected_effort)
        backend = ui.catalog_backend
        if backend is None or str(backend.profile.get("executable", "codex") or "codex").strip() != executable:
            if backend is not None:
                backend.close()
            backend = CodexBackend({"executable": executable})
            ui.catalog_backend = backend
        try:
            catalog = backend.list_models()
        except CodexAuthenticationRequired as exc:
            return gr.update(), gr.update(), str(exc)
        except Exception as exc:
            backend.close()
            ui.catalog_backend = None
            return gr.update(), gr.update(), f"**Could not refresh Codex models:** {exc}"
        backend.close()
        ui.catalog_backend = None
        save_model_catalog(server_config, server_config_filename, ENGINE_CODEX, catalog)
        return gr.update(choices=codex_model_choices(catalog, selected_model), value=selected_model), gr.update(choices=codex_reasoning_effort_choices(catalog, selected_model, selected_effort), value=selected_effort), f"**Refreshed and cached {len(catalog)} Codex models.** The selected model and reasoning effort were not changed."

    def update_reasoning_efforts(model, selected_effort):
        model = normalize_codex_model_selection(model)
        selected_effort = normalize_reasoning_effort_selection(selected_effort)
        catalog = cached_model_catalog(server_config, ENGINE_CODEX)
        return gr.update(choices=codex_reasoning_effort_choices(catalog, model, selected_effort), value=selected_effort)

    ui.model.change(fn=update_reasoning_efforts, inputs=[ui.model, ui.reasoning_effort], outputs=[ui.reasoning_effort], show_progress="hidden")
    ui.refresh.click(fn=refresh_catalog, inputs=[ui.executable, ui.model, ui.reasoning_effort], outputs=[ui.model, ui.reasoning_effort, ui.status], show_progress="hidden")


def codex_profile_from_values(executable: Any, model: Any, reasoning_effort: Any, model_catalog: Any) -> dict[str, Any]:
    return {"executable": str(executable or "codex").strip() or "codex", "model": normalize_codex_model_selection(model), "reasoning_effort": normalize_reasoning_effort_selection(reasoning_effort), "model_catalog": model_catalog}


def validate_codex_profile(profile: dict[str, Any]) -> None:
    if not profile["executable"]:
        raise ValueError("Codex executable is required.")
