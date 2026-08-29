from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .claude_backend import CLAUDE_AUTH_DOCS_URL, CLAUDE_SDK_INSTALL_SPEC, ClaudeBackend
from .config import ENGINE_CLAUDE, claude_model_choices, claude_reasoning_effort_choices, normalize_claude_model_selection, normalize_reasoning_effort_selection
from .config_ui_common import cached_model_catalog, save_model_catalog


@dataclass(slots=True)
class ClaudeConfigUI:
    group: Any
    executable: Any
    model: Any
    reasoning_effort: Any
    refresh: Any
    status: Any

    @property
    def save_components(self) -> list[Any]:
        return [self.executable, self.model, self.reasoning_effort]


def create_claude_config_ui(gr, profile: dict[str, Any], *, visible: bool, lock_config: bool) -> ClaudeConfigUI:
    with gr.Group(visible=visible, elem_classes=["wangp-transparent-group"]) as group:
        gr.Markdown(f"**Claude Code setup** — install the compatible Python bridge with `pip install {CLAUDE_SDK_INSTALL_SPEC}`. Do not use an unpinned SDK upgrade because current releases replace WanGP's MCP/Pydantic stack. WanGP automatically reuses Claude Code from PATH or compatible VS Code, Cursor, Windsurf, and VS Code Insiders extensions. Authenticate Claude Code once before using it in WanGP. [Authentication help]({CLAUDE_AUTH_DOCS_URL})")
        with gr.Row(elem_classes=["wangp-bottom-aligned-row"]):
            executable = gr.Textbox(value=profile["executable"], label="Claude executable (`claude` = auto-detect)")
            model = gr.Dropdown(choices=claude_model_choices(profile["model_catalog"], profile["model"]), value=profile["model"], allow_custom_value=True, label="Claude model")
            reasoning_effort = gr.Dropdown(choices=claude_reasoning_effort_choices(profile["model_catalog"], profile["model"], profile["reasoning_effort"]), value=profile["reasoning_effort"], allow_custom_value=True, label="Reasoning effort")
            refresh = gr.Button("Refresh", min_width=80, scale=0, interactive=not lock_config)
        status = gr.Markdown()
    return ClaudeConfigUI(group, executable, model, reasoning_effort, refresh, status)


def bind_claude_config_ui(gr, ui: ClaudeConfigUI, server_config: dict[str, Any], server_config_filename: str) -> None:
    def refresh_catalog(executable, selected_model, selected_effort):
        executable = str(executable or "claude").strip() or "claude"
        selected_model = normalize_claude_model_selection(selected_model)
        selected_effort = normalize_reasoning_effort_selection(selected_effort)
        backend = ClaudeBackend({"executable": executable})
        try:
            catalog = backend.list_models()
        except Exception as exc:
            return gr.update(), gr.update(), f"**Could not refresh Claude models:** {exc}"
        finally:
            backend.close()
        save_model_catalog(server_config, server_config_filename, ENGINE_CLAUDE, catalog)
        return gr.update(choices=claude_model_choices(catalog, selected_model), value=selected_model), gr.update(choices=claude_reasoning_effort_choices(catalog, selected_model, selected_effort), value=selected_effort), f"**Refreshed and cached {len(catalog)} Claude models.** The selected model and reasoning effort were not changed."

    def update_reasoning_efforts(model, selected_effort):
        model = normalize_claude_model_selection(model)
        selected_effort = normalize_reasoning_effort_selection(selected_effort)
        catalog = cached_model_catalog(server_config, ENGINE_CLAUDE)
        return gr.update(choices=claude_reasoning_effort_choices(catalog, model, selected_effort), value=selected_effort)

    ui.model.change(fn=update_reasoning_efforts, inputs=[ui.model, ui.reasoning_effort], outputs=[ui.reasoning_effort], show_progress="hidden")
    ui.refresh.click(fn=refresh_catalog, inputs=[ui.executable, ui.model, ui.reasoning_effort], outputs=[ui.model, ui.reasoning_effort, ui.status], show_progress="hidden")


def claude_profile_from_values(executable: Any, model: Any, reasoning_effort: Any, model_catalog: Any) -> dict[str, Any]:
    return {"executable": str(executable or "claude").strip() or "claude", "model": normalize_claude_model_selection(model), "reasoning_effort": normalize_reasoning_effort_selection(reasoning_effort), "model_catalog": model_catalog}


def validate_claude_profile(profile: dict[str, Any]) -> None:
    if not profile["executable"]:
        raise ValueError("Claude Code executable is required.")
