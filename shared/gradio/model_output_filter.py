import html
import time
from dataclasses import dataclass
from typing import Callable

import gradio as gr

from shared import model_dropdowns


FILTER_ICONS = {
    model_dropdowns.MODEL_OUTPUT_FILTER_ALL: """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7.5 15.5c-2 0-3.5-1.6-3.5-3.5s1.5-3.5 3.5-3.5c3.2 0 5.8 7 9 7 2 0 3.5-1.6 3.5-3.5s-1.5-3.5-3.5-3.5c-3.2 0-5.8 7-9 7Z"/>
        </svg>
    """,
    model_dropdowns.MODEL_OUTPUT_FILTER_VIDEO: """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 8h16v11H4V8Z"/>
            <path d="M4 8 7 4h4L8 8"/>
            <path d="m12 8 3-4h4l-3 4"/>
            <path d="M10 12.5v3l3-1.5-3-1.5Z"/>
        </svg>
    """,
    model_dropdowns.MODEL_OUTPUT_FILTER_IMAGE: """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 5h16v14H4V5Z"/>
            <path d="m6.5 16 4.5-4.5 3 3 2-2 3.5 3.5"/>
            <path d="M15.5 8.5h.01"/>
        </svg>
    """,
    model_dropdowns.MODEL_OUTPUT_FILTER_AUDIO: """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 12h2.5l2-4v8l2.5-8 2 8 2.5-4H20"/>
        </svg>
    """,
}

FILTER_LABELS = {
    model_dropdowns.MODEL_OUTPUT_FILTER_ALL: "All model families",
    model_dropdowns.MODEL_OUTPUT_FILTER_VIDEO: "Movie model families",
    model_dropdowns.MODEL_OUTPUT_FILTER_IMAGE: "Image-only model families",
    model_dropdowns.MODEL_OUTPUT_FILTER_AUDIO: "Audio-only model families",
}


@dataclass
class ModelOutputFilterControls:
    html: gr.HTML
    target: gr.Textbox
    apply_button: gr.Button
    refresh_trigger: gr.Textbox


def render_filter(selected_filter):
    selected_filter = model_dropdowns.normalize_model_output_filter(selected_filter)
    items = []
    for filter_key in (
        model_dropdowns.MODEL_OUTPUT_FILTER_ALL,
        model_dropdowns.MODEL_OUTPUT_FILTER_VIDEO,
        model_dropdowns.MODEL_OUTPUT_FILTER_IMAGE,
        model_dropdowns.MODEL_OUTPUT_FILTER_AUDIO,
    ):
        active = filter_key == selected_filter
        classes = "wangp-model-output-filter-link" + (" wangp-model-output-filter-link-active" if active else "")
        items.append(
            "<button type='button' class='{classes}' data-model-output-filter='{filter_key}' title='{label}' aria-label='{label}' aria-pressed='{pressed}'>{icon}</button>".format(
                classes=classes,
                filter_key=html.escape(filter_key, quote=True),
                label=html.escape(FILTER_LABELS[filter_key], quote=True),
                pressed=str(active).lower(),
                icon=FILTER_ICONS[filter_key],
            )
        )
    return "<div class='wangp-model-output-filter' role='group' aria-label='Model family filter'>{}</div>".format("".join(items))


def create_filter(selected_filter):
    with gr.Column(scale=2, min_width=210, elem_classes=["wangp-model-output-filter-wrap"]):
        filter_html = gr.HTML(value=render_filter(selected_filter), elem_id="wangp_model_output_filter")
        with gr.Row(visible=False, elem_classes=["wangp-model-output-filter-hidden-controls"]):
            target = gr.Textbox(value="", show_label=False, elem_id="wangp_model_output_filter_target")
            apply_button = gr.Button("Apply model output filter", elem_id="wangp_model_output_filter_apply")
            refresh_trigger = gr.Textbox(value="", show_label=False, elem_id="wangp_model_output_filter_refresh")
    return ModelOutputFilterControls(filter_html, target, apply_button, refresh_trigger)


def apply_filter_selection(deps, state, requested_filter, current_model_type_getter):
    selected_filter = model_dropdowns.normalize_model_output_filter(requested_filter)
    if isinstance(state, dict):
        state[model_dropdowns.MODEL_OUTPUT_FILTER_CONFIG_KEY] = selected_filter
    current_model_type = current_model_type_getter(state)
    selected_model_type = model_dropdowns.select_model_for_output_filter(deps, state, current_model_type, selected_filter)
    model_dropdowns.debug_model_selector_event("filter.click", requested_filter=selected_filter, current_model=current_model_type, target=selected_model_type)
    return f"{selected_model_type}|{time.time()}"


def refresh_for_model_target(deps, state, model_type):
    before_filter = model_dropdowns.get_model_output_filter(deps, state)
    selected_filter, changed = model_dropdowns.reconcile_model_output_filter_for_model_type(deps, state, model_type)
    model_dropdowns.debug_model_selector_event("filter.refresh_for_target", target=model_type, before_filter=before_filter, after_filter=selected_filter, changed=changed)
    return f"{selected_filter}|{time.time()}" if changed else gr.update()


def render_filter_from_trigger(refresh_value):
    selected_filter = str(refresh_value or "").split("|", 1)[0]
    model_dropdowns.debug_model_selector_event("filter.render_trigger", trigger=refresh_value, selected_filter=selected_filter)
    return gr.update(value=render_filter(selected_filter)) if len(selected_filter) > 0 else gr.update()


def bind_filter(controls: ModelOutputFilterControls, *, deps_factory: Callable, state, model_choice_target, current_model_type_getter: Callable):
    controls.apply_button.click(
        fn=lambda state_value, requested_filter: apply_filter_selection(deps_factory(), state_value, requested_filter, current_model_type_getter),
        inputs=[state, controls.target],
        outputs=[model_choice_target],
        trigger_mode="always_last",
        show_progress="hidden",
    )
    controls.refresh_trigger.change(
        fn=render_filter_from_trigger,
        inputs=[controls.refresh_trigger],
        outputs=[controls.html],
        trigger_mode="always_last",
        show_progress="hidden",
    )


def get_javascript():
    return r"""
    (function () {
        function root() {
            if (window.gradioApp) return window.gradioApp();
            const app = document.querySelector("gradio-app");
            return app ? (app.shadowRoot || app) : document;
        }

        function targetInput() {
            return root().querySelector("#wangp_model_output_filter_target textarea, #wangp_model_output_filter_target input");
        }

        function applyButton() {
            const el = root().querySelector("#wangp_model_output_filter_apply");
            return el?.matches("button") ? el : el?.querySelector("button");
        }

        function selectFilterLink(link) {
            const area = link.closest(".wangp-model-output-filter");
            if (!area) return;
            area.querySelectorAll("[data-model-output-filter]").forEach((item) => {
                const active = item === link;
                item.classList.toggle("wangp-model-output-filter-link-active", active);
                item.setAttribute("aria-pressed", active ? "true" : "false");
            });
        }

        function eventFilterLink(event) {
            const path = event.composedPath ? event.composedPath() : [];
            for (const node of path) {
                if (node?.matches?.("#wangp_model_output_filter [data-model-output-filter]")) return node;
                const closest = node?.closest?.("#wangp_model_output_filter [data-model-output-filter]");
                if (closest) return closest;
            }
            return event.target?.closest?.("#wangp_model_output_filter [data-model-output-filter]");
        }

        document.addEventListener("click", (event) => {
            const link = eventFilterLink(event);
            if (!link) return;
            event.preventDefault();
            if (link.getAttribute("aria-pressed") === "true") return;
            const target = targetInput();
            const button = applyButton();
            if (!target || !button) return;
            selectFilterLink(link);
            link.blur();
            target.value = link.dataset.modelOutputFilter || "all";
            target.dispatchEvent(new Event("input", { bubbles: true }));
            button.click();
        });
    })();
    """


def get_css():
    return """
    .wangp-model-output-filter-wrap {
        position: relative;
        align-self: center;
        width: 100%;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    #wangp_model_output_filter {
        position: relative;
        z-index: 2;
        width: 100%;
        min-width: 0 !important;
        margin: 0 auto !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    #wangp_model_output_filter > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    .wangp-model-output-filter {
        display: grid;
        grid-template-columns: repeat(4, minmax(30px, 42px));
        align-items: center;
        justify-content: center;
        width: min(100%, 196px);
        margin: 0 auto;
        gap: clamp(2px, 1.2vw, 6px);
        padding: 0 4px;
        background: transparent;
        border-radius: 6px;
        box-sizing: border-box;
    }
    .wangp-model-output-filter-link {
        appearance: none;
        border: 0;
        background: transparent;
        box-shadow: none;
        color: var(--neutral-500, #858585);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 30px;
        width: 100%;
        max-width: 42px;
        height: 32px;
        padding: 0;
        line-height: 1;
        opacity: 0.58;
        outline: none !important;
        -webkit-tap-highlight-color: transparent;
        user-select: none;
    }
    .wangp-model-output-filter-link svg {
        display: block;
        width: 25px;
        height: 25px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.8;
        stroke-linecap: round;
        stroke-linejoin: round;
        filter: drop-shadow(0 0 2px var(--body-background-fill));
    }
    .wangp-model-output-filter-link:hover {
        background: transparent !important;
        box-shadow: none !important;
        color: var(--body-text-color) !important;
        opacity: 0.96;
        text-decoration: underline;
        text-underline-offset: 3px;
    }
    .wangp-model-output-filter-link:focus,
    .wangp-model-output-filter-link:focus-visible,
    .wangp-model-output-filter-link:active {
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .wangp-model-output-filter-link-active {
        color: var(--button-primary-background-fill) !important;
        opacity: 1;
        font-weight: 700;
    }
    .wangp-model-output-filter-link-active svg {
        stroke-width: 2.35;
    }
    .wangp-model-output-filter-hidden-controls {
        display: none !important;
    }
    """
