from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from . import common
from . import constants as ui_constants
from . import frame_planning as frames
from . import process_catalog as catalog


@dataclass(frozen=True)
class ProcessFormState:
    process_model_type: str
    process_name: str
    source_path: str
    process_strength: float
    output_path: str
    prompt: str
    continue_enabled: bool
    source_audio_track: str
    output_resolution: str
    target_ratio: str
    chunk_size_seconds: float
    sliding_window_overlap: int
    start_seconds: str
    end_seconds: str

    def to_dict(self) -> dict:
        return {
            "process_model_type": self.process_model_type,
            "process_name": self.process_name,
            "source_path": self.source_path,
            "process_strength": self.process_strength,
            "output_path": self.output_path,
            "prompt": self.prompt,
            "continue_enabled": self.continue_enabled,
            "source_audio_track": self.source_audio_track,
            "output_resolution": self.output_resolution,
            "target_ratio": self.target_ratio,
            "chunk_size_seconds": self.chunk_size_seconds,
            "sliding_window_overlap": self.sliding_window_overlap,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


@dataclass(frozen=True)
class FormComponentValues:
    source_path: object
    process_strength: object
    output_path: object
    prompt_text: object
    continue_enabled: object
    source_audio_track: object
    output_resolution: object
    target_ratio: object
    chunk_size_seconds: object
    sliding_window_overlap: object
    start_seconds: object
    end_seconds: object

    def to_raw_state(self) -> dict:
        return {
            "source_path": self.source_path,
            "process_strength": self.process_strength,
            "output_path": self.output_path,
            "prompt": self.prompt_text,
            "continue_enabled": self.continue_enabled,
            "source_audio_track": self.source_audio_track,
            "output_resolution": self.output_resolution,
            "target_ratio": self.target_ratio,
            "chunk_size_seconds": self.chunk_size_seconds,
            "sliding_window_overlap": self.sliding_window_overlap,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


@dataclass(frozen=True)
class InitialFormPatch:
    model_type_choices: list[tuple[str, str]]
    process_choices: list[tuple[str, str]]
    model_type: str
    process_name: str
    form_state: ProcessFormState
    overlap_step: int
    overlap_max: int
    process_strength_visible: bool
    target_ratio_visible: bool


RESTORED_FORM_OUTPUT_COUNT = 12


def skipped_restored_form_outputs() -> tuple:
    return tuple(gr.skip() for _ in range(RESTORED_FORM_OUTPUT_COUNT))


class ProcessFormController:
    def __init__(
        self,
        *,
        library,
        get_model_def,
        output_resolution_values: set[str],
        source_audio_track_values: set[str],
        ratio_values: set[str],
        default_model_type: str | None = None,
    ) -> None:
        self.library = library
        self.get_model_def = get_model_def
        self.output_resolution_values = output_resolution_values
        self.source_audio_track_values = source_audio_track_values
        self.ratio_values = ratio_values
        self.default_model_type = default_model_type or catalog.DEFAULT_MODEL_TYPE

    @staticmethod
    def _process_strength_slider_bounds(value: float) -> tuple[float, float]:
        lower = value if value < 0.0 else 0.0
        upper = value if value > 3.0 else 3.0
        return lower, upper

    @staticmethod
    def _fit_overlap_slider_value(value: int, maximum: int) -> int:
        if value < 1:
            return 1
        if value > maximum:
            return maximum
        return value

    def build_initial_form(self, saved_ui_settings: dict, main_state: dict | None, initial_user_refs: list[str]) -> InitialFormPatch:
        saved_process_name = str(saved_ui_settings.get("process_name") or "").strip()
        saved_model_type = str(saved_ui_settings.get("process_model_type") or "").strip()
        saved_process_definition = self.library.process_definition(saved_process_name, main_state, initial_user_refs)
        if saved_process_definition is not None:
            saved_model_type = self.library.process_definition_model_type(saved_process_definition) or saved_model_type
        process_names_by_model_type = self.library.process_values_by_model_type(initial_user_refs)
        default_model_type = (
            saved_model_type
            if saved_model_type in process_names_by_model_type
            else catalog.DEFAULT_MODEL_TYPE
            if catalog.DEFAULT_MODEL_TYPE in process_names_by_model_type
            else next(iter(process_names_by_model_type), catalog.DEFAULT_MODEL_TYPE)
        )
        default_process_choices = self.library.normal_process_choices(default_model_type, initial_user_refs)
        default_process_values = [value for _label, value in default_process_choices]
        default_process_name = (
            saved_process_name
            if saved_process_name in default_process_values
            else catalog.DEFAULT_PROCESS_NAME
            if catalog.DEFAULT_PROCESS_NAME in default_process_values
            else (default_process_values[0] if default_process_values else catalog.DEFAULT_PROCESS_NAME)
        )
        form_state = self.build_form_state(default_process_name, saved_ui_settings, main_state, initial_user_refs)
        self.default_model_type = default_model_type
        return InitialFormPatch(
            model_type_choices=self.library.model_type_choices(initial_user_refs),
            process_choices=default_process_choices,
            model_type=default_model_type,
            process_name=default_process_name,
            form_state=form_state,
            overlap_step=frames.get_vae_temporal_latent_size(default_model_type, self.get_model_def),
            overlap_max=frames.get_overlap_slider_max(default_model_type, self.get_model_def),
            process_strength_visible=self.library.is_process_strength_visible(default_process_name, main_state, initial_user_refs),
            target_ratio_visible=self.library.has_process_outpaint(default_process_name, main_state, initial_user_refs),
        )

    def user_settings_hint_update(self, process_choices: list[tuple[str, str]]):
        return gr.update(visible=self.library.process_choices_have_user_settings(process_choices))

    @staticmethod
    def settings_action_updates(process_model_type_value: str, process_value: str) -> tuple:
        add_visible = process_model_type_value == ui_constants.ADD_USER_SETTINGS_MODEL_TYPE
        delete_visible = catalog.is_user_process_value(process_value)
        placeholder_visible = not add_visible and not delete_visible
        return gr.update(visible=add_visible), gr.update(visible=delete_visible), gr.update(visible=placeholder_visible)

    def target_ratio_update(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, target_ratio: str | None = None):
        visible = self.library.has_process_outpaint(process_name, main_state, user_refs)
        return gr.update(value=target_ratio if visible else "", visible=visible, choices=ui_constants.RATIO_CHOICES if visible else ui_constants.RATIO_CHOICES_WITH_EMPTY)

    def process_strength_update(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, process_strength: float | None = None):
        process_definition = self.library.process_definition(process_name, main_state, user_refs)
        visible = self.library.is_process_strength_visible(process_name, main_state, user_refs)
        default_value = common.get_default_process_strength((process_definition or {}).get("settings", {}))
        if isinstance(process_definition, dict) and process_definition.get("source") == "user":
            user_default = self.library.user_lora_strength_override_default(process_definition)
            if user_default is not None:
                default_value = user_default
        value = common.coerce_float(process_strength, default_value) if visible else default_value
        minimum, maximum = self._process_strength_slider_bounds(value)
        return gr.update(value=value, visible=visible, minimum=minimum, maximum=maximum)

    def overlap_control_updates(self, process_name: str, main_state: dict | None, user_refs: list[str] | None):
        process_definition = self.library.process_definition_or_default(process_name, main_state, user_refs)
        settings = process_definition.get("settings", {})
        model_type = str(settings.get("model_type") or "")
        step = frames.get_vae_temporal_latent_size(model_type, self.get_model_def)
        maximum = frames.get_overlap_slider_max(model_type, self.get_model_def)
        value = common.coerce_int(settings.get("sliding_window_overlap"), 1, minimum=1)
        value = self._fit_overlap_slider_value(frames.normalize_overlap_frames(value, frame_step=step), maximum)
        return gr.update(minimum=1, maximum=maximum, step=step, value=value)

    def build_form_state(self, process_name: str, raw_state: dict | None = None, main_state: dict | None = None, user_refs: list[str] | None = None) -> ProcessFormState:
        process_definition = self.library.process_definition_or_default(process_name, main_state, user_refs)
        process_settings = process_definition.get("settings", {})
        model_type = str(process_settings.get("model_type") or catalog.DEFAULT_MODEL_TYPE)
        step = frames.get_vae_temporal_latent_size(model_type, self.get_model_def)
        maximum = frames.get_overlap_slider_max(model_type, self.get_model_def)
        raw_state = raw_state if isinstance(raw_state, dict) else {}

        default_strength = common.get_default_process_strength(process_settings)
        saved_process_strength = raw_state.get("process_strength", raw_state.get("control_video_strength"))
        process_strength = default_strength if saved_process_strength is None else common.coerce_float(saved_process_strength, default_strength)
        source_audio_track = str(raw_state.get("source_audio_track") or "").strip()
        output_resolution = str(raw_state.get("output_resolution") or "").strip()
        target_ratio = str(raw_state.get("target_ratio") or "4:3").strip()
        overlap_default = self._fit_overlap_slider_value(frames.normalize_overlap_frames(common.coerce_int(process_settings.get("sliding_window_overlap"), 1, minimum=1), frame_step=step), maximum)
        overlap_value = common.coerce_int(raw_state.get("sliding_window_overlap"), overlap_default, minimum=1)
        sliding_window_overlap = self._fit_overlap_slider_value(frames.normalize_overlap_frames(overlap_value, frame_step=step), maximum)

        return ProcessFormState(
            process_model_type=model_type,
            process_name=process_name,
            source_path=str(raw_state.get("source_path") or ui_constants.DEFAULT_SOURCE_PATH),
            process_strength=process_strength,
            output_path=str(raw_state.get("output_path") or ui_constants.DEFAULT_OUTPUT_PATH),
            prompt=str(raw_state.get("prompt") or "") if "prompt" in raw_state else str(process_settings.get("prompt") or ""),
            continue_enabled=common.coerce_bool(raw_state.get("continue_enabled"), True),
            source_audio_track=source_audio_track if source_audio_track in self.source_audio_track_values else "",
            output_resolution=output_resolution if output_resolution in self.output_resolution_values else "720p",
            target_ratio=target_ratio if target_ratio in self.ratio_values else "4:3",
            chunk_size_seconds=common.coerce_float(raw_state.get("chunk_size_seconds"), 10.0, minimum=0.1),
            sliding_window_overlap=sliding_window_overlap,
            start_seconds="" if raw_state.get("start_seconds") in (None, "") else str(raw_state.get("start_seconds")),
            end_seconds="" if raw_state.get("end_seconds") in (None, "") else str(raw_state.get("end_seconds")),
        )

    def build_state(self, process_name: str, raw_state: dict | None = None, main_state: dict | None = None, user_refs: list[str] | None = None) -> dict:
        return self.build_form_state(process_name, raw_state, main_state, user_refs).to_dict()

    def snapshot_state(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, values: FormComponentValues) -> dict:
        return self.build_state(process_name, values.to_raw_state(), main_state, user_refs)

    def store_memory(self, memory_state: dict | None, current_process_name: str, main_state: dict | None, user_refs: list[str] | None, values: FormComponentValues):
        updated_memory = dict(memory_state) if isinstance(memory_state, dict) else {}
        current_process_name = str(current_process_name or "").strip()
        if self.library.process_definition(current_process_name, main_state, user_refs) is not None:
            updated_memory[current_process_name] = self.snapshot_state(current_process_name, main_state, user_refs, values)
        return updated_memory

    def restore_state(self, memory_state: dict | None, process_name: str, current_source_path: str, main_state: dict | None, user_refs: list[str] | None) -> tuple:
        state = self.build_form_state(process_name, (memory_state or {}).get(process_name), main_state, user_refs)
        source_path_value = current_source_path.strip() or state.source_path
        return (
            source_path_value,
            self.process_strength_update(process_name, main_state, user_refs, state.process_strength),
            state.output_path,
            state.prompt,
            state.continue_enabled,
            state.source_audio_track,
            state.output_resolution,
            self.target_ratio_update(process_name, main_state, user_refs, state.target_ratio),
            state.chunk_size_seconds,
            self.overlap_control_updates(process_name, main_state, user_refs),
            state.start_seconds,
            state.end_seconds,
        )

    def change_process_model_type(self, memory_state: dict | None, current_process_name: str, next_model_type: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None, values: FormComponentValues) -> tuple:
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        updated_memory = self.store_memory(memory_state, current_process_name, main_state, refs, values)
        process_choices, next_process_name = self.library.process_choices(next_model_type, main_state, main_lset_name, refs)
        next_process_name = str(next_process_name or ui_constants.NO_USER_SETTINGS_VALUE).strip()
        return (
            updated_memory,
            next_process_name,
            gr.update(choices=process_choices, value=next_process_name),
            self.user_settings_hint_update(process_choices),
            *self.settings_action_updates(next_model_type, next_process_name),
            *self.restore_state(updated_memory, next_process_name, values.source_path, main_state, refs),
        )

    def change_process_name(self, memory_state: dict | None, current_process_name: str, next_process_name: str, process_model_type_value: str, main_state: dict | None, user_refs: list[str] | None, values: FormComponentValues) -> tuple:
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        updated_memory = self.store_memory(memory_state, current_process_name, main_state, refs, values)
        next_process_name = str(next_process_name or "").strip()
        _add_update, delete_update, placeholder_update = self.settings_action_updates(process_model_type_value, next_process_name)
        return (
            updated_memory,
            next_process_name,
            delete_update,
            placeholder_update,
            *self.restore_state(updated_memory, next_process_name, values.source_path, main_state, refs),
        )

    def refresh_from_main(self, _refresh_id, memory_state: dict | None, current_process_name: str, process_model_type_value: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None, values: FormComponentValues) -> tuple:
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        model_choices = self.library.model_type_choices(refs)
        process_model_type_value = str(process_model_type_value or "").strip()
        if process_model_type_value != ui_constants.ADD_USER_SETTINGS_MODEL_TYPE:
            valid_model_values = {value for _label, value in model_choices}
            model_value = process_model_type_value if process_model_type_value in valid_model_values else self.default_model_type
            return (
                gr.update(choices=model_choices, value=model_value),
                gr.update(),
                gr.update(),
                memory_state,
                current_process_name,
                *self.settings_action_updates(model_value, current_process_name),
                *skipped_restored_form_outputs(),
            )
        updated_memory = self.store_memory(memory_state, current_process_name, main_state, refs, values)
        process_choices, next_process_name = self.library.current_user_settings_choices(main_state, main_lset_name)
        return (
            gr.update(choices=model_choices, value=ui_constants.ADD_USER_SETTINGS_MODEL_TYPE),
            gr.update(choices=process_choices, value=next_process_name),
            self.user_settings_hint_update(process_choices),
            updated_memory,
            next_process_name,
            *self.settings_action_updates(ui_constants.ADD_USER_SETTINGS_MODEL_TYPE, next_process_name),
            *self.restore_state(updated_memory, next_process_name, values.source_path, main_state, refs),
        )
