from typing import Any, Callable

from postprocessing.audio_processors import (
    AUDIO_PROCESSOR_TYPE_SOUNDTRACK,
    MMAUDIO_METHOD,
    method_metadata,
    read_config_section,
)


class MMAudioProcessor:
    MODE_OFF = 0
    PERSIST_UNLOAD = 1
    PERSIST_RAM = 2

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def query_audio_processor_def(cls) -> dict[str, Any]:
        return {
            "name": "MMAudio",
            "processor_types": (AUDIO_PROCESSOR_TYPE_SOUNDTRACK,),
            "methods": [("MMAudio (generate Audio Based on Video Content)", MMAUDIO_METHOD)],
            "method_types": {MMAUDIO_METHOD: (AUDIO_PROCESSOR_TYPE_SOUNDTRACK,)},
            "needs_prompt": {MMAUDIO_METHOD: True},
            "needs_negative_prompt": {MMAUDIO_METHOD: True},
            "supports_repeat": {MMAUDIO_METHOD: True},
            "status": {MMAUDIO_METHOD: "MMAudio Soundtrack Generation"},
            "config_key": "mmaudio",
            "pos": 20,
        }

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        from postprocessing.mmaudio import MMAUDIO_DEFAULT_MODE

        return {"mode": MMAUDIO_DEFAULT_MODE, "persistence": cls.PERSIST_UNLOAD}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        from postprocessing.mmaudio import MMAUDIO_MODE_NEW, MMAUDIO_MODE_V2

        from postprocessing.mmaudio import MMAUDIO_DEFAULT_MODE

        mode = config.get("mode", MMAUDIO_DEFAULT_MODE)
        persistence = config.get("persistence", cls.PERSIST_UNLOAD)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = MMAUDIO_DEFAULT_MODE
        try:
            persistence = int(persistence)
        except (TypeError, ValueError):
            persistence = cls.PERSIST_UNLOAD
        if mode not in (MMAUDIO_MODE_V2, MMAUDIO_MODE_NEW):
            mode = MMAUDIO_DEFAULT_MODE
        if persistence not in (cls.PERSIST_UNLOAD, cls.PERSIST_RAM):
            persistence = cls.PERSIST_UNLOAD
        return {"mode": mode, "persistence": persistence}

    def config(self, server_config: dict[str, Any] | None = None) -> dict[str, Any]:
        return read_config_section(self.server_config if server_config is None else server_config, self)

    def enabled(self) -> bool:
        return True

    def _settings(self, config: dict[str, Any] | None = None) -> tuple[bool, int, int, str | None, str | None]:
        from postprocessing.mmaudio import MMAUDIO_ALTERNATE, MMAUDIO_MODE_NEW, MMAUDIO_MODE_V2, MMAUDIO_STANDARD

        values = self.config() if config is None else self.normalize_config_section(config)
        mode, persistence = int(values["mode"]), int(values["persistence"])
        if mode == MMAUDIO_MODE_V2:
            return True, mode, persistence, "large_44k_v2", MMAUDIO_STANDARD
        if mode == MMAUDIO_MODE_NEW:
            return True, mode, persistence, "large_44k", MMAUDIO_ALTERNATE
        return False, mode, persistence, None, None

    def validate_method(self, method, video_source=None, frames_count=None, fps=None, duration=None, **_kwargs) -> str:
        enabled, _, _, _, _ = self._settings()
        if not enabled:
            return "MMAudio is disabled in Configuration > Extensions"
        if video_source is not None:
            from shared.utils.utils import get_video_info

            fps, _, _, frames_count = get_video_info(video_source)
        elif duration is not None:
            try:
                return "" if float(duration) >= 1 else "MMAudio can generate an Audio track only if the Video is at least 1s long"
            except (TypeError, ValueError):
                return ""
        if frames_count is None or fps is None:
            return ""
        return "" if frames_count >= round(float(fps)) else "MMAudio can generate an Audio track only if the Video is at least 1s long"

    def query_download_defs(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        from postprocessing.mmaudio import MMAUDIO_ALTERNATE, MMAUDIO_MODE_V2, MMAUDIO_STANDARD

        enabled, mode, _, _, _ = self._settings()
        if enabled_only and not enabled:
            return []
        model_files = [MMAUDIO_STANDARD if mode == MMAUDIO_MODE_V2 else MMAUDIO_ALTERNATE] if enabled_only else [MMAUDIO_STANDARD, MMAUDIO_ALTERNATE]
        return [{
            "repoId": "DeepBeepMeep/Wan2.1",
            "sourceFolderList": ["mmaudio", "DFN5B-CLIP-ViT-H-14-378", "bigvgan_v2_44khz_128band_512x"],
            "fileList": [["synchformer_state_dict.pth", "v1-44.pth", *model_files], ["open_clip_config.json", "open_clip_pytorch_model.bin"], ["config.json", "bigvgan_generator.pt"]],
        }]

    def download(self, method, process_files: Callable[..., Any], send_cmd=None, status_text=None, **_kwargs) -> bool:
        from shared.utils.download import process_files_def_if_needed

        defs = self.query_download_defs(enabled_only=True)
        return process_files_def_if_needed(defs[0] if defs else None, send_cmd=send_cmd, status_text=status_text or "Downloading MMAudio model files...")

    def generate_soundtrack(self, method, video_path, prompt="", negative_prompt="", seed=-1, duration=0, output_path=None, send_cmd=None, status_callback=None, verbose_level=1, audio_codec_key="aac_128", **_kwargs) -> str:
        enabled, _, persistence, model_name, model_path = self._settings()
        if not enabled:
            raise RuntimeError("MMAudio is disabled in Configuration > Extensions")
        if status_callback is not None:
            status_callback(method_metadata(method)["status"])
        from postprocessing.mmaudio.mmaudio import video_to_audio

        video_to_audio(
            video_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_steps=25,
            cfg_strength=4.5,
            duration=duration,
            save_path=output_path,
            persistent_models=persistence == self.PERSIST_RAM,
            audio_file_only=True,
            verboseLevel=verbose_level,
            model_name=model_name,
            model_path=model_path,
            audio_codec_key=audio_codec_key,
        )
        return output_path

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        from postprocessing.mmaudio import MMAUDIO_DEFAULT_MODE, MMAUDIO_MODE_CHOICES

        config = self.normalize_config_section(config)
        mode = gr.Dropdown(choices=MMAUDIO_MODE_CHOICES, value=config.get("mode", MMAUDIO_DEFAULT_MODE), label="MMAudio Soundtrack Generation (requires 10GB extra download)", interactive=not lock_config)
        persistence = gr.Dropdown(choices=[("Unload after use", self.PERSIST_UNLOAD), ("Persistent in RAM", self.PERSIST_RAM)], value=config.get("persistence", self.PERSIST_UNLOAD), label="MMAudio Model Persistence", interactive=not lock_config)
        return [("mode", mode), ("persistence", persistence)]

    def release_vram(self) -> None:
        try:
            from postprocessing.mmaudio.mmaudio import release_models

            release_models()
        except ImportError:
            pass
