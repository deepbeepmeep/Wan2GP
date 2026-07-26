from __future__ import annotations

import os
from typing import Any, Callable

import torch

from postprocessing.spatial_upsamplers import SimpleScaleSuffixMixin, UPSAMPLER_TYPE_POSTPROCESSING


class ChainOfZoomBridge(SimpleScaleSuffixMixin):
    """Chain-of-Zoom extreme super-resolution (https://github.com/bryanswkim/Chain-of-Zoom).

    Autoregressive chain of x2/x4 one-step SD3 (OSEDiff) super-resolution steps with
    multi-scale aware tile prompts extracted by a Qwen2.5-VL-3B VLM.
    """

    UPSAMPLING_VALUE_PREFIX = "coz"
    MULTIPLIERS = (2.0, 4.0, 8.0, 16.0)
    PERSIST_UNLOAD = 1
    PERSIST_RAM = 2
    batch_image_inputs = False
    uses_image_profile = True
    PERSISTENCE_CHOICES = [("Unload after use", PERSIST_UNLOAD), ("Persistent in RAM", PERSIST_RAM)]
    PARALLEL_TILE_PROCESSING_CHOICES = [("Auto", "auto"), ("2", 2), ("4", 4), ("8", 8)]

    # all Chain-of-Zoom specific files live flat in a single folder (two-level ckpts layout);
    # the VLM config/processor/tokenizer files are the Qwen2.5-VL ones
    COZ_FOLDER = "chain_of_zoom"
    TRANSFORMER_FILENAME = "CoZ_sd3_medium_srlora_bf16.safetensors"
    VAE_FILENAME = "CoZ_sd3_vae_srlora_bf16.safetensors"
    VLM_FILENAME = "CoZ_qwen2.5-vl-3B_srprompt_bf16.safetensors"
    VLM_QUANTIZED_FILENAME = "CoZ_qwen2.5-vl-3B_srprompt_quanto_bf16_int8.safetensors"
    VLM_EXTRA_FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    # SD3's CLIP-L weights differ from the shared clip_vit_large model.safetensors; only the tokenizer is shared.
    # SD3's CLIP-G uses the same CLIP vocab (pad token "!"), so it ships without tokenizer files.
    CLIP_L_FOLDER = "clip_vit_large_text_with_projection"
    CLIP_L_FILENAME = "clip_vit_large_text_with_projection_bf16.safetensors"
    CLIP_G_FOLDER = "clip_vit_bigG_14_text"
    CLIP_G_FILENAME = "clip_vit_bigG_14_text_bf16.safetensors"
    CLIP_CONFIG_FILENAME = "config.json"
    CLIP_TOKENIZER_FOLDER = "clip_vit_large_patch14"
    CLIP_TOKENIZER_FILES = ["merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
    T5_FOLDER = "T5_xxl_1.1"
    T5_FILENAME_BF16 = "T5_xxl_1.1_enc_bf16.safetensors"
    T5_FILENAME_INT8 = "T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"
    T5_EXTRA_FILES = ["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"]

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"persistence": cls.PERSIST_UNLOAD, "parallel_tile_processing": "auto"}

    @classmethod
    def legacy_config_keys(cls) -> tuple[str, ...]:
        return ("coz_persistence",)

    @classmethod
    def legacy_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return {"persistence": config.get("coz_persistence", cls.PERSIST_UNLOAD)}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        if config:
            normalized["persistence"] = config.get("persistence", normalized["persistence"])
            normalized["parallel_tile_processing"] = config.get("parallel_tile_processing", normalized["parallel_tile_processing"])
        try:
            normalized["persistence"] = int(normalized.get("persistence", cls.PERSIST_UNLOAD))
        except (TypeError, ValueError):
            normalized["persistence"] = cls.PERSIST_UNLOAD
        if normalized["persistence"] not in (cls.PERSIST_UNLOAD, cls.PERSIST_RAM):
            normalized["persistence"] = cls.PERSIST_UNLOAD
        if str(normalized["parallel_tile_processing"]).strip().lower() == "auto":
            normalized["parallel_tile_processing"] = "auto"
        else:
            try:
                normalized["parallel_tile_processing"] = int(normalized["parallel_tile_processing"])
            except (TypeError, ValueError):
                normalized["parallel_tile_processing"] = "auto"
            if normalized["parallel_tile_processing"] not in (2, 4, 8):
                normalized["parallel_tile_processing"] = "auto"
        return normalized

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api

        return upsampler_api.read_config_section(self.server_config, self)

    @staticmethod
    def _auto_parallel_tile_processing() -> int:
        if not torch.cuda.is_available():
            return 2
        total_vram_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
        if total_vram_gb <= 8:
            return 2
        if total_vram_gb <= 16:
            return 4
        return 8

    def parallel_tile_processing(self) -> int:
        value = self.config()["parallel_tile_processing"]
        return self._auto_parallel_tile_processing() if value == "auto" else int(value)

    def persistent_models(self) -> bool:
        return int(self.config()["persistence"] or self.PERSIST_UNLOAD) == self.PERSIST_RAM

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "Chain-of-Zoom",
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("image",),
            "profile": "image",
            "config_key": "chain_of_zoom",
            "pos": 50,
            "method_pos": {cls.UPSAMPLING_VALUE_PREFIX: 50},
            "methods": [("Chain-of-Zoom", cls.UPSAMPLING_VALUE_PREFIX)],
            "vae_methods": [],
            "multipliers": {cls.UPSAMPLING_VALUE_PREFIX: cls.MULTIPLIERS},
            "default_spatial_upsampling": f"{cls.UPSAMPLING_VALUE_PREFIX}4",
        }

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Group():
            with gr.Row():
                persistence = gr.Dropdown(choices=self.PERSISTENCE_CHOICES, value=config["persistence"], label="Chain-of-Zoom Model Persistence", interactive=not lock_config)
                parallel_tile_processing = gr.Dropdown(choices=self.PARALLEL_TILE_PROCESSING_CHOICES, value=config["parallel_tile_processing"], label="Parallel Tile Processing", interactive=not lock_config)
        return [("persistence", persistence), ("parallel_tile_processing", parallel_tile_processing)]

    def config_requires_release(self, old_config: dict[str, Any], new_config: dict[str, Any], changed_keys: set[str]) -> bool:
        return old_config != new_config or bool({"image_profile", "lm_decoder_engine"} & changed_keys)

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        if not self.is_upsampling(spatial_upsampling):
            return ""
        if image_mode == 0:
            return "Chain-of-Zoom Spatial Upsampling is only available for Images"
        scale = self.split_value(spatial_upsampling)[1]
        if scale not in self.MULTIPLIERS:
            return f"Chain-of-Zoom only supports x{', x'.join(f'{value:g}' for value in self.MULTIPLIERS)} upsampling"
        return ""

    def _t5_filename(self) -> str:
        return self.T5_FILENAME_INT8

    def _vlm_filename(self) -> str:
        return self.VLM_QUANTIZED_FILENAME

    def query_download_defs(self) -> list[dict[str, Any]]:
        return [
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": [self.COZ_FOLDER],
                "fileList": [[self.TRANSFORMER_FILENAME, self.VAE_FILENAME, self._vlm_filename()] + self.VLM_EXTRA_FILES],
            },
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": [self.CLIP_L_FOLDER, self.CLIP_G_FOLDER],
                "fileList": [[self.CLIP_L_FILENAME, self.CLIP_CONFIG_FILENAME], [self.CLIP_G_FILENAME, self.CLIP_CONFIG_FILENAME]],
            },
            {
                "repoId": "DeepBeepMeep/HunyuanVideo",
                "sourceFolderList": [self.CLIP_TOKENIZER_FOLDER],
                "fileList": [self.CLIP_TOKENIZER_FILES],
            },
            {
                "repoId": "DeepBeepMeep/LTX_Video",
                "sourceFolderList": [self.T5_FOLDER],
                "fileList": [[self._t5_filename()] + self.T5_EXTRA_FILES],
            },
        ]

    def _required_files(self) -> list[str]:
        coz_files = [self.TRANSFORMER_FILENAME, self.VAE_FILENAME, self._vlm_filename()] + self.VLM_EXTRA_FILES
        required = [os.path.join(self.COZ_FOLDER, filename) for filename in coz_files]
        required += [os.path.join(self.CLIP_L_FOLDER, filename) for filename in [self.CLIP_L_FILENAME, self.CLIP_CONFIG_FILENAME]]
        required += [os.path.join(self.CLIP_G_FOLDER, filename) for filename in [self.CLIP_G_FILENAME, self.CLIP_CONFIG_FILENAME]]
        required += [os.path.join(self.CLIP_TOKENIZER_FOLDER, filename) for filename in self.CLIP_TOKENIZER_FILES]
        required += [os.path.join(self.T5_FOLDER, filename) for filename in [self._t5_filename()] + self.T5_EXTRA_FILES]
        return required

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        if all(self.files_locator.locate_file(path, error_if_none=False) is not None for path in self._required_files()):
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        for download_def in self.query_download_defs():
            process_files(**download_def)
        return True

    def paths(self):
        from postprocessing.chain_of_zoom.runtime import CoZPaths

        fl = self.files_locator
        return CoZPaths(
            transformer=fl.locate_file(os.path.join(self.COZ_FOLDER, self.TRANSFORMER_FILENAME)),
            vae=fl.locate_file(os.path.join(self.COZ_FOLDER, self.VAE_FILENAME)),
            clip_l=fl.locate_file(os.path.join(self.CLIP_L_FOLDER, self.CLIP_L_FILENAME)),
            clip_tokenizer_dir=os.path.dirname(fl.locate_file(os.path.join(self.CLIP_TOKENIZER_FOLDER, "tokenizer_config.json"))),
            clip_g=fl.locate_file(os.path.join(self.CLIP_G_FOLDER, self.CLIP_G_FILENAME)),
            t5=fl.locate_file(os.path.join(self.T5_FOLDER, self._t5_filename())),
            vlm=fl.locate_file(os.path.join(self.COZ_FOLDER, self._vlm_filename())),
            vlm_dir=os.path.dirname(fl.locate_file(os.path.join(self.COZ_FOLDER, "config.json"))),
        )

    def load_upsampler(
        self,
        spatial_upsampling,
        *,
        process_files: Callable[..., Any],
        init_pipe: Callable[..., int],
        profile,
        progress_callback=None,
        **kwargs,
    ):
        split = self.split_value(spatial_upsampling)
        if split is None or split[1] not in self.MULTIPLIERS:
            raise ValueError(f"Unknown Chain-of-Zoom upsampling mode: {spatial_upsampling}")
        self.download(process_files)
        from postprocessing.chain_of_zoom.runtime import load_models
        from shared.llm_engines.nanovllm.vllm_support import resolve_lm_decoder_engine

        lm_decoder_engine = resolve_lm_decoder_engine(self.server_config.get("lm_decoder_engine", ""), ["cg", "vllm"])
        parallel_tile_processing = self.parallel_tile_processing()
        load_models(self.paths(), lm_decoder_engine=lm_decoder_engine, vlm_vision_batch=parallel_tile_processing, vlm_prompt_batch=parallel_tile_processing * 2, init_pipe=init_pipe, profile=profile, progress_callback=progress_callback)

    def upscale(
        self,
        sample,
        spatial_upsampling,
        *,
        seed=0,
        abort_callback=None,
        progress_callback=None,
        **kwargs,
    ):
        split = self.split_value(spatial_upsampling)
        if split is None or split[1] not in self.MULTIPLIERS:
            raise ValueError(f"Unknown Chain-of-Zoom upsampling mode: {spatial_upsampling}")
        from postprocessing.chain_of_zoom.runtime import upscale_image

        return upscale_image(sample, split[1], seed=seed, abort_callback=abort_callback, progress_callback=progress_callback), None

    def release_vram(self) -> None:
        from postprocessing.chain_of_zoom.runtime import release_models

        release_models()
