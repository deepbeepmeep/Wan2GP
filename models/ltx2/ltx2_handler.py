import os
import torch
from shared.utils import files_locator as fl


_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_GEMMA_MERGED_FILENAME = f"{_GEMMA_FOLDER}.safetensors"
_GEMMA_QUANTO_FILENAME = f"{_GEMMA_FOLDER}_quanto_bf16_int8.safetensors"
_SPATIAL_UPSCALER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
_DISTILLED_LORA_FILENAME = "ltx-2-19b-distilled-lora-384.safetensors"


def get_text_encoder_name(text_encoder_quantization):
    text_encoder_filename = f"{_GEMMA_FOLDER}/{_GEMMA_MERGED_FILENAME}"
    if text_encoder_quantization == "int8":
        text_encoder_filename = f"{_GEMMA_FOLDER}/{_GEMMA_QUANTO_FILENAME}"
    return fl.locate_file(text_encoder_filename, True)


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["ltx2_19B"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "ltx2"

    @staticmethod
    def query_family_infos():
        return {"ltx2": (40, "LTX-2")}


    @staticmethod
    def query_model_def(base_model_type, model_def):
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        extra_model_def = {
            "fps": 24,
            "frames_minimum": 17,
            "frames_steps": 8,
            "sliding_window": True,
            "image_prompt_types_allowed": "TSEV",
            "returns_audio": True,
            "profiles_dir": ["ltx2_19B"],
        }
        extra_model_def["sliding_window_defaults"] = {
            "overlap_min": 1,
            "overlap_max": 97,
            "overlap_step": 8,
            "overlap_default": 9,
            "window_min": 5,
            "window_max": 501,
            "window_step": 4,
            "window_default": 241,
        }
        if pipeline_kind == "distilled":
            extra_model_def.update(
                {
                    "lock_inference_steps": True,
                    "no_negative_prompt": True,
                    "guidance_max_phases": 1,
                }
            )
        else:
            extra_model_def["guidance_max_phases"] = 2
            extra_model_def["virtual_higher_phases"] = True
            extra_model_def["lock_guidance_phases"] = True
        return extra_model_def

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("ltx2")

    @staticmethod
    def register_lora_cli_args(parser):
        parser.add_argument(
            "--lora-dir-ltx2",
            type=str,
            default=os.path.join("loras", "ltx2"),
            help="Path to a directory that contains LTX-2 LoRAs",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args):
        return args.lora_dir_ltx2

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 64

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_text_encoder_name(text_encoder_quantization)
        gemma_files = [
            "added_tokens.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ] + computeList(text_encoder_filename)

        download_def = [
            {
                "repoId": "DeepBeepMeep/LTX-2",
                "sourceFolderList": [""],
                "fileList": [[_SPATIAL_UPSCALER_FILENAME, _DISTILLED_LORA_FILENAME] + computeList(model_filename)],
            },
            {
                "repoId": "DeepBeepMeep/LTX-2",
                "sourceFolderList": [_GEMMA_FOLDER],
                "fileList": [gemma_files],
            },
        ]
        return download_def

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        override_text_encoder=None,
        **kwargs,
    ):
        from .ltx2 import LTX2

        ltx2_model = LTX2(
            model_filename=model_filename,
            model_type=model_type,
            base_model_type=base_model_type,
            model_def=model_def,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            override_text_encoder=override_text_encoder,
        )

        pipe = {
            "transformer": ltx2_model.model,
            "text_encoder": ltx2_model.text_encoder,
            "text_embedding_projection": ltx2_model.text_embedding_projection,
            "text_embeddings_connector": ltx2_model.text_embeddings_connector,
            "vae": ltx2_model.video_decoder,
            "video_encoder": ltx2_model.video_encoder,
            "audio_decoder": ltx2_model.audio_decoder,
            "vocoder": ltx2_model.vocoder,
            "spatial_upsampler": ltx2_model.spatial_upsampler,
        }
        if ltx2_model.model2 is not None:
            pipe["transformer2"] = ltx2_model.model2
        return ltx2_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind != "distilled" and ui_defaults.get("guidance_phases", 0) < 2:
            ui_defaults["guidance_phases"] = 2

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "sliding_window_size": 481,
                "sliding_window_overlap": 9,
            }
        )
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind != "distilled":
            ui_defaults.setdefault("guidance_phases", 2)
        else:
            ui_defaults.setdefault("guidance_phases", 1)
