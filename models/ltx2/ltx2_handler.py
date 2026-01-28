import os
import torch
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url

_GEMMA_FOLDER_URL = "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/gemma-3-12b-it-qat-q4_0-unquantized/"
_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_GEMMA_FILENAME = f"{_GEMMA_FOLDER}.safetensors"
_GEMMA_QUANTO_FILENAME = f"{_GEMMA_FOLDER}_quanto_bf16_int8.safetensors"
_SPATIAL_UPSCALER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
_DISTILLED_LORA_FILENAME = "ltx-2-19b-distilled-lora-384.safetensors"
_VIDEO_VAE_FILENAME = "ltx-2-19b_vae.safetensors"
_AUDIO_VAE_FILENAME = "ltx-2-19b_audio_vae.safetensors"
_VOCODER_FILENAME = "ltx-2-19b_vocoder.safetensors"
_TEXT_EMBEDDING_PROJECTION_FILENAME = "ltx-2-19b_text_embedding_projection.safetensors"
_DEV_EMBEDDINGS_CONNECTOR_FILENAME = "ltx-2-19b-dev_embeddings_connector.safetensors"
_DISTILLED_EMBEDDINGS_CONNECTOR_FILENAME = "ltx-2-19b-distilled_embeddings_connector.safetensors"


def _get_embeddings_connector_filename(model_def):
    pipeline_kind = (model_def or {}).get("ltx2_pipeline", "two_stage")
    if pipeline_kind == "distilled":
        return _DISTILLED_EMBEDDINGS_CONNECTOR_FILENAME
    return _DEV_EMBEDDINGS_CONNECTOR_FILENAME


def _get_multi_file_names(model_def):
    return {
        "video_vae": _VIDEO_VAE_FILENAME,
        "audio_vae": _AUDIO_VAE_FILENAME,
        "vocoder": _VOCODER_FILENAME,
        "text_embedding_projection": _TEXT_EMBEDDING_PROJECTION_FILENAME,
        "text_embeddings_connector": _get_embeddings_connector_filename(model_def),
    }


def _resolve_multi_file_paths(model_def):
    return {key: fl.locate_file(name) for key, name in _get_multi_file_names(model_def).items()}




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

        distilled = pipeline_kind == "distilled"

        extra_model_def = {
            "text_encoder_folder": _GEMMA_FOLDER,
            "text_encoder_URLs": [
                build_hf_url("DeepBeepMeep/LTX-2", _GEMMA_FOLDER, _GEMMA_FILENAME),
                build_hf_url("DeepBeepMeep/LTX-2", _GEMMA_FOLDER, _GEMMA_QUANTO_FILENAME),
            ],
            "dtype": "bf16",
            "fps": 24,
            "frames_minimum": 17,
            "frames_steps": 8,
            "sliding_window": True,
            "image_prompt_types_allowed": "TSEV",
            "returns_audio": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "one_speaker_only": True,
            "audio_guide_label": "Audio Prompt (Soundtrack)",
            "audio_scale_name": "Prompt Audio Strength",
            "audio_prompt_type_sources": {
                "selection": ["", "A", "K"],
                "labels": {
                    "": "Generate Video & Soundtrack based on Text Prompt",
                    "A": "Generate Video based on Soundtrack and Text Prompt",
                    "K": "Generate Video based on Control Video + its Audio Track and Text Prompt",
                },
                "show_label": False,
            },
            "audio_guide_window_slicing": True,
            "output_audio_is_input_audio": True,
            "custom_denoising_strength": True,
            "profiles_dir": ["ltx2_19B"],
        }
        extra_model_def["extra_control_frames"] = 1
        extra_model_def["dont_cat_preguide"] = True
        extra_model_def["input_video_strength"] = "Image / Source Video Strength (you may try values lower value than 1 to get more motion)"
        extra_model_def["guide_preprocessing"] = {
            "selection": ["", "PVG", "DVG", "EVG", "VG"],
            "labels": {
                "PVG": "Transfer Human Motion",
                "DVG": "Transfer Depth",
                "EVG": "Transfer Canny Edges",
                "VG": "Use LTX-2 raw format",
            },
        }
        extra_model_def["mask_preprocessing"] = {
            "selection": ["", "A", "NA", "XA", "XNA"],
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
        if distilled:
            extra_model_def.update(
                {
                    "lock_inference_steps": True,
                    "no_negative_prompt": True,
                }
            )
        extra_model_def["guidance_max_phases"] = 2
        extra_model_def["visible_phases"] = 0 if distilled else 1
        extra_model_def["lock_guidance_phases"] = True
        return extra_model_def

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("ltx2")

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-ltx2",
            type=str,
            default=None,
            help=f"Path to a directory that contains LTX-2 LoRAs (default: {os.path.join(lora_root, 'ltx2')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_ltx2", None) or os.path.join(lora_root, "ltx2")

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 64

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        gemma_files = [
            "added_tokens.json",
            "chat_template.json",
            "config_light.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ] 

        file_list = [_SPATIAL_UPSCALER_FILENAME] 
        for name in _get_multi_file_names(model_def).values():
            if name not in file_list:
                file_list.append(name)

        download_def = [
            {
                "repoId": "DeepBeepMeep/LTX-2",
                "sourceFolderList": [""],
                "fileList": [file_list],
            },
            {
                "repoId": "DeepBeepMeep/LTX-2",
                "sourceFolderList": [_GEMMA_FOLDER],
                "fileList": [gemma_files],
            },
        ]
        return download_def

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        audio_prompt_type = inputs.get("audio_prompt_type") or ""
        if "A" in audio_prompt_type and inputs.get("audio_guide") is None:
            audio_source = inputs.get("audio_source")
            if audio_source is not None:
                inputs["audio_guide"] = audio_source

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
        text_encoder_filename=None,
        **kwargs,
    ):
        from .ltx2 import LTX2

        checkpoint_paths = _resolve_multi_file_paths(model_def)
        transformer_path = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        checkpoint_paths["transformer"] = transformer_path

        ltx2_model = LTX2(
            model_filename=model_filename,
            model_type=model_type,
            base_model_type=base_model_type,
            model_def=model_def,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            text_encoder_filename=text_encoder_filename,
            text_encoder_filepath = model_def.get("text_encoder_folder", os.path.dirname(text_encoder_filename)),
            checkpoint_paths=checkpoint_paths,
        )

        pipe = {
            "transformer": ltx2_model.model,
            "text_encoder": ltx2_model.text_encoder,
            "text_embedding_projection": ltx2_model.text_embedding_projection,
            "text_embeddings_connector": ltx2_model.text_embeddings_connector,
            "vae": ltx2_model.video_decoder,
            "video_encoder": ltx2_model.video_encoder,
            "audio_encoder": ltx2_model.audio_encoder,
            "audio_decoder": ltx2_model.audio_decoder,
            "vocoder": ltx2_model.vocoder,
            "spatial_upsampler": ltx2_model.spatial_upsampler,
        }
        if ltx2_model.model2 is not None:
            pipe["transformer2"] = ltx2_model.model2

        if model_def.get("ltx2_pipeline", "") != "distilled":
            pipe = { "pipe": pipe, "loras" : ["text_embedding_projection", "text_embeddings_connector"] }

        return ltx2_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind != "distilled" and ui_defaults.get("guidance_phases", 0) < 2:
            ui_defaults["guidance_phases"] = 2

        if settings_version < 2.43:
            ui_defaults.update(
                {
                    "denoising_strength": 1.0,
                    "masking_strength": 0,
                }
            )

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "sliding_window_size": 481,
                "sliding_window_overlap": 17,
                "denoising_strength": 1.0,
                "masking_strength": 0,
                "audio_prompt_type": "",
            }
        )
        ui_defaults.setdefault("audio_scale", 1.0)
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind != "distilled":
            ui_defaults.setdefault("guidance_phases", 2)
        else:
            ui_defaults.setdefault("guidance_phases", 1)
