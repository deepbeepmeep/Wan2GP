import os
import torch
from shared.utils import files_locator as fl


def get_longcat_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
    if text_encoder_quantization == "int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_int8")
    return fl.locate_file(text_encoder_filename, True)


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["longcat_video", "longcat_avatar"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "longcat"

    @staticmethod
    def query_family_infos():
        return {"longcat": (60, "LongCat")}

    @staticmethod
    def register_lora_cli_args(parser):
        parser.add_argument(
            "--lora-dir-longcat",
            type=str,
            default=os.path.join("loras", "longcat"),
            help="Path to a directory that contains LongCat Video LoRAs",
        )
        parser.add_argument(
            "--lora-dir-longcat-avatar",
            type=str,
            default=os.path.join("loras", "longcat_avatar"),
            help="Path to a directory that contains LongCat Avatar LoRAs",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args):
        if base_model_type == "longcat_avatar":
            return args.lora_dir_longcat_avatar
        return args.lora_dir_longcat

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "frames_minimum": 5,
            "frames_steps": 4,
            "sliding_window": True,
            "guidance_max_phases": 1,
            "image_prompt_types_allowed": "TSVL",
            "video_continuation": True,
            "sample_solvers": [
                ("Auto (Continuation = Enhanced HF)", "auto"),
                ("Default", ""),
                ("Enhanced HF", "enhance_hf"),
                ("Distill", "distill"),
            ],
        }

        if base_model_type == "longcat_video":
            extra_model_def.update(
                {
                    "fps": 15,
                    "profiles_dir": ["longcat_video"],
                }
            )
        elif base_model_type == "longcat_avatar":
            extra_model_def.update(
                {
                    "fps": 16,
                    "profiles_dir": [base_model_type],
                    "audio_guide_label": "Voice to follow",
                    "audio_guide2_label": "Voice to follow #2",
                    "audio_guidance": True,
                    "any_audio_prompt": True,
                    "audio_prompt_choices": True,                
                    "image_ref_choices": {
                        "choices": [("None", ""), ("Anchor Reference Image", "KI")],
                        "letters_filter": "KI",
                        "visible": True,
                        "label": "Anchor Reference Image",
                    },
                    "reference_image_enabled": True,
                    "no_background_removal": True,
                    "image_prompt_types_allowed": "TSVL",
                }
            )


        return extra_model_def

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("wan")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_longcat_text_encoder_filename(text_encoder_quantization)
        download_def = [
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": ["umt5-xxl", "chinese-wav2vec2-base"],
                "fileList": [
                    ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"]
                    + computeList(text_encoder_filename),
                    [
                        "config.json",
                        "preprocessor_config.json",
                        "pytorch_model.bin",
                        "readme.txt",
                    ],
                ],
            }
        ]
        download_def += [
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": [""],
                "fileList": [["Wan2.1_VAE_bf16.safetensors"]],
            }
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
        from .longcat_main import LongCatModel

        longcat_model = LongCatModel(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=get_longcat_text_encoder_filename(text_encoder_quantization)
            if override_text_encoder is None
            else override_text_encoder,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
        )

        pipe = {
            "transformer": longcat_model.transformer,
            "vae": longcat_model.vae,
            "text_encoder": longcat_model.text_encoder.model,
        }
        if longcat_model.audio_encoder is not None:
            pipe["wav2vec"] = longcat_model.audio_encoder

        return longcat_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "audio_guidance_scale": 4.0,
                "sliding_window_overlap": 13,
                
            }
        )
        if base_model_type == "longcat_video":
            ui_defaults.update({"video_length": 93})

        if base_model_type in ["longcat_avatar"]:
            ui_defaults.update({"video_length": 93, "video_prompt_type": ""})

        if ui_defaults.get("sample_solver", "") == "":
            ui_defaults["sample_solver"] = "auto"
