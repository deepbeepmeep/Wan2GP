import os
import torch
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url

_GEMMA_FOLDER_URL = "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/gemma-3-12b-it-qat-q4_0-unquantized/"
_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_GEMMA_FILENAME = f"{_GEMMA_FOLDER}.safetensors"
_GEMMA_QUANTO_FILENAME = f"{_GEMMA_FOLDER}_quanto_bf16_int8.safetensors"

_ARCH_SPECS = {
    "ltx2_19B": {
        "repo_id": "DeepBeepMeep/LTX-2",
        "config_file": "ltx2_19b_config.json",
        "spatial_upscaler": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "temporal_upscaler": "ltx-2-temporal-upscaler-x2-1.0.safetensors",
        "distilled_lora": "ltx-2-19b-distilled-lora-384.safetensors",
        "video_vae": "ltx-2-19b_vae.safetensors",
        "audio_vae": "ltx-2-19b_audio_vae.safetensors",
        "vocoder": "ltx-2-19b_vocoder.safetensors",
        "text_embedding_projection": "ltx-2-19b_text_embedding_projection.safetensors",
        "dev_embeddings_connector": "ltx-2-19b-dev_embeddings_connector.safetensors",
        "distilled_embeddings_connector": "ltx-2-19b-distilled_embeddings_connector.safetensors",
        "profiles_dir": "ltx2_19B",
        "lora_dir": "ltx2",
    },
    "ltx2_22B": {
        "repo_id": "DeepBeepMeep/LTX-2",
        "config_file": "ltx2_22b_config.json",
        "spatial_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "temporal_upscaler": "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "union_control_lora": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "video_vae": "ltx-2.3-22b_vae.safetensors",
        "audio_vae": "ltx-2.3-22b_audio_vae.safetensors",
        "vocoder": "ltx-2.3-22b_vocoder.safetensors",
        "text_embedding_projection": "ltx-2.3-22b_text_embedding_projection.safetensors",
        "embeddings_connector": "ltx-2.3-22b_embeddings_connector.safetensors",
        "profiles_dir": "ltx2_22B",
        "lora_dir": "ltx2_22B",
    },
}


def _get_arch_spec(base_model_type: str | None) -> dict:
    return _ARCH_SPECS.get(base_model_type or "", _ARCH_SPECS["ltx2_19B"])


def _default_perturbation_layers(base_model_type: str | None) -> list[int]:
    return [28] if base_model_type == "ltx2_22B" else [29]


def _default_dev_settings(base_model_type: str | None) -> dict:
    return {
        "num_inference_steps": 30 if base_model_type == "ltx2_22B" else 40,
        "guidance_scale": 3.0,
        # "audio_guidance_scale": 7.0,
        # "alt_guidance_scale": 3.0,
        # "alt_scale": 0.7,
        # "perturbation_switch": 2,
        "perturbation_layers": _default_perturbation_layers(base_model_type),
        "perturbation_start_perc": 0,
        "perturbation_end_perc": 100,
        "apg_switch": 0,
        "cfg_star_switch": 0,
        "guidance_phases": 2,
    }


def _get_embeddings_connector_filename(model_def, base_model_type):
    spec = _get_arch_spec(base_model_type)
    shared_connector = spec.get("embeddings_connector")
    if shared_connector:
        return shared_connector
    pipeline_kind = (model_def or {}).get("ltx2_pipeline", "two_stage")
    if pipeline_kind == "distilled":
        return spec["distilled_embeddings_connector"]
    return spec["dev_embeddings_connector"]


def _get_multi_file_names(model_def, base_model_type):
    spec = _get_arch_spec(base_model_type)
    return {
        "video_vae": spec["video_vae"],
        "audio_vae": spec["audio_vae"],
        "vocoder": spec["vocoder"],
        "text_embedding_projection": spec["text_embedding_projection"],
        "text_embeddings_connector": _get_embeddings_connector_filename(model_def, base_model_type),
    }


def _resolve_multi_file_paths(model_def, base_model_type):
    spec = _get_arch_spec(base_model_type)
    paths = {key: fl.locate_file(name) for key, name in _get_multi_file_names(model_def, base_model_type).items()}
    paths["spatial_upsampler"] = fl.locate_file(spec["spatial_upscaler"])
    model_config = os.path.join(os.path.dirname(__file__), "configs", spec["config_file"])
    if not os.path.isfile(model_config):
        raise FileNotFoundError(f"Missing LTX config file: {model_config}")
    paths["model_config"] = model_config
    return paths




class family_handler:
    @staticmethod
    def query_supported_types():
        return ["ltx2_19B", "ltx2_22B"]

    @staticmethod
    def query_family_maps():

        models_eqv_map = {
            "ltx2_19B" : "ltx2_22B",
        }

        models_comp_map = { 
                    "ltx2_19B" : [ "ltx2_22B"],
                    }
        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "ltx2"

    @staticmethod
    def query_family_infos():
        return {"ltx2": (40, "LTX-2")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        spec = _get_arch_spec(base_model_type)
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")

        distilled = pipeline_kind == "distilled"
        audio_prompt_selection = ["", "A", "K"] if distilled else ["", "A"]
        audio_prompt_labels = {
            "": "Generate Video & Soundtrack based on Text Prompt",
            "A": "Generate Video based on Soundtrack and Text Prompt",
        }
        if distilled:
            audio_prompt_labels["K"] = "Generate Video based on Control Video + its Audio Track and Text Prompt"


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
            "end_frames_always_enabled": True,
            "returns_audio": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "one_speaker_only": True,
            "audio_guide_label": "Audio Prompt (Soundtrack)",
            "audio_scale_name": "Prompt Audio Strength",
            "multiple_images_as_text_prompts": True,
            "audio_prompt_type_sources": {
                "selection": audio_prompt_selection,
                "labels": audio_prompt_labels,
                "show_label": False,
            },
            "audio_guide_window_slicing": True,
            "output_audio_is_input_audio": True,
            "multimedia_generation": True,
            "custom_denoising_strength": distilled,
            "profiles_dir": [spec["profiles_dir"]],
            "ltx2_spatial_upscaler_file": spec["spatial_upscaler"],
            "self_refiner": True,
            "self_refiner_max_plans": 2,
            "no_background_removal": True,
            "vae_block_size": 64,
        }
        extra_model_def["extra_control_frames"] = 1
        extra_model_def["dont_cat_preguide"] = True
        extra_model_def["input_video_strength"] = "Image / Source Video Strength (you may try values lower value than 1 to get more motion)"
        
        control_choices = [("No Video Process", "")]
        control_choices += [ ("Transfer Human Motion", "PVG") , ("Transfer Depth", "DVG") , ("Transfer Canny Edges", "EVG"), ("Use LTX-2 raw format Control Video", "VG")] if distilled else []
        control_choices +=   [("Inject Frames", "KFI")]
        extra_model_def["guide_custom_choices"] = {
            "choices": control_choices,
            "letters_filter": "PDEVGKFI",
            "default": "",
            "label": "Control Video / Frames Injection"
        }

        extra_model_def["custom_frames_injection"] = True

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
        else:
            extra_model_def.update(
                {
                    "audio_guidance": True,
                    "adaptive_projected_guidance": True,
                    "cfg_star": True,
                    "perturbation": True,
                    "alt_guidance": "Modality Guidance",
                    "alt_scale": "Guidance Rescale",
                    "perturbation_choices": [
                        ("Off", 0),
                        ("Skip Layer Guidance", 1),
                        ("Skip Self Attention", 2),
                    ],
                    "perturbation_layers_max": 48,
                }
            )
        extra_model_def["guidance_max_phases"] = 2
        extra_model_def["visible_phases"] = 0 if distilled else 1
        extra_model_def["lock_guidance_phases"] = True

        # extra_model_def["custom_video_selection"] = {
        #     "choices":[
        #         ("None", ""),
        #         ("Inject Frames", "FI"),
        #     ],
        #     "label": "Inject Frames",
        #     "type": "checkbox",
        #     "letters_filter": "FI",
        #     "show_label" : False,
        #     "scale": 1,
        #     }

        return extra_model_def

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("ltx2", base_model_type)

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-ltx2",
            type=str,
            default=None,
            help=f"Path to a directory that contains LTX-2 LoRAs (default: {os.path.join(lora_root, 'ltx2')})",
        )
        # parser.add_argument(
        #     "--lora-dir-ltx2-22b",
        #     type=str,
        #     default=None,
        #     help=f"Path to a directory that contains LTX-2.3 22B LoRAs (default: {os.path.join(lora_root, 'ltx2_22B')})",
        # )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        # if base_model_type == "ltx2_22B":
        #     return getattr(args, "lora_dir_ltx2_22b", None) or os.path.join(lora_root, "ltx2_22B")
        return getattr(args, "lora_dir_ltx2", None) or os.path.join(lora_root, "ltx2")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        spec = _get_arch_spec(base_model_type)
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

        file_list = [spec["spatial_upscaler"], spec["temporal_upscaler"]]
        for name in _get_multi_file_names(model_def, base_model_type).values():
            if name not in file_list:
                file_list.append(name)

        download_def = [
            {
                "repoId": spec["repo_id"],
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
        if model_def.get("ltx2_pipeline", "two_stage") == "distilled":
            inputs.update(
                {
                    "num_inference_steps": 8,
                    "guidance_scale": 1.0,
                    "audio_guidance_scale": 1.0,
                    "audio_cfg_scale": 1.0,
                    "alt_guidance_scale": 1.0,
                    "alt_scale": 0.0,
                }
            )
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

        checkpoint_paths = _resolve_multi_file_paths(model_def, base_model_type)
        transformer_path = list(model_filename) if isinstance(model_filename, (list, tuple)) else model_filename
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

        if save_quantized:
            from wgp import save_quantized_model

            quantized_source = transformer_path[0] if isinstance(transformer_path, (list, tuple)) else transformer_path
            quantized_transformer = getattr(ltx2_model.model, "velocity_model", ltx2_model.model)
            save_quantized_model(
                quantized_transformer,
                model_type,
                quantized_source,
                dtype,
                checkpoint_paths["model_config"],
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
        default_perturbation_layers = _default_perturbation_layers(base_model_type)
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

        if settings_version < 2.45:
            ui_defaults.update(
                {
                    "alt_guidance_scale": 1.0,
                    "perturbation_layers": default_perturbation_layers,
                }
            )

        if settings_version < 2.49:
            ui_defaults.update(
                {
                    "self_refiner_plan": "2-8:3",
                }
            )

        if settings_version < 2.55 and pipeline_kind != "distilled":
            ui_defaults.update({
                "audio_guidance_scale": 1.0,
                "alt_guidance_scale": 1.0,
                "alt_scale": 0.0,
                })

                # _default_dev_settings(base_model_type)

        if settings_version < 2.52:
            plan = ui_defaults.get("self_refiner_plan")
            if isinstance(plan, list):
                from shared.utils.self_refiner import convert_refiner_list_to_string
                ui_defaults["self_refiner_plan"] = convert_refiner_list_to_string(plan)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        default_perturbation_layers = _default_perturbation_layers(base_model_type)
        ui_defaults.update(
            {
                "sliding_window_size": 481,
                "sliding_window_overlap": 17,
                "denoising_strength": 1.0,
                "masking_strength": 0,
                "audio_prompt_type": "",
                "perturbation_layers": default_perturbation_layers,
	            }
        )
        ui_defaults.setdefault("audio_scale", 1.0)
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind != "distilled":
            ui_defaults.update(_default_dev_settings(base_model_type))
        else:
            ui_defaults.setdefault("guidance_phases", 1)
