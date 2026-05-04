import os
import shutil
import sys
import torch
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url
from shared.utils.loras_mutipliers import parse_loras_multipliers
import gradio as gr
from pathlib import Path

from .lora_utils import control_video_phase2_message

_GEMMA_FOLDER_URL = "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/gemma-3-12b-it-qat-q4_0-unquantized/"
_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_GEMMA_FILENAME = f"{_GEMMA_FOLDER}.safetensors"
_GEMMA_QUANTO_FILENAME = f"{_GEMMA_FOLDER}_quanto_bf16_int8.safetensors"
_LORAS_MIGRATED = False
_LORA_SPEC_KEYS = ("distilled_lora", "distilled_1_1_lora", "union_control_lora", "id_lora", "outpaint_lora", "hdr_lora")

_ARCH_SPECS = {
    "ltx2_19B": {
        "repo_id": "DeepBeepMeep/LTX-2",
        "config_file": "ltx2_19b_config.json",
        "spatial_upscaler": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "temporal_upscaler": "ltx-2-temporal-upscaler-x2-1.0.safetensors",
        "distilled_lora": "ltx-2-19b-distilled-lora-384.safetensors",
        "union_control_lora": "ltx-2-19b-ic-lora-union-control-ref0.5.safetensors",
        "id_lora": "id-lora-celebvhq-ltx2.safetensors",
        "video_vae": "ltx-2-19b_vae.safetensors",
        "audio_vae": "ltx-2-19b_audio_vae.safetensors",
        "vocoder": "ltx-2-19b_vocoder.safetensors",
        "text_embedding_projection": "ltx-2-19b_text_embedding_projection.safetensors",
        "dev_embeddings_connector": "ltx-2-19b-dev_embeddings_connector.safetensors",
        "distilled_embeddings_connector": "ltx-2-19b-distilled_embeddings_connector.safetensors",
        "profiles_dir": "ltx2",
        "preset_profiles_dir": "ltx2_presets",
        "distilled_preset_profiles_dir": "ltx2_distilled_presets",
        "lora_dir": "ltx2",
    },
    "ltx2_22B": {
        "repo_id": "DeepBeepMeep/LTX-2",
        "config_file": "ltx2_22b_config.json",
        "spatial_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "temporal_upscaler": "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "distilled_1_1_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "union_control_lora": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "id_lora": "id-lora-celebvhq-ltx2.3.safetensors",
        "outpaint_lora": "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "hdr_lora": "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
        "hdr_scene_embeddings": "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
        "video_vae": "ltx-2.3-22b_vae.safetensors",
        "audio_vae": "ltx-2.3-22b_audio_vae.safetensors",
        "vocoder": "ltx-2.3-22b_vocoder.safetensors",
        "text_embedding_projection": "ltx-2.3-22b_text_embedding_projection.safetensors",
        "embeddings_connector": "ltx-2.3-22b_embeddings_connector.safetensors",
        "profiles_dir": "ltx2",
        "preset_profiles_dir": "ltx2_presets",
        "distilled_preset_profiles_dir": "ltx2_distilled_presets",
        "lora_dir": "ltx2",
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


def _migrate_loras():
    global _LORAS_MIGRATED
    if _LORAS_MIGRATED:
        return
    wgp = sys.modules.get("wgp")
    lora_root = wgp.get_lora_root()

    lora_dir = Path(lora_root) / _ARCH_SPECS["ltx2_19B"]["lora_dir"]
    lora_dir.mkdir(parents=True, exist_ok=True)

    moved = set()
    for spec in _ARCH_SPECS.values():
        for key in _LORA_SPEC_KEYS:
            filename = spec.get(key, None)
            if filename is None or filename in moved:
                continue
            source = fl.locate_file(filename, error_if_none=False)
            if source is None:
                continue
            target = lora_dir / filename
            if Path(source).resolve() == target.resolve() or target.exists():
                moved.add(filename)
                continue
            shutil.move(source, target)
            print(f"[WAN2GP][LTX2] Moved {key} LoRA '{source}' -> '{target}'")
            moved.add(filename)
            
    _LORAS_MIGRATED = True


def _notify_control_video_phase2(base_model_type, model_def, inputs, any_outpainting):
    video_prompt_type = inputs.get("video_prompt_type", "") or ""
    if int(inputs.get("guidance_phases", 1)) != 2 or "V" not in video_prompt_type or inputs.get("video_guide") is None:
        return ""
    wgp = sys.modules.get("wgp")
    lora_dir = wgp.get_lora_dir(base_model_type) if wgp is not None and hasattr(wgp, "get_lora_dir") else None
    selected = {os.path.basename(lora).lower() for lora in inputs.get("activated_loras", []) or []}
    spec = _get_arch_spec(base_model_type)
    builtins = [] if model_def.get("ltx2_pipeline", "two_stage") != "distilled" else [
        spec.get("hdr_lora") if base_model_type == "ltx2_22B" and "&" in video_prompt_type else None,
        spec.get("union_control_lora") if any(letter in video_prompt_type for letter in "OPDE") else None,
        spec.get("outpaint_lora") if base_model_type == "ltx2_22B" and any_outpainting else None,
    ]
    extra_loras = [os.path.join(lora_dir, name) if lora_dir else name for name in builtins if name and name.lower() not in selected]
    extra_mults = [1.0] * len(extra_loras)
    activated_loras = [os.path.join(lora_dir, os.path.basename(lora)) if lora_dir else lora for lora in inputs.get("activated_loras", []) or []]
    steps, switch_phase = int(inputs.get("num_inference_steps", 1)), inputs.get("model_switch_phase", 1)
    _, loras_slists, errors = parse_loras_multipliers(extra_mults, len(extra_loras), steps, nb_phases=2, model_switch_phase=switch_phase)
    if not errors:
        _, loras_slists, errors = parse_loras_multipliers(inputs.get("loras_multipliers", ""), len(activated_loras), steps, nb_phases=2, merge_slist=loras_slists, model_switch_phase=switch_phase)
    if errors:
        return f"Error parsing Loras: {errors}"
    loras_selected = extra_loras + activated_loras
    msg = control_video_phase2_message(loras_selected, loras_slists)
    print(msg)
    gr.Info(msg)
    return ""


class family_handler:
    @staticmethod
    def query_supported_types():
        _migrate_loras()
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
        preload_urls = model_def.get("preload_URLs")
        spec = _get_arch_spec(base_model_type)
        if isinstance(preload_urls, list): 
            # migrate old finetunes
            lora_filenames = {spec[key] for key in _LORA_SPEC_KEYS if key in spec}
            def add_lora_dir_suffix(entry):
                if not isinstance(entry, str) or "|%lora_dir" in entry:
                    return entry
                source_entry = entry.split("|", 1)[0]
                if source_entry.startswith("http") and os.path.basename(source_entry) in lora_filenames:
                    return f"{source_entry}|%lora_dir"
                return entry
            model_def["preload_URLs"] = [add_lora_dir_suffix(entry) for entry in preload_urls]

        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")

        distilled = pipeline_kind == "distilled"
        audio_prompt_selection = ["", "A", "K", "2"] if distilled else ["", "A", "A1OF"]
        audio_prompt_labels = {
            "": "Generate Video & Soundtrack based on Text Prompt",
            "A": "Generate Video based on Soundtrack and Text Prompt",
            "A1OF": "Generate Video based on Reference Voice (ID-LoRA) and Text Prompt",
        }
        if distilled:
            audio_prompt_labels["K"] = "Generate Video based on Control Video + its Audio Track and Text Prompt"
            audio_prompt_labels["2"] = "Generate Audio based on Control Video and Text Prompt"


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
            "audio_guide_label": "Audio Prompt (Soundtrack, leave blank to to use a Null Audio)",
            "audio_scale_name": "Prompt Audio Strength",
            "audio_prompt_type_sources": {
                "selection": audio_prompt_selection,
                "labels": audio_prompt_labels,
                "custom_flags": {
                    "1": "Reference Voice (ID-LoRA)",
                    "2": "Generate Audio based on Control Video and Text Prompt",
                },
                "letters_filter": "A1OFK2",
                "show_label": False,
            },
            "auto_null_audio": True,
            "audio_guide_window_slicing": True,
            "video_length_not_limited_by_audio": True,
            "output_audio_is_input_audio": True,
            "multimedia_generation": True,
            "multiple_images_as_text_prompts": True,
            "custom_denoising_strength": distilled,
            "profiles_dir": [spec["profiles_dir"]],
            "ltx2_spatial_upscaler_file": spec["spatial_upscaler"],
            "ltx2_hdr_lora_file": spec.get("hdr_lora", ""),
            "ltx2_hdr_scene_embeddings_file": spec.get("hdr_scene_embeddings", ""),
            "self_refiner": True,
            "self_refiner_max_plans": 2,
            "no_background_removal": True,
            "vae_block_size": 64,
            "keep_frames_video_guide_not_supported": True,
        }
        
        if distilled and base_model_type in ["ltx2_22B"]:
            extra_model_def["video_guide_outpainting"] = [0,1]
            extra_model_def["video_guide_outpainting_label"] = "Enable Spatial Outpainting on Control Video using Ic Lora Outpaint"
            extra_model_def["guide_inpaint_color"] = 0

        extra_model_def["preset_profiles_dir"] = [spec.get("distilled_preset_profiles_dir") if distilled else spec.get("preset_profiles_dir")]
        extra_model_def["extra_control_frames"] = 1
        extra_model_def["dont_cat_preguide"] = True
        extra_model_def["input_video_strength"] = {
            "label": "Start Image / Source Strength (lower values may create more motion)",
            "name": "Start Image / Source Strength",
        }
        extra_model_def["denoising_strength"] = {
            "label": "Control Video Strength (higher = closer to the Control Video)",
            "name": "Control Video Strength",
        }
        extra_model_def["masking_strength"] = {
            "label": "Masked Control Duration (higher = longer masked reinjection)",
            "name": "Masked Control Duration",
        }
        
        control_choices = [("No Video Process", "")]
        control_choices += [ ("Transfer Human Motion", "PVG"), ("Transfer Human Motion With Pose Alignment", "OVG")  , ("Transfer Depth", "DVG") , ("Transfer Canny Edges", "EVG"), ("LTX2 Raw Format / Control Video for Ic Lora", "VG")] if distilled else []
        if distilled and base_model_type == "ltx2_22B":
            control_choices += [("Convert SDR to HDR (IC-LoRA)", f"V&G")]
        control_choices +=   [("Inject Frames", "KFI")]
        extra_model_def["guide_custom_choices"] = {
            "choices": control_choices,
            "letters_filter": f"OPDEVG&KFI",
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
                    "NAG": True,
                    "no_negative_prompt": False,
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
            if base_model_type == "ltx2_22B":
                extra_model_def["sample_solvers"] = [("Euler", "euler"), ("HQ (res2s)", "res2s")]
        extra_model_def["guidance_max_phases"] = 2
        extra_model_def["visible_phases"] = 0 if distilled else 1
        # extra_model_def["lock_guidance_phases"] = True

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
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind == "distilled":
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
            if inputs.get("perturbation",0) == 2:
                inputs["perturbation"] = 0
        else:
            sample_solver = inputs.get("sample_solver", "euler" if base_model_type == "ltx2_22B" else "").lower()
            if base_model_type == "ltx2_22B":
                if sample_solver not in {"euler", "res2s"}:
                    return f"Unsupported LTX2 sampler '{sample_solver}'."
                inputs["sample_solver"] = sample_solver
                if sample_solver == "res2s":
                    if inputs.get("apg_switch", 0):
                        return "HQ sampler does not support APG yet."
                    if inputs.get("cfg_star_switch", 0):
                        return "HQ sampler does not support CFG Star yet."
                    if inputs.get("self_refiner_setting", 0):
                        return "HQ sampler does not support Self Refiner yet."
                    inputs["perturbation_switch"] = 0
            elif sample_solver not in {"", "euler"}:
                return f"Sampler '{sample_solver}' is not supported for {base_model_type}."
        video_guide_outpainting = inputs.get("video_guide_outpainting", None) 
        video_guide_outpainting_ratio = inputs.get("video_guide_outpainting_ratio", "") 
        video_prompt_type = inputs.get("video_prompt_type", "") or ""
        audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
        from shared.utils.utils import get_outpainting_dims 
        any_outpainting = get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio) is not None        
        if "2" in audio_prompt_type:
            if pipeline_kind != "distilled":
                return "LTX2 audio generation from Control Video is supported only with distilled models."
            if any(letter in audio_prompt_type for letter in "AK"):
                return "LTX2 audio generation from Control Video must use the dedicated audio option, without an Audio Source or Control Video Audio Track prompt."
            if "V" not in video_prompt_type or "G" not in video_prompt_type:
                return "LTX2 audio generation from Control Video requires 'LTX2 Raw Format / Control Video for Ic Lora'."
            if any(letter in video_prompt_type for letter in "OPDE&AFKI") or any_outpainting:
                return "LTX2 audio generation from Control Video supports only raw Control Video, without Pose/Depth/Canny/HDR/Outpaint/Mask/Inject Frames."
            if inputs.get("video_guide") is None:
                return "You must provide a Control Video to generate audio from it."
        if "&" in video_prompt_type:
            if pipeline_kind != "distilled" or base_model_type != "ltx2_22B":
                return "LTX2 HDR IC-LoRA is supported only with LTX-2.3 22B distilled."
            if any(letter in video_prompt_type for letter in "OPDE") or any_outpainting:
                return "LTX2 HDR IC-LoRA is not compatible with Pose/Depth/Canny/Outpaint control modes."
            if "F" in video_prompt_type:
                return "LTX2 HDR IC-LoRA is not yet compatible with Inject Frames."
        if pipeline_kind == "distilled" and any_outpainting:
            if "V" in video_prompt_type :
                if any(letter in video_prompt_type for letter in "OPDE"):
                    return "LTX2 outpainting on Control Video supports only LTX2 Raw Format  / Contro Video for Ic Lora."
                if "1" in audio_prompt_type:
                    return "LTX2 outpainting on Control Video is not compatible with the ID-LoRA option."
                if "F" in video_prompt_type :
                    return "LTX2 outpainting is not yet compatible with Inject Frames."
                if "A" in video_prompt_type :
                    return "LTX2 outpainting doesnt support Video Mask."

        guide_phases = inputs.get("guidance_phases", 1)
        if guide_phases !=1 and "V" in video_prompt_type and any_outpainting:
            inputs["guidance_phases"]=  1            
            gr.Info("Number of Phases has been set to 1 as Outpainting is enabled")
        if "2" not in audio_prompt_type:
            error = _notify_control_video_phase2(base_model_type, model_def, inputs, any_outpainting)
            if error:
                return error
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
        if pipeline_kind != "distilled" and ui_defaults.get("sample_solver", "") in {"", None}:
            ui_defaults["sample_solver"] = "euler"

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

        if settings_version < 2.58 and pipeline_kind == "distilled":
            ui_defaults["guidance_phases"]=2
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
                "guidance_phases": 2,
	            }
        )
        ui_defaults.setdefault("audio_scale", 1.0)
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind != "distilled":
            ui_defaults.update(_default_dev_settings(base_model_type))
            ui_defaults.setdefault("sample_solver", "euler")

    @staticmethod
    def get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs):
        from .prompt_enhancer import  get_custom_prompt_enhancer_instructions
        return get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs)
