import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from shared.utils import files_locator as fl


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["ovi"]

    @staticmethod
    def query_family_maps() -> Tuple[Dict[str, str], Dict[str, list]]:
        return {}, {}

    @staticmethod
    def query_model_family():
        return "wan"

    @staticmethod
    def query_family_infos():
        return {}

    @staticmethod
    def register_lora_cli_args(parser):
        from .wan_handler import family_handler as wan_family_handler

        return wan_family_handler.register_lora_cli_args(parser)

    @staticmethod
    def get_wan_text_encoder_filename(text_encoder_quantization):
        text_encoder_filename =  "umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
        if text_encoder_quantization =="int8":
            text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_int8") 
        return  fl.locate_file(text_encoder_filename, True)

    @staticmethod
    def query_model_def(base_model_type: str, model_def: Dict[str, Any]):
        cfg = {
            "wan_5B_class": True,
            "profiles_dir": ["wan_2_2_ovi"],
            "group": "wan2_2",
            "fps": 24,
            "frames_minimum": 121,
            "frames_steps": 120,
            "sliding_window": False,
            "multiple_submodels": False,
            "guidance_max_phases": 1,
            "skip_layer_guidance": True,
            "returns_audio": True,
            "sample_solvers": [
                ("unipc", "unipc"),
                ("dpm++", "dpm++"),
                ("euler", "euler"),
            ],
            "flow_shift": True,
            "audio_guidance": True,
            "image_prompt_types_allowed" : "TSVL",
            "sliding_window": True,
            "sliding_window_size_locked": True,
            "sliding_window_defaults" : { "overlap_min" : 1, "overlap_max" : 1, "overlap_step": 0, "overlap_default": 1},
            "compile":  ["transformer", "transformer2"]
        }
        cfg.update(model_def)
        return cfg

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):

        from .wan_handler import family_handler
        download_def = family_handler.query_model_files(computeList, "ti2v_2_2", model_filename, text_encoder_quantization)
        if not isinstance(download_def, list):
            download_def = [download_def]
        download_def  += [{
            "repoId" : "DeepBeepMeep/Wan2.1", 
            "sourceFolderList" :  ["mmaudio", ],
            "fileList" : [ [ "v1-16.pth", "best_netG.pt"]]   
        }]

        return download_def

    @staticmethod
    def get_lora_dir(base_model_type, args):
        from .wan_handler import family_handler as wan_family_handler

        return wan_family_handler.get_lora_dir(base_model_type, args)

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
    ):
        from .ovi_fusion_engine import OviFusionEngine 

        checkpoint_dir = "ckpts"

        ovi_model = OviFusionEngine(
            config=None,
            checkpoint_dir=checkpoint_dir,
            model_def=model_def,
            model_filename = model_filename, 
            text_encoder_filename = family_handler.get_wan_text_encoder_filename(text_encoder_quantization),
            dtype=dtype,
        )

        pipe = {
            "transformer": ovi_model.model.video_model,
            "transformer2": ovi_model.model.audio_model,
            "text_encoder": ovi_model.text_encoder.model,
            "vae": ovi_model.vae.model,
            "vae2": ovi_model.audio_vae,
        }
        cotenants_map = { 
                            "transformer": ["transformer2"],
                            "transformer2": ["transformer"],                             
        }

        dict = { "pipe": pipe, "coTenantsMap": cotenants_map}
        return ovi_model, dict

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pass

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({  "sample_solver": "unipc",
                        "flow_shift": 5.0,
                        "guidance_scale":  4.0,
                        "audio_guidance_scale": 3.0,
                        "num_inference_steps": 50,
                        "slg_switch": 1,
                        "sliding_window_size": 121,
                        "video_length": 121,
                        "slg_layers" : [11]
        })


    @staticmethod
    def get_vae_block_size(base_model_type):
        return 32

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("wan", "ti2v_2_2")
