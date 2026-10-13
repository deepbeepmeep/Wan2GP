# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from .prompts import TRELLIS2_DEEPY_INFOS, TRELLIS2_DEEPY_PROMPT_INFOS, TRELLIS2_INFOS, TRELLIS2_PROMPT_INFOS
from .settings import CUSTOM_SETTINGS, MODEL_MODES

ARCHITECTURE = "trellis2"
REPO = "DeepBeepMeep/3D"


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "trellis2"

    @staticmethod
    def query_family_infos():
        return {"trellis2": (3020, "TRELLIS.2")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "image_outputs": True,
            "mesh_outputs": True,
            "device_explicit": True,
            "dtype": "bf16",
            "image_ref_choices": {"choices": [("Object Image", "I")], "letters_filter": "I", "default": "I"},
            "one_image_ref_needed": True,
            "one_image_ref_only": True,
            "image_refs_keep_alpha": True,
            "fit_into_canvas_image_refs": 0,
            "prompt_description": "Asset Name / Description (optional, used for file names)",
            "prompt_optional": True,  # the prompt only names the output files
            "no_negative_prompt": True,
            "embedded_guidance": False,
            "inference_steps": True,
            "guidance_max_phases": 1,
            "alt_guidance": "Texture Guidance",
            "model_modes": {**MODEL_MODES, "choices": list(MODEL_MODES["choices"])},
            "custom_settings": [one.copy() for one in CUSTOM_SETTINGS],
            "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
            "profiles_dir": [base_model_type],
            "specialities": [{"name": "image to 3D", "aliases": ["3D mesh", "GLB", "3D asset"], "description": "Turns one picture of an object into a 3D mesh."},
                             {"name": "PBR textured 3D asset", "aliases": ["textured 3D model", "textured GLB", "image to textured 3D", "game asset"],
                              "description": "Turns one picture of an object into a GLB with baked PBR textures (base colour, metallic, roughness) in one pass, "
                                             "keeping thin and intricate structures. Prefer Hunyuan3D to texture an existing mesh, TripoSG for an untextured "
                                             "shape and TripoSplat for a Gaussian splat."}],
            "infos": TRELLIS2_INFOS,
            "prompt_infos": TRELLIS2_PROMPT_INFOS,
            "deepy_infos": TRELLIS2_DEEPY_INFOS,
            "deepy_prompt_infos": TRELLIS2_DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": REPO,
            "sourceFolderList": ["trellis2", "dinov3_vitl16"],
            "fileList": [["trellis2_ss_decoder_fp16.safetensors", "trellis2_shape_decoder_fp16.safetensors", "trellis2_texture_decoder_fp16.safetensors"],
                         ["dinov3_vitl16_lvd1689m.safetensors", "LICENSE_DINOv3.md"]],
        }]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, **kwargs):
        from .trellis2_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        return model, model.pipe()

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "video_prompt_type": "I",
            "num_inference_steps": 12,
            "guidance_scale": 7.5,
            "alt_guidance_scale": 1.0,
            "model_mode": MODEL_MODES["default"],
            "batch_size": 1,
            "remove_background_images_ref": 1,
        })
