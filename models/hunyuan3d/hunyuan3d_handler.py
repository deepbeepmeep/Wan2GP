# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch

from .prompts import (ASSET_NAME_INSTRUCTIONS, DEEPY_PROMPT_INFOS, PROMPT_INFOS, SHAPE_DEEPY_INFOS, SHAPE_INFOS, TEXTURE_DEEPY_INFOS,
                      TEXTURE_INFOS)

SHAPE_ARCH = "hunyuan3d_21_shape"
TEXTURE_ARCH = "hunyuan3d_21"
REPO = "DeepBeepMeep/3D"
PROJECT = "hunyuan3d"
SHAPE_FILES = ["hunyuan3d_21_shape_vae_fp16.safetensors", "dinov2_large_fp16.safetensors"]
PAINT_FILES = ["hunyuan3d_21_paint_vae_fp16.safetensors", "dinov2_giant_fp16.safetensors", "realesrgan_x4plus_fp16.safetensors"]
MESH_FILE_TYPES = [".glb", ".obj", ".ply", ".stl"]
MODES = [("Shape + PBR Texture", "shape_texture"), ("Shape Only", "shape"), ("Texture a Mesh", "texture")]
TEXTURE_VIEWS = [("6 Views at 512 px (fast)", "6_512"), ("8 Views at 768 px", "8_768"), ("9 Views at 768 px", "9_768")]


def _custom_settings(textured):
    settings = [
        {"id": "octree_resolution", "name": "Mesh Resolution", "label": "Mesh Resolution (surface grid)", "type": "dropdown", "default": 256,
         "choices": [("196 (Fast)", 196), ("256 (Standard)", 256), ("384 (High)", 384), ("512 (Very High)", 512)],
         "info": "Resolution of the grid the surface is extracted from: higher values keep finer details and give more faces."},
        {"id": "max_faces", "name": "Max Faces", "label": "Max Faces (0 = keep all)" if not textured else "Max Faces (textured meshes, 0 = 40000)",
         "type": "int", "min": 0, "max": 1000000, "inc": 1000, "default": 40000 if textured else 0,
         "info": "Reduces the mesh to this number of faces (quadric decimation)."},
    ]
    if textured:
        settings += [
            {"id": "texture_views", "name": "Texture Views", "label": "Texture Views", "type": "dropdown", "default": "8_768", "choices": TEXTURE_VIEWS,
             "info": "Views generated to paint the texture: more views and 768 px cover the object better and look sharper, 6 at 512 px is faster."},
            {"id": "texture_size", "name": "Texture Size", "label": "Texture Size", "type": "dropdown", "default": 2048,
             "choices": [("1024", 1024), ("2048", 2048), ("4096", 4096)], "info": "Size of the PBR texture images."},
        ]
    return settings


class family_handler:
    @staticmethod
    def query_supported_types():
        return [SHAPE_ARCH, TEXTURE_ARCH]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "hunyuan3d"

    @staticmethod
    def query_family_infos():
        return {"hunyuan3d": (1300, "Hunyuan3D")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        textured = base_model_type == TEXTURE_ARCH
        extra_model_def = {
            "image_outputs": True,
            "mesh_outputs": True,
            "device_explicit": True,
            "dtype": "fp16",
            "image_ref_choices": {"choices": [("Object Image", "I")], "letters_filter": "I", "default": "I"},
            "one_image_ref_needed": True,
            "one_image_ref_only": True,
            "image_refs_keep_alpha": True,
            "no_lora": True,
            "prompt_description": "Asset Name / Description (optional, used for file names)",
            "prompt_optional": True,  # the prompt only names the output files
            "no_negative_prompt": True,
            "guidance_max_phases": 1,
            "embedded_guidance": False,
            "NAG": False,
            "inference_steps": True,
            "profiles_dir": [base_model_type],
            "custom_settings": _custom_settings(textured),
            "prompt_enhancer_def": {"selection": ["TI"], "labels": {"TI": "An Asset Name from the {image_inputs}"}, "default": ""},
            "image_prompt_enhancer_instructions": ASSET_NAME_INSTRUCTIONS,
            "image_prompt_enhancer_max_tokens": 32,
            "infos": TEXTURE_INFOS if textured else SHAPE_INFOS,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": TEXTURE_DEEPY_INFOS if textured else SHAPE_DEEPY_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
            "specialities": [{"name": "image to 3D", "aliases": ["3D mesh", "GLB", "3D asset"], "description": "Turns one picture of an object into a 3D mesh."}],
        }
        if textured:
            extra_model_def.update({
                "model_modes": {"choices": MODES, "default": "shape_texture", "label": "Mode"},
                "alt_guidance": "Texture Guidance",
                "custom_guide": {"id": "custom_guide", "name": "Mesh", "label": "Input Mesh (Texture a Mesh)", "type": "file",
                                 "default": None, "required": False, "file_types": MESH_FILE_TYPES},
            })
            extra_model_def["specialities"].append({"name": "PBR texturing", "aliases": ["textured mesh", "texture a mesh", "retexture"],
                                                    "description": "Paints a generated or supplied mesh with base colour and metallic-roughness textures matching a picture."})
        return extra_model_def

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        files = SHAPE_FILES + (PAINT_FILES if base_model_type == TEXTURE_ARCH else [])
        return [{"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [files]}]

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if inputs.get("model_mode") == "texture" and inputs.get("custom_guide") is None:
            return "Texture a Mesh needs an Input Mesh (.glb, .obj, .ply or .stl)."
        return None

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, submodel_no_list=None, **kwargs):
        from .hunyuan3d_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        pipe = {"transformer": model.transformer, "vae": model.vae, "image_encoder": model.image_encoder}
        if model.paint is not None:
            pipe.update({"transformer2": model.paint.unet, "reference_unet": model.paint.unet_dual, "paint_vae": model.paint_vae,
                         "image_encoder2": model.paint_image_encoder, "upscaler": model.upscaler})
        return model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "batch_size": 1,
            "video_prompt_type": "I",
            "remove_background_images_ref": 1,
            "prompt_enhancer": "",
        })
        if base_model_type == TEXTURE_ARCH:
            ui_defaults.update({"model_mode": "shape_texture", "alt_guidance_scale": 3.0})
