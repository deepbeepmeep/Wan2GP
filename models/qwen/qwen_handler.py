import os
import torch
import gradio as gr
from shared.utils.hf import build_hf_url

class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "image_outputs" : True,
            "sample_solvers":[
                            ("Default", "default"),
                            ("Lightning", "lightning")],
            "guidance_max_phases" : 1,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": ["qwen"],
        }
        text_encoder_folder = "Qwen2.5-VL-7B-Instruct"
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("DeepBeepMeep/Qwen_image", text_encoder_folder, "Qwen2.5-VL-7B-Instruct_bf16.safetensors"),
            build_hf_url("DeepBeepMeep/Qwen_image", text_encoder_folder, "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = text_encoder_folder

        extra_model_def["vae_upsampler"] = [1,2]

        if base_model_type in ["qwen_image_layered_20B"]:
            extra_model_def["batch_size_label"] = "Number of Layers"
            extra_model_def["set_video_prompt_type"] = "V"
            extra_model_def["guide_preprocessing"] = {
                "selection": ["V"],
                "labels": {"V": "Control Image"},
                "visible": False,
            }
            extra_model_def["vae_upsampler"] = [1]
            extra_model_def["sample_solvers"] = [("Default", "default")]

        if base_model_type in ["qwen_image_20B"]:
            extra_model_def["inpaint_support"] = True
            extra_model_def["inpaint_video_prompt_type"] = "VA"
            extra_model_def["image_video_prompt_type"] = ""            
            extra_model_def["video_guide_outpainting"] = [2]
            extra_model_def["model_modes"] = {
                        "choices": [
                            ("LanPaint (2 steps): ~2x slower, easy task", 2),
                            ("LanPaint (5 steps): ~5x slower, medium task", 3),
                            ("LanPaint (10 steps): ~10x slower, hard task", 4),
                            ("LanPaint (15 steps): ~15x slower, very hard task", 5),
                            ],
                        "default": 2,
                        "label" : "Inpainting Method",
                        "image_modes" : [2],
            }

        if base_model_type in ["qwen_image_edit_20B", "qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]:
            extra_model_def["inpaint_support"] = True
            if base_model_type in ["qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]:
                extra_model_def["inpaint_video_prompt_type"]= "VAGI"            
            extra_model_def["image_ref_inpaint"]=  base_model_type in ["qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]
            extra_model_def["image_ref_choices"] = {
            "choices": [
                ("None", ""),
                ("Conditional Image is first Main Subject / Landscape and may be followed by People / Objects", "KI"),
                ("Conditional Images are People / Objects", "I"),
                ],
            "letters_filter": "KI",
            }
            extra_model_def["background_removal_label"]= "Remove Backgrounds only behind People / Objects except main Subject / Landscape" 
            extra_model_def["video_guide_outpainting"] = [2]
            extra_model_def["model_modes"] = {
                        "choices": [
                            ("Lora Inpainting: Inpainted area completely unrelated to masked content", 1),
                            ("Masked Denoising : Inpainted area may reuse some content that has been masked", 0),
                            ("LanPaint (2 steps): ~2x slower, easy task", 2),
                            ("LanPaint (5 steps): ~5x slower, medium task", 3),
                            ("LanPaint (10 steps): ~10x slower, hard task", 4),
                            ("LanPaint (15 steps): ~15x slower, very hard task", 5),
                            ],
                        "default": 1,
                        "label" : "Inpainting Method",
                        "image_modes" : [2],
            }
            extra_model_def["inpaint_color"] = "FF0000"

        if base_model_type in ["qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]:
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["", "PV", "DV", "SV", "CV", "V"], #, "MV" 
                    "labels": {"V": "Qwen Raw Format"},
                }

            extra_model_def["mask_strength_always_enabled"] = True

            extra_model_def["mask_preprocessing"] = {
                    "selection": ["", "A"],
                    "visible": True,
                }
        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["qwen_image_20B", "qwen_image_edit_20B", "qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B", "qwen_image_layered_20B"]

    @staticmethod
    def query_family_maps():
        models_eqv_map = {
            "qwen_image_edit_plus2_20B": "qwen_image_edit_plus_20B",
        }
        models_comp_map = {
            "qwen_image_edit_plus_20B": ["qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"],
        }
        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "qwen"

    @staticmethod
    def query_family_infos():
        return {"qwen":(110, "Qwen")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-qwen",
            type=str,
            default=None,
            help=f"Path to a directory that contains qwen images Loras (default: {os.path.join(lora_root, 'qwen')})"
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_qwen", None) or os.path.join(lora_root, "qwen")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        vae_files = ["qwen_vae.safetensors", "qwen_vae_config.json"]
        if base_model_type in ["qwen_image_layered_20B"]:
            vae_files.append("qwen_image_layered_vae_bf16.safetensors")
        download_def = [{  
            "repoId" : "DeepBeepMeep/Qwen_image", 
            "sourceFolderList" :  ["", "Qwen2.5-VL-7B-Instruct"],
            "fileList" : [ vae_files, ["merges.txt", "tokenizer_config.json", "config.json", "vocab.json", "video_preprocessor_config.json", "preprocessor_config.json", "chat_template.json"]  ]
            }]

        download_def  += [{
            "repoId" : "DeepBeepMeep/Wan2.1", 
            "sourceFolderList" :  [""  ],
            "fileList" : [ ["Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"]  ]   
        }]
        return download_def

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False, submodel_no_list = None, text_encoder_filename = None, VAE_upsampling = None, **kwargs):
        from .qwen_main import model_factory
        from mmgp import offload

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type = model_type, 
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized,
            VAE_upsampling = VAE_upsampling,
        )

        from ..wan.modules.vae import WanVAE
        pipe = {"tokenizer" : pipe_processor.tokenizer, "transformer" : pipe_processor.transformer, "text_encoder" : pipe_processor.text_encoder, "vae" : pipe_processor.vae.model if isinstance(pipe_processor.vae, WanVAE) else pipe_processor.vae }

        return pipe_processor, pipe


    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if ui_defaults.get("sample_solver", "") == "": 
            ui_defaults["sample_solver"] = "default"

        if settings_version < 2.32:
            ui_defaults["denoising_strength"] = 1.
                            
    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "guidance_scale":  4,
            "sample_solver": "default",
        })            
        if base_model_type in ["qwen_image_edit_20B"]: 
            ui_defaults.update({
                "video_prompt_type": "KI",
                "denoising_strength" : 1.,
                "model_mode" : 0,
            })
        elif base_model_type in ["qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]:
            ui_defaults.update({
                "video_prompt_type": "",
                "denoising_strength" : 1.,
                "model_mode" : 0,
            })
        elif base_model_type in ["qwen_image_layered_20B"]:
            ui_defaults.update({
                "video_prompt_type": "V",
            })

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type in ["qwen_image_20B", "qwen_image_edit_20B", "qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]:
            model_mode = inputs["model_mode"]
            denoising_strength = inputs["denoising_strength"]
            masking_strength = inputs["masking_strength"]
            model_mode_int = None
            if model_mode is not None:
                try:
                    model_mode_int = int(model_mode)
                except (TypeError, ValueError):
                    model_mode_int = None

            if model_mode_int in (2, 3, 4, 5):
                if denoising_strength != 1 or masking_strength != 1:
                    gr.Info("LanPaint forces Denoising Strength and Masking Strength to 1; non-1 values will be ignored.")
            elif denoising_strength < 1 and model_mode_int != 0:
                gr.Info("Denoising Strength will be ignored if Masked Denoising is not used")

        if base_model_type in ["qwen_image_layered_20B"]:
            if inputs.get("image_guide") is None:
                return "Qwen Image Layered requires a Control Image."

    @staticmethod
    def custom_prompt_preprocess(prompt, video_guide_outpainting, model_mode, **kwargs):
        if model_mode == 0:
            # from wgp import get_outpainting_dims
            if len(video_guide_outpainting) and not video_guide_outpainting.startswith("#") and video_guide_outpainting != "0 0 0 0":
                if not prompt.endswith("."): prompt += "."
                prompt += "Remove the red paddings on the sides and show what's behind them."
        return prompt  


    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("qwen")
        return latent_rgb_factors, latent_rgb_factors_bias
