import os
import torch
from shared.utils import files_locator as fl


def get_z_image_text_encoder_filename(text_encoder_quantization=None):
    text_encoder_filename = "Qwen3/qwen3_bf16.safetensors"
    if text_encoder_quantization =="int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8") 
    return fl.locate_file(text_encoder_filename, True)


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "image_outputs": True,
            "guidance_max_phases": 0,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [],
            "no_negative_prompt": True,
        }

        if base_model_type == "z_image_control":
            extra_model_def["mask_preprocessing"] = {
                "selection":[ ""],
                "visible": False
            }

            extra_model_def["control_net_weight_name"] = "Control"
            extra_model_def["control_net_weight_size"] = 1

            extra_model_def["guide_preprocessing"] = {
                "selection": ["", "PV", "DV", "EV", "V"],
                "labels" : { "V": "Use Z-Image Raw Format"},
            }

        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["z_image", "z_image_control"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "z_image"

    @staticmethod
    def query_family_infos():
        return {"z_image": (50, "Z-Image"), "z_image_control": (51, "Z-Image ControlNet")}

    @staticmethod
    def register_lora_cli_args(parser):
        parser.add_argument(
            "--lora-dir-z-image",
            type=str,
            default=os.path.join("loras", "z_image"),
            help="Path to a directory that contains z image settings"
        )

    @staticmethod
    def get_lora_dir(base_model_type, args):
        return args.lora_dir_z_image

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_z_image_text_encoder_filename(text_encoder_quantization)
        download_def = [
            {
                "repoId": "DeepBeepMeep/Z-Image",
                "sourceFolderList": ["Qwen3", ""],
                "fileList": [                    
                    ["tokenizer.json", "tokenizer_config.json", "vocab.json", "config.json", "merges.txt"]+ computeList(text_encoder_filename),
                    ["ZImageTurbo_VAE_bf16_config.json", "ZImageTurbo_VAE_bf16.safetensors", "ZImageTurbo_scheduler_config.json"],                
                ],
            }
        ]
        return download_def

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
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
        from .z_image_main import model_factory

        text_encoder_filename = (
            override_text_encoder if override_text_encoder is not None else get_z_image_text_encoder_filename(text_encoder_quantization)
        )

        # Detect if this is the control variant
        is_control = base_model_type == "z_image_control"

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
            is_control=is_control,
        )

        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.vae,
        }
        return pipe_processor, pipe

    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("flux")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "guidance_scale": 0.0,
                "num_inference_steps": ui_defaults.get("num_inference_steps", 9),
            }
        )

        # Add control defaults for z_image_control
        if base_model_type == "z_image_control":
            ui_defaults.update(
                {
                    "control_net_weight":  0.75,
                }
            )
