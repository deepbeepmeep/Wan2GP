import os
import torch
from shared.utils import files_locator as fl 


def get_ltxv_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors"
    if text_encoder_quantization =="int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8") 
    return fl.locate_file(text_encoder_filename, True)

class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        LTXV_config = model_def.get("LTXV_config", "")
        distilled= "distilled" in LTXV_config 
        extra_model_def = {}
        if distilled:
            extra_model_def.update({
            "lock_inference_steps": True,
            "no_negative_prompt" : True,
        })


        extra_model_def["fps"] = 30
        extra_model_def["frames_minimum"] = 17
        extra_model_def["frames_steps"] = 8
        extra_model_def["sliding_window"] = True
        extra_model_def["image_prompt_types_allowed"] = "TSEV"

        extra_model_def["guide_preprocessing"] = {
            "selection": ["", "PV", "DV", "EV", "V"],
            "labels" : { "V": "Use LTXV raw format"}
        }

        extra_model_def["mask_preprocessing"] = {
            "selection": ["", "A", "NA", "XA", "XNA"],
        }

        extra_model_def["extra_control_frames"] = 1
        extra_model_def["dont_cat_preguide"]= True
        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["ltxv_13B"]
    
    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("ltxv")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def query_model_family():
        return "ltxv"

    @staticmethod
    def query_family_infos():
        return {"ltxv":(10, "LTX Video")}

    @staticmethod
    def register_lora_cli_args(parser):
        parser.add_argument(
            "--lora-dir-ltxv",
            type=str,
            default=os.path.join("loras", "ltxv"),
            help="Path to a directory that contains LTX Videos Loras"
        )

    @staticmethod
    def get_lora_dir(base_model_type, args):
        return args.lora_dir_ltxv

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 32

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_ltxv_text_encoder_filename(text_encoder_quantization)    
        return {
            "repoId" : "DeepBeepMeep/LTX_Video", 
            "sourceFolderList" :  ["T5_xxl_1.1",  ""  ],
            "fileList" : [ ["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"] + computeList(text_encoder_filename), ["ltxv_0.9.7_VAE.safetensors", "ltxv_0.9.7_spatial_upscaler.safetensors", "ltxv_scheduler.json"] + computeList(model_filename) ]   
        }


    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False, submodel_no_list = None, override_text_encoder = None):
        from .ltxv import LTXV

        ltxv_model = LTXV(
            model_filepath = model_filename,
            text_encoder_filepath = get_ltxv_text_encoder_filename(text_encoder_quantization) if override_text_encoder is None else override_text_encoder,
            model_type = model_type, 
            base_model_type = base_model_type,
            model_def = model_def,
            dtype = dtype,
            # quantizeTransformer = quantizeTransformer,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer
        )

        pipeline = ltxv_model.pipeline 
        pipe = {"transformer" : pipeline.video_pipeline.transformer, "vae" : pipeline.vae, "text_encoder" : pipeline.video_pipeline.text_encoder, "latent_upsampler" : pipeline.latent_upsampler}

        return ltxv_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        pass
   
