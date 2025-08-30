import torch

def get_ltxv_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors"
    if text_encoder_quantization =="int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8") 
    return text_encoder_filename

class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        flux_model = model_def.get("flux-model", "flux-dev")
        flux_schnell = flux_model == "flux-schnell" 
        flux_chroma = flux_model == "flux-chroma" 
        flux_uso = flux_model == "flux-dev-uso"
        model_def_output = {
            "image_outputs" : True,
            "no_negative_prompt" : not flux_chroma,
        }
        if flux_chroma:
            model_def_output["guidance_max_phases"] = 1
        elif not flux_schnell:
            model_def_output["embedded_guidance"] = True
        if flux_uso :
            model_def_output["any_image_refs_relative_size"] = True
            model_def_output["no_background_removal"] = True

            model_def_output["image_ref_choices"] = {
                "choices":[("No Reference Image", ""),("First Image is a Reference Image, and then the next ones (up to two) are Style Images", "I"),
                            ("Up to two Images are Style Images", "IJ")],
                "default": "I",
                "letters_filter": "IJ",
                "label": "Reference Images / Style Images"
            }

        return model_def_output

    @staticmethod
    def query_supported_types():
        return ["flux"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("flux")
        return latent_rgb_factors, latent_rgb_factors_bias


    @staticmethod
    def query_model_family():
        return "flux"

    @staticmethod
    def query_family_infos():
        return {"flux":(30, "Flux 1")}

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_ltxv_text_encoder_filename(text_encoder_quantization)    
        return [
            {  
            "repoId" : "DeepBeepMeep/Flux", 
            "sourceFolderList" :  ["siglip-so400m-patch14-384", "",],
            "fileList" : [ ["config.json", "preprocessor_config.json", "model.safetensors"], ["flux_vae.safetensors"] ]   
            },
            {  
            "repoId" : "DeepBeepMeep/LTX_Video", 
            "sourceFolderList" :  ["T5_xxl_1.1"],
            "fileList" : [ ["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"] + computeList(text_encoder_filename)  ]   
            },
            {  
            "repoId" : "DeepBeepMeep/HunyuanVideo", 
            "sourceFolderList" :  [  "clip_vit_large_patch14",   ],
            "fileList" :[ 
                            ["config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
                            ]
            } 
        ]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False):
        from .flux_main  import model_factory

        flux_model = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type = model_type, 
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= get_ltxv_text_encoder_filename(text_encoder_quantization),
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = { "transformer": flux_model.model, "vae" : flux_model.vae, "text_encoder" : flux_model.clip, "text_encoder_2" : flux_model.t5}

        if flux_model.vision_encoder is not None:
            pipe["siglip_model"] = flux_model.vision_encoder 
        if flux_model.feature_embedder is not None:
            pipe["feature_embedder"] = flux_model.feature_embedder 
        return flux_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        flux_model = model_def.get("flux-model", "flux-dev")
        flux_uso = flux_model == "flux-dev-uso"
        ui_defaults.update({
            "embedded_guidance":  2.5,
        })            
        if model_def.get("reference_image", False):
            ui_defaults.update({
                "video_prompt_type": "I" if flux_uso else "KI",
            })

