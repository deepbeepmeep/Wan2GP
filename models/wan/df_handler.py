import torch

class family_handler():

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        if base_model_type == "sky_df_1.3B":
            coefficients= [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        else: 
            coefficients= [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]

        skip_steps_cache.coefficients = coefficients

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {}
        if base_model_type in ["sky_df_14B"]:
            fps = 24
        else:
            fps = 16
        extra_model_def["fps"] =fps
        extra_model_def["frames_minimum"] = 17
        extra_model_def["frames_steps"] = 20
        extra_model_def["latent_size"] = 4
        extra_model_def["sliding_window"] = True
        extra_model_def["skip_layer_guidance"] = True
        extra_model_def["tea_cache"] = True
        extra_model_def["guidance_max_phases"] = 1
        extra_model_def["flow_shift"] = True
        extra_model_def["model_modes"] = {
                    "choices": [
                        ("Synchronous", 0),
                        ("Asynchronous (better quality but around 50% extra steps added)", 5),
                        ],
                    "default": 0,
                    "label" : "Generation Type"
        }

        extra_model_def["image_prompt_types_allowed"] = "TSV"


        return extra_model_def 

    @staticmethod
    def query_supported_types():
        return ["sky_df_1.3B", "sky_df_14B"]


    @staticmethod
    def query_family_maps():
        models_eqv_map = {
            "sky_df_1.3B" : "sky_df_14B",
        }

        models_comp_map = { 
                    "sky_df_14B": ["sky_df_1.3B"],
                    }
        return models_eqv_map, models_comp_map



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
    def get_lora_dir(base_model_type, args):
        from .wan_handler import family_handler as wan_family_handler

        return wan_family_handler.get_lora_dir(base_model_type, args)

    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("wan", base_model_type)
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        from .wan_handler import family_handler
        return family_handler.query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization)
    
    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized= False, submodel_no_list = None, override_text_encoder = None):
        from .configs import WAN_CONFIGS
        from .wan_handler import family_handler
        cfg = WAN_CONFIGS['t2v-14B']
        from . import DTT2V
        wan_model = DTT2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type = model_type,        
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= family_handler.get_wan_text_encoder_filename(text_encoder_quantization) if override_text_encoder is None else override_text_encoder,
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model, "vae": wan_model.vae.model }
        return wan_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "guidance_scale": 6.0,
            "flow_shift": 8,
            "sliding_window_discard_last_frames" : 0,
            "resolution": "1280x720" if "720" in base_model_type else "960x544",
            "sliding_window_size" : 121 if "720" in base_model_type else 97,
            "RIFLEx_setting": 2,
            "guidance_scale": 6,
            "flow_shift": 8,
        })
