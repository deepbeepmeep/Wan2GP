import torch

def get_qwen_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "ckpts/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_bf16.safetensors"
    if text_encoder_quantization =="int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8") 
    return text_encoder_filename

class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        model_def_output = {
            "image_outputs" : True,
            "sample_solvers":[
                            ("Default", "default"),
                            ("Lightning", "lightning")],
            "guidance_max_phases" : 1,
        }


        return model_def_output

    @staticmethod
    def query_supported_types():
        return ["qwen_image_20B", "qwen_image_edit_20B"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "qwen"

    @staticmethod
    def query_family_infos():
        return {"qwen":(40, "Qwen")}

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_qwen_text_encoder_filename(text_encoder_quantization)    
        return  {  
            "repoId" : "DeepBeepMeep/Qwen_image", 
            "sourceFolderList" :  ["", "Qwen2.5-VL-7B-Instruct"],
            "fileList" : [ ["qwen_vae.safetensors", "qwen_vae_config.json"], ["merges.txt", "tokenizer_config.json", "config.json", "vocab.json", "video_preprocessor_config.json", "preprocessor_config.json"] + computeList(text_encoder_filename)  ]
            }

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False):
        from .qwen_main import model_factory
        from mmgp import offload

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type = model_type, 
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= get_qwen_text_encoder_filename(text_encoder_quantization),
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = {"tokenizer" : pipe_processor.tokenizer, "transformer" : pipe_processor.transformer, "text_encoder" : pipe_processor.text_encoder, "vae" : pipe_processor.vae}

        return pipe_processor, pipe


    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if ui_defaults.get("sample_solver", "") == "": 
            ui_defaults["sample_solver"] = "default"

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "guidance_scale":  4,
            "sample_solver": "default",
        })            
        if model_def.get("reference_image", False):
            ui_defaults.update({
                "video_prompt_type": "KI",
            })

