"""Shared RGB Qwen Image VAE choices, downloads and checkpoint-native loading."""


VAE_FILES = {
    "real": "krea2_real_vae_fp32.safetensors",
    "hd": "krea2_hd_vae_fp32.safetensors",
}
UPSCALE_VAE_FILE = "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"


def vae_configs():
    return {
        "_name": "VAE",
        "_default_label": "Qwen Image (Default)",
        "real": {"name": "Krea 2 Real", "qwen_vae_variant": "real"},
        "hd": {"name": "Krea 2 HD", "qwen_vae_variant": "hd"},
    }


def query_vae_files(model_def):
    variant = model_def.get("qwen_vae_variant", "default")
    files = ["qwen_vae_config.json"]
    if variant == "default":
        files.insert(0, "qwen_vae.safetensors")
    downloads = [{"repoId": "DeepBeepMeep/Qwen_image", "sourceFolderList": [""], "fileList": [files]}]
    if variant != "default":
        downloads.append({"repoId": "DeepBeepMeep/krea-2", "sourceFolderList": ["krea2"], "fileList": [[VAE_FILES[variant]]]})
    downloads.append({"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": [""], "fileList": [[UPSCALE_VAE_FILE]]})
    return downloads


def load_vae(model_def, VAE_upsampling=None):
    from mmgp import offload
    from shared.utils import files_locator as fl
    from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    from .convert_diffusers_qwen_vae import convert_state_dict

    variant = model_def.get("qwen_vae_variant", "default")
    if VAE_upsampling is not None:
        filename, factor, preprocess = UPSCALE_VAE_FILE, 2, convert_state_dict
    elif variant == "default":
        filename, factor, preprocess = "qwen_vae.safetensors", 1, None
    else:
        filename, factor, preprocess = "krea2/" + VAE_FILES[variant], 1, convert_state_dict
    vae = offload.fast_load_transformers_model(fl.locate_file(filename), writable_tensors=False, modelClass=AutoencoderKLQwenImage, defaultConfigPath=fl.locate_file("qwen_vae_config.json"), default_dtype=None, configKwargs={"upsampler_factor": factor}, preprocess_sd=preprocess)
    vae._convertWeightsFloatTo = None
    vae.upsampling_set = VAE_upsampling
    return vae
