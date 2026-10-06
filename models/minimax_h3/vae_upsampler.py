"""Capabilities and optional assets for MiniMax H3's X2 VAE decoder.

Checkpoint: https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE
Revision: af8c92d267c6849fec5032c35a65d5766737b338. See X2_VAE_LICENSE and X2_VAE_NOTICE.
"""

from postprocessing.spatial_upsamplers import SimpleScaleSuffixMixin


X2_VAE_FILE = "minimax_h3/MiniMax-H3-X2-Detail-v1.safetensors"
X2_VAE_INT8_FILE = "minimax_h3/MiniMax-H3-X2-Detail-v1_int8_convrot.safetensors"  # built by tools/build_h3_x2_vae_int8_convrot.py
X2_VAE_METHOD = "h3_vae"
X2_VAE_VALUE = "h3_vae*2"
X1_VAE_VALUE = "h3_vae*1"
X2_VAE_DESCRIPTION = (
    "Replaces MiniMax H3's default VAE and handles VAE decoding and upsampling together. "
    "Choose x2 to double the output width and height, or x1 to keep the original size."
)


def query_x2_vae_files(filename=X2_VAE_FILE):
    return {"repoId": "DeepBeepMeep/MiniMax-H3", "sourceFolderList": ["minimax_h3"],
            "fileList": [[filename.rsplit("/", 1)[-1]]]}


class MiniMaxH3VaeUpsampler(SimpleScaleSuffixMixin):
    def __init__(self, server_config=None, files_locator=None):
        pass

    @staticmethod
    def query_upsampler_def():
        return {"name": "MiniMax H3 VAE", "upsampler_types": ("vae",), "media": ("video", "image"),
                "profile": "video", "pos": 31, "methods": [], "vae_methods": [("MiniMax H3 VAE", X2_VAE_METHOD)],
                "multipliers": {X2_VAE_METHOD: (1.0, 2.0)}, "default_spatial_upsampling": X2_VAE_VALUE,
                "description": X2_VAE_DESCRIPTION}

    def validate_upsampling(self, value, image_mode):
        return "" if self.split_value(value) in ((X2_VAE_METHOD, 1.0), (X2_VAE_METHOD, 2.0)) else "MiniMax H3 VAE Upsampling only supports x1 and x2"

    def supports_model_vae_method(self, method, model_type, model_def, image_mode):
        return method == X2_VAE_METHOD and image_mode in model_def.get("vae_upsamplers", {}).get(X2_VAE_METHOD, ())

    def validate_model_vae_upsampling(self, value, image_mode, model_type, model_def, medium):
        error = self.validate_upsampling(value, image_mode)
        if error:
            return error
        return "" if self.supports_model_vae_method(X2_VAE_METHOD, model_type, model_def, image_mode) else f"MiniMax H3 VAE Upsampling is not available for {medium}"

    def model_load_upsampling_value(self, value, model_type, model_def, image_mode):
        # Both scales share the X2 decoder; switching scale must not reload its weights.
        return X2_VAE_VALUE if self.supports_model_vae_method(X2_VAE_METHOD, model_type, model_def, image_mode) else None

    @staticmethod
    def loaded_model_vae_upsampling_value(model):
        return X2_VAE_VALUE if model is not None and getattr(getattr(model, "vae", None), "upsampling_set", None) == X2_VAE_VALUE else None

    @staticmethod
    def model_load_kwargs_for_vae_upsampling(value, model_type, model_def, image_mode):
        return {"VAE_upsampling": X2_VAE_VALUE}

    def post_model_process_vae_upsampling(self, sample, spatial_upsampling):
        if self.split_value(spatial_upsampling)[1] == 1.0:
            from PIL import Image
            from postprocessing.lanczos import resize_lanczos_spatial

            return resize_lanczos_spatial(sample, 0.5, method=Image.Resampling.BICUBIC)
        return sample
