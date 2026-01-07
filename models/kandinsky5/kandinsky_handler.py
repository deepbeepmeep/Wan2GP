import os
import torch
from omegaconf import OmegaConf
from shared.utils import files_locator as fl


def get_kandinsky_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_bf16.safetensors"
    if text_encoder_quantization == "int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8")
    return fl.locate_file(text_encoder_filename, True)


_MAGCACHE_RATIOS_CACHE = {}


def _load_magcache_ratios(config_name):
    ratios = _MAGCACHE_RATIOS_CACHE.get(config_name)
    if ratios is not None:
        return ratios
    config_path = os.path.join("models", "kandinsky5", "configs", config_name)
    if not os.path.isfile(config_path):
        _MAGCACHE_RATIOS_CACHE[config_name] = None
        return None
    conf = OmegaConf.load(config_path)
    ratios = None
    if hasattr(conf, "magcache") and "mag_ratios" in conf.magcache:
        ratios = list(conf.magcache.mag_ratios)
    _MAGCACHE_RATIOS_CACHE[config_name] = ratios
    return ratios


def _select_k5_bucket(width, height, is_video):
    if width is None or height is None:
        return 512
    area = width * height
    if is_video:
        bucket_areas = {512: 512 * 768, 1024: 1024 * 1024}
    else:
        bucket_areas = {512: 512 * 512, 1024: 1024 * 1024}
    return min(bucket_areas, key=lambda res: abs(area - bucket_areas[res]))


def _is_k5_sparse(model_type, model_def):
    if model_type and "sparse" in model_type.lower():
        return True
    overrides = (model_def or {}).get("k5_config_overrides", {})
    attention = overrides.get("model", {}).get("attention", {})
    return attention.get("type") == "nabla"


def _select_k5_magcache_config(base_model_type, model_type, bucket, is_sparse):
    bucket_tag = "hd" if bucket == 1024 else "sd"
    model_type = (model_type or "").lower()
    if base_model_type == "k5_pro_t2v":
        if "10s" in model_type:
            return f"k5_pro_t2v_10s_sft_{bucket_tag}.yaml"
        return f"k5_pro_t2v_5s_sft_{bucket_tag}.yaml"
    if base_model_type == "k5_pro_i2v":
        return f"k5_pro_i2v_5s_sft_{bucket_tag}.yaml"
    if base_model_type == "k5_lite_t2v":
        if "10s" in model_type:
            return "k5_lite_t2v_10s_sft_sd.yaml"
        return "k5_lite_t2v_5s_sft_sd.yaml"
    if base_model_type == "k5_lite_i2v":
        return "k5_lite_i2v_5s_sft_sd.yaml"
    return None


def _infer_task(base_model_type):
    if not base_model_type:
        return "t2v"
    base = base_model_type.lower()
    if "i2v" in base:
        return "i2v"
    if "t2v" in base:
        return "t2v"
    if "i2i" in base:
        return "i2i"
    if "t2i" in base:
        return "t2i"
    return "t2v"


class family_handler:
    @staticmethod
    def query_supported_types():
        return [
            "k5_lite_t2v",
            "k5_lite_i2v",
            "k5_pro_t2v",
            "k5_pro_i2v",
        ]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "kandinsky5"

    @staticmethod
    def query_family_infos():
        return {
            "kandinsky5": (50, "Kandinsky 5"),
        }

    @staticmethod
    def register_lora_cli_args(parser):
        parser.add_argument(
            "--lora-dir-kandinsky5",
            type=str,
            default=os.path.join("loras", "kandinsky5"),
            help="Base path for Kandinsky 5 loras (per-architecture subfolders are used).",
        )
        parser.add_argument(
            "--lora-dir-k5-lite-t2v",
            type=str,
            default=os.path.join("loras", "k5_lite_t2v"),
            help="Path to a directory that contains Kandinsky 5 Lite T2V loras.",
        )
        parser.add_argument(
            "--lora-dir-k5-lite-i2v",
            type=str,
            default=os.path.join("loras", "k5_lite_i2v"),
            help="Path to a directory that contains Kandinsky 5 Lite I2V loras.",
        )
        parser.add_argument(
            "--lora-dir-k5-pro-t2v",
            type=str,
            default=os.path.join("loras", "k5_pro_t2v"),
            help="Path to a directory that contains Kandinsky 5 Pro T2V loras.",
        )
        parser.add_argument(
            "--lora-dir-k5-pro-i2v",
            type=str,
            default=os.path.join("loras", "k5_pro_i2v"),
            help="Path to a directory that contains Kandinsky 5 Pro I2V loras.",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args):
        base_dir = getattr(args, "lora_dir_kandinsky5", None) or os.path.join("loras", "kandinsky5")
        per_arch = {
            "k5_lite_t2v": getattr(args, "lora_dir_k5_lite_t2v", None) or os.path.join("loras", "k5_lite_t2v"),
            "k5_lite_i2v": getattr(args, "lora_dir_k5_lite_i2v", None) or os.path.join("loras", "k5_lite_i2v"),
            "k5_pro_t2v": getattr(args, "lora_dir_k5_pro_t2v", None) or os.path.join("loras", "k5_pro_t2v"),
            "k5_pro_i2v": getattr(args, "lora_dir_k5_pro_i2v", None) or os.path.join("loras", "k5_pro_i2v"),
        }
        if base_model_type in per_arch:
            return per_arch[base_model_type]
        if base_model_type and base_model_type != "kandinsky5":
            return os.path.join(base_dir, base_model_type)
        return base_dir

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        if cache_type != "mag":
            return
        skip_steps_cache.update({
            "magcache_thresh": 0,
            "magcache_K": 2,
        })
        resolution = inputs.get("resolution")
        width = height = None
        if isinstance(resolution, str) and "x" in resolution:
            width_str, height_str = resolution.split("x", 1)
            if width_str.isdigit() and height_str.isdigit():
                width = int(width_str)
                height = int(height_str)
        bucket = _select_k5_bucket(width, height, is_video=True)
        model_type = inputs.get("model_type")
        is_sparse = _is_k5_sparse(model_type, model_def)
        config_name = _select_k5_magcache_config(base_model_type, model_type, bucket, is_sparse)
        if not config_name:
            return
        ratios = _load_magcache_ratios(config_name)
        if ratios:
            skip_steps_cache.def_mag_ratios = ratios

    @staticmethod
    def query_model_def(base_model_type, model_def):
        task = _infer_task(base_model_type)
        is_video = task in ("t2v", "i2v")
        is_image = task in ("t2i", "i2i")

        profiles_dir = base_model_type or "kandinsky5"
        extra_model_def = {
            "i2v_class": task == "i2v",
            "t2v_class": task == "t2v",
            "image_outputs": is_image,
            "guidance_max_phases": 1,
            "sliding_window": False,
            "flow_shift": True,
            "mag_cache": True,
            "profiles_dir": [profiles_dir],
        }

        if is_video:
            extra_model_def.update(
                {
                    "fps": 24,
                    "frames_minimum": 5,
                    "frames_steps": 4,
                }
            )
        else:
            extra_model_def.update(
                {
                    "fps": 1,
                    "frames_minimum": 1,
                    "frames_steps": 1,
                }
            )

        if task in ("i2v", "i2i"):
            extra_model_def["image_prompt_types_allowed"] = "S"
        else:
            extra_model_def["image_prompt_types_allowed"] = ""

        return extra_model_def

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        text_encoder_filename = get_kandinsky_text_encoder_filename(text_encoder_quantization)
        return [
            {
                "repoId": "DeepBeepMeep/Qwen_image",
                "sourceFolderList": ["", "Qwen2.5-VL-7B-Instruct"],
                "fileList": [
                    ["qwen_vae.safetensors", "qwen_vae_config.json"],
                    [
                        "merges.txt",
                        "tokenizer_config.json",
                        "config.json",
                        "vocab.json",
                        "video_preprocessor_config.json",
                        "preprocessor_config.json",
                        "chat_template.json",
                    ]
                    + computeList(text_encoder_filename),
                ],
            },
            {
                "repoId": "DeepBeepMeep/HunyuanVideo",
                "sourceFolderList": ["clip_vit_large_patch14", ""],
                "fileList": [
                    [
                        "config.json",
                        "merges.txt",
                        "model.safetensors",
                        "preprocessor_config.json",
                        "special_tokens_map.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                    ],
                    [
                        "hunyuan_video_VAE_fp32.safetensors",
                        "hunyuan_video_VAE_config.json",
                    ],
                ],
            },
        ]

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
        from .kandinsky_main import model_factory

        kandinsky = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=override_text_encoder,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
        )

        pipe = {
            "transformer": kandinsky.transformer,
            "text_encoder": kandinsky.text_embedder.embedder.model,
            "text_encoder_2": kandinsky.text_embedder.clip_embedder.model,
            "vae": kandinsky.vae,
        }
        for module in pipe.values():
            if isinstance(module, torch.nn.Module):
                module.to("cpu")
        return kandinsky, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        task = _infer_task(base_model_type)
        if task in ("t2i", "i2i"):
            ui_defaults["image_mode"] = 1
        if task in ("i2v", "i2i"):
            ui_defaults["image_prompt_type"] = "S"

        ui_defaults["skip_steps_start_step_perc"] = 20

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("hunyuan")
