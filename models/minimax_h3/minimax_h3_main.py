"""Model construction and checkpoint loading for MiniMax H3."""

import os

import torch
from accelerate import init_empty_weights

from mmgp import offload, quant_router
from shared.utils import files_locator as fl

from .audio_vae import MiniMaxH3AudioVAE
from .convrot_layout import get_convrot_layout, restore_interleaved_h3_qkv
from .pipeline import MiniMaxH3Pipeline
from .text_encoder import MiniMaxH3TextEncoder, load_h3_qwen_config
from .transformer import MiniMaxH3Model, get_linear_split_map
from .video_vae import MiniMaxH3VideoVAE, get_video_vae_linear_split_map


VIDEO_VAE_FILE = "MiniMax-H3-video_vae_fp16.safetensors"
AUDIO_VAE_FILE = "MiniMax-H3-audio_vae_fp32.safetensors"
TEXT_ENCODER_FOLDER = "Qwen3-VL-32B-Instruct"
PROBE_CHECKPOINT_ARCHITECTURE = False
ADALN_CURVE_GRID = 1001
ADALN_CURVE_DIM = 64
SPLIT_QKV_PROJECTIONS = True


def _strip_wrappers(state_dict, quantization_map=None, tied_weights_map=None):
    restore_interleaved_h3_qkv(state_dict)
    prefixes = ("model.diffusion_model.", "diffusion_model.")

    def strip(mapping):
        if mapping is None:
            return None
        result = {}
        for key, value in mapping.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            result[key] = value
        return result

    return strip(state_dict), strip(quantization_map), strip(tied_weights_map)


def probe_h3_checkpoint(filename):
    """Inspect tensor headers before allocating the network."""
    state_dict, _ = quant_router.load_metadata_state_dict(filename)
    normalized = {}
    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value
    table = next((tensor for key, tensor in normalized.items() if key == "adaln_t_table"), None)
    if table is None:
        return {"compressed_modulation": False, "adaln_curve_grid": None, "time_embed_dim": 2688}
    if len(table.shape) != 2 or table.shape[0] < 2:
        raise ValueError(f"Invalid H3 AdaLN curve table shape: {tuple(table.shape)}")
    return {"compressed_modulation": True, "adaln_curve_grid": int(table.shape[0]), "time_embed_dim": int(table.shape[1])}


def _load_transformer(filename, dtype, compressed_modulation):
    checkpoint = probe_h3_checkpoint(filename) if PROBE_CHECKPOINT_ARCHITECTURE else {
        "compressed_modulation": compressed_modulation,
        "adaln_curve_grid": ADALN_CURVE_GRID if compressed_modulation else None,
        "time_embed_dim": ADALN_CURVE_DIM if compressed_modulation else 2688,
    }
    with init_empty_weights(include_buffers=True):
        transformer = MiniMaxH3Model(adaln_curve_grid=checkpoint["adaln_curve_grid"], time_embed_dim=checkpoint["time_embed_dim"],
                                     dtype=dtype, device="meta")
    split_map = get_linear_split_map(transformer.attention_inner_size) if SPLIT_QKV_PROJECTIONS else None
    if split_map is not None:
        offload.split_linear_modules(transformer, split_map)
    transformer.requires_grad_(False)
    offload.load_model_data(transformer, filename, writable_tensors=False, default_dtype=dtype, preprocess_sd=_strip_wrappers,
                            fused_split_map=split_map)
    transformer.eval().requires_grad_(False)
    transformer.h3_checkpoint_info = checkpoint
    transformer.split_linear_modules_map = split_map
    return transformer


def _load_text_encoder(filename, dtype):
    config_path = fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "config.json"))
    config = load_h3_qwen_config(config_path)
    with init_empty_weights(include_buffers=True):
        text_encoder = MiniMaxH3TextEncoder(config)
    text_encoder.visual.rotary_pos_emb.reset_inv_freq()
    text_encoder.language_model.rotary_emb.reset_inv_freq()
    state_dict, _ = quant_router.load_metadata_state_dict(filename)
    prefix = "model" if any(key.startswith("model.") for key in state_dict) else None
    offload.load_model_data(text_encoder, filename, modelPrefix=prefix, writable_tensors=False, default_dtype=dtype,
                            ignore_unused_weights=True)
    text_encoder.set_tokenizer(os.path.dirname(config_path))
    text_encoder.eval().requires_grad_(False)
    return text_encoder


def _load_video_vae(filename, dtype):
    filename = fl.locate_file(filename)
    with init_empty_weights(include_buffers=False):
        vae = MiniMaxH3VideoVAE()
    split_map = get_video_vae_linear_split_map() if SPLIT_QKV_PROJECTIONS else None
    if split_map is not None:
        offload.split_linear_modules(vae, split_map)
    def preprocess(state_dict):
        state_dict.pop("latents_mean", None)
        state_dict.pop("latents_std", None)
        return state_dict

    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=dtype, preprocess_sd=preprocess,
                            fused_split_map=split_map)
    vae.split_linear_modules_map = split_map
    vae._model_dtype = dtype
    vae.eval().requires_grad_(False)
    return vae


def _load_audio_vae(filename):
    filename = fl.locate_file(filename)
    with init_empty_weights(include_buffers=False):
        vae = MiniMaxH3AudioVAE()

    def preprocess(state_dict):
        state_dict.pop("latents_mean", None)
        state_dict.pop("latents_std", None)
        for key in list(state_dict):
            if key.endswith(".weight_g"):
                state_dict[key.removesuffix("weight_g") + "parametrizations.weight.original0"] = state_dict.pop(key)
            elif key.endswith(".weight_v"):
                state_dict[key.removesuffix("weight_v") + "parametrizations.weight.original1"] = state_dict.pop(key)
        return state_dict

    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=torch.float32, preprocess_sd=preprocess)
    vae.eval().requires_grad_(False)
    return vae


def model_factory(model_filename, text_encoder_filename, dtype=torch.bfloat16, VAE_dtype=torch.float32, save_quantized=False,
                  model_type="minimax_h3_fl2va", compressed_modulation=False, reference_mode=False, video_vae_filename=VIDEO_VAE_FILE,
                  audio_vae_filename=AUDIO_VAE_FILE):
    transformer = _load_transformer(model_filename, dtype, compressed_modulation)
    text_encoder = _load_text_encoder(text_encoder_filename, dtype)
    video_vae = _load_video_vae(video_vae_filename, VAE_dtype)
    audio_vae = _load_audio_vae(audio_vae_filename)
    pipeline = MiniMaxH3Pipeline(transformer, text_encoder, video_vae, audio_vae, reference_mode=reference_mode, dtype=dtype)
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, model_filename[0], dtype, None,
                             convrot_layout=get_convrot_layout(transformer))
    return pipeline


__all__ = ["ADALN_CURVE_DIM", "ADALN_CURVE_GRID", "AUDIO_VAE_FILE", "PROBE_CHECKPOINT_ARCHITECTURE",
           "SPLIT_QKV_PROJECTIONS", "TEXT_ENCODER_FOLDER", "VIDEO_VAE_FILE", "model_factory", "probe_h3_checkpoint"]
