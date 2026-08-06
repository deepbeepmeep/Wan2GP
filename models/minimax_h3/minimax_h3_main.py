"""Model construction and checkpoint loading for MiniMax H3."""

import os
from functools import partial

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
ADALN_CURVE_DIM = 8


def _strip_wrappers(state_dict, quantization_map=None, tied_weights_map=None, qkv_splitting=True):
    if qkv_splitting:
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


def _strip_text_encoder_wrapper(state_dict, quantization_map=None, tied_weights_map=None):
    # Strip Comfy quantization metadata markers
    keys_to_remove = [k for k in list(state_dict.keys()) if k.endswith(".comfy_quant")]
    for k in keys_to_remove:
        del state_dict[k]

    # Dequantize INT8 embedding to BF16 so MMGP sees consistent dtypes
    embed_weight_key = "model.embed_tokens.weight"
    embed_scale_key = "model.embed_tokens.weight_scale"
    if embed_weight_key in state_dict and embed_scale_key in state_dict:
        w = state_dict[embed_weight_key]
        s = state_dict[embed_scale_key]
        if w.dtype == torch.int8:
            dequant = w.to(torch.float32) * s.to(torch.float32)
            state_dict[embed_weight_key] = dequant.to(torch.bfloat16)
            del state_dict[embed_scale_key]
            print("[H3 FIX] Dequantized INT8 embedding to BF16")

    root = "language_model" if embed_weight_key in state_dict else ""
    return tuple(offload.map_state_dict([state_dict, quantization_map, tied_weights_map], rules={"model": root}))


def probe_h3_checkpoint(filename):
    """Inspect tensor headers before allocating the network."""
    checkpoint_path = filename[0] if isinstance(filename, (list, tuple)) else filename
    state_dict, _ = quant_router.load_metadata_state_dict(checkpoint_path)
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
    rank = int(table.shape[1])
    if rank != ADALN_CURVE_DIM:
        print(
            f"MiniMax H3 pruned checkpoint '{checkpoint_path}' uses non-official AdaLN rank {rank}; the official rank is {ADALN_CURVE_DIM}. "
            "This file may be incompatible with MiniMax H3 LoRAs. Delete it and retry so WanGP can automatically download the official replacement."
        )
    return {"compressed_modulation": True, "adaln_curve_grid": int(table.shape[0]), "time_embed_dim": rank}


def _load_transformer(filename, dtype, qkv_splitting=True):
    checkpoint = probe_h3_checkpoint(filename)
    with init_empty_weights(include_buffers=True):
        transformer = MiniMaxH3Model(adaln_curve_grid=checkpoint["adaln_curve_grid"], time_embed_dim=checkpoint["time_embed_dim"],
                                     dtype=dtype, device="meta")
    filenames = filename if isinstance(filename, (list, tuple)) else [filename]
    split_map = get_linear_split_map(transformer.attention_inner_size) if qkv_splitting and not any(path.lower().endswith(".gguf") for path in filenames) else None
    if split_map is not None:
        offload.split_linear_modules(transformer, split_map)
    transformer.requires_grad_(False)
    preprocess_sd = partial(_strip_wrappers, qkv_splitting=qkv_splitting)
    offload.load_model_data(transformer, filename, writable_tensors=False, default_dtype=dtype, preprocess_sd=preprocess_sd,
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
    text_encoder.requires_grad_(False)
    offload.load_model_data(text_encoder, filename, writable_tensors=False, default_dtype=dtype,
                            preprocess_sd=_strip_text_encoder_wrapper, ignore_unused_weights=True)
    text_encoder.set_tokenizer(os.path.dirname(config_path))

    # Fix AWQ pre_quant_scale: load directly from checkpoint since MMGP strips them.
    # Map checkpoint key prefix "model." → module path "language_model."
    import safetensors.torch as _st
    _awq_count = 0
    try:
        with _st.safe_open(filename, framework="pt", device="cpu") as _sf:
            for _key in _sf.keys():
                if ".pre_quant_scale" in _key and _key.startswith("model."):
                    # Convert "model.layers.N.mlp.down_proj.pre_quant_scale"
                    # to "language_model.layers.N.mlp.down_proj"
                    _mod_path = "language_model" + _key[len("model"):_key.rfind(".pre_quant_scale")]
                    _parts = _mod_path.split(".")
                    _mod = text_encoder
                    for _p in _parts:
                        if _p.isdigit():
                            _mod = _mod[int(_p)]
                        else:
                            _mod = getattr(_mod, _p)
                    _scale = _sf.get_tensor(_key).to(dtype=dtype)
                    # Register as buffer and monkey-patch forward
                    if hasattr(_mod, "forward"):
                        _mod.register_buffer("pre_quant_scale", _scale, persistent=False)
                        _orig_fwd = _mod.forward
                        def _make_awq_fwd(orig, pqs):
                            def awq_forward(input):
                                scaled = input * pqs.to(device=input.device, dtype=input.dtype)
                                return orig(scaled)
                            return awq_forward
                        _mod.forward = _make_awq_fwd(_orig_fwd, _scale)
                        _awq_count += 1
    except Exception as _e:
        print(f"[H3 FIX] Warning: could not load AWQ scales: {_e}")
    print(f"[H3 FIX] Patched {_awq_count} AWQ linears with pre_quant_scale from checkpoint")

    text_encoder.eval().requires_grad_(False)
    return text_encoder


def _load_video_vae(filename, dtype, qkv_splitting=True):
    filename = fl.locate_file(filename)
    with init_empty_weights(include_buffers=False):
        vae = MiniMaxH3VideoVAE()
    split_map = get_video_vae_linear_split_map() if qkv_splitting else None
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


def model_factory(model_filename, text_encoder_filename, qkv_splitting, dtype=torch.bfloat16, VAE_dtype=torch.float32, save_quantized=False,
                  model_type="minimax_h3_fl2va", reference_mode=False, video_vae_filename=VIDEO_VAE_FILE,
                  audio_vae_filename=AUDIO_VAE_FILE):
    transformer = _load_transformer(model_filename, dtype, qkv_splitting)
    text_encoder = _load_text_encoder(text_encoder_filename, dtype)
    video_vae = _load_video_vae(video_vae_filename, VAE_dtype, qkv_splitting)
    audio_vae = _load_audio_vae(audio_vae_filename)
    pipeline = MiniMaxH3Pipeline(transformer, text_encoder, video_vae, audio_vae, reference_mode=reference_mode, dtype=dtype)
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, model_filename[0], dtype, None,
                             convrot_layout=get_convrot_layout(transformer))
    return pipeline


__all__ = ["ADALN_CURVE_DIM", "AUDIO_VAE_FILE",
           "TEXT_ENCODER_FOLDER", "VIDEO_VAE_FILE", "model_factory", "probe_h3_checkpoint"]
