"""Model construction and checkpoint loading for MiniMax H3."""

import os
import re
from functools import partial

import torch
from accelerate import init_empty_weights

from mmgp import offload, quant_router
from shared.utils import files_locator as fl

from .audio_vae import MiniMaxH3AudioVAE
from .convrot_layout import get_convrot_layout, restore_interleaved_h3_qkv
from .latent_upscaler import MiniMaxH3LatentUpscaler
from .pipeline import MiniMaxH3Pipeline
from .text_encoder import MiniMaxH3TextEncoder, load_h3_qwen_config
from .transformer import MiniMaxH3Model, get_linear_split_map
from .video_vae import MiniMaxH3VideoVAE, get_video_vae_linear_split_map


VIDEO_VAE_FILE = "MiniMax-H3-video_vae_fp16.safetensors"
VIDEO_VAE_FP8MIX_FILE = "minimax_h3_video_vae_fp8mix.safetensors"
AUDIO_VAE_FILE = "MiniMax-H3-audio_vae_fp32.safetensors"
LATENT_UPSCALER_FOLDER = "minimax_h3"
LATENT_UPSCALER_FILE = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
TEXT_ENCODER_FOLDER = "Qwen3-VL-32B-Instruct"
ADALN_CURVE_DIM = 8
NUM_TRANSFORMER_BLOCKS = 50


def _hybrid_ref2va_blocks(filename):
    filenames = filename if isinstance(filename, (list, tuple)) else [filename]
    ranges = set()
    for path in filenames:
        match = re.search(r"hybrid[_-]fl2va[_-]ref2va[_-]b(\d+)-(\d+)", os.path.basename(path), re.IGNORECASE)
        if match:
            ranges.add(tuple(map(int, match.groups())))
    if not ranges:
        return None
    if len(ranges) != 1:
        raise ValueError(f"MiniMax H3 checkpoint files specify conflicting hybrid block ranges: {sorted(ranges)}")
    start, end = ranges.pop()
    if not 0 <= start <= end < NUM_TRANSFORMER_BLOCKS:
        raise ValueError(f"Invalid MiniMax H3 hybrid Ref2VA block range b{start}-{end}; expected blocks 0-{NUM_TRANSFORMER_BLOCKS - 1}")
    return start, end


def _strip_wrappers(state_dict, quantization_map=None, tied_weights_map=None, qkv_splitting=True, pdd=False,
                    ref2va_adaln_t_table=None):
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
            if pdd and (key.startswith("final_layer.video_out.") or key.startswith("final_layer.audio_out.") or
                        key in ("final_layer.video_out", "final_layer.audio_out")):
                continue
            if pdd and key.startswith("pdd_heads."):
                key = "final_layer." + key[len("pdd_heads."):]
            result[key] = value
        return result

    state_dict = strip(state_dict)
    if ref2va_adaln_t_table is not None:
        state_dict["ref2va_adaln_t_table"] = ref2va_adaln_t_table
    return state_dict, strip(quantization_map), strip(tied_weights_map)


def _strip_text_encoder_wrapper(state_dict, quantization_map=None, tied_weights_map=None):
    root = "language_model" if "model.embed_tokens.weight" in state_dict else ""
    return tuple(offload.map_state_dict([state_dict, quantization_map, tied_weights_map], rules={"model": root}))


def probe_h3_checkpoint(filename):
    """Inspect tensor headers before allocating the network."""
    checkpoint_path = filename[0] if isinstance(filename, (list, tuple)) else filename
    hybrid_ref2va_blocks = _hybrid_ref2va_blocks(filename)
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
        if hybrid_ref2va_blocks is not None:
            raise ValueError("MiniMax H3 FL2VA/Ref2VA hybrid checkpoints require compressed AdaLN tables")
        return {"compressed_modulation": False, "adaln_curve_grid": None, "time_embed_dim": 2688, "adaln_dtype": None}
    if not table.dtype.is_floating_point:
        raise ValueError(f"H3 AdaLN curve table must use an unquantized floating-point GGUF qtype, got {table.dtype}")
    if len(table.shape) != 2 or table.shape[0] < 2:
        raise ValueError(f"Invalid H3 AdaLN curve table shape: {tuple(table.shape)}")
    rank = int(table.shape[1])
    if rank != ADALN_CURVE_DIM:
        print(
            f"MiniMax H3 pruned checkpoint '{checkpoint_path}' uses non-official AdaLN rank {rank}; the official rank is {ADALN_CURVE_DIM}. "
            "This file may be incompatible with MiniMax H3 LoRAs. Delete it and retry so WanGP can automatically download the official replacement."
        )
    return {"compressed_modulation": True, "adaln_curve_grid": int(table.shape[0]), "time_embed_dim": rank, "adaln_dtype": table.dtype,
            "hybrid_ref2va_blocks": hybrid_ref2va_blocks}


def _resample_adaln_table(table, rows, dtype):
    if table.shape[0] != rows:
        position = torch.linspace(0, table.shape[0] - 1, rows)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        table = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    return table.to(dtype=dtype)


def _load_transformer(filename, dtype, qkv_splitting=True, qkv_layout="interleaved", pdd=False,
                      pdd_num_steps=None, pdd_block_size=None):
    checkpoint = probe_h3_checkpoint(filename)
    hybrid_ref2va_blocks = checkpoint.get("hybrid_ref2va_blocks")
    filenames = list(filename) if isinstance(filename, (list, tuple)) else [filename]
    ref2va_adaln_t_table = None
    if hybrid_ref2va_blocks is not None:
        from .lora_affine import _load_affine_package
        ref2va_adaln_t_table = _resample_adaln_table(_load_affine_package("ref2va", checkpoint["time_embed_dim"])[0],
                                                     checkpoint["adaln_curve_grid"], checkpoint["adaln_dtype"])
        start, end = hybrid_ref2va_blocks
        print(f"MiniMax H3: detected FL2VA/Ref2VA hybrid checkpoint; blocks {start}-{end} use the Ref2VA AdaLN table, "
              "and all other blocks plus the final layer use the FL2VA table.")
    if pdd and (int(pdd_num_steps) < 1 or int(pdd_block_size) < 1 or int(pdd_num_steps) % int(pdd_block_size)):
        raise ValueError(f"Invalid MiniMax H3 PDD grid={pdd_num_steps}, block={pdd_block_size}")
    with init_empty_weights(include_buffers=True):
        transformer = MiniMaxH3Model(adaln_curve_grid=checkpoint["adaln_curve_grid"], time_embed_dim=checkpoint["time_embed_dim"], adaln_dtype=checkpoint["adaln_dtype"],
                                     hybrid_ref2va_blocks=hybrid_ref2va_blocks, pdd_num_steps=pdd_num_steps,
                                     pdd_block_size=pdd_block_size, dtype=dtype, device="meta")
    split_map = get_linear_split_map(transformer.attention_inner_size, qkv_layout=qkv_layout) if qkv_splitting and not any(path.lower().endswith(".gguf") for path in filenames) else None
    if split_map is not None:
        offload.split_linear_modules(transformer, split_map)
    transformer.requires_grad_(False)
    preprocess_sd = partial(_strip_wrappers, qkv_splitting=qkv_splitting, pdd=pdd, ref2va_adaln_t_table=ref2va_adaln_t_table)
    offload.load_model_data(transformer, filenames, writable_tensors=False, default_dtype=dtype, preprocess_sd=preprocess_sd,
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
    text_encoder.eval().requires_grad_(False)
    return text_encoder


def _load_video_vae(filename, dtype, qkv_splitting=True):
    filename = fl.locate_file(filename)
    print(f"Loading MiniMax H3 Video VAE '{filename}'...")
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


def _load_latent_upscaler(filename):
    filename = fl.locate_file(filename)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=True):
            upscaler = MiniMaxH3LatentUpscaler()
    finally:
        torch.set_default_dtype(previous_dtype)
    offload.load_model_data(upscaler, filename, writable_tensors=False, default_dtype=torch.bfloat16)
    return upscaler.eval().requires_grad_(False)


def model_factory(model_filename, text_encoder_filename, qkv_splitting, dtype=torch.bfloat16, VAE_dtype=torch.float32, save_quantized=False,
                  model_type="minimax_h3_fl2va", reference_mode=False, video_vae_filename=VIDEO_VAE_FILE,
                  audio_vae_filename=AUDIO_VAE_FILE, latent_upscaler_filename=os.path.join(LATENT_UPSCALER_FOLDER, LATENT_UPSCALER_FILE),
                  shared_h3_pipeline=None, qkv_layout="interleaved", pdd=False, pdd_num_steps=None, pdd_block_size=None):
    transformer = _load_transformer(model_filename, dtype, qkv_splitting, qkv_layout, pdd, pdd_num_steps, pdd_block_size)
    if shared_h3_pipeline is None:
        text_encoder = _load_text_encoder(text_encoder_filename, dtype)
        video_vae_qkv_splitting = qkv_splitting and video_vae_filename == VIDEO_VAE_FILE
        video_vae = _load_video_vae(video_vae_filename, VAE_dtype, video_vae_qkv_splitting)
        audio_vae = _load_audio_vae(audio_vae_filename)
        latent_upscaler = _load_latent_upscaler(latent_upscaler_filename)
    else:
        text_encoder = shared_h3_pipeline.text_encoder
        video_vae = shared_h3_pipeline.vae
        audio_vae = shared_h3_pipeline.audio_vae
        latent_upscaler = shared_h3_pipeline.latent_upscaler
    pipeline = MiniMaxH3Pipeline(transformer, text_encoder, video_vae, audio_vae, latent_upscaler=latent_upscaler, reference_mode=reference_mode, dtype=dtype)
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, model_filename[0], dtype, None,
                             convrot_layout=get_convrot_layout(transformer))
    return pipeline


__all__ = ["ADALN_CURVE_DIM", "AUDIO_VAE_FILE", "LATENT_UPSCALER_FILE", "LATENT_UPSCALER_FOLDER", "TEXT_ENCODER_FOLDER",
           "VIDEO_VAE_FILE", "VIDEO_VAE_FP8MIX_FILE", "model_factory", "probe_h3_checkpoint"]
