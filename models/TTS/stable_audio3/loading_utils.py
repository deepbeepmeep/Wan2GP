import json

import torch
from mmgp import offload, safetensors2

from models.TTS.stable_audio3.factory import create_autoencoder_from_config, create_diffusion_cond_from_config


def copy_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    state_dict = remap_state_dict_keys(state_dict, model_state_dict)
    for key, value in state_dict.items():
        if key in model_state_dict and value.shape == model_state_dict[key].shape:
            model_state_dict[key] = value
        else:
            print(f"Key {key} not found in target state_dict or shape mismatch. Skipping.")
    model.load_state_dict(model_state_dict, strict=False)


def _load_prefixed_state_dict(ckpt_path, prefix):
    with safetensors2.safe_open(ckpt_path, framework="pt", device="cpu", writable_tensors=False) as reader:
        keys = list(reader.keys())
        return {key[len(prefix) :]: reader.get_tensor(key) for key in keys if key.startswith(prefix)}


def load_autoencoder(config_path: str, ckpt_path: str, device: str = "cpu"):
    with open(config_path, "r", encoding="utf-8") as reader:
        config = json.load(reader)
    autoencoder = create_autoencoder_from_config(config["model"], config["sample_rate"])
    nested_prefix = "pretransform.model."
    state_dict = _load_prefixed_state_dict(ckpt_path, nested_prefix)
    if state_dict:
        copy_state_dict(autoencoder, state_dict)
    else:
        offload.load_model_data(autoencoder, ckpt_path, writable_tensors=False)
    return autoencoder


def load_pretransform(pretransform, ckpt_path: str, dtype: torch.dtype = torch.bfloat16):
    autoencoder = pretransform.model
    nested_prefix = "pretransform.model."
    state_dict = _load_prefixed_state_dict(ckpt_path, nested_prefix)
    if state_dict:
        copy_state_dict(autoencoder, state_dict)
    else:
        offload.load_model_data(autoencoder, ckpt_path, default_dtype=dtype, writable_tensors=False)
    return pretransform


def load_diffusion_cond(
    model_config,
    ckpt_path: str,
    pretransform_ckpt_path: str | None = None,
    text_encoder_weights_path: str | None = None,
    text_encoder_tokenizer_dir: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    model = create_diffusion_cond_from_config(
        model_config,
        text_encoder_weights_path=text_encoder_weights_path,
        text_encoder_tokenizer_dir=text_encoder_tokenizer_dir,
        dtype=dtype,
    )
    offload.load_model_data(model, ckpt_path, default_dtype=dtype, writable_tensors=False, ignore_missing_keys=pretransform_ckpt_path is not None)
    if pretransform_ckpt_path is not None and model.pretransform is not None:
        load_pretransform(model.pretransform, pretransform_ckpt_path, dtype=dtype)
    model.eval().requires_grad_(False)
    return model


def remap_state_dict_keys(state_dict, model_state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key not in model_state_dict:
            parts = key.split(".")
            for i in range(1, len(parts)):
                candidate = ".".join(parts[:i]) + "." + ".".join(parts[i + 1 :])
                if candidate in model_state_dict:
                    key = candidate
                    break
        remapped[key] = value
    return remapped
