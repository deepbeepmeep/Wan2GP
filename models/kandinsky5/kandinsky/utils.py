import os
from typing import Optional, Union

import numpy as np
import torch

from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .models.dit import get_dit
from .models.text_embedders import get_text_embedder
from .models.vae import build_vae
from .i2v_pipeline import Kandinsky5I2VPipeline
from .t2v_pipeline import Kandinsky5T2VPipeline
from .t2i_pipeline import Kandinsky5T2IPipeline
from .i2i_pipeline import Kandinsky5I2IPipeline
from .magcache_utils import set_magcache_params

from PIL import Image
from safetensors.torch import load_file

torch._dynamo.config.suppress_errors = True


HF_TOKEN = None


def get_hf_token():
    return HF_TOKEN


def set_hf_token(hf_token):
    global HF_TOKEN
    HF_TOKEN = hf_token


def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
) -> Kandinsky5T2VPipeline:
    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id="kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s",
            allow_patterns="model/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        dit_path = os.path.join(cache_dir, "model/kandinsky5lite_t2v_sft_5s.safetensors")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        vae_path = os.path.join(cache_dir, "vae/")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
            token=get_hf_token()
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
            token=get_hf_token()
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path
        )
    else:
        conf = OmegaConf.load(conf_path)
    conf.model.dit_params.attention_engine = attention_engine

    conf.model.text_embedder.qwen.mode = "t2v"
    text_embedder = get_text_embedder(conf.model.text_embedder, device=torch.device("cpu"),
                                      quantized_qwen=quantized_qwen, text_token_padding=text_token_padding)

    vae = build_vae(conf.model.vae)
    vae = vae.eval()

    dit = get_dit(conf.model.dit_params, text_token_padding=text_token_padding)

    if magcache:
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg, start_step=0)

    state_dict = load_file(conf.model.checkpoint_path, device='cpu')
    dit.load_state_dict(state_dict, assign=True)

    return Kandinsky5T2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        conf=conf,
    )

def get_I2V_pipeline(
    device_map: Union[str, torch.device, dict],
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
) -> Kandinsky5T2VPipeline:
    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id="kandinskylab/Kandinsky-5.0-I2V-Lite-5s",
            allow_patterns="model/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        dit_path = os.path.join(cache_dir, "model/kandinsky5lite_i2v_5s.safetensors")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        vae_path = os.path.join(cache_dir, "vae/")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
            token=get_hf_token()
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
            token=get_hf_token()
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path
        )
    else:
        conf = OmegaConf.load(conf_path)
    conf.model.dit_params.attention_engine = attention_engine

    conf.model.text_embedder.qwen.mode = "i2v"
    text_embedder = get_text_embedder(conf.model.text_embedder, device=torch.device("cpu"),
                                      quantized_qwen=quantized_qwen, text_token_padding=text_token_padding)
    
    vae = build_vae(conf.model.vae)
    vae = vae.eval()

    dit = get_dit(conf.model.dit_params, text_token_padding=text_token_padding)

    if magcache:
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg, start_step=0)

    state_dict = load_file(conf.model.checkpoint_path, device='cpu')
    dit.load_state_dict(state_dict, assign=True)

    return Kandinsky5I2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        conf=conf,
    )


def _get_TI2I_params(
    instruct_type: bool,
    model_name: str,
    weights_name: str,
    device_map: Union[str, torch.device, dict],
    resolution: int = 1024,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
) -> Kandinsky5T2IPipeline:
    assert resolution in [1024]

    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None and conf_path is None:
        dit_path = snapshot_download(
            repo_id=f"kandinskylab/{model_name}",
            allow_patterns="model/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        dit_path = os.path.join(cache_dir, f"model/{weights_name}")

    if vae_path is None and conf_path is None:
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
            token=get_hf_token()
        )
        vae_path = os.path.join(cache_dir, "vae/")

    if text_encoder_path is None and conf_path is None:
        text_encoder_path = snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=os.path.join(cache_dir, "text_encoder/"),
            token=get_hf_token()
        )
        text_encoder_path = os.path.join(cache_dir, "text_encoder/")

    if text_encoder2_path is None and conf_path is None:
        text_encoder2_path = snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=os.path.join(cache_dir, "text_encoder2/"),
            token=get_hf_token()
        )
        text_encoder2_path = os.path.join(cache_dir, "text_encoder2/")

    if conf_path is None:
        conf = get_default_ti2i_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path, instruct_type=instruct_type,
        )
    else:
        conf = OmegaConf.load(conf_path)
    conf.model.dit_params.attention_engine = attention_engine

    conf.model.text_embedder.qwen.mode = "t2i"
    text_embedder = get_text_embedder(conf.model.text_embedder, device=torch.device("cpu"),
                                      quantized_qwen=quantized_qwen, text_token_padding=text_token_padding)

    vae = build_vae(conf.model.vae)
    vae = vae.eval()

    dit = get_dit(conf.model.dit_params, text_token_padding=text_token_padding)

    if magcache:
        mag_ratios = conf.magcache.mag_ratios
        num_steps = conf.model.num_steps
        no_cfg = False
        if conf.model.guidance_weight == 1.0:
            no_cfg = True
        set_magcache_params(dit, mag_ratios, num_steps, no_cfg, start_step=0)

    state_dict = load_file(conf.model.checkpoint_path)
    dit.load_state_dict(state_dict, assign=True)

    return dict(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        resolution=resolution,
        conf=conf,
    )


def get_T2I_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 1024,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
) -> Kandinsky5T2IPipeline:
    kwargs = _get_TI2I_params(
        instruct_type=None,
        model_name='Kandinsky-5.0-T2I-Lite',
        weights_name='kandinsky5lite_t2i.safetensors',
        device_map=device_map,
        resolution=resolution,
        cache_dir=cache_dir,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        text_encoder2_path=text_encoder2_path,
        vae_path=vae_path,
        conf_path=conf_path,
        magcache=magcache,
        quantized_qwen=quantized_qwen,
        text_token_padding=text_token_padding,
        attention_engine=attention_engine,
    )

    return Kandinsky5T2IPipeline(**kwargs)


def get_I2I_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 1024,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    magcache: bool = False,
    quantized_qwen: bool = False,
    text_token_padding: bool = False,
    attention_engine: str = "auto",
) -> Kandinsky5T2IPipeline:
    kwargs = _get_TI2I_params(
        instruct_type='channel',
        model_name='Kandinsky-5.0-I2I-Lite',
        weights_name='kandinsky5lite_i2i.safetensors',
        device_map=device_map,
        resolution=resolution,
        cache_dir=cache_dir,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        text_encoder2_path=text_encoder2_path,
        vae_path=vae_path,
        conf_path=conf_path,
        magcache=magcache,
        quantized_qwen=quantized_qwen,
        text_token_padding=text_token_padding,
        attention_engine=attention_engine,
    )

    return Kandinsky5I2IPipeline(**kwargs)


def get_default_conf(
    dit_path,
    vae_path,
    text_encoder_path,
    text_encoder2_path,
) -> DictConfig:
    dit_params = {
        "in_visual_dim": 16,
        "out_visual_dim": 16,
        "time_dim": 512,
        "patch_size": [1, 2, 2],
        "model_dim": 1792,
        "ff_dim": 7168,
        "num_text_blocks": 2,
        "num_visual_blocks": 32,
        "axes_dims": [16, 24, 24],
        "visual_cond": True,
        "in_text_dim": 3584,
        "in_text_dim2": 768,
    }

    attention = {
        "type": "flash",
        "causal": False,
        "local": False,
        "glob": False,
        "window": 3,
    }

    vae = {
        "checkpoint_path": vae_path,
        "name": "hunyuan",
    }

    text_embedder = {
        "qwen": {
            "emb_size": 3584,
            "checkpoint_path": text_encoder_path,
            "max_length": 256,
        },
        "clip": {
            "checkpoint_path": text_encoder2_path,
            "emb_size": 768,
            "max_length": 77,
        },
    }

    conf = {
        "model": {
            "checkpoint_path": dit_path,
            "vae": vae,
            "text_embedder": text_embedder,
            "dit_params": dit_params,
            "attention": attention,
            "num_steps": 50,
            "guidance_weight": 5.0,
        },
        "metrics": {"scale_factor": (1, 2, 2), "resolution": 512,},
    }

    return DictConfig(conf)


def get_default_ti2i_conf(
    dit_path,
    vae_path,
    text_encoder_path,
    text_encoder2_path,
    instruct_type=None,
) -> DictConfig:
    dit_params = {
        "instruct_type": instruct_type,
        "in_visual_dim": 16,
        "out_visual_dim": 16,
        "time_dim": 512,
        "patch_size": [1, 2, 2],
        "model_dim": 2560,
        "ff_dim": 10240,
        "num_text_blocks": 2,
        "num_visual_blocks": 50,
        "axes_dims": [32,48, 48],
    }

    attention = {
        "type": "flash",
        "causal": False,
        "local": False,
        "glob": False,
        "window": 3,
    }

    vae = {
        "checkpoint_path": vae_path,
        "name": "hunyuan",
    }

    text_embedder = {
        "qwen": {
            "emb_size": 3584,
            "checkpoint_path": text_encoder_path,
            "max_length": 512,
        },
        "clip": {
            "checkpoint_path": text_encoder2_path,
            "emb_size": 768,
            "max_length": 77,
        },
    }

    conf = {
        "model": {
            "checkpoint_path": dit_path,
            "vae": vae,
            "text_embedder": text_embedder,
            "dit_params": dit_params,
            "attention": attention,
            "num_steps": 50,
            "guidance_weight": 3.5,
        },
        "metrics": {"scale_factor": (1, 2, 2), "resolution": 1024},
    }

    return DictConfig(conf)
