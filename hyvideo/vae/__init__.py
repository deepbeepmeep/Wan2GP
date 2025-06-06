from pathlib import Path

import torch

from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ..constants import VAE_PATH, PRECISION_TO_TYPE

def load_vae(vae_type: str="884-16c-hy",
             vae_precision: str=None,
             sample_size: tuple=None,
             vae_path: str=None,
             vae_config_path: str=None,
             logger=None,
             device=None
             ):
    """the fucntion to load the 3D VAE model

    Args:
        vae_type (str): the type of the 3D VAE model. Defaults to "884-16c-hy".
        vae_precision (str, optional): the precision to load vae. Defaults to None.
        sample_size (tuple, optional): the tiling size. Defaults to None.
        vae_path (str, optional): the path to vae. Defaults to None.
        logger (_type_, optional): logger. Defaults to None.
        device (_type_, optional): device to load vae. Defaults to None.
    """
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]
    
    if logger is not None:
        logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")

    # config = AutoencoderKLCausal3D.load_config("ckpts/hunyuan_video_VAE_config.json")
    # config = AutoencoderKLCausal3D.load_config("c:/temp/hvae/config_vae.json")
    config = AutoencoderKLCausal3D.load_config(vae_config_path)
    if sample_size:
        vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
    else:
        vae = AutoencoderKLCausal3D.from_config(config)

    vae_ckpt = Path(vae_path) 
    # vae_ckpt = Path("ckpts/hunyuan_video_VAE.pt") 
    # vae_ckpt = Path("c:/temp/hvae/pytorch_model.pt")
    assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"
    
    from mmgp import offload

    # ckpt = torch.load(vae_ckpt, weights_only=True, map_location=vae.device)
    # if "state_dict" in ckpt:
    #     ckpt = ckpt["state_dict"]
    # if any(k.startswith("vae.") for k in ckpt.keys()):
    #     ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
    # a,b = vae.load_state_dict(ckpt)

    # offload.save_model(vae, "vae_32.safetensors")
    # vae.to(torch.bfloat16)
    # offload.save_model(vae, "vae_16.safetensors")
    offload.load_model_data(vae, vae_path )
    # ckpt = torch.load(vae_ckpt, weights_only=True, map_location=vae.device)

    spatial_compression_ratio = vae.config.spatial_compression_ratio
    time_compression_ratio = vae.config.time_compression_ratio
    
    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    vae.eval()

    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
