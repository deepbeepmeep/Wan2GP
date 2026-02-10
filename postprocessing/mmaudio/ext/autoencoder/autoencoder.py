import os
from typing import Literal, Optional

import torch
import torch.nn as nn
from mmgp import offload
from shared.utils import files_locator as fl

from ..autoencoder.vae import VAE, get_my_vae
from ..bigvgan import BigVGAN
from ..bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from ...model.utils.distributions import DiagonalGaussianDistribution

_BIGVGAN_V2_FOLDER = "bigvgan_v2_44khz_128band_512x"


def _resolve_bigvgan_v2_files():
    weights_path = fl.locate_file(
        os.path.join(_BIGVGAN_V2_FOLDER, "bigvgan_generator.pt"), error_if_none=False
    )
    config_path = fl.locate_file(
        os.path.join(_BIGVGAN_V2_FOLDER, "config.json"), error_if_none=False
    )
    if weights_path is None or config_path is None:
        raise FileNotFoundError(
            f"Missing BigVGANv2 files in '{_BIGVGAN_V2_FOLDER}'. "
            "Expected 'config.json' and 'bigvgan_generator.pt'."
        )
    return weights_path, config_path


def _preprocess_bigvgan_v2_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
    if isinstance(state_dict, dict) and isinstance(state_dict.get("generator"), dict):
        state_dict = state_dict["generator"]
    return state_dict, quantization_map, tied_weights_map


class AutoEncoderModule(nn.Module):

    def __init__(self,
                 *,
                 vae_ckpt_path,
                 vocoder_ckpt_path: Optional[str] = None,
                 mode: Literal['16k', '44k'],
                 need_vae_encoder: bool = True):
        super().__init__()
        self.vae: VAE = get_my_vae(mode).eval()
        vae_state_dict = torch.load(vae_ckpt_path, weights_only=True, map_location='cpu')
        self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()

        if mode == '16k':
            assert vocoder_ckpt_path is not None
            self.vocoder = BigVGAN(vocoder_ckpt_path).eval()
        elif mode == '44k':
            vocoder_ckpt_path, vocoder_config_path = _resolve_bigvgan_v2_files()
            self.vocoder = offload.fast_load_transformers_model(
                vocoder_ckpt_path,
                modelClass=BigVGANv2,
                forcedConfigPath=vocoder_config_path,
                preprocess_sd=_preprocess_bigvgan_v2_state_dict,
                configKwargs={"use_cuda_kernel": False},
                writable_tensors=False,
                default_dtype=torch.float32,
            )
            self.vocoder.remove_weight_norm()
            self.vocoder.eval()
        else:
            raise ValueError(f'Unknown mode: {mode}')

        for param in self.parameters():
            param.requires_grad = False

        if not need_vae_encoder:
            del self.vae.encoder

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
