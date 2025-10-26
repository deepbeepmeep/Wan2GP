from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .models import BigVGANVocoder

_bigvgan_vocoder_path = Path(__file__).parent / 'bigvgan_vocoder.yml'


class BigVGAN(nn.Module):

    def __init__(self, ckpt_path, config_path=_bigvgan_vocoder_path):
        super().__init__()
        vocoder_cfg = OmegaConf.load(config_path)
        self.vocoder = BigVGANVocoder(vocoder_cfg).eval()
        vocoder_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)['generator']
        self.vocoder.load_state_dict(vocoder_ckpt)

        self.weight_norm_removed = False
        self.remove_weight_norm()

    @torch.inference_mode()
    def forward(self, x):
        assert self.weight_norm_removed, 'call remove_weight_norm() before inference'
        return self.vocoder(x)

    def remove_weight_norm(self):
        self.vocoder.remove_weight_norm()
        self.weight_norm_removed = True
        return self
