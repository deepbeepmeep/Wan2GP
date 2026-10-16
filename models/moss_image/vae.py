# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from torch import nn
from models.flux3.vae import Flux3VAE
from models.flux.modules.autoencoder_flux2 import Decoder
from shared.utils.conv_bands import band_convs


class MOSSImageVAE(Flux3VAE):
    def __init__(self):
        nn.Module.__init__(self)
        self.decoder = Decoder(ch=96, out_ch=3, ch_mult=[1, 2, 4, 4], num_res_blocks=2, in_channels=3, resolution=1024, z_channels=32)
        self.bn_eps, self.ps = 1e-4, [2, 2]
        self.bn = nn.BatchNorm2d(128, eps=self.bn_eps, affine=False)
        self.bn._lock_dtype = torch.float32
        self._convertWeightsFloatTo = None
        band_convs(self)
