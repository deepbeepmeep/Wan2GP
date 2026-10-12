"""Continuous DAC codec, adapted from Prism/MOVA. See LICENSE and NOTICE."""
import math
import torch
from torch import nn
from shared.utils.phase_progress import check_abort

WNConv1d = nn.Conv1d
WNConvTranspose1d = nn.ConvTranspose1d

def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-09).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

class ResidualUnit(nn.Module):

    def __init__(self, dim: int=16, dilation: int=1):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.block = nn.Sequential(Snake1d(dim), WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad), Snake1d(dim), WNConv1d(dim, dim, kernel_size=1))

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y

class EncoderBlock(nn.Module):

    def __init__(self, dim: int=16, stride: int=1):
        super().__init__()
        self.block = nn.Sequential(ResidualUnit(dim // 2, dilation=1), ResidualUnit(dim // 2, dilation=3), ResidualUnit(dim // 2, dilation=9), Snake1d(dim // 2), WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)))

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):

    def __init__(self, d_model: int=64, strides: list=[2, 4, 8, 8], d_latent: int=64):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):

    def __init__(self, input_dim: int=16, output_dim: int=8, stride: int=1):
        super().__init__()
        self.block = nn.Sequential(Snake1d(input_dim), WNConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2), ResidualUnit(output_dim, dilation=1), ResidualUnit(output_dim, dilation=3), ResidualUnit(output_dim, dilation=9))

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):

    def __init__(self, input_channel, channels, rates, d_out: int=1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [Snake1d(output_dim), WNConv1d(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AudioVAE(nn.Module):
    _offload_hooks = ['decode']

    def __init__(self, config):
        super().__init__()
        self.sample_rate = config['sample_rate']
        self.hop_length = math.prod(config['encoder_rates'])
        self.encoder = Encoder(config['encoder_dim'], config['encoder_rates'], config['latent_dim'])
        self.decoder = Decoder(config['latent_dim'], config['decoder_dim'], config['decoder_rates'])
        self.quant_conv = nn.Conv1d(config['latent_dim'], 2 * config['latent_dim'], 1)
        self.post_quant_conv = nn.Conv1d(config['latent_dim'], config['latent_dim'], 1)
        self._convertWeightsFloatTo = None

    def decode(self, z):
        x = self.post_quant_conv(z)
        for block in self.decoder.model:
            x = block(x)
            check_abort()
        return x

