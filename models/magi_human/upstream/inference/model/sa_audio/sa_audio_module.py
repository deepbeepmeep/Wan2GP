# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
from typing import Any, Dict, Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm as parametrized_weight_norm


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    # Adapted from BigVGAN activation.
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def vae_sample(mean, scale):
    stdev = F.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class VAEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x, return_info=False, **kwargs):
        info = {}
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info["kl"] = kl
        if return_info:
            return x, info
        return x

    def decode(self, x):
        return x


def WNConv1d(*args, **kwargs):
    return parametrized_weight_norm(nn.Conv1d(*args, **kwargs), name="weight")


def WNConvTranspose1d(*args, **kwargs):
    return parametrized_weight_norm(nn.ConvTranspose1d(*args, **kwargs), name="weight")


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def get_activation(
    activation: Literal["elu", "snake", "none"], antialias: bool = False, channels=None
) -> nn.Module:
    if antialias:
        raise NotImplementedError("antialias activation is not supported in sa_audio")

    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        if self.training:
            y = checkpoint(self.layers, x)
        else:
            y = self.layers(x)
        return y + x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, 3, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, 9, use_snake=use_snake),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
    ):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, 1, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 3, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()

        c_mults = [1] + c_mults
        depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]

        for i in range(depth - 1):
            layers.append(
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            )

        layers.extend(
            [
                get_activation(
                    "snake" if use_snake else "elu",
                    antialias=antialias_activation,
                    channels=c_mults[-1] * channels,
                ),
                WNConv1d(
                    in_channels=c_mults[-1] * channels,
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                ),
            ]
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake: bool = False,
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
        final_tanh: bool = True,
    ):
        super().__init__()

        c_mults = [1] + c_mults
        depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            )
        ]

        for i in range(depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            )

        layers.extend(
            [
                get_activation(
                    "snake" if use_snake else "elu",
                    antialias=antialias_activation,
                    channels=c_mults[0] * channels,
                ),
                WNConv1d(
                    in_channels=c_mults[0] * channels,
                    out_channels=out_channels,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int,
        io_channels: int = 2,
        bottleneck: nn.Module | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        soft_clip: bool = False,
    ):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = in_channels if in_channels is not None else io_channels
        self.out_channels = out_channels if out_channels is not None else io_channels
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.soft_clip = soft_clip

    def encode(self, audio, skip_bottleneck: bool = False, return_info: bool = False, **kwargs):
        info = {}
        latents = self.encoder(audio)
        info["pre_bottleneck_latents"] = latents

        if self.bottleneck is not None and not skip_bottleneck:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)

        if return_info:
            return latents, info
        return latents

    def decode(self, latents, skip_bottleneck: bool = False, **kwargs):
        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents, **kwargs)
        if self.soft_clip:
            decoded = torch.tanh(decoded)
        return decoded


# AE factories

def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"
    if encoder_type != "oobleck":
        raise ValueError(f"Only encoder type 'oobleck' is supported, got: {encoder_type}")

    encoder = OobleckEncoder(**encoder_config["config"])
    if not encoder_config.get("requires_grad", True):
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder


def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"
    if decoder_type != "oobleck":
        raise ValueError(f"Only decoder type 'oobleck' is supported, got: {decoder_type}")

    decoder = OobleckDecoder(**decoder_config["config"])
    if not decoder_config.get("requires_grad", True):
        for param in decoder.parameters():
            param.requires_grad = False
    return decoder


def create_bottleneck_from_config(bottleneck_config: Dict[str, Any]):
    bottleneck_type = bottleneck_config.get("type", None)
    assert bottleneck_type is not None, "type must be specified in bottleneck config"

    if bottleneck_type != "vae":
        raise NotImplementedError(
            f"Only bottleneck type 'vae' is supported, got: {bottleneck_type}"
        )

    bottleneck = VAEBottleneck()
    if not bottleneck_config.get("requires_grad", True):
        for param in bottleneck.parameters():
            param.requires_grad = False
    return bottleneck


def create_autoencoder_from_config(config: Dict[str, Any]):
    ae_config = config["model"]

    if ae_config.get("pretransform") is not None:
        raise NotImplementedError("Nested pretransform is not supported in sa_audio")

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck_cfg = ae_config.get("bottleneck")
    bottleneck = create_bottleneck_from_config(bottleneck_cfg) if bottleneck_cfg else None

    latent_dim = ae_config.get("latent_dim")
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio")
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels")
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate")
    assert sample_rate is not None, "sample_rate must be specified in model config"

    return AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        io_channels=io_channels,
        bottleneck=bottleneck,
        in_channels=ae_config.get("in_channels"),
        out_channels=ae_config.get("out_channels"),
        soft_clip=ae_config["decoder"].get("soft_clip", False),
    )


def create_model_from_config(model_config: Dict[str, Any]):
    model_type = model_config.get("model_type", None)
    assert model_type is not None, "model_type must be specified in model config"

    if model_type != "autoencoder":
        raise NotImplementedError(f"Only 'autoencoder' is supported, got: {model_type}")

    return create_autoencoder_from_config(model_config)
