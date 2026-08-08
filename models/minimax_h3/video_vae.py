# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# WanGP boundary adapter around the official MiniMax-H3 video VAE.

import torch

from .components.video_autoencoder import AutoencoderKLMiniMaxH3, get_linear_split_map as get_video_vae_linear_split_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


class MiniMaxH3VideoVAE(AutoencoderKLMiniMaxH3):
    def __init__(self):
        super().__init__(latents_mean=LATENTS_MEAN, latents_std=LATENTS_STD)
        self.register_buffer("_latents_mean", torch.tensor(LATENTS_MEAN, dtype=torch.float32), persistent=False)
        self.register_buffer("_latents_std", torch.tensor(LATENTS_STD, dtype=torch.float32), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        self._interrupt = False

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision):
        if vae_config == 0:
            return 256
        return {1: 0, 2: 512, 3: 256}.get(int(vae_config), 128)

    @property
    def vae_ratio(self):
        return self.spatial_compression_ratio

    @property
    def vae_ratio_t(self):
        return self.temporal_compression_ratio

    @property
    def _interrupt(self):
        return getattr(self, "_abort", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._abort = bool(value)
        if hasattr(self, "encoder"):
            self.encoder._interrupt = self._abort
        if hasattr(self, "decoder"):
            self.decoder._interrupt = self._abort

    def _pixels(self, video):
        video = video.float().add(1.0).mul_(0.5)
        video.sub_(self.pixel_mean.to(video)).div_(self.pixel_std.to(video))
        return video.to(self._model_dtype)

    def _normalize(self, latents):
        mean = self._latents_mean.view(1, -1, 1, 1, 1).to(latents)
        std = self._latents_std.view(1, -1, 1, 1, 1).to(latents)
        return (latents - mean) / std

    def encode(self, video):
        posterior = super().encode(self._pixels(video), return_dict=False)[0]
        return self._normalize(posterior.mode().float())

    def encode_condition(self, video, keep_all_latents=False):
        pixels = self._pixels(video)
        if pixels.shape[2] == 1:
            moments = self._encode_clip(pixels)
        elif keep_all_latents:
            clip_length = self.config.clip_length
            moments = torch.cat([
                self._encode_clip(pixels[:, :, start:start + clip_length])
                for start in range(0, pixels.shape[2], clip_length)
            ], dim=2)
        else:
            moments = self._encode(pixels)
        mean, logvar = moments.float().chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42), dtype=torch.float32, device="cpu")
        latents = (mean + std * noise.to(mean.device)).to(torch.float16).float()
        return self._normalize(latents)

    def decode(self, latents):
        mean = self._latents_mean.view(1, -1, 1, 1, 1).to(latents)
        std = self._latents_std.view(1, -1, 1, 1, 1).to(latents)
        decoded = super().decode((latents * std + mean).to(self._model_dtype), return_dict=False)[0].float()
        decoded.mul_(self.pixel_std.to(decoded)).add_(self.pixel_mean.to(decoded))
        return decoded.clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)


__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "LATENTS_MEAN", "LATENTS_STD", "MiniMaxH3VideoVAE",
           "get_video_vae_linear_split_map"]
