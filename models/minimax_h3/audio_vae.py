# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
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
# Modified for WanGP: stereo boundary layout, latent normalization, MMGP dtype locking,
# and the compact encode/decode API used by the WanGP pipeline.

import torch

from .components.audio_autoencoder import AutoencoderKLMiniMaxH3Audio


LATENTS_MEAN = (
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373,
    0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
    -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614,
    -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797,
)
LATENTS_STD = (
    1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940418612,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
)


class MiniMaxH3AudioVAE(AutoencoderKLMiniMaxH3Audio):
    def __init__(self):
        super().__init__(latents_mean=list(LATENTS_MEAN), latents_std=list(LATENTS_STD))
        self.register_buffer("_latents_mean", torch.tensor(LATENTS_MEAN, dtype=torch.float32), persistent=False)
        self.register_buffer("_latents_std", torch.tensor(LATENTS_STD, dtype=torch.float32), persistent=False)
        # MMGP treats this VAE as one independently managed root model. Keep its
        # DAC/BigVGAN weights and runtime inputs in the checkpoint's FP32 dtype.
        self._lock_dtype = torch.float32
        self._model_dtype = torch.float32
        self._interrupt = False

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

    def encode(self, waveform):
        # WanGP boundary: [1, stereo, samples] -> official mono-batch [stereo, 1, samples].
        posterior = super().encode(waveform[0].unsqueeze(1), return_dict=False)[0]
        latents = posterior.mode()
        mean = self._latents_mean.view(1, -1, 1).to(latents)
        std = self._latents_std.view(1, -1, 1).to(latents)
        return ((latents - mean) / std).permute(1, 0, 2).unsqueeze(0)

    def decode(self, latents):
        # WanGP boundary: [1, channels, stereo, frames] -> official mono-batch.
        latents = latents[0].permute(1, 0, 2)
        mean = self._latents_mean.view(1, -1, 1).to(latents)
        std = self._latents_std.view(1, -1, 1).to(latents)
        waveform = super().decode(latents * std + mean, return_dict=False)[0]
        return waveform.transpose(0, 1)


__all__ = ["LATENTS_MEAN", "LATENTS_STD", "MiniMaxH3AudioVAE"]
