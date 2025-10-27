from typing import Literal, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize

from .ext.autoencoder import AutoEncoderModule
from .ext.autoencoder.distributions import DiagonalGaussianDistribution
from .ext.mel_converter import get_mel_converter


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: str, 
        bigvgan_vocoder_ckpt: Optional[str] = None,
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                        vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                        mode=mode,
                                        need_vae_encoder=need_vae_encoder)

    def compile(self):

        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def wrapped_decode(self, z):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            mel_decoded = self.decode(z)
            audio = self.vocode(mel_decoded)

            return audio 

    @torch.no_grad()
    def wrapped_encode(self, audio):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            dist = self.encode_audio(audio)

            return dist.mean