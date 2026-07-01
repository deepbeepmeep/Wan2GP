from typing import Literal, Optional
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Normalize
from PrismAudio.models.factory import create_model_from_config
from PrismAudio.models.utils import load_ckpt_state_dict
import einshape
import sys
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os
import numpy as np
log = logging.getLogger()

from data_utils.ext.synchformer import Synchformer


def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    for key in state_dict:
        if key not in model_state_dict:
            unexpected_keys.append(key)
        elif state_dict[key].shape != model_state_dict[key].shape:
            unexpected_keys.append(key)

    for key in model_state_dict:
        if key not in state_dict:
            missing_keys.append(key)

    print("Missing keys in state_dict:", missing_keys)
    print("Unexpected keys in state_dict:", unexpected_keys)
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
        
    model.load_state_dict(model_state_dict, strict=False)
    

class FeaturesUtils(nn.Module):
 
    def __init__(
        self,
        *, 
        vae_ckpt: Optional[str] = None,
        vae_config: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        need_vae_encoder: bool = True,
        t5_model_path: str = "google/t5gemma-l-l-ul2-it",
        videoprism_ckpt_path: Optional[str] = None,
        videoprism_tokenizer_path: Optional[str] = None,
    ):
        super().__init__()
        self.videoprism_ckpt_path = videoprism_ckpt_path
        self.videoprism_tokenizer_path = videoprism_tokenizer_path
        
        if enable_conditions:
            
            local_t5 = os.path.isdir(t5_model_path)
            self.t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path, local_files_only=local_t5).get_encoder()
            self.t5tokenizer = AutoTokenizer.from_pretrained(t5_model_path, local_files_only=local_t5)
            
            self.synchformer = Synchformer()
            if synchformer_ckpt is not None:
                self.synchformer.load_state_dict(torch.load(synchformer_ckpt, weights_only=True, map_location='cpu'))

  
        else:

            self.synchformer = None
            self.tokenizer = None

        if vae_ckpt is not None:
            with open(vae_config) as f:
                vae_config = json.load(f)
            self.vae = create_model_from_config(vae_config)
            print(f"Loading model checkpoint from {vae_ckpt}")
            # Load checkpoint
            copy_state_dict(self.vae, load_ckpt_state_dict(vae_ckpt,prefix='autoencoder.'))#,prefix='autoencoder.'

        
    def _init_jax(self):
        if hasattr(self, "flax_model"):
            return  # already init
        import jax
        from videoprism import models as vp

        self._jax = jax
        self._vp = vp
        backend = jax.default_backend()
        if backend != 'gpu':
            self.jax_dev = jax.devices()[0]  # CPU只有一个设备
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            devices = jax.devices()
            device_idx = local_rank % len(devices)
            self.jax_dev = devices[device_idx]
        
        model_name = 'videoprism_lvt_public_v1_large'
        self.flax_model = vp.get_model(model_name)
        state = vp.load_pretrained_weights(model_name, checkpoint_path=self.videoprism_ckpt_path)
        self.loaded_state = jax.device_put(state, device=self.jax_dev)
        if self.videoprism_tokenizer_path is None:
            self.text_tokenizer = vp.load_text_tokenizer('c4_en')
        else:
            from videoprism import tokenizers as vp_tokenizers

            self.text_tokenizer = vp_tokenizers.SentencePieceTokenizer(self.videoprism_tokenizer_path)

        self.apply_jit = jax.jit(lambda state, x, y, z: self.flax_model.apply(
            state, x, y, z, train=False, return_intermediate=True
        ), device=self.jax_dev)

    # def train(self, mode: bool) -> None:
    #     return super().train(False)
    
    def encode_video_and_text_with_videoprism(self, x: torch.Tensor, cot: str, batch_size: int = -1) -> torch.Tensor:
        self._init_jax()
        jax = self._jax
        vp = self._vp

        b, t, h, w, c = x.shape
        assert c == 3 and h == 288 and w == 288
        text_ids, text_paddings = vp.tokenize_texts(self.text_tokenizer, cot)

        x = jax.device_put(x.cpu().numpy(), device=self.jax_dev)

        text_ids      = jax.device_put(text_ids,   device=self.jax_dev)
        text_paddings = jax.device_put(text_paddings, device=self.jax_dev)

        video_embeddings, text_embeddings, outputs = self.apply_jit(self.loaded_state, x, text_ids, text_paddings)

        frame_embed = outputs['frame_embeddings']
        spatialtemporal_embed = einshape.jax_einshape(
            'b(ts)d->btsd', outputs['spatiotemporal_features'], t=frame_embed.shape[0]
        )

        return video_embeddings[0],frame_embed[0],spatialtemporal_embed[0][0],text_embeddings

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.synchformer is not None, 'Synchformer is not loaded'

        b, t, c, h, w = x.shape


        assert c == 3 and h == 224 and w == 224


        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x

    @torch.inference_mode()
    def encode_t5_text(self, text: list[str]) -> torch.Tensor:
        assert self.t5 is not None, 'T5 model is not loaded'
        assert self.t5tokenizer is not None, 'T5 Tokenizer is not loaded'
        # x: (B, L)
        inputs = {key: value.to(self.device) for key, value in self.t5tokenizer(text, padding=True, truncation=False, return_tensors="pt").items()}
        inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
        
        text_features = self.t5(**inputs).last_hidden_state
        return text_features
    
    @torch.inference_mode()
    def encode_audio(self, x) -> torch.Tensor:
        x = self.vae.encode(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


