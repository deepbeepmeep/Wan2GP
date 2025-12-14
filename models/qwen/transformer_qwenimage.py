# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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


import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from shared.attention import pay_attention
import functools

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen_inplace(x_list: list, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings in-place using real arithmetic (no complex numbers).

    Args:
        x_list: Single-element list containing query or key tensor [B, S, H, D].
                The list is cleared after use to free memory early.
        freqs_cis: Precomputed frequency tensor with cos/sin stacked [S, D//2, 2]

    Returns:
        Tensor with rotary embeddings applied [B, S, H, D]
    """
    x_in = x_list[0]
    dtype = x_in.dtype
    # Reshape to separate real/imag pairs: [B, S, H, D] -> [B, S, H, D//2, 2]
    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
    x_in = None
    x_list.clear()

    # freqs_cis shape: [S, D//2, 2] where [..., 0] is cos, [..., 1] is sin
    # Add batch and head dims: [1, S, 1, D//2]
    cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(2)
    sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(2)

    # Get views into x for in-place modification
    x0 = x[..., 0]  # real part
    x1 = x[..., 1]  # imag part

    # Apply rotation in-place:
    # new_x0 = x0 * cos - x1 * sin
    # new_x1 = x1 * cos + x0 * sin
    x0_orig = x0.clone()
    x0.mul_(cos)
    x0.addcmul_(x1, sin, value=-1)
    x1.mul_(cos)
    x1.addcmul_(x0_orig, sin)

    return x.flatten(3).to(dtype)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        Returns:
            Tensor of shape [len(index), dim//2, 2] with cos/sin stacked in last dim
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        # Stack cos and sin instead of using complex numbers (better for torch.compile)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling() and False:
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        # freqs_pos/neg now have shape [4096, axes_dim[i]//2, 2] with cos/sin in last dim
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        # Keep the cos/sin dimension (2) separate throughout
        freqs_frame = freqs_pos[0][idx : idx + frame]  # [frame, d0//2, 2]
        freqs_frame = freqs_frame.view(frame, 1, 1, -1, 2).expand(frame, height, width, -1, 2)

        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)

        # Concatenate on freq dimension (dim -2), keep cos/sin dimension (dim -1) intact
        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-2)  # [F, H, W, total//2, 2]
        freqs = freqs.reshape(seq_lens, -1, 2)  # [seq_lens, total//2, 2]
        return freqs.clone().contiguous()


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states_list = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        hidden_states, encoder_hidden_states = hidden_states_list
        hidden_states_list.clear()

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        del hidden_states
        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        del encoder_hidden_states
        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE (in-place, no complex numbers for better torch.compile support)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            x_list = [img_query]
            del img_query
            img_query = apply_rotary_emb_qwen_inplace(x_list, img_freqs)
            x_list = [img_key]
            del img_key
            img_key = apply_rotary_emb_qwen_inplace(x_list, img_freqs)
            x_list = [txt_query]
            del txt_query
            txt_query = apply_rotary_emb_qwen_inplace(x_list, txt_freqs)
            x_list = [txt_key]
            del txt_key
            txt_key = apply_rotary_emb_qwen_inplace(x_list, txt_freqs)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        del txt_query, img_query
        joint_key = torch.cat([txt_key, img_key], dim=1)
        del txt_key, img_key
        joint_value = torch.cat([txt_value, img_value], dim=1)
        del txt_value, img_value

        # Compute joint attention
        dtype = joint_query.dtype
        qkv_list = [joint_query, joint_key, joint_value ]
        del joint_query, joint_key, joint_value
        joint_hidden_states = pay_attention(qkv_list)

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part
        del joint_hidden_states
        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # FFN expansion factor for chunking (diffusers FeedForward default is 4x)
        self.ffn_mult = 4

    def _apply_ffn_chunked(self, ffn_in: torch.Tensor, mlp: nn.Module) -> None:
        _, seq_len, dim = ffn_in.shape
        ffn_in_flat = ffn_in.reshape(-1, dim)
        chunk_size = max(seq_len // self.ffn_mult, 1)
        for ffn_chunk in torch.split(ffn_in_flat, chunk_size):
            ffn_chunk[...] = mlp(ffn_chunk)
        # ffn_in is already modified in-place via ffn_in_flat view

    def _modulate_inplace(self, x, shift, scale, gate):
        scale.add_(1.0)
        x.mul_(scale.unsqueeze(1))
        x.add_(shift.unsqueeze(1))
        return gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams and split into shift, scale, gate
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self.img_mod(temb).chunk(6, dim=-1)
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.txt_mod(temb).chunk(6, dim=-1)

        # Process image stream - norm1 + modulation (in-place)
        img_normed = self.img_norm1(hidden_states)
        img_gate1 = self._modulate_inplace(img_normed, img_shift1, img_scale1, img_gate1)

        # Process text stream - norm1 + modulation (in-place)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_gate1 = self._modulate_inplace(txt_normed, txt_shift1, txt_scale1, txt_gate1)


        hidden_states_list = [img_normed, txt_normed]
        del img_normed, txt_normed
        joint_attention_kwargs = joint_attention_kwargs or {}
        img_attn_output, txt_attn_output = self.attn.processor(self.attn,
            hidden_states_list=hidden_states_list,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Apply attention gates and add residual in-place
        hidden_states.addcmul_(img_attn_output, img_gate1)
        encoder_hidden_states.addcmul_(txt_attn_output, txt_gate1)

        img_normed2 = self.img_norm2(hidden_states)
        img_gate2 = self._modulate_inplace(img_normed2, img_shift2, img_scale2, img_gate2)
        self._apply_ffn_chunked(img_normed2, self.img_mlp)
        # img_normed2 = self.img_mlp(img_normed2)

        hidden_states.addcmul_(img_normed2, img_gate2)

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_gate2 = self._modulate_inplace(txt_normed2, txt_shift2, txt_scale2, txt_gate2)
        txt_normed2 = self.txt_mlp(txt_normed2)

        encoder_hidden_states.addcmul_(txt_normed2, txt_gate2)

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(nn.Module): 
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]


    def preprocess_loras(self, model_type, sd):

        first = next(iter(sd), None)
        if first == None:
            return sd

        new_sd = {}
        for k,v in sd.items():
            k = k.replace(".lora.", ".lora_")
            k = k.replace(".default.weight", ".weight")
            new_sd[k] = v
        sd = new_sd  

        prefix_list = ["lora_unet_transformer_blocks"]
        for prefix in prefix_list: 
            if first.startswith(prefix):
                repl_list = ["attn", "img_mlp", "txt_mlp", "img_mod", "txt_mod"]
                src_list = ["_" + k + "_" for k in repl_list]
                tgt_list = ["." + k + "." for k in repl_list]
                src_list2 = ["_0_", "_0.", "_1.", "_2."]
                tgt_list2 = [".0.", ".0.", ".1.", ".2."]
                new_sd = {}
                for k,v in sd.items():
                    k = "diffusion_model.transformer_blocks." + k[len(prefix)+1:]
                    for s,t in zip(src_list, tgt_list):
                        k = k.replace(s,t)
                    for s,t in zip(src_list2, tgt_list2):
                        k = k.replace(s,t)
                    new_sd[k] = v
                sd = new_sd  
                return sd

        prefix_list = ["transformer_blocks"]
        for prefix in prefix_list: 
            if first.startswith(prefix):
                new_sd = {}
                for k,v in sd.items():
                    if k.startswith(prefix):
                        k = "diffusion_model." + k
                        new_sd[k] = v
                sd = new_sd  
                return sd
        
        return sd

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds
        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states_list  = None,
        encoder_hidden_states_mask_list = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens_list  = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback= None,
        pipeline =None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        hidden_states_list = [hidden_states if i == 0 else hidden_states.clone() for i, _ in enumerate(encoder_hidden_states_list)]
        
        new_encoder_hidden_states_list = []
        for encoder_hidden_states in encoder_hidden_states_list:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)
            new_encoder_hidden_states_list.append(encoder_hidden_states)
        encoder_hidden_states_list = new_encoder_hidden_states_list
        new_encoder_hidden_states_list = encoder_hidden_states = None
        
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb_list = [ self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device) for txt_seq_lens in txt_seq_lens_list] 

        hidden_states = None

        for index_block, block in enumerate(self.transformer_blocks):
            if callback != None:
                callback(-1, None, False, True)
            if pipeline._interrupt:
                return [None] * len(hidden_states_list)
            for hidden_states, encoder_hidden_states, encoder_hidden_states_mask, image_rotary_emb in zip(hidden_states_list, encoder_hidden_states_list, encoder_hidden_states_mask_list, image_rotary_emb_list):
                encoder_hidden_states[...], hidden_states[...] = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        output_list = []
        for i in range(len(hidden_states_list)):
            hidden_states = self.norm_out(hidden_states_list[i], temb)
            hidden_states_list[i] = None
            output_list.append(self.proj_out(hidden_states))

        return output_list
