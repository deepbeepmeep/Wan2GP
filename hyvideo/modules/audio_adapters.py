"""
This module provides the implementation of an Audio Projection Model, which is designed for
audio processing tasks. The model takes audio embeddings as input and outputs context tokens
that can be used for various downstream applications, such as audio analysis or synthesis.

The AudioProjModel class is based on the ModelMixin class from the diffusers library, which
provides a foundation for building custom models. This implementation includes multiple linear
layers with ReLU activation functions and a LayerNorm for normalization.

Key Features:
- Audio embedding input with flexible sequence length and block structure.
- Multiple linear layers for feature transformation.
- ReLU activation for non-linear transformation.
- LayerNorm for stabilizing and speeding up training.
- Rearrangement of input embeddings to match the model's expected input shape.
- Customizable number of blocks, channels, and context tokens for adaptability.

The module is structured to be easily integrated into larger systems or used as a standalone
component for audio feature extraction and processing.

Classes:
- AudioProjModel: A class representing the audio projection model with configurable parameters.

Functions:
- (none)

Dependencies:
- torch: For tensor operations and neural network components.
- diffusers: For the ModelMixin base class.
- einops: For tensor rearrangement operations.

"""

import torch
from diffusers import ModelMixin
from einops import rearrange

import math
import torch.nn as nn

class AudioProjNet2(ModelMixin):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=4,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)


    def forward(self, audio_embeds):
          
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )
        context_tokens = self.norm(context_tokens)
        out_all = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return out_all


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) 
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=1024, heads=33):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head #* heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        import torch.nn.init as init
        init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            init.zeros_(self.to_out.bias)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, t, aa, D)
            latent (torch.Tensor): latent features
                shape (b, t, hw, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        # print("latents shape: ", latents.shape)
        # print("x shape: ", x.shape)
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)


        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        # out = out.permute(0, 2, 1, 3)
        return self.to_out(out)
    #def forward(self, x, latents):
    #    """
    #    Args:
    #        x (torch.Tensor): image features
    #            shape (b, t, aa, D)
    #        latent (torch.Tensor): latent features
    #            shape (b, t, hw, D)
    #    """
    #    if get_sequence_parallel_state():
    #        sp_size = nccl_info.sp_size
    #        sp_rank = nccl_info.rank_within_group
    #        print("rank:", latents.shape, sp_size, sp_rank)
    #        latents = torch.chunk(latents, sp_size, dim=1)[sp_rank]

    #    x = self.norm1(x)
    #    latents = self.norm2(latents)
    #    # print("latents shape: ", latents.shape)
    #    # print("x shape: ", x.shape)
    #    q = self.to_q(latents)
    #    k, v = self.to_kv(x).chunk(2, dim=-1)

    #    # print("q, k, v: ", q.shape, k.shape, v.shape)

    #    # attention
    #    #scale = 1 / math.sqrt(math.sqrt(self.dim_head))
    #    #weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
    #    #weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    #    #out = weight @ v
    #    def shrink_head(encoder_state, dim):
    #        local_heads = encoder_state.shape[dim] // nccl_info.sp_size
    #        return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads, local_heads)

    #    if get_sequence_parallel_state():
    #    # batch_size, seq_len, attn_heads, head_dim
    #        q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)  # [2, 32256, 24, 128]
    #        k = shrink_head(k ,dim=2)
    #        v = shrink_head(v ,dim=2)
    #    qkv = torch.stack([query, key, value], dim=2)
    #    attn = flash_attn_no_pad(qkv, causal=False, dropout_p=0.0, softmax_scale=None)
    #    # out = out.permute(0, 2, 1, 3)
    #    #b, s, a, d = attn.shape
    #    #attn = attn.reshape(b, s, -1)
    #        
    #    out = self.to_out(attn)
    #    if get_sequence_parallel_state():
    #        out = all_gather(out, dim=1)
    #    return out
