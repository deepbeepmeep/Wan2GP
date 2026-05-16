import torch
from torch import nn
import math

from .dit_model import ModelArgs, Transformer
from ..commons import sequence_mask

from torch.nn.utils import weight_norm

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, scale=1000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = scale * t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT(torch.nn.Module):
    def __init__(
        self,
        time_as_token,
        style_as_token,
        uvit_skip_connection,
        block_size,
        depth,
        num_heads,
        hidden_dim,
        in_channels,
        content_dim,
        style_encoder_dim,
        class_dropout_prob,
        dropout_rate,
        attn_dropout_rate,
    ):
        super(DiT, self).__init__()
        self.time_as_token = time_as_token
        self.style_as_token = style_as_token
        self.uvit_skip_connection = uvit_skip_connection
        model_args = ModelArgs(
            block_size=block_size,
            n_layer=depth,
            n_head=num_heads,
            dim=hidden_dim,
            head_dim=hidden_dim // num_heads,
            vocab_size=1, # we don't use this
            uvit_skip_connection=self.uvit_skip_connection,
            time_as_token=self.time_as_token,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = weight_norm(nn.Linear(in_channels, hidden_dim, bias=True))

        self.content_dim = content_dim # for continuous content
        self.cond_projection = nn.Linear(content_dim, hidden_dim, bias=True) # continuous content

        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.final_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, in_channels),
        )

        self.class_dropout_prob = class_dropout_prob

        self.cond_x_merge_linear = nn.Linear(hidden_dim + in_channels + in_channels, hidden_dim)
        self.style_in = nn.Linear(style_encoder_dim, hidden_dim)

    def forward(self, x, prompt_x, x_lens, t, style, cond):
        class_dropout = False
        content_dropout = False
        if self.training and torch.rand(1) < self.class_dropout_prob:
            class_dropout = True
            if self.training and torch.rand(1) < 0.5:
                content_dropout = True
        cond_in_module = self.cond_projection

        B, _, T = x.size()

        t1 = self.t_embedder(t)  # (N, D)
        cond = cond_in_module(cond)

        x = x.transpose(1, 2)
        prompt_x = prompt_x.transpose(1, 2)

        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        if class_dropout:
            x_in[..., self.in_channels:self.in_channels*2] = 0
            if content_dropout:
                x_in[..., self.in_channels*2:] = 0
        x_in = self.cond_x_merge_linear(x_in)  # (N, T, D)

        style = self.style_in(style)
        style = torch.zeros_like(style) if class_dropout else style
        if self.style_as_token:
            x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)
        if self.time_as_token:
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)
        x_mask = sequence_mask(x_lens + self.style_as_token + self.time_as_token, max_length=x_in.size(1)).to(x.device).unsqueeze(1)
        input_pos = torch.arange(x_in.size(1)).to(x.device)
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1)
        x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)
        x_res = x_res[:, 1:] if self.time_as_token else x_res
        x_res = x_res[:, 1:] if self.style_as_token else x_res
        x = self.final_mlp(x_res)
        x = x.transpose(1, 2)
        return x
