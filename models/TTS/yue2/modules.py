"""YuE2 building blocks adapted from upstream (Apache-2.0)."""
import math
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from shared.attention import pay_attention

class YuE2Config(PretrainedConfig):
    model_type = "yue2"

    _hf_fields = frozenset({
        "model_type", "architectures", "auto_map", "transformers_version",
        "dtype", "torch_dtype", "return_dict", "output_hidden_states",
        "output_attentions", "use_cache", "tie_word_embeddings", "torchscript",
        "is_decoder", "is_encoder_decoder", "add_cross_attention",
        "bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id",
        "attn_implementation",
    })

    def to_dict(self):
        return {key: value for key, value in super().to_dict().items()
                if key in self._hf_fields or key in self._inference_fields}

    _inference_fields = frozenset(['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size', 'rms_norm_eps', 'rope_theta', 'max_position_embeddings', 'tie_word_embeddings', 'latent_type', 'latent_dim', 'max_latent_frames', 'timestep_shift', 'ffn_chunk_size'])

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 6144,
        vocab_size: int = 184704,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 24576,
        tie_word_embeddings: bool = False,
        # Acoustic inference architecture
        latent_type: str = "vae",
        latent_dim: int = 64,
        max_latent_frames: int = 24576,
        timestep_shift: float = 1.0,
        ffn_chunk_size: int = 1024,
        **kwargs,
    ):
        if latent_type != "vae":
            raise ValueError("YuE2 inference supports only latent_type='vae'")
        # Serialize only the documented model and Transformers configuration.
        kwargs = {key: value for key, value in kwargs.items() if key in self._hf_fields}
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.latent_type = latent_type
        self.latent_dim = latent_dim
        self.max_latent_frames = max_latent_frames
        self.timestep_shift = timestep_shift
        self.ffn_chunk_size = ffn_chunk_size


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x_list) -> torch.Tensor:
        x = x_list.pop()
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._inv_freq: Optional[torch.Tensor] = None

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._inv_freq is None or self._inv_freq.device != position_ids.device:
            self._inv_freq = 1.0 / (self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device) / self.head_dim
            ))
        pos = position_ids.float().unsqueeze(-1)
        angles = pos * self._inv_freq
        return angles.cos(), angles.sin()


def _apply_rotary(x_list, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x_list.pop()
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    scratch = torch.empty_like(x)
    torch.mul(x1, sin, out=scratch[..., :half])
    torch.mul(x2, sin, out=scratch[..., half:])
    x1.mul_(cos).sub_(scratch[..., half:])
    x2.mul_(cos).add_(scratch[..., :half])
    return x


class Attention(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x_list, cache, cos, sin, engine):
        x = x_list.pop()
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        del x
        q_list, k_list = [q], [k]
        q = k = None
        q_list.append(self.q_norm(q_list))
        k_list.append(self.k_norm(k_list))
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        q = _apply_rotary(q_list, rc, rs)
        k = _apply_rotary(k_list, rc, rs)
        # Assemble prefix + audio directly in the attention head layout, including GQA.
        qkv_list = [q, k, v]
        q = k = v = None
        for index in (1, 2):
            source = qkv_list[index]
            prefix = cache[index - 1]
            combined = source.new_empty(B, prefix.shape[0] + T, self.num_heads, self.head_dim)
            grouped = combined.view(B, -1, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            grouped[:, :prefix.shape[0]].copy_(prefix[None, :, :, None, :])
            grouped[:, prefix.shape[0]:].copy_(source.unsqueeze(-2))
            qkv_list[index] = combined
            del source, combined, grouped
        h = pay_attention(qkv_list, force_attention="flash" if engine == "vllm" else "sdpa", recycle_q=True)
        return self.o_proj(h.flatten(2))



class MLP(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x_list) -> torch.Tensor:
        x = x_list.pop()
        gate, up = self.gate_proj(x), self.up_proj(x)
        del x
        F.silu(gate, inplace=True).mul_(up)
        del up
        return self.down_proj(gate)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → MLP → hidden_size (same as modules.py)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.parameters()).dtype))


class AudioPositionEmbedding(nn.Module):
    """Non-learnable 1D sinusoidal PE for audio latent frames."""

    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_frames, hidden_size)
        position = torch.arange(0, max_frames, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, position_ids):
        return self.pe[position_ids]

