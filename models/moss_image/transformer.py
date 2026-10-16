# Adapted from OpenMOSS MOSS-Image, Apache-2.0; see LICENSE.
import math
from types import SimpleNamespace

import torch
from torch import nn
from shared import attention_kit
from shared.kernels import ops
from shared.utils.lora_mapping import convert_lora_keys
from shared.utils.phase_progress import GenerationAborted


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return ops.rms_norm_rounded([x], self.eps, manual=True) * self.weight


def situ(hidden, gate):
    # MOSS's bounded SiLU gate and value saturation are checkpoint-native.
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attend(x_list, projections, q_norm, k_norm, rope, heads):
    def normalize(q, k, group):
        for tensor, norm in ((q, q_norm), (k, k_norm)):
            if tensor is not None:
                attention_kit.rows_(tensor, lambda part, first, last: rotate(norm(part), rope[:, first:last]))
    return attention_kit.qkv_attention(x_list, *projections, heads, q_norm.weight.numel(), normalize).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        x = x + self.proj(attend([self.norm1(x)], (self.q, self.k, self.v), self.q_norm, self.k_norm, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.expansion = 2 * ffn_dim / dim
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')})
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, streams, ropes, modulation, skip_branch=-1):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        for branch in range(len(streams)):
            if branch == skip_branch:
                continue
            x, streams[branch] = streams[branch], None
            y = ops.norm_scale_shift([x], self.norm1, scale_a[:, None], shift_a[:, None])
            y_list, y = [y], None
            attended = attend(y_list, tuple(self.proj[name] for name in ('q', 'k', 'v')), self.q_norm, self.k_norm, ropes[branch], self.heads)
            ops.gated_residual_(x, [self.proj['o'](attended)], gate_a[:, None])
            del attended
            chunk = math.ceil(x.shape[1] / self.expansion) if x.shape[1] > 1024 else x.shape[1]
            for start in range(0, x.shape[1], chunk):
                part = x[:, start:start + chunk]
                y = ops.norm_scale_shift([part], self.norm2, scale_f[:, None], shift_f[:, None])
                hidden, gate = self.proj['ffn_up_gate'](y).chunk(2, -1)
                del y
                activated = situ(hidden, gate)
                del hidden, gate
                ops.gated_residual_(part, [self.proj['ffn_down'](activated)], gate_f[:, None])
                del activated
            streams[branch] = x
        return streams


def rope_embed(height, width, text_length, device, head_dim=128, base=10000.0):
    bands = 1 / (base ** (torch.arange(head_dim // 8, device='cpu', dtype=torch.float32) / (head_dim // 8)))
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands.to(device)
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        self.config = SimpleNamespace(dim=dim, freq_dim=freq_dim, num_heads=num_heads, latent_channels=latent_channels, rope_base=rope_base)
        self._interrupt = False
        self.split_linear_modules_map = {'qkv': {'mapped_modules': ['q', 'k', 'v'], 'split_sizes': [dim] * 3}}
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys(state_dict, dict(self.named_modules()), split_linear_modules_map=self.split_linear_modules_map)

    def forward(self, latents, timestep, contexts, ropes, skip_layer=22):
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=latents.device, dtype=torch.float64) / half)
        phase = timestep.double().reshape(1, 1) * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        modulation = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(1, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(1, 2, -1)
        video = self.video_embedder(latents.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous())
        tokens = video.shape[1]
        streams = []
        for context, rope in zip(contexts, ropes):
            text = self.text_embedder.proj(context.to(self.text_embedder.proj.weight.dtype))
            for block in self.text_embedder.blocks:
                text = block(text, rope[:, tokens:])
                if self._interrupt:
                    raise GenerationAborted
            streams.append(torch.cat((video, text.expand(video.shape[0], -1, -1)), 1))
        del video, text
        for index, layer in enumerate(self.layers):
            streams = layer(streams, ropes, modulation, 0 if len(contexts) == 2 and index == skip_layer else -1)
            if self._interrupt:
                raise GenerationAborted
        results = []
        for x in streams:
            x = self.final_norm_video(x[:, :tokens].to(self.final_norm_video.weight.dtype))
            x = ops.scale_shift([x], head[:, 1, None], head[:, 0, None])
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            results.append(x.transpose(1, 2).reshape_as(latents).float())
        return results
