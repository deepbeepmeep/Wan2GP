# Adapted from Black Forest Labs' flux-3-image-klein, licensed under Apache-2.0 (LICENSE).
import math

import torch
import torch.nn.functional as F
from torch import nn
from mmgp import cuda_graphs
from shared import attention_kit
from shared.kernels import ops as kernels
from shared.utils.phase_progress import GenerationAborted


def timestep_embedding(t, dim=256):
    freqs = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device, dtype=torch.float32) / (dim // 2))
    args = (1000 * t)[:, None].float() * freqs[None]
    return torch.cat((args.cos(), args.sin()), -1).to(t)


def rope_embed(ids):
    scale = torch.arange(0, 32, 2, dtype=torch.float64, device=ids.device) / 32
    angles = ids.double()[..., None] * (1.0 / (10000 ** scale))
    angles = angles.flatten(-2)
    return torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), -1).unflatten(-1, (2, 2)).float()[None, :, None]


def norm_rope(x, weight, freqs):
    x_list, x = [x], None
    return kernels.matrix_rms_rope(x_list, weight, freqs)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm, self.key_norm = RMSNorm(dim), RMSNorm(dim)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec):
        return self.lin(F.silu(vec)).chunk(3, -1)


def modulate(x, shift, scale):
    x_list, x = [x], None
    return kernels.layer_norm_modulate(x_list, shift, scale)


class LastLayer(nn.Module):
    def __init__(self, hidden, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, -1)
        return self.linear(modulate(x, shift, scale))


class Block(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(self.head_dim)
        self.mlp_expansion = 2 * mlp_ratio

    def forward(self, streams, frequencies, modulations):
        # CFG branches execute sequentially while this block's weights are loaded.
        for branch in range(len(streams)):
            x = streams[branch]
            streams[branch] = None
            spans, freqs = modulations[branch], frequencies[branch]
            # Keep expanded visual FFN intermediates roughly input-sized, while
            # short text streams use one projection call.
            chunk_size = math.ceil(x.shape[0] / self.mlp_expansion) if x.shape[0] > 1024 else x.shape[0]
            x_mod = torch.empty_like(x)
            for start, end, (shift, scale, gate) in spans:
                for first in range(start, end, chunk_size):
                    last = min(first + chunk_size, end)
                    x_mod[first:last] = modulate(x[first:last], shift, scale)

            def normalize(q, k, group):
                for tensor, weight in ((q, self.norm.query_norm.scale), (k, self.norm.key_norm.scale)):
                    if tensor is not None:
                        attention_kit.rows_(tensor, lambda part, first, last: norm_rope(part, weight, freqs[:, first:last]))

            attn = attention_kit.qkv_attention([x_mod], self.q_proj, self.k_proj, self.v_proj, self.heads, self.head_dim, normalize).flatten(2)[0]
            for start, end, (shift, scale, gate) in spans:
                for first in range(start, end, chunk_size):
                    last = min(first + chunk_size, end)
                    activated = kernels.swiglu([self.mlp_in(x_mod[first:last])])
                    result = self.attn_out(attn[first:last]) + self.mlp_out(activated)
                    del activated
                    result_list, result = [result], None
                    kernels.gated_residual_(x[first:last], result_list, gate)
            streams[branch] = x
        return streams


class DiT(nn.Module):
    def __init__(self, config, flow_map=False):
        super().__init__()
        self.flow_map = flow_map
        self._interrupt = False
        hidden, heads = config['hidden_size'], config['num_heads']
        self.split_linear_modules_map = {'qkv': {'mapped_modules': ['q_proj', 'k_proj', 'v_proj'], 'split_sizes': [hidden] * 3}}
        self.depth, self.depth_single = config['depth'], config['depth_single']
        self.emb_in = nn.ModuleDict({m: nn.Linear(128, hidden, bias=False) for m in ('image', 'image_cond')})
        self.txt_in = nn.Linear(config['context_dim'], hidden, bias=False)
        self.time_in = MLPEmbedder(256, hidden)
        if flow_map:
            self.ref_time_in = MLPEmbedder(256, hidden)
        self.early_stream_modulations = nn.ModuleDict({m: ModeModulation(hidden) for m in ('image', 'image_cond', 'txt')})
        self.single_stream_modulations = nn.ModuleDict({m: ModeModulation(hidden) for m in ('image', 'image_cond', 'txt')})
        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([Block(hidden, heads, config['mlp_ratio']) for _ in range(self.depth)]) for m in ('image', 'image_cond')})
        self.txt_mode_blocks = nn.ModuleList([Block(hidden, heads, config['mlp_ratio']) for _ in range(self.depth)])
        self.single_blocks = nn.ModuleList([Block(hidden, heads, config['mlp_ratio']) for _ in range(self.depth_single)])
        self.final_layer = nn.ModuleDict({'image': LastLayer(hidden, 128)})

    def preprocess_loras(self, model_type, state_dict):
        from shared.utils.lora_mapping import convert_lora_keys
        return convert_lora_keys(state_dict, dict(self.named_modules()), split_linear_modules_map=self.split_linear_modules_map, compose_lokr=True)

    def check_abort(self):
        if self._interrupt:
            raise GenerationAborted

    def forward(self, image, texts, reference, frequencies, timestep, target_timestep=None):
        branches = len(texts)
        modes = ['image'] + (['image_cond'] if reference is not None else []) + ['text']
        dtype = texts[0].dtype
        time = torch.stack((timestep, timestep.new_zeros(()))).to(dtype)
        vecs = self.time_in(timestep_embedding(time))
        if self.flow_map:
            vecs = vecs + self.ref_time_in(timestep_embedding(target_timestep.reshape(1).to(dtype)))
        vec = {m: vecs[0:1] if m == 'image' else vecs[1:2] for m in modes}
        key = lambda m: 'txt' if m == 'text' else m
        early = {m: self.early_stream_modulations[key(m)](vec[m]) for m in modes}
        single = {m: self.single_stream_modulations[key(m)](vec[m]) for m in modes}
        x = {'image': [self.emb_in['image'](image.to(dtype)) for _ in texts], 'text': [self.txt_in(text) for text in texts]}
        if reference is not None:
            x['image_cond'] = [self.emb_in['image_cond'](reference) for _ in texts]
        spans = {m: [[(0, item.shape[0], early[m])] for item in x[m]] for m in modes}
        with cuda_graphs.scope():
            for depth in range(self.depth):
                for mode in modes:
                    block = self.txt_mode_blocks[depth] if mode == 'text' else self.content_mode_blocks[mode][depth]
                    x[mode] = block(x[mode], frequencies[mode], spans[mode])
                    self.check_abort()
            combined, mods = [], []
            for branch in range(branches):
                offset, branch_mods = 0, []
                for mode in modes:
                    end = offset + x[mode][branch].shape[0]
                    branch_mods.append((offset, end, single[mode]))
                    offset = end
                combined.append(torch.cat([x[m][branch] for m in modes]))
                mods.append(branch_mods)
            del x, early, spans
            for block in self.single_blocks:
                combined = block(combined, frequencies['single'], mods)
                self.check_abort()
        return [self.final_layer['image'](item[:image.shape[0]], vec['image']) for item in combined]
