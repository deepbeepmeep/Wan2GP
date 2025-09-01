import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ext.rotary_embeddings import compute_rope_rotations
from .embeddings import TimestepEmbedder
from .low_level import MLP, ChannelLastConv1d, ConvMLP
from .transformer_layers import (FinalBlock, JointBlock, MMDitSingleBlock)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor


# Partially from https://github.com/facebookresearch/DiT
class MMAudio(nn.Module):

    def __init__(self,
                 *,
                 latent_dim: int,
                 clip_dim: int,
                 sync_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 clip_seq_len: int,
                 sync_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 v2: bool = False) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if v2:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )
        else:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )

        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        if v2:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=hidden_dim,
                                            max_period=1)
        else:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=256,
                                            max_period=10000)
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                       num_heads,
                       mlp_ratio=mlp_ratio,
                       pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)

        self.initialize_weights()
        self.initialize_rotations()

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        clip_rot = compute_rope_rotations(self._clip_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * self._latent_seq_len /
                                          self._clip_seq_len,
                                          device=self.device)

        self.latent_rot = latent_rot #, persistent=False)
        self.clip_rot = clip_rot #, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        # empty string feat shall be initialized by a CLIP encoder
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self, clip_f: torch.Tensor, sync_f: torch.Tensor,
                              text_f: torch.Tensor) -> PreprocessedConditions:
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        assert clip_f.shape[1] == self._clip_seq_len, f'{clip_f.shape=} {self._clip_seq_len=}'
        assert sync_f.shape[1] == self._sync_seq_len, f'{sync_f.shape=} {self._sync_seq_len=}'
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = clip_f.shape[0]

        # B * num_segments (24) * 8 * 768
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)

        # extend vf to match x
        clip_f = self.clip_input_proj(clip_f)  # (B, VN, D)
        sync_f = self.sync_input_proj(sync_f)  # (B, VN, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        # upsample the sync features to match the audio
        sync_f = sync_f.transpose(1, 2)  # (B, D, VN)
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode='nearest-exact')
        sync_f = sync_f.transpose(1, 2)  # (B, N, D)

        # get conditional features from the clip side
        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))  # (B, D)
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)

        return PreprocessedConditions(clip_f=clip_f,
                                      sync_f=sync_f,
                                      text_f=text_f,
                                      clip_f_c=clip_f_c,
                                      text_f_c=text_f_c)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        """
        for non-cacheable computations
        """
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'

        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        text_f = conditions.text_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c

        latent = self.audio_input_proj(latent)  # (B, N, D)
        global_c = self.global_cond_mlp(clip_f_c + text_f_c)  # (B, D)

        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)  # (B, D)
        extended_c = global_c + sync_f



        self.latent_rot = self.latent_rot.to("cuda")
        self.clip_rot = self.clip_rot.to("cuda")
        for block in self.joint_blocks:
            latent, clip_f, text_f = block(latent, clip_f, text_f, global_c, extended_c,
                                           self.latent_rot, self.clip_rot)  # (B, N, D)

        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)
        self.latent_rot = self.latent_rot.to("cpu")
        self.clip_rot = self.clip_rot.to("cpu")

        # should be extended_c; this is a minor implementation error #55
        flow = self.final_layer(latent, global_c)  # (B, N, out_dim), remove t
        return flow

    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, sync_f: torch.Tensor,
                text_f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, N, C) 
        vf: (B, T, C_V)
        t: (B,)
        """
        conditions = self.preprocess_conditions(clip_f, sync_f, text_f)
        flow = self.predict_flow(latent, t, conditions)
        return flow

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)

        empty_clip = self.get_empty_clip_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)
        conditions = self.preprocess_conditions(empty_clip, empty_sync, empty_text)
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        else:
            return (cfg_strength * self.predict_flow(latent, t, conditions) +
                    (1 - cfg_strength) * self.predict_flow(latent, t, empty_conditions))

    def load_weights(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'clip_rot' in src_dict:
            del src_dict['clip_rot']

        a,b = self.load_state_dict(src_dict, strict=True, assign= True)
        pass

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len


def small_16k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=20,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=250,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def small_44k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def medium_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k_v2(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   v2=True,
                   **kwargs)


def get_my_mmaudio(name: str, **kwargs) -> MMAudio:
    if name == 'small_16k':
        return small_16k(**kwargs)
    if name == 'small_44k':
        return small_44k(**kwargs)
    if name == 'medium_44k':
        return medium_44k(**kwargs)
    if name == 'large_44k':
        return large_44k(**kwargs)
    if name == 'large_44k_v2':
        return large_44k_v2(**kwargs)

    raise ValueError(f'Unknown model name: {name}')


if __name__ == '__main__':
    network = get_my_mmaudio('small_16k')

    # print the number of parameters in terms of millions
    num_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')
