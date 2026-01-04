from typing import Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding


# @torch.jit.script
def _weighted_pooling(masks: torch.Tensor, value: torch.Tensor,
                      logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    # value: B*num_objects*H*W*value_dim
    # logits: B*num_objects*H*W*num_summaries
    # masks: B*num_objects*H*W*num_summaries: 1 if allowed
    weights = logits.sigmoid() * masks
    # B*num_objects*num_summaries*value_dim
    sums = torch.einsum('bkhwq,bkhwc->bkqc', weights, value)
    # B*num_objects*H*W*num_summaries -> B*num_objects*num_summaries*1
    area = weights.flatten(start_dim=2, end_dim=3).sum(2).unsqueeze(-1)

    # B*num_objects*num_summaries*value_dim
    return sums, area


class ObjectSummarizer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_summarizer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature

        if self.add_pe:
            self.pos_enc = PositionalEncoding(self.embed_dim,
                                              scale=self.pixel_pe_scale,
                                              temperature=self.pixel_pe_temperature)

        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_summaries),
        )

    def forward(self,
                masks: torch.Tensor,
                value: torch.Tensor,
                need_weights: bool = False) -> (torch.Tensor, Optional[torch.Tensor]):
        # masks: B*num_objects*(H0)*(W0)
        # value: B*num_objects*value_dim*H*W
        # -> B*num_objects*H*W*value_dim
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode='area')
        masks = masks.unsqueeze(-1)
        inv_masks = 1 - masks
        repeated_masks = torch.cat([
            masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
            inv_masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
        ],
                                   dim=-1)

        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe

        with torch.amp.autocast("cuda"):
            value = value.float()
            feature = self.feature_pred(value)
            logits = self.weights_pred(value)
            sums, area = _weighted_pooling(repeated_masks, feature, logits)

        summaries = torch.cat([sums, area], dim=-1)

        if need_weights:
            return summaries, logits
        else:
            return summaries, None