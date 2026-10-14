# Adapted from DeepPhonemizer (https://github.com/as-ideas/DeepPhonemizer, commit 5dce7e2, dp/model/model.py and dp/model/utils.py).
# MIT License, Copyright (c) 2021 Axel Springer News Media & Tech GmbH & Co. KG - Ideas Engineering; see LICENSE-DeepPhonemizer.
# WanGP changes: inference only (ForwardTransformer.generate and get_dedup_tokens), the positional table is the checkpoint
# parameter (not recomputed), the nested-tensor path upstream could not use is disabled explicitly.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=False)

    def forward(self, x):  # [T, N, C]
        return x + self.scale * self.pe[:x.size(0)]


class ForwardTransformer(nn.Module):
    """Non-autoregressive G2P: each character is repeated `char_repeats` times and every position predicts one phoneme or padding."""

    def __init__(self, encoder_vocab_size: int, decoder_vocab_size: int, d_model: int = 512, d_fft: int = 1024, layers: int = 4, heads: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_fft, dropout=0.1, activation="relu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, text: torch.Tensor) -> torch.Tensor:  # [N, T] token ids, 0 = padding -> [N, T, V] logits
        x = text.transpose(0, 1)
        x = self.encoder(self.pos_encoder(self.embedding(x)), src_key_padding_mask=text == 0)
        return self.fc_out(x).transpose(0, 1)

    def generate(self, text: torch.Tensor) -> list[list[int]]:
        """Most probable phoneme token per position, padding removed and consecutive repeats merged (upstream get_dedup_tokens)."""
        tokens = self.forward(text).argmax(-1)
        return [torch.unique_consecutive(row[row != 0]).tolist() for row in tokens]
