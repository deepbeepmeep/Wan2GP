from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from .utils.maskgct.models.codec.amphion_codec.quantize import ResidualVQ
from .utils.maskgct.models.codec.kmeans.vocos import VocosBackbone


class EnhancedCodec(nn.Module):
    def __init__(self, codebook_size=8192, hidden_size=1024, codebook_dim=8, vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12, num_quantizers=1, downsample_scale=2, cfg=None):
        super().__init__()
        if cfg is not None:
            codebook_size = cfg.get("codebook_size", codebook_size)
            hidden_size = cfg.get("hidden_size", hidden_size)
            codebook_dim = cfg.get("codebook_dim", codebook_dim)
            vocos_dim = cfg.get("vocos_dim", vocos_dim)
            vocos_intermediate_dim = cfg.get("vocos_intermediate_dim", vocos_intermediate_dim)
            vocos_num_layers = cfg.get("vocos_num_layers", vocos_num_layers)
            num_quantizers = cfg.get("num_quantizers", num_quantizers)
            downsample_scale = cfg.get("downsample_scale", downsample_scale)
        self.downsample_scale = downsample_scale
        if downsample_scale is not None and downsample_scale > 1:
            self.down = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
            self.up = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            VocosBackbone(input_channels=hidden_size, dim=vocos_dim, intermediate_dim=vocos_intermediate_dim, num_layers=vocos_num_layers, adanorm_num_embeddings=None),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(input_channels=hidden_size, dim=vocos_dim, intermediate_dim=vocos_intermediate_dim, num_layers=vocos_num_layers, adanorm_num_embeddings=None),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.quantizer = ResidualVQ(input_dim=hidden_size, num_quantizers=num_quantizers, codebook_size=codebook_size, codebook_dim=codebook_dim, quantizer_type="fvq", quantizer_dropout=0.0, commitment=0.15, codebook_loss_weight=1.0, use_l2_normlize=True)

    def quantize(self, hidden_states):
        if self.downsample_scale is not None and self.downsample_scale > 1:
            hidden_states = F.gelu(self.down(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = self.encoder(hidden_states.transpose(1, 2)).transpose(1, 2)
        quantized, indices, _, _, _ = self.quantizer(hidden_states)
        if indices.shape[0] == 1:
            indices = indices.squeeze(0)
        return indices, quantized.transpose(1, 2)

    def decode(self, codes):
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        hidden_states = self.decoder(self.quantizer.vq2emb(codes))
        if self.downsample_scale is not None and self.downsample_scale > 1:
            hidden_states = F.interpolate(hidden_states.transpose(1, 2), scale_factor=self.downsample_scale, mode="nearest")
            hidden_states = self.up(hidden_states).transpose(1, 2)
        return hidden_states

    def forward(self, codes):
        return self.decode(codes)
