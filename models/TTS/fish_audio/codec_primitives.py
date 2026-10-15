# Inference forms of the DAC layers already used by WanGP's IndexTTS integration.
# Derived from Descript Audio Codec (MIT), Copyright Descript, Inc. See LICENSE_DAC.
import torch
from torch import nn
from torch.nn import functional as F


def WNConv1d(*args, **kwargs):
    # Weight normalization is folded once during checkpoint preparation.
    return nn.Conv1d(*args, **kwargs)


def WNConvTranspose1d(*args, **kwargs):
    return nn.ConvTranspose1d(*args, **kwargs)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + torch.sin(self.alpha * x).square() / (self.alpha + 1e-9)


class VectorQuantize(nn.Module):
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.codebook_size = codebook_size
        self.in_proj = WNConv1d(input_dim, codebook_dim, 1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, 1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, ids):
        return self.codebook(ids).transpose(1, 2)

    def forward(self, x):
        latent = self.in_proj(x)
        rows = F.normalize(latent.transpose(1, 2).reshape(-1, latent.shape[1]), dim=-1)
        codebook = F.normalize(self.codebook.weight, dim=-1)
        distance = rows.square().sum(1, keepdim=True) - 2 * rows @ codebook.T + codebook.square().sum(1)[None]
        ids = (-distance).argmax(1).reshape(x.shape[0], -1)
        quantized = self.decode_code(ids)
        # Inference does not calculate training losses or straight-through gradients.
        zero = x.new_zeros(())
        return self.out_proj(quantized), zero, zero, ids, latent


class ResidualVectorQuantize(nn.Module):
    def __init__(self, input_dim=512, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.0):
        super().__init__()
        self.n_codebooks, self.codebook_size = n_codebooks, codebook_size
        self.quantizers = nn.ModuleList([VectorQuantize(input_dim, codebook_size, codebook_dim) for _ in range(n_codebooks)])

    def forward(self, x, n_quantizers=None):
        output, residual, codes, latents = torch.zeros_like(x), x, [], []
        for quantizer in self.quantizers[:n_quantizers or self.n_codebooks]:
            value, _, _, ids, latent = quantizer(residual)
            output = output + value
            residual = residual - value
            codes.append(ids)
            latents.append(latent)
        zero = x.new_zeros(())
        return output, torch.stack(codes, 1), torch.cat(latents, 1), zero, zero

    def from_codes(self, codes):
        latents = [self.quantizers[index].decode_code(codes[:, index]) for index in range(codes.shape[1])]
        output = self.quantizers[0].out_proj(latents[0])
        for index in range(1, len(latents)):
            output = output + self.quantizers[index].out_proj(latents[index])
        return output, torch.cat(latents, 1), codes
