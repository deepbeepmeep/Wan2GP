# References:
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/rotary_positional_embedding.py

import torch
import torch.nn as nn

from einops import rearrange, repeat

def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((-x_imag, x_real), dim=-1).flatten(-2)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self,
                 head_dim,
                 cp_split_hw=None
                 ):
        """Rotary positional embedding for 3D
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
        """
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, 'Dim must be a multiply of 8 for 3D RoPE.'
        self.cp_split_hw = cp_split_hw
        # We take the assumption that the longest side of grid will not larger than 512, i.e, 512 * 8 = 4098 input pixels
        self.base = 10000
        self.freqs_dict = {}

    def register_grid_size(self, grid_size):
        key = tuple(grid_size)
        if key not in self.freqs_dict:
            self.freqs_dict[key] = self.precompute_freqs_cis_3d(grid_size)

    def precompute_freqs_cis_3d(self, grid_size):
        num_frames, height, width = grid_size     
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        cpu = torch.device("cpu")
        freqs_t = 1.0 / (
            self.base ** (torch.arange(0, dim_t, 2, device=cpu, dtype=torch.float32)[: (dim_t // 2)] / dim_t)
        )
        freqs_h = 1.0 / (
            self.base ** (torch.arange(0, dim_h, 2, device=cpu, dtype=torch.float32)[: (dim_h // 2)] / dim_h)
        )
        freqs_w = 1.0 / (
            self.base ** (torch.arange(0, dim_w, 2, device=cpu, dtype=torch.float32)[: (dim_w // 2)] / dim_w)
        )
        grid_t = torch.arange(num_frames, device=cpu, dtype=torch.float32)
        grid_h = torch.arange(height, device=cpu, dtype=torch.float32)
        grid_w = torch.arange(width, device=cpu, dtype=torch.float32)
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        # (T H W D)
        freqs = rearrange(freqs, "T H W D -> (T H W) D")

        return freqs

    def forward(self, q, k, grid_size):
        """3D RoPE.

        Args:
            query: [B, head, seq, head_dim]
            key: [B, head, seq, head_dim]
        Returns:
            query and key with the same shape as input.
        """

        key = tuple(grid_size)
        if key not in self.freqs_dict:
            self.register_grid_size(grid_size)

        freqs = self.freqs_dict[key].to(device=q.device, dtype=torch.float32)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        q_out = q.float()
        k_out = k.float()
        q_rot = rotate_half(q_out)
        k_rot = rotate_half(k_out)
        q_out.mul_(cos).add_(q_rot.mul_(sin))
        k_out.mul_(cos).add_(k_rot.mul_(sin))

        return q_out.to(q.dtype), k_out.to(k.dtype)
