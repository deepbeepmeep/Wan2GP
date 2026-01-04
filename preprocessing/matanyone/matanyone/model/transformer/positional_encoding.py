# Reference:
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/position_encoding.py
# https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py

import math

import numpy as np
import torch
from torch import nn


def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim: int,
                 scale: float = math.pi * 2,
                 temperature: float = 10000,
                 normalize: bool = True,
                 channel_last: bool = True,
                 transpose_output: bool = False):
        super().__init__()
        dim = int(np.ceil(dim / 4) * 2)
        self.dim = dim
        inv_freq = 1.0 / (temperature**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.normalize = normalize
        self.scale = scale
        self.eps = 1e-6
        self.channel_last = channel_last
        self.transpose_output = transpose_output

        self.cached_penc = None  # the cache is irrespective of the number of objects

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 4/5d tensor of size 
            channel_last=True: (batch_size, h, w, c) or (batch_size, k, h, w, c)
            channel_last=False: (batch_size, c, h, w) or (batch_size, k, c, h, w)
        :return: positional encoding tensor that has the same shape as the input if the input is 4d
                 if the input is 5d, the output is broadcastable along the k-dimension
        """
        if len(tensor.shape) != 4 and len(tensor.shape) != 5:
            raise RuntimeError(f'The input tensor has to be 4/5d, got {tensor.shape}!')

        if len(tensor.shape) == 5:
            # take a sample from the k dimension
            num_objects = tensor.shape[1]
            tensor = tensor[:, 0]
        else:
            num_objects = None

        if self.channel_last:
            batch_size, h, w, c = tensor.shape
        else:
            batch_size, c, h, w = tensor.shape

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            if num_objects is None:
                return self.cached_penc
            else:
                return self.cached_penc.unsqueeze(1)

        self.cached_penc = None

        pos_y = torch.arange(h, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_x = torch.arange(w, device=tensor.device, dtype=self.inv_freq.dtype)
        if self.normalize:
            pos_y = pos_y / (pos_y[-1] + self.eps) * self.scale
            pos_x = pos_x / (pos_x[-1] + self.eps) * self.scale

        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_x = get_emb(sin_inp_x)

        emb = torch.zeros((h, w, self.dim * 2), device=tensor.device, dtype=tensor.dtype)
        emb[:, :, :self.dim] = emb_x
        emb[:, :, self.dim:] = emb_y

        if not self.channel_last and self.transpose_output:
            # cancelled out
            pass
        elif (not self.channel_last) or (self.transpose_output):
            emb = emb.permute(2, 0, 1)

        self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if num_objects is None:
            return self.cached_penc
        else:
            return self.cached_penc.unsqueeze(1)


if __name__ == '__main__':
    pe = PositionalEncoding(8).cuda()
    input = torch.ones((1, 8, 8, 8)).cuda()
    output = pe(input)
    # print(output)
    print(output[0, :, 0, 0])
    print(output[0, :, 0, 5])
    print(output[0, 0, :, 0])
    print(output[0, 0, 0, :])
