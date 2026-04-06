import torch
import torch.nn as nn


class UnfoldNd(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        if len(self.kernel_size) != 3 or len(self.stride) != 3:
            raise ValueError("UnfoldNd expects 3D kernel_size and stride tuples.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected a 5D tensor for UnfoldNd, got shape {tuple(x.shape)}.")
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        patches = x.unfold(2, kt, st).unfold(3, kh, sh).unfold(4, kw, sw)
        patches = patches.permute(0, 1, 5, 6, 7, 2, 3, 4).contiguous()
        return patches.view(x.shape[0], x.shape[1] * kt * kh * kw, -1)
