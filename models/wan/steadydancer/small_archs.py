import torch
import torch.nn as nn


class FactorConv3d(nn.Module):
    """
    (2+1)D 分解 3D 卷积：1×H×W 空间卷积 → Swish → T×1×1 时间卷积
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride: int = 1,
                 dilation: int = 1):
        super().__init__()

        if isinstance(kernel_size, int):
            k_t, k_h, k_w = kernel_size, kernel_size, kernel_size
        else:
            k_t, k_h, k_w = kernel_size

        pad_t  = (k_t - 1) * dilation // 2
        pad_hw = (k_h - 1) * dilation // 2

        self.spatial = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(1, k_h, k_w),
            stride=(1, stride, stride),
            padding=(0, pad_hw, pad_hw),
            dilation=(1, dilation, dilation),
            groups=in_channels,
            bias=False
        )

        self.temporal = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(k_t, 1, 1),
            stride=(stride, 1, 1),
            padding=(pad_t, 0, 0),
            dilation=(dilation, 1, 1),
            bias=True
        )

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.spatial(x)
        x = self.act(x)
        x = self.temporal(x)
        return x


class LayerNorm2D(nn.Module):
    """
    LayerNorm over C for a 4-D tensor (B, C, H, W)
    """
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)        # (B, 1, H, W)
        var  = x.var (dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class PoseRefNetNoBNV3(nn.Module):
    def __init__(self,
                 in_channels_c: int,
                 in_channels_x: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.nhead = num_heads

        self.proj_p = nn.Conv2d(in_channels_c, hidden_dim, kernel_size=1)
        self.proj_r = nn.Conv2d(in_channels_x, hidden_dim, kernel_size=1)

        self.proj_p_back = nn.Conv2d(hidden_dim, in_channels_c, kernel_size=1)

        self.cross_attn = nn.MultiheadAttention(hidden_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)

        self.ffn_pose = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.norm1 = LayerNorm2D(hidden_dim)
        self.norm2 = LayerNorm2D(hidden_dim)

    def forward(self, pose, ref, mask=None):
        """
        pose : (B, C1, T, H, W)
        ref  : (B, C2, T, H, W)
        mask : (B, T*H*W) 可选 key_padding_mask
        return: (B, d_model, T, H, W)
        """
        B, _, T, H, W = pose.shape
        L = H * W

        p_trans = pose.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)
        r_trans = ref.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)

        p_trans = self.proj_p(p_trans)
        r_trans = self.proj_r(r_trans)

        p_trans = p_trans.flatten(2).transpose(1, 2)
        r_trans = r_trans.flatten(2).transpose(1, 2)

        out = self.cross_attn(query=r_trans,
                              key=p_trans,
                              value=p_trans,
                              key_padding_mask=mask)[0]

        out = out.transpose(1, 2).contiguous().view(B*T, -1, H, W)
        out = self.norm1(out)

        ffn_out = self.ffn_pose(out)
        out = out + ffn_out
        out = self.norm2(out)
        out = self.proj_p_back(out)
        out = out.view(B, T, -1, H, W).contiguous().transpose(1, 2)

        return out
