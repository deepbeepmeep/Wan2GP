import torch
from einops import rearrange
from torch import Tensor
from shared.attention import pay_attention


def attention(qkv_list, pe: Tensor) -> Tensor:
    q, k, v = qkv_list
    qkv_list.clear()
    q_list = [q] 
    q = None
    q = apply_rope_(q_list, pe)
    k_list = [k] 
    k = None
    k = apply_rope_(k_list, pe)
    qkv_list = [q.transpose(1,2), k.transpose(1,2) ,v.transpose(1,2)]
    del q,k, v
    x = pay_attention(qkv_list).transpose(1,2)
    # x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope_(q_list, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq= q_list[0]
    xqshape = xq.shape
    xqdtype= xq.dtype
    q_list.clear()
    xq = xq.float().reshape(*xqshape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq[..., 0]
    xq = freqs_cis[..., 1] * xq[..., 1]

    xq_out.add_(xq)
    # xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]

    return xq_out.reshape(*xqshape).to(xqdtype)

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
