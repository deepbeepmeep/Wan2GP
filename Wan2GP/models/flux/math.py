import torch
from einops import rearrange
from torch import Tensor
from shared.attention import pay_attention


def attention(qkv_list, pe: Tensor, *, txt_len: int | None = None, NAG: dict | None = None) -> Tensor:
    q, k, v = qkv_list
    qkv_list.clear()
    q_list = [q]
    q = None
    q = apply_rope_(q_list, pe)
    k_list = [k]
    k = None
    k = apply_rope_(k_list, pe)

    if NAG is not None and txt_len is not None:
        cap_len = int(NAG.get("cap_embed_len", 0) or 0)
        prefix_len = int(NAG.get("prefix_len", 0) or 0)
        total_len = q.shape[2]
        img_start = txt_len
        packed_len = txt_len - prefix_len
        if cap_len > 0 and packed_len == (cap_len * 2) and img_start <= total_len:
            pos_start = prefix_len
            pos_end = pos_start + cap_len
            neg_start = pos_end
            neg_end = neg_start + cap_len
            if neg_end <= txt_len:
                # Build pos/neg sequences that share prefix + image tokens.
                q_neg = torch.cat( (q[:, :, :prefix_len], q[:, :, neg_start:neg_end], q[:, :, img_start:]), dim=2, )
                k_neg = torch.cat( (k[:, :, :prefix_len], k[:, :, neg_start:neg_end], k[:, :, img_start:]), dim=2, )
                v_neg = torch.cat( (v[:, :, :prefix_len], v[:, :, neg_start:neg_end], v[:, :, img_start:]), dim=2, )

                q_pos = torch.cat((q[:, :, :pos_end], q[:, :, img_start:]), dim=2)
                k_pos = torch.cat((k[:, :, :pos_end], k[:, :, img_start:]), dim=2)
                v_pos = torch.cat((v[:, :, :pos_end], v[:, :, img_start:]), dim=2)
                del q, k, v

                qkv_pos = [q_pos.transpose(1, 2), k_pos.transpose(1, 2), v_pos.transpose(1, 2)]
                q_pos = k_pos = v_pos = None
                x_pos = pay_attention(qkv_pos)
                x_pos = x_pos.flatten(2, 3)

                qkv_neg = [q_neg.transpose(1, 2), k_neg.transpose(1, 2), v_neg.transpose(1, 2)]
                q_neg = k_neg = v_neg = None
                x_neg = pay_attention(qkv_neg)
                x_neg = x_neg.flatten(2, 3)

                neg_slice_end = prefix_len + cap_len
                neg_out = x_neg[:, prefix_len:neg_slice_end].clone()
                nag_scale = NAG["scale"]
                nag_alpha = NAG["alpha"]
                nag_tau = NAG["tau"]
                dtype = x_pos.dtype

                x_guidance = x_neg
                x_guidance.mul_(1 - nag_scale)
                x_guidance.add_(x_pos, alpha=nag_scale)
                norm_positive = torch.norm(x_pos, p=1, dim=-1, keepdim=True)
                norm_guidance = torch.norm(x_guidance, p=1, dim=-1, keepdim=True)
                scale = norm_guidance / norm_positive
                torch.nan_to_num(scale, nan=10.0, posinf=10.0, neginf=10.0, out=scale)
                factor = (1 / (norm_guidance + 1e-7) * norm_positive * nag_tau).to(x_guidance.dtype)
                x_guidance = torch.where(scale > nag_tau, x_guidance * factor, x_guidance).to(dtype)
                del norm_positive, norm_guidance, scale, factor

                x_guidance.mul_(nag_alpha)
                x_guidance.add_(x_pos, alpha=(1 - nag_alpha))
                x_pos = None

                prefix_pos_guidance = x_guidance[:, :pos_end]
                img_guidance = x_guidance[:, pos_end:]
                x_guidance = None

                out = torch.cat([prefix_pos_guidance, neg_out, img_guidance], dim=1)
                prefix_pos_guidance = neg_out = img_guidance = None
                return out

    qkv_list = [q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)]
    del q, k, v
    x = pay_attention(qkv_list).transpose(1, 2)
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
