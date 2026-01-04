from typing import List, Iterable
import torch
import torch.nn.functional as F


# STM
def pad_divide_by(in_img: torch.Tensor, d: int) -> (torch.Tensor, Iterable[int]):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array


def unpad(img: torch.Tensor, pad: Iterable[int]) -> torch.Tensor:
    if len(img.shape) == 4:
        if pad[2] + pad[3] > 0:
            img = img[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2] + pad[3] > 0:
            img = img[:, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, pad[0]:-pad[1]]
    elif len(img.shape) == 5:
        if pad[2] + pad[3] > 0:
            img = img[:, :, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, :, pad[0]:-pad[1]]
    else:
        raise NotImplementedError
    return img


# @torch.jit.script
def aggregate(prob: torch.Tensor, dim: int) -> torch.Tensor:
    with torch.amp.autocast("cuda"):
        prob = prob.float()
        new_prob = torch.cat([torch.prod(1 - prob, dim=dim, keepdim=True), prob],
                             dim).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))  # (0, 1) --> (-inf, inf)

        return logits


# @torch.jit.script
def cls_to_one_hot(cls_gt: torch.Tensor, num_objects: int) -> torch.Tensor:
    # cls_gt: B*1*H*W
    B, _, H, W = cls_gt.shape
    one_hot = torch.zeros(B, num_objects + 1, H, W, device=cls_gt.device).scatter_(1, cls_gt, 1)
    return one_hot