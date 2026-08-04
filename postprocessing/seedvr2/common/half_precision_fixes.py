import torch.nn.functional as F


def safe_pad_operation(x, padding, mode="constant", value=0.0):
    return F.pad(x, padding, mode=mode, value=value)


def safe_interpolate_operation(x, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None):
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
