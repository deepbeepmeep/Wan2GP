"""Model utilities for NLF."""
import torch


def heatmap_to_image(coords: torch.Tensor, proc_side: int, stride: int, centered_stride: bool):
    last_image_pixel = proc_side - 1
    last_receptive_center = last_image_pixel - (last_image_pixel % stride)
    coords_out = coords * last_receptive_center

    if centered_stride:
        coords_out = coords_out + stride // 2

    return coords_out


def heatmap_to_metric(
    coords: torch.Tensor, proc_side: int, stride: int, centered_stride: bool, box_size_m: float
):
    xy = coords[..., :2]
    coords2d = heatmap_to_image(xy, proc_side, stride, centered_stride) * box_size_m / proc_side
    return torch.cat([coords2d, coords[..., 2:] * box_size_m], dim=-1)
