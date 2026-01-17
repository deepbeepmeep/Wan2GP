"""NLF (Neural Localizer Fields) model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ptu, ptu3d
from . import util as model_util


@dataclass(frozen=True)
class NLFModelConfig:
    proc_side: int
    stride_test: int
    centered_stride: bool
    backbone_link_dim: int
    depth: int
    box_size_m: float
    uncert_bias: float = 0.0
    uncert_bias2: float = 0.001
    fix_uncert_factor: bool = False
    mix_3d_inside_fov: float = 0.5
    weak_perspective: bool = False


class NLFModel(nn.Module):
    def __init__(
        self,
        backbone,
        weight_field,
        normalizer,
        backbone_channels=1280,
        *,
        config: Optional[NLFModelConfig] = None,
        n_joints: Optional[int] = None,
        n_left_joints: Optional[int] = None,
        n_center_joints: Optional[int] = None,
    ):
        super().__init__()
        if config is None:
            raise ValueError("config must be provided")
        self.backbone = backbone
        self.heatmap_head = LocalizerHead(
            weight_field, normalizer, in_channels=backbone_channels, config=config
        )
        self.input_resolution = int(config.proc_side)

        if n_joints is None or n_left_joints is None or n_center_joints is None:
            raise ValueError("n_joints, n_left_joints, n_center_joints must be provided")

        inv_permutation = torch.arange(int(n_joints), dtype=torch.int64)
        canonical_locs_init = torch.zeros((int(n_joints), 3), dtype=torch.float32)
        canonical_delta_mask = torch.ones((int(n_joints),), dtype=torch.float32)

        self.inv_permutation = nn.Buffer(inv_permutation.to(dtype=torch.int64), persistent=False)
        self.canonical_lefts = nn.Parameter(
            torch.zeros((int(n_left_joints), 3), dtype=torch.float32)
        )
        self.canonical_centers = nn.Parameter(
            torch.zeros((int(n_center_joints), 2), dtype=torch.float32)
        )
        self.canonical_locs_init = nn.Buffer(
            canonical_locs_init.to(dtype=torch.float32), persistent=False
        )
        self.canonical_delta_mask = nn.Buffer(
            canonical_delta_mask.to(dtype=torch.float32), persistent=False
        )

    @torch.jit.export
    def canonical_locs(self):
        canonical_rights = torch.cat(
            [-self.canonical_lefts[:, :1], self.canonical_lefts[:, 1:]], dim=1
        )
        canonical_centers = torch.cat(
            [torch.zeros_like(self.canonical_centers[:, :1]), self.canonical_centers], dim=1
        )
        permuted = torch.cat([self.canonical_lefts, canonical_rights, canonical_centers], dim=0)
        return (
            permuted.index_select(0, self.inv_permutation)
            * self.canonical_delta_mask[:, torch.newaxis]
            + self.canonical_locs_init
        )

    @torch.jit.export
    def get_features(self, image: torch.Tensor):
        f = self.backbone(image)
        return self.heatmap_head.layer(f)

    @torch.jit.export
    def predict_multi_same_weights(
        self,
        image: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        flip_canonicals_per_image: torch.Tensor,
    ):
        features_processed = self.get_features(image)
        coords2d, coords3d, uncertainties = self.heatmap_head.decode_features_multi_same_weights(
            features_processed, weights, flip_canonicals_per_image
        )

        with torch.amp.autocast('cuda', enabled=False):
            return self.heatmap_head.reconstruct_absolute(
                coords2d.float(), coords3d.float(), uncertainties.float(), intrinsic_matrix.float()
            )

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        return self.heatmap_head.get_weights_for_canonical_points(canonical_points)


class LocalizerHead(nn.Module):
    def __init__(self, weight_field, normalizer, in_channels=1280, *, config: NLFModelConfig):
        super().__init__()
        self.uncert_bias = float(config.uncert_bias)
        self.uncert_bias2 = float(config.uncert_bias2)
        self.depth = int(config.depth)
        self.weight_field = weight_field
        self.stride_test = int(config.stride_test)
        self.centered_stride = bool(config.centered_stride)
        self.box_size_m = float(config.box_size_m)
        self.proc_side = int(config.proc_side)
        self.backbone_link_dim = int(config.backbone_link_dim)
        self.fix_uncert_factor = bool(config.fix_uncert_factor)
        self.mix_3d_inside_fov = float(config.mix_3d_inside_fov)
        self.weak_perspective = bool(config.weak_perspective)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, self.backbone_link_dim, kernel_size=1, bias=False),
            normalizer(self.backbone_link_dim),
            nn.SiLU(),
        )

    @torch.jit.export
    def transpose_weights(self, weights: torch.Tensor, n_in_channels: int):
        n_out_channels = 2 + self.depth
        weights_resh = torch.unflatten(weights, -1, (n_in_channels + 1, n_out_channels))
        w_tensor = weights_resh[..., :-1, :]
        b_tensor = weights_resh[..., -1, :]
        w_tensor = w_tensor.permute(0, 2, 1)
        return w_tensor.contiguous(), b_tensor.contiguous()

    @torch.jit.export
    def apply_weights3d_same_canonicals_impl(
        self, features: torch.Tensor, w_tensor: torch.Tensor, b_tensor: torch.Tensor
    ):
        n_out_channels = 2 + self.depth
        w_tensor = torch.flatten(w_tensor, start_dim=0, end_dim=1).unsqueeze(-1).unsqueeze(-1)
        b_tensor = b_tensor.reshape(-1)

        logits = F.conv2d(features, w_tensor, bias=b_tensor).float()
        logits = torch.unflatten(logits, 1, (-1, n_out_channels))
        uncertainty_map = logits[:, :, 0]

        coords_metric_xy = ptu.soft_argmax(logits[:, :, 1], dim=[3, 2])
        heatmap25d = ptu.softmax(logits[:, :, 2:], dim=[4, 3, 2])
        heatmap2d = torch.sum(heatmap25d, dim=2)

        uncertainties = torch.einsum('nphw,nphw->np', uncertainty_map, heatmap2d.detach())
        uncertainties = F.softplus(uncertainties + self.uncert_bias) + self.uncert_bias2
        coords25d = ptu.decode_heatmap(heatmap25d, dim=[4, 3, 2])
        coords2d = coords25d[..., :2]
        coords3d = torch.cat([coords_metric_xy, coords25d[..., 2:]], dim=-1)
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        weights = self.weight_field(canonical_points)
        w_tensor, b_tensor = self.transpose_weights(weights.half(), self.backbone_link_dim)
        weights_fl = self.weight_field(
            canonical_points
            * torch.tensor([-1, 1, 1], dtype=torch.float32, device=canonical_points.device)
        )
        w_tensor_fl, b_tensor_fl = self.transpose_weights(weights_fl.half(), self.backbone_link_dim)
        return dict(
            w_tensor=w_tensor,
            b_tensor=b_tensor,
            w_tensor_flipped=w_tensor_fl,
            b_tensor_flipped=b_tensor_fl,
        )

    @torch.jit.export
    def decode_features_multi_same_weights(
        self,
        features: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        flip_canonicals_per_image: torch.Tensor,
    ):
        features_processed = features
        flip_canonicals_per_image_ind = flip_canonicals_per_image.to(torch.int32)

        nfl_features_processed, fl_features_processed = ptu.dynamic_partition(
            features_processed, flip_canonicals_per_image_ind, 2
        )
        partitioned_indices = ptu.dynamic_partition(
            torch.arange(features_processed.shape[0], device=flip_canonicals_per_image_ind.device),
            flip_canonicals_per_image_ind,
            2,
        )
        nfl_coords2d, nfl_coords3d, nfl_uncertainties = self.apply_weights3d_same_canonicals_impl(
            nfl_features_processed, weights['w_tensor'], weights['b_tensor']
        )
        fl_coords2d, fl_coords3d, fl_uncertainties = self.apply_weights3d_same_canonicals_impl(
            fl_features_processed, weights['w_tensor_flipped'], weights['b_tensor_flipped']
        )
        coords2d = ptu.dynamic_stitch(partitioned_indices, [nfl_coords2d, fl_coords2d])
        coords3d = ptu.dynamic_stitch(partitioned_indices, [nfl_coords3d, fl_coords3d])
        uncertainties = ptu.dynamic_stitch(
            partitioned_indices, [nfl_uncertainties, fl_uncertainties]
        )

        coords2d = model_util.heatmap_to_image(
            coords2d, self.proc_side, self.stride_test, self.centered_stride
        )
        coords3d = model_util.heatmap_to_metric(
            coords3d,
            self.proc_side,
            self.stride_test,
            self.centered_stride,
            self.box_size_m,
        )
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def reconstruct_absolute(
        self,
        coords2d: torch.Tensor,
        coords3d: torch.Tensor,
        uncertainties: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
    ):
        coords3d_abs = (
            ptu3d.reconstruct_absolute(
                coords2d,
                coords3d,
                intrinsic_matrix,
                proc_side=self.proc_side,
                stride=self.stride_test,
                centered_stride=self.centered_stride,
                weak_perspective=self.weak_perspective,
                mix_3d_inside_fov=0.5,
                point_validity_mask=uncertainties < 0.3,
                border_factor1=1.0,
                border_factor2=0.6,
                mix_based_on_3d=True,
            )
            * 1000
        )
        factor = 1 if self.fix_uncert_factor else 3
        return coords3d_abs, uncertainties * factor
