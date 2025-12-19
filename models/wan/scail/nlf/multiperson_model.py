"""Multiperson NLF model."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import smplfitter.pt
import torch
import torchvision.transforms.v2.functional

from . import ptu, ptu3d
from . import plausibility_check as plausib
from . import warping


class MultipersonNLF(torch.nn.Module):
    def __init__(
        self,
        crop_model,
        detector,
        skeleton_infos,
        pad_white_pixels=True,
        *,
        cano_all: Optional[Dict[str, torch.Tensor]] = None,
        body_models: Optional[torch.nn.ModuleDict] = None,
        fitters: Optional[torch.nn.ModuleDict] = None,
    ):
        super().__init__()

        self.crop_model = crop_model
        self.detector = detector
        device = next(self.crop_model.parameters()).device

        if body_models is None:
            raise ValueError("body_models must be provided for eager mode")
        self.body_models = body_models

        if fitters is None:
            raise ValueError("fitters must be provided for eager mode")
        if cano_all is None:
            raise ValueError("cano_all must be provided for eager mode")
        self.fitters = fitters

        self.cano_all = {
            k: v.to(device=device, dtype=torch.float32) for k, v in cano_all.items()
        }
        nothing = torch.zeros([], device=device, dtype=torch.float32)

        self.weights = {
            k: dict(
                w_tensor=nothing,
                b_tensor=nothing,
                w_tensor_flipped=nothing,
                b_tensor_flipped=nothing,
            )
            for k in self.body_models
        }
        self.per_skeleton_indices = {
            k: torch.as_tensor(v['indices'], dtype=torch.int32) for k, v in skeleton_infos.items()
        }
        self.per_skeleton_joint_names = {k: v['names'] for k, v in skeleton_infos.items()}
        self.per_skeleton_joint_edges = {
            k: torch.as_tensor(v['edges'], dtype=torch.int32) for k, v in skeleton_infos.items()
        }
        self.skeleton_joint_indices_table = {k: v['indices'] for k, v in skeleton_infos.items()}
        self.pad_white_pixels = pad_white_pixels

    def detect_parametric_batched(
        self,
        images: torch.Tensor,
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
        detector_threshold: float = 0.3,
        detector_nms_iou_threshold: float = 0.7,
        max_detections: int = 150,
        detector_flip_aug: bool = False,
        detector_both_flip_aug: bool = False,
        suppress_implausible_poses: bool = True,
        beta_regularizer: float = 10.0,
        beta_regularizer2: float = 0.0,
        model_name: str = 'smpl',
        extra_boxes: Optional[List[torch.Tensor]] = None,
    ):
        images = im_to_linear(images)

        boxes = self.detector(
            images=images,
            threshold=detector_threshold,
            nms_iou_threshold=detector_nms_iou_threshold,
            max_detections=max_detections,
            extrinsic_matrix=extrinsic_matrix,
            world_up_vector=world_up_vector,
            flip_aug=detector_flip_aug,
            bothflip_aug=detector_both_flip_aug,
            extra_boxes=extra_boxes,
        )
        return self._estimate_parametric_batched(
            images,
            boxes,
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses,
            beta_regularizer,
            beta_regularizer2,
            model_name,
        )

    detect_smpl_batched = detect_parametric_batched

    def _estimate_parametric_batched(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor],
        distortion_coeffs: Optional[torch.Tensor],
        extrinsic_matrix: Optional[torch.Tensor],
        world_up_vector: Optional[torch.Tensor],
        default_fov_degrees: float,
        internal_batch_size: int,
        antialias_factor: int,
        num_aug: int,
        rot_aug_max_degrees: float,
        suppress_implausible_poses: bool,
        beta_regularizer: float,
        beta_regularizer2: float,
        model_name: str,
    ):
        if model_name not in self.body_models:
            raise ValueError(
                f'Unknown model name {model_name}, use one of {self.body_models.keys()}'
            )

        if self.weights[model_name]['w_tensor'].ndim == 0:
            self.weights[model_name] = self.get_weights_for_canonical_points(
                self.cano_all[model_name]
            )

        result = self._estimate_poses_batched(
            images,
            boxes,
            self.weights[model_name],
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses=suppress_implausible_poses,
        )
        boxes = result['boxes']
        n_pose_per_image_list = [len(b) for b in boxes]
        if sum(n_pose_per_image_list) == 0:
            return self._predict_empty_parametric(images, model_name)

        poses3d_flat = torch.cat(result['poses3d'], dim=0)
        mean_poses = torch.mean(poses3d_flat, dim=-2, keepdim=True)
        poses3d_flat = poses3d_flat - mean_poses
        poses2d_flat = torch.cat(result['poses2d'], dim=0)
        uncertainties_flat = torch.cat(result['uncertainties'], dim=0)

        fitter = self.fitters['smpl'] if model_name == 'smpl' else self.fitters['smplx']
        vertices_flat, joints_flat = torch.split(
            poses3d_flat, [fitter.body_model.num_vertices, fitter.body_model.num_joints], dim=-2
        )
        vertex_uncertainties_flat, joint_uncertainties_flat = torch.split(
            uncertainties_flat,
            [fitter.body_model.num_vertices, fitter.body_model.num_joints],
            dim=-1,
        )

        vertex_weights = vertex_uncertainties_flat**-1.5
        vertex_weights = vertex_weights / torch.mean(vertex_weights, dim=-1, keepdim=True)
        joint_weights = joint_uncertainties_flat**-1.5
        joint_weights = joint_weights / torch.mean(joint_weights, dim=-1, keepdim=True)

        fit = fitter.fit(
            vertices_flat / 1000,
            joints_flat / 1000,
            vertex_weights=vertex_weights,
            joint_weights=joint_weights,
            num_iter=3,
            beta_regularizer=beta_regularizer,
            beta_regularizer2=beta_regularizer2,
            final_adjust_rots=True,
            requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
        )

        result['pose'] = torch.split(fit['pose_rotvecs'], n_pose_per_image_list)
        result['betas'] = torch.split(fit['shape_betas'], n_pose_per_image_list)
        result['trans'] = torch.split(
            fit['trans'] + mean_poses.squeeze(-2) / 1000, n_pose_per_image_list
        )

        body_model: smplfitter.pt.BodyModel = (
            self.body_models['smpl'] if model_name == 'smpl' else self.body_models['smplx']
        )
        fit_res = body_model.forward(
            fit['pose_rotvecs'], fit['shape_betas'], fit['trans'] + mean_poses.squeeze(-2) / 1000
        )
        fit_vertices_flat = fit_res['vertices'] * 1000
        fit_joints_flat = fit_res['joints'] * 1000
        result['vertices3d'] = torch.split(fit_vertices_flat, n_pose_per_image_list)
        result['joints3d'] = torch.split(fit_joints_flat, n_pose_per_image_list)

        result['vertices2d'] = project_ragged(
            images,
            [x for x in result['vertices3d']],
            extrinsic_matrix,
            intrinsic_matrix,
            distortion_coeffs,
            default_fov_degrees,
        )
        result['joints2d'] = project_ragged(
            images,
            [x for x in result['joints3d']],
            extrinsic_matrix,
            intrinsic_matrix,
            distortion_coeffs,
            default_fov_degrees,
        )

        result['vertices3d_nonparam'] = torch.split(
            vertices_flat + mean_poses, n_pose_per_image_list
        )
        result['joints3d_nonparam'] = torch.split(joints_flat + mean_poses, n_pose_per_image_list)
        vertices2d, joints2d = torch.split(
            poses2d_flat, [fitter.body_model.num_vertices, fitter.body_model.num_joints], dim=-2
        )

        result['vertices2d_nonparam'] = torch.split(vertices2d, n_pose_per_image_list)
        result['joints2d_nonparam'] = torch.split(joints2d, n_pose_per_image_list)

        result['vertex_uncertainties'] = torch.split(
            vertex_uncertainties_flat * 1000, n_pose_per_image_list
        )
        result['joint_uncertainties'] = torch.split(
            joint_uncertainties_flat * 1000, n_pose_per_image_list
        )

        del result['poses3d']
        del result['poses2d']
        del result['uncertainties']
        return result

    def _estimate_poses_batched(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor],
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor],
        distortion_coeffs: Optional[torch.Tensor],
        extrinsic_matrix: Optional[torch.Tensor],
        world_up_vector: Optional[torch.Tensor],
        default_fov_degrees: float,
        internal_batch_size: int,
        antialias_factor: int,
        num_aug: int,
        rot_aug_max_degrees: float,
        suppress_implausible_poses: bool,
    ):
        if sum(len(b) for b in boxes) == 0:
            return self._predict_empty(images, weights)

        images = im_to_linear(images)

        n_images = len(images)
        device = images.device

        if intrinsic_matrix is None:
            intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(
                default_fov_degrees, images.shape[2:4], device=device
            )

        if len(intrinsic_matrix) == 1:
            intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_images, dim=0)

        if distortion_coeffs is None:
            distortion_coeffs = torch.zeros((n_images, 5), device=device)
        if len(distortion_coeffs) == 1:
            distortion_coeffs = torch.repeat_interleave(distortion_coeffs, n_images, dim=0)

        if extrinsic_matrix is None:
            extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
        if len(extrinsic_matrix) == 1:
            extrinsic_matrix = torch.repeat_interleave(extrinsic_matrix, n_images, dim=0)

        n_box_per_image_list = [len(b) for b in boxes]
        n_box_per_image = torch.tensor([len(b) for b in boxes], device=device)

        intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_box_per_image, dim=0)
        distortion_coeffs = torch.repeat_interleave(distortion_coeffs, n_box_per_image, dim=0)

        if world_up_vector is None:
            world_up_vector = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)

        camspace_up = torch.einsum('c,bCc->bC', world_up_vector, extrinsic_matrix[..., :3, :3])
        camspace_up = torch.repeat_interleave(camspace_up, n_box_per_image, dim=0)

        aug_gammas = ptu.linspace(0.6, 1.0, num_aug, dtype=torch.float32, device=device)

        aug_angle_range = rot_aug_max_degrees * (torch.pi / 180.0)
        aug_angles = ptu.linspace(
            -aug_angle_range, aug_angle_range, num_aug, dtype=torch.float32, device=device
        )

        if num_aug == 1:
            aug_scales = torch.tensor([1.0], device=device, dtype=torch.float32)
        else:
            aug_scales = torch.cat(
                [
                    ptu.linspace(
                        0.8, 1.0, num_aug // 2, endpoint=False, dtype=torch.float32, device=device
                    ),
                    torch.linspace(
                        1.0, 1.1, num_aug - num_aug // 2, dtype=torch.float32, device=device
                    ),
                ],
                dim=0,
            )
        aug_should_flip = (torch.arange(0, num_aug, device=device) - num_aug // 2) % 2 != 0
        aug_flipmat = torch.tensor(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device
        )
        aug_maybe_flipmat = torch.where(
            aug_should_flip[:, torch.newaxis, torch.newaxis],
            aug_flipmat,
            torch.eye(3, device=device),
        )
        aug_rotmat = ptu3d.rotation_mat(-aug_angles, rot_axis='z')
        aug_rotflipmat = aug_maybe_flipmat @ aug_rotmat

        poses3d_flat, uncert_flat = self._predict_in_batches(
            images,
            weights,
            intrinsic_matrix,
            distortion_coeffs,
            camspace_up,
            boxes,
            internal_batch_size,
            aug_should_flip,
            aug_rotflipmat,
            aug_gammas,
            aug_scales,
            antialias_factor,
        )
        poses3d_flat = plausib.scale_align(poses3d_flat)
        mean = torch.mean(poses3d_flat, dim=(-3, -2), keepdim=True)
        poses3d_flat_submean = (poses3d_flat - mean).float()
        poses3d_flat_submean, final_weights = weighted_geometric_median(
            poses3d_flat_submean, uncert_flat**-1.5, dim=-3, n_iter=10, eps=50.0
        )
        poses3d_flat = poses3d_flat_submean.double() + mean.squeeze(1)

        uncert_flat = weighted_mean(uncert_flat, final_weights, dim=-2)

        poses2d_flat_normalized = ptu3d.to_homogeneous(
            warping.distort_points(ptu3d.project(poses3d_flat.float()), distortion_coeffs)
        )
        poses2d_flat = torch.einsum(
            'bnk,bjk->bnj', poses2d_flat_normalized, intrinsic_matrix[:, :2, :]
        )

        poses3d = torch.split(poses3d_flat, n_box_per_image_list)
        poses2d = torch.split(poses2d_flat, n_box_per_image_list)
        uncert = torch.split(uncert_flat, n_box_per_image_list)

        if suppress_implausible_poses:
            boxes, poses3d, poses2d, uncert = self._filter_poses(boxes, poses3d, poses2d, uncert)

        n_box_per_image_list = [len(b) for b in boxes]

        if sum(n_box_per_image_list) == 0:
            return self._predict_empty(images, weights)

        n_box_per_image = torch.tensor(n_box_per_image_list, device=device)
        inv_extrinsic_matrix = torch.repeat_interleave(
            torch.linalg.inv(extrinsic_matrix.double()), n_box_per_image, dim=0
        )
        poses3d_flat = torch.einsum(
            'bnk,bjk->bnj',
            ptu3d.to_homogeneous(torch.cat(poses3d)),
            inv_extrinsic_matrix[:, :3, :],
        )
        poses3d = torch.split(poses3d_flat.float(), n_box_per_image_list)

        result = dict(boxes=boxes, poses3d=poses3d, poses2d=poses2d, uncertainties=uncert)
        return result

    def _filter_poses(
        self,
        boxes: List[torch.Tensor],
        poses3d: List[torch.Tensor],
        poses2d: List[torch.Tensor],
        uncert: List[torch.Tensor],
    ):
        boxes_out = []
        poses3d_out = []
        poses2d_out = []
        uncert_out = []
        for boxes_, poses3d_, poses2d_, uncert_ in zip(boxes, poses3d, poses2d, uncert):
            is_uncert_low = plausib.is_uncertainty_low(uncert_)
            is_pose_consi = plausib.is_pose_consistent_with_box(poses2d_, boxes_)
            plausible_mask = torch.logical_and(is_uncert_low, is_pose_consi)
            nms_indices = plausib.pose_non_max_suppression(
                poses3d_, boxes_[..., 4] / torch.mean(uncert_, dim=-1), plausible_mask
            )
            boxes_out.append(boxes_[nms_indices])
            poses3d_out.append(poses3d_[nms_indices])
            poses2d_out.append(poses2d_[nms_indices])
            uncert_out.append(uncert_[nms_indices])
        return boxes_out, poses3d_out, poses2d_out, uncert_out

    def _predict_in_batches(
        self,
        images: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: List[torch.Tensor],
        internal_batch_size: int,
        aug_should_flip: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_gammas: torch.Tensor,
        aug_scales: torch.Tensor,
        antialias_factor: int,
    ):
        num_aug = len(aug_gammas)
        boxes_per_batch = internal_batch_size // num_aug
        boxes_flat = torch.cat(boxes, dim=0)
        image_id_per_box = torch.repeat_interleave(
            torch.arange(len(boxes)), torch.tensor([len(b) for b in boxes])
        )

        if boxes_per_batch == 0:
            return self._predict_single_batch(
                images,
                weights,
                intrinsic_matrix,
                distortion_coeffs,
                camspace_up,
                boxes_flat,
                image_id_per_box,
                aug_rotflipmat,
                aug_should_flip,
                aug_scales,
                aug_gammas,
                antialias_factor,
            )
        else:
            n_total_boxes = len(boxes_flat)
            n_batches = int(math.ceil(n_total_boxes / boxes_per_batch))
            poses3d_batches = []
            uncert_batches = []
            for i in range(n_batches):
                batch_slice = slice(i * boxes_per_batch, (i + 1) * boxes_per_batch)
                poses3d, uncert = self._predict_single_batch(
                    images,
                    weights,
                    intrinsic_matrix[batch_slice],
                    distortion_coeffs[batch_slice],
                    camspace_up[batch_slice],
                    boxes_flat[batch_slice],
                    image_id_per_box[batch_slice],
                    aug_rotflipmat,
                    aug_should_flip,
                    aug_scales,
                    aug_gammas,
                    antialias_factor,
                )
                poses3d_batches.append(poses3d)
                uncert_batches.append(uncert)
            return torch.cat(poses3d_batches, dim=0), torch.cat(uncert_batches, dim=0)

    def _predict_single_batch(
        self,
        images: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: torch.Tensor,
        image_ids: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_should_flip: torch.Tensor,
        aug_scales: torch.Tensor,
        aug_gammas: torch.Tensor,
        antialias_factor: int,
    ):
        crops, new_intrinsic_matrix, R = self._get_crops(
            images,
            intrinsic_matrix,
            distortion_coeffs,
            camspace_up,
            boxes,
            image_ids,
            aug_rotflipmat,
            aug_scales,
            aug_gammas,
            antialias_factor,
        )

        new_intrinsic_matrix_flat = torch.reshape(new_intrinsic_matrix, (-1, 3, 3))
        res = self.crop_model.input_resolution
        crops_flat = torch.reshape(crops, (-1, 3, res, res))

        n_cases = crops.shape[1]
        aug_should_flip_flat = torch.repeat_interleave(aug_should_flip, n_cases, dim=0)

        poses_flat, uncert_flat = self.crop_model.predict_multi_same_weights(
            crops_flat, new_intrinsic_matrix_flat, weights, aug_should_flip_flat
        )
        poses_flat = poses_flat.double()
        n_joints = poses_flat.shape[-2]

        poses = torch.reshape(poses_flat, [-1, n_cases, n_joints, 3])
        uncert = torch.reshape(uncert_flat, [-1, n_cases, n_joints])
        poses_orig_camspace = poses @ R.double()

        return poses_orig_camspace.transpose(0, 1), uncert.transpose(0, 1)

    def _get_crops(
        self,
        images: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: torch.Tensor,
        image_ids: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_scales: torch.Tensor,
        aug_gammas: torch.Tensor,
        antialias_factor: int,
    ):
        R_noaug, box_scales = self._get_new_rotation_and_scale(
            intrinsic_matrix, distortion_coeffs, camspace_up, boxes
        )

        device = images.device
        crop_scales = aug_scales[:, torch.newaxis] * box_scales[torch.newaxis, :]
        num_box = boxes.shape[0]
        num_aug = aug_gammas.shape[0]
        res = self.crop_model.input_resolution
        new_intrinsic_matrix = torch.cat(
            [
                torch.cat(
                    [
                        intrinsic_matrix[torch.newaxis, :, :2, :2]
                        * crop_scales[:, :, torch.newaxis, torch.newaxis],
                        torch.full(
                            (num_aug, num_box, 2, 1), res / 2, dtype=torch.float32, device=device
                        ),
                    ],
                    dim=3,
                ),
                torch.cat(
                    [
                        torch.zeros((num_aug, num_box, 1, 2), dtype=torch.float32, device=device),
                        torch.ones((num_aug, num_box, 1, 1), dtype=torch.float32, device=device),
                    ],
                    dim=3,
                ),
            ],
            dim=2,
        )
        R = aug_rotflipmat[:, torch.newaxis] @ R_noaug
        new_invprojmat = torch.linalg.inv(new_intrinsic_matrix @ R)

        if antialias_factor > 1:
            scaling_mat = warping.corner_aligned_scale_mat(1 / antialias_factor)
            new_invprojmat = new_invprojmat @ scaling_mat.to(new_invprojmat.device)

        if self.pad_white_pixels:
            images.neg_().add_(1)

        crops = warping.warp_images_with_pyramid(
            images,
            intrinsic_matrix=torch.tile(intrinsic_matrix, [num_aug, 1, 1]),
            new_invprojmats=torch.reshape(new_invprojmat, [-1, 3, 3]),
            distortion_coeffs=torch.tile(distortion_coeffs, [num_aug, 1]),
            crop_scales=torch.reshape(crop_scales, [-1]) * antialias_factor,
            output_shape=(res * antialias_factor, res * antialias_factor),
            image_ids=torch.tile(image_ids, [num_aug]),
        )
        if self.pad_white_pixels:
            crops.neg_().add_(1).clamp_(0, 1)

        if antialias_factor == 2:
            crops = torch.nn.functional.avg_pool2d(crops, 2, 2)
        elif antialias_factor == 4:
            crops = torch.nn.functional.avg_pool2d(crops, 4, 4)
        elif antialias_factor > 4:
            crops = torchvision.transforms.v2.functional.resize(
                crops,
                (res, res),
                torchvision.transforms.v2.functional.InterpolationMode.BILINEAR,
                antialias=True,
            )
        crops = torch.reshape(crops, [num_aug, num_box, 3, res, res])
        crops **= torch.reshape(aug_gammas.to(crops.dtype) / 2.2, [-1, 1, 1, 1, 1])
        return crops, new_intrinsic_matrix, R

    def _get_new_rotation_and_scale(self, intrinsic_matrix, distortion_coeffs, camspace_up, boxes):
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxpoints_homog = ptu3d.to_homogeneous(
            torch.stack(
                [
                    torch.stack([x + w / 2, y + h / 2], dim=1),
                    torch.stack([x + w / 2, y], dim=1),
                    torch.stack([x + w, y + h / 2], dim=1),
                    torch.stack([x + w / 2, y + h], dim=1),
                    torch.stack([x, y + h / 2], dim=1),
                ],
                dim=1,
            )
        )

        boxpoints_camspace = torch.einsum(
            'bpc,bCc->bpC', boxpoints_homog, torch.linalg.inv(intrinsic_matrix)
        )
        boxpoints_camspace = ptu3d.to_homogeneous(
            warping.undistort_points(boxpoints_camspace[:, :, :2], distortion_coeffs)
        )
        box_center_camspace = boxpoints_camspace[:, 0]
        R_noaug = ptu3d.lookat_matrix(forward_vector=box_center_camspace, up_vector=camspace_up)

        sidepoints_camspace = boxpoints_camspace[:, 1:5]
        sidepoints_new = ptu3d.project(
            torch.einsum('bpc,bCc->bpC', sidepoints_camspace, intrinsic_matrix @ R_noaug)
        )

        vertical_size = torch.linalg.norm(sidepoints_new[:, 0] - sidepoints_new[:, 2], dim=-1)
        horiz_size = torch.linalg.norm(sidepoints_new[:, 1] - sidepoints_new[:, 3], dim=-1)
        box_size_new = torch.maximum(vertical_size, horiz_size)

        box_scales = (
            torch.tensor(self.crop_model.input_resolution, dtype=box_size_new.dtype) / box_size_new
        )
        return R_noaug, box_scales

    def _predict_empty(self, image: torch.Tensor, weights: Dict[str, torch.Tensor]):
        device = image.device
        n_joints = weights['w_tensor'].shape[0]
        poses3d = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        poses2d = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        uncert = torch.zeros((0, n_joints), dtype=torch.float32, device=device)
        boxes = torch.zeros((0, 5), dtype=torch.float32, device=device)
        n_images = image.shape[0]

        result = dict(
            boxes=[boxes] * n_images,
            poses3d=[poses3d] * n_images,
            poses2d=[poses2d] * n_images,
            uncertainties=[uncert] * n_images,
        )
        return result

    def _predict_empty_parametric(self, image: torch.Tensor, model_name: str):
        device = image.device
        fitter = self.fitters['smpl'] if model_name == 'smpl' else self.fitters['smplx']
        n_joints = fitter.body_model.num_joints
        n_verts = fitter.body_model.num_vertices
        pose = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        betas = torch.zeros((0, fitter.body_model.num_betas), dtype=torch.float32, device=device)
        trans = torch.zeros((0, 3), dtype=torch.float32, device=device)
        vertices3d = torch.zeros((0, n_verts, 3), dtype=torch.float32, device=device)
        joints3d = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        vertices2d = torch.zeros((0, n_verts, 2), dtype=torch.float32, device=device)
        joints2d = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        vertices3d_nonparam = torch.zeros((0, n_verts, 3), dtype=torch.float32, device=device)
        joints3d_nonparam = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        vertices2d_nonparam = torch.zeros((0, n_verts, 2), dtype=torch.float32, device=device)
        joints2d_nonparam = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        vertex_uncertainties = torch.zeros((0, n_verts), dtype=torch.float32, device=device)
        joint_uncertainties = torch.zeros((0, n_joints), dtype=torch.float32, device=device)
        n_images = image.shape[0]
        result = dict(
            pose=[pose] * n_images,
            betas=[betas] * n_images,
            trans=[trans] * n_images,
            vertices3d=[vertices3d] * n_images,
            joints3d=[joints3d] * n_images,
            vertices2d=[vertices2d] * n_images,
            joints2d=[joints2d] * n_images,
            vertices3d_nonparam=[vertices3d_nonparam] * n_images,
            joints3d_nonparam=[joints3d_nonparam] * n_images,
            vertices2d_nonparam=[vertices2d_nonparam] * n_images,
            joints2d_nonparam=[joints2d_nonparam] * n_images,
            vertex_uncertainties=[vertex_uncertainties] * n_images,
            joint_uncertainties=[joint_uncertainties] * n_images,
        )
        return result

    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        return self.crop_model.get_weights_for_canonical_points(canonical_points)


def im_to_linear(im: torch.Tensor):
    if im.dtype == torch.uint8:
        return im.to(dtype=torch.float16).mul_(1.0 / 255.0).pow_(2.2)
    elif im.dtype == torch.uint16:
        return im.to(dtype=torch.float16).mul_(1.0 / 65504.0).nan_to_num_(posinf=1.0).pow_(2.2)
    elif im.dtype == torch.float16:
        return im**2.2
    else:
        return im.to(dtype=torch.float16).pow_(2.2)


def project_ragged(
    images: torch.Tensor,
    poses3d: List[torch.Tensor],
    extrinsic_matrix: Optional[torch.Tensor],
    intrinsic_matrix: Optional[torch.Tensor],
    distortion_coeffs: Optional[torch.Tensor],
    default_fov_degrees: float = 55.0,
):
    device = images.device
    n_box_per_image_list = [len(b) for b in poses3d]
    n_box_per_image = torch.tensor([len(b) for b in poses3d], device=device)

    n_images = images.shape[0]

    if intrinsic_matrix is None:
        intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(
            default_fov_degrees, images.shape[2:4], device=device
        )

    if intrinsic_matrix.shape[0] == 1:
        intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_images, dim=0)

    if distortion_coeffs is None:
        distortion_coeffs = torch.zeros((n_images, 5), device=device)

    if distortion_coeffs.shape[0] == 1:
        distortion_coeffs = torch.repeat_interleave(distortion_coeffs, n_images, dim=0)

    if extrinsic_matrix is None:
        extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
    if extrinsic_matrix.shape[0] == 1:
        extrinsic_matrix = torch.repeat_interleave(extrinsic_matrix, n_images, dim=0)

    intrinsic_matrix_rep = torch.repeat_interleave(intrinsic_matrix, n_box_per_image, dim=0)
    distortion_coeffs_rep = torch.repeat_interleave(distortion_coeffs, n_box_per_image, dim=0)
    extrinsic_matrix_rep = torch.repeat_interleave(extrinsic_matrix, n_box_per_image, dim=0)

    poses3d_flat = torch.cat(poses3d, dim=0)
    poses3d_flat = torch.einsum(
        'bnk,bjk->bnj', ptu3d.to_homogeneous(poses3d_flat), extrinsic_matrix_rep[:, :3, :]
    )

    poses2d_flat_normalized = ptu3d.to_homogeneous(
        warping.distort_points(ptu3d.project(poses3d_flat), distortion_coeffs_rep)
    )
    poses2d_flat = torch.einsum(
        'bnk,bjk->bnj', poses2d_flat_normalized, intrinsic_matrix_rep[:, :2, :]
    )
    poses2d = torch.split(poses2d_flat, n_box_per_image_list)
    return poses2d


def weighted_geometric_median(
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    n_iter: int = 10,
    dim: int = -2,
    eps: float = 1e-1,
    keepdim: bool = False,
):
    if dim < 0:
        dim = len(x.shape) + dim

    if w is None:
        w = torch.ones_like(x[..., :1])
    else:
        w = w.unsqueeze(-1)

    new_weights = w
    y = weighted_mean(x, new_weights, dim=dim, keepdim=True)
    for _ in range(n_iter):
        dist = torch.norm(x - y, dim=-1, keepdim=True)
        new_weights = w / (dist + eps)
        y = weighted_mean(x, new_weights, dim=dim, keepdim=True)

    if not keepdim:
        y = y.squeeze(dim)

    return y, new_weights.squeeze(-1)


def weighted_mean(x: torch.Tensor, w: torch.Tensor, dim: int = -2, keepdim: bool = False):
    return (x * w).sum(dim=dim, keepdim=keepdim) / w.sum(dim=dim, keepdim=keepdim)
