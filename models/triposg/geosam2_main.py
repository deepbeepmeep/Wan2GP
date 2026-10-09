# GeoSAM2 for WanGP: prompt-based or automatic 3D part segmentation of a mesh, ported from VAST-AI-Research/GeoSAM2 inference.py,
# single_view_point_prompt_infer.py, geosam2_render.py, utils/inference_utils.py, sam2/sam2_video_predictor_geosam2.py,
# sam2/sam2_image_predictor_geosam2.py, sam2/automatic_mask_generator_geosam2.py, sam2/utils/{amg.py, misc.py, transforms.py}
# (Copyright (c) Meta Platforms, Inc. and affiliates; modifications copyright 2025 VAST-AI-Research and the GeoSAM2 authors; Apache
# License 2.0, see LICENSE_GeoSAM2 and NOTICE_GeoSAM2 in this folder).
# Modified by DeepBeepMeep for WanGP: the 12 views rendered with the shared rasterizer instead of Blender (same cameras, normalization,
# world normals over white and position maps); the view features computed once per mesh and shared by the automatic mask generator
# and the propagation (upstream recomputes the same features); masks kept as tensors instead of RLE / numpy round trips; the mask
# connected-component post-processing of the optional CUDA extension done with OpenCV; the C++ label mode and the Python face loops
# vectorized; the mesh placed in the rendered frame for lifting as it is rendered (upstream translates before scaling, which only
# matches for meshes already normalized); faces adjacent through shared positions (upstream's unprocessed trimesh only links faces
# sharing vertex indices, which cuts parts along UV seams); a multi-part GLB (one node per segment) instead of a recoloured mesh.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from mmgp import offload
from torch import nn

from shared.mesh3d import Mesh, load_mesh, meshes_to_glb
from shared.mesh3d.mesh import face_normals, submesh, weld
from shared.mesh3d.raster import interpolate, rasterize
from shared.mesh3d.render import look_at, perspective, preview
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status
from .geosam2_model import NO_OBJ_SCORE, GeoSAM2

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "geosam2.json")
AZIMUTHS = (180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
ELEVATIONS = (0.0, 25.0, 0.0, -25.0) * 3
VIEWS, RESOLUTION, LENS, SENSOR = 12, 1024, 50.0, 36.0
SAMPLES_PER_FACE = 5  # upstream SAMPLE_NUM: points sampled on each face to lift the view labels
IMAGE_MEAN, IMAGE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PREVIEW_SIZE = 512
DEFAULT_SETTINGS = {"postprocess_pa": 0.02, "auto_segment": 1}
# GeoSAM2 colours segments randomly; these fixed distinct colours keep the parts recognizable from one generation to the next
PART_COLORS = [(82, 170, 220), (215, 91, 78), (45, 136, 117), (247, 172, 83), (124, 121, 121), (127, 171, 209), (243, 152, 101), (145, 204, 192),
               (150, 59, 121), (181, 206, 78), (189, 119, 149), (199, 193, 222), (200, 151, 54), (236, 110, 102), (238, 182, 212), (96, 96, 168)]


def _keep_checkpoint_dtypes(model):
    """FP32 checkpoint kept in FP32 (upstream runs it under BF16 autocast, which casts the weights per operation)."""
    model._convertWeightsFloatTo = None
    for module in [model, *model.children()]:
        if not isinstance(module, nn.ModuleList) and not hasattr(module, "_lock_dtype"):
            module._lock_dtype = None
    return model.eval().requires_grad_(False)


def part_color(index, device):
    if index < len(PART_COLORS):
        return torch.tensor(PART_COLORS[index], dtype=torch.float32, device=device) / 255
    hue = (index * 0.618034) % 1.0  # golden-ratio hues beyond the palette
    rgb = cv2.cvtColor(np.uint8([[[int(hue * 180), 160, 220]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return torch.tensor(rgb, dtype=torch.float32, device=device) / 255


# ------------------------------------------------------------------------------------------------ views (geosam2_render.py)

def scene_frame(vertices):
    """Blender's glTF import (+Z up: x, -z, y) then normalize_scene(1.0): largest box side 1, box centred on the origin. Returns the
    vertices and the box size."""
    vertices = torch.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], 1).double()
    vertices = vertices / (vertices.amax(0) - vertices.amin(0)).max()
    low, high = vertices.amin(0), vertices.amax(0)
    return (vertices - (low + high) / 2).float(), high - low


def view_cameras(box_size, device):
    """The 12 cameras of get_solid_points_on_sphere at 50/36 times the box diagonal, looking at the origin with +Z up (Blender
    track -Z / up Y). Returns world-to-camera matrices [12, 4, 4] (OpenGL) and the horizontal field of view in degrees."""
    distance = LENS / SENSOR * float(box_size.norm())
    eyes = torch.tensor([[distance * math.cos(math.radians(e)) * math.cos(math.radians(a)), distance * math.cos(math.radians(e)) * math.sin(math.radians(a)),
                          distance * math.sin(math.radians(e))] for a, e in zip(AZIMUTHS, ELEVATIONS)], dtype=torch.float32, device=device)
    return look_at(eyes, up=(0.0, 0.0, 1.0)), math.degrees(2 * math.atan(SENSOR / 2 / LENS))


def render_views(mesh, views, fov):
    """Coverage [12, H, W] bool, planar depth [12, H, W] (1e10 where empty), position maps [12, 3, H, W] (clipped to [-1, 1], 0 where
    empty) and world normal maps [12, H, W, 3] uint8 ((n + 1) / 2 over white) of the mesh in the scene frame. Normals: the mesh's
    own (Blender's import keeps them), else flat (Blender imports meshes without normals flat shaded)."""
    device = mesh.vertices.device
    if mesh.normals is None:
        vertices, faces = mesh.vertices[mesh.faces].reshape(-1, 3), torch.arange(3 * len(mesh.faces), device=device).view(-1, 3)
        normals = F.normalize(face_normals(mesh.vertices, mesh.faces), dim=1).repeat_interleave(3, 0)
    else:
        vertices, faces, normals = mesh.vertices, mesh.faces, mesh.normals
    projection = perspective(fov, 1.0, 0.1, 100.0, device=device)
    homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], 1)
    camera = homogeneous @ views.transpose(1, 2)
    clip = camera @ projection.T
    clip[..., 1].neg_()
    masks, depths, positions, normal_maps = [], [], [], []
    for view in range(len(views)):
        face_id, bary, _ = rasterize(clip[view], faces, RESOLUTION)
        covered = face_id[0] >= 0
        attributes = interpolate(torch.cat([vertices, -camera[view, :, 2:3], normals], 1), faces, face_id, bary)[0]
        masks.append(covered)
        depths.append(torch.where(covered, attributes[..., 3], 1e10))
        positions.append((attributes[..., :3].clamp(-1, 1) * covered[..., None]).permute(2, 0, 1))
        normal = (F.normalize(attributes[..., 4:7], dim=-1) + 1) / 2
        normal_maps.append(torch.where(covered[..., None], (normal * 255).round(), 255.0).to(torch.uint8))
        check_abort()
    return torch.stack(masks), torch.stack(depths), torch.stack(positions), torch.stack(normal_maps)


def normalized_image(normal_map):
    """SAM2 image input: [0, 1] then ImageNet normalization, (3, H, W)."""
    image = normal_map.permute(2, 0, 1).float() / 255
    mean, std = (torch.tensor(v, device=image.device).view(-1, 1, 1) for v in (IMAGE_MEAN, IMAGE_STD))
    return (image - mean) / std


# ------------------------------------------------------------------------------------------------ mask utilities

def connected_component_areas(masks):
    """Area of the 8-connected component of every pixel of binary masks (N, H, W) (0 outside), as the optional CUDA kernel of
    sam2/csrc gives it."""
    out = torch.zeros(masks.shape, dtype=torch.int32)
    for i, mask in enumerate(masks.to(torch.uint8).cpu().numpy()):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA].copy()
        areas[0] = 0
        out[i] = torch.from_numpy(np.where(labels > 0, areas[labels], 0).astype(np.int32))
    return out.to(masks.device)


def postprocess_masks(masks, max_area, threshold=0.0):
    """SAM2Transforms.postprocess_masks on low-resolution logits (B, C, h, w): holes and sprinkles up to max_area pixels removed (before
    the upsampling to the image size)."""
    masks = masks.float()
    flat = masks.flatten(0, 1)
    holes = (connected_component_areas(flat <= threshold) <= max_area) & (flat <= threshold)
    masks = torch.where(holes.reshape_as(masks), threshold + 10.0, masks)
    flat = masks.flatten(0, 1)
    sprinkles = (connected_component_areas(flat > threshold) <= max_area) & (flat > threshold)
    return torch.where(sprinkles.reshape_as(masks), threshold - 10.0, masks)


def fill_holes(mask, max_area):
    """fill_holes_in_mask_scores: background holes up to max_area pixels set to 0.1."""
    holes = (connected_component_areas((mask <= 0).flatten(0, 1)) <= max_area) & (mask <= 0).flatten(0, 1)
    return torch.where(holes.reshape_as(mask), 0.1, mask)


def stability_score(logits, threshold=0.0, offset=0.7):
    intersections = (logits > threshold + offset).sum((-1, -2), dtype=torch.int32)
    unions = (logits > threshold - offset).sum((-1, -2), dtype=torch.int32)
    return intersections / unions


def mask_boxes(masks):
    """batched_mask_to_box (x0, y0, x1, y1), zeros for empty masks."""
    h, w = masks.shape[-2:]
    rows, cols = masks.any(-1), masks.any(-2)
    ys, xs = torch.arange(h, device=masks.device), torch.arange(w, device=masks.device)
    bottom, right = (rows * ys).amax(-1), (cols * xs).amax(-1)
    top, left = (rows * ys + h * ~rows).amin(-1), (cols * xs + w * ~cols).amin(-1)
    boxes = torch.stack([left, top, right, bottom], -1)
    return boxes * ~((right < left) | (bottom < top))[..., None]


def iou(a, b):
    return (a & b).sum().item() / ((a | b).sum().item() + 1e-6)


def mode_ignoring_zero(values):
    """Upstream mode_except_negative_one (C++): the most frequent value of each row, the most frequent non-zero one when there is
    one. values (N, K) int."""
    out = torch.empty(len(values), dtype=values.dtype, device=values.device)
    for start in range(0, len(values), 1 << 18):
        rows = values[start:start + (1 << 18)]
        counts = (rows[:, :, None] == rows[:, None, :]).sum(-1)
        counts = torch.where(rows != 0, counts + rows.shape[1], counts)  # any non-zero value outranks 0
        out[start:start + len(rows)] = rows.gather(1, counts.argmax(1, keepdim=True))[:, 0]
    return out


# ------------------------------------------------------------------------------------------------ propagation (SAM2VideoPredictor)

class Tracker:
    """inference_state and the methods of upstream SAM2VideoPredictor used by GeoSAM2 (forward tracking, fill_hole_area 8,
    binarized memories of prompted frames, propagate_in_video_v2's rotated order). frames: the view index of each frame."""

    def __init__(self, model, features, frames):
        self.model, self.features, self.frames = model, features, frames
        self.num_frames = len(frames) + 1
        self.constants = {}
        self.reset()

    def reset(self):
        self.obj_ids, self.point_inputs, self.mask_inputs, self.outputs, self.temp_outputs, self.tracked = [], [], [], [], [], []

    def _obj_idx(self, obj_id):
        if obj_id not in self.obj_ids:
            self.obj_ids.append(obj_id)
            for container in (self.point_inputs, self.mask_inputs, self.tracked):
                container.append({})
            self.outputs.append({"cond_frame_outputs": {}, "non_cond_frame_outputs": {}})
            self.temp_outputs.append({"cond_frame_outputs": {}, "non_cond_frame_outputs": {}})
        return self.obj_ids.index(obj_id)

    def _frame_features(self, frame_idx_real):
        """_get_image_feature + _prepare_backbone_features, batch 1: (HW, 1, C) features and position encodings, and sizes."""
        fpn, pos = self.features[self.frames[frame_idx_real]]
        return [x.flatten(2).permute(2, 0, 1) for x in fpn], [x.flatten(2).permute(2, 0, 1) for x in pos], [(x.shape[-2], x.shape[-1]) for x in pos]

    def _maskmem_pos_enc(self, maskmem_pos_enc):
        if "maskmem_pos_enc" not in self.constants:
            self.constants["maskmem_pos_enc"] = [x[0:1].clone() for x in maskmem_pos_enc]
        return self.constants["maskmem_pos_enc"]

    def _run_single_frame_inference(self, output_dict, frame_idx, is_init_cond_frame, point_inputs, mask_inputs, run_mem_encoder, frame_idx_real,
                                    prev_sam_mask_logits=None):
        feats, pos, sizes = self._frame_features(frame_idx_real)
        out = self.model.track_step(frame_idx, is_init_cond_frame, feats, pos, sizes, point_inputs, mask_inputs, output_dict, self.num_frames, run_mem_encoder,
                                    prev_sam_mask_logits)
        maskmem_features = out["maskmem_features"]
        pred_masks = fill_holes(out["pred_masks"], 8)
        compact = {"maskmem_features": None if maskmem_features is None else maskmem_features.to(torch.bfloat16),
                   "maskmem_pos_enc": None if out["maskmem_pos_enc"] is None else self._maskmem_pos_enc(out["maskmem_pos_enc"]),
                   "pred_masks": pred_masks, "obj_ptr": out["obj_ptr"], "object_score_logits": out["object_score_logits"], "ious": out["ious"]}
        return compact, pred_masks

    def add_points(self, frame_idx, obj_id, points, labels):
        """add_new_points_or_box (new points, pixel coordinates at the model resolution)."""
        obj_idx = self._obj_idx(obj_id)
        point_inputs = {"point_coords": points[None].float(), "point_labels": labels[None].to(torch.int32)}
        self.point_inputs[obj_idx][frame_idx] = point_inputs
        self.mask_inputs[obj_idx].pop(frame_idx, None)
        prev = self.temp_outputs[obj_idx]["cond_frame_outputs"].get(frame_idx) or self.outputs[obj_idx]["cond_frame_outputs"].get(frame_idx) \
            or self.outputs[obj_idx]["non_cond_frame_outputs"].get(frame_idx)
        prev_logits = None if prev is None else torch.clamp(prev["pred_masks"], -32.0, 32.0)
        out, _ = self._run_single_frame_inference(self.outputs[obj_idx], frame_idx, frame_idx not in self.tracked[obj_idx], point_inputs, None, False,
                                                  frame_idx, prev_logits)
        self.temp_outputs[obj_idx]["cond_frame_outputs"][frame_idx] = out

    def add_mask(self, frame_idx, obj_id, mask):
        """add_new_mask: a binary mask at the model resolution."""
        obj_idx = self._obj_idx(obj_id)
        mask_inputs = mask[None, None].float()
        self.mask_inputs[obj_idx][frame_idx] = mask_inputs
        self.point_inputs[obj_idx].pop(frame_idx, None)
        out, _ = self._run_single_frame_inference(self.outputs[obj_idx], frame_idx, frame_idx not in self.tracked[obj_idx], None, mask_inputs, False, frame_idx)
        self.temp_outputs[obj_idx]["cond_frame_outputs"][frame_idx] = out

    def _preflight(self):
        for obj_idx in range(len(self.obj_ids)):
            for storage_key in ("non_cond_frame_outputs", "cond_frame_outputs"):
                for frame_idx, out in self.temp_outputs[obj_idx][storage_key].items():
                    if out["maskmem_features"] is None:
                        high_res_masks = F.interpolate(out["pred_masks"], size=(self.model.image_size,) * 2, mode="bilinear", align_corners=False)
                        feats, _, sizes = self._frame_features(frame_idx)
                        maskmem_features, maskmem_pos_enc = self.model.encode_new_memory(feats, sizes, high_res_masks, out["object_score_logits"], True)
                        out["maskmem_features"], out["maskmem_pos_enc"] = maskmem_features.to(torch.bfloat16), self._maskmem_pos_enc(maskmem_pos_enc)
                    self.outputs[obj_idx][storage_key][frame_idx] = out
                self.temp_outputs[obj_idx][storage_key].clear()
            for frame_idx in self.outputs[obj_idx]["cond_frame_outputs"]:
                self.outputs[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

    def propagate(self, start_frame_idx, progress=None):
        """propagate_in_video_v2: the seed frame, then the views in circular order from the seed (the seed view tracked again first),
        on rotated frame indices. Yields (view frame, low-resolution logits (objects, 1, 256, 256)), which upstream upsamples bilinearly
        to the view resolution."""
        self._preflight()
        end_frame_idx = min(start_frame_idx + self.num_frames, self.num_frames - 1)
        order = [start_frame_idx] + list(range(start_frame_idx, end_frame_idx)) + list(range(0, start_frame_idx))
        for position, frame_idx_real in enumerate(order):
            frame_idx = (position + start_frame_idx) % 13
            pred_masks_per_obj = []
            for obj_idx in range(len(self.obj_ids)):
                output_dict = self.outputs[obj_idx]
                if frame_idx in output_dict["cond_frame_outputs"]:
                    pred_masks = output_dict["cond_frame_outputs"][frame_idx]["pred_masks"]
                else:
                    out, pred_masks = self._run_single_frame_inference(output_dict, frame_idx, False, None, None, True, frame_idx_real)
                    output_dict["non_cond_frame_outputs"][frame_idx] = out
                self.tracked[obj_idx][frame_idx] = True
                pred_masks_per_obj.append(pred_masks)
                check_abort()
            if progress is not None:
                progress.advance()
            yield frame_idx_real, torch.cat(pred_masks_per_obj, dim=0)


# ------------------------------------------------------------------------------------------------ automatic masks (SAM2AutomaticMaskGenerator)

def _predict(model, features, points, mask_input=None, multimask_output=True):
    """SAM2ImagePredictor._predict on one view (no-memory embedding added, dynamic multimask off as in the image model): low-res
    logits (B, M, 256, 256) and IoU predictions."""
    fpn, _ = features
    labels = torch.ones(points.shape[0], 1, dtype=torch.int32, device=points.device)
    sparse, dense = model.sam_prompt_encoder((points[:, None, :], labels), mask_input)
    image_embed = fpn[-1] + model.no_mem_embed.view(1, -1, 1, 1)
    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(image_embed, model.sam_prompt_encoder.get_dense_pe(), sparse, dense, multimask_output,
                                                                  points.shape[0] > 1, [fpn[0], fpn[1]], False)
    return low_res_masks, iou_predictions


def automatic_masks(model, features, region, rng, points_per_batch=128, iou_threshold=0.7, stability_threshold=0.7, stability_offset=0.7, nms_threshold=0.7,
                    min_region_area=25):
    """GeoSAM2's mask generator: up to 4096 points drawn in region (H, W bool) prompt the SAM decoder (3 masks each), every mask is
    refined once with its own logits (use_m2m), filtered by predicted IoU and stability, deduplicated by box NMS; then show_anns keeps
    the masks overlapping the better ones by at most 85% IoU, sorted by area. Returns [(point (x, y), mask (H, W) bool)]."""
    true_indices = np.argwhere(region.cpu().numpy())
    if len(true_indices) == 0:
        return []
    chosen = true_indices[rng.choice(true_indices.shape[0], min(4096, true_indices.shape[0]), replace=False)]
    all_points = torch.from_numpy(np.float64(chosen)[:, [1, 0]]).float().to(region.device)
    masks, ious, points, boxes = [], [], [], []
    for start in range(0, len(all_points), points_per_batch):
        batch = all_points[start:start + points_per_batch]
        low_res, _ = _predict(model, features, batch)
        low_res = torch.clamp(low_res, -32.0, 32.0).flatten(0, 1)
        batch = batch.repeat_interleave(3, dim=0)
        for sub in range(0, len(batch), points_per_batch):  # refine_with_m2m
            refined, refined_iou = _predict(model, features, batch[sub:sub + points_per_batch], low_res[sub:sub + points_per_batch, None], False)
            logits = F.interpolate(postprocess_masks(refined, min_region_area), (RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)[:, 0]
            keep = refined_iou[:, 0] > iou_threshold
            keep &= stability_score(logits, 0.0, stability_offset) >= stability_threshold
            binary = logits[keep] > 0
            del logits
            masks.append(np.packbits(binary.cpu().numpy(), axis=-1))  # 1 bit per pixel: thousands of candidates before the NMS
            ious.append(refined_iou[keep, 0].float())
            points.append(batch[sub:sub + points_per_batch][keep])
            boxes.append(mask_boxes(binary))
            check_abort()
    from torchvision.ops.boxes import batched_nms

    masks, ious, points, boxes = np.concatenate(masks), torch.cat(ious), torch.cat(points), torch.cat(boxes)
    keep = batched_nms(boxes.float(), ious, torch.zeros_like(boxes[:, 0]), iou_threshold=nms_threshold).cpu()
    candidates = sorted(((ious[i].item(), points[i].tolist(), torch.from_numpy(np.unpackbits(masks[i], axis=-1)).bool()) for i in keep.tolist()),
                        key=lambda x: x[0], reverse=True)
    accepted = []
    for _, point, mask in candidates:
        if all(iou(mask, other) * 100 <= 85 for _, other in accepted):
            accepted.append((point, mask))
    return sorted(accepted, key=lambda item: int(item[1].sum()), reverse=True)


# ------------------------------------------------------------------------------------------------ 2D mask filters (utils/inference_utils.py)

def shrink_masks(logits):
    """shrink_mask: pixels removed by an opening of 13x13 (3 erosions then 3 dilations of 5x5) get the logit 1024. logits (V, H, W)."""
    vanilla = (logits > 0).float()[:, None]
    opened = vanilla
    for _ in range(3):
        opened = -F.max_pool2d(-opened, 5, 1, 2)
    for _ in range(3):
        opened = F.max_pool2d(opened, 5, 1, 2)
    return torch.where((vanilla > 0.5)[:, 0] & ~(opened > 0.5)[:, 0], 1024.0, logits)


def upsample(low_res):
    """Low-resolution logits (..., 256, 256) to the view resolution (bilinear), as _get_orig_video_res_output."""
    shape = low_res.shape
    return F.interpolate(low_res.reshape(-1, 1, *shape[-2:]), size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False).reshape(*shape[:-2], RESOLUTION, RESOLUTION)


def filter_segments(segments, track_id, stability_threshold, area_alpha):
    """shrink_mask, filter_mask_stability, filter_mask_area then trans2bool on {obj_id: low-resolution logits (V, h, w)}: an object is
    dropped when its stability falls under the threshold (0.07 lower outside the seed view) on a view where it is visible, or when its
    area on a view exceeds area_alpha times its seed view area. Returns {obj_id: masks (V, H, W) bool}."""
    kept = {}
    for obj_id, low_res in segments.items():
        logits = shrink_masks(upsample(low_res))
        visible = (logits > 0).flatten(1).any(1)
        thresholds = torch.full((len(logits),), stability_threshold - 0.07, device=logits.device)
        thresholds[track_id] = stability_threshold
        if (visible & (stability_score(logits) < thresholds)).any():
            continue
        areas = (logits > 0).flatten(1).sum(1).float()
        others = torch.ones_like(visible)
        others[track_id] = False
        if (areas[others] / max(areas[track_id].item(), 1e-6) > area_alpha).any():
            continue
        kept[obj_id] = logits > 0
    return kept


def filter_overlaps(all_segments, segments, track_id):
    """filter_iou: within the pass, objects overlapping an accepted one by more than 0.8 IoU on the seed view are dropped, then those
    overlapping an object of the previous passes."""
    accepted = {}
    for obj_id, masks in segments.items():
        if all(iou(masks[track_id], other[track_id]) <= 0.8 for other in accepted.values()):
            accepted[obj_id] = masks
    return {obj_id: masks for obj_id, masks in accepted.items() if not any(0.8 < iou(masks[track_id], other[track_id]) <= 1 for other in all_segments.values())}


# ------------------------------------------------------------------------------------------------ lifting and labels

def sample_face_points(face_vertices, count, rng):
    """sample_points_on_faces_parallel: count uniform barycentric samples per face (F, 3, 3) -> (F * count, 3)."""
    u, v = rng.rand(len(face_vertices), count, 1), rng.rand(len(face_vertices), count, 1)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    w = 1 - u - v
    points = u * face_vertices[:, None, 0] + v * face_vertices[:, None, 1] + w * face_vertices[:, None, 2]
    return torch.from_numpy(points).float().reshape(-1, 3)


def lift_labels(segments, prior_keys, coverage, depths, views, fov, points, samples_per_face):
    """lift_2dmask_3d: the masks painted on each view (largest objects first, prompted objects last, 999 for covered pixels without an
    object), every sampled point visible in a view (depth within 1e-3, covered) reads the label under it, each point takes its most
    frequent non-zero label over the views and each face over its points."""
    device = coverage.device
    labels = torch.full((VIEWS, 1, RESOLUTION, RESOLUTION), 999.0, device=device)
    prompted = torch.zeros_like(labels)
    for obj_id in sorted(segments, key=lambda key: int(segments[key].sum()), reverse=True):
        target = prompted if obj_id in prior_keys else labels
        target[segments[obj_id][:, None]] = float(obj_id)
    labels = torch.where(prompted != 0, prompted, labels)
    tan_half = math.tan(math.radians(fov) / 2)
    projection = torch.zeros(4, 4, device=device)
    projection[0, 0], projection[1, 1] = 1 / tan_half, -1 / tan_half
    projection[2, 2], projection[2, 3], projection[3, 2] = -(10.0 + 0.1) / (10.0 - 0.1), -2 * 10.0 * 0.1 / (10.0 - 0.1), -1
    point_labels = []
    for start in range(0, len(points), 1 << 20):
        chunk = points[start:start + (1 << 20)].to(device)
        homogeneous = torch.cat([chunk, torch.ones_like(chunk[:, :1])], 1)
        camera = homogeneous @ views.transpose(1, 2)
        clip = camera @ projection.T
        ndc = clip[..., :2] / clip[..., 3:4]
        grid = ndc[:, None]
        covered = F.grid_sample(coverage[:, None].float(), grid, align_corners=True, mode="nearest")[:, 0, 0]
        depth = F.grid_sample(depths[:, None], grid, align_corners=True, mode="nearest")[:, 0, 0]
        valid = ((depth - -camera[..., 2]).abs() < 1e-3) & (covered > 0.9)
        pixels = ((ndc + 1) / 2 * RESOLUTION).round().clamp(0, RESOLUTION - 1).int()
        sampled = F.grid_sample(labels, (pixels.float() / (RESOLUTION - 1) * 2 - 1)[:, None], mode="nearest", align_corners=False)[:, 0, 0]  # upstream's default
        point_labels.append(mode_ignoring_zero(torch.where(valid, sampled, 0).T.int()))
        check_abort()
    return mode_ignoring_zero(torch.cat(point_labels).reshape(-1, samples_per_face))


def face_adjacency(faces):
    """Pairs of faces sharing an edge (as trimesh's face_adjacency: consecutive pairs on edges of more than two faces), both directions."""
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).sort(1).values
    owners = torch.arange(len(faces), device=faces.device).repeat(3)
    _, inverse = torch.unique(edges, dim=0, return_inverse=True)
    order = inverse.argsort(stable=True)
    sorted_edges, sorted_owners = inverse[order], owners[order]
    same = sorted_edges[1:] == sorted_edges[:-1]
    pairs = torch.stack([sorted_owners[:-1][same], sorted_owners[1:][same]], 1)
    return torch.cat([pairs, pairs.flip(1)])


def label_components(labels, pairs):
    """Connected faces sharing the same non-zero label: [(faces tensor)] in the order of their first face."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    same = (labels[pairs[:, 0]] == labels[pairs[:, 1]]) & (labels[pairs[:, 0]] != 0)
    pairs = pairs[same].cpu().numpy()
    count = len(labels)
    graph = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(count, count))
    _, component = connected_components(graph, directed=False)
    component = torch.from_numpy(component).to(labels.device)
    labelled = (labels != 0).nonzero()[:, 0]
    ids, groups = torch.unique(component[labelled], return_inverse=True)
    first_face = torch.full((len(ids),), count, dtype=torch.long, device=labels.device).scatter_reduce(0, groups, labelled, "amin")
    members = labelled[groups.argsort(stable=True)].split(torch.bincount(groups, minlength=len(ids)).tolist())
    return [members[i] for i in first_face.argsort().tolist()]


def complete_labels(labels, mesh, pa):
    """complete_labels(smooth_type="adjacent"): components smaller than pa of the largest (in faces and in area) are cleared, unlabeled
    faces take the most common label of their labeled neighbours (64 passes), then of the 3 nearest labeled face centres, and every
    connected region gets its own label. Returns the face labels."""
    labels = labels.clone()
    pairs = face_adjacency(weld(mesh.vertices)[1][mesh.faces])  # through shared positions: UV seams and split primitives do not cut parts
    components = sorted(label_components(labels, pairs), key=len, reverse=True)
    if not components:
        raise ValueError("GeoSAM2 could not label any face of this mesh from its 12 views.")
    areas = face_normals(mesh.vertices.double(), mesh.faces).norm(dim=1) / 2
    sizes = torch.tensor([len(c) for c in components], dtype=torch.float64)
    comp_areas = torch.stack([areas[c].sum() for c in components]).cpu()
    for i in ((sizes < sizes.max() * pa) & (comp_areas < comp_areas.max() * pa)).nonzero()[:, 0].tolist():
        labels[components[i]] = 0
    for _ in range(64):
        source, target = pairs[:, 1], pairs[:, 0]
        usable = (labels[target] == 0) & (labels[source] != 0)
        if not usable.any():
            break
        candidates = torch.stack([target[usable], labels[source[usable]].long()], 1)
        keys, counts = torch.unique(candidates, dim=0, return_counts=True)
        order = torch.argsort(counts.double() * -(1 << 20) + keys[:, 1].double(), stable=True)  # most votes, then lowest label
        keys = keys[order]
        first = torch.ones(len(keys), dtype=torch.bool, device=keys.device)
        faces_sorted, index = keys[:, 0].sort(stable=True)
        first[index[1:][faces_sorted[1:] == faces_sorted[:-1]]] = False
        labels[keys[first, 0]] = keys[first, 1].to(labels.dtype)
    unlabeled, labeled = (labels == 0).nonzero()[:, 0], (labels != 0).nonzero()[:, 0]
    if len(unlabeled) and len(labeled):
        centers = mesh.vertices[mesh.faces].float().mean(1)
        nearest = torch.cat([torch.cdist(centers[unlabeled[s:s + 4096]], centers[labeled]).topk(min(3, len(labeled)), largest=False).indices
                             for s in range(0, len(unlabeled), 4096)])
        labels[unlabeled] = torch.mode(labels[labeled][nearest], dim=1).values
    seen, next_label = set(), int(labels.max().item()) + 1
    for component in components:
        label = int(labels[component.min()].item())
        if label == 0 or label in seen:
            labels[component] = next_label
            next_label += 1
        seen.add(label)
    return labels


# ------------------------------------------------------------------------------------------------ prompts

def parse_point_prompts(text):
    """Upstream point prompt file: a JSON list of {"frame_idx": view, "obj_id": part, "point": [x, y], "label": 1 | 0} with x, y in
    pixels of the 1024 x 1024 rendered view. Returns (view, {obj_id: (points (N, 2), labels (N,))}) or None for an empty text."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        prompts = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The Point Prompts are not valid JSON: {error}") from error
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("The Point Prompts must be a JSON list of {\"frame_idx\", \"obj_id\", \"point\": [x, y], \"label\"} objects.")
    views = {int(item["frame_idx"]) for item in prompts}
    if len(views) != 1 or not 0 <= next(iter(views)) < VIEWS:
        raise ValueError("All the Point Prompts must use the same view (frame_idx 0 to 11).")
    grouped = {}
    for item in prompts:
        point, label = item["point"], int(item.get("label", 1))
        if len(point) != 2 or label not in (0, 1):
            raise ValueError(f"Invalid point prompt {item}: point must be [x, y] and label 0 or 1.")
        grouped.setdefault(int(item["obj_id"]), ([], []))
        grouped[int(item["obj_id"])][0].append([float(point[0]), float(point[1])])
        grouped[int(item["obj_id"])][1].append(label)
    return next(iter(views)), {obj_id: (torch.tensor(p), torch.tensor(l)) for obj_id, (p, l) in grouped.items()}


# ------------------------------------------------------------------------------------------------ model

class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        self._interrupt = False
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.config = json.load(reader)
        filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        with init_empty_weights():
            model = GeoSAM2(**self.config["model"])
        offload.load_model_data(model, filename, writable_tensors=False, default_dtype=None)
        self.transformer = _keep_checkpoint_dtypes(model)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, filename, torch.float32, None)

    def _view_features(self, pos_maps, normal_maps):
        features = []
        with PhaseProgress(VIEWS, "Encoding Views", "views") as progress:
            for view in range(VIEWS):
                fpn, pos = self.transformer.forward_image(pos_maps[view][None], normalized_image(normal_maps[view])[None])
                features.append((fpn, pos))
                progress.advance()
        return features

    def _prompt_masks(self, features, view, prompts):
        """single_view_point_prompt_infer: the points of each part tracked on their view alone (two passes of propagate_in_video_v2),
        painted in part order (later parts over earlier ones), parts under 64 pixels dropped. Returns {obj_id: mask (H, W) bool}."""
        tracker = Tracker(self.transformer, features, [view])
        for obj_id, (points, labels) in sorted(prompts.items()):
            tracker.add_points(0, obj_id, points.to(self.device), labels.to(self.device))
        logits = None
        for _, logits in tracker.propagate(0):
            pass
        label_map = torch.zeros((RESOLUTION, RESOLUTION), dtype=torch.int64, device=self.device)
        for obj_id, mask in zip(tracker.obj_ids, upsample(logits)[:, 0] > 0):
            label_map[mask] = obj_id
        return {int(obj_id): label_map == obj_id for obj_id in torch.unique(label_map).tolist() if obj_id != 0 and (label_map == obj_id).sum() >= 64}

    def _segment(self, features, coverage, prompt_masks, start, auto_segment, rng):
        """segment_with_mask_prompts: the start view gets the prompt masks (or automatic masks), the opposite view automatic masks of
        what is left; each pass is propagated to all the views and filtered. Returns ({obj_id: masks (V, H, W) bool}, prompted ids)."""
        all_segments, prior_keys, next_obj_id = {}, set(), 1
        seeds = [start, (start + 6) % VIEWS]
        with PhaseProgress(len(seeds) * (VIEWS + 1), "Propagating Masks", "views") as progress:
            for seed in seeds:
                tracker = Tracker(self.transformer, features, list(range(VIEWS)))
                prompted = seed == start and bool(prompt_masks)
                if prompted:
                    for obj_id, mask in prompt_masks.items():
                        tracker.add_mask(seed, obj_id, mask)
                        prior_keys.add(obj_id)
                        next_obj_id = max(next_obj_id, obj_id + 1)
                elif auto_segment:
                    set_phase_status("Generating Automatic Masks")
                    region = coverage[seed].clone()
                    for masks in all_segments.values():
                        region &= ~masks[seed]
                    for point, _ in automatic_masks(self.transformer, features[seed], region, rng):
                        tracker.add_points(seed, next_obj_id, torch.tensor([point], device=self.device), torch.ones(1, device=self.device))
                        next_obj_id += 1
                if not tracker.obj_ids:
                    progress.advance(VIEWS + 1)
                    continue
                low_res = torch.empty(len(tracker.obj_ids), VIEWS, *self.config["low_res_size"], device=self.device)
                for view, masks in tracker.propagate(seed, progress):
                    low_res[:, view] = masks[:, 0]
                segments = filter_segments({obj_id: low_res[i] for i, obj_id in enumerate(tracker.obj_ids)}, seed, 0.0 if prompted else 0.7, 10000 if prompted else 1)
                del low_res, tracker
                all_segments.update(filter_overlaps(all_segments, segments, seed))
                check_abort()
        return all_segments, prior_keys

    @generation_progress
    @torch.inference_mode()
    def generate(self, seed, custom_settings, input_custom, alt_prompt=None, callback=None, set_progress_status=None, **kwargs):
        settings = DEFAULT_SETTINGS | custom_settings
        prompts = parse_point_prompts(alt_prompt)
        rng = np.random.RandomState(seed % (1 << 32))
        set_phase_status("Loading Mesh")
        source = load_mesh(input_custom, device=self.device)
        vertices, box_size = scene_frame(source.vertices.float())
        scene = Mesh(vertices, source.faces, normals=None if source.normals is None else scene_frame_normals(source.normals))
        views, fov = view_cameras(box_size, self.device)
        set_phase_status("Rendering Views")
        coverage, depths, pos_maps, normal_maps = render_views(scene, views, fov)
        points = sample_face_points(vertices[source.faces].double().cpu().numpy(), SAMPLES_PER_FACE, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = self._view_features(pos_maps, normal_maps)
            start, prompt_masks = 0, {}
            if prompts is not None:
                set_phase_status("Segmenting Prompted View")
                start, obj_prompts = prompts
                prompt_masks = self._prompt_masks(features, start, obj_prompts)
            segments, prior_keys = self._segment(features, coverage, prompt_masks, start, bool(int(settings["auto_segment"])), rng)
        del features
        if not segments:
            raise ValueError("GeoSAM2 found no part on this mesh: check that the mesh is visible (not empty or degenerate), or give Point Prompts.")
        set_phase_status("Lifting Masks to Faces")
        labels = lift_labels(segments, prior_keys, coverage, depths, views, fov, points, SAMPLES_PER_FACE)
        del segments, coverage, depths, pos_maps
        set_phase_status("Completing Labels")
        labels = complete_labels(labels, Mesh(source.vertices.float(), source.faces), float(settings["postprocess_pa"]))
        set_phase_status("Exporting Mesh")
        ids, counts = torch.unique(labels, return_counts=True)
        parts = [submesh(source, labels == label) for label in ids[counts.argsort(descending=True)].tolist()]
        textured = source.base_color is not None and source.uvs is not None
        colors = [part_color(i, self.device) for i in range(len(parts))]
        if not textured:
            parts = [part.replace(colors=color.repeat(len(part.vertices), 1)) for part, color in zip(parts, colors)]
        glb = meshes_to_glb([part.to("cpu") for part in parts])
        offsets = np.cumsum([0] + [len(part.vertices) for part in parts[:-1]])
        merged = Mesh(torch.cat([part.vertices for part in parts]), torch.cat([part.faces + int(offset) for part, offset in zip(parts, offsets)]),
                      colors=torch.cat([color.repeat(len(part.vertices), 1) for part, color in zip(parts, colors)]))
        print(f"[GeoSAM2] {len(parts)} parts")
        return {"x": preview(merged, PREVIEW_SIZE).cpu()[:, None], "meshes": [glb], "side_files": {"_views.png": views_sheet(normal_maps)}}


def views_sheet(normal_maps, tile=512):
    """The 12 rendered normal views in a 4 x 3 sheet of tile-pixel views, each labelled with its frame_idx, for placing Point Prompts
    (view pixels are 1024 / tile times the sheet ones)."""
    rows = []
    for row in range(3):
        tiles = []
        for view in range(row * 4, row * 4 + 4):
            image = cv2.resize(normal_maps[view].cpu().numpy(), (tile, tile), interpolation=cv2.INTER_AREA)
            cv2.putText(image, f"frame_idx {view}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(cv2.rectangle(image, (0, 0), (tile - 1, tile - 1), (160, 160, 160), 1))
        rows.append(np.concatenate(tiles, 1))
    return cv2.imencode(".png", cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))[1].tobytes()


def scene_frame_normals(normals):
    """Normals in Blender's import frame (x, -z, y); the uniform scale and translation of normalize_scene leave them unchanged."""
    return torch.stack([normals[:, 0], -normals[:, 2], normals[:, 1]], 1).float()
