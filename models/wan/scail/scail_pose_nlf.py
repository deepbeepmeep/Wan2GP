"""
NLFPose extraction + rendering helpers from the official SCAIL-Pose implementation.

Upstream: https://github.com/zai-org/SCAIL-Pose
- NLFPoseExtract/extract_nlfpose_batch.py (process_video_nlf)
- NLFPoseExtract/nlf_draw.py (intrinsic_matrix_from_field_of_view, process_data_to_COCO_format)
- NLFPoseExtract/nlf_render.py (render_nlf_as_images and helpers)
- NLFPoseExtract/process_pose.py (recollect_* and scale_faces)

"""

from __future__ import annotations

import copy
import math
import random
from typing import List, Optional

import numpy as np
import torch

from .scail_pose_draw import draw_pose_to_canvas_np, scale_image_hw_keep_size
from .scail_pose_taichi_cylinder import render_whole


def process_data_to_COCO_format(joints: np.ndarray) -> np.ndarray:
    """Map 24-joint NLF pose to (18, dim) COCO-like format used by SCAIL-Pose."""
    if joints.ndim != 2:
        raise ValueError(f"Expected shape (24,2) or (24,3), got {joints.shape}")

    dim = joints.shape[1]
    mapping = {
        15: 0,  # head
        12: 1,  # neck
        17: 2,  # left shoulder
        16: 5,  # right shoulder
        19: 3,  # left elbow
        18: 6,  # right elbow
        21: 4,  # left hand
        20: 7,  # right hand
        2: 8,  # left pelvis
        1: 11,  # right pelvis
        5: 9,  # left knee
        4: 12,  # right knee
        8: 10,  # left feet
        7: 13,  # right feet
    }

    new_joints = np.zeros((18, dim), dtype=joints.dtype)
    for src, dst in mapping.items():
        new_joints[dst] = joints[src]
    return new_joints


def intrinsic_matrix_from_field_of_view(imshape, fov_degrees: float = 55):
    imshape = np.array(imshape)
    fov_radians = fov_degrees * np.array(np.pi / 180)
    larger_side = np.max(imshape)
    focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
    return np.array(
        [
            [focal_length, 0, imshape[1] / 2],
            [0, focal_length, imshape[0] / 2],
            [0, 0, 1],
        ]
    )


def process_video_nlf(model, vr_frames: torch.Tensor, bboxes):
    pose_meta_list = []
    vr_frames = vr_frames.cuda()
    height, width = vr_frames.shape[1], vr_frames.shape[2]
    result_list = []

    batch_size = 64
    buffer = torch.zeros((batch_size, height, width, 3), dtype=vr_frames.dtype, device="cuda")
    buffer_count = 0
    with torch.inference_mode():
        for frame, bbox_list in zip(vr_frames, bboxes):
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                x1_px = max(0, math.floor(x1 * width - width * 0.025))
                y1_px = max(0, math.floor(y1 * height - height * 0.05))
                x2_px = min(width, math.ceil(x2 * width + width * 0.025))
                y2_px = min(height, math.ceil(y2 * height + height * 0.05))

                cropped_region = frame[y1_px:y2_px, x1_px:x2_px, :]
                buffer[buffer_count, y1_px:y2_px, x1_px:x2_px, :] = cropped_region
                buffer_count += 1

                if buffer_count == batch_size:
                    frame_batch = buffer.permute(0, 3, 1, 2)
                    pred = model.detect_smpl_batched(frame_batch)
                    if "joints3d_nonparam" in pred:
                        result_list.extend(pred["joints3d_nonparam"])
                    else:
                        result_list.extend([None] * buffer_count)

                    buffer.zero_()
                    buffer_count = 0

        if buffer_count > 0:
            frame_batch = buffer[:buffer_count].permute(0, 3, 1, 2)
            pred = model.detect_smpl_batched(frame_batch)
            if "joints3d_nonparam" in pred:
                result_list.extend(pred["joints3d_nonparam"])
            else:
                result_list.extend([None] * buffer_count)

    index = 0
    for bbox_list in bboxes:
        n = len(bbox_list)
        pose_meta_list.append(
            {
                "video_height": height,
                "video_width": width,
                "bboxes": bbox_list,
                "nlfpose": result_list[index : index + n],
            }
        )
        index += n

    del buffer
    torch.cuda.empty_cache()
    return pose_meta_list


def process_video_multi_nlf(model, vr_frames_list: List[torch.Tensor]):
    """Run NLF on pre-segmented per-person videos and keep results separate per person.

    This mirrors the official SCAIL-Pose multi-human pipeline: segment each character,
    run NLF independently per character-video, then render them together.

    Args:
        vr_frames_list: list of per-person tensors shaped (T,H,W,3) uint8.

    Returns:
        list of dicts (length T) with keys: video_height, video_width, bboxes=None,
        nlfpose=[per_person_prediction, ...] where each prediction is a tensor of
        shape (n_detections, 24, 3) (or empty) for that person.
    """
    if not vr_frames_list:
        return []

    n_people = len(vr_frames_list)
    lengths = [int(v.shape[0]) for v in vr_frames_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"All person videos must have the same length, got {lengths}")

    vr_frames_list = [v.cuda() for v in vr_frames_list]
    vr_frames_first = vr_frames_list[0]
    height, width = int(vr_frames_first.shape[1]), int(vr_frames_first.shape[2])

    pose_meta_list = []
    result_list = []

    batch_size = 64
    buffer = torch.zeros((batch_size, height, width, 3), dtype=vr_frames_first.dtype, device="cuda")
    buffer_count = 0

    with torch.inference_mode():
        for frame_idx in range(lengths[0]):
            for person_idx in range(n_people):
                buffer[buffer_count] = vr_frames_list[person_idx][frame_idx]
                buffer_count += 1

                if buffer_count == batch_size:
                    frame_batch = buffer.permute(0, 3, 1, 2)
                    pred = model.detect_smpl_batched(frame_batch)
                    if "joints3d_nonparam" in pred:
                        result_list.extend(pred["joints3d_nonparam"])
                    else:
                        result_list.extend([None] * buffer_count)

                    buffer.zero_()
                    buffer_count = 0

        if buffer_count > 0:
            frame_batch = buffer[:buffer_count].permute(0, 3, 1, 2)
            pred = model.detect_smpl_batched(frame_batch)
            if "joints3d_nonparam" in pred:
                result_list.extend(pred["joints3d_nonparam"])
            else:
                result_list.extend([None] * buffer_count)

    index = 0
    for _ in range(lengths[0]):
        pose_meta_list.append(
            {"video_height": height, "video_width": width, "bboxes": None, "nlfpose": result_list[index : index + n_people]}
        )
        index += n_people

    del buffer
    torch.cuda.empty_cache()
    return pose_meta_list


def recollect_nlf(data):
    new_data = []
    for item in data:
        new_item = item.copy()
        if len(item["bboxes"]) > 0:
            new_item["bboxes"] = item["bboxes"][:1]
            new_item["nlfpose"] = item["nlfpose"][:1]
        new_data.append(new_item)
    return new_data


def recollect_dwposes(poses):
    new_poses = []
    for pose in poses:
        for i in range(1):
            bodies = pose["bodies"]
            faces = pose["faces"][i : i + 1]
            hands = pose["hands"][2 * i : 2 * i + 2]
            candidate = bodies["candidate"][i : i + 1]
            subset = bodies["subset"][i : i + 1]
            new_pose = {"bodies": {"candidate": candidate, "subset": subset}, "faces": faces, "hands": hands}
        new_poses.append(new_pose)
    return new_poses


def scale_faces(poses, pose_2d_ref):
    ref = pose_2d_ref[0]
    pose_0 = poses[0]

    face_0 = pose_0["faces"]
    face_ref = ref["faces"]

    face_0 = np.array(face_0[0])
    face_ref = np.array(face_ref[0])

    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose["faces"]
        face = np.array(face[0])
        center = face[center_idx]
        scaled_face = (face - center) * scale_n + center
        poses[i]["faces"][0] = scaled_face

        body = pose["bodies"]
        candidate = body["candidate"]
        candidate_np = np.array(candidate[0])
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]["bodies"]["candidate"][0] = scaled_candidate

    pose["faces"][0] = scaled_face
    return scale_n


def p3d_single_p2d(points, intrinsic_matrix):
    X, Y, Z = points[0], points[1], points[2]
    u = (intrinsic_matrix[0, 0] * X / Z) + intrinsic_matrix[0, 2]
    v = (intrinsic_matrix[1, 1] * Y / Z) + intrinsic_matrix[1, 2]
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    return np.array([u_np, v_np])


def shift_dwpose_according_to_nlf(smpl_poses, aligned_poses, ori_intrinstics, modified_intrinstics, height, width):
    for i in range(len(smpl_poses)):
        persons_joints_list = smpl_poses[i]
        poses_list = aligned_poses[i]
        for person_idx, person_joints in enumerate(persons_joints_list):
            face = poses_list["faces"][person_idx]
            right_hand = poses_list["hands"][2 * person_idx]
            left_hand = poses_list["hands"][2 * person_idx + 1]
            candidate = poses_list["bodies"]["candidate"][person_idx]

            person_joint_15_2d_shift = (
                p3d_single_p2d(person_joints[15], modified_intrinstics) - p3d_single_p2d(person_joints[15], ori_intrinstics)
                if person_joints[15, 2] > 0.01
                else np.array([0.0, 0.0])
            )
            person_joint_20_2d_shift = (
                p3d_single_p2d(person_joints[20], modified_intrinstics) - p3d_single_p2d(person_joints[20], ori_intrinstics)
                if person_joints[20, 2] > 0.01
                else np.array([0.0, 0.0])
            )
            person_joint_21_2d_shift = (
                p3d_single_p2d(person_joints[21], modified_intrinstics) - p3d_single_p2d(person_joints[21], ori_intrinstics)
                if person_joints[21, 2] > 0.01
                else np.array([0.0, 0.0])
            )

            face[:, 0] += person_joint_15_2d_shift[0] / width
            face[:, 1] += person_joint_15_2d_shift[1] / height
            right_hand[:, 0] += person_joint_20_2d_shift[0] / width
            right_hand[:, 1] += person_joint_20_2d_shift[1] / height
            left_hand[:, 0] += person_joint_21_2d_shift[0] / width
            left_hand[:, 1] += person_joint_21_2d_shift[1] / height
            candidate[:, 0] += person_joint_15_2d_shift[0] / width
            candidate[:, 1] += person_joint_15_2d_shift[1] / height


def get_single_pose_cylinder_specs(args):
    idx, pose, focal, princpt, height, width, colors, limb_seq, draw_seq = args
    cylinder_specs = []

    for joints3d in pose:
        joints3d = joints3d.cpu().numpy()
        joints3d = process_data_to_COCO_format(joints3d)
        for line_idx in draw_seq:
            line = limb_seq[line_idx]
            start, end = line[0], line[1]
            if np.sum(joints3d[start]) == 0 or np.sum(joints3d[end]) == 0:
                continue
            cylinder_specs.append((joints3d[start], joints3d[end], colors[line_idx]))
    return cylinder_specs


def collect_smpl_poses(data):
    uncollected_smpl_poses = [item["nlfpose"] for item in data]
    smpl_poses = [[] for _ in range(len(uncollected_smpl_poses))]
    for frame_idx in range(len(uncollected_smpl_poses)):
        for person_idx in range(len(uncollected_smpl_poses[frame_idx])):
            if len(uncollected_smpl_poses[frame_idx][person_idx]) > 0:
                smpl_poses[frame_idx].append(uncollected_smpl_poses[frame_idx][person_idx][0])
            else:
                smpl_poses[frame_idx].append(torch.zeros((24, 3), dtype=torch.float32))
    return smpl_poses


def render_nlf_as_images(
    data,
    poses,
    reshape_pool=None,
    intrinsic_matrix=None,
    draw_2d=True,
    aug_2d=False,
    aug_cam=False,
):
    height, width = data[0]["video_height"], data[0]["video_width"]
    video_length = len(data)

    base_colors_255_dict = {
        "Red": [255, 0, 0],
        "Orange": [255, 85, 0],
        "Golden Orange": [255, 170, 0],
        "Yellow": [255, 240, 0],
        "Yellow-Green": [180, 255, 0],
        "Bright Green": [0, 255, 0],
        "Light Green-Blue": [0, 255, 85],
        "Aqua": [0, 255, 170],
        "Cyan": [0, 255, 255],
        "Sky Blue": [0, 170, 255],
        "Medium Blue": [0, 85, 255],
        "Pure Blue": [0, 0, 255],
        "Purple-Blue": [85, 0, 255],
        "Medium Purple": [170, 0, 255],
        "Grey": [150, 150, 150],
        "Pink-Magenta": [255, 0, 170],
        "Dark Pink": [255, 0, 85],
        "Violet": [100, 0, 255],
        "Dark Violet": [50, 0, 255],
    }

    ordered_colors_255 = [
        base_colors_255_dict["Red"],
        base_colors_255_dict["Cyan"],
        base_colors_255_dict["Orange"],
        base_colors_255_dict["Golden Orange"],
        base_colors_255_dict["Sky Blue"],
        base_colors_255_dict["Medium Blue"],
        base_colors_255_dict["Yellow-Green"],
        base_colors_255_dict["Bright Green"],
        base_colors_255_dict["Light Green-Blue"],
        base_colors_255_dict["Pure Blue"],
        base_colors_255_dict["Purple-Blue"],
        base_colors_255_dict["Medium Purple"],
        base_colors_255_dict["Grey"],
        base_colors_255_dict["Pink-Magenta"],
        base_colors_255_dict["Dark Violet"],
        base_colors_255_dict["Pink-Magenta"],
        base_colors_255_dict["Dark Violet"],
    ]

    limb_seq = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 0],
        [0, 14],
        [14, 16],
        [0, 15],
        [15, 17],
    ]

    draw_seq = [
        0,
        2,
        3,
        1,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]

    colors = [[c / 300 + 0.15 for c in color_rgb] + [0.8] for color_rgb in ordered_colors_255]

    if poses is not None:
        smpl_poses = collect_smpl_poses(data)
        aligned_poses = copy.deepcopy(poses)
        if reshape_pool is not None:
            for i in range(video_length):
                persons_joints_list = smpl_poses[i]
                poses_list = aligned_poses[i]
                for person_idx, person_joints in enumerate(persons_joints_list):
                    candidate = poses_list["bodies"]["candidate"][person_idx]
                    subset = poses_list["bodies"]["subset"][person_idx]
                    face = poses_list["faces"][person_idx]
                    right_hand = poses_list["hands"][2 * person_idx]
                    left_hand = poses_list["hands"][2 * person_idx + 1]
                    reshape_pool.apply_random_reshapes(person_joints, candidate, left_hand, right_hand, face, subset)
    else:
        smpl_poses = [item["nlfpose"] for item in data]
        aligned_poses = None

    if intrinsic_matrix is None:
        intrinsic_matrix = intrinsic_matrix_from_field_of_view((height, width))
    focal_x = intrinsic_matrix[0, 0]
    focal_y = intrinsic_matrix[1, 1]
    princpt = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])

    if aug_cam and random.random() < 0.3 and aligned_poses is not None:
        w_shift_factor = random.uniform(-0.04, 0.04)
        h_shift_factor = random.uniform(-0.04, 0.04)
        princpt = (princpt[0] - w_shift_factor * width, princpt[1] - h_shift_factor * height)
        new_intrinsic_matrix = copy.deepcopy(intrinsic_matrix)
        new_intrinsic_matrix[0, 2] = princpt[0]
        new_intrinsic_matrix[1, 2] = princpt[1]
        shift_dwpose_according_to_nlf(smpl_poses, aligned_poses, intrinsic_matrix, new_intrinsic_matrix, height, width)

    cylinder_specs_list = []
    for i in range(video_length):
        cylinder_specs = get_single_pose_cylinder_specs((i, smpl_poses[i], None, None, None, None, colors, limb_seq, draw_seq))
        cylinder_specs_list.append(cylinder_specs)

    frames_np_rgba = render_whole(
        cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1]
    )

    if poses is not None and draw_2d:
        canvas_2d = draw_pose_to_canvas_np(
            aligned_poses,
            pool=None,
            H=height,
            W=width,
            reshape_scale=0,
            show_feet_flag=False,
            show_body_flag=False,
            show_cheek_flag=True,
            dw_hand=True,
        )
        scale_h = random.uniform(0.85, 1.15)
        scale_w = random.uniform(0.85, 1.15)
        rescale_flag = random.random() < 0.4 if reshape_pool is not None else False
        for i in range(len(frames_np_rgba)):
            frame_img = frames_np_rgba[i]
            canvas_img = canvas_2d[i]
            mask = canvas_img != 0
            frame_img[:, :, :3][mask] = canvas_img[mask]
            frames_np_rgba[i] = frame_img
            if aug_2d:
                if rescale_flag:
                    frames_np_rgba[i] = scale_image_hw_keep_size(frames_np_rgba[i], scale_h, scale_w)
                if reshape_pool is not None:
                    if random.random() < 0.04:
                        frames_np_rgba[i][:, :, 0:3] = 0
    else:
        scale_h = random.uniform(0.85, 1.15)
        scale_w = random.uniform(0.85, 1.15)
        rescale_flag = random.random() < 0.4 if reshape_pool is not None else False
        for i in range(len(frames_np_rgba)):
            if aug_2d:
                if rescale_flag:
                    frames_np_rgba[i] = scale_image_hw_keep_size(frames_np_rgba[i], scale_h, scale_w)
                if reshape_pool is not None:
                    if random.random() < 0.04:
                        frames_np_rgba[i][:, :, 0:3] = 0

    return frames_np_rgba


def render_multi_nlf_as_images(
    data,
    poses,
    reshape_pool=None,
    intrinsic_matrix=None,
    draw_2d=True,
    aug_2d=False,
    aug_cam=False,
):
    """Render multi-person NLF results with per-person color palettes (SCAIL-Pose multi-human)."""
    height, width = data[0]["video_height"], data[0]["video_width"]
    video_length = len(data)

    # Pastel-ish palette for person 1, saturated palette for person 2 (matches upstream).
    second_person_base_colors_255_dict = {
        "Red": [255, 20, 20],
        "Orange": [255, 60, 0],
        "Golden Orange": [255, 110, 0],
        "Yellow": [255, 200, 0],
        "Yellow-Green": [160, 255, 40],
        "Bright Green": [0, 255, 50],
        "Light Green-Blue": [0, 255, 100],
        "Aqua": [0, 255, 200],
        "Cyan": [0, 230, 255],
        "Sky Blue": [0, 130, 255],
        "Medium Blue": [0, 70, 255],
        "Pure Blue": [0, 0, 255],
        "Purple-Blue": [80, 0, 255],
        "Medium Purple": [160, 0, 255],
        "Grey": [130, 130, 130],
        "Pink-Magenta": [255, 0, 150],
        "Dark Pink": [255, 0, 100],
        "Violet": [120, 0, 255],
        "Dark Violet": [60, 0, 255],
    }

    first_person_base_colors_255_dict = {
        "Red": [255, 150, 150],
        "Orange": [255, 180, 140],
        "Golden Orange": [255, 215, 150],
        "Yellow": [255, 240, 170],
        "Yellow-Green": [200, 255, 100],
        "Bright Green": [100, 255, 100],
        "Light Green-Blue": [140, 255, 180],
        "Aqua": [150, 240, 200],
        "Cyan": [180, 230, 240],
        "Sky Blue": [160, 200, 255],
        "Medium Blue": [100, 120, 255],
        "Pure Blue": [120, 140, 255],
        "Purple-Blue": [180, 90, 255],
        "Medium Purple": [190, 120, 255],
        "Grey": [210, 210, 210],
        "Pink-Magenta": [255, 120, 200],
        "Dark Pink": [255, 150, 180],
        "Violet": [200, 90, 255],
        "Dark Violet": [130, 80, 255],
    }

    ordered_colors_255_list = []
    for base_colors_255_dict in [first_person_base_colors_255_dict, second_person_base_colors_255_dict]:
        ordered_colors_255_list.append(
            [
                base_colors_255_dict["Red"],
                base_colors_255_dict["Cyan"],
                base_colors_255_dict["Orange"],
                base_colors_255_dict["Golden Orange"],
                base_colors_255_dict["Sky Blue"],
                base_colors_255_dict["Medium Blue"],
                base_colors_255_dict["Yellow-Green"],
                base_colors_255_dict["Bright Green"],
                base_colors_255_dict["Light Green-Blue"],
                base_colors_255_dict["Pure Blue"],
                base_colors_255_dict["Purple-Blue"],
                base_colors_255_dict["Medium Purple"],
                base_colors_255_dict["Grey"],
                base_colors_255_dict["Pink-Magenta"],
                base_colors_255_dict["Dark Violet"],
                base_colors_255_dict["Pink-Magenta"],
                base_colors_255_dict["Dark Violet"],
            ]
        )

    limb_seq = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 0],
        [0, 14],
        [14, 16],
        [0, 15],
        [15, 17],
    ]

    draw_seq = [0, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    smpl_poses = collect_smpl_poses(data)
    max_people = max((len(p) for p in smpl_poses), default=0)
    colors_per_person = []
    for person_idx in range(max_people):
        palette_idx = min(person_idx, len(ordered_colors_255_list) - 1)
        colors_per_person.append([[c / 300 + 0.15 for c in rgb] + [0.8] for rgb in ordered_colors_255_list[palette_idx]])

    if intrinsic_matrix is None:
        intrinsic_matrix = intrinsic_matrix_from_field_of_view((height, width))
    focal_x = intrinsic_matrix[0, 0]
    focal_y = intrinsic_matrix[1, 1]
    princpt = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])

    cylinder_specs_list = []
    for frame_idx in range(video_length):
        specs = []
        for person_idx, joints3d in enumerate(smpl_poses[frame_idx]):
            if person_idx >= len(colors_per_person):
                break
            joints3d_np = process_data_to_COCO_format(joints3d.cpu().numpy())
            for line_idx in draw_seq:
                start, end = limb_seq[line_idx]
                if np.sum(joints3d_np[start]) == 0 or np.sum(joints3d_np[end]) == 0:
                    continue
                specs.append((joints3d_np[start], joints3d_np[end], colors_per_person[person_idx][line_idx]))
        cylinder_specs_list.append(specs)

    frames_np_rgba = render_whole(cylinder_specs_list, H=height, W=width, fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])

    if poses is not None and draw_2d:
        aligned_poses = copy.deepcopy(poses)
        canvas_2d = draw_pose_to_canvas_np(
            aligned_poses,
            pool=None,
            H=height,
            W=width,
            reshape_scale=0,
            show_feet_flag=False,
            show_body_flag=False,
            show_cheek_flag=True,
            dw_hand=True,
        )
        for i in range(len(frames_np_rgba)):
            frame_img = frames_np_rgba[i]
            canvas_img = canvas_2d[i]
            mask = canvas_img != 0
            frame_img[:, :, :3][mask] = canvas_img[mask]
            frames_np_rgba[i] = frame_img

    return frames_np_rgba
