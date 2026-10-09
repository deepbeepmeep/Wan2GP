"""Camera paths shared by the Wan camera-control models (Fun Control-Camera and Uni3C).

Presets are Vista4D's local camera moves (models/wan/vista4d/preprocess.py). A custom path is either a Vista4D target-camera .npz
(cam_c2w [F, 4, 4] OpenCV camera-to-world, intrinsics [F, 4] = fx, fy, cx, cy in pixels) or a CameraCtrl / RealEstate10K pose .txt
(one line per frame: index fx fy cx cy 0 0 then the 3x4 world-to-camera matrix, intrinsics normalized by the frame size, first line
skipped as in VideoX-Fun). Every path is returned relative to its first camera, with intrinsics in pixels of the output frame.
"""
import numpy as np

from ..vista4d.preprocess import _generate_target_camera_trajectory, _infer_intrinsics_size, _load_cameras, _resize_intrinsics


def _load_pose_txt(path, frame_num, height, width):
    with open(path, "r", encoding="utf-8") as reader:
        rows = [[float(value) for value in line.split()] for line in reader.read().splitlines()[1:] if line.strip()]
    if len(rows) < frame_num:
        raise ValueError(f"Camera path {path} has {len(rows)} poses, {frame_num} are needed.")
    rows = np.asarray(rows[:frame_num], dtype=np.float64)
    fx, fy, cx, cy = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    # VideoX-Fun process_pose_file: the normalized intrinsics describe 1280x720 frames cropped to the output aspect ratio
    if 1280 / 720 > width / height:
        fx = fx * height * (1280 / 720) / width
    else:
        fy = fy * width / (1280 / 720) / height
    w2c = np.tile(np.eye(4), (frame_num, 1, 1))
    w2c[:, :3] = rows[:, 7:19].reshape(-1, 3, 4)
    return np.linalg.inv(w2c), np.stack([fx * width, fy * height, cx * width, cy * height], axis=1)


def _load_npz(path, frame_num, height, width):
    cam_c2w, intrinsics = _load_cameras(path, frame_num)
    camera_height, camera_width = _infer_intrinsics_size(intrinsics, height, width)
    if (camera_height, camera_width) != (height, width):
        intrinsics = _resize_intrinsics(intrinsics, height, width, camera_height, camera_width)
    return cam_c2w.astype(np.float64), intrinsics


def build_camera_path(model_mode, custom_path, custom_settings, frame_num, height, width, intrinsics, focus_depth):
    """Returns (c2w [F, 4, 4], intrinsics [F, 4]) relative to the first camera. `intrinsics` [4] and `focus_depth` (distance of the
    point the arcs turn around, which also scales the translations) only shape the preset moves."""
    if custom_path:
        c2w, intrinsics = (_load_pose_txt if str(custom_path).lower().endswith(".txt") else _load_npz)(custom_path, frame_num, height, width)
        return (np.linalg.inv(c2w[:1]) @ c2w).astype(np.float32), intrinsics.astype(np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], frame_num, axis=0)
    intrinsics = np.repeat(np.asarray(intrinsics, dtype=np.float32)[None], frame_num, axis=0)
    strength = float(custom_settings.get("camera_strength", 100.0)) / 100.0
    return _generate_target_camera_trajectory(c2w, intrinsics, np.full((1, 1, 1), focus_depth, dtype=np.float32), None, model_mode, strength)
