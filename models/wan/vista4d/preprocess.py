import hashlib
import os
import shutil
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from .camera import get_plucker_embedding

VISTA4D_REUSE_GENERATED_MAP_IN_MEMORY = True
VISTA4D_DA3_PROCESS_RES = 896
VISTA4D_DA3_CHUNK_SIZE = -1
VISTA4D_DA3_CHUNK_OVERLAP = 8
_VISTA4D_MAP_CACHE = {}
_VISTA4D_MAP_CACHE_MAX_ITEMS = 2


def _center_slice(length, frame_num):
    if length < frame_num:
        raise ValueError(f"Vista4D input needs at least {frame_num} frames, got {length}.")
    start = (length - frame_num) // 2
    return slice(start, start + frame_num)


def _tensor_to_video_np(frames):
    video = frames.detach().cpu().float().permute(1, 2, 3, 0).numpy()
    return ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def _video_np_to_tensor(video, device, dtype):
    video = torch.from_numpy(video).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    return video.to(device=device, dtype=dtype)


def _crop_resize_video(video, height, width, resample=Image.Resampling.LANCZOS):
    if video.shape[1:3] == (height, width):
        return video.copy()
    frames = []
    for frame in video:
        image = Image.fromarray(frame)
        in_w, in_h = image.size
        if in_h / in_w > height / width:
            crop_h = int(in_w * height / width)
            top = (in_h - crop_h) // 2
            image = image.crop((0, top, in_w, top + crop_h))
        else:
            crop_w = int(in_h * width / height)
            left = (in_w - crop_w) // 2
            image = image.crop((left, 0, left + crop_w, in_h))
        frames.append(np.asarray(image.resize((width, height), resample)))
    return np.stack(frames, axis=0)


def _resize_intrinsics(intrinsics, height, width, height_input, width_input):
    intrinsics = intrinsics.copy()
    if height_input / width_input > height / width:
        crop_h = int(width_input * height / width)
        intrinsics[..., 3] -= (height_input - crop_h) / 2
        height_input = crop_h
    else:
        crop_w = int(height_input * width / height)
        intrinsics[..., 2] -= (width_input - crop_w) / 2
        width_input = crop_w
    intrinsics[..., 0] *= width / width_input
    intrinsics[..., 1] *= height / height_input
    intrinsics[..., 2] *= width / width_input
    intrinsics[..., 3] *= height / height_input
    return intrinsics


def _infer_intrinsics_size(intrinsics, fallback_height, fallback_width):
    cx = float(np.nanmedian(intrinsics[..., 2]))
    cy = float(np.nanmedian(intrinsics[..., 3]))
    if np.isfinite(cx) and np.isfinite(cy) and cx > 0 and cy > 0:
        return max(1, int(round(cy * 2))), max(1, int(round(cx * 2)))
    return fallback_height, fallback_width


def _read_video(path):
    cap = cv2.VideoCapture(os.fspath(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open Vista4D video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Vista4D video has no frames: {path}")
    return np.stack(frames, axis=0)


def _save_video(path, video, fps):
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    writer = imageio.get_writer(os.fspath(path), fps=fps, quality=9, macro_block_size=1)
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def _load_masks(folder, frame_num, height, width, default):
    folder = Path(folder)
    if not folder.is_dir():
        fill = np.ones if default else np.zeros
        return fill((frame_num, height, width), dtype=np.bool_)
    files = sorted(folder.glob("*.png"))
    if len(files) == 0:
        raise ValueError(f"Vista4D mask folder is empty: {folder}")
    masks = []
    for file in files:
        frame = cv2.imread(os.fspath(file), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Could not load Vista4D mask: {file}")
        masks.append(frame > 127)
    masks = np.stack(masks, axis=0)
    masks = masks[_center_slice(masks.shape[0], frame_num)]
    if masks.shape[1:3] != (height, width):
        masks = F.interpolate(torch.from_numpy(masks[:, None].astype(np.float32)), size=(height, width), mode="nearest")[:, 0].numpy() > 0.5
    return masks


def _save_masks(folder, masks):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        cv2.imwrite(os.fspath(folder / f"{idx:05d}.png"), mask.astype(np.uint8) * 255)


def _pixel_grid(height, width):
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    return xs.reshape(-1), ys.reshape(-1)


def _frame_points(video, depths, sky_mask, dynamic_mask, cam_c2w, intrinsics, frame_idx, xs, ys):
    depth = depths[frame_idx].reshape(-1).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 1e-5)
    if sky_mask is not None:
        valid &= ~sky_mask[frame_idx].reshape(-1).astype(np.bool_)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.bool_)
    fx, fy, cx, cy = intrinsics[frame_idx]
    z = depth[valid]
    cam_points = np.stack(((xs[valid] - cx) * z / fx, (ys[valid] - cy) * z / fy, z), axis=1)
    world_points = cam_points @ cam_c2w[frame_idx, :3, :3].T + cam_c2w[frame_idx, :3, 3]
    colors = video[frame_idx].reshape(-1, 3)[valid]
    point_dynamic = dynamic_mask[frame_idx].reshape(-1)[valid] if dynamic_mask is not None else np.zeros(colors.shape[0], dtype=np.bool_)
    return world_points.astype(np.float32), colors, point_dynamic.astype(np.bool_)


def _project_points(points, colors, cam_w2c, intrinsics, height, width):
    if points.shape[0] == 0:
        return None
    cam_points = points @ cam_w2c[:3, :3].T + cam_w2c[:3, 3]
    z = cam_points[:, 2]
    valid = z > 1e-5
    if not np.any(valid):
        return None
    fx, fy, cx, cy = intrinsics
    valid_idx = np.nonzero(valid)[0]
    x = np.rint(cam_points[valid, 0] * fx / z[valid] + cx).astype(np.int32)
    y = np.rint(cam_points[valid, 1] * fy / z[valid] + cy).astype(np.int32)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(inside):
        return None
    return y[inside] * width + x[inside], z[valid][inside], colors[valid][inside], valid_idx[inside]


def _render_point_chunks(point_chunks, cam_w2c, intrinsics, height, width):
    empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
    empty_mask = np.zeros((height, width), dtype=np.bool_)
    zbuf = np.full(height * width, np.inf, dtype=np.float32)
    projected = []
    for points, colors, is_dynamic in point_chunks:
        item = _project_points(points, colors, cam_w2c, intrinsics, height, width)
        if item is None:
            continue
        pix, z, colors, _ = item
        np.minimum.at(zbuf, pix, z)
        projected.append((pix, z, colors, is_dynamic))
    if len(projected) == 0:
        return empty_frame, empty_mask, empty_mask.copy()
    output = np.zeros((height * width, 3), dtype=np.uint8)
    dynamic = np.zeros(height * width, dtype=np.bool_)
    for pix, z, colors, is_dynamic in projected:
        keep = z <= zbuf[pix] + 1e-4
        output[pix[keep]] = colors[keep]
        if is_dynamic:
            dynamic[pix[keep]] = True
    mask = np.isfinite(zbuf).reshape(height, width)
    return output.reshape(height, width, 3), mask, dynamic.reshape(height, width) & mask


def _to_torch_point_cache(point_cache, device):
    return [(torch.from_numpy(points).to(device=device, dtype=torch.float32), torch.from_numpy(colors).to(device=device)) for points, colors in point_cache]


def _project_points_torch(points, colors, cam_w2c, intrinsics, height, width):
    if points.shape[0] == 0:
        return None
    cam_points = points @ cam_w2c[:3, :3].T + cam_w2c[:3, 3]
    z_all = cam_points[:, 2]
    valid = z_all > 1e-5
    if not bool(valid.any()):
        return None
    fx, fy, cx, cy = intrinsics.unbind(0)
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    z = z_all[valid_idx]
    x = torch.round(cam_points[valid_idx, 0] * fx / z + cx).to(torch.int64)
    y = torch.round(cam_points[valid_idx, 1] * fy / z + cy).to(torch.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not bool(inside.any()):
        return None
    return y[inside] * width + x[inside], z[inside], colors[valid_idx[inside]]


def _render_point_chunks_torch(point_chunks, cam_w2c, intrinsics, height, width):
    device = cam_w2c.device
    zbuf = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    projected = []
    order_start = 0
    for points, colors, is_dynamic in point_chunks:
        item = _project_points_torch(points, colors, cam_w2c, intrinsics, height, width)
        if item is None:
            order_start += points.shape[0]
            continue
        pix, z, colors = item
        order = torch.arange(order_start, order_start + pix.shape[0], dtype=torch.int32, device=device)
        zbuf.scatter_reduce_(0, pix, z, reduce="amin", include_self=True)
        projected.append((pix, z, colors, order, is_dynamic))
        order_start += points.shape[0]

    frame = torch.zeros((height * width, 3), dtype=torch.uint8, device=device)
    dynamic = torch.zeros((height * width,), dtype=torch.bool, device=device)
    if len(projected) > 0:
        winner_order = torch.full((height * width,), -1, dtype=torch.int32, device=device)
        for pix, z, _, order, is_dynamic in projected:
            keep = z <= zbuf[pix] + 1e-4
            if keep.any():
                winner_order.scatter_reduce_(0, pix[keep], order[keep], reduce="amax", include_self=True)
                if is_dynamic:
                    dynamic[pix[keep]] = True
        for pix, _, colors, order, _ in projected:
            selected = order == winner_order[pix]
            if selected.any():
                frame[pix[selected]] = colors[selected]

    mask = torch.isfinite(zbuf)
    return frame.reshape(height, width, 3).cpu().numpy(), mask.reshape(height, width).cpu().numpy(), (dynamic & mask).reshape(height, width).cpu().numpy()


def _render_point_cloud_video_torch(static_cache, dynamic_cache, target_cam_w2c, target_intrinsics, height, width, window_radius):
    device = torch.device("cuda")
    static_cache = _to_torch_point_cache(static_cache, device)
    dynamic_cache = _to_torch_point_cache(dynamic_cache, device)
    target_cam_w2c = torch.from_numpy(target_cam_w2c).to(device=device, dtype=torch.float32)
    target_intrinsics = torch.from_numpy(target_intrinsics).to(device=device, dtype=torch.float32)
    rendered = []
    alpha_masks = []
    dynamic_masks = []
    context = torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), context:
        for target_idx in range(len(dynamic_cache)):
            if window_radius < 0:
                point_chunks = [(points, colors, False) for points, colors in static_cache]
            else:
                start = max(0, target_idx - window_radius)
                end = min(len(dynamic_cache), target_idx + window_radius + 1)
                point_chunks = [(static_cache[idx][0], static_cache[idx][1], False) for idx in range(start, end) if static_cache[idx][0].shape[0] > 0]
            dynamic_points, dynamic_colors = dynamic_cache[target_idx]
            if dynamic_points.shape[0] > 0:
                point_chunks.append((dynamic_points, dynamic_colors, True))
            frame, mask, rendered_dynamic = _render_point_chunks_torch(point_chunks, target_cam_w2c[target_idx], target_intrinsics[target_idx], height, width)
            rendered.append(frame)
            alpha_masks.append(mask)
            dynamic_masks.append(rendered_dynamic)
    return np.stack(rendered, axis=0), np.stack(alpha_masks, axis=0), np.stack(dynamic_masks, axis=0)


def _render_point_cloud_video(video, depths, sky_mask, dynamic_mask, cam_c2w, intrinsics, target_cam_c2w=None, target_intrinsics=None, window_radius=-1):
    frame_num, height, width = video.shape[:3]
    if target_cam_c2w is None:
        target_cam_c2w = cam_c2w
    if target_intrinsics is None:
        target_intrinsics = intrinsics
    xs, ys = _pixel_grid(height, width)
    static_cache = []
    dynamic_cache = []
    for idx in range(frame_num):
        points, colors, point_dynamic = _frame_points(video, depths, sky_mask, dynamic_mask, cam_c2w, intrinsics, idx, xs, ys)
        static = ~point_dynamic
        static_cache.append((points[static], colors[static]))
        dynamic_cache.append((points[point_dynamic], colors[point_dynamic]))
    target_cam_w2c = np.linalg.inv(target_cam_c2w).astype(np.float32)

    rendered = []
    alpha_masks = []
    dynamic_masks = []
    window_radius = int(window_radius)
    if window_radius < 0:
        static_points = [points for points, _ in static_cache if points.shape[0] > 0]
        static_colors = [colors for points, colors in static_cache if points.shape[0] > 0]
        static_cache = [(np.concatenate(static_points, axis=0), np.concatenate(static_colors, axis=0))] if static_points else []
    if torch.cuda.is_available():
        return _render_point_cloud_video_torch(static_cache, dynamic_cache, target_cam_w2c, target_intrinsics, height, width, window_radius)
    for target_idx in range(frame_num):
        if window_radius < 0:
            point_chunks = [(points, colors, False) for points, colors in static_cache]
        else:
            start = max(0, target_idx - window_radius)
            end = min(frame_num, target_idx + window_radius + 1)
            point_chunks = [(static_cache[idx][0], static_cache[idx][1], False) for idx in range(start, end) if static_cache[idx][0].shape[0] > 0]
        dynamic_points, dynamic_colors = dynamic_cache[target_idx]
        if dynamic_points.shape[0] > 0:
            point_chunks.append((dynamic_points, dynamic_colors, True))
        frame, mask, rendered_dynamic = _render_point_chunks(point_chunks, target_cam_w2c[target_idx], target_intrinsics[target_idx], height, width)
        rendered.append(frame)
        alpha_masks.append(mask)
        dynamic_masks.append(rendered_dynamic)
    return np.stack(rendered, axis=0), np.stack(alpha_masks, axis=0), np.stack(dynamic_masks, axis=0)


def _load_cameras(path, frame_num):
    data = np.load(path)
    cam_c2w = data["cam_c2w"][_center_slice(data["cam_c2w"].shape[0], frame_num)]
    intrinsics = data["intrinsics"][_center_slice(data["intrinsics"].shape[0], frame_num)]
    return cam_c2w.astype(np.float32), intrinsics.astype(np.float32)


def _save_cameras(path, cam_c2w, intrinsics):
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    np.savez(path, cam_c2w=cam_c2w.astype(np.float32), intrinsics=intrinsics.astype(np.float32))


def _is_camera_npz(path):
    if Path(path).suffix.lower() != ".npz":
        return False
    try:
        with np.load(path) as data:
            return "cam_c2w" in data.files and "intrinsics" in data.files
    except Exception:
        return False


def _extract_zip(path):
    digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    target = Path("ckpts") / "temp" / "vista4d_maps" / digest
    if target.is_dir():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    roots = [item for item in target.iterdir() if item.is_dir()]
    if len(roots) == 1 and (roots[0] / "video_src.mp4").is_file():
        return roots[0]
    return target


def _resolve_custom_input(input_custom):
    if input_custom is None or str(input_custom).strip() == "":
        return None, None
    path = Path(input_custom)
    if path.is_file() and _is_camera_npz(path):
        return None, path
    if path.is_file() and path.suffix.lower() == ".zip":
        return _extract_zip(path), None
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"Vista4D custom_guide must be a preprocessed map folder/zip or target camera .npz, got: {input_custom}")
    return path, None


def _default_output_dir():
    from wgp import save_path
    return save_path


def _hash_array(hasher, array):
    array = np.ascontiguousarray(array)
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.view(np.uint8))


def _hash_file(path):
    hasher = hashlib.sha1()
    with open(path, "rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _make_generated_map_cache_key(video, frame_num, height, width, fps, process_res, chunk_size, chunk_overlap, custom_settings, target_camera_path, model_mode):
    hasher = hashlib.sha1()
    hasher.update(b"vista4d-memory-map-v2")
    hasher.update(f"{frame_num}|{height}|{width}|{fps}|{process_res}|{chunk_size}|{chunk_overlap}".encode("utf-8"))
    if target_camera_path is None:
        hasher.update(str(model_mode or "").encode("utf-8"))
    if isinstance(custom_settings, dict):
        hasher.update(str(custom_settings.get("vista4d_seg_keywords", "_all_")).encode("utf-8"))
        hasher.update(str(custom_settings.get("vista4d_scene_scale", 1.0)).encode("utf-8"))
        if target_camera_path is None:
            hasher.update(str(custom_settings.get("vista4d_camera_strength", 100.0)).encode("utf-8"))
    if target_camera_path is not None:
        hasher.update(os.path.abspath(os.fspath(target_camera_path)).encode("utf-8"))
        hasher.update(_hash_file(target_camera_path).encode("utf-8"))
    _hash_array(hasher, video)
    return hasher.hexdigest()


def _cache_generated_map(cache_key, data):
    _VISTA4D_MAP_CACHE[cache_key] = data
    while len(_VISTA4D_MAP_CACHE) > _VISTA4D_MAP_CACHE_MAX_ITEMS:
        _VISTA4D_MAP_CACHE.pop(next(iter(_VISTA4D_MAP_CACHE)))


def _parse_keywords(custom_settings):
    raw = "_all_"
    if isinstance(custom_settings, dict):
        raw = str(custom_settings.get("vista4d_seg_keywords", raw))
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def _camera_strength(custom_settings):
    if not isinstance(custom_settings, dict):
        return 1.0
    return float(custom_settings.get("vista4d_camera_strength", 100.0) or 0.0) / 100.0


def _get_da3_settings():
    return VISTA4D_DA3_PROCESS_RES, VISTA4D_DA3_CHUNK_SIZE, VISTA4D_DA3_CHUNK_OVERLAP


def _save_generated_map_to_disk(data, source_cam_c2w, source_intrinsics, depths, sky_mask, fps):
    map_dir = Path(_default_output_dir()) / f"vista4d_map_{time.strftime('%Y%m%d_%H%M%S')}"
    map_dir.mkdir(parents=True, exist_ok=True)
    _save_video(map_dir / "video_src.mp4", data["source_video"], fps=fps)
    _save_video(map_dir / "video_pc.mp4", data["point_cloud_video"], fps=fps)
    _save_masks(map_dir / "alpha_mask_src", data["source_alpha_mask"])
    _save_masks(map_dir / "dynamic_mask_src", data["source_motion_mask"])
    _save_masks(map_dir / "static_mask_src", data["source_alpha_mask"] & ~data["source_motion_mask"])
    _save_masks(map_dir / "alpha_mask_pc", data["point_cloud_alpha_mask"])
    _save_masks(map_dir / "dynamic_mask_pc", data["point_cloud_motion_mask"])
    _save_masks(map_dir / "static_mask_pc", data["point_cloud_alpha_mask"] & ~data["point_cloud_motion_mask"])
    _save_masks(map_dir / "sky_mask_src", sky_mask.astype(np.bool_))
    _save_cameras(map_dir / "cameras_src.npz", source_cam_c2w, source_intrinsics)
    _save_cameras(map_dir / "cameras_tgt.npz", data["cam_c2w"], data["intrinsics"])
    np.save(map_dir / "depths_da3.npy", depths.astype(np.float16))
    shutil.make_archive(os.fspath(map_dir), "zip", root_dir=map_dir)
    print(f"Vista4D map saved to: {map_dir}")
    return map_dir


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _estimate_focus_depth(depths, sky_mask):
    valid = np.isfinite(depths) & (depths > 1e-5)
    if sky_mask is not None:
        valid &= ~sky_mask.astype(np.bool_)
    if not np.any(valid):
        return 1.0
    samples = depths[valid]
    if samples.size > 200000:
        samples = samples[np.linspace(0, samples.size - 1, 200000, dtype=np.int64)]
    return max(float(np.median(samples)), 1e-3)


def _apply_local_camera_delta(cam_c2w, translations, rotations):
    target = cam_c2w.copy()
    for idx in range(cam_c2w.shape[0]):
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = rotations[idx]
        delta[:3, 3] = translations[idx]
        target[idx] = cam_c2w[idx] @ delta
    return target


def _generate_target_camera_trajectory(cam_c2w, intrinsics, depths, sky_mask, model_mode, strength):
    mode = str(model_mode or "dolly_zoom")
    if mode not in {
        "dolly_zoom", "left_front_zoom", "right_front_zoom", "close_crane_above", "close_crane_below", "arc_right_45", "arc_left_45",
        "push_in", "pull_back", "truck_right", "truck_left", "pedestal_up", "pedestal_down",
        "pan_right_45", "pan_left_45", "tilt_up_45", "tilt_down_45", "zoom_in", "zoom_out", "bird_view",
        "crane_above_right", "crane_above_left", "crane_below_right", "crane_below_left",
    }:
        mode = "dolly_zoom"
    focus = _estimate_focus_depth(depths, sky_mask)
    frame_num = cam_c2w.shape[0]
    progress = np.linspace(0.0, 1.0, frame_num, dtype=np.float32) * np.float32(strength)
    translations = np.zeros((frame_num, 3), dtype=np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], frame_num, axis=0)
    target_intrinsics = intrinsics.copy()
    focal_scale = np.ones(frame_num, dtype=np.float32)

    if mode == "dolly_zoom":
        translations[:, 2] = 0.35 * focus * progress
        focal_scale = np.clip((focus - translations[:, 2]) / focus, 0.25, 4.0).astype(np.float32)
    elif mode in ("left_front_zoom", "right_front_zoom"):
        translations[:, 0] = (-0.22 if mode == "left_front_zoom" else 0.22) * focus * progress
        translations[:, 2] = 0.20 * focus * progress
        focal_scale = np.clip(1.0 + 0.25 * progress, 0.25, 4.0).astype(np.float32)
    elif mode in ("close_crane_above", "close_crane_below"):
        translations[:, 1] = (-0.22 if mode == "close_crane_above" else 0.22) * focus * progress
        translations[:, 2] = 0.18 * focus * progress
        for idx, amount in enumerate(progress):
            rotations[idx] = _rotation_x(np.deg2rad(-15.0 if mode == "close_crane_above" else 15.0) * amount)
    elif mode in ("arc_right_45", "arc_left_45"):
        sign = -1.0 if mode == "arc_right_45" else 1.0
        angles = np.deg2rad(45.0) * progress * sign
        translations[:, 0] = -focus * np.sin(angles)
        translations[:, 2] = focus * (1.0 - np.cos(angles))
        for idx, angle in enumerate(angles):
            rotations[idx] = _rotation_y(angle)
    elif mode == "push_in":
        translations[:, 2] = 0.25 * focus * progress
    elif mode == "pull_back":
        translations[:, 2] = -0.25 * focus * progress
    elif mode == "truck_right":
        translations[:, 0] = 0.25 * focus * progress
    elif mode == "truck_left":
        translations[:, 0] = -0.25 * focus * progress
    elif mode == "pedestal_up":
        translations[:, 1] = -0.25 * focus * progress
    elif mode == "pedestal_down":
        translations[:, 1] = 0.25 * focus * progress
    elif mode == "pan_right_45":
        for idx, amount in enumerate(progress):
            rotations[idx] = _rotation_y(np.deg2rad(45.0) * amount)
    elif mode == "pan_left_45":
        for idx, amount in enumerate(progress):
            rotations[idx] = _rotation_y(np.deg2rad(-45.0) * amount)
    elif mode == "tilt_up_45":
        for idx, amount in enumerate(progress):
            rotations[idx] = _rotation_x(np.deg2rad(45.0) * amount)
    elif mode == "tilt_down_45":
        for idx, amount in enumerate(progress):
            rotations[idx] = _rotation_x(np.deg2rad(-45.0) * amount)
    elif mode == "zoom_in":
        focal_scale = np.clip(1.0 + 0.60 * progress, 0.25, 4.0).astype(np.float32)
    elif mode == "zoom_out":
        focal_scale = np.clip(1.0 - 0.35 * progress, 0.25, 4.0).astype(np.float32)
    elif mode == "bird_view":
        translations[:, 1] = -0.70 * focus * progress
        translations[:, 2] = -0.12 * focus * progress
        for idx, amount in enumerate(progress):
            rotations[idx] = _rotation_x(np.deg2rad(-60.0) * amount)
    elif mode in ("crane_above_right", "crane_above_left", "crane_below_right", "crane_below_left"):
        above = "above" in mode
        right = "right" in mode
        translations[:, 0] = (0.18 if right else -0.18) * focus * progress
        translations[:, 1] = (-0.24 if above else 0.24) * focus * progress
        translations[:, 2] = 0.16 * focus * progress
        yaw = np.deg2rad(-12.0 if right else 12.0)
        pitch = np.deg2rad(-18.0 if above else 18.0)
        for idx, amount in enumerate(progress):
            rotations[idx] = _rotation_y(yaw * amount) @ _rotation_x(pitch * amount)

    target_intrinsics[:, 0:2] *= focal_scale[:, None]
    return _apply_local_camera_delta(cam_c2w, translations, rotations), target_intrinsics


def _create_map_from_control_video(input_frames, frame_num, height, width, fps, custom_settings, target_camera_path=None, model_mode=None):
    video = _tensor_to_video_np(input_frames)
    video = video[_center_slice(video.shape[0], frame_num)]
    height_input, width_input = video.shape[1:3]
    # video = _crop_resize_video(video, height, width)

    scene_scale = 1.0
    if isinstance(custom_settings, dict):
        scene_scale = float(custom_settings.get("vista4d_scene_scale", 1.0) or 1.0)
    process_res, chunk_size, chunk_overlap = _get_da3_settings()
    if process_res <= 0:
        process_res = width

    from preprocessing.depth_anything_v3.depth import resolve_da3_chunk_size, run_da3_reconstruction
    chunk_size = resolve_da3_chunk_size(chunk_size)
    cache_key = None
    if VISTA4D_REUSE_GENERATED_MAP_IN_MEMORY:
        cache_key = _make_generated_map_cache_key(video, frame_num, height, width, fps, process_res, chunk_size, chunk_overlap, custom_settings, target_camera_path, model_mode)
        cached = _VISTA4D_MAP_CACHE.get(cache_key)
        if cached is not None:
            print("Vista4D map reused from memory cache.")
            return cached

    depths, sky_mask, cam_c2w, intrinsics = run_da3_reconstruction(video, process_res=process_res, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if scene_scale != 1.0:
        depths *= scene_scale
        cam_c2w[:, :3, 3] *= scene_scale
    source_cam_c2w = cam_c2w
    source_intrinsics = intrinsics
    target_cam_c2w = cam_c2w
    target_intrinsics = intrinsics
    if target_camera_path is not None:
        target_cam_c2w, target_intrinsics = _load_cameras(target_camera_path, frame_num)
        camera_height, camera_width = _infer_intrinsics_size(target_intrinsics, height_input, width_input)
        if (camera_height, camera_width) != (height, width):
            target_intrinsics = _resize_intrinsics(target_intrinsics, height, width, camera_height, camera_width)
        if scene_scale != 1.0:
            target_cam_c2w[:, :3, 3] *= scene_scale
    else:
        target_cam_c2w, target_intrinsics = _generate_target_camera_trajectory(cam_c2w, intrinsics, depths, sky_mask, model_mode, _camera_strength(custom_settings))

    keywords = _parse_keywords(custom_settings)
    if len(keywords) == 0:
        dynamic_mask = np.zeros((frame_num, height, width), dtype=np.bool_)
    elif "_all_" in keywords:
        dynamic_mask = np.ones((frame_num, height, width), dtype=np.bool_)
    else:
        from preprocessing.sam3.preprocessor import run_sam3_video
        dynamic_mask = run_sam3_video(video, keywords)

    source_alpha_mask = np.ones((frame_num, height, width), dtype=np.bool_)
    point_cloud_video, point_cloud_alpha_mask, point_cloud_dynamic_mask = _render_point_cloud_video(video, depths, sky_mask, dynamic_mask, cam_c2w, intrinsics, target_cam_c2w, target_intrinsics)
    data = {
        "source_video": video,
        "point_cloud_video": point_cloud_video,
        "source_alpha_mask": source_alpha_mask,
        "source_motion_mask": dynamic_mask,
        "point_cloud_alpha_mask": point_cloud_alpha_mask,
        "point_cloud_motion_mask": point_cloud_dynamic_mask,
        "cam_c2w": target_cam_c2w,
        "intrinsics": target_intrinsics,
    }
    if VISTA4D_REUSE_GENERATED_MAP_IN_MEMORY:
        _cache_generated_map(cache_key, data)
        print("Vista4D map kept in memory cache.")
        return data
    return _save_generated_map_to_disk(data, source_cam_c2w, source_intrinsics, depths, sky_mask, fps)


def _load_map(map_dir, frame_num, height, width):
    video_src = _read_video(map_dir / "video_src.mp4")
    video_pc = _read_video(map_dir / "video_pc.mp4")
    video_src = video_src[_center_slice(video_src.shape[0], frame_num)]
    video_pc = video_pc[_center_slice(video_pc.shape[0], frame_num)]
    src_h, src_w = video_src.shape[1:3]
    video_src = _crop_resize_video(video_src, height, width)
    video_pc = _crop_resize_video(video_pc, height, width)
    cam_c2w, intrinsics = _load_cameras(map_dir / "cameras_tgt.npz", frame_num)
    if (src_h, src_w) != (height, width):
        intrinsics = _resize_intrinsics(intrinsics, height, width, src_h, src_w)
    return {
        "source_video": video_src,
        "point_cloud_video": video_pc,
        "source_alpha_mask": _load_masks(map_dir / "alpha_mask_src", frame_num, height, width, True),
        "source_motion_mask": _load_masks(map_dir / "dynamic_mask_src", frame_num, height, width, False),
        "point_cloud_alpha_mask": _load_masks(map_dir / "alpha_mask_pc", frame_num, height, width, True),
        "point_cloud_motion_mask": _load_masks(map_dir / "dynamic_mask_pc", frame_num, height, width, False),
        "cam_c2w": cam_c2w,
        "intrinsics": intrinsics,
    }


def _pack_masks(alpha_mask, motion_mask, device, dtype):
    masks = np.stack((alpha_mask, motion_mask), axis=0)[None].astype(np.float32)
    masks = torch.from_numpy(masks).to(device=device, dtype=dtype)
    b, c, f, h, w = masks.shape
    masks = torch.cat((torch.repeat_interleave(masks[:, :, 0:1], repeats=4, dim=2), masks[:, :, 1:]), dim=2)
    masks = rearrange(masks, "b c (f sf) (h sh) (w sw) -> b (c sf sh sw) f h w", sf=4, sh=8, sw=8)
    return masks


def prepare_vista4d_condition(pipeline, input_frames, input_custom, frame_num, height, width, tile_size, fps=16, custom_settings=None, model_mode=None):
    map_dir, target_camera_path = _resolve_custom_input(input_custom)
    data = None
    if map_dir is None:
        if input_frames is None:
            raise ValueError("Vista4D needs a control video or a preprocessed map in custom_guide.")
        map_ref = _create_map_from_control_video(input_frames, frame_num, height, width, fps, custom_settings, target_camera_path, model_mode=model_mode)
        if isinstance(map_ref, dict):
            data = map_ref
        else:
            map_dir = map_ref

    if data is None:
        data = _load_map(map_dir, frame_num, height, width)
    device = pipeline.device
    dtype = pipeline.dtype
    vae_dtype = pipeline.VAE_dtype
    source_video = _video_np_to_tensor(data["source_video"], device, vae_dtype)
    point_video = _video_np_to_tensor(data["point_cloud_video"], device, vae_dtype)
    source_latents = pipeline.vae.encode([source_video], tile_size=tile_size)[0].unsqueeze(0).to(device=device, dtype=dtype)
    point_latents = pipeline.vae.encode([point_video], tile_size=tile_size)[0].unsqueeze(0).to(device=device, dtype=dtype)
    source_masks = _pack_masks(data["source_alpha_mask"], data["source_motion_mask"], device, dtype)
    point_masks = _pack_masks(data["point_cloud_alpha_mask"], data["point_cloud_motion_mask"], device, dtype)

    lat_h = height // pipeline.vae_stride[1]
    lat_w = width // pipeline.vae_stride[2]
    cam_c2w = torch.from_numpy(data["cam_c2w"][None]).to(device=device, dtype=dtype)
    intrinsics = torch.from_numpy(data["intrinsics"][None]).to(device=device, dtype=dtype)
    cam_emb = get_plucker_embedding(intrinsics, cam_c2w, height, width, height_dit=lat_h // 2, width_dit=lat_w // 2)
    cam_emb = cam_emb[:, ::pipeline.vae_stride[0]].to(dtype=dtype)

    return {
        "vista": {
            "source_latents": source_latents,
            "point_latents": point_latents,
            "source_mask_latents": source_masks,
            "point_mask_latents": point_masks,
            "cam_emb": cam_emb,
        }
    }
