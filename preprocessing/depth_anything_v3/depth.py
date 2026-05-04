from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_PACKAGE_ROOT = Path(__file__).resolve().parent
DA3_BF16_MODEL = "depth/depth_anything_v3_vitl_bf16.safetensors"
DA3_METRIC_BF16_MODEL = "depth/depth_anything_v3_metric_large_bf16.safetensors"


def resolve_da3_chunk_size(chunk_size=-1, device=None):
    chunk_size = int(chunk_size if chunk_size is not None else -1)
    if chunk_size != -1:
        return chunk_size
    if not torch.cuda.is_available():
        return 33
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        return 33
    device_index = torch.cuda.current_device() if device.index is None else device.index
    vram_gb = torch.cuda.get_device_properties(device_index).total_memory / 1_000_000_000
    if vram_gb < 8:
        return 33
    if vram_gb < 24:
        return 65
    return 97


def _load_da3(pretrained_model, device, model_name="da3-large"):
    from mmgp import offload
    from safetensors import safe_open

    from .api import DepthAnything3

    model = DepthAnything3(model_name=model_name)
    pretrained_model = str(pretrained_model)
    if not pretrained_model.endswith(".safetensors"):
        raise ValueError(f"Depth Anything 3 now expects the bf16 safetensors checkpoint, got: {pretrained_model}")
    model_keys = set(model.state_dict().keys())
    with safe_open(pretrained_model, framework="pt", device="cpu") as f:
        checkpoint_keys = set(f.keys())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    allowed_missing = tuple(f"model.head.scratch.output_conv2_aux.{idx}.2." for idx in range(1, 4))
    unsupported_missing = [key for key in missing if not key.startswith(allowed_missing)]
    if unexpected or unsupported_missing:
        raise RuntimeError(f"Unexpected DA3 checkpoint keys: unexpected={unexpected}, missing={unsupported_missing}")
    offload.load_model_data(model, pretrained_model, writable_tensors=False, default_dtype=torch.bfloat16, ignore_missing_keys=True)
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def _resize_2d(array, height, width, mode="bilinear", inverse=False):
    if array.shape[-2:] == (height, width):
        return array.copy()
    dtype = array.dtype
    tensor = torch.from_numpy(array).to(torch.float64)
    leading = tensor.shape[:-2]
    tensor = tensor.reshape(-1, *tensor.shape[-2:])
    if inverse:
        tensor = 1 / tensor
    tensor = F.interpolate(tensor[:, None], size=(height, width), mode=mode)[:, 0]
    if inverse:
        tensor = 1 / tensor
    tensor = tensor.reshape(*leading, height, width)
    if dtype == np.bool_:
        tensor = tensor >= 0.5
    return tensor.numpy().astype(dtype)


def _k_to_intrinsics(k):
    intrinsics = np.zeros((k.shape[0], 4), dtype=np.float32)
    intrinsics[:, 0] = k[:, 0, 0]
    intrinsics[:, 1] = k[:, 1, 1]
    intrinsics[:, 2] = k[:, 0, 2]
    intrinsics[:, 3] = k[:, 1, 2]
    return intrinsics


def _prediction_to_arrays(prediction, height, width):
    depths = prediction.depth.astype(np.float32)
    sky = getattr(prediction, "sky", None)
    if sky is None:
        sky = np.zeros_like(depths, dtype=np.bool_)
    else:
        sky = sky.astype(np.bool_)
    cam_w2c = prediction.extrinsics.astype(np.float32)
    intrinsics = _k_to_intrinsics(prediction.intrinsics.astype(np.float32))
    processed = prediction.processed_images
    proc_h, proc_w = processed.shape[1:3]

    depths = _resize_2d(depths, height, width, mode="bilinear", inverse=True)
    sky = _resize_2d(sky, height, width, mode="nearest", inverse=False)
    intrinsics[:, 0::2] *= width / proc_w
    intrinsics[:, 1::2] *= height / proc_h
    return depths, sky, cam_w2c, intrinsics


def _camera_w2c_to_c2w(cam_w2c):
    cam_w2c_44 = np.zeros((cam_w2c.shape[0], 4, 4), dtype=np.float32)
    cam_w2c_44[:, :3, :4] = cam_w2c
    cam_w2c_44[:, 3, 3] = 1.0
    cam_c2w = np.linalg.inv(cam_w2c_44)
    return (np.linalg.inv(cam_c2w[0])[None] @ cam_c2w).astype(np.float32)


def _w2c_to_pose(cam_w2c):
    cam_w2c_44 = np.zeros((cam_w2c.shape[0], 4, 4), dtype=np.float64)
    cam_w2c_44[:, :3, :4] = cam_w2c.astype(np.float64)
    cam_w2c_44[:, 3, 3] = 1.0
    return np.linalg.inv(cam_w2c_44)


def _closest_rotation(matrix):
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _pose_based_chunk_alignment(ref_w2c, est_w2c):
    ref_pose = _w2c_to_pose(ref_w2c)
    est_pose = _w2c_to_pose(est_w2c)
    rotation = _closest_rotation(np.mean(ref_pose[:, :3, :3] @ np.swapaxes(est_pose[:, :3, :3], -1, -2), axis=0))

    ref_centers = ref_pose[:, :3, 3]
    est_centers = est_pose[:, :3, 3]
    pair_i, pair_j = np.triu_indices(ref_centers.shape[0], k=1)
    ref_dists = np.linalg.norm(ref_centers[pair_i] - ref_centers[pair_j], axis=1)
    est_dists = np.linalg.norm(est_centers[pair_i] - est_centers[pair_j], axis=1)
    valid = est_dists > np.finfo(np.float64).eps
    scale = float(np.median(ref_dists[valid] / est_dists[valid])) if valid.any() else 1.0

    est_mean = est_centers.mean(axis=0)
    ref_mean = ref_centers.mean(axis=0)
    translation = ref_mean - scale * (rotation @ est_mean)
    return rotation.astype(np.float32), translation.astype(np.float32), np.float32(scale)


def _apply_sim3_to_w2c(cam_w2c, rotation, translation, scale):
    cam_w2c_44 = np.zeros((cam_w2c.shape[0], 4, 4), dtype=np.float32)
    cam_w2c_44[:, :3, :4] = cam_w2c
    cam_w2c_44[:, 3, 3] = 1.0
    poses = np.linalg.inv(cam_w2c_44)
    aligned = poses.copy()
    aligned[:, :3, :3] = rotation @ poses[:, :3, :3]
    aligned[:, :3, 3] = (rotation @ (scale * poses[:, :3, 3]).T).T + translation
    return np.linalg.inv(aligned)[:, :3, :4].astype(np.float32)


def _chunk_ranges(frame_count, chunk_size, overlap):
    if chunk_size <= 0 or chunk_size >= frame_count:
        return [(0, frame_count)]
    if overlap < 8:
        raise ValueError("DA3 temporal chunking requires at least 8 overlap frames")
    if overlap >= chunk_size:
        raise ValueError("DA3 temporal chunk overlap must be smaller than the chunk size")
    ranges, start, step = [], 0, chunk_size - overlap
    while True:
        end = start + chunk_size
        if end >= frame_count:
            ranges.append((frame_count - chunk_size, frame_count))
            break
        ranges.append((start, end))
        next_start = start + step
        final_start = frame_count - chunk_size
        start = final_start if end - final_start >= overlap else next_start
    return ranges


def _infer_da3_prediction(model, video, frame_indices, process_res):
    frames = [Image.fromarray(video[i]) for i in frame_indices]
    return model.inference(frames, process_res=process_res, export_format="npz")


def _infer_da3_depth_prediction(model, video, frame_indices, process_res):
    frames = [Image.fromarray(video[i]) for i in frame_indices]
    prediction = model.inference(frames, process_res=process_res, export_format="npz")
    return _resize_2d(prediction.depth.astype(np.float32), video.shape[1], video.shape[2], mode="bilinear", inverse=True)


def _run_da3_prediction(model, video, process_res, chunk_size=0, chunk_overlap=8):
    frame_count, height, width = video.shape[:3]
    chunk_size = resolve_da3_chunk_size(chunk_size)
    ranges = _chunk_ranges(frame_count, chunk_size, chunk_overlap)
    if len(ranges) == 1:
        prediction = _infer_da3_prediction(model, video, range(frame_count), process_res)
        depths, sky, cam_w2c, intrinsics = _prediction_to_arrays(prediction, height, width)
        return depths, sky, _camera_w2c_to_c2w(cam_w2c), intrinsics

    depths_all = np.empty((frame_count, height, width), dtype=np.float32)
    sky_all = np.empty((frame_count, height, width), dtype=np.bool_)
    cam_w2c_all = np.empty((frame_count, 3, 4), dtype=np.float32)
    intrinsics_all = np.empty((frame_count, 4), dtype=np.float32)
    filled = np.zeros(frame_count, dtype=np.bool_)

    for start, end in ranges:
        indices = np.arange(start, end)
        prediction = _infer_da3_prediction(model, video, indices, process_res)
        depths, sky, cam_w2c, intrinsics = _prediction_to_arrays(prediction, height, width)
        overlap_mask = filled[indices]
        if overlap_mask.any():
            if int(overlap_mask.sum()) < 3:
                raise ValueError("DA3 temporal chunking produced fewer than 3 overlap frames for alignment")
            ref_w2c = cam_w2c_all[indices[overlap_mask]]
            est_w2c = cam_w2c[overlap_mask]
            rotation, translation, scale = _pose_based_chunk_alignment(ref_w2c, est_w2c)
            cam_w2c = _apply_sim3_to_w2c(cam_w2c, rotation, translation, scale)
            depths *= np.float32(scale)
        keep_mask = ~filled[indices]
        keep_indices = indices[keep_mask]
        depths_all[keep_indices] = depths[keep_mask]
        sky_all[keep_indices] = sky[keep_mask]
        cam_w2c_all[keep_indices] = cam_w2c[keep_mask]
        intrinsics_all[keep_indices] = intrinsics[keep_mask]
        filled[keep_indices] = True
        del prediction, depths, sky, cam_w2c, intrinsics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not filled.all():
        missing = np.flatnonzero(~filled).tolist()
        raise RuntimeError(f"DA3 temporal chunking failed to fill frames: {missing}")
    return depths_all, sky_all, _camera_w2c_to_c2w(cam_w2c_all), intrinsics_all


def _run_da3_depth_prediction(model, video, process_res, chunk_size=0):
    frame_count, height, width = video.shape[:3]
    chunk_size = resolve_da3_chunk_size(chunk_size)
    if chunk_size <= 0 or chunk_size >= frame_count:
        return _infer_da3_depth_prediction(model, video, range(frame_count), process_res)
    depth_all = np.empty((frame_count, height, width), dtype=np.float32)
    for start in range(0, frame_count, chunk_size):
        end = min(frame_count, start + chunk_size)
        depth_all[start:end] = _infer_da3_depth_prediction(model, video, range(start, end), process_res)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return depth_all


@torch.inference_mode()
def run_da3_reconstruction(video, pretrained_model=None, process_res=0, device=None, chunk_size=0, chunk_overlap=8):
    from shared.utils import files_locator as fl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    chunk_size = resolve_da3_chunk_size(chunk_size, device)
    pretrained_model = pretrained_model or fl.locate_file(DA3_BF16_MODEL)
    model = _load_da3(pretrained_model, device, model_name="da3-large")
    height, width = video.shape[1:3]
    if process_res <= 0:
        process_res = width
    depths, sky, cam_c2w, intrinsics = _run_da3_prediction(model, video, process_res, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return depths, sky, cam_c2w.astype(np.float32), intrinsics.astype(np.float32)


class DepthV3VideoAnnotator:
    def __init__(self, cfg, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.process_res = int(cfg.get("PROCESS_RES", 0) or 0)
        self.chunk_size = resolve_da3_chunk_size(cfg.get("CHUNK_SIZE", -1), self.device)
        self.chunk_overlap = int(cfg.get("CHUNK_OVERLAP", 8) or 8)
        self.model_name = cfg.get("MODEL_NAME", "da3-large")
        self.model = _load_da3(cfg["PRETRAINED_MODEL"], self.device, model_name=self.model_name)

    @torch.inference_mode()
    def forward(self, frames):
        video = np.stack([np.asarray(frame) for frame in frames], axis=0)
        if self.model_name == "da3metric-large":
            depth = _run_da3_depth_prediction(self.model, video, self.process_res or video.shape[2], chunk_size=self.chunk_size)
        else:
            depth, _, _, _ = _run_da3_prediction(self.model, video, self.process_res or video.shape[2], chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        disp = 1.0 / np.maximum(depth, 1e-6)
        disp -= disp.min()
        disp /= max(float(disp.max()), 1e-6)
        depth_video = (disp * 255.0).clip(0, 255).astype(np.uint8)
        return [np.repeat(frame[..., None], 3, axis=2) for frame in depth_video]

    def close(self):
        if self.model is not None:
            self.model.to("cpu")
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
