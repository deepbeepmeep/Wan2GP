# models/wan/frames2video/core.py
# Frames2Video adapter with robust input handling and verbose debug logging
# - accepts PIL.Image, numpy.ndarray, or torch.Tensor
# - normalizes to [-1,1] when inputs are in [0,1]
# - ensures explicit frame dim [C,1,H,W] for start/end
# - converts to model's VAE dtype and device
# - logs shapes, dtypes and devices at each step

from typing import List, Optional, Any, Dict
import torch
import torchvision.transforms.functional as TF
import numpy as np
import traceback


def _tensor_from_pil_or_ndarray(pic: Any) -> torch.Tensor:
    """Return a float32 tensor in [0,1] with shape [C,H,W] from PIL or ndarray."""
    if isinstance(pic, torch.Tensor):
        return pic
    if isinstance(pic, np.ndarray):
        arr = pic
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            raise TypeError(f"[F2V debug] unsupported ndarray shape: {arr.shape}")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).float() / 255.0
    # assume PIL Image
    return TF.to_tensor(pic)  # float32 [C,H,W] in [0,1]


def _normalize_to_minus1_plus1_if_needed(t: torch.Tensor) -> torch.Tensor:
    """
    If tensor looks like [0,1], convert to [-1,1]. Otherwise leave as-is.
    """
    try:
        mx = float(t.max())
        mn = float(t.min())
        if mx <= 1.01 and mn >= 0.0:
            return t.mul(2.0).sub(1.0)
    except Exception:
        pass
    return t


def _to_tensor_frame_generic(img: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert input (PIL.Image, ndarray, or torch.Tensor) to a tensor (typically normalized to [-1,1]),
    add a frame dim if missing, and move to device/dtype.

    Returns:
      - If input is PIL/ndarray: [C,1,H,W]
      - If input is tensor:
          * [C,H,W] -> [C,1,H,W]
          * [C,1,H,W] kept
          * [B,C,H,W] -> take first batch -> [C,H,W] then -> [C,1,H,W]
          * [C,F,H,W] kept as-is (caller decides if this is valid downstream)
          * [B,C,F,H,W] kept as-is (caller decides)
    """
    if isinstance(img, torch.Tensor):
        t = img

        if t.dim() == 5:
            # [B,C,F,H,W] (or similar) - do not reshape, just normalize and move
            t = _normalize_to_minus1_plus1_if_needed(t)
            return t.to(device=device, dtype=dtype)

        if t.dim() == 4:
            # could be [C,F,H,W] or [B,C,H,W]
            if t.shape[0] > 4 and t.shape[1] <= 4:
                # likely [B,C,H,W]
                t = t[0]  # -> [C,H,W]
                t = _normalize_to_minus1_plus1_if_needed(t)
                t = t.unsqueeze(1)  # -> [C,1,H,W]
                return t.to(device=device, dtype=dtype)

            # treat as [C,F,H,W] (could include F==1)
            t = _normalize_to_minus1_plus1_if_needed(t)
            return t.to(device=device, dtype=dtype)

        if t.dim() == 3:
            # [C,H,W] -> [C,1,H,W]
            t = _normalize_to_minus1_plus1_if_needed(t)
            t = t.unsqueeze(1)
            return t.to(device=device, dtype=dtype)

        raise TypeError(f"[F2V debug] unsupported tensor shape: {tuple(t.shape)}")

    # PIL / ndarray path
    t = _tensor_from_pil_or_ndarray(img)          # [C,H,W] in [0,1]
    t = _normalize_to_minus1_plus1_if_needed(t)   # -> [-1,1]
    t = t.unsqueeze(1)                             # [C,1,H,W]
    return t.to(device=device, dtype=dtype)


def _prepare_middle_images(middle_images: Optional[List[Any]], device: torch.device, dtype: torch.dtype):
    """Convert optional list of middle images (PIL/ndarray/tensor) to list of tensors."""
    if middle_images is None:
        return None
    out = []
    for i, img in enumerate(middle_images):
        try:
            out.append(_to_tensor_frame_generic(img, device=device, dtype=dtype))
        except Exception as e:
            print(f"[F2V debug] failed to convert middle image #{i}: {e}")
            traceback.print_exc()
            out.append(None)
    return out


def run_frames2video(self, model_def: dict, inputs: Dict[str, Any]):
    """
    Entry point used by WanAny2V when frames2video_class is set.

    Accepts image_start/image_end as PIL.Image, numpy.ndarray, or torch.Tensor.
    Prepares tensors and calls self._generate_from_preprocessed(...).
    """
    try:
        if not hasattr(self, "_generate_from_preprocessed"):
            raise AttributeError(
                "WanAny2V is missing _generate_from_preprocessed; apply the any2video.py refactor first."
            )

        image_start = inputs.get("image_start", None)
        image_end = inputs.get("image_end", None)
        middle_images = inputs.get("middle_images", None)

        if image_start is None or image_end is None:
            raise ValueError("run_frames2video requires 'image_start' and 'image_end' in inputs")

        frame_num = int(inputs.get("frame_num", 81))
        sampling_steps = int(inputs.get("sampling_steps", 40))
        guide_scale = float(inputs.get("guide_scale", 5.0))

        device = getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        vae_dtype = getattr(self, "VAE_dtype", torch.float32)

        print("[F2V debug] run_frames2video called")
        print("[F2V debug] input types:", type(image_start), type(image_end), "middle_images:",
              None if middle_images is None else f"list(len={len(middle_images)})")
        print("[F2V debug] model device:", device, "model VAE_dtype:", vae_dtype)

        start_tensor = _to_tensor_frame_generic(image_start, device=device, dtype=vae_dtype)
        end_tensor = _to_tensor_frame_generic(image_end, device=device, dtype=vae_dtype)
        middle_tensors = _prepare_middle_images(middle_images, device=device, dtype=vae_dtype)

        # Hard assertions for Frames2Video start/end
        if start_tensor.dim() != 4 or start_tensor.shape[0] != 3 or start_tensor.shape[1] != 1:
            raise ValueError(f"[F2V debug] start_tensor must be [3,1,H,W], got {tuple(start_tensor.shape)}")
        if end_tensor.dim() != 4 or end_tensor.shape[0] != 3 or end_tensor.shape[1] != 1:
            raise ValueError(f"[F2V debug] end_tensor must be [3,1,H,W], got {tuple(end_tensor.shape)}")

        def _log_tensor_info(name, t):
            if t is None:
                print(f"[F2V debug] {name}: None")
                return
            try:
                print(
                    f"[F2V debug] {name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
                    f"min={float(t.min()):.6f}, max={float(t.max()):.6f}, mean={float(t.mean()):.6f}"
                )
            except Exception as e:
                print(
                    f"[F2V debug] {name}: shape={getattr(t, 'shape', None)}, dtype={getattr(t, 'dtype', None)}, "
                    f"device={getattr(t, 'device', None)} (stats unavailable: {e})"
                )

        _log_tensor_info("start_tensor", start_tensor)
        _log_tensor_info("end_tensor", end_tensor)
        if middle_tensors is not None:
            for i, mt in enumerate(middle_tensors):
                _log_tensor_info(f"middle_tensor[{i}]", mt)

        gen_kwargs = dict(
            img=start_tensor,
            img_end=end_tensor,
            middle_images=middle_tensors,
            frame_num=frame_num,
            sample_solver="unipc",
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            offload_model=True,
            input_prompt="",
        )

        extra = inputs.get("generate_kwargs", {})
        if isinstance(extra, dict):
            gen_kwargs.update(extra)

        print("[F2V debug] calling _generate_from_preprocessed with frame_num:", frame_num,
              "sampling_steps:", sampling_steps, "guide_scale:", guide_scale)
        print("[F2V debug] generate kwargs keys:", list(gen_kwargs.keys()))

        result = self._generate_from_preprocessed(
            gen_kwargs.pop("img"),
            gen_kwargs.pop("img_end"),
            middle_images=gen_kwargs.pop("middle_images", None),
            **gen_kwargs
        )

        if isinstance(result, torch.Tensor):
            print("[F2V debug] generate returned tensor: shape=", tuple(result.shape),
                  "dtype=", result.dtype, "device=", result.device)
            try:
                print("[F2V debug] result stats: min={:.6f}, max={:.6f}, mean={:.6f}".format(
                    float(result.min()), float(result.max()), float(result.mean())
                ))
            except Exception:
                pass
        elif isinstance(result, (list, tuple)):
            print(f"[F2V debug] generate returned {type(result).__name__} of length {len(result)}")
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    print(f"[F2V debug] result[{i}] shape={tuple(r.shape)}, dtype={r.dtype}, device={r.device}")
                else:
                    print(f"[F2V debug] result[{i}] type={type(r)}")
        else:
            print("[F2V debug] generate returned unexpected type:", type(result))

        print("[F2V debug] run_frames2video finished successfully")
        return result

    except Exception as e:
        print("[F2V debug] run_frames2video failed:", e)
        traceback.print_exc()
        raise
