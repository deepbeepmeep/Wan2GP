from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

HDR_REFERENCE_WHITE_NITS = 203.0
HDR10_MASTER_DISPLAY = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
HDR10_MAX_CLL = "10000,400"
VIDEO_PROMPT_HDR_OUTPUT_FLAG = "&"


def hdr10_zscale_filter(*, reference_white_nits: float = HDR_REFERENCE_WHITE_NITS) -> str:
    return (
        "zscale=pin=709:tin=linear:min=gbr:rin=full:"
        f"p=2020:t=smpte2084:m=2020_ncl:r=limited:npl={float(reference_white_nits):.12g},"
        "format=yuv420p10le"
    )


def hdr10_x265_params() -> str:
    return f"hdr10=1:repeat-headers=1:master-display={HDR10_MASTER_DISPLAY}:max-cll={HDR10_MAX_CLL}:log-level=none"


class LogC3:
    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591

    def compress(self, hdr: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(hdr, min=0.0)
        log_part = self.C * torch.log10(self.A * x + self.B) + self.D
        lin_part = self.E * x + self.F
        return torch.where(x >= self.CUT, log_part, lin_part).clamp_(0.0, 1.0)

    def compress_ldr(self, ldr: torch.Tensor) -> torch.Tensor:
        return torch.clamp(ldr, 0.0, 1.0)

    def decompress(self, logc: torch.Tensor) -> torch.Tensor:
        logc = torch.clamp(logc, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        lin_from_log = (torch.pow(10.0, (logc - self.D) / self.C) - self.B) / self.A
        lin_from_lin = (logc - self.F) / self.E
        return torch.where(logc >= cut_log, lin_from_log, lin_from_lin).clamp_(min=0.0)


class ACEScct:
    """ACEScct log encoding of linear ACEScg (AP1 primaries), as used by the LTX-2.5 SDR-to-HDR IC-LoRA."""

    A = 10.5402377416545
    B = 0.0729055341958355
    X_BREAK = 0.0078125
    Y_BREAK = 0.155251141552511
    LOG_SCALE = 17.52
    LOG_OFFSET = 9.72

    def compress(self, linear_ap1: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(linear_ap1, min=0.0)
        log_part = (torch.log2(torch.clamp(x, min=1e-12)) + self.LOG_OFFSET) / self.LOG_SCALE
        return torch.where(x > self.X_BREAK, log_part, self.A * x + self.B).clamp_(0.0, 1.0)

    def decompress(self, acescct: torch.Tensor) -> torch.Tensor:
        ct = torch.clamp(acescct, 0.0, 1.0)
        return torch.where(ct > self.Y_BREAK, torch.pow(2.0, ct * self.LOG_SCALE - self.LOG_OFFSET), (ct - self.B) / self.A)


# Linear AP1 (ACEScg) <-> Rec.709 primaries with a Bradford CAT: colour-science values, as in the LTX-2.5 reference pipeline.
_AP1_TO_REC709 = ((1.7050509452819824, -0.6217921376228333, -0.08325887471437454), (-0.13025641441345215, 1.1408047676086426, -0.010548318736255169), (-0.02400335669517517, -0.1289689689874649, 1.1529723405838013))
_REC709_TO_AP1 = ((0.6130974292755127, 0.33952316641807556, 0.04737945273518562), (0.07019372284412384, 0.9163538813591003, 0.013452397659420967), (0.0206155925989151, 0.10956976562738419, 0.8698146343231201))


def _apply_primaries(frames: torch.Tensor, matrix: tuple, channel_dim: int) -> torch.Tensor:
    matrix = torch.tensor(matrix, dtype=frames.dtype, device=frames.device)
    return torch.tensordot(matrix, frames.movedim(channel_dim, 0), dims=1).movedim(0, channel_dim)


def srgb_eotf(srgb: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(srgb, 0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def hdr_linear_to_vae_range(frames: torch.Tensor, *, transform: str = "logc3", channel_dim: int) -> torch.Tensor:
    """Scene-linear Rec.709 frames to the [-1, 1] VAE range of the HDR working space."""
    frames = frames.to(dtype=torch.float32)
    if transform == "logc3":
        return LogC3().compress(frames).mul_(2.0).sub_(1.0)
    if transform == "acescct":
        return ACEScct().compress(_apply_primaries(frames, _REC709_TO_AP1, channel_dim)).mul_(2.0).sub_(1.0)
    raise ValueError(f"Unsupported HDR transform: {transform}")


def sdr_to_vae_range(frames: torch.Tensor, *, transform: str, channel_dim: int) -> torch.Tensor:
    """sRGB-encoded SDR frames in [-1, 1] to the [-1, 1] VAE range of the HDR working space."""
    frames = frames.to(dtype=torch.float32).add(1.0).mul_(0.5)
    if transform == "logc3":
        return LogC3().compress_ldr(frames).mul_(2.0).sub_(1.0)
    if transform == "acescct":
        return hdr_linear_to_vae_range(srgb_eotf(frames), transform=transform, channel_dim=channel_dim)
    raise ValueError(f"Unsupported HDR transform: {transform}")


def vae_range_to_hdr_linear(frames: torch.Tensor, *, transform: str = "logc3", channel_dim: int) -> torch.Tensor:
    """[-1, 1] VAE output of the HDR working space to scene-linear Rec.709 frames."""
    frames = frames.to(dtype=torch.float32).add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    if transform == "logc3":
        return LogC3().decompress(frames)
    if transform == "acescct":
        return _apply_primaries(ACEScct().decompress(frames), _AP1_TO_REC709, channel_dim).clamp_(min=0.0)
    raise ValueError(f"Unsupported HDR transform: {transform}")


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    linear = torch.clamp(linear, 0.0, 1.0)
    low = linear * 12.92
    high = 1.055 * torch.pow(linear, 1.0 / 2.4) - 0.055
    return torch.where(linear <= 0.0031308, low, high).clamp_(0.0, 1.0)


def tonemap_hdr_tensor_to_uint8(video: torch.Tensor, *, exposure: float = 0.0) -> torch.Tensor:
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected [C,F,H,W] HDR tensor, got {tuple(video.shape)}.")
    scale = float(2.0 ** float(exposure))
    srgb = linear_to_srgb(video.to(dtype=torch.float32).mul(scale))
    return srgb.mul(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)


def iter_video_chunks(video: torch.Tensor | Iterable[torch.Tensor]):
    if torch.is_tensor(video):
        yield video
        return
    for chunk in video:
        if chunk is not None:
            yield chunk


def iter_hdr_gbrpf32_frames(video: torch.Tensor | Iterable[torch.Tensor]):
    for chunk in iter_video_chunks(video):
        if chunk is None:
            continue
        if chunk.ndim == 5 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 4:
            raise ValueError(f"Expected [C,F,H,W] HDR tensor, got {tuple(chunk.shape)}.")
        frames = chunk.detach().cpu()
        for frame in frames.permute(1, 0, 2, 3):
            frame = frame.to(dtype=torch.float32)
            yield frame[[1, 2, 0]].contiguous().numpy().astype(np.float32, copy=False).tobytes()


def write_hdr_exr_frames(
    video: torch.Tensor | Iterable[torch.Tensor],
    output_dir: str | os.PathLike[str],
    *,
    start_index: int = 0,
    exr_half: bool = True,
) -> int:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if exr_half and hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_HALF"):
        params = [int(cv2.IMWRITE_EXR_TYPE), int(cv2.IMWRITE_EXR_TYPE_HALF)]
    frame_count = 0
    for chunk in iter_video_chunks(video):
        if chunk.ndim == 5 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 4:
            raise ValueError(f"Expected [C,F,H,W] HDR tensor, got {tuple(chunk.shape)}.")
        frames = chunk.detach().cpu().permute(1, 2, 3, 0)
        for frame in frames:
            rgb = frame.to(dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            path = os.path.join(os.fspath(output_dir), f"frame_{int(start_index) + frame_count:06d}.exr")
            if not cv2.imwrite(path, bgr, params):
                raise RuntimeError(f"Failed to write HDR EXR frame: {path}")
            frame_count += 1
    return frame_count


def read_hdr_exr_frames(
    output_dir: str | os.PathLike[str],
    *,
    start_index: int,
    frame_count: int,
) -> torch.Tensor | None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    frames = []
    for idx in range(int(start_index), int(start_index) + int(frame_count)):
        path = os.path.join(os.fspath(output_dir), f"frame_{idx:06d}.exr")
        if not os.path.isfile(path):
            return None
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            return None
        rgb = np.ascontiguousarray(bgr[..., ::-1]).astype(np.float32, copy=False)
        frames.append(torch.from_numpy(rgb))
    if not frames:
        return None
    return torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous()
