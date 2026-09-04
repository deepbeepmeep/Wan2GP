from __future__ import annotations

import collections
import argparse
import json
import math
import os
import re
import struct
import subprocess
import threading
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch


RUNTIME = Path(__file__).resolve().parents[2] / "dlss5"
HOST_DIR = RUNTIME / "host"
DLSS_DIR = RUNTIME / "dlss"
DLSSG_DIR = RUNTIME / "dlssg"
USE_DEPTH_GUIDE = True
DEPTH_MODEL_VARIANT = "vitl"
LEGACY_NR_WORKER = HOST_DIR / "nvngx.dll"
DEPTH_NR_WORKER = HOST_DIR / "nr-depth-worker.exe"
NR_WORKER = DEPTH_NR_WORKER if USE_DEPTH_GUIDE else LEGACY_NR_WORKER
DLSSG_WORKER = DLSSG_DIR / "dlssg-worker.exe"

NR_FILES = (
    HOST_DIR / "dxgi.dll",
    HOST_DIR / "renodx-dlss5.addon64",
    HOST_DIR / "nvngx_dlssnr.dll",
    DLSS_DIR / "nvngx_dlss.dll",
    NR_WORKER,
)
DLSSG_FILES = (DLSSG_DIR / "nvngx_dlssg.dll", DLSSG_WORKER)

NR_MODES = {
    1.0: ("DLAA", 5),
    1.5: ("Quality", 2),
    1.724: ("Balanced", 1),
    2.0: ("Performance", 0),
    3.0: ("Ultra Performance", 3),
}

DEPTH_RESOLUTION_DIVISORS = {"full": 1, "half": 2, "quarter": 4}
MOTION_VECTOR_METHODS = ("original", "raft")
RAFT_ITERATIONS = 20
_RAFT_MODELS = {}
_DEPTH_MODELS = {}


def configure_depth_estimator(server_config):
    global DEPTH_MODEL_VARIANT
    DEPTH_MODEL_VARIANT = server_config.get("depth_anything_v2_variant", "vitl")


def _raft_model(device: torch.device):
    key = str(device)
    if key not in _RAFT_MODELS:
        from preprocessing.raft.raft import RAFT
        from shared.utils import files_locator as fl

        model = RAFT(argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False))
        weights = torch.load(fl.locate_file("flow/raft-things.pth"), map_location="cpu", weights_only=True)
        model.load_state_dict({name.removeprefix("module."): value for name, value in weights.items()})
        _RAFT_MODELS[key] = model.to(device).eval()
    return _RAFT_MODELS[key]


def release_flow_model():
    for annotator in _DEPTH_MODELS.values():
        if hasattr(annotator, "close"):
            annotator.close()
        else:
            annotator.model.to("cpu")
    _DEPTH_MODELS.clear()
    _RAFT_MODELS.clear()


def _depth_model(device: torch.device):
    variant = DEPTH_MODEL_VARIANT
    key = (variant, str(device))
    if key not in _DEPTH_MODELS:
        from shared.utils import files_locator as fl

        if variant == "da3_metric_large":
            from preprocessing.depth_anything_v3.depth import DepthV3VideoAnnotator

            _DEPTH_MODELS[key] = DepthV3VideoAnnotator({"PRETRAINED_MODEL": fl.locate_file("depth/depth_anything_v3_metric_large_bf16.safetensors"), "MODEL_NAME": "da3metric-large", "PROCESS_RES": 0, "CHUNK_SIZE": 1, "CHUNK_OVERLAP": 8}, device=device)
        else:
            from preprocessing.depth_anything_v2.depth import DepthV2Annotator

            filename = f"depth/depth_anything_v2_{variant}.pth"
            _DEPTH_MODELS[key] = DepthV2Annotator({"PRETRAINED_MODEL": fl.locate_file(filename), "MODEL_VARIANT": variant}, device=device)
    return _DEPTH_MODELS[key]


def _missing(files: tuple[Path, ...]) -> list[Path]:
    return [path for path in files if not path.is_file()]


@lru_cache(maxsize=1)
def dlssg_capabilities() -> dict:
    if _missing(DLSSG_FILES) or os.name != "nt":
        return {}
    try:
        result = subprocess.run([str(DLSSG_WORKER), "--probe"], cwd=str(DLSSG_DIR), capture_output=True, text=True, timeout=15, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        capabilities = json.loads(result.stdout.strip())
        return capabilities if result.returncode == 0 and capabilities.get("available") else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def _gpu_series() -> int:
    if os.name != "nt":
        return 0
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return 0
    series = [int(match.group(1)) for match in re.finditer(r"GeForce\s+RTX\s+(\d{2})\d{2}", result.stdout, re.IGNORECASE)]
    return max(series, default=0)


def is_rtx_50_series() -> bool:
    return _gpu_series() == 50


@lru_cache(maxsize=1)
def _hags_enabled() -> bool:
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers") as key:
            value, _kind = winreg.QueryValueEx(key, "HwSchMode")
        return int(value) == 2
    except (OSError, ValueError):
        return False


def unavailable_reason(*, temporal: bool) -> str:
    missing = _missing(DLSSG_FILES if temporal else NR_FILES)
    if missing:
        return f"missing {missing[0].name}"
    if os.name != "nt":
        return "Windows 11 required"
    series = _gpu_series()
    minimum = 40 if temporal else 30
    if series < minimum:
        return f"RTX {minimum}+ required"
    if temporal and not _hags_enabled():
        return "HAGS disabled"
    return ""


def require_runtime(*, temporal: bool) -> None:
    reason = unavailable_reason(temporal=temporal)
    if reason:
        raise RuntimeError(f"DLSS {'Frame Generation' if temporal else 'Neural Rendering'} is unavailable: {reason}. See docs/DLSS5.md.")


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise RuntimeError(f"DLSS worker closed after {len(data)} of {size} output bytes")
        data.extend(block)
    return bytes(data)


def _read_array(stream, shape: tuple[int, ...], output: np.ndarray | None = None) -> np.ndarray:
    output = np.empty(shape, dtype=np.uint8) if output is None else output
    view = memoryview(output).cast("B")
    offset = 0
    while offset < len(view):
        count = stream.readinto(view[offset:])
        if not count:
            raise RuntimeError(f"DLSS worker closed after {offset} of {len(view)} output bytes")
        offset += count
    return output


class Worker:
    def __init__(self, command: list[str], cwd: Path):
        self.logs: collections.deque[str] = collections.deque(maxlen=100)
        self.process = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._log_thread = threading.Thread(target=self._read_logs, daemon=True)
        self._log_thread.start()

    def _read_logs(self):
        assert self.process.stderr is not None
        for raw in iter(self.process.stderr.readline, b""):
            self.logs.append(raw.decode("utf-8", "replace").rstrip())

    def error(self) -> str:
        return "\n".join(self.logs)

    def close(self, *, abort: bool = False):
        process = self.process
        if process.poll() is None:
            try:
                if abort:
                    process.terminate()
                elif process.stdin and not process.stdin.closed:
                    process.stdin.close()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
        self._log_thread.join(timeout=1)
        if not abort and process.returncode:
            raise RuntimeError(f"DLSS worker exited with code {process.returncode}.\n{self.error()}")


class FlowGuides:
    def __init__(self, width: int, height: int, motion_vector: str):
        self.width, self.height = width, height
        self.use_raft = motion_vector == "raft"
        scale = min(1.0, 640 / width)
        self.flow_width = max(64, round(width * scale / 2) * 2)
        self.flow_height = max(64, round(height * scale / 2) * 2)
        self.previous = None
        self.zero = np.zeros((height, width, 2), dtype=np.float16)
        if self.use_raft:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.flow = _raft_model(self.device)
        else:
            self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.flow.setUseSpatialPropagation(True)
            self.flow.setFinestScale(1)

    def _raft(self, current: np.ndarray) -> np.ndarray:
        from preprocessing.raft.utils.utils import InputPadder

        current_tensor = torch.from_numpy(current[..., :3]).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        previous_tensor = torch.from_numpy(self.previous[..., :3]).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        padder = InputPadder(current_tensor.shape)
        current_tensor, previous_tensor = padder.pad(current_tensor, previous_tensor)
        with torch.inference_mode():
            _low, motion = self.flow(current_tensor, previous_tensor, iters=RAFT_ITERATIONS, test_mode=True)
        return padder.unpad(motion)[0].permute(1, 2, 0).cpu().numpy()

    def process(self, rgba: np.ndarray) -> tuple[np.ndarray, bool]:
        resized = cv2.resize(rgba, (self.flow_width, self.flow_height), interpolation=cv2.INTER_AREA)
        current = resized if self.use_raft else cv2.cvtColor(resized, cv2.COLOR_RGBA2GRAY)
        if self.previous is None:
            motion, reset = self.zero, True
        else:
            score = float(np.mean(cv2.absdiff(current, self.previous))) / 255.0
            duplicate = score < 0.0005
            reset = score > 0.24
            if reset or duplicate:
                motion = self.zero
            else:
                low_resolution_motion = self._raft(current) if self.use_raft else self.flow.calc(current, self.previous, None)
                motion = cv2.resize(low_resolution_motion, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                motion[..., 0] *= self.width / self.flow_width
                motion[..., 1] *= self.height / self.flow_height
                finite = np.isfinite(motion).all(axis=2)
                motion[~finite] = 0
                reset = float(np.mean(finite)) < 0.98
                if reset:
                    motion = self.zero
                motion = np.ascontiguousarray(motion, dtype=np.float16)
        self.previous = current
        return motion, reset


class DepthGuides:
    def __init__(self, width: int, height: int, depth_resolution: str):
        self.width, self.height = width, height
        divisor = DEPTH_RESOLUTION_DIVISORS[depth_resolution]
        self.inference_width = width // divisor
        self.inference_height = height // divisor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.variant = DEPTH_MODEL_VARIANT
        self.estimator = _depth_model(self.device)
        self.low = self.high = None

    def process(self, rgba: np.ndarray, reset: bool) -> np.ndarray:
        inference_frame = rgba if (self.inference_width, self.inference_height) == (self.width, self.height) else cv2.resize(rgba, (self.inference_width, self.inference_height), interpolation=cv2.INTER_AREA)
        if self.variant == "da3_metric_large":
            from preprocessing.depth_anything_v3.depth import _run_da3_depth_prediction

            metric_depth = _run_da3_depth_prediction(self.estimator.model, inference_frame[None, ..., :3], self.inference_width, chunk_size=1)[0]
            disparity = 1.0 / np.maximum(metric_depth, 1e-6)
            del metric_depth
        else:
            bgr = cv2.cvtColor(inference_frame, cv2.COLOR_RGBA2BGR)
            disparity = self.estimator.model.infer_image(bgr, input_size=min(self.inference_width, self.inference_height))
            del bgr
        del inference_frame
        low, high = (float(value) for value in np.nanpercentile(disparity, (2, 98)))
        if reset or self.low is None:
            self.low, self.high = low, high
        else:
            self.low = 0.95 * self.low + 0.05 * low
            self.high = 0.95 * self.high + 0.05 * high
        disparity -= self.low
        disparity /= max(self.high - self.low, 1e-6)
        np.clip(disparity, 0, 1, out=disparity)
        depth = disparity
        if disparity.shape != (self.height, self.width):
            depth = cv2.resize(disparity, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            del disparity
        return np.ascontiguousarray(depth, dtype=np.float32)


def _sample_info(sample: torch.Tensor) -> tuple[torch.dtype, torch.device, int, int, int, int]:
    if sample.ndim != 4 or sample.shape[0] not in (3, 4):
        raise ValueError("DLSS expects a [3|4, frames, height, width] tensor")
    channels, frames, height, width = sample.shape
    if frames < 1:
        raise ValueError("DLSS requires at least one frame")
    return sample.dtype, sample.device, channels, frames, height, width


def _frame_to_rgba(sample: torch.Tensor, index: int) -> np.ndarray:
    channels = sample.shape[0]
    frame = sample[:, index].detach()
    if frame.dtype != torch.uint8:
        frame = frame.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
    data = frame.to(device="cpu").permute(1, 2, 0).contiguous().numpy()
    if channels == 3:
        rgba = np.empty((*data.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3], rgba[..., 3] = data, 255
        return rgba
    return data


def _from_rgba_frames(data: np.ndarray, dtype: torch.dtype, device: torch.device, channels: int) -> torch.Tensor:
    output = torch.from_numpy(data).permute(3, 0, 1, 2)
    if channels == 3:
        output = output[:3]
    if device.type != "cpu":
        output = output.to(device=device)
    if dtype != torch.uint8:
        output = output.to(dtype=dtype).div_(127.5).sub_(1)
    return output


class NeuralRenderingSession(Worker):
    VIDEO_MAGIC = 0x34563544
    SETUP_MAGIC = 0x34505553
    DEPTH_VIDEO_MAGIC = 0x3144574E
    DEPTH_SETUP_MAGIC = 0x3152574E
    FRAME_MAGIC = 0x314D5246
    OUT_MAGIC = 0x3154554F

    def __init__(self, width: int, height: int, frames: int, scale: float, intensity: float = 1.0):
        output_width = max(2, math.floor(width * scale / 2 + 0.5) * 2)
        output_height = max(2, math.floor(height * scale / 2 + 0.5) * 2)
        if max(output_width, output_height) > 7680 or min(output_width, output_height) > 4320:
            raise ValueError(f"DLSS output {output_width}x{output_height} exceeds the 7680x4320 limit")
        intensity = max(0.0, min(2.0, float(intensity)))
        command = [str(NR_WORKER), "--nr-intensity", f"{intensity:.4f}", "--wangp-video"] if USE_DEPTH_GUIDE else [str(NR_WORKER), "--video"]
        super().__init__(command, HOST_DIR)
        self.output_width, self.output_height = output_width, output_height
        assert self.process.stdin is not None and self.process.stdout is not None
        if USE_DEPTH_GUIDE:
            self.process.stdin.write(struct.pack("<10I", self.DEPTH_VIDEO_MAGIC, width, height, output_width, output_height, frames, 0, 60, 0, 0))
        else:
            _name, perf_quality = NR_MODES[scale]
            self.process.stdin.write(struct.pack("<14I4f", self.VIDEO_MAGIC, width, height, output_width, output_height, 0, frames, perf_quality, 0, 0, 0, 0, 0, 0, intensity, 1.0, 1.0, -1.0))
        self.process.stdin.flush()
        if USE_DEPTH_GUIDE:
            response = struct.unpack("<6I", _read_exact(self.process.stdout, struct.calcsize("<6I")))
            if response[0] != self.DEPTH_SETUP_MAGIC or response[1]:
                self.close(abort=True)
                raise RuntimeError(f"DLSS Neural Rendering depth setup failed (NGX 0x{response[1]:08X}).\n{self.error()}")
            self.render_width, self.render_height = response[2], response[3]
            negotiated_output = response[4], response[5]
        else:
            response = struct.unpack("<12I", _read_exact(self.process.stdout, struct.calcsize("<12I")))
            if response[0] != self.SETUP_MAGIC or not response[1]:
                self.close(abort=True)
                raise RuntimeError(f"DLSS Neural Rendering setup failed (NGX 0x{response[2]:08X}).\n{self.error()}")
            self.render_width, self.render_height = response[3], response[4]
            negotiated_output = response[5], response[6]
        if negotiated_output != (output_width, output_height):
            self.close(abort=True)
            raise RuntimeError("DLSS worker negotiated unexpected output dimensions")

    def process_frame(self, index: int, rgba: np.ndarray, motion: np.ndarray, reset: bool, output: np.ndarray | None = None, *, depth: np.ndarray | None = None) -> np.ndarray:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(struct.pack("<4Iq", self.FRAME_MAGIC, index, int(reset), 0, index))
        self.process.stdin.write(memoryview(np.ascontiguousarray(rgba, dtype=np.uint8)).cast("B"))
        self.process.stdin.write(memoryview(np.ascontiguousarray(motion, dtype=np.float16)).cast("B"))
        if USE_DEPTH_GUIDE:
            if depth is None:
                raise ValueError("DLSS Neural Rendering depth mode requires a depth guide")
            self.process.stdin.write(memoryview(np.ascontiguousarray(depth, dtype=np.float32)).cast("B"))
        self.process.stdin.flush()
        magic, out_index, ok, byte_count, ngx_result, _pts = struct.unpack("<5Iq", _read_exact(self.process.stdout, struct.calcsize("<5Iq")))
        expected = self.output_width * self.output_height * 4
        if magic != self.OUT_MAGIC or out_index != index or not ok or byte_count != expected or ngx_result != 1:
            raise RuntimeError(f"DLSS Neural Rendering failed on frame {index} (NGX 0x{ngx_result:08X}).\n{self.error()}")
        return _read_array(self.process.stdout, (self.output_height, self.output_width, 4), output)


def neural_render(sample: torch.Tensor, scale: float, *, still_image: bool, depth_resolution: str, motion_vector: str, intensity: float = 1.0, abort_callback=None, progress_callback=None) -> torch.Tensor | None:
    require_runtime(temporal=False)
    dtype, device, channels, frame_count, height, width = _sample_info(sample)
    session = NeuralRenderingSession(width, height, frame_count, scale, intensity)
    output = np.empty((frame_count, session.output_height, session.output_width, 4), dtype=np.uint8)
    try:
        flow_guides = None if still_image else FlowGuides(session.render_width, session.render_height, motion_vector)
        zero_motion = np.zeros((session.render_height, session.render_width, 2), dtype=np.float16) if still_image else None
        depth_guides = DepthGuides(session.render_width, session.render_height, depth_resolution) if USE_DEPTH_GUIDE else None
        for index in range(frame_count):
            if abort_callback is not None and abort_callback():
                session.close(abort=True)
                return None
            frame = _frame_to_rgba(sample, index)
            render_frame = frame if (width, height) == (session.render_width, session.render_height) else cv2.resize(frame, (session.render_width, session.render_height), interpolation=cv2.INTER_LANCZOS4)
            motion, reset = (zero_motion, True) if still_image else flow_guides.process(render_frame)
            depth = depth_guides.process(render_frame, reset) if depth_guides is not None else None
            processed = session.process_frame(index, render_frame, motion, reset, output[index], depth=depth)
            if channels == 4:
                processed[..., 3] = cv2.resize(frame[..., 3], (session.output_width, session.output_height), interpolation=cv2.INTER_LANCZOS4)
            if progress_callback is not None:
                progress_callback("DLSS 5 Neural Rendering", index + 1, frame_count)
            del frame, render_frame, motion, depth, processed
        session.close()
    except Exception:
        session.close(abort=True)
        raise
    return _from_rgba_frames(output, dtype, device, channels)


class FrameGenerationSession(Worker):
    SETUP_MAGIC = 0x31534746
    SETUP_OUT_MAGIC = 0x31524746
    FRAME_MAGIC = 0x31464746
    FRAME_OUT_MAGIC = 0x314F4746

    def __init__(self, width: int, height: int, frames: int, generated_count: int):
        super().__init__([str(DLSSG_WORKER), "--serve"], DLSSG_DIR)
        self.width, self.height = width, height
        self.generated_count = generated_count
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(struct.pack("<5I", self.SETUP_MAGIC, width, height, max(1, frames), generated_count))
        self.process.stdin.flush()
        magic, status, maximum, _reserved = struct.unpack("<4I", _read_exact(self.process.stdout, struct.calcsize("<4I")))
        if magic != self.SETUP_OUT_MAGIC or status or maximum < generated_count:
            self.close(abort=True)
            raise RuntimeError(f"DLSS Frame Generation {generated_count + 1}x setup failed (status {status}, runtime maximum {maximum + 1}x).\n{self.error()}")

    def process_frame(self, index: int, rgba: np.ndarray, motion: np.ndarray, timestamp: Fraction, reset: bool, output: np.ndarray | None = None) -> int:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(struct.pack("<4I2q", self.FRAME_MAGIC, index, int(reset), 0, timestamp.numerator, timestamp.denominator))
        self.process.stdin.write(memoryview(np.ascontiguousarray(rgba, dtype=np.uint8)).cast("B"))
        self.process.stdin.write(memoryview(np.ascontiguousarray(motion, dtype=np.float16)).cast("B"))
        self.process.stdin.flush()
        magic, status, generated, disabled = struct.unpack("<4I", _read_exact(self.process.stdout, struct.calcsize("<4I")))
        if magic != self.FRAME_OUT_MAGIC or status:
            raise RuntimeError(f"DLSS Frame Generation failed on frame {index} (status {status}).\n{self.error()}")
        if disabled or generated == 0:
            return 0
        if generated != self.generated_count:
            raise RuntimeError(f"DLSS Frame Generation returned {generated} frames when {self.generated_count} were requested")
        if output is None:
            raise RuntimeError("DLSS Frame Generation returned a frame without a destination buffer")
        for index in range(generated):
            _read_array(self.process.stdout, (self.height, self.width, 4), output[index])
        return generated


def _interpolate(frame_count: int, width: int, height: int, frame_getter, fps: float, scale: int, motion_vector: str, abort_callback=None, progress_callback=None) -> np.ndarray | None:
    generated_count = scale - 1
    session = FrameGenerationSession(width, height, frame_count, generated_count)
    guides = FlowGuides(width, height, motion_vector)
    output = np.empty(((frame_count - 1) * scale + 1, height, width, 4), dtype=np.uint8)
    try:
        for index in range(frame_count):
            if abort_callback is not None and abort_callback():
                session.close(abort=True)
                return None
            frame = frame_getter(index)
            motion, reset = guides.process(frame)
            target_index = index * scale
            output[target_index] = frame
            if index == 0:
                session.process_frame(index, frame, motion, Fraction(index, 1) / Fraction(str(fps)), reset)
            else:
                generated_output = output[target_index - generated_count:target_index]
                generated = session.process_frame(index, frame, motion, Fraction(index, 1) / Fraction(str(fps)), reset, generated_output)
                if reset or not generated:
                    generated_output[:] = output[target_index - scale]
            if progress_callback is not None:
                progress_callback("DLSS Frame Generation", index + 1, frame_count)
            del frame, motion
        session.close()
    except Exception:
        session.close(abort=True)
        raise
    return output


def frame_generate(sample: torch.Tensor, previous_last_frame: torch.Tensor | None, fps: float, scale: int, *, motion_vector: str, abort_callback=None, progress_callback=None) -> tuple[torch.Tensor | None, torch.Tensor, float]:
    require_runtime(temporal=True)
    dtype, device, channels, frame_count, height, width = _sample_info(sample)
    next_previous = sample[:, -1:].clone()
    has_previous = previous_last_frame is not None
    input_count = frame_count + int(has_previous)

    def input_frame(index):
        return _frame_to_rgba(previous_last_frame if has_previous and index == 0 else sample, index - 1 if has_previous else index)

    output = _interpolate(input_count, width, height, input_frame, fps, scale, motion_vector, abort_callback=abort_callback, progress_callback=progress_callback)
    if output is None:
        return None, next_previous, fps * scale
    if has_previous:
        output = output[1:]
    result = _from_rgba_frames(output, dtype, device, channels)
    return result, next_previous, fps * scale
