from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from scripts.dual_run_compare.status import SECTION_MISSING_EVIDENCE, SECTION_NOT_APPLICABLE


IMAGE_METRIC_KEYS: tuple[str, ...] = (
    "image_phash_normalized_hamming",
    "image_ssim",
    "image_pixel_dimensions",
    "image_format_container",
)

VIDEO_METRIC_KEYS: tuple[str, ...] = (
    "video_frame_count",
    "video_phash_mean",
    "video_phash_p95",
    "video_duration_ms",
    "video_fps",
    "video_audio_duration_ms",
)


def _missing_result(metric_keys: Iterable[str], *, media_kind: str, path: Path, role: str) -> dict[str, Any]:
    missing = [
        {
            "key": metric_key,
            "status": SECTION_MISSING_EVIDENCE,
            "media_kind": media_kind,
            "path": str(path),
            "role": role,
            "reason": "artifact does not exist",
        }
        for metric_key in metric_keys
    ]
    return {
        "status": SECTION_MISSING_EVIDENCE,
        "observations": {},
        "required_metric_keys": [item["key"] for item in missing],
        "missing_evidence": missing,
    }


def _phash_array_from_pil(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    dct = cv2.dct(pixels)
    low_frequency = dct[:8, :8]
    median = np.median(low_frequency[1:, 1:])
    return low_frequency > median


def _normalized_hamming(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"hash shape mismatch: {left.shape} != {right.shape}")
    return float(np.count_nonzero(left != right) / left.size)


def _image_format(path: Path) -> str:
    with Image.open(path) as image:
        return (image.format or path.suffix.lstrip(".")).lower()


def _image_dimensions(path: Path) -> str:
    with Image.open(path) as image:
        return f"{image.width}x{image.height}"


def _image_ssim(reference_path: Path, candidate_path: Path) -> float:
    with Image.open(reference_path) as reference_image, Image.open(candidate_path) as candidate_image:
        reference = np.asarray(reference_image.convert("L"), dtype=np.float32)
        candidate = np.asarray(candidate_image.convert("L").resize(reference_image.size), dtype=np.float32)
    return float(structural_similarity(reference, candidate, data_range=255))


def compare_image_artifacts(reference_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    reference = Path(reference_path)
    candidate = Path(candidate_path)
    if not reference.is_file():
        return _missing_result(IMAGE_METRIC_KEYS, media_kind="image", path=reference, role="reference")
    if not candidate.is_file():
        return _missing_result(IMAGE_METRIC_KEYS, media_kind="image", path=candidate, role="candidate")

    with Image.open(reference) as reference_image, Image.open(candidate) as candidate_image:
        reference_hash = _phash_array_from_pil(reference_image)
        candidate_hash = _phash_array_from_pil(candidate_image)

    observations = {
        "image_phash_normalized_hamming": _normalized_hamming(reference_hash, candidate_hash),
        "image_ssim": _image_ssim(reference, candidate),
        "image_pixel_dimensions": {
            "observed": _image_dimensions(candidate),
            "expected": _image_dimensions(reference),
        },
        "image_format_container": {
            "observed": _image_format(candidate),
            "expected": _image_format(reference),
        },
    }
    return {
        "status": "observed",
        "observations": observations,
        "required_metric_keys": list(IMAGE_METRIC_KEYS),
        "missing_evidence": [],
    }


def _frame_hash(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _phash_array_from_pil(Image.fromarray(rgb))


def _sample_frame_hashes(path: Path, max_samples: int = 8) -> tuple[list[np.ndarray], int, float, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video artifact: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_ms = (frame_count / fps * 1000.0) if fps else 0.0
        if frame_count <= 0:
            return [], frame_count, fps, duration_ms
        sample_count = min(max_samples, frame_count)
        indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        hashes: list[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                hashes.append(_frame_hash(frame))
        return hashes, frame_count, fps, duration_ms
    finally:
        capture.release()


def _video_phash_stats(reference_hashes: list[np.ndarray], candidate_hashes: list[np.ndarray]) -> tuple[float, float]:
    pair_count = min(len(reference_hashes), len(candidate_hashes))
    if pair_count == 0:
        return math.nan, math.nan
    distances = [
        _normalized_hamming(reference_hashes[index], candidate_hashes[index])
        for index in range(pair_count)
    ]
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def _audio_duration_ms(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return {
            "status": SECTION_NOT_APPLICABLE,
            "observed": None,
            "reason": "audio stream absent or ffprobe unavailable",
        }
    try:
        return {"observed": float(output.splitlines()[0]) * 1000.0}
    except ValueError:
        return {
            "status": SECTION_NOT_APPLICABLE,
            "observed": None,
            "reason": "audio stream duration unavailable",
        }


def _video_format(path: Path) -> str:
    return path.suffix.lstrip(".").lower()


def compare_video_artifacts(reference_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    reference = Path(reference_path)
    candidate = Path(candidate_path)
    if not reference.is_file():
        return _missing_result(VIDEO_METRIC_KEYS, media_kind="video", path=reference, role="reference")
    if not candidate.is_file():
        return _missing_result(VIDEO_METRIC_KEYS, media_kind="video", path=candidate, role="candidate")

    reference_hashes, reference_frame_count, reference_fps, reference_duration_ms = _sample_frame_hashes(reference)
    candidate_hashes, candidate_frame_count, candidate_fps, candidate_duration_ms = _sample_frame_hashes(candidate)
    phash_mean, phash_p95 = _video_phash_stats(reference_hashes, candidate_hashes)
    audio_duration = _audio_duration_ms(candidate)
    reference_audio_duration = _audio_duration_ms(reference)

    observations: dict[str, Any] = {
        "video_frame_count": {
            "observed": candidate_frame_count,
            "expected": reference_frame_count,
        },
        "video_phash_mean": phash_mean,
        "video_phash_p95": phash_p95,
        "video_duration_ms": {
            "observed": candidate_duration_ms,
            "expected": reference_duration_ms,
        },
        "video_fps": {
            "observed": candidate_fps,
            "expected": reference_fps,
        },
    }
    if audio_duration.get("status") != SECTION_NOT_APPLICABLE:
        observations["video_audio_duration_ms"] = {
            "observed": audio_duration["observed"],
            "expected": reference_audio_duration.get("observed"),
        }

    return {
        "status": "observed",
        "observations": observations,
        "required_metric_keys": [
            key for key in VIDEO_METRIC_KEYS if key in observations
        ],
        "optional_metric_results": {
            "video_audio_duration_ms": audio_duration,
            "video_format_container": {
                "observed": _video_format(candidate),
                "expected": _video_format(reference),
            },
        },
        "missing_evidence": [],
    }
