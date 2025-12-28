import argparse
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def _frame_stats(frame: np.ndarray) -> tuple[float, float]:
    if frame is None:
        return 0.0, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray)), float(np.max(gray) - np.min(gray))


def _check_image(path: Path, min_std: float, min_range: float) -> tuple[bool, str]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "unable to decode image"
    std, rng = _frame_stats(img)
    if std < min_std or rng < min_range:
        return False, f"low variance (std={std:.2f}, range={rng:.2f})"
    return True, f"ok (std={std:.2f}, range={rng:.2f})"


def _sample_video_frames(cap: cv2.VideoCapture, indices: list[int]) -> list[np.ndarray]:
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def _check_video(path: Path, min_std: float, min_range: float) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "unable to open video"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return False, "no frames detected"
    indices = [0, frame_count // 2, max(frame_count - 1, 0)]
    frames = _sample_video_frames(cap, indices)
    cap.release()
    if not frames:
        return False, "failed to read sample frames"
    stats = [_frame_stats(frame) for frame in frames]
    avg_std = sum(s[0] for s in stats) / len(stats)
    avg_rng = sum(s[1] for s in stats) / len(stats)
    if avg_std < min_std or avg_rng < min_range:
        return False, f"low variance (avg std={avg_std:.2f}, avg range={avg_rng:.2f})"
    return True, f"ok (avg std={avg_std:.2f}, avg range={avg_rng:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated outputs are not garbage.")
    parser.add_argument("output_dir", help="Output directory to scan.")
    parser.add_argument("--min-std", type=float, default=2.0, help="Minimum grayscale stddev.")
    parser.add_argument("--min-range", type=float, default=10.0, help="Minimum grayscale range.")
    args = parser.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        raise SystemExit(f"Output directory not found: {root}")

    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)]
    if not files:
        raise SystemExit(f"No output media files found in {root}")

    failures = []
    for path in sorted(files):
        if path.suffix.lower() in IMAGE_EXTS:
            ok, detail = _check_image(path, args.min_std, args.min_range)
        else:
            ok, detail = _check_video(path, args.min_std, args.min_range)
        status = "ok" if ok else "fail"
        print(f"[{status}] {path}: {detail}")
        if not ok:
            failures.append(path)

    if failures:
        raise SystemExit(f"{len(failures)} file(s) failed validation.")
    print(f"Validated {len(files)} file(s).")


if __name__ == "__main__":
    main()
