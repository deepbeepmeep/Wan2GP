"""Dependency-light preview transport/scheduler benchmark for the Tiny VAE subsystem.

Run the real decoder/generation matrix from the implementation plan on the
target WanGP runtime; this script isolates the bounded preview transport.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.preview.encoding import encode_preview
from shared.preview.scheduler import CaptureScheduler
from shared.preview.types import PreviewContext, PreviewOptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--preview-fps", type=int, choices=(2, 4, 8, 16), default=16)
    args = parser.parse_args()

    frames = [Image.new("RGB", (args.width, args.height), (index * 13 % 256, 64, 128)) for index in range(args.frames)]
    context = PreviewContext("benchmark", "benchmark", 1, "ltx2_22B", "ltx2_22B", "taeltx2_3", 1, args.steps, fps=8)
    options = PreviewOptions(max_edge=min(1024, max(128, args.width)), preview_fps=args.preview_fps)
    scheduler = CaptureScheduler(args.steps)
    capture_steps = [step for step in range(1, args.steps + 1) if scheduler.should_capture(step)]
    elapsed = []
    media = None
    for _ in range(max(1, args.repeats)):
        started = time.perf_counter()
        media = encode_preview(frames, context, options, decode_ms=0.0)
        elapsed.append((time.perf_counter() - started) * 1000)
    print(json.dumps({
        "encoder_ms_mean": round(sum(elapsed) / len(elapsed), 3),
        "encoder_ms_min": round(min(elapsed), 3),
        "encoder_ms_max": round(max(elapsed), 3),
        "media_bytes": len(media.data) if media else 0,
        "mime_type": media.mime_type if media else None,
        "media_kind": media.media_kind if media else None,
        "encoded_frames": media.frame_count if media else 0,
        "capture_steps": capture_steps,
        "pending_capacity": 1,
    }, indent=2))


if __name__ == "__main__":
    main()
