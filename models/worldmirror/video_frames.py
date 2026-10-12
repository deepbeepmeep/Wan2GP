# Motion-aware frame extraction of WorldMirror 2.0, ported from Tencent HY-World 2.0 (hyworld2/worldrecon/hyworldmirror/utils/
# video_utils.py, video_to_image_frames_new). Copyright (C) 2026 Tencent, licensed under the Tencent HY-WORLD 2.0 Community License
# Agreement (LICENSE in this folder). Modified by DeepBeepMeep for WanGP (2026-10-08): frames decoded with WanGP's ffmpeg helpers (virtual
# media, HDR tone mapping) instead of cv2.VideoCapture, returned in memory instead of JPEG files, no metadata files, abort checks.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import cv2
import numpy as np
from PIL import Image

from shared.utils.phase_progress import check_abort
from shared.utils.video_decode import decode_video_frame_indices_ffmpeg, decode_video_frames_ffmpeg, probe_video_stream_metadata

FLOW_EDGE, CLARITY_EDGE, SEARCH_WINDOW, DECODE_CHUNK = 320, 480, 3, 32


def _resize(frame, long_edge):
    height, width = frame.shape[:2]
    if max(height, width) <= long_edge:
        return frame
    scale = long_edge / float(max(height, width))
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _histogram(image):
    """Normalized HSV histogram (8 x 4 x 4 bins), used to skip near-duplicate views."""
    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_RGB2HSV)], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _duplicate(hist, selected):
    return any(cv2.compareHist(hist, other, cv2.HISTCMP_CORREL) > 0.999 for other in selected)


def _sparse_motion(path, fps, total):
    """Phase 1: a frame every ~0.5 s with its mean DIS optical-flow magnitude from the previous sample (320 px) and its histogram."""
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    indices = list(range(0, total, max(1, int(fps * 0.5))))
    samples, previous = [], None
    for start in range(0, len(indices), DECODE_CHUNK):
        for index, frame in zip(indices[start:start + DECODE_CHUNK], decode_video_frame_indices_ffmpeg(path, indices[start:start + DECODE_CHUNK], bridge="numpy")):
            small = _resize(frame, FLOW_EDGE)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            motion = 0.0 if previous is None else float(np.mean(np.linalg.norm(flow.calc(previous, gray, None), axis=-1)))
            samples.append({"idx": index, "motion": motion, "hist": _histogram(small)})
            previous = gray
        check_abort()
    return samples


def _adaptive_selection(samples, fps, max_frames):
    """Phase 2: a new view whenever the accumulated motion reaches total motion / max_frames (at least 5 px) or 4 s have passed, unless
    it duplicates a selected view; the last sample is checked too."""
    motions = [sample["motion"] for sample in samples[1:]]
    if not motions:
        return [samples[0]["idx"]]
    step = max(5.0, sum(motions) / max_frames)
    selected, hists, accumulated, last = [samples[0]["idx"]], [samples[0]["hist"]], 0.0, samples[0]["idx"]
    for sample in samples[1:]:
        accumulated += sample["motion"] if sample["motion"] >= 1.0 else 0.0
        if (accumulated >= step or sample["idx"] - last > 4.0 * fps) and not _duplicate(sample["hist"], hists):
            selected.append(sample["idx"])
            hists.append(sample["hist"])
            accumulated, last = 0.0, sample["idx"]
    if samples[-1]["idx"] != selected[-1] and not _duplicate(samples[-1]["hist"], hists):
        selected.append(samples[-1]["idx"])
    return sorted(set(selected))


def _frame_constraints(selected, samples, min_frames, max_frames):
    if len(selected) < min_frames:
        indices = [sample["idx"] for sample in samples]
        extras = np.linspace(0, len(indices) - 1, min_frames - len(selected) + 2)[1:-1]
        selected = sorted(set(selected + [indices[int(extra)] for extra in extras]))
    if len(selected) > max_frames:
        selected = [selected[int(round(position))] for position in np.linspace(0, len(selected) - 1, max_frames)]
    return selected


def _merge_windows(selected):
    """Search windows of +/- SEARCH_WINDOW frames around the targets, merged when they overlap: [(start, end, targets)]."""
    windows = []
    for target in sorted(selected):
        start, end = max(0, target - SEARCH_WINDOW), target + SEARCH_WINDOW
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], end, windows[-1][2] + [target])
        else:
            windows.append((start, end, [target]))
    return windows


def select_frames(path, max_frames, min_frames=1):
    """video_to_image_frames_new: up to max_frames views of a video chosen by camera motion, each replaced by the sharpest frame (variance
    of the Laplacian at 480 px) within +/- 3 frames. Returns (RGB PIL images, frame indices) in time order."""
    max_frames = int(np.clip(max_frames, 1, 64))
    min_frames = int(np.clip(min_frames, 1, max_frames))
    metadata = probe_video_stream_metadata(path)
    if metadata is None or not metadata["frame_count"]:
        raise ValueError(f"Unable to read the video {path}")
    fps, total = float(metadata.get("fps_float") or metadata["fps"] or 30.0), int(metadata["frame_count"])
    samples = _sparse_motion(path, fps, total)
    selected = _frame_constraints(_adaptive_selection(samples, fps, max_frames), samples, min_frames, max_frames)
    best = {}
    for start, end, targets in _merge_windows(selected):
        frames = decode_video_frames_ffmpeg(path, start, min(end + 1, total) - start, bridge="numpy")
        scored = [(start + offset, frame, cv2.Laplacian(cv2.cvtColor(_resize(frame, CLARITY_EDGE), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
                  for offset, frame in enumerate(frames)]
        for target in targets:
            candidates = [item for item in scored if abs(item[0] - target) <= SEARCH_WINDOW]
            if candidates:
                best[target] = max(candidates, key=lambda item: item[2])[:2]
            elif scored:
                best[target] = min(scored, key=lambda item: abs(item[0] - target))[:2]
        check_abort()
    chosen = [best[target] for target in sorted(selected) if target in best]
    return [Image.fromarray(np.ascontiguousarray(frame)) for _, frame in chosen], [index for index, _ in chosen]
