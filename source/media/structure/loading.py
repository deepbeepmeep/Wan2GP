"""Structure video frame loading and resampling."""

import numpy as np
from typing import List, Tuple

from source.core.log import generation_logger
from source.core.params.structure_guidance import normalize_structure_treatment

__all__ = [
    "load_structure_video_frames",
]


def _resample_frame_indices(video_fps: float, video_frames_count: int, max_target_frames_count: int, target_fps: float, start_target_frame: int) -> List[int]:
    """
    Calculate which frame indices to extract for FPS conversion.
    Ported from Wan2GP/shared/utils/utils.py to avoid importing wgp.py
    """
    import math

    video_frame_duration = 1 / video_fps
    target_frame_duration = 1 / target_fps

    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)
    cur_time = frame_no * video_frame_duration
    frame_ids = []

    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count:
            break
        diff = round((target_time - cur_time) / video_frame_duration, 5)
        add_frames_count = math.ceil(diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration

    if max_target_frames_count > 0:
        frame_ids = frame_ids[:max_target_frames_count]
    return frame_ids


def load_structure_video_frames(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: Tuple[int, int],
    treatment: str = "adjust",
    crop_to_fit: bool = True,
) -> List[np.ndarray]:
    """
    Load structure video frames with treatment mode.

    Args:
        structure_video_path: Path to structure video
        target_frame_count: Number of frames to load (note: loads target_frame_count+1 to get enough flows)
        target_fps: Target FPS (used for clip mode temporal sampling)
        target_resolution: (width, height) tuple
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        crop_to_fit: If True, center-crop to match target aspect ratio before resizing

    Returns:
        List of numpy uint8 arrays [H, W, C] in RGB format
    """
    from PIL import Image

    treatment = normalize_structure_treatment(
        treatment,
        context="structure video loading",
    )

    # Try decord first, then fall back to cv2 when it is unavailable.
    use_decord = False
    try:
        import decord

        decord.bridge.set_bridge('torch')
        use_decord = True
    except ImportError:
        generation_logger.debug("[STRUCTURE_VIDEO] decord not available, using cv2 fallback")

    if not use_decord:
        import cv2

    # Load N+1 frames to ensure we get N flows (RAFT produces N-1 flows for N frames)
    frames_to_load = target_frame_count + 1

    # Load video
    if use_decord:
        reader = decord.VideoReader(structure_video_path)
        video_fps = round(reader.get_avg_fps())
        video_frame_count = len(reader)
    else:
        cap = cv2.VideoCapture(structure_video_path)
        video_fps = round(cap.get(cv2.CAP_PROP_FPS))
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    generation_logger.debug(f"[STRUCTURE_VIDEO] Loading frames from structure video:")
    generation_logger.debug(f"  Video: {video_frame_count} frames @ {video_fps}fps")
    generation_logger.debug(f"  Needed: {frames_to_load} frames")
    generation_logger.debug(f"  Treatment: {treatment}")

    # Calculate frame indices based on treatment mode
    if treatment == "adjust":
        # ADJUST MODE: Stretch/compress entire video to match needed frame count
        # Linearly interpolate frame indices across the entire video
        if video_frame_count >= frames_to_load:
            # Compress: Sample evenly across video
            frame_indices = [int(i * (video_frame_count - 1) / (frames_to_load - 1)) for i in range(frames_to_load)]
            dropped = video_frame_count - frames_to_load
            generation_logger.debug(f"  Adjust mode: Your input video has {video_frame_count} frames so we'll drop {dropped} frames to compress your guide video to the {frames_to_load} frames your input images cover.")
        else:
            # Stretch: Repeat frames to reach target count
            # Use linear interpolation indices (will repeat frames)
            frame_indices = [int(i * (video_frame_count - 1) / (frames_to_load - 1)) for i in range(frames_to_load)]
            duplicates = frames_to_load - len(set(frame_indices))
            generation_logger.debug(f"  Adjust mode: Your input video has {video_frame_count} frames so we'll duplicate {duplicates} frames to stretch your guide video to the {frames_to_load} frames your input images cover.")
    else:
        # CLIP MODE: Temporal sampling based on FPS
        frame_indices = _resample_frame_indices(
            video_fps=video_fps,
            video_frames_count=video_frame_count,
            max_target_frames_count=frames_to_load,
            target_fps=target_fps,
            start_target_frame=0
        )

        if not frame_indices and video_frame_count > 0:
            # When temporal resampling yields nothing (very short clips, edge rounding),
            # seed from the first available frame so the fill loop can make progress.
            frame_indices = [0]

        # If video is too short, loop back to start
        if len(frame_indices) < frames_to_load:
            generation_logger.debug(f"  Clip mode: Video too short ({len(frame_indices)} frames < {frames_to_load} needed), looping back to start to fill remaining frames")
            while len(frame_indices) < frames_to_load:
                remaining = frames_to_load - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])
        elif video_frame_count > frames_to_load:
            ignored = video_frame_count - frames_to_load
            generation_logger.debug(f"  Clip mode: Your video will guide {frames_to_load} frames of your timeline. The last {ignored} frames of your video (frames {frames_to_load + 1}-{video_frame_count}) will be ignored.")

        generation_logger.debug(f"  Clip mode: Temporal sampling extracted {len(frame_indices)} frames")

    if not frame_indices:
        raise ValueError(f"No frames could be extracted from structure video: {structure_video_path}")

    if use_decord:
        frames = reader.get_batch(frame_indices)  # Returns torch tensors [T, H, W, C]
        generation_logger.debug(f"[STRUCTURE_VIDEO] Loaded {len(frames)} frames")
    else:
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            elif frames:
                frames.append(frames[-1].copy())
        cap.release()
        generation_logger.debug(f"[STRUCTURE_VIDEO] Loaded {len(frames)} frames via cv2 fallback")

    # Process frames to target resolution (WGP pattern from line 3826-3830)
    w, h = target_resolution
    target_aspect = w / h
    processed_frames = []

    for i, frame in enumerate(frames):
        # Convert decord/torch tensor to numpy (WGP pattern line 3826)
        if hasattr(frame, 'cpu'):
            frame_np = frame.cpu().numpy()  # [H, W, C] uint8
        else:
            frame_np = np.array(frame)

        # Ensure uint8
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        # Convert to PIL for processing (WGP pattern line 3830)
        frame_pil = Image.fromarray(frame_np)

        # Center crop to match target aspect ratio if requested
        if crop_to_fit:
            src_w, src_h = frame_pil.size
            src_aspect = src_w / src_h

            if abs(src_aspect - target_aspect) > 0.01:  # Different aspect ratios
                if src_aspect > target_aspect:
                    # Source is wider - crop width
                    new_w = int(src_h * target_aspect)
                    left = (src_w - new_w) // 2
                    frame_pil = frame_pil.crop((left, 0, left + new_w, src_h))
                else:
                    # Source is taller - crop height
                    new_h = int(src_w / target_aspect)
                    top = (src_h - new_h) // 2
                    frame_pil = frame_pil.crop((0, top, src_w, top + new_h))

                if i == 0:
                    generation_logger.debug(f"[STRUCTURE_VIDEO] Center-cropped from {src_w}x{src_h} to {frame_pil.size[0]}x{frame_pil.size[1]}")

        # Resize to target resolution
        frame_resized = frame_pil.resize((w, h), resample=Image.Resampling.LANCZOS)
        frame_resized_np = np.array(frame_resized)

        processed_frames.append(frame_resized_np)

    generation_logger.debug(f"[STRUCTURE_VIDEO] Preprocessed {len(processed_frames)} frames to {w}x{h}")

    return processed_frames
