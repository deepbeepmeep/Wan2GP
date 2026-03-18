"""Frame-level operations for structure guidance: neutral frames and ranged loading."""

import numpy as np
from typing import List, Tuple, Optional

from source.core.log import generation_logger
from source.media.structure.loading import _resample_frame_indices

__all__ = [
    "create_neutral_frame",
    "load_structure_video_frames_with_range",
]


def create_neutral_frame(structure_type: str, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Create a neutral frame for gaps in structure video coverage.

    These frames don't bias the generation - they represent "no guidance signal".

    IMPORTANT for Uni3C (structure_type="raw"):
        Black pixel frames are detected by _encode_uni3c_guide() in any2video.py
        and converted to zeros in latent space. This is critical because:
        - VAE-encoded black pixels != zero latents
        - Zero latents = true "no control"
        - VAE-encoded black = "control toward black output"

        The detection happens automatically during VAE encoding, so black frames
        created here will be properly handled as "no guidance".

    Args:
        structure_type: Type of structure preprocessing ("flow", "canny", "depth", "raw", or "uni3c")
        resolution: (width, height) tuple

    Returns:
        numpy array [H, W, C] uint8 RGB
    """
    w, h = resolution

    if structure_type == "flow":
        # Gray = center of HSV color wheel = no motion
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "canny":
        # Black = no edges detected
        return np.zeros((h, w, 3), dtype=np.uint8)
    elif structure_type == "depth":
        # Mid-gray = neutral depth (not close, not far)
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "raw":
        # Black = no structure signal for Uni3C
        # NOTE: These black frames will be detected during VAE encoding and
        # their latents will be zeroed for true "no control" behavior.
        return np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Default to black
        return np.zeros((h, w, 3), dtype=np.uint8)


def load_structure_video_frames_with_range(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: Tuple[int, int],
    treatment: str = "adjust",
    crop_to_fit: bool = True,
    source_start_frame: int = 0,
    source_end_frame: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Load structure video frames with optional source range extraction.

    Extended version of load_structure_video_frames that supports extracting
    a specific range from the source video before applying treatment.

    Args:
        structure_video_path: Path to structure video
        target_frame_count: Number of output frames needed
        target_fps: Target FPS (used for clip mode)
        target_resolution: (width, height) tuple
        treatment: "adjust" (stretch/compress) or "clip" (temporal sample)
        crop_to_fit: Center-crop to match target aspect ratio
        source_start_frame: First frame to extract from source (default: 0)
        source_end_frame: Last frame (exclusive) to extract (default: None = end of video)

    Returns:
        List of numpy uint8 arrays [H, W, C] in RGB format
    """
    import cv2
    from PIL import Image

    # Try decord first (faster), fall back to cv2
    use_decord = False
    try:
        import decord
        decord.bridge.set_bridge('torch')
        use_decord = True
    except ImportError:
        generation_logger.debug("[STRUCTURE_VIDEO_RANGE] decord not available, using cv2 fallback")

    if use_decord:
        try:
            reader = decord.VideoReader(structure_video_path)
            video_fps = round(reader.get_avg_fps())
            total_video_frames = len(reader)
        except (ModuleNotFoundError, RuntimeError, OSError, ValueError) as exc:
            generation_logger.debug(f"[STRUCTURE_VIDEO_RANGE] decord reader unavailable, falling back to cv2: {exc}")
            use_decord = False

    if not use_decord:
        cap = cv2.VideoCapture(structure_video_path)
        video_fps = round(cap.get(cv2.CAP_PROP_FPS))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resolve source range
    actual_start = source_start_frame
    actual_end = source_end_frame if source_end_frame is not None else total_video_frames

    # Clamp to valid range
    actual_start = max(0, min(actual_start, total_video_frames - 1))
    actual_end = max(actual_start + 1, min(actual_end, total_video_frames))

    effective_source_frames = actual_end - actual_start

    generation_logger.debug(f"[STRUCTURE_VIDEO_RANGE] Loading from source video:")
    generation_logger.debug(f"  Total source frames: {total_video_frames} @ {video_fps}fps")
    generation_logger.debug(f"  Source range: [{actual_start}, {actual_end}) = {effective_source_frames} frames")
    generation_logger.debug(f"  Target frames needed: {target_frame_count}")
    generation_logger.debug(f"  Treatment: {treatment}")

    # Load N+1 frames for flow processing (RAFT needs N frames to produce N-1 flows)
    frames_to_load = target_frame_count + 1

    # Calculate frame indices based on treatment mode
    if treatment == "adjust":
        # ADJUST: Stretch/compress source range to match needed count
        if effective_source_frames >= frames_to_load:
            # Compress: sample evenly
            frame_indices = [
                actual_start + int(i * (effective_source_frames - 1) / (frames_to_load - 1))
                for i in range(frames_to_load)
            ]
            generation_logger.debug(f"  Adjust: Compressing {effective_source_frames} -> {frames_to_load} frames")
        else:
            # Stretch: repeat frames
            frame_indices = [
                actual_start + int(i * (effective_source_frames - 1) / (frames_to_load - 1))
                for i in range(frames_to_load)
            ]
            generation_logger.debug(f"  Adjust: Stretching {effective_source_frames} -> {frames_to_load} frames")
    else:
        # CLIP: Temporal sampling from the source range
        # Calculate indices relative to source range
        frame_indices = _resample_frame_indices(
            video_fps=video_fps,
            video_frames_count=effective_source_frames,
            max_target_frames_count=frames_to_load,
            target_fps=target_fps,
            start_target_frame=0
        )
        # Offset to actual source positions
        frame_indices = [actual_start + idx for idx in frame_indices]

        # Handle if source range is too short
        if len(frame_indices) < frames_to_load:
            generation_logger.debug(f"  Clip: Source range too short, looping to fill {frames_to_load} frames")
            while len(frame_indices) < frames_to_load:
                remaining = frames_to_load - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])

        generation_logger.debug(f"  Clip: Extracted {len(frame_indices)} frame indices")

    if not frame_indices:
        raise ValueError(f"No frames could be extracted from range [{actual_start}, {actual_end})")

    # Extract frames
    if use_decord:
        frames = reader.get_batch(frame_indices)
        raw_frames = []
        for frame in frames:
            if hasattr(frame, 'cpu'):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = np.array(frame)
            raw_frames.append(frame_np)
    else:
        # cv2 fallback - read frames sequentially
        raw_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw_frames.append(frame_rgb)
            else:
                # Fallback: use last frame or black
                if raw_frames:
                    raw_frames.append(raw_frames[-1].copy())
                else:
                    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    raw_frames.append(np.zeros((h_src, w_src, 3), dtype=np.uint8))
        cap.release()

    # Process frames to target resolution
    w, h = target_resolution
    target_aspect = w / h
    processed_frames = []

    for i, frame_np in enumerate(raw_frames):
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        frame_pil = Image.fromarray(frame_np)

        # Center crop if needed
        if crop_to_fit:
            src_w, src_h = frame_pil.size
            src_aspect = src_w / src_h

            if abs(src_aspect - target_aspect) > 0.01:
                if src_aspect > target_aspect:
                    new_w = int(src_h * target_aspect)
                    left = (src_w - new_w) // 2
                    frame_pil = frame_pil.crop((left, 0, left + new_w, src_h))
                else:
                    new_h = int(src_w / target_aspect)
                    top = (src_h - new_h) // 2
                    frame_pil = frame_pil.crop((0, top, src_w, top + new_h))

        frame_resized = frame_pil.resize((w, h), resample=Image.Resampling.LANCZOS)
        processed_frames.append(np.array(frame_resized))

    generation_logger.debug(f"[STRUCTURE_VIDEO_RANGE] Loaded {len(processed_frames)} frames at {w}x{h}")

    return processed_frames
