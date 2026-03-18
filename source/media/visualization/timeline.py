"""Timeline clip creation and video treatment utilities."""

import numpy as np
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

from source.core.log import headless_logger
from source.media.visualization.timeline_thumbnails import _load_timeline_thumbnail


def _apply_video_treatment(clip, target_duration: float, target_fps: int, treatment: str, video_name: str = "video"):
    """
    Apply treatment to video clip to match target duration using FRAME SAMPLING.

    This matches the exact logic from structure_video_guidance.py:load_structure_video_frames()
    by sampling specific frame indices rather than changing playback speed.

    Args:
        clip: MoviePy VideoFileClip
        target_duration: Target duration in seconds
        target_fps: Target FPS for output
        treatment: "adjust" (linear frame sampling) or "clip" (FPS-based sampling)
        video_name: Name for logging

    Returns:
        Treated VideoFileClip with resampled frames
    """
    from moviepy.editor import ImageSequenceClip

    current_fps = clip.fps
    current_frame_count = int(clip.duration * clip.fps)
    target_frame_count = int(target_duration * target_fps)
    if current_frame_count <= 0 or target_frame_count <= 0:
        raise ValueError("Video treatment requires at least one input and target frame")

    if treatment == "adjust":
        # ADJUST MODE: Linear interpolation sampling (matches generation logic line 201)
        if target_frame_count == 1:
            frame_indices = [0]
            headless_logger.debug(f"  {video_name}: adjust mode - collapsing to a single frame")
        elif current_frame_count == 1:
            frame_indices = [0 for _ in range(target_frame_count)]
            headless_logger.debug(f"  {video_name}: adjust mode - repeating single source frame")
        elif current_frame_count >= target_frame_count:
            frame_indices = [int(i * (current_frame_count - 1) / (target_frame_count - 1))
                           for i in range(target_frame_count)]
            headless_logger.debug(f"  {video_name}: adjust mode - sampling {target_frame_count} from {current_frame_count} frames (compress)")
        else:
            frame_indices = [int(i * (current_frame_count - 1) / (target_frame_count - 1))
                           for i in range(target_frame_count)]
            duplicates = target_frame_count - len(set(frame_indices))
            headless_logger.debug(f"  {video_name}: adjust mode - sampling {target_frame_count} from {current_frame_count} frames (stretch, {duplicates} duplicates)")

    elif treatment == "clip":
        # CLIP MODE: FPS-based temporal sampling
        def resample_frame_indices(video_fps, video_frames_count, max_target_frames, target_fps):
            """Matches _resample_frame_indices from structure_video_guidance.py"""
            import math
            video_frame_duration = 1 / video_fps
            target_frame_duration = 1 / target_fps

            target_time = 0
            frame_no = math.ceil(target_time / video_frame_duration)
            cur_time = frame_no * video_frame_duration
            frame_ids = []

            while True:
                if len(frame_ids) >= max_target_frames:
                    break
                diff = round((target_time - cur_time) / video_frame_duration, 5)
                add_frames_count = math.ceil(diff)
                frame_no += add_frames_count
                if frame_no >= video_frames_count:
                    break
                frame_ids.append(frame_no)
                cur_time += add_frames_count * video_frame_duration
                target_time += target_frame_duration

            return frame_ids[:max_target_frames]

        frame_indices = resample_frame_indices(current_fps, current_frame_count, target_frame_count, target_fps)

        # If video too short, loop
        if not frame_indices and current_frame_count > 0:
            frame_indices = [0]
        if len(frame_indices) < target_frame_count:
            headless_logger.debug(f"  {video_name}: clip mode - video too short, looping to fill {target_frame_count} frames")
            while len(frame_indices) < target_frame_count:
                remaining = target_frame_count - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])

        headless_logger.debug(f"  {video_name}: clip mode - sampled {len(frame_indices)} frames from {current_frame_count}")

    else:
        raise ValueError(f"Invalid treatment: {treatment}. Must be 'adjust' or 'clip'")

    # Extract frames at the specified indices
    headless_logger.debug(f"  {video_name}: extracting {len(frame_indices)} frames...")
    frames = []
    for idx in frame_indices:
        time_at_frame = idx / clip.fps
        frame = clip.get_frame(time_at_frame)
        frames.append(frame)

    # Create new clip from resampled frames
    resampled_clip = ImageSequenceClip(frames, fps=target_fps)

    headless_logger.debug(f"  {video_name}: resampled to {len(frames)} frames @ {target_fps}fps ({resampled_clip.duration:.2f}s)")

    return resampled_clip


def _create_timeline_clip(
    duration: float,
    width: int,
    height: int,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    vertical: bool = False
):
    """
    Create an animated timeline clip showing:
    - Thumbnail strip of input images
    - Progress bar with current segment highlighted
    - Current frame number and segment info

    Args:
        frame_overlaps: List of overlap frame counts between segments.
                       Overlaps are subtracted when calculating final positions.
    """
    from moviepy.editor import VideoClip

    # Calculate segment boundaries ACCOUNTING FOR OVERLAPS first
    overlaps = frame_overlaps if frame_overlaps else [0] * (len(segment_frames) - 1)

    segment_boundaries = []
    cumulative_start = 0

    for i, seg_frames in enumerate(segment_frames):
        segment_end = cumulative_start + seg_frames - 1
        segment_boundaries.append((cumulative_start, segment_end))

        if i < len(overlaps):
            cumulative_start = segment_end + 1 - overlaps[i]
        else:
            cumulative_start = segment_end + 1

    # Calculate actual total frames (accounting for overlaps)
    total_frames = sum(segment_frames) - sum(overlaps)

    # Calculate image positions
    image_frame_positions = [0]

    for start, end in segment_boundaries:
        image_frame_positions.append(end)

    if len(input_image_paths) == len(segment_frames):
        image_frame_positions = [end for start, end in segment_boundaries]
    else:
        image_frame_positions = [0] + [end for start, end in segment_boundaries]

    # Calculate width available for images/bar (with margins)
    margin = 20
    available_width = width - (2 * margin)

    # Load and prepare input images with consistent sizing
    num_images = len(input_image_paths)

    if vertical:
        max_thumb_width = int(width * 0.8)
        available_height = height - (2 * margin)
        max_thumb_height = int(available_height / (num_images + 0.5))
    else:
        spacing = 10
        total_spacing = spacing * (num_images - 1)
        max_thumb_width = int((available_width - total_spacing) / num_images)
        max_thumb_width = int(max_thumb_width * 0.9)
        max_thumb_height = int(height * 0.6)

    thumbnails = []
    for img_path in input_image_paths:
        img = Image.open(img_path).convert('RGB')
        aspect = img.height / img.width
        target_h = int(max_thumb_width * aspect)
        if target_h > max_thumb_height:
            target_h = max_thumb_height
            thumb_w = int(max_thumb_height / aspect)
        else:
            thumb_w = max_thumb_width
        img = img.resize((thumb_w, target_h), Image.Resampling.LANCZOS)
        thumbnails.append(np.array(img))

    def make_frame(t):
        """Generate frame at time t."""
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

        current_frame = int(t * fps)

        # Determine active image
        active_image_index = 0
        if len(input_image_paths) == len(segment_frames) + 1:
            for i in range(len(image_frame_positions) - 1):
                current_img_frame = image_frame_positions[i]
                next_img_frame = image_frame_positions[i + 1]
                halfway_frame = (current_img_frame + next_img_frame) / 2

                if current_frame >= halfway_frame:
                    active_image_index = i + 1
                else:
                    break
            active_image_index = min(active_image_index, len(input_image_paths) - 1)
        else:
            current_segment = 0
            for i, (start, end) in enumerate(segment_boundaries):
                if start <= current_frame <= end:
                    current_segment = i
                    break
            if current_frame > segment_boundaries[-1][1]:
                current_segment = len(segment_boundaries) - 1
            active_image_index = current_segment

        # Calculate image positions
        if vertical:
            available_height_v = height - (2 * margin) - 50
            num_images_v = len(image_frame_positions)

            image_y_positions = []
            top_y = margin + 30

            image_y_positions.append(top_y)

            if num_images_v > 1:
                remaining_height = available_height_v - 100
                spacing_y = remaining_height / max(num_images_v - 1, 1)
                for i in range(1, num_images_v):
                    y_pos = top_y + 100 + int((i - 1) * spacing_y)
                    image_y_positions.append(y_pos)

            ordered_positions = image_y_positions.copy()
            if active_image_index < len(ordered_positions):
                temp_positions = [0] * num_images_v
                temp_positions[active_image_index] = ordered_positions[0]

                other_idx = 1
                for i in range(num_images_v):
                    if i != active_image_index:
                        temp_positions[i] = ordered_positions[other_idx]
                        other_idx += 1
                image_y_positions = temp_positions

            image_x_positions = [width // 2] * num_images_v
        else:
            bar_start_x = margin
            bar_end_x = margin + available_width
            bar_width = available_width

            image_x_positions = []
            for i, frame_pos in enumerate(image_frame_positions):
                progress_ratio = frame_pos / (total_frames - 1) if total_frames > 1 else 0
                x_pos = bar_start_x + int(progress_ratio * bar_width)
                image_x_positions.append(x_pos)

            base_y = 80
            image_y_positions = [base_y] * len(image_x_positions)

        # Draw thumbnails
        for i, (thumb, x_center, y_center) in enumerate(zip(thumbnails, image_x_positions, image_y_positions)):
            should_highlight = (i == active_image_index)

            if should_highlight:
                thumb_img = Image.fromarray(thumb)
                scaled_w = int(thumb_img.width * 1.15)
                scaled_h = int(thumb_img.height * 1.15)
                thumb_img = thumb_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

                x_start = x_center - scaled_w // 2
                y_start = y_center - 40
                x_start = max(0, min(x_start, width - scaled_w))
                y_start = max(0, min(y_start, height - scaled_h))

                border_color = (255, 100, 0)
                draw.rectangle(
                    [x_start - 3, y_start - 3,
                     x_start + scaled_w + 3, y_start + scaled_h + 3],
                    outline=border_color,
                    width=4
                )

                img.paste(thumb_img, (x_start, y_start))
            else:
                x_start = x_center - thumb.shape[1] // 2
                y_start = y_center
                x_start = max(0, min(x_start, width - thumb.shape[1]))
                y_start = max(0, min(y_start, height - thumb.shape[0]))

                thumb_img = Image.fromarray(thumb)
                img.paste(thumb_img, (x_start, y_start))

        # Draw progress bar
        if vertical:
            bar_width_v = 12
            bar_x = width - margin - bar_width_v
            bar_start_y = margin
            bar_end_y = height - margin
            bar_height_v = bar_end_y - bar_start_y

            draw.rectangle(
                [bar_x, bar_start_y, bar_x + bar_width_v, bar_end_y],
                fill=(200, 200, 200),
                outline=(150, 150, 150)
            )

            progress = min(t / duration, 1.0)
            progress_y = bar_start_y + int(bar_height_v * progress)

            if progress_y <= bar_start_y and progress > 0:
                progress_y = bar_start_y + 1

            draw.rectangle(
                [bar_x, bar_start_y, bar_x + bar_width_v, progress_y],
                fill=(0, 150, 255),
                outline=None
            )
        else:
            max_thumb_h = max(thumb.shape[0] for thumb in thumbnails)

            bar_y = 10 + max_thumb_h + 10
            bar_height = 12

            draw.rectangle(
                [bar_start_x, bar_y, bar_end_x, bar_y + bar_height],
                fill=(200, 200, 200),
                outline=(150, 150, 150)
            )

            progress = min(t / duration, 1.0)
            progress_x = bar_start_x + int(bar_width * progress)

            if progress_x <= bar_start_x and progress > 0:
                progress_x = bar_start_x + 1

            draw.rectangle(
                [bar_start_x, bar_y, progress_x, bar_y + bar_height],
                fill=(0, 150, 255),
                outline=None
            )

        # Draw image markers on progress bar (horizontal only)
        if not vertical:
            for i, x_pos in enumerate(image_x_positions):
                marker_active = (i == active_image_index)

                marker_color = (255, 100, 0) if marker_active else (100, 100, 100)
                draw.line(
                    [(x_pos, bar_y - 5), (x_pos, bar_y + bar_height + 5)],
                    fill=marker_color,
                    width=3 if marker_active else 2
                )

                circle_r = 4
                draw.ellipse(
                    [x_pos - circle_r, bar_y + bar_height//2 - circle_r,
                     x_pos + circle_r, bar_y + bar_height//2 + circle_r],
                    fill=marker_color,
                    outline=(255, 255, 255)
                )

        # Current segment for text
        current_segment = 0
        for i, (start, end) in enumerate(segment_boundaries):
            if start <= current_frame <= end:
                current_segment = i
                break
        if current_frame > segment_boundaries[-1][1]:
            current_segment = len(segment_boundaries) - 1

        # Draw text info
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        text = f"Frame {current_frame}/{total_frames-1} | Segment {current_segment + 1}/{len(segment_boundaries)}"
        if segment_prompts and current_segment < len(segment_prompts):
            prompt_preview = segment_prompts[current_segment][:50]
            text += f" | {prompt_preview}..."

        if vertical:
            text_y = height - margin - 30
        else:
            text_y = bar_y + 20
        draw.text((margin, text_y), text, fill=(50, 50, 50), font=font)

        return np.array(img)

    return VideoClip(make_frame, duration=duration)
