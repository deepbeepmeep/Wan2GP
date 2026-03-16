"""
Uni3C Validation Utilities

Extracts frames from videos and creates comparison images for VLM-based validation.
Claude (or other VLMs) can view these images to judge if Uni3C output matches
the guide video's motion better than baseline.

Usage:
    from scripts.uni3c_validation import create_uni3c_comparison
    
    # Create comparison grid for VLM review
    comparison_path = create_uni3c_comparison(
        guide_video="path/to/guide.mp4",
        baseline_output="path/to/baseline.mp4",
        uni3c_output="path/to/uni3c.mp4",
        output_dir="./test_results"
    )
    
    # Now show comparison_path to Claude/VLM for judgment
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


def run_subprocess(cmd, **kwargs):
    """Small indirection for subprocess execution so tests can stub probes/extraction."""
    return subprocess.run(cmd, **kwargs)


def extract_frames_from_video(
    video_path: str,
    num_frames: int = 5,
    output_dir: Optional[str] = None,
    allow_fallbacks: bool = False,
) -> List[str]:
    """
    Extract evenly-spaced frames from a video using ffmpeg.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default 5: start, 25%, 50%, 75%, end)
        output_dir: Directory to save frames (uses temp dir if None)
        
    Returns:
        List of paths to extracted frame images
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create output directory
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="uni3c_frames_"))
    
    # Get video duration using ffprobe
    try:
        result = run_subprocess(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True, timeout=300
        )
        duration = float(result.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        if not allow_fallbacks and "ffprobe" in str(e).lower():
            raise RuntimeError(f"Could not resolve duration for {video_path}") from e
        print(f"[UNI3C_VAL] Warning: Could not get duration, using 5s default: {e}")
        duration = 5.0
    
    # Calculate timestamps for evenly-spaced frames
    if num_frames == 1:
        timestamps = [duration / 2]
    else:
        timestamps = [duration * i / (num_frames - 1) for i in range(num_frames)]
    
    frame_paths = []
    video_name = video_path.stem
    
    for i, ts in enumerate(timestamps):
        frame_path = out_dir / f"{video_name}_frame_{i:02d}.jpg"
        
        try:
            run_subprocess(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
                 "-vframes", "1", "-q:v", "2", str(frame_path)],
                capture_output=True, check=True, timeout=300
            )
            if frame_path.exists():
                frame_paths.append(str(frame_path))
        except (subprocess.SubprocessError, OSError) as e:
            if not allow_fallbacks:
                print(f"[UNI3C_VAL] Warning: Failed to extract frame at {ts}s: {e}")
                continue
            print(f"[UNI3C_VAL] Warning: Failed to extract frame at {ts}s: {e}")

    if not frame_paths:
        raise RuntimeError(f"Failed to extract any frames from {video_path}")

    return frame_paths


def create_frame_strip(
    frame_paths: List[str],
    label: str,
    target_height: int = 200,
    border_color: Tuple[int, int, int] = (100, 100, 100)
) -> Image.Image:
    """
    Create a horizontal strip of frames with a label.
    
    Args:
        frame_paths: List of paths to frame images
        label: Label for this strip (e.g., "Guide Video", "Baseline", "Uni3C")
        target_height: Height to resize frames to
        border_color: RGB color for the label background
        
    Returns:
        PIL Image of the frame strip
    """
    if not frame_paths:
        # Return placeholder
        placeholder = Image.new('RGB', (400, target_height + 40), border_color)
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), f"{label}: No frames", fill=(255, 255, 255))
        return placeholder
    
    # Load and resize frames
    frames = []
    for fp in frame_paths:
        try:
            img = Image.open(fp)
            # Maintain aspect ratio
            aspect = img.width / img.height
            new_width = int(target_height * aspect)
            img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            frames.append(img)
        except (OSError, ValueError) as e:
            print(f"[UNI3C_VAL] Warning: Could not load frame {fp}: {e}")
    
    if not frames:
        placeholder = Image.new('RGB', (400, target_height + 40), border_color)
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), f"{label}: Failed to load frames", fill=(255, 255, 255))
        return placeholder
    
    # Calculate strip dimensions
    gap = 4
    label_height = 40
    total_width = sum(f.width for f in frames) + gap * (len(frames) - 1) + 20
    total_height = target_height + label_height
    
    # Create strip canvas
    strip = Image.new('RGB', (total_width, total_height), (30, 30, 30))
    draw = ImageDraw.Draw(strip)
    
    # Draw label background
    draw.rectangle([0, 0, total_width, label_height], fill=border_color)
    
    # Draw label text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (OSError, ValueError):
        font = ImageFont.load_default()
    
    draw.text((10, 8), label, fill=(255, 255, 255), font=font)
    
    # Add frame number indicators
    frame_labels = ["Start", "25%", "50%", "75%", "End"][:len(frames)]
    
    # Paste frames
    x_offset = 10
    for i, (frame, fl) in enumerate(zip(frames, frame_labels)):
        strip.paste(frame, (x_offset, label_height))
        
        # Draw frame label below
        try:
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (OSError, ValueError):
            small_font = ImageFont.load_default()
        
        # Draw on the frame itself (bottom)
        frame_draw = ImageDraw.Draw(strip)
        text_y = label_height + target_height - 20
        frame_draw.rectangle([x_offset, text_y, x_offset + 40, text_y + 18], fill=(0, 0, 0, 180))
        frame_draw.text((x_offset + 4, text_y + 2), fl, fill=(255, 255, 255), font=small_font)
        
        x_offset += frame.width + gap
    
    return strip


def create_uni3c_comparison(
    guide_video: str,
    baseline_output: str,
    uni3c_output: str,
    output_dir: str = "./test_results/uni3c_validation",
    num_frames: int = 5,
    task_id: str = "unknown",
    allow_fallbacks: bool = False,
) -> str:
    """
    Create a comparison image grid for VLM validation.
    
    Shows three rows:
    1. Guide Video frames (what motion should look like)
    2. Baseline Output frames (no Uni3C)
    3. Uni3C Output frames (with Uni3C applied)
    
    Args:
        guide_video: Path to guide video (motion reference)
        baseline_output: Path to baseline output (no Uni3C)
        uni3c_output: Path to Uni3C output
        output_dir: Directory to save comparison image
        num_frames: Number of frames to extract from each video
        task_id: Task ID for naming
        
    Returns:
        Path to the comparison image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[UNI3C_VAL] Creating comparison for task {task_id}")
    print(f"[UNI3C_VAL]   Guide: {guide_video}")
    print(f"[UNI3C_VAL]   Baseline: {baseline_output}")
    print(f"[UNI3C_VAL]   Uni3C: {uni3c_output}")
    
    # Extract frames from each video
    frames_dir = output_dir / "frames" / task_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    guide_frames = extract_frames_from_video(
        guide_video,
        num_frames,
        str(frames_dir / "guide"),
        allow_fallbacks=allow_fallbacks,
    )
    baseline_frames = extract_frames_from_video(
        baseline_output,
        num_frames,
        str(frames_dir / "baseline"),
        allow_fallbacks=allow_fallbacks,
    )
    uni3c_frames = extract_frames_from_video(
        uni3c_output,
        num_frames,
        str(frames_dir / "uni3c"),
        allow_fallbacks=allow_fallbacks,
    )
    
    print(f"[UNI3C_VAL] Extracted frames: guide={len(guide_frames)}, baseline={len(baseline_frames)}, uni3c={len(uni3c_frames)}")
    
    # Create frame strips
    guide_strip = create_frame_strip(guide_frames, "GUIDE VIDEO (Target Motion)", 
                                      border_color=(70, 130, 180))  # Steel blue
    baseline_strip = create_frame_strip(baseline_frames, "BASELINE (No Uni3C)", 
                                         border_color=(139, 69, 19))  # Saddle brown
    uni3c_strip = create_frame_strip(uni3c_frames, "UNI3C OUTPUT (With Motion Guidance)", 
                                      border_color=(34, 139, 34))  # Forest green
    
    # Combine into single comparison image
    max_width = max(guide_strip.width, baseline_strip.width, uni3c_strip.width)
    total_height = guide_strip.height + baseline_strip.height + uni3c_strip.height + 120  # Extra for title/instructions
    
    comparison = Image.new('RGB', (max_width, total_height), (20, 20, 20))
    draw = ImageDraw.Draw(comparison)
    
    # Title
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        inst_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, ValueError):
        title_font = ImageFont.load_default()
        inst_font = ImageFont.load_default()
    
    draw.text((10, 10), f"Uni3C Validation - Task {task_id}", fill=(255, 255, 255), font=title_font)
    
    instructions = (
        "QUESTION: Does the Uni3C output (green) match the Guide Video's motion better than Baseline (brown)?\n"
        "Look at: camera movement, subject motion direction, overall flow between frames."
    )
    draw.text((10, 50), instructions, fill=(200, 200, 200), font=inst_font)
    
    # Paste strips
    y_offset = 100
    comparison.paste(guide_strip, (0, y_offset))
    y_offset += guide_strip.height + 10
    comparison.paste(baseline_strip, (0, y_offset))
    y_offset += baseline_strip.height + 10
    comparison.paste(uni3c_strip, (0, y_offset))
    
    # Save comparison image
    comparison_path = output_dir / f"uni3c_comparison_{task_id}.jpg"
    comparison.save(str(comparison_path), quality=90)
    
    print(f"[UNI3C_VAL] ✅ Saved comparison image: {comparison_path}")
    
    return str(comparison_path)


def create_vlm_validation_prompt(comparison_image_path: str) -> str:
    """
    Generate a prompt for Claude/VLM to evaluate the Uni3C comparison.
    
    Args:
        comparison_image_path: Path to the comparison image
        
    Returns:
        Prompt string for the VLM
    """
    return f"""Please analyze this Uni3C motion guidance validation image.

The image shows three rows of video frames:
1. **GUIDE VIDEO** (blue header): This is the reference motion we want to transfer
2. **BASELINE** (brown header): Output WITHOUT Uni3C motion guidance  
3. **UNI3C OUTPUT** (green header): Output WITH Uni3C motion guidance applied

Each row shows 5 frames: Start, 25%, 50%, 75%, End of the video.

**Your task:**
1. Describe the motion/camera movement visible in the GUIDE VIDEO frames
2. Describe the motion in the BASELINE frames
3. Describe the motion in the UNI3C OUTPUT frames
4. **VERDICT**: Does the Uni3C output exhibit motion more similar to the Guide Video than the Baseline?

Answer format:
- Guide Motion: [description]
- Baseline Motion: [description]  
- Uni3C Motion: [description]
- VERDICT: [YES/NO/INCONCLUSIVE] - Uni3C [matches/does not match] guide better than baseline
- Confidence: [HIGH/MEDIUM/LOW]
- Reasoning: [1-2 sentences explaining your verdict]

Image path: {comparison_image_path}
"""


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    allow_fallbacks = False
    if "--allow-fallbacks" in argv:
        argv.remove("--allow-fallbacks")
        allow_fallbacks = True

    if len(argv) < 4:
        print("Usage: python uni3c_validation.py <guide_video> <baseline_output> <uni3c_output> <task_id> [--allow-fallbacks]")
        print("\nCreates a comparison image for VLM validation of Uni3C motion guidance.")
        return 1

    guide, baseline, uni3c, task_id = argv[:4]
    comparison_path = create_uni3c_comparison(
        guide,
        baseline,
        uni3c,
        task_id=task_id,
        allow_fallbacks=allow_fallbacks,
    )
    print(f"\n✅ Comparison image created: {comparison_path}")
    print(f"\n📋 VLM Prompt:\n{create_vlm_validation_prompt(comparison_path)}")
    return 0


# CLI interface for testing
if __name__ == "__main__":
    raise SystemExit(main())
