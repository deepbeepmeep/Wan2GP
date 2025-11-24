"""
Transition effects and FFmpeg concatenation for storyboard scenes.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TransitionType:
    """Definition of a transition type"""
    id: str
    name: str
    description: str
    ffmpeg_filter: str  # FFmpeg xfade transition name
    supports_duration: bool = True


# Available transition types
TRANSITIONS = {
    "cut": TransitionType(
        id="cut",
        name="Cut",
        description="Hard cut, no transition effect",
        ffmpeg_filter="",
        supports_duration=False
    ),
    "crossfade": TransitionType(
        id="crossfade",
        name="Crossfade",
        description="Smooth blend between scenes",
        ffmpeg_filter="fade"
    ),
    "fade_black": TransitionType(
        id="fade_black",
        name="Fade through Black",
        description="Fade out to black, then fade in",
        ffmpeg_filter="fadeblack"
    ),
    "fade_white": TransitionType(
        id="fade_white",
        name="Fade through White",
        description="Fade out to white, then fade in",
        ffmpeg_filter="fadewhite"
    ),
    "wipe_left": TransitionType(
        id="wipe_left",
        name="Wipe Left",
        description="Wipe from right to left",
        ffmpeg_filter="wipeleft"
    ),
    "wipe_right": TransitionType(
        id="wipe_right",
        name="Wipe Right",
        description="Wipe from left to right",
        ffmpeg_filter="wiperight"
    ),
    "wipe_up": TransitionType(
        id="wipe_up",
        name="Wipe Up",
        description="Wipe from bottom to top",
        ffmpeg_filter="wipeup"
    ),
    "wipe_down": TransitionType(
        id="wipe_down",
        name="Wipe Down",
        description="Wipe from top to bottom",
        ffmpeg_filter="wipedown"
    ),
    "slide_left": TransitionType(
        id="slide_left",
        name="Slide Left",
        description="Slide in from right",
        ffmpeg_filter="slideleft"
    ),
    "slide_right": TransitionType(
        id="slide_right",
        name="Slide Right",
        description="Slide in from left",
        ffmpeg_filter="slideright"
    ),
    "dissolve": TransitionType(
        id="dissolve",
        name="Dissolve",
        description="Pixelated dissolve effect",
        ffmpeg_filter="dissolve"
    ),
    "zoom_in": TransitionType(
        id="zoom_in",
        name="Zoom In",
        description="Zoom into next scene",
        ffmpeg_filter="zoomin"
    ),
}


def get_transition_choices() -> List[Tuple[str, str]]:
    """Get list of transition choices for dropdown"""
    return [(t.name, t.id) for t in TRANSITIONS.values()]


def get_transition_names() -> List[str]:
    """Get list of transition display names"""
    return [t.name for t in TRANSITIONS.values()]


def get_transition_ids() -> List[str]:
    """Get list of transition IDs"""
    return list(TRANSITIONS.keys())


class SceneConcatenator:
    """Handles concatenation of video scenes with transitions using FFmpeg"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return 0.0

    def concatenate_simple(self, video_paths: List[str], output_name: str) -> str:
        """Simple concatenation without transitions (all cuts)"""
        if not video_paths:
            raise ValueError("No videos to concatenate")

        output_path = self.output_dir / f"{output_name}.mp4"

        # Create concat file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
            concat_file = f.name

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            os.unlink(concat_file)

        return str(output_path)

    def concatenate_with_transitions(
        self,
        scenes: List[Dict],
        output_name: str,
        fps: int = 16,
        progress_callback=None
    ) -> str:
        """
        Concatenate scenes with transitions using FFmpeg filter_complex.

        Args:
            scenes: List of dicts with keys: output_path, transition_type, transition_duration_frames
            output_name: Name for output file (without extension)
            fps: Frames per second
            progress_callback: Optional callback(progress: float, message: str)

        Returns:
            Path to output video
        """
        if not scenes:
            raise ValueError("No scenes to concatenate")

        if len(scenes) == 1:
            # Single scene, just copy
            return scenes[0]["output_path"]

        output_path = self.output_dir / f"{output_name}.mp4"

        # Check if any transitions are non-cut
        has_transitions = any(
            s.get("transition_type", "cut") != "cut" and s.get("transition_duration_frames", 0) > 0
            for s in scenes[:-1]  # Last scene has no transition
        )

        if not has_transitions:
            # All cuts, use simple concat
            return self.concatenate_simple(
                [s["output_path"] for s in scenes],
                output_name
            )

        # Build complex filter for transitions
        inputs = []
        filter_parts = []
        current_label = "[0:v]"

        # Add all inputs
        for i, scene in enumerate(scenes):
            inputs.extend(["-i", scene["output_path"]])

        # Calculate offsets and build xfade filters
        cumulative_offset = 0
        for i, scene in enumerate(scenes[:-1]):
            duration = self.get_video_duration(scene["output_path"])
            transition_type = scene.get("transition_type", "cut")
            transition_frames = scene.get("transition_duration_frames", 0)
            transition_duration = transition_frames / fps

            if transition_type == "cut" or transition_duration <= 0:
                # No transition, just concatenate
                if i == 0:
                    cumulative_offset = duration
                else:
                    cumulative_offset += duration
                continue

            # Get FFmpeg filter name
            trans_def = TRANSITIONS.get(transition_type)
            if not trans_def or not trans_def.ffmpeg_filter:
                cumulative_offset += duration
                continue

            # Calculate offset (when transition starts)
            offset = cumulative_offset + duration - transition_duration

            # Build xfade filter
            next_label = f"[v{i}]"
            if i == len(scenes) - 2:
                # Last transition outputs to final
                next_label = "[vout]"

            if i == 0:
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition={trans_def.ffmpeg_filter}:"
                    f"duration={transition_duration:.3f}:offset={offset:.3f}{next_label}"
                )
            else:
                filter_parts.append(
                    f"{current_label}[{i+1}:v]xfade=transition={trans_def.ffmpeg_filter}:"
                    f"duration={transition_duration:.3f}:offset={offset:.3f}{next_label}"
                )

            current_label = next_label
            cumulative_offset = offset  # Transition overlaps, so offset is new baseline

            if progress_callback:
                progress_callback(i / len(scenes), f"Processing scene {i+1}/{len(scenes)}")

        if not filter_parts:
            # No actual transitions built, fall back to simple concat
            return self.concatenate_simple(
                [s["output_path"] for s in scenes],
                output_name
            )

        # Build final command
        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            str(output_path)
        ]

        if progress_callback:
            progress_callback(0.9, "Encoding final video...")

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # Fall back to simple concat on error
            print(f"[Storyboard] Transition encoding failed: {e.stderr.decode()}")
            print("[Storyboard] Falling back to simple concatenation")
            return self.concatenate_simple(
                [s["output_path"] for s in scenes],
                output_name
            )

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return str(output_path)

    def create_thumbnail(self, video_path: str, output_path: str, time_offset: float = 0.5) -> str:
        """Extract a thumbnail from a video"""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(time_offset),
            "-i", video_path,
            "-vframes", "1",
            "-vf", "scale=160:-1",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            return ""
