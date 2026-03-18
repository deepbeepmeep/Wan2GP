"""
Structure Guidance Configuration - Unified VACE/Uni3C Parameter Handling.

This module consolidates the previously scattered structure video and Uni3C
parameters into a single, coherent configuration object.

Key Design Decisions:
- VACE and Uni3C are mutually exclusive (one target per generation)
- Single `strength` parameter works for both targets
- Backward compatible with legacy parameter formats
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging

from .base import ParamGroup

logger = logging.getLogger(__name__)


def normalize_structure_treatment(
    treatment: str | None,
    *,
    context: str = "structure guidance",
) -> Literal["adjust", "clip"]:
    normalized = "adjust" if treatment is None else str(treatment).strip().lower()
    if normalized not in {"adjust", "clip"}:
        raise ValueError(f"{context}: invalid treatment {treatment!r}")
    return normalized  # type: ignore[return-value]


@dataclass
class StructureVideoEntry:
    """
    A single source video for structure guidance.

    Represents one segment of a potentially multi-source structure guidance setup.
    """
    path: str
    start_frame: int = 0
    end_frame: Optional[int] = None  # None = entire video
    treatment: Literal["adjust", "clip"] = "adjust"
    source_start_frame: int = 0
    source_end_frame: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StructureVideoEntry':
        """Parse from dict format (e.g., from structure_videos array)."""
        return cls(
            path=d.get("path", ""),
            start_frame=d.get("start_frame", 0),
            end_frame=d.get("end_frame"),
            treatment=normalize_structure_treatment(d.get("treatment", "adjust")),
            source_start_frame=d.get("source_start_frame", 0),
            source_end_frame=d.get("source_end_frame"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format."""
        return {
            "path": self.path,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "treatment": self.treatment,
            "source_start_frame": self.source_start_frame,
            "source_end_frame": self.source_end_frame,
        }


@dataclass
class StructureGuidanceConfig(ParamGroup):
    """
    Unified structure guidance configuration for VACE and Uni3C.

    This consolidates 20+ legacy parameters into a clean, coherent config:

    OLD (VACE):                          NEW:
    - structure_type                     - target + preprocessing
    - structure_video_path               - videos[]
    - structure_videos[]                 - videos[]
    - structure_video_treatment          - videos[].treatment
    - structure_video_motion_strength    - strength
    - structure_canny_intensity          - canny_intensity
    - structure_depth_contrast           - depth_contrast
    - structure_guidance_video_url       - (internal) _guidance_video_url
    - structure_guidance_frame_offset    - (internal) _frame_offset

    OLD (Uni3C):                         NEW:
    - use_uni3c                          - target == "uni3c"
    - uni3c_guide_video                  - (internal) _guidance_video_url
    - uni3c_strength                     - strength
    - uni3c_start_percent                - step_window[0]
    - uni3c_end_percent                  - step_window[1]
    - uni3c_frame_policy                 - frame_policy
    - uni3c_zero_empty_frames            - zero_empty_frames
    - uni3c_guidance_frame_offset        - (internal) _frame_offset
    """

    # Source videos
    videos: List[StructureVideoEntry] = field(default_factory=list)

    # Target system: "vace" or "uni3c"
    target: Literal["vace", "uni3c"] = "vace"

    # Preprocessing type (VACE only, ignored for uni3c)
    preprocessing: Literal["flow", "canny", "depth", "none"] = "flow"

    # Unified strength (applies to both VACE motion strength and Uni3C weight)
    strength: float = 1.0

    # VACE preprocessing modifiers
    canny_intensity: float = 1.0
    depth_contrast: float = 1.0

    # Uni3C-specific parameters
    step_window: Tuple[float, float] = (0.0, 1.0)  # (start_percent, end_percent)
    frame_policy: str = "fit"
    zero_empty_frames: bool = True
    keep_on_gpu: bool = False

    # Internal computed fields (set during processing, not from input)
    _guidance_video_url: Optional[str] = None
    _frame_offset: int = 0
    _original_video_url: Optional[str] = None
    _trimmed_video_url: Optional[str] = None

    @classmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'StructureGuidanceConfig':
        """
        Parse structure guidance config from parameters.

        Supports both new unified format and legacy formats for backward compatibility.
        """
        # Check for new unified format first
        if "structure_guidance" in params and isinstance(params["structure_guidance"], dict):
            return cls._from_new_format(params["structure_guidance"], **context)

        # Fall back to legacy format detection
        return cls._from_legacy_format(params, **context)

    @classmethod
    def _from_new_format(cls, sg: Dict[str, Any], **context) -> 'StructureGuidanceConfig':
        """Parse from new unified structure_guidance format."""
        config = cls()

        # Parse videos array
        videos_raw = sg.get("videos", [])
        config.videos = [StructureVideoEntry.from_dict(v) for v in videos_raw]

        # Target
        config.target = sg.get("target", "vace")

        # Preprocessing
        config.preprocessing = sg.get("preprocessing", "flow")

        # Strength
        config.strength = float(sg.get("strength", 1.0))

        # VACE modifiers
        config.canny_intensity = float(sg.get("canny_intensity", 1.0))
        config.depth_contrast = float(sg.get("depth_contrast", 1.0))

        # Uni3C specific
        step_window = sg.get("step_window", [0.0, 1.0])
        if isinstance(step_window, (list, tuple)) and len(step_window) >= 2:
            config.step_window = (float(step_window[0]), float(step_window[1]))
        config.frame_policy = sg.get("frame_policy", "fit")
        config.zero_empty_frames = sg.get("zero_empty_frames", True)
        config.keep_on_gpu = sg.get("keep_on_gpu", False)

        # Optional internal computed fields (may be included when segment payloads are generated)
        config._guidance_video_url = sg.get("_guidance_video_url")
        config._frame_offset = int(sg.get("_frame_offset", 0) or 0)
        config._original_video_url = sg.get("_original_video_url")
        config._trimmed_video_url = sg.get("_trimmed_video_url")

        return config

    @classmethod
    def _from_legacy_format(cls, params: Dict[str, Any], **context) -> 'StructureGuidanceConfig':
        """
        Convert legacy parameters to unified config.

        Handles all the old parameter names for backward compatibility.
        """
        config = cls()

        # === Detect target from structure_type or use_uni3c ===
        structure_type = (
            params.get("structure_type") or
            params.get("structure_video_type")
        )
        use_uni3c = params.get("use_uni3c", False)

        if structure_type == "uni3c" or use_uni3c:
            config.target = "uni3c"
            config.preprocessing = "none"
        elif structure_type in ["flow", "canny", "depth"]:
            config.target = "vace"
            config.preprocessing = structure_type
        elif structure_type == "raw":
            config.target = "vace"
            config.preprocessing = "none"
        elif structure_type is None:
            # No structure guidance
            config.target = "vace"
            config.preprocessing = "flow"

        # === Parse videos from multiple possible sources ===

        # New format: structure_videos array
        structure_videos = params.get("structure_videos", [])
        if structure_videos:
            config.videos = [StructureVideoEntry.from_dict(v) for v in structure_videos]
            # Extract structure_type from first config if not set at top level
            if not structure_type and config.videos:
                first_type = structure_videos[0].get("structure_type", structure_videos[0].get("type"))
                if first_type:
                    if first_type == "uni3c":
                        config.target = "uni3c"
                        config.preprocessing = "none"
                    elif first_type == "raw":
                        config.target = "vace"
                        config.preprocessing = "none"
                    elif first_type in ["flow", "canny", "depth"]:
                        config.target = "vace"
                        config.preprocessing = first_type

        # Legacy format: single structure_video_path
        elif params.get("structure_video_path"):
            config.videos = [StructureVideoEntry(
                path=params["structure_video_path"],
                treatment=normalize_structure_treatment(params.get("structure_video_treatment", "adjust")),
            )]

        # === Strength (unified) ===
        if config.target == "uni3c":
            # Prefer explicit uni3c_strength, fall back to motion_strength
            config.strength = float(
                params.get("uni3c_strength") or
                params.get("structure_video_motion_strength") or
                1.0
            )
        else:
            config.strength = float(
                params.get("structure_video_motion_strength") or
                1.0
            )

        # === VACE preprocessing modifiers ===
        config.canny_intensity = float(params.get("structure_canny_intensity", 1.0))
        config.depth_contrast = float(params.get("structure_depth_contrast", 1.0))

        # === Uni3C specific ===
        config.step_window = (
            float(params.get("uni3c_start_percent", 0.0)),
            float(params.get("uni3c_end_percent", 1.0)),
        )
        config.frame_policy = params.get("uni3c_frame_policy", "fit")
        config.zero_empty_frames = params.get("uni3c_zero_empty_frames", True)
        config.keep_on_gpu = params.get("uni3c_keep_on_gpu", False)

        # === Internal computed fields (from pre-computed guidance) ===
        config._guidance_video_url = (
            params.get("structure_guidance_video_url") or
            params.get("structure_motion_video_url") or
            params.get("uni3c_guide_video")
        )
        config._frame_offset = (
            params.get("structure_guidance_frame_offset") or
            params.get("structure_motion_frame_offset") or
            params.get("uni3c_guidance_frame_offset") or
            0
        )
        config._original_video_url = params.get("structure_original_video_url")
        config._trimmed_video_url = params.get("structure_trimmed_video_url")

        return config

    def to_wgp_format(self) -> Dict[str, Any]:
        """
        Convert to WGP-compatible format.

        Returns the appropriate params based on target (VACE or Uni3C).
        """
        if self.target == "uni3c":
            return self.to_uni3c_params()
        else:
            return self.to_vace_params()

    def to_vace_params(self) -> Dict[str, Any]:
        """
        Output parameters for VACE consumption.

        Maps unified config back to legacy VACE parameter names.
        """
        result = {}

        # Map preprocessing back to structure_type
        if self.preprocessing == "none":
            result["structure_type"] = "raw"
        else:
            result["structure_type"] = self.preprocessing

        # Videos
        if self.videos:
            result["structure_videos"] = [v.to_dict() for v in self.videos]
            if len(self.videos) == 1:
                result["structure_video_path"] = self.videos[0].path
                result["structure_video_treatment"] = self.videos[0].treatment

        # Strength and modifiers
        result["structure_video_motion_strength"] = self.strength
        result["structure_canny_intensity"] = self.canny_intensity
        result["structure_depth_contrast"] = self.depth_contrast

        # Internal computed
        if self._guidance_video_url:
            result["structure_guidance_video_url"] = self._guidance_video_url
        if self._frame_offset:
            result["structure_guidance_frame_offset"] = self._frame_offset
        if self._original_video_url:
            result["structure_original_video_url"] = self._original_video_url
        if self._trimmed_video_url:
            result["structure_trimmed_video_url"] = self._trimmed_video_url

        return result

    def to_uni3c_params(self) -> Dict[str, Any]:
        """
        Output parameters for Uni3C consumption.

        Maps unified config to WGP's uni3c_* parameters.
        """
        return {
            "use_uni3c": True,
            "uni3c_guide_video": self._guidance_video_url,
            "uni3c_strength": self.strength,
            "uni3c_start_percent": self.step_window[0],
            "uni3c_end_percent": self.step_window[1],
            "uni3c_frame_policy": self.frame_policy,
            "uni3c_zero_empty_frames": self.zero_empty_frames,
            "uni3c_keep_on_gpu": self.keep_on_gpu,
        }

    def to_segment_payload(self, segment_index: int = 0, frame_offset: int = 0) -> Dict[str, Any]:
        """
        Create a segment-specific payload for enqueueing segment tasks.

        Args:
            segment_index: Index of this segment
            frame_offset: Frame offset in the guidance video for this segment

        Returns:
            Dict with structure guidance config for this segment
        """
        payload = {
            "structure_guidance": {
                "videos": [v.to_dict() for v in self.videos],
                "target": self.target,
                "preprocessing": self.preprocessing,
                "strength": self.strength,
                "canny_intensity": self.canny_intensity,
                "depth_contrast": self.depth_contrast,
                "step_window": list(self.step_window),
                "frame_policy": self.frame_policy,
                "zero_empty_frames": self.zero_empty_frames,
                "keep_on_gpu": self.keep_on_gpu,
            }
        }

        # Include computed values if available
        if self._guidance_video_url:
            payload["structure_guidance"]["_guidance_video_url"] = self._guidance_video_url
        payload["structure_guidance"]["_frame_offset"] = frame_offset

        return payload

    def validate(self) -> List[str]:
        """Validate structure guidance configuration."""
        errors = []

        # Target must be valid
        if self.target not in ("vace", "uni3c"):
            errors.append(f"Invalid target: {self.target}. Must be 'vace' or 'uni3c'")

        # Preprocessing must be valid for VACE
        if self.target == "vace" and self.preprocessing not in ("flow", "canny", "depth", "none"):
            errors.append(f"Invalid preprocessing: {self.preprocessing}. Must be 'flow', 'canny', 'depth', or 'none'")

        # Strength must be positive
        if self.strength < 0:
            errors.append(f"Strength must be non-negative, got: {self.strength}")

        # Step window must be valid
        if self.step_window[0] < 0 or self.step_window[1] > 1:
            errors.append(f"Step window must be within [0, 1], got: {self.step_window}")
        if self.step_window[0] > self.step_window[1]:
            errors.append(f"Step window start must be <= end, got: {self.step_window}")

        # Videos must have valid paths if present
        for i, video in enumerate(self.videos):
            if not video.path:
                errors.append(f"Video {i} has empty path")

        return errors

    @property
    def has_guidance(self) -> bool:
        """Check if any structure guidance is configured."""
        return bool(self.videos) or bool(self._guidance_video_url)

    @property
    def guidance_video_url(self) -> Optional[str]:
        """Public accessor for the computed/derived guidance video URL/path (if any)."""
        return self._guidance_video_url

    @property
    def is_uni3c(self) -> bool:
        """Check if this config targets Uni3C."""
        return self.target == "uni3c"

    @property
    def is_vace(self) -> bool:
        """Check if this config targets VACE."""
        return self.target == "vace"

    @property
    def legacy_structure_type(self) -> str:
        """
        Get the legacy structure_type value for backward compatibility.

        Maps:
        - target="uni3c" -> "uni3c"
        - target="vace", preprocessing="none" -> "raw"
        - target="vace", preprocessing=X -> X
        """
        if self.target == "uni3c":
            return "uni3c"
        elif self.preprocessing == "none":
            return "raw"
        else:
            return self.preprocessing

    def __repr__(self) -> str:
        return (
            f"StructureGuidanceConfig("
            f"target={self.target!r}, "
            f"preprocessing={self.preprocessing!r}, "
            f"strength={self.strength}, "
            f"videos={len(self.videos)}, "
            f"has_guidance={self.has_guidance})"
        )
