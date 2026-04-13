"""Travel guidance contract for orchestrator and segment payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from source.runtime.wgp_bridge import get_model_def

from .base import ParamGroup
from .structure_guidance import StructureGuidanceConfig, StructureVideoEntry

_TRAVEL_GUIDANCE_KIND = Literal["none", "vace", "ltx_control", "ltx_hybrid", "uni3c"]
_TRAVEL_GUIDANCE_LEGACY_KEYS = (
    "structure_type",
    "structure_video_path",
    "structure_videos",
    "use_uni3c",
)
_LTX_CONTROL_MODES = {"pose", "depth", "canny", "video", "cameraman"}


def _has_payload_value(value: Any) -> bool:
    """Return True when a payload field should count as "present"."""
    if value is None or value is False:
        return False
    if isinstance(value, (list, dict, str)):
        return bool(value)
    return True


def _infer_allowed_kinds(model_name: str) -> set[str]:
    """Infer which travel guidance kinds are allowed for a model."""
    lowered = (model_name or "").lower()

    if "ltx2" in lowered:
        pipeline_kind = None
        try:
            model_def = get_model_def(model_name)
            if isinstance(model_def, dict):
                pipeline_kind = model_def.get("ltx2_pipeline")
        except (ImportError, TypeError, ValueError, KeyError, AttributeError):
            pipeline_kind = None

        is_distilled = pipeline_kind == "distilled" or (
            pipeline_kind is None and "distilled" in lowered
        )
        if is_distilled:
            return {"none", "ltx_control", "ltx_hybrid", "uni3c"}
        return {"none"}

    return {"none", "vace", "uni3c"}


@dataclass
class AnchorEntry:
    """Single positioned image anchor for hybrid travel guidance."""

    image_url: str
    frame_position: int
    strength: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "AnchorEntry":
        return cls(
            image_url=str(d.get("image_url", "") or ""),
            frame_position=int(d.get("frame_position", 0) or 0),
            strength=float(d.get("strength", 1.0)),
        )

    def to_dict(self) -> dict:
        return {
            "image_url": self.image_url,
            "frame_position": self.frame_position,
            "strength": self.strength,
        }


@dataclass
class AudioConditioningConfig:
    """Optional generation-time audio conditioning for hybrid travel guidance."""

    source: Literal["external", "control_track"]
    audio_url: Optional[str] = None
    strength: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConditioningConfig":
        return cls(
            source=str(d.get("source", "") or ""),  # type: ignore[arg-type]
            audio_url=d.get("audio_url"),
            strength=float(d.get("strength", 1.0)),
        )

    def to_dict(self) -> dict:
        result = {
            "source": self.source,
            "strength": self.strength,
        }
        if self.audio_url:
            result["audio_url"] = self.audio_url
        return result


@dataclass
class TravelGuidanceConfig(ParamGroup):
    """Discriminated union for travel guidance."""

    kind: _TRAVEL_GUIDANCE_KIND = "none"

    # Shared fields for all non-none kinds
    videos: List[StructureVideoEntry] = field(default_factory=list)
    strength: float = 0.0

    # Hybrid-only fields
    anchors: List[AnchorEntry] = field(default_factory=list)
    control_strength: float = 1.0
    audio: Optional[AudioConditioningConfig] = None

    # VACE / LTX control
    mode: str = ""
    canny_intensity: float = 1.0
    depth_contrast: float = 1.0

    # Uni3C
    step_window: Tuple[float, float] = (0.0, 1.0)
    frame_policy: str = "fit"
    zero_empty_frames: bool = True
    keep_on_gpu: bool = False

    # Internal orchestration state
    _guidance_video_url: Optional[str] = None
    _frame_offset: int = 0

    @classmethod
    def from_params(cls, params: Dict[str, Any], **context) -> "TravelGuidanceConfig":
        model_name = (
            context.get("model_name")
            or context.get("model")
            or params.get("model_name")
            or params.get("model")
            or ""
        )
        return cls.from_payload(params, model_name)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], model_name: str) -> "TravelGuidanceConfig":
        """Parse from either a full params dict or a raw ``travel_guidance`` dict."""
        if not isinstance(payload, dict):
            raise ValueError("travel_guidance payload must be a dict")

        if "travel_guidance" in payload:
            raw = payload.get("travel_guidance") or {}
            if not isinstance(raw, dict):
                raise ValueError("travel_guidance payload must be a dict")
            cls._validate_exclusive_payload(payload)
        else:
            raw = payload

        kind = str(raw.get("kind", "none") or "none")
        mode = str(raw.get("mode", "") or "")
        strength_default = cls._default_strength(kind, mode)
        audio_raw = raw.get("audio")
        anchors_raw = raw.get("anchors", []) or []

        config = cls(
            kind=kind,  # type: ignore[arg-type]
            videos=[StructureVideoEntry.from_dict(v) for v in raw.get("videos", []) or []],
            strength=float(raw.get("strength", strength_default)),
            anchors=[AnchorEntry.from_dict(anchor) for anchor in anchors_raw],
            control_strength=float(raw.get("control_strength", 1.0)),
            audio=AudioConditioningConfig.from_dict(audio_raw) if isinstance(audio_raw, dict) else None,
            mode=mode,
            canny_intensity=float(raw.get("canny_intensity", 1.0)),
            depth_contrast=float(raw.get("depth_contrast", 1.0)),
            frame_policy=str(raw.get("frame_policy", "fit") or "fit"),
            zero_empty_frames=bool(raw.get("zero_empty_frames", True)),
            keep_on_gpu=bool(raw.get("keep_on_gpu", False)),
            _guidance_video_url=raw.get("_guidance_video_url"),
            _frame_offset=int(raw.get("_frame_offset", 0) or 0),
        )

        step_window = raw.get("step_window", [0.0, 1.0])
        if isinstance(step_window, (list, tuple)) and len(step_window) >= 2:
            config.step_window = (float(step_window[0]), float(step_window[1]))

        errors = config.validate(model_name)
        if errors:
            raise ValueError("; ".join(errors))

        return config

    def to_wgp_format(self) -> Dict[str, Any]:
        """Travel guidance is resolved before the WGP boundary."""
        if self.kind in {"vace", "uni3c"}:
            return self.to_structure_guidance_config().to_wgp_format()
        return {}

    @staticmethod
    def _validate_exclusive_payload(payload: Dict[str, Any]) -> None:
        """Reject mixed public travel-guidance contracts."""
        if _has_payload_value(payload.get("structure_guidance")):
            raise ValueError("travel_guidance cannot be combined with structure_guidance")

        conflicting = [
            key for key in _TRAVEL_GUIDANCE_LEGACY_KEYS if _has_payload_value(payload.get(key))
        ]
        if conflicting:
            raise ValueError(
                "travel_guidance cannot be combined with legacy structure guidance "
                f"fields: {sorted(conflicting)}"
            )

    @staticmethod
    def _default_strength(kind: str, mode: str) -> float:
        if kind == "ltx_control":
            if mode in ("video", "cameraman"):
                return 1.0
            return 0.5
        if kind in ("vace", "uni3c"):
            return 1.0
        if kind == "ltx_hybrid":
            return 0.0
        return 0.0

    def to_structure_guidance_config(self) -> StructureGuidanceConfig:
        """Bridge into the existing WGP-facing structure guidance config."""
        if self.kind not in ("vace", "uni3c"):
            raise ValueError(
                f"travel_guidance kind '{self.kind}' cannot be converted to StructureGuidanceConfig"
            )

        preprocessing = "none"
        target: Literal["vace", "uni3c"] = "vace"

        if self.kind == "vace":
            target = "vace"
            preprocessing = "none" if self.mode == "raw" else self.mode
        elif self.kind == "uni3c":
            target = "uni3c"

        return StructureGuidanceConfig(
            videos=list(self.videos),
            target=target,
            preprocessing=preprocessing,  # type: ignore[arg-type]
            strength=self.strength,
            canny_intensity=self.canny_intensity,
            depth_contrast=self.depth_contrast,
            step_window=self.step_window,
            frame_policy=self.frame_policy,
            zero_empty_frames=self.zero_empty_frames,
            keep_on_gpu=self.keep_on_gpu,
            _guidance_video_url=self._guidance_video_url,
            _frame_offset=self._frame_offset,
        )

    def needs_ic_lora(self) -> bool:
        """Return True when LTX control requires the union IC-LoRA."""
        if self.kind == "ltx_control":
            return self.mode in {"pose", "depth", "canny", "cameraman"}
        if self.kind == "ltx_hybrid":
            return self.has_control and self.mode in {"pose", "depth", "canny"}
        return False

    def get_preprocessor_type(self) -> str:
        """Return the preprocessor/compositing type used for this guidance."""
        if self.kind == "vace":
            return "raw" if self.mode == "raw" else self.mode
        if self.kind == "ltx_control":
            return "raw" if self.mode in {"video", "cameraman"} else self.mode
        if self.kind == "ltx_hybrid":
            if not self.has_control:
                return "raw"
            return "raw" if self.mode == "video" else self.mode
        if self.kind == "uni3c":
            return "uni3c"
        return "raw"

    def to_segment_payload(self, frame_offset: int) -> Dict[str, Any]:
        """Serialize to the segment-level ``travel_guidance`` payload."""
        if self.kind == "none":
            return {"kind": "none"}

        payload: Dict[str, Any] = {
            "kind": self.kind,
            "videos": [video.to_dict() for video in self.videos],
            "strength": self.strength,
            "_frame_offset": frame_offset,
        }

        if self.mode:
            payload["mode"] = self.mode
        if self.kind in {"vace", "ltx_control"}:
            payload["canny_intensity"] = self.canny_intensity
            payload["depth_contrast"] = self.depth_contrast
        if self.kind == "ltx_hybrid":
            payload["anchors"] = [anchor.to_dict() for anchor in self.anchors]
            payload["control_strength"] = self.control_strength
            payload["canny_intensity"] = self.canny_intensity
            payload["depth_contrast"] = self.depth_contrast
            if self.audio is not None:
                payload["audio"] = self.audio.to_dict()
        if self.kind == "uni3c":
            payload["step_window"] = list(self.step_window)
            payload["frame_policy"] = self.frame_policy
            payload["zero_empty_frames"] = self.zero_empty_frames
            payload["keep_on_gpu"] = self.keep_on_gpu
        if self._guidance_video_url:
            payload["_guidance_video_url"] = self._guidance_video_url

        return payload

    def validate(self, model_name: str) -> List[str]:
        """Validate the parsed config against payload and model constraints."""
        errors: List[str] = []
        allowed_kinds = _infer_allowed_kinds(model_name)

        if self.kind not in {"none", "vace", "ltx_control", "ltx_hybrid", "uni3c"}:
            errors.append(
                "Invalid travel_guidance.kind "
                f"'{self.kind}'. Must be one of none, vace, ltx_control, ltx_hybrid, uni3c"
            )
            return errors

        if self.kind not in allowed_kinds:
            errors.append(
                f"Model '{model_name}' does not support travel_guidance kind '{self.kind}'"
            )

        if self.kind == "none":
            return errors

        if self.kind in {"vace", "ltx_control", "uni3c"} and not self.videos:
            errors.append(f"travel_guidance kind '{self.kind}' requires at least one video")

        if self.kind == "ltx_hybrid" and not (self.has_control or self.has_anchors):
            errors.append("travel_guidance kind 'ltx_hybrid' requires anchors or videos")

        if self.kind == "vace" and self.mode not in {"flow", "canny", "depth", "raw"}:
            errors.append(
                "travel_guidance kind 'vace' requires mode to be one of flow, canny, depth, raw"
            )
        if self.kind == "ltx_control" and self.mode not in _LTX_CONTROL_MODES:
            errors.append(
                "travel_guidance kind 'ltx_control' requires mode to be one of pose, depth, canny, video"
            )
        if self.kind == "ltx_hybrid":
            if self.has_control and self.mode not in _LTX_CONTROL_MODES:
                errors.append(
                    "travel_guidance kind 'ltx_hybrid' requires mode to be one of pose, depth, canny, video when videos are present"
                )
            if not self.has_control and self.mode == "raw":
                errors.append("travel_guidance kind 'ltx_hybrid' does not accept mode='raw'")
        if self.kind == "uni3c" and self.mode:
            errors.append("travel_guidance kind 'uni3c' does not accept mode")

        if self.kind == "ltx_control" and not 0.0 <= self.strength <= 2.0:
            errors.append(
                f"travel_guidance strength for ltx_control must be within [0, 2], got {self.strength}"
            )
        elif self.strength < 0:
            errors.append(f"travel_guidance strength must be non-negative, got {self.strength}")

        if self.kind == "ltx_hybrid" and not 0.0 <= self.control_strength <= 1.0:
            errors.append(
                f"travel_guidance control_strength for ltx_hybrid must be within [0, 1], got {self.control_strength}"
            )

        if self.step_window[0] < 0 or self.step_window[1] > 1:
            errors.append(f"travel_guidance step_window must be within [0, 1], got {self.step_window}")
        if self.step_window[0] > self.step_window[1]:
            errors.append(
                f"travel_guidance step_window start must be <= end, got {self.step_window}"
            )

        for index, video in enumerate(self.videos):
            if not video.path:
                errors.append(f"travel_guidance video {index} has an empty path")

        if self.kind == "ltx_hybrid":
            for index, anchor in enumerate(self.anchors):
                if not anchor.image_url:
                    errors.append(f"travel_guidance anchor {index} has an empty image_url")
                if not 0.0 <= anchor.strength <= 1.0:
                    errors.append(
                        f"travel_guidance anchor {index} strength must be within [0, 1], got {anchor.strength}"
                    )

            if self.audio is not None:
                if self.audio.source not in {"external", "control_track"}:
                    errors.append(
                        "travel_guidance audio.source must be one of external, control_track"
                    )
                if self.audio.source == "control_track" and not self.has_control:
                    errors.append(
                        "travel_guidance audio.source='control_track' requires control videos"
                    )
                if self.audio.source == "external" and not self.audio.audio_url:
                    errors.append(
                        "travel_guidance audio.source='external' requires audio.audio_url"
                    )
                if not 0.0 <= self.audio.strength <= 1.0:
                    errors.append(
                        f"travel_guidance audio.strength must be within [0, 1], got {self.audio.strength}"
                    )

        return errors

    @property
    def has_guidance(self) -> bool:
        return self.kind != "none" and (self.has_anchors or self.has_control)

    @property
    def guidance_video_url(self) -> Optional[str]:
        return self._guidance_video_url

    @property
    def frame_offset(self) -> int:
        return self._frame_offset

    @property
    def is_uni3c(self) -> bool:
        return self.kind == "uni3c"

    @property
    def is_vace(self) -> bool:
        return self.kind == "vace"

    @property
    def is_ltx_control(self) -> bool:
        return self.kind == "ltx_control"

    @property
    def is_ltx_hybrid(self) -> bool:
        return self.kind == "ltx_hybrid"

    @property
    def has_control(self) -> bool:
        return bool(self.videos)

    @property
    def has_anchors(self) -> bool:
        return bool(self.anchors)

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @property
    def legacy_structure_type(self) -> str:
        if self.kind == "uni3c":
            return "uni3c"
        return self.get_preprocessor_type()

    def __repr__(self) -> str:
        return (
            "TravelGuidanceConfig("
            f"kind={self.kind!r}, "
            f"mode={self.mode!r}, "
            f"strength={self.strength}, "
            f"control_strength={self.control_strength}, "
            f"videos={len(self.videos)}, "
            f"anchors={len(self.anchors)}, "
            f"has_guidance={self.has_guidance})"
        )
