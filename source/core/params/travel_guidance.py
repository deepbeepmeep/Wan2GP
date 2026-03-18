"""Travel guidance contract for orchestrator and segment payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from .base import ParamGroup
from .structure_guidance import StructureGuidanceConfig, StructureVideoEntry

_TRAVEL_GUIDANCE_KIND = Literal["none", "vace", "ltx_control", "uni3c"]
_TRAVEL_GUIDANCE_LEGACY_KEYS = (
    "structure_type",
    "structure_video_path",
    "structure_videos",
    "use_uni3c",
)


def _has_payload_value(value: Any) -> bool:
    """Return True when a payload field should count as "present".

    Treats None, empty collections, empty strings, and False as absent.
    This matters for boolean legacy keys like ``use_uni3c`` where
    ``False`` is the inactive/default state and should not conflict
    with ``travel_guidance``.
    """
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
            from wgp import get_model_def

            model_def = get_model_def(model_name)
            if isinstance(model_def, dict):
                pipeline_kind = model_def.get("ltx2_pipeline")
        except (ImportError, TypeError, ValueError, KeyError, AttributeError):
            pipeline_kind = None

        is_distilled = pipeline_kind == "distilled" or (
            pipeline_kind is None and "distilled" in lowered
        )
        if is_distilled:
            return {"none", "ltx_control", "uni3c"}
        return {"none"}

    # Preserve existing Wan/VACE-style travel guidance behavior for non-LTX models.
    return {"none", "vace", "uni3c"}


@dataclass
class TravelGuidanceConfig(ParamGroup):
    """Discriminated union for travel guidance."""

    kind: _TRAVEL_GUIDANCE_KIND = "none"

    # Shared fields for all non-none kinds
    videos: List[StructureVideoEntry] = field(default_factory=list)
    strength: float = 0.0

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
        """ParamGroup entry point."""
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

        config = cls(
            kind=kind,  # type: ignore[arg-type]
            videos=[StructureVideoEntry.from_dict(v) for v in raw.get("videos", []) or []],
            strength=float(raw.get("strength", strength_default)),
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
            raise ValueError(
                "travel_guidance cannot be combined with structure_guidance"
            )

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
            return 1.0 if mode == "video" else 0.5
        if kind in ("vace", "uni3c"):
            return 1.0
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
        return self.kind == "ltx_control" and self.mode in {"pose", "depth", "canny"}

    def get_preprocessor_type(self) -> str:
        """Return the preprocessor/compositing type used for this guidance."""
        if self.kind == "vace":
            return "raw" if self.mode == "raw" else self.mode
        if self.kind == "ltx_control":
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

        if self.kind not in {"none", "vace", "ltx_control", "uni3c"}:
            errors.append(
                f"Invalid travel_guidance.kind '{self.kind}'. Must be one of none, vace, ltx_control, uni3c"
            )
            return errors

        if self.kind not in allowed_kinds:
            errors.append(
                f"Model '{model_name}' does not support travel_guidance kind '{self.kind}'"
            )

        if self.kind == "none":
            return errors

        if not self.videos:
            errors.append(f"travel_guidance kind '{self.kind}' requires at least one video")

        if self.kind == "vace" and self.mode not in {"flow", "canny", "depth", "raw"}:
            errors.append(
                "travel_guidance kind 'vace' requires mode to be one of flow, canny, depth, raw"
            )
        if self.kind == "ltx_control" and self.mode not in {"pose", "depth", "canny", "video"}:
            errors.append(
                "travel_guidance kind 'ltx_control' requires mode to be one of pose, depth, canny, video"
            )
        if self.kind == "uni3c" and self.mode:
            errors.append("travel_guidance kind 'uni3c' does not accept mode")

        if self.kind == "ltx_control" and not 0.0 <= self.strength <= 1.0:
            errors.append(
                f"travel_guidance strength for ltx_control must be within [0, 1], got {self.strength}"
            )
        elif self.strength < 0:
            errors.append(f"travel_guidance strength must be non-negative, got {self.strength}")

        if self.step_window[0] < 0 or self.step_window[1] > 1:
            errors.append(f"travel_guidance step_window must be within [0, 1], got {self.step_window}")
        if self.step_window[0] > self.step_window[1]:
            errors.append(
                f"travel_guidance step_window start must be <= end, got {self.step_window}"
            )

        for index, video in enumerate(self.videos):
            if not video.path:
                errors.append(f"travel_guidance video {index} has an empty path")

        return errors

    @property
    def has_guidance(self) -> bool:
        return self.kind != "none" and (bool(self.videos) or bool(self._guidance_video_url))

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
            f"videos={len(self.videos)}, "
            f"has_guidance={self.has_guidance})"
        )
