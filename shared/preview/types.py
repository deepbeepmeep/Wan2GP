from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from typing import Any, Literal


PreviewMode = Literal["off", "rgb", "tae"]
PreviewDevice = Literal["auto", "cuda", "cpu"]
PreviewUpdateRate = Literal["adaptive", "every_step", "every_2", "every_4"]


@dataclass(frozen=True)
class PreviewOptions:
    mode: PreviewMode = "rgb"
    device: PreviewDevice = "auto"
    update_rate: PreviewUpdateRate = "adaptive"
    max_edge: int = 512
    preview_fps: int = 16
    webp_quality: int = 72
    target_updates: int = 7

    @classmethod
    def from_value(cls, value: Any) -> "PreviewOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("preview options must be an object")
        data = value.get("_preview", value)
        if not isinstance(data, dict):
            raise ValueError("_preview must be an object")
        mode = str(data.get("mode", cls.mode)).strip().lower()
        device = str(data.get("device", cls.device)).strip().lower()
        update_rate = str(data.get("update_rate", cls.update_rate)).strip().lower()
        if mode not in {"off", "rgb", "tae"}:
            raise ValueError(f"unknown preview mode: {mode}")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"unknown preview device: {device}")
        if update_rate not in {"adaptive", "every_step", "every_2", "every_4"}:
            raise ValueError(f"unknown preview update rate: {update_rate}")

        def integer(name: str, default: int, low: int, high: int) -> int:
            try:
                return max(low, min(high, int(data.get(name, default))))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"preview option {name} must be an integer") from exc

        preview_fps = integer("preview_fps", cls.preview_fps, 2, 16)
        if preview_fps not in {2, 4, 8, 16}:
            raise ValueError("preview option preview_fps must be one of 2, 4, 8, or 16")
        return cls(
            mode=mode,
            device=device,
            update_rate=update_rate,
            max_edge=integer("max_edge", cls.max_edge, 128, 1024),
            preview_fps=preview_fps,
            webp_quality=integer("webp_quality", cls.webp_quality, 1, 100),
            target_updates=integer("target_updates", cls.target_updates, 2, 16),
        )

    def with_mode(self, mode: PreviewMode) -> "PreviewOptions":
        return replace(self, mode=mode)


@dataclass(frozen=True)
class PreviewContext:
    generation_id: str
    context_id: str
    sequence: int
    model_type: str
    architecture: str
    decoder_id: str | None
    step: int
    total_steps: int
    pass_no: int | None = None
    window_no: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreviewMedia:
    generation_id: str
    context_id: str
    sequence: int
    media_kind: Literal["image", "animated_image", "video"]
    mime_type: str
    data: bytes
    width: int
    height: int
    frame_count: int
    fps: float | None
    duration_ms: int | None
    step: int
    total_steps: int
    decoder_id: str
    decode_ms: float
    encode_ms: float
    dropped_count: int = 0
    warning: str | None = None
    pass_no: int | None = None
    window_no: int | None = None
    first_frame: Any = field(default=None, compare=False, repr=False)

    def without_frame(self) -> "PreviewMedia":
        return replace(self, first_frame=None)

    def to_dict(self, *, encode_data: bool = True) -> dict[str, Any]:
        """Return a JSON-bridge-friendly representation without serializing the PIL fallback frame."""
        data: bytes | str = self.data
        if encode_data:
            data = base64.b64encode(data).decode("ascii")
        return {
            "generation_id": self.generation_id,
            "context_id": self.context_id,
            "sequence": self.sequence,
            "media_kind": self.media_kind,
            "mime_type": self.mime_type,
            "data": data,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration_ms": self.duration_ms,
            "step": self.step,
            "total_steps": self.total_steps,
            "decoder_id": self.decoder_id,
            "decode_ms": self.decode_ms,
            "encode_ms": self.encode_ms,
            "dropped_count": self.dropped_count,
            "warning": self.warning,
            "pass_no": self.pass_no,
            "window_no": self.window_no,
        }
