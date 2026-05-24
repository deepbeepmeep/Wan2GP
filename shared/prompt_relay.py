from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

_RELAY_MARKER_RE = re.compile(r"\[([^\]]+)\]")
_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")
_SECONDS_RE = re.compile(r"^(\d+(?:\.\d+)?)(?:s|sec|secs|second|seconds)$", re.IGNORECASE)

__all__ = [
    "PromptRelayBound",
    "PromptRelayConditioning",
    "PromptRelayMaskBuilder",
    "PromptRelayPlan",
    "PromptRelaySegment",
    "encode_prompt_relay",
    "parse_prompt_relay",
]


@dataclass(frozen=True)
class PromptRelayBound:
    value: float
    unit: str

    def resolve(self, total_seconds: float, total_frames: int, inclusive_end: bool = False) -> float:
        if self.unit == "percent":
            return max(0.0, min(1.0, self.value))
        if self.unit == "frame":
            if total_frames <= 1:
                return 0.0
            frame_index = self.value if inclusive_end else self.value - 1.0
            frame_index = max(frame_index, 0.0)
            return max(0.0, min(1.0, frame_index / float(total_frames - 1)))
        if total_seconds <= 0:
            return 0.0
        return max(0.0, min(1.0, self.value / total_seconds))


@dataclass(frozen=True)
class PromptRelaySegment:
    start: PromptRelayBound
    end: PromptRelayBound | None
    prompt: str
    key_start: int = 0
    key_end: int = 0


@dataclass(frozen=True)
class PromptRelayPlan:
    global_prompt: str
    segments: tuple[PromptRelaySegment, ...]


@dataclass(frozen=True)
class _RuntimeSegment:
    start: float
    end: float
    key_start: int
    key_end: int


@dataclass(frozen=True)
class PromptRelayConditioning:
    video_context: torch.Tensor
    audio_context: torch.Tensor | None
    video_mask_builder: PromptRelayMaskBuilder | None
    audio_mask_builder: PromptRelayMaskBuilder | None

    @property
    def mask_builder(self) -> PromptRelayMaskBuilder | None:
        return self.video_mask_builder


class PromptRelayMaskBuilder:
    def __init__(
        self,
        key_valid: torch.Tensor,
        segments: list[_RuntimeSegment],
        positive_key_count: int,
        visible_start_ratio: float = 0.0,
        epsilon: float = 1e-3,
        padding_bias: float = -100.0,
    ) -> None:
        self.key_valid = key_valid.detach().to("cpu", dtype=torch.bool)
        self.segments = tuple(segments)
        self.positive_key_count = int(positive_key_count)
        self.visible_start_ratio = max(0.0, min(1.0, float(visible_start_ratio)))
        self.sigma = float(1.0 / math.log(1.0 / epsilon)) if 0 < epsilon < 1 else 0.1448
        self.padding_bias = float(padding_bias)

    def __call__(self, state: Any, frame_indices: torch.Tensor | None, context: Any) -> torch.Tensor | None:
        if not self.segments:
            return None
        context_len = _context_seq_len(context)
        if context_len <= 0:
            return None
        device = state.latent.device
        dtype = state.latent.dtype if torch.is_floating_point(state.latent) else torch.float32
        frame_indices = _resolve_frame_indices(state, frame_indices).to(device=device)
        batch_size, query_len = frame_indices.shape
        key_valid = self.key_valid.to(device=device)
        if key_valid.numel() < self.positive_key_count:
            key_valid = torch.cat([key_valid, torch.ones(self.positive_key_count - key_valid.numel(), device=device, dtype=torch.bool)])
        key_valid = key_valid[: self.positive_key_count]
        if context_len > self.positive_key_count:
            key_valid = torch.cat([key_valid, torch.ones(context_len - self.positive_key_count, device=device, dtype=torch.bool)])
        else:
            key_valid = key_valid[:context_len]

        positive_len = min(self.positive_key_count, context_len)
        mask = torch.zeros((batch_size, query_len, context_len), device=device, dtype=torch.float32)

        raw_query_frames = frame_indices.to(torch.float32)
        raw_max_frame = raw_query_frames.amax(dim=1, keepdim=True).clamp_min(1.0)
        visible_start = raw_max_frame * self.visible_start_ratio
        query_frames = raw_query_frames - visible_start
        max_frame = (raw_max_frame - visible_start).clamp_min(1.0)
        sigma_sq = 2.0 * self.sigma * self.sigma
        for segment in self.segments:
            start_key = min(segment.key_start, positive_len)
            end_key = min(segment.key_end, positive_len)
            if start_key >= end_key:
                continue
            start = torch.tensor(segment.start, device=device, dtype=torch.float32) * max_frame
            end = torch.tensor(segment.end, device=device, dtype=torch.float32) * max_frame
            length = (end - start).clamp_min(1.0)
            midpoint = (start + end) * 0.5
            window = (length * 0.5 - 2.0).clamp_min(0.0)
            distance = (query_frames - midpoint).abs()
            cost = torch.relu(distance - window).square() / sigma_sq
            mask[:, :, start_key:end_key] = -cost.unsqueeze(-1)

        if key_valid.numel() < context_len:
            key_valid = torch.cat([key_valid, torch.zeros(context_len - key_valid.numel(), device=device, dtype=torch.bool)])
        mask[:, :, ~key_valid[:context_len]] = self.padding_bias
        return mask.to(dtype=dtype).unsqueeze(2)


def parse_prompt_relay(prompt: str) -> PromptRelayPlan | None:
    current_bounds: tuple[PromptRelayBound, PromptRelayBound | None] | None = None
    last_valid_end = 0
    global_parts: list[str] = []
    segments: list[PromptRelaySegment] = []
    for match in _RELAY_MARKER_RE.finditer(prompt):
        bounds = _parse_marker(match.group(1))
        if bounds is None:
            continue
        if current_bounds is None:
            global_parts.append(prompt[last_valid_end : match.start()])
        else:
            segment_prompt = prompt[last_valid_end : match.start()].strip()
            if segment_prompt:
                segments.append(PromptRelaySegment(current_bounds[0], current_bounds[1], segment_prompt))
        current_bounds = bounds
        last_valid_end = match.end()
    if current_bounds is None:
        return None
    segment_prompt = prompt[last_valid_end:].strip()
    if segment_prompt:
        segments.append(PromptRelaySegment(current_bounds[0], current_bounds[1], segment_prompt))
    if not segments:
        return None
    return PromptRelayPlan("".join(global_parts).strip(), tuple(segments))


def encode_prompt_relay(
    prompt: str,
    encode_fn: Callable[[list[str]], list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]]],
    text_encoder_cache: Any,
    device: torch.device | str,
    num_frames: int,
    frame_rate: float,
    tokenizer: Any,
    visible_frame_offset: int = 0,
) -> PromptRelayConditioning | None:
    plan = parse_prompt_relay(prompt)
    if plan is None:
        return None
    full_prompt, token_ranges = _build_full_prompt_and_token_ranges(plan, tokenizer)
    encoded = text_encoder_cache.encode(
        encode_fn,
        [full_prompt],
        device=device,
        parallel=True,
        cache_keys=[("prompt_relay_full", full_prompt)],
    )[0]
    video_context, audio_context, video_mask, audio_mask = encoded
    return PromptRelayConditioning(
        video_context=video_context,
        audio_context=audio_context,
        video_mask_builder=_build_mask_builder(plan, video_context, video_mask, token_ranges, num_frames, frame_rate, visible_frame_offset),
        audio_mask_builder=None if audio_context is None else _build_mask_builder(plan, audio_context, audio_mask, token_ranges, num_frames, frame_rate, visible_frame_offset),
    )


def _build_mask_builder(
    plan: PromptRelayPlan,
    context: torch.Tensor,
    mask: torch.Tensor,
    token_ranges: list[tuple[int, int]],
    num_frames: int,
    frame_rate: float,
    visible_frame_offset: int = 0,
) -> PromptRelayMaskBuilder | None:
    runtime_segments = []
    num_frames = max(int(num_frames), 1)
    visible_frame_offset = min(max(int(visible_frame_offset), 0), max(num_frames - 1, 0))
    visible_num_frames = max(num_frames - visible_frame_offset, 1)
    visible_start_ratio = float(visible_frame_offset) / float(num_frames - 1) if num_frames > 1 else 0.0
    total_seconds = max((visible_num_frames - 1) / max(float(frame_rate), 1e-6), 0.0)
    seq_len = _seq_len(context)
    for segment, (start_key, end_key) in zip(plan.segments, token_ranges, strict=True):
        start_key = min(max(int(start_key), 0), seq_len)
        end_key = min(max(int(end_key), start_key), seq_len)
        if start_key >= end_key:
            continue
        start = segment.start.resolve(total_seconds, visible_num_frames)
        end = 1.0 if segment.end is None else segment.end.resolve(total_seconds, visible_num_frames, inclusive_end=True)
        end = max(start, end)
        runtime_segments.append(_RuntimeSegment(start, end, start_key, end_key))
    if not any(segment.start > 0.0 or segment.end < 1.0 for segment in runtime_segments):
        return None
    return PromptRelayMaskBuilder(_normalize_key_mask(mask, seq_len), runtime_segments, seq_len, visible_start_ratio=visible_start_ratio)


def _parse_marker(marker: str) -> tuple[PromptRelayBound, PromptRelayBound | None] | None:
    candidate = None
    for index, char in enumerate(marker):
        if char != ":":
            continue
        start = _parse_bound(marker[:index].strip())
        if start is None:
            continue
        end_text = marker[index + 1 :].strip()
        end = None if not end_text else _parse_bound(end_text)
        if end_text and end is None:
            continue
        if end is not None and end.unit != start.unit:
            continue
        if end is not None and end.value < start.value:
            continue
        candidate = (start, end)
    return candidate


def _parse_bound(text: str) -> PromptRelayBound | None:
    if not text:
        return None
    if text.endswith("%"):
        value = text[:-1].strip()
        return PromptRelayBound(float(value) / 100.0, "percent") if _NUMERIC_RE.match(value) else None
    seconds_match = _SECONDS_RE.match(text)
    if seconds_match:
        return PromptRelayBound(float(seconds_match.group(1)), "seconds")
    if ":" in text:
        parts = text.split(":")
        if not all(_NUMERIC_RE.match(part) for part in parts):
            return None
        total = 0.0
        for part in parts:
            total = total * 60.0 + float(part)
        return PromptRelayBound(total, "seconds")
    if _NUMERIC_RE.match(text):
        return PromptRelayBound(float(text), "frame")
    return None


def _build_full_prompt_and_token_ranges(plan: PromptRelayPlan, tokenizer: Any) -> tuple[str, list[tuple[int, int]]]:
    full_prompt = plan.global_prompt.strip()
    token_ranges: list[tuple[int, int]] = []
    for segment in plan.segments:
        separator = "\n" if full_prompt else ""
        prefix = full_prompt + separator
        full_prompt = prefix + segment.prompt.strip()
        token_ranges.append((_content_token_count(tokenizer, prefix), _content_token_count(tokenizer, full_prompt)))
    return full_prompt, token_ranges


def _content_token_count(tokenizer: Any, text: str) -> int:
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    max_length = int(getattr(tokenizer, "max_length", getattr(raw_tokenizer, "model_max_length", 1024)))
    encoded = raw_tokenizer(
        text.strip(),
        padding=False,
        max_length=max_length,
        truncation=True,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    eos_token_id = getattr(raw_tokenizer, "eos_token_id", None)
    if eos_token_id is not None and input_ids and input_ids[-1] == eos_token_id:
        input_ids = input_ids[:-1]
    return len(input_ids)


def _seq_len(context: torch.Tensor) -> int:
    return int(context.shape[0] if context.dim() == 2 else context.shape[1])


def _normalize_key_mask(mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    mask = mask.detach().to("cpu", dtype=torch.bool).reshape(-1)
    if mask.numel() < seq_len:
        mask = torch.cat([mask, torch.ones(seq_len - mask.numel(), dtype=torch.bool)])
    return mask[:seq_len]


def _context_seq_len(context: Any) -> int:
    tensor = getattr(context, "projected_context", None)
    if tensor is None:
        tensor = getattr(context, "context", context)
    if tensor is None:
        return 0
    return _seq_len(tensor)


def _resolve_frame_indices(state: Any, frame_indices: torch.Tensor | None) -> torch.Tensor:
    if frame_indices is not None:
        return frame_indices
    positions = state.positions
    if positions is not None and positions.ndim >= 4 and positions.shape[1] > 0:
        frame_times = positions[:, 0, :, 0]
        changes = torch.cat(
            [torch.zeros((frame_times.shape[0], 1), device=frame_times.device, dtype=torch.long), (frame_times[:, 1:] != frame_times[:, :-1]).to(torch.long)],
            dim=1,
        )
        return torch.cumsum(changes, dim=1)
    return torch.zeros((state.latent.shape[0], state.latent.shape[1]), device=state.latent.device, dtype=torch.long)
