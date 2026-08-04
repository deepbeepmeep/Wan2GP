from dataclasses import dataclass

import torch

from shared.prompt_relay import parse_prompt_relay
from .modules.posemb_layers import get_1d_rotary_pos_embed


@dataclass(frozen=True)
class ShotPlanPrompt:
    prompt: str
    cut_frames: tuple[int, ...]


def compile_shotplan_prompt(prompt: str, num_frames: int, fps: float) -> ShotPlanPrompt:
    plan = parse_prompt_relay(prompt)
    if plan is None:
        return ShotPlanPrompt(prompt, ())

    total_seconds = (num_frames - 1) / fps
    ranges = []
    for segment in plan.segments:
        start = segment.start.resolve(total_seconds, num_frames)
        end = 1.0 if segment.end is None else segment.end.resolve(total_seconds, num_frames, inclusive_end=True)
        ranges.append((start, end))

    if ranges[0][0] != 0:
        raise ValueError("The first ShotPlan relay segment must start at the beginning of the video.")
    for previous, current in zip(ranges, ranges[1:]):
        if abs(previous[1] - current[0]) > 1e-6:
            raise ValueError("ShotPlan relay segments must be contiguous and must not overlap.")
    if abs(ranges[-1][1] - 1.0) > 1e-6:
        raise ValueError("The final ShotPlan relay segment must reach the end of the video.")

    cut_frames = tuple(round(start * (num_frames - 1)) for start, _ in ranges[1:])
    if len(set(cut_frames)) != len(cut_frames):
        raise ValueError("Each ShotPlan relay segment must begin on a distinct output frame.")

    shot_prompts = "\n".join(f"Shot {index}: {segment.prompt}" for index, segment in enumerate(plan.segments, 1))
    compiled_prompt = f"{plan.global_prompt}\n{shot_prompts}" if plan.global_prompt else shot_prompts
    return ShotPlanPrompt(compiled_prompt, cut_frames)


def inject_shotplan_tokens(x, freqs, hardcut_embedding, cut_frames, grid_sizes, vae_scale=4):
    frames, height, width = (int(value) for value in grid_sizes)
    tokens_per_frame = height * width
    cut_positions = tuple(1.0 + frame / vae_scale for frame in cut_frames)
    token = hardcut_embedding.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
    cos, sin = freqs
    position_dtype = cos.dtype
    zero = torch.zeros(1, device=cos.device, dtype=position_dtype)
    spatial_cos_h, spatial_sin_h = get_1d_rotary_pos_embed(42, zero, use_real=True)
    spatial_cos_w, spatial_sin_w = get_1d_rotary_pos_embed(42, zero, use_real=True)

    x_parts, cos_parts, sin_parts, keep_parts = [], [], [], []
    for frame in range(frames):
        start = frame * tokens_per_frame
        end = start + tokens_per_frame
        x_parts.append(x[:, start:end])
        cos_parts.append(cos[start:end])
        sin_parts.append(sin[start:end])
        keep_parts.append(torch.ones(tokens_per_frame, device=x.device, dtype=torch.bool))
        for position in cut_positions:
            if frame < position <= frame + 1:
                temporal_position = torch.tensor([position], device=cos.device, dtype=position_dtype)
                temporal_cos, temporal_sin = get_1d_rotary_pos_embed(44, temporal_position, use_real=True)
                x_parts.append(token)
                cos_parts.append(torch.cat((temporal_cos, spatial_cos_h, spatial_cos_w), dim=1))
                sin_parts.append(torch.cat((temporal_sin, spatial_sin_h, spatial_sin_w), dim=1))
                keep_parts.append(torch.zeros(1, device=x.device, dtype=torch.bool))

    return torch.cat(x_parts, dim=1), (torch.cat(cos_parts), torch.cat(sin_parts)), torch.cat(keep_parts)
