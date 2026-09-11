"""Windowed temporal repair composed from WanGP's H3 video-to-video API.

The planner measures unusually large latent transitions against nearby motion.
Each selected source frame occupies several frames during denoising; sampling
the repaired sequence at the original boundaries restores the source timeline.

Independently implemented from the behavior specification and existing WanGP
APIs, without reading upstream or the earlier temporal refiner implementation.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.minimax_h3.components.packing import _FRAME_PER_TOKEN
from models.minimax_h3.interrupt import GenerationInterrupted
from shared.utils.frame_scheduler import normalize_frame_count


NATIVE_CLIP = sum(_FRAME_PER_TOKEN)
MAX_WINDOW = 360


def latent_starts(count):
    """Pixel-frame starts of H3 temporal tokens, on CPU."""
    spans = torch.tensor(_FRAME_PER_TOKEN, dtype=torch.long, device="cpu").repeat((count + 4) // 5)[:count]
    return spans.cumsum(0) - spans


def frame_durations(latents, frame_count, maximum):
    """Return CPU integer durations from BCTHW H3 latents (one batch)."""
    latent = latents.detach().to(device="cpu", dtype=torch.float32)[0]
    count = latent.shape[1]
    durations = torch.ones(frame_count, dtype=torch.long, device="cpu")
    phase_count = len(_FRAME_PER_TOKEN)
    if count <= phase_count or maximum == 1:
        return durations
    # Adjacent H3 latent phases differ even for a perfectly static source.
    # Compare equal phases in neighboring native clips to measure scene motion.
    energy = (latent[:, phase_count:] - latent[:, :-phase_count]).square().mean(dim=(0, 2, 3)).sqrt()
    padded = F.pad(energy[None, None], (4, 4), mode="replicate")[0, 0]
    baseline = padded.unfold(0, 9, 1).median(dim=1).values
    # H3 latents are normalized. One tenth of a latent standard deviation
    # rejects encoding jitter; the absolute term also catches sustained action.
    activity = ((energy - 0.1) / 0.75).clamp(0, 1)
    abruptness = ((energy - baseline * 1.5 - 0.1) / (baseline + 0.1)).clamp(0, 1)
    transition_duration = 1 + torch.ceil(torch.maximum(activity, abruptness) * (maximum - 1)).long()
    latent_duration = torch.ones(count, dtype=torch.long, device="cpu")
    latent_duration[:-phase_count] = transition_duration
    latent_duration[phase_count:] = torch.maximum(latent_duration[phase_count:], transition_duration)
    starts = latent_starts(count + 1).tolist()
    for index in range(count):
        durations[starts[index]:min(starts[index + 1], frame_count)] = latent_duration[index]
    return durations


def plan_windows(durations, context=NATIVE_CLIP, limit=MAX_WINDOW):
    """Return (context_start, body_start, body_stop, context_stop) intervals.

    Bodies are disjoint. Nearby selected frames share a body, while each body
    receives a native clip of source context on both sides where available.
    The bound includes repetition, context and H3 frame-count normalization.
    """
    active = torch.nonzero(durations > 1).flatten().tolist()
    if not active:
        return []
    runs = []
    start = previous = active[0]
    for position in active[1:]:
        if position - previous > context:
            runs.append((start, previous + 1))
            start = position
        previous = position
    runs.append((start, previous + 1))
    windows = []
    frames = durations.numel()
    for start, stop in runs:
        while start < stop:
            left = max(0, start - context)
            end, expanded = start, start - left
            while end < stop:
                candidate = expanded + int(durations[end])
                right = min(frames, end + 1 + context)
                if normalize_frame_count(candidate + right - end - 1, 5, 17, 5) > limit:
                    break
                expanded, end = candidate, end + 1
            if end == start:
                raise ValueError("Temporal repair window limit cannot fit one frame and its context")
            windows.append((left, start, end, min(frames, end + context)))
            start = end
    return windows


def inverse_clock(durations, expanded_frames):
    """Evaluate inverse duration map at H3 token starts, in source-frame units."""
    boundaries = torch.cat((torch.zeros(1, dtype=torch.long, device="cpu"), durations.cumsum(0)))
    token_count = 2 + (normalize_frame_count(expanded_frames, 5, 17, 5) - 5) // 17 * 5
    positions = latent_starts(token_count).clamp(max=int(boundaries[-1]))
    source = torch.searchsorted(boundaries[1:], positions, right=True).clamp(max=durations.numel() - 1)
    return source.double() + (positions - boundaries[source]).double() / durations[source]


def stretch_audio(waveform, sample_rate, fps, start, durations, expanded_frames):
    """Pitch-preserving conditioning audio; samples x channels in/out."""
    from torchaudio.functional import phase_vocoder

    audio = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = audio.T
    pieces = []
    window = torch.hann_window(1024, device="cpu")
    phase = torch.linspace(0, torch.pi * 256, 513, device="cpu")[:, None]
    offset, expanded = 0, 0
    for duration, length in zip(*torch.unique_consecutive(durations, return_counts=True)):
        duration, length = int(duration), int(length)
        first, last = round((start + offset) * sample_rate / fps), round((start + offset + length) * sample_rate / fps)
        piece = audio[:, first:last]
        piece = F.pad(piece, (0, last - first - piece.shape[-1]))
        target = round((expanded + length * duration) * sample_rate / fps) - round(expanded * sample_rate / fps)
        if duration > 1:
            spectrum = torch.stft(piece, n_fft=1024, hop_length=256, window=window, pad_mode="constant", return_complex=True)
            spectrum = phase_vocoder(spectrum, 1.0 / duration, phase)
            piece = torch.istft(spectrum, n_fft=1024, hop_length=256, window=window, length=target)
        else:
            piece = F.pad(piece[..., :target], (0, max(0, target - piece.shape[-1])))
        pieces.append(piece)
        offset += length
        expanded += length * duration
    audio = torch.cat(pieces, dim=-1)
    target = round(expanded_frames * sample_rate / fps)
    return F.pad(audio[..., :target], (0, max(0, target - audio.shape[-1]))).T.numpy()


@contextmanager
def base_denoiser(pipeline, check_abort):
    from mmgp import offload

    model = pipeline.transformer
    cache = model.cache
    # These optional attributes are owned by MMGP and are absent before LoRA use.
    adapters = list(getattr(model, "_loras_active_adapters", ()))
    scaling = getattr(model, "_loras_scaling", {})
    multipliers = [scaling[name] for name in adapters]
    step = getattr(model, "_lora_step_no", 0)
    hooks = []
    try:
        model.cache = None
        offload.activate_loras(model, [])
        for block in model.blocks:
            hooks.append(block.register_forward_pre_hook(lambda _module, _inputs: check_abort()))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        model.cache = cache
        offload.activate_loras(model, adapters, multipliers)
        offload.set_step_no_for_lora(model, step)


@torch.inference_mode()
def repair(video, *, pipeline, strength, maximum, prompt, fps, seed, latents=None, audio_waveform=None,
           audio_sample_rate=32000, reference_images=None, vae_tile_size=None, abort_callback=None,
           progress_callback=None, dyrope=False):
    """Repair CTHW video, preserving shape, dtype and original frame timing."""
    if not 0 <= strength <= 1 or maximum not in (1, 2, 3, 4) or int(maximum) != maximum:
        raise ValueError("Temporal repair requires strength in [0, 1] and integer maximum in [1, 4]")
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] < 1:
        raise ValueError("Temporal repair requires nonempty RGB CTHW video")
    if fps <= 0:
        raise ValueError("Temporal repair requires positive fps")
    if strength == 0 or maximum == 1:
        return video

    def check_abort():
        if abort_callback is not None and abort_callback():
            pipeline._interrupt = True
        pipeline._check_abort()

    def report(status, current=None, total=None):
        check_abort()
        if progress_callback is not None:
            progress_callback(status, current, total)

    source = video.detach().to(device="cpu")
    output = source.clone()
    spatial_padding = (0, -source.shape[-1] % 32, 0, -source.shape[-2] % 32)

    def pixels(chunk):
        chunk = chunk.float()
        return chunk.div(127.5).sub(1) if source.dtype == torch.uint8 else chunk

    check_abort()
    pipeline._use_shared_components()
    pipeline._configure_tiling(vae_tile_size)
    if latents is None:
        encoded = []
        clip_count = (source.shape[1] + NATIVE_CLIP - 1) // NATIVE_CLIP
        for index in tqdm(range(clip_count), desc="H3 temporal activity"):
            report("Analyzing temporal activity", index, clip_count)
            chunk = F.pad(pixels(source[:, index * NATIVE_CLIP:(index + 1) * NATIVE_CLIP]), spatial_padding, mode="replicate")
            chunk = F.pad(chunk, (0, 0, 0, 0, 0, NATIVE_CLIP - chunk.shape[1]), mode="replicate")
            encoded.append(pipeline._encode_video(chunk, keep_all_latents=True))
        latents = torch.cat(encoded, dim=2)
        del encoded, chunk
    durations = frame_durations(latents, source.shape[1], int(maximum))
    del latents
    windows = plan_windows(durations)
    if not windows:
        report("Temporal repair complete", 0, 0)
        return output
    steps = max(1, round(25 * strength))
    start_sigma = 12 * (steps / 25) / (1 + 11 * (steps / 25))
    with base_denoiser(pipeline, check_abort):
        for number, (left, start, stop, right) in enumerate(tqdm(windows, desc="H3 temporal repair")):
            report(f"Repairing temporal window {number + 1}/{len(windows)}", 0, steps)
            local = torch.ones(right - left, dtype=torch.long, device="cpu")
            local[start - left:stop - left] = durations[start:stop]
            boundaries = local.cumsum(0) - local
            indices = torch.repeat_interleave(torch.arange(left, right, device="cpu"), local)
            count = normalize_frame_count(indices.numel(), 5, 17, 5)
            indices = F.pad(indices, (0, count - indices.numel()), value=right - 1)
            expanded = F.pad(pixels(source[:, indices]), spatial_padding, mode="replicate")
            mask = ((indices >= start) & (indices < stop)).float().view(1, -1, 1, 1)
            audio = None if audio_waveform is None else stretch_audio(audio_waveform, audio_sample_rate, fps, left, local, count)

            def callback(index, _preview, _initial, **_kwargs):
                report(f"Repairing temporal window {number + 1}/{len(windows)}", max(0, int(index) + 1), steps)

            result = pipeline.generate(
                input_prompt=prompt or "Preserve the source scene, subjects and appearance with smooth, natural motion.",
                input_frames=expanded, input_masks=mask, input_ref_images=reference_images if pipeline.reference_mode else None,
                input_waveform=audio, input_waveform_sample_rate=audio_sample_rate, audio_prompt_type="A" if audio is not None else "",
                frame_num=count, height=expanded.shape[-2], width=expanded.shape[-1], fps=fps, seed=int(seed) + number,
                sampling_steps=steps, shift=12.0, starting_sigma=start_sigma, sample_solver="res_multistep",
                denoising_strength=1.0, masking_strength=1.0, refinement_mode=True, guide_phases=1,
                VAE_tile_size=vae_tile_size, callback=callback, set_progress_status=report,
                temporal_rope_clock=inverse_clock(local, count) if dyrope else None)
            del expanded, audio, mask, indices
            if result is None:
                raise GenerationInterrupted
            check_abort()
            repaired = result["x"].detach().to(device="cpu")[:, boundaries, :source.shape[-2], :source.shape[-1]]
            del result
            alpha = torch.ones(right - left, dtype=torch.float32, device="cpu")
            if start > left:
                alpha[:start - left] = torch.linspace(0, 1, start - left + 1, device="cpu")[:-1]
            if stop < right:
                alpha[stop - left:] = torch.linspace(1, 0, right - stop + 1, device="cpu")[1:]
            # Context from a later window must not overwrite another body.
            alpha[:start - left].masked_fill_(durations[left:start] > 1, 0)
            alpha[stop - left:].masked_fill_(durations[stop:right] > 1, 0)
            mixed = torch.lerp(pixels(output[:, left:right]), repaired.float(), alpha[None, :, None, None])
            if source.dtype == torch.uint8:
                mixed = mixed.clamp(-1, 1).add_(1).mul_(127.5).round_()
            output[:, left:right] = mixed.to(source.dtype)
            del repaired, mixed
    report("Temporal repair complete", len(windows) * steps, len(windows) * steps)
    return output
