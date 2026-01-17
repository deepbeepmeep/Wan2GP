import gc
import inspect
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mmgp import offload

from ...ltx_core.components.noisers import Noiser
from ...ltx_core.components.protocols import DiffusionStepProtocol, GuiderProtocol
from ...ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)
from ...ltx_core.model.transformer import Modality, X0Model
from ...ltx_core.model.video_vae import VideoEncoder, TilingConfig, encode_video as vae_encode_video
from ...ltx_core.text_encoders.gemma import GemmaTextEncoderModelBase
from ...ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ...ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ...ltx_core.utils import to_denoised, to_velocity
from .media_io import decode_image, load_image_conditioning, load_video_conditioning, resize_aspect_ratio_preserving
from .types import (
    DenoisingFunc,
    DenoisingLoopFunc,
    PipelineComponents,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def image_conditionings_by_replacing_latent(
    images: list[tuple],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    conditionings = []
    for image_entry in images:
        if len(image_entry) == 4:
            image_path, frame_idx, strength, resample = image_entry
        else:
            image_path, frame_idx, strength = image_entry
            resample = None
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            resample=resample,
        )
        encoded_image = vae_encode_video(image, video_encoder, tiling_config)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=strength,
                latent_idx=frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[tuple],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    conditionings = []
    for image_entry in images:
        if len(image_entry) == 4:
            image_path, frame_idx, strength, resample = image_entry
        else:
            image_path, frame_idx, strength = image_entry
            resample = None
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            resample=resample,
        )
        encoded_image = vae_encode_video(image, video_encoder, tiling_config)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=frame_idx, strength=strength)
        )
    return conditionings


def video_conditionings_by_keyframe(
    video_conditioning: list[tuple],
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    conditionings = []
    for entry in video_conditioning:
        if len(entry) == 2:
            video_path, strength = entry
            frame_idx = 0
        elif len(entry) == 3:
            video_path, frame_idx, strength = entry
        else:
            raise ValueError("Video conditioning entries must be (video, strength) or (video, frame_idx, strength).")
        video = load_video_conditioning(
            video_path=video_path,
            height=height,
            width=width,
            frame_cap=num_frames,
            dtype=dtype,
            device=device,
        )
        # remove_prepend = False
        # if frame_idx < 0:
        #     remove_prepend = True
        #     frame_idx = -frame_idx
        # if frame_idx < 0:
        #     encoded_video = vae_encode_video(video, video_encoder, tiling_config)
        #     encoded_video = encoded_video[:, :, 1:]
        #     frame_idx = -frame_idx + 1
        # else:
        #     encoded_video = vae_encode_video(video, video_encoder, tiling_config)

        encoded_video = vae_encode_video(video, video_encoder, tiling_config)
        cond =VideoConditionByKeyframeIndex(keyframes=encoded_video, frame_idx=frame_idx, strength=strength)
        conditionings.append(cond)

    return conditionings


def latent_conditionings_by_latent_sequence(
    latents: torch.Tensor,
    strength: float = 1.0,
    start_index: int = 0,
) -> list[ConditioningItem]:
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    if latents.dim() != 5:
        raise ValueError(f"Expected latent tensor with 5 dimensions; got {latents.shape}.")
    if latents.shape[2] == 0:
        return []
    conditionings = []
    for latent_idx in range(latents.shape[2]):
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=latents[:, :, latent_idx : latent_idx + 1],
                strength=strength,
                latent_idx=start_index + latent_idx,
            )
        )
    return conditionings


@dataclass(frozen=True)
class MaskInjection:
    mask_tokens: torch.Tensor
    source_tokens: torch.Tensor
    noise_tokens: torch.Tensor
    token_slice: slice
    masked_steps: int


def _pixel_to_latent_index(frame_idx: int, stride: int) -> int:
    if frame_idx <= 0:
        return 0
    return (frame_idx - 1) // stride + 1


def _coerce_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 5:
        if mask.shape[1] in (1, 3, 4):
            return mask[:, :1]
        if mask.shape[-1] in (1, 3, 4):
            return mask.permute(0, 4, 1, 2, 3)[:, :1]
    elif mask.ndim == 4:
        if mask.shape[0] in (1, 3, 4):
            return mask.unsqueeze(0)[:, :1]
        if mask.shape[-1] in (1, 3, 4):
            return mask.permute(3, 0, 1, 2).unsqueeze(0)[:, :1]
        return mask.unsqueeze(1)
    elif mask.ndim == 3:
        if mask.shape[-1] in (1, 3, 4):
            return mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)[:, :1]
        if mask.shape[0] in (1, 3, 4):
            return mask.unsqueeze(0).unsqueeze(2)[:, :1]
        return mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported mask tensor shape: {tuple(mask.shape)}")


def _normalize_mask_values(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if mask.min() < 0.0:
        mask = (mask + 1.0) * 0.5
    elif mask.max() > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def _resize_mask_spatial(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[3] == height and mask.shape[4] == width:
        return mask
    return F.interpolate(mask, size=(mask.shape[2], height, width), mode="nearest")


def _mask_to_latents(mask: torch.Tensor, target_frames: int, target_h: int, target_w: int) -> torch.Tensor:
    if target_frames <= 0 or mask.shape[2] == 0:
        raise ValueError("Mask has no frames to map into latent space.")
    if mask.shape[2] == 1:
        mask = F.interpolate(mask, size=(1, target_h, target_w), mode="nearest")
        if target_frames > 1:
            mask = mask.expand(-1, -1, target_frames, -1, -1)
        return mask
    if target_frames == 1:
        return F.interpolate(mask[:, :, :1], size=(1, target_h, target_w), mode="nearest")
    first = F.interpolate(mask[:, :, :1], size=(1, target_h, target_w), mode="nearest")
    rest = mask[:, :, 1:]
    if rest.shape[2] == 0:
        rest = torch.ones(
            (mask.shape[0], 1, target_frames - 1, target_h, target_w),
            device=mask.device,
            dtype=mask.dtype,
        )
    else:
        rest = F.interpolate(rest, size=(target_frames - 1, target_h, target_w), mode="nearest")
    return torch.cat([first, rest], dim=2)


def prepare_mask_injection(  # noqa: PLR0913
    masking_source: dict | None,
    masking_strength: float | None,
    output_shape: VideoPixelShape,
    video_encoder: VideoEncoder,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None,
    generator: torch.Generator,
    num_steps: int,
) -> MaskInjection | None:
    if masking_source is None:
        return None
    try:
        strength = float(masking_strength or 0.0)
    except (TypeError, ValueError):
        return None
    strength = max(0.0, min(1.0, strength))
    if strength <= 0.0 or num_steps <= 0:
        return None
    masked_steps = min(num_steps, int(math.ceil(num_steps * strength)))
    if masked_steps <= 0:
        return None

    video = masking_source.get("video")
    mask = masking_source.get("mask")
    if video is None or mask is None:
        return None
    start_frame = int(masking_source.get("start_frame") or 0)

    video_tensor = load_video_conditioning(
        video_path=video,
        height=output_shape.height,
        width=output_shape.width,
        frame_cap=None,
        dtype=dtype,
        device=device,
    )

    mask_tensor = _coerce_mask_tensor(mask).to(device=device)
    mask_tensor = _normalize_mask_values(mask_tensor)
    if mask_tensor.shape[0] != video_tensor.shape[0]:
        if mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.expand(video_tensor.shape[0], -1, -1, -1, -1)
        else:
            return None
    mask_tensor = _resize_mask_spatial(mask_tensor, output_shape.height, output_shape.width)
    if mask_tensor.shape[2] < video_tensor.shape[2]:
        pad_frames = video_tensor.shape[2] - mask_tensor.shape[2]
        pad = torch.ones(
            (mask_tensor.shape[0], 1, pad_frames, mask_tensor.shape[3], mask_tensor.shape[4]),
            device=mask_tensor.device,
            dtype=mask_tensor.dtype,
        )
        mask_tensor = torch.cat([mask_tensor, pad], dim=2)
    elif mask_tensor.shape[2] > video_tensor.shape[2]:
        mask_tensor = mask_tensor[:, :, : video_tensor.shape[2]]
    if video_tensor.shape[2] == 0 or mask_tensor.shape[2] == 0:
        return None

    source_latents = vae_encode_video(video_tensor, video_encoder, tiling_config).to(device=device, dtype=dtype)
    try:
        mask_latents = _mask_to_latents(
            mask_tensor, source_latents.shape[2], source_latents.shape[3], source_latents.shape[4]
        )
    except ValueError:
        return None
    mask_latents = (mask_latents >= 0.5).to(dtype)

    output_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    start_latent = _pixel_to_latent_index(start_frame, components.video_scale_factors.time)
    if start_latent >= output_latent_shape.frames:
        return None
    available_frames = output_latent_shape.frames - start_latent
    control_frames = min(source_latents.shape[2], available_frames)
    if control_frames <= 0:
        return None
    source_latents = source_latents[:, :, :control_frames]
    mask_latents = mask_latents[:, :, :control_frames]

    source_tokens = components.video_patchifier.patchify(source_latents)
    mask_tokens = components.video_patchifier.patchify(mask_latents).to(dtype=source_tokens.dtype)
    noise_tokens = torch.randn(
        source_tokens.shape,
        device=source_tokens.device,
        dtype=source_tokens.dtype,
        generator=generator,
    )

    patch_t, patch_h, patch_w = components.video_patchifier.patch_size
    if patch_t != 1:
        raise ValueError("Mask injection expects temporal patch size of 1.")
    tokens_per_frame = (output_latent_shape.height // patch_h) * (output_latent_shape.width // patch_w)
    token_offset = start_latent * tokens_per_frame
    token_count = control_frames * tokens_per_frame
    token_slice = slice(token_offset, token_offset + token_count)

    return MaskInjection(
        mask_tokens=mask_tokens,
        source_tokens=source_tokens,
        noise_tokens=noise_tokens,
        token_slice=token_slice,
        masked_steps=masked_steps,
    )


def _apply_mask_injection(
    video_state: LatentState,
    sigmas: torch.Tensor,
    step_idx: int,
    mask_context: MaskInjection,
) -> None:
    if step_idx >= mask_context.masked_steps:
        return
    sigma_next = sigmas[step_idx + 1].to(mask_context.source_tokens.dtype)
    token_slice = mask_context.token_slice
    current = video_state.latent[:, token_slice]
    noisy_source = mask_context.noise_tokens * sigma_next + (1 - sigma_next) * mask_context.source_tokens
    video_state.latent[:, token_slice] = noisy_source * (1 - mask_context.mask_tokens) + mask_context.mask_tokens * current


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    *,
    mask_context: MaskInjection | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    callback: Callable[..., None] | None = None,
    preview_tools: VideoLatentTools | None = None,
    pass_no: int = 0,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Perform the joint audio-video denoising loop over a diffusion schedule.
    This function iterates over all but the final value in ``sigmas`` and, at
    each diffusion step, calls ``denoise_fn`` to obtain denoised video and
    audio latents. The denoised latents are post-processed with their
    respective denoise masks and clean latents, then passed to ``stepper`` to
    advance the noisy latents one step along the diffusion schedule.
    ### Parameters
    sigmas:
        A 1D tensor of noise levels (diffusion sigmas) defining the sampling
        schedule. All steps except the last element are iterated over.
    video_state:
        The current video :class:`LatentState`, containing the noisy latent,
        its clean reference latent, and the denoising mask.
    audio_state:
        The current audio :class:`LatentState`, analogous to ``video_state``
        but for the audio modality.
    stepper:
        An implementation of :class:`DiffusionStepProtocol` that updates a
        latent given the current latent, its denoised estimate, the full
        ``sigmas`` schedule, and the current step index.
    denoise_fn:
        A callable implementing :class:`DenoisingFunc`. It is invoked as
        ``denoise_fn(video_state, audio_state, sigmas, step_index)`` and must
        return a tuple ``(denoised_video, denoised_audio)``, where each element
        is a tensor with the same shape as the corresponding latent.
    ### Returns
    tuple[LatentState, LatentState]
        A pair ``(video_state, audio_state)`` containing the final video and
        audio latent states after completing the denoising loop.
    """
    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        if interrupt_check is not None and interrupt_check():
            return None, None
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
        if denoised_video is None and denoised_audio is None:
            return None, None

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))
        if mask_context is not None:
            _apply_mask_injection(video_state, sigmas, step_idx, mask_context)
        _invoke_callback(callback, step_idx, pass_no, video_state, preview_tools)

    return (video_state, audio_state)


def gradient_estimating_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    ge_gamma: float = 2.0,
    *,
    mask_context: MaskInjection | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    callback: Callable[..., None] | None = None,
    preview_tools: VideoLatentTools | None = None,
    pass_no: int = 0,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Perform the joint audio-video denoising loop using gradient-estimation sampling.
    This function is similar to :func:`euler_denoising_loop`, but applies
    gradient estimation to improve the denoised estimates by tracking velocity
    changes across steps. See the referenced function for detailed parameter
    documentation.
    ### Parameters
    ge_gamma:
        Gradient estimation coefficient controlling the velocity correction term.
        Default is 2.0. Paper: https://openreview.net/pdf?id=o2ND9v0CeK
    sigmas, video_state, audio_state, stepper, denoise_fn:
        See :func:`euler_denoising_loop` for parameter descriptions.
    ### Returns
    tuple[LatentState, LatentState]
        See :func:`euler_denoising_loop` for return value description.
    """

    previous_audio_velocity = None
    previous_video_velocity = None

    def update_velocity_and_sample(
        noisy_sample: torch.Tensor, denoised_sample: torch.Tensor, sigma: float, previous_velocity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        if interrupt_check is not None and interrupt_check():
            return None, None
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
        if denoised_video is None and denoised_audio is None:
            return None, None

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        if sigmas[step_idx + 1] == 0:
            _invoke_callback(
                callback,
                step_idx,
                pass_no,
                replace(video_state, latent=denoised_video),
                preview_tools,
            )
            return replace(video_state, latent=denoised_video), replace(audio_state, latent=denoised_audio)

        previous_video_velocity, denoised_video = update_velocity_and_sample(
            video_state.latent, denoised_video, sigmas[step_idx], previous_video_velocity
        )
        previous_audio_velocity, denoised_audio = update_velocity_and_sample(
            audio_state.latent, denoised_audio, sigmas[step_idx], previous_audio_velocity
        )

        video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))
        if mask_context is not None:
            _apply_mask_injection(video_state, sigmas, step_idx, mask_context)
        _invoke_callback(callback, step_idx, pass_no, video_state, preview_tools)

    return (video_state, audio_state)


def noise_video_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, VideoLatentTools]:
    """Initialize and noise a video latent state for the diffusion pipeline.
    Creates a video latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    video latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, output_shape.fps)
    video_state = create_noised_state(
        tools=video_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return video_state, video_tools


def bind_interrupt_check(transformer: object, interrupt_check: Callable[[], bool] | None) -> None:
    if interrupt_check is None or transformer is None:
        return
    target = getattr(transformer, "velocity_model", transformer)
    if hasattr(target, "interrupt_check"):
        target.interrupt_check = interrupt_check


def noise_audio_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, AudioLatentTools]:
    """Initialize and noise an audio latent state for the diffusion pipeline.
    Creates an audio latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    audio latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
    audio_state = create_noised_state(
        tools=audio_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return audio_state, audio_tools


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState, context: torch.Tensor, sigma: float | torch.Tensor, enabled: bool = True
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    """
    timesteps, frame_indices = timesteps_from_mask(state.denoise_mask, sigma, positions=state.positions)
    return Modality(
        enabled=enabled,
        latent=state.latent,
        timesteps=timesteps,
        positions=state.positions,
        context=context,
        context_mask=None,
        frame_indices=frame_indices,
    )


def timesteps_from_mask(
    denoise_mask: torch.Tensor, sigma: float | torch.Tensor, positions: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute timesteps from a denoise mask and sigma value.
    Multiplies the denoise mask by sigma to produce timesteps for each position
    in the latent state. Areas where the mask is 0 will have zero timesteps.
    """
    if positions is None or positions.ndim < 4 or positions.shape[1] != 3:
        return denoise_mask * sigma, None

    token_mask = denoise_mask
    if token_mask.ndim > 2:
        token_mask = token_mask.mean(dim=-1)

    batch_size = token_mask.shape[0]
    frame_times = positions[:, 0, :, 0]

    frame_indices_list = []
    frame_masks = []
    for b in range(batch_size):
        unique_times, inverse = torch.unique(frame_times[b], sorted=True, return_inverse=True)
        frame_indices_list.append(inverse)

        frame_count = unique_times.numel()
        frame_min = torch.full(
            (frame_count,),
            torch.finfo(token_mask.dtype).max,
            device=token_mask.device,
            dtype=token_mask.dtype,
        )
        frame_max = torch.full(
            (frame_count,),
            torch.finfo(token_mask.dtype).min,
            device=token_mask.device,
            dtype=token_mask.dtype,
        )
        frame_min.scatter_reduce_(0, inverse, token_mask[b], reduce="amin", include_self=True)
        frame_max.scatter_reduce_(0, inverse, token_mask[b], reduce="amax", include_self=True)

        if not torch.allclose(frame_min, frame_max, atol=1e-6):
            return denoise_mask * sigma, None
        frame_masks.append(frame_min)

    frame_timesteps = torch.stack(frame_masks, dim=0) * sigma
    frame_indices = torch.stack(frame_indices_list, dim=0)
    return frame_timesteps, frame_indices


def simple_denoising_func(
    video_context: torch.Tensor, audio_context: torch.Tensor, transformer: X0Model
) -> DenoisingFunc:
    def simple_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, video_context, sigma)
        pos_audio = modality_from_latent_state(audio_state, audio_context, sigma)

        if transformer is not None:
            offload.set_step_no_for_lora(transformer, step_index)
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        if denoised_video is None and denoised_audio is None:
            return None, None
        return denoised_video, denoised_audio

    return simple_denoising_step


def guider_denoising_func(
    guider: GuiderProtocol,
    v_context_p: torch.Tensor,
    v_context_n: torch.Tensor,
    a_context_p: torch.Tensor,
    a_context_n: torch.Tensor,
    transformer: X0Model,
) -> DenoisingFunc:
    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
        pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

        if transformer is not None:
            offload.set_step_no_for_lora(transformer, step_index)
        if guider.enabled():
            neg_video = modality_from_latent_state(video_state, v_context_n, sigma)
            neg_audio = modality_from_latent_state(audio_state, a_context_n, sigma)
            denoised_video_list, denoised_audio_list = transformer(
                video=[pos_video, neg_video],
                audio=[pos_audio, neg_audio],
                perturbations=None,
            )
            if denoised_video_list is None and denoised_audio_list is None:
                return None, None
            denoised_video, neg_denoised_video = denoised_video_list
            denoised_audio, neg_denoised_audio = denoised_audio_list
            if denoised_video is None and denoised_audio is None:
                return None, None

            denoised_video = denoised_video + guider.delta(denoised_video, neg_denoised_video)
            denoised_audio = denoised_audio + guider.delta(denoised_audio, neg_denoised_audio)
            neg_video = neg_audio = neg_denoised_video = neg_denoised_audio = None
        else:
            denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
            if denoised_video is None and denoised_audio is None:
                return None, None

        pos_video = pos_audio = None
        return denoised_video, denoised_audio

    return guider_denoising_step


def denoise_audio_video(  # noqa: PLR0913
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: torch.Tensor,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    audio_conditionings: list[ConditioningItem] | None = None,
    noise_scale: float = 1.0,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
    mask_context: MaskInjection | None = None,
) -> tuple[LatentState | None, LatentState | None]:
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )
    audio_state, audio_tools = noise_audio_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=audio_conditionings or [],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_audio_latent,
    )

    loop_kwargs = {}
    if "preview_tools" in inspect.signature(denoising_loop_fn).parameters:
        loop_kwargs["preview_tools"] = video_tools
    if "mask_context" in inspect.signature(denoising_loop_fn).parameters:
        loop_kwargs["mask_context"] = mask_context
    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
        **loop_kwargs,
    )

    if video_state is None or audio_state is None:
        return None, None

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


def _invoke_callback(
    callback: Callable[..., None] | None,
    step_idx: int,
    pass_no: int,
    video_state: LatentState | None,
    preview_tools: VideoLatentTools | None,
) -> None:
    if callback is None or video_state is None:
        return
    preview_latents = None
    if preview_tools is not None:
        preview_state = preview_tools.clear_conditioning(video_state)
        preview_state = preview_tools.unpatchify(preview_state)
        preview_latents = preview_state.latent[0].detach()
    callback(step_idx, preview_latents, False, pass_no=pass_no)


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoderModelBase,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    image = None
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info(f"Enhanced prompt: {prompt}")
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    """Assert that the resolution is divisible by the required divisor.
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    """
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )
