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
    VideoConditionByReferenceLatent,
)
from ...ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ...ltx_core.model.transformer import Modality, X0Model
from ...ltx_core.model.video_vae import VideoEncoder, TilingConfig, encode_video as vae_encode_video
from ...ltx_core.text_encoders.gemma import GemmaTextEncoderModelBase
from ...ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ...ltx_core.types import AudioLatentShape, LatentState, TimestepCompressionPlan, VideoLatentShape, VideoPixelShape
from ...ltx_core.utils import to_denoised, to_velocity
from .media_io import decode_image, load_image_conditioning, load_video_conditioning, resize_aspect_ratio_preserving
from .types import (
    DenoisingFunc,
    DenoisingLoopFunc,
    PipelineComponents,
)
from shared.utils.self_refiner import run_refinement_loop_multi


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


def video_conditionings_by_reference_latent(
    video_conditioning: list[tuple],
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    downscale_factor: int = 1,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    scale = max(1, int(downscale_factor))
    if scale > 1 and (height % scale != 0 or width % scale != 0):
        raise ValueError(f"Output dimensions ({height}x{width}) must be divisible by reference downscale factor {scale}.")

    ref_height = height // scale
    ref_width = width // scale
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
            height=ref_height,
            width=ref_width,
            frame_cap=num_frames,
            dtype=dtype,
            device=device,
        )
        encoded_video = vae_encode_video(video, video_encoder, tiling_config)
        conditionings.append(
            VideoConditionByReferenceLatent(
                latent=encoded_video,
                frame_idx=frame_idx,
                strength=strength,
                downscale_factor=scale,
            )
        )
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
    transformer=None,
    self_refiner_handler=None,
    self_refiner_handler_audio=None,
    self_refiner_generator: torch.Generator | None = None,
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
    prewarm = getattr(denoise_fn, "_prewarm", None)
    cleanup = getattr(denoise_fn, "_cleanup", None)
    if callable(prewarm):
        prewarm(video_state, audio_state, sigmas)

    try:
        for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
            if interrupt_check is not None and interrupt_check():
                return None, None

            offload.set_step_no_for_lora(transformer, step_idx)
            denoised_video = denoised_audio = None
            denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
            if denoised_video is None or denoised_audio is None:
                return None, None

            denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
            denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

            refiner_steps = 0
            if self_refiner_handler is not None:
                self_refiner_handler.reset_buffer()
                refiner_steps = self_refiner_handler.get_anneal_steps(step_idx)
            if self_refiner_handler_audio is not None:
                self_refiner_handler_audio.reset_buffer()
                if refiner_steps == 0:
                    refiner_steps = self_refiner_handler_audio.get_anneal_steps(step_idx)

            use_audio_refiner = (
                self_refiner_handler is not None
                and self_refiner_handler_audio is not None
                and denoised_audio is not None
            )

            if use_audio_refiner and refiner_steps > 1:
                current_sigma = float(sigmas[step_idx].item()) if torch.is_tensor(sigmas[step_idx]) else float(sigmas[step_idx])
                next_sigma = float(sigmas[step_idx + 1].item()) if step_idx + 1 < len(sigmas) else 0.0
                refine_failed = False

                def denoise_multi(latents_list):
                    nonlocal refine_failed
                    temp_video_state = replace(video_state, latent=latents_list[0])
                    temp_audio_state = replace(audio_state, latent=latents_list[1])
                    denoised_video_loop, denoised_audio_loop = denoise_fn(temp_video_state, temp_audio_state, sigmas, step_idx)
                    if denoised_video_loop is None and denoised_audio_loop is None:
                        refine_failed = True
                        return [denoised_video, denoised_audio]
                    if denoised_video_loop is None or denoised_audio_loop is None:
                        refine_failed = True
                        return [denoised_video, denoised_audio]
                    denoised_video_loop = post_process_latent(
                        denoised_video_loop, temp_video_state.denoise_mask, temp_video_state.clean_latent
                    )
                    denoised_audio_loop = post_process_latent(
                        denoised_audio_loop, temp_audio_state.denoise_mask, temp_audio_state.clean_latent
                    )
                    return [denoised_video_loop, denoised_audio_loop]

                def step_func(noise_preds, latents_list):
                    latents_next_video = stepper.step(latents_list[0], noise_preds[0], sigmas, step_idx)
                    latents_next_audio = stepper.step(latents_list[1], noise_preds[1], sigmas, step_idx)
                    return [latents_next_video, latents_next_audio], noise_preds

                refined_latents = run_refinement_loop_multi(
                    handlers=[self_refiner_handler, self_refiner_handler_audio],
                    latents_list=[video_state.latent, audio_state.latent],
                    noise_pred_list=[denoised_video, denoised_audio],
                    current_sigma=current_sigma,
                    next_sigma=next_sigma,
                    m_steps=refiner_steps,
                    denoise_func=denoise_multi,
                    step_func=step_func,
                    generators=[self_refiner_generator, self_refiner_generator],
                    devices=[video_state.latent.device, audio_state.latent.device],
                    noise_masks=[video_state.denoise_mask, audio_state.denoise_mask],
                )
                if refine_failed or refined_latents is None or any(latent is None for latent in refined_latents):
                    return None, None
                video_state = replace(video_state, latent=refined_latents[0])
                audio_state = replace(audio_state, latent=refined_latents[1])
            elif self_refiner_handler is not None and refiner_steps > 1:
                current_sigma = float(sigmas[step_idx].item()) if torch.is_tensor(sigmas[step_idx]) else float(sigmas[step_idx])
                next_sigma = float(sigmas[step_idx + 1].item()) if step_idx + 1 < len(sigmas) else 0.0
                denoised_audio_final = denoised_audio
                refine_failed = False

                def denoise_video(latents_in):
                    nonlocal denoised_audio_final, refine_failed
                    temp_video_state = replace(video_state, latent=latents_in)
                    denoised_video_loop, denoised_audio_loop = denoise_fn(temp_video_state, audio_state, sigmas, step_idx)
                    if denoised_video_loop is None and denoised_audio_loop is None:
                        refine_failed = True
                        return denoised_video
                    if denoised_video_loop is None:
                        refine_failed = True
                        return denoised_video
                    denoised_video_loop = post_process_latent(
                        denoised_video_loop, temp_video_state.denoise_mask, temp_video_state.clean_latent
                    )
                    if denoised_audio_loop is not None:
                        denoised_audio_loop = post_process_latent(
                            denoised_audio_loop, audio_state.denoise_mask, audio_state.clean_latent
                        )
                        denoised_audio_final = denoised_audio_loop
                    return denoised_video_loop

                def step_func(n_pred_in, latents_in):
                    latents_next = stepper.step(latents_in, n_pred_in, sigmas, step_idx)
                    return latents_next, n_pred_in

                latents_refined = self_refiner_handler.run_refinement_loop(
                    latents=video_state.latent,
                    noise_pred=denoised_video,
                    current_sigma=current_sigma,
                    next_sigma=next_sigma,
                    m_steps=refiner_steps,
                    denoise_func=denoise_video,
                    step_func=step_func,
                    generator=self_refiner_generator,
                    device=video_state.latent.device,
                    noise_mask=video_state.denoise_mask,
                )
                if refine_failed or latents_refined is None:
                    return None, None
                video_state = replace(video_state, latent=latents_refined)
                audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio_final, sigmas, step_idx))
            else:
                video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
                audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))

            if mask_context is not None:
                _apply_mask_injection(video_state, sigmas, step_idx, mask_context)
            _invoke_callback(callback, step_idx, pass_no, video_state, preview_tools)

        return video_state, audio_state
    finally:
        if callable(cleanup):
            cleanup()


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
    cleanup = getattr(denoise_fn, "_cleanup", None)

    def update_velocity_and_sample(
        noisy_sample: torch.Tensor, denoised_sample: torch.Tensor, sigma: float, previous_velocity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    try:
        for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
            if interrupt_check is not None and interrupt_check():
                return None, None
            denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
            if denoised_video is None or denoised_audio is None:
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

        return video_state, audio_state
    finally:
        if callable(cleanup):
            cleanup()


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
    state: LatentState,
    context: torch.Tensor,
    sigma: float | torch.Tensor,
    enabled: bool = True,
    step_index: int | None = None,
    sigma_schedule: torch.Tensor | None = None,
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    """
    runtime_cache = state.runtime_cache
    if runtime_cache.timestep_plan is None:
        runtime_cache.timestep_plan = build_timestep_compression_plan(state.denoise_mask, state.positions)
    timesteps, frame_indices = timesteps_from_mask(state.denoise_mask, sigma, plan=runtime_cache.timestep_plan)
    sigma_tensor = sigma if torch.is_tensor(sigma) else torch.tensor(sigma, device=state.latent.device)
    sigma_tensor = sigma_tensor.to(device=state.latent.device, dtype=state.latent.dtype)
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.expand(state.latent.shape[0])
    elif sigma_tensor.ndim == 1 and sigma_tensor.shape[0] == 1:
        sigma_tensor = sigma_tensor.expand(state.latent.shape[0])
    elif sigma_tensor.ndim > 1:
        sigma_tensor = sigma_tensor.reshape(state.latent.shape[0], -1)[:, 0]
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma_tensor,
        timesteps=timesteps,
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=None,
        frame_indices=frame_indices,
        runtime_cache=runtime_cache,
        step_index=step_index,
        sigma_schedule=sigma_schedule,
    )


def _get_batch_size(video_state: LatentState | None, audio_state: LatentState | None) -> int:
    if video_state is not None:
        return int(video_state.latent.shape[0])
    if audio_state is not None:
        return int(audio_state.latent.shape[0])
    return 1


def _cross_attn_perturbations(batch_size: int) -> BatchedPerturbationConfig:
    perts = [
        PerturbationConfig(
            [
                Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, None),
                Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, None),
            ]
        )
        for _ in range(batch_size)
    ]
    return BatchedPerturbationConfig(perts)


PERTURBATION_perturbation = 1
PERTURBATION_SKIP_SELF_ATTENTION = 2


def _normalize_perturbation_layers(perturbation_layers) -> list[int] | None:
    if perturbation_layers is None:
        return None
    if isinstance(perturbation_layers, (list, tuple)):
        return [int(layer) for layer in perturbation_layers if str(layer).strip() != ""]
    if isinstance(perturbation_layers, (int, float)):
        return [int(perturbation_layers)]
    if isinstance(perturbation_layers, str):
        return [int(layer.strip()) for layer in perturbation_layers.split(",") if layer.strip()]
    return None


def _legacy_perturbation_layer_configs(
    batch_size: int, perturbation_layers: list[int] | None
) -> BatchedPerturbationConfig:
    perts = [
        PerturbationConfig(
            [
                Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_AUDIO_SELF_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, perturbation_layers),
            ]
        )
        for _ in range(batch_size)
    ]
    return BatchedPerturbationConfig(perts)


def _self_attn_perturbation_configs(
    batch_size: int, perturbation_layers: list[int] | None
) -> BatchedPerturbationConfig:
    perts = [
        PerturbationConfig(
            [
                Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_AUDIO_SELF_ATTN, perturbation_layers),
            ]
        )
        for _ in range(batch_size)
    ]
    return BatchedPerturbationConfig(perts)


def _perturbation_active(
    step_index: int,
    sigmas: torch.Tensor,
    perturbation_start: float,
    perturbation_end: float,
) -> bool:
    total_steps = max(len(sigmas) - 1, 1)
    start_step = int(perturbation_start * total_steps)
    end_step = int(perturbation_end * total_steps)
    return start_step <= step_index < end_step


def _rescale_prediction(cond: torch.Tensor | None, pred: torch.Tensor | None, rescale_scale: float) -> torch.Tensor | None:
    if cond is None or pred is None or math.isclose(rescale_scale, 0.0):
        return pred
    pred_std = pred.std().clamp_min(1e-6)
    factor = cond.std() / pred_std
    factor = rescale_scale * factor + (1.0 - rescale_scale)
    return pred * factor


def build_timestep_compression_plan(
    denoise_mask: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> TimestepCompressionPlan:
    if positions is None or positions.ndim < 4 or positions.shape[1] != 3:
        return TimestepCompressionPlan()

    token_mask = denoise_mask
    if token_mask.ndim > 2:
        token_mask = token_mask.mean(dim=-1)

    batch_size = token_mask.shape[0]
    frame_times = positions[:, 0, :, 0]
    run_lengths = None
    quantum = 0
    frame_masks = []
    frame_indices = []
    token_count = frame_times.shape[1]
    for b in range(batch_size):
        changes = torch.nonzero(frame_times[b, 1:] != frame_times[b, :-1]).flatten() + 1
        bounds = torch.cat([changes.new_zeros(1), changes, changes.new_full((1,), token_count)])
        lengths = (bounds[1:] - bounds[:-1]).tolist()
        if run_lengths is None:
            run_lengths = lengths
            quantum = run_lengths[0]
            for length in run_lengths[1:]:
                quantum = math.gcd(quantum, length)
        elif lengths != run_lengths:
            return TimestepCompressionPlan()

        group_values = []
        group_indices = []
        group_idx = 0
        for start, end, length in zip(bounds[:-1].tolist(), bounds[1:].tolist(), run_lengths):
            run_mask = token_mask[b, start:end]
            if not torch.allclose(run_mask, run_mask[:1], atol=1e-6):
                return TimestepCompressionPlan()
            repeat = length // quantum
            group_values.append(run_mask[:1].expand(repeat))
            group_indices.append(
                torch.arange(group_idx, group_idx + repeat, device=token_mask.device).repeat_interleave(quantum)
            )
            group_idx += repeat
        frame_masks.append(torch.cat(group_values))
        frame_indices.append(torch.cat(group_indices))

    return TimestepCompressionPlan(frame_mask=torch.stack(frame_masks, dim=0), frame_indices=torch.stack(frame_indices, dim=0))


def timesteps_from_mask(
    denoise_mask: torch.Tensor,
    sigma: float | torch.Tensor,
    positions: torch.Tensor | None = None,
    plan: TimestepCompressionPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute timesteps from a denoise mask and sigma value."""
    if plan is None:
        plan = build_timestep_compression_plan(denoise_mask, positions)
    if plan.frame_mask is None or plan.frame_indices is None:
        return denoise_mask * sigma, None
    return plan.frame_mask * sigma, plan.frame_indices


def _prepare_conditioning_context(
    transformer: X0Model | None,
    state: LatentState,
    context: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    is_audio: bool,
):
    if transformer is None:
        return context
    velocity_model = getattr(transformer, "velocity_model", transformer)
    preprocessor_name = "audio_args_preprocessor" if is_audio else "video_args_preprocessor"
    preprocessor = getattr(velocity_model, preprocessor_name, None)
    if preprocessor is None or not hasattr(preprocessor, "build_prepared_conditioning"):
        return context
    seed_modality = modality_from_latent_state(state, context, sigmas[0], step_index=0, sigma_schedule=sigmas)
    return preprocessor.build_prepared_conditioning(seed_modality)


def _clear_phase_timestep_embedders(transformer: X0Model | None) -> None:
    if transformer is None:
        return
    velocity_model = getattr(transformer, "velocity_model", transformer)
    for preprocessor_name in ("video_args_preprocessor", "audio_args_preprocessor"):
        preprocessor = getattr(velocity_model, preprocessor_name, None)
        clear_fn = None if preprocessor is None else getattr(preprocessor, "clear_phase_timestep_embedders", None)
        if callable(clear_fn):
            clear_fn()


def _get_audio_reference_token_count(audio_state: LatentState) -> int:
    positions = audio_state.positions
    if positions is None or positions.ndim < 4 or positions.shape[1] < 1:
        return 0
    ref_mask = positions[:, 0, :, 1] < 0
    if ref_mask.ndim != 2 or not torch.any(ref_mask):
        return 0
    counts = ref_mask.sum(dim=1)
    if not torch.all(counts == counts[:1]):
        return 0
    ref_tokens = int(counts[0].item())
    total_tokens = int(ref_mask.shape[1])
    if ref_tokens <= 0 or ref_tokens >= total_tokens:
        return 0
    if not torch.all(ref_mask[:, :ref_tokens]) or torch.any(ref_mask[:, ref_tokens:]):
        return 0
    return ref_tokens


def _slice_audio_target_state(audio_state: LatentState, ref_audio_tokens: int) -> LatentState | None:
    ref_audio_tokens = int(ref_audio_tokens)
    total_tokens = int(audio_state.latent.shape[1])
    if ref_audio_tokens <= 0 or ref_audio_tokens >= total_tokens:
        return None
    return LatentState(
        latent=audio_state.latent[:, ref_audio_tokens:],
        denoise_mask=audio_state.denoise_mask[:, ref_audio_tokens:],
        positions=audio_state.positions[:, :, ref_audio_tokens:],
        clean_latent=audio_state.clean_latent[:, ref_audio_tokens:],
    )


def simple_denoising_func(
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    transformer: X0Model,
    alt_guidance_scale: float = 1.0,
    audio_context_n: torch.Tensor | None = None,
    audio_guidance_scale: float = 1.0,
    audio_identity_guidance_scale: float = 0.0,
) -> DenoisingFunc:
    prepared_video_context = prepared_audio_context = None
    prepared_audio_context_n = prepared_audio_context_id = None

    def _prewarm(video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor) -> None:
        nonlocal prepared_video_context, prepared_audio_context
        if prepared_video_context is None:
            prepared_video_context = _prepare_conditioning_context(
                transformer, video_state, video_context, sigmas, is_audio=False
            )
        if prepared_audio_context is None:
            prepared_audio_context = _prepare_conditioning_context(
                transformer, audio_state, audio_context, sigmas, is_audio=True
            )

    def _cleanup() -> None:
        nonlocal prepared_video_context, prepared_audio_context, prepared_audio_context_n, prepared_audio_context_id
        prepared_video_context = None
        prepared_audio_context = None
        prepared_audio_context_n = None
        prepared_audio_context_id = None
        _clear_phase_timestep_embedders(transformer)

    def simple_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal prepared_audio_context_n, prepared_audio_context_id
        _prewarm(video_state, audio_state, sigmas)
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(
            video_state, prepared_video_context, sigma, step_index=step_index, sigma_schedule=sigmas
        )
        pos_audio = modality_from_latent_state(
            audio_state, prepared_audio_context, sigma, step_index=step_index, sigma_schedule=sigmas
        )

        if transformer is not None:
            offload.set_step_no_for_lora(transformer, step_index)
        use_alt = not math.isclose(alt_guidance_scale, 1.0)
        use_audio_cfg = audio_context_n is not None and not math.isclose(audio_guidance_scale, 1.0)
        ref_audio_tokens = _get_audio_reference_token_count(audio_state) if audio_identity_guidance_scale > 0 else 0
        id_audio_state = _slice_audio_target_state(audio_state, ref_audio_tokens) if ref_audio_tokens > 0 else None
        use_id = id_audio_state is not None
        if use_audio_cfg and prepared_audio_context_n is None:
            prepared_audio_context_n = _prepare_conditioning_context(
                transformer, audio_state, audio_context_n, sigmas, is_audio=True
            )
        if use_id and prepared_audio_context_id is None:
            prepared_audio_context_id = _prepare_conditioning_context(
                transformer, id_audio_state, audio_context, sigmas, is_audio=True
            )
        if not use_alt and not use_audio_cfg and not use_id:
            denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
            if denoised_video is None and denoised_audio is None:
                return None, None
            return denoised_video, denoised_audio

        batch_size = _get_batch_size(video_state, audio_state)
        video_list = [pos_video]
        audio_list = [pos_audio]
        perturbations = [None]
        neg_index = None
        alt_index = None
        id_index = None
        if use_audio_cfg:
            neg_index = len(video_list)
            video_list.append(
                modality_from_latent_state(
                    video_state, prepared_video_context, sigma, step_index=step_index, sigma_schedule=sigmas
                )
            )
            audio_list.append(
                modality_from_latent_state(
                    audio_state, prepared_audio_context_n, sigma, step_index=step_index, sigma_schedule=sigmas
                )
            )
            perturbations.append(None)
        if use_alt:
            alt_index = len(video_list)
            video_list.append(
                modality_from_latent_state(
                    video_state, prepared_video_context, sigma, step_index=step_index, sigma_schedule=sigmas
                )
            )
            audio_list.append(
                modality_from_latent_state(
                    audio_state, prepared_audio_context, sigma, step_index=step_index, sigma_schedule=sigmas
                )
            )
            perturbations.append(_cross_attn_perturbations(batch_size))
        if use_id:
            id_index = len(video_list)
            video_list.append(
                modality_from_latent_state(
                    video_state, prepared_video_context, sigma, step_index=step_index, sigma_schedule=sigmas
                )
            )
            audio_list.append(
                modality_from_latent_state(
                    id_audio_state, prepared_audio_context_id, sigma, step_index=step_index, sigma_schedule=sigmas
                )
            )
            perturbations.append(None)

        denoised_video_list, denoised_audio_list = transformer(
            video=video_list,
            audio=audio_list,
            perturbations=perturbations,
        )
        if denoised_video_list is None and denoised_audio_list is None:
            return None, None
        pos_denoised_video = denoised_video_list[0]
        pos_denoised_audio = denoised_audio_list[0]
        if pos_denoised_video is None and pos_denoised_audio is None:
            return None, None
        denoised_video = pos_denoised_video
        denoised_audio = pos_denoised_audio
        if use_audio_cfg and neg_index is not None and denoised_audio is not None:
            neg_denoised_audio = denoised_audio_list[neg_index]
            if neg_denoised_audio is not None:
                denoised_audio = denoised_audio + (audio_guidance_scale - 1.0) * (
                    pos_denoised_audio - neg_denoised_audio
                )
        if use_alt and alt_index is not None:
            alt_denoised_video = denoised_video_list[alt_index]
            alt_denoised_audio = denoised_audio_list[alt_index]
            if denoised_video is not None and alt_denoised_video is not None:
                denoised_video = denoised_video + (alt_guidance_scale - 1.0) * (
                    pos_denoised_video - alt_denoised_video
                )
            if denoised_audio is not None and alt_denoised_audio is not None:
                denoised_audio = denoised_audio + (alt_guidance_scale - 1.0) * (
                    pos_denoised_audio - alt_denoised_audio
                )
        if use_id and id_index is not None and denoised_audio is not None:
            id_denoised_audio = denoised_audio_list[id_index]
            target_audio_tokens = int(pos_denoised_audio.shape[1]) - int(ref_audio_tokens)
            if (
                id_denoised_audio is not None
                and target_audio_tokens > 0
                and id_denoised_audio.shape[1] == target_audio_tokens
            ):
                denoised_audio = denoised_audio.clone()
                denoised_audio[:, ref_audio_tokens:] = denoised_audio[:, ref_audio_tokens:] + audio_identity_guidance_scale * (
                    pos_denoised_audio[:, ref_audio_tokens:] - id_denoised_audio
                )
        return denoised_video, denoised_audio

    simple_denoising_step._prewarm = _prewarm
    simple_denoising_step._cleanup = _cleanup
    return simple_denoising_step


def guider_denoising_func(
    video_guider: GuiderProtocol,
    audio_guider: GuiderProtocol,
    v_context_p: torch.Tensor,
    v_context_n: torch.Tensor,
    a_context_p: torch.Tensor,
    a_context_n: torch.Tensor,
    transformer: X0Model,
    alt_guidance_scale: float = 1.0,
    alt_scale: float = 0.0,
    perturbation_switch: int = 0,
    perturbation_layers: list[int] | None = None,
    perturbation_start: float = 0.0,
    perturbation_end: float = 1.0,
    audio_identity_guidance_scale: float = 0.0,
) -> DenoisingFunc:
    perturb_all_layers = perturbation_layers is None
    perturbation_layers_norm = _normalize_perturbation_layers(perturbation_layers)
    prepared_v_context_p = prepared_v_context_n = None
    prepared_a_context_p = prepared_a_context_n = prepared_a_context_id = None

    def _prewarm(video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor) -> None:
        nonlocal prepared_v_context_p, prepared_v_context_n, prepared_a_context_p, prepared_a_context_n
        if prepared_v_context_p is None:
            prepared_v_context_p = _prepare_conditioning_context(transformer, video_state, v_context_p, sigmas, is_audio=False)
        if prepared_a_context_p is None:
            prepared_a_context_p = _prepare_conditioning_context(transformer, audio_state, a_context_p, sigmas, is_audio=True)

    def _cleanup() -> None:
        nonlocal prepared_v_context_p, prepared_v_context_n, prepared_a_context_p, prepared_a_context_n, prepared_a_context_id
        prepared_v_context_p = prepared_v_context_n = None
        prepared_a_context_p = prepared_a_context_n = prepared_a_context_id = None
        _clear_phase_timestep_embedders(transformer)

    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal prepared_v_context_n, prepared_a_context_n, prepared_a_context_id
        _prewarm(video_state, audio_state, sigmas)
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(
            video_state, prepared_v_context_p, sigma, step_index=step_index, sigma_schedule=sigmas
        )
        pos_audio = modality_from_latent_state(
            audio_state, prepared_a_context_p, sigma, step_index=step_index, sigma_schedule=sigmas
        )

        if transformer is not None:
            offload.set_step_no_for_lora(transformer, step_index)
        use_video_cfg = video_guider.enabled()
        use_audio_cfg = audio_guider.enabled()
        use_cfg = use_video_cfg or use_audio_cfg
        use_alt = not math.isclose(alt_guidance_scale, 1.0)
        ref_audio_tokens = _get_audio_reference_token_count(audio_state) if audio_identity_guidance_scale > 0 else 0
        id_audio_state = _slice_audio_target_state(audio_state, ref_audio_tokens) if ref_audio_tokens > 0 else None
        use_id = id_audio_state is not None
        if use_id and prepared_a_context_id is None:
            prepared_a_context_id = _prepare_conditioning_context(
                transformer, id_audio_state, a_context_p, sigmas, is_audio=True
            )
        use_perturbation = _perturbation_active(step_index, sigmas, perturbation_start, perturbation_end)
        has_perturbation_layers = perturb_all_layers or bool(perturbation_layers_norm)
        use_legacy_perturbation = (
            perturbation_switch == PERTURBATION_perturbation and use_cfg and use_perturbation and has_perturbation_layers
        )
        use_stg = (
            perturbation_switch == PERTURBATION_SKIP_SELF_ATTENTION and use_perturbation and has_perturbation_layers
        )
        selected_layers = None if perturb_all_layers else perturbation_layers_norm
        if use_cfg or use_alt or use_stg or use_id:
            batch_size = _get_batch_size(video_state, audio_state)
            video_list = [pos_video]
            audio_list = [pos_audio]
            perturbations: list[BatchedPerturbationConfig | None] = [None]
            neg_index = None
            stg_index = None
            alt_index = None
            id_index = None

            if use_cfg:
                if use_video_cfg and prepared_v_context_n is None:
                    prepared_v_context_n = _prepare_conditioning_context(
                        transformer, video_state, v_context_n, sigmas, is_audio=False
                    )
                if use_audio_cfg and prepared_a_context_n is None:
                    prepared_a_context_n = _prepare_conditioning_context(
                        transformer, audio_state, a_context_n, sigmas, is_audio=True
                    )
                neg_video_context = prepared_v_context_n if use_video_cfg else prepared_v_context_p
                neg_audio_context = prepared_a_context_n if use_audio_cfg else prepared_a_context_p
                neg_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(video_state, neg_video_context, sigma, step_index=step_index, sigma_schedule=sigmas)
                )
                audio_list.append(
                    modality_from_latent_state(audio_state, neg_audio_context, sigma, step_index=step_index, sigma_schedule=sigmas)
                )
                perturbations.append(
                    _legacy_perturbation_layer_configs(batch_size, selected_layers) if use_legacy_perturbation else None
                )

            if use_stg:
                stg_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(video_state, prepared_v_context_p, sigma, step_index=step_index, sigma_schedule=sigmas)
                )
                audio_list.append(
                    modality_from_latent_state(audio_state, prepared_a_context_p, sigma, step_index=step_index, sigma_schedule=sigmas)
                )
                perturbations.append(_self_attn_perturbation_configs(batch_size, selected_layers))

            if use_alt:
                alt_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(video_state, prepared_v_context_p, sigma, step_index=step_index, sigma_schedule=sigmas)
                )
                audio_list.append(
                    modality_from_latent_state(audio_state, prepared_a_context_p, sigma, step_index=step_index, sigma_schedule=sigmas)
                )
                perturbations.append(_cross_attn_perturbations(batch_size))

            if use_id:
                id_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state, prepared_v_context_p, sigma, step_index=step_index, sigma_schedule=sigmas
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        id_audio_state, prepared_a_context_id, sigma, step_index=step_index, sigma_schedule=sigmas
                    )
                )
                perturbations.append(None)

            denoised_video_list, denoised_audio_list = transformer(
                video=video_list,
                audio=audio_list,
                perturbations=perturbations,
            )
            if denoised_video_list is None and denoised_audio_list is None:
                return None, None
            pos_denoised_video = denoised_video_list[0]
            pos_denoised_audio = denoised_audio_list[0]
            if pos_denoised_video is None and pos_denoised_audio is None:
                return None, None

            denoised_video = pos_denoised_video
            denoised_audio = pos_denoised_audio

            if use_cfg and neg_index is not None:
                neg_denoised_video = denoised_video_list[neg_index]
                neg_denoised_audio = denoised_audio_list[neg_index]
                if denoised_video is not None and neg_denoised_video is not None and use_video_cfg:
                    denoised_video = denoised_video + video_guider.delta(pos_denoised_video, neg_denoised_video)
                if denoised_audio is not None and neg_denoised_audio is not None and use_audio_cfg:
                    denoised_audio = denoised_audio + audio_guider.delta(pos_denoised_audio, neg_denoised_audio)

            if use_stg and stg_index is not None:
                stg_denoised_video = denoised_video_list[stg_index]
                stg_denoised_audio = denoised_audio_list[stg_index]
                if denoised_video is not None and stg_denoised_video is not None:
                    denoised_video = denoised_video + (pos_denoised_video - stg_denoised_video)
                if denoised_audio is not None and stg_denoised_audio is not None:
                    denoised_audio = denoised_audio + (pos_denoised_audio - stg_denoised_audio)

            if use_alt and alt_index is not None:
                alt_denoised_video = denoised_video_list[alt_index]
                alt_denoised_audio = denoised_audio_list[alt_index]
                if denoised_video is not None and alt_denoised_video is not None:
                    denoised_video = denoised_video + (alt_guidance_scale - 1.0) * (
                        pos_denoised_video - alt_denoised_video
                    )
                if denoised_audio is not None and alt_denoised_audio is not None:
                    denoised_audio = denoised_audio + (alt_guidance_scale - 1.0) * (
                        pos_denoised_audio - alt_denoised_audio
                    )
            if use_id and id_index is not None and denoised_audio is not None:
                id_denoised_audio = denoised_audio_list[id_index]
                target_audio_tokens = int(pos_denoised_audio.shape[1]) - int(ref_audio_tokens)
                if (
                    id_denoised_audio is not None
                    and target_audio_tokens > 0
                    and id_denoised_audio.shape[1] == target_audio_tokens
                ):
                    denoised_audio = denoised_audio.clone()
                    denoised_audio[:, ref_audio_tokens:] = denoised_audio[:, ref_audio_tokens:] + audio_identity_guidance_scale * (
                        pos_denoised_audio[:, ref_audio_tokens:] - id_denoised_audio
                    )
        else:
            denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
            if denoised_video is None and denoised_audio is None:
                return None, None
            pos_denoised_video = denoised_video
            pos_denoised_audio = denoised_audio

        denoised_video = _rescale_prediction(pos_denoised_video, denoised_video, alt_scale)
        denoised_audio = _rescale_prediction(pos_denoised_audio, denoised_audio, alt_scale)

        pos_video = pos_audio = None
        return denoised_video, denoised_audio

    guider_denoising_step._prewarm = _prewarm
    guider_denoising_step._cleanup = _cleanup
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
    audio_noise_scale: float | None = None,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
    mask_context: MaskInjection | None = None,
    freeze_audio: bool = False,
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
        noise_scale=noise_scale if audio_noise_scale is None else audio_noise_scale,
        initial_latent=initial_audio_latent,
    )
    if freeze_audio:
        audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

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


def assert_resolution(
    height: int,
    width: int,
    is_two_stage: bool | None = None,
    upscale_passes: int | None = None,
) -> None:
    """Assert that the resolution is divisible by the required divisor."""
    if upscale_passes is None:
        upscale_passes = 1 if is_two_stage else 0
    upscale_passes = max(0, int(upscale_passes))
    divisor = 32 * (2 ** upscale_passes)
    if upscale_passes == 0:
        pipeline_label = "one-stage"
    elif upscale_passes == 1:
        pipeline_label = "two-stage"
    else:
        pipeline_label = f"{upscale_passes + 1}-phase"
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {pipeline_label} pipelines, "
            f"height and width must be multiples of {divisor}."
        )
