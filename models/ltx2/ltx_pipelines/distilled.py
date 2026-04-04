import logging
import os
import time
from collections.abc import Callable, Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoPixelShape
from .utils import ModelLedger
from .utils.args import default_2_stage_distilled_arg_parser
from .utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from .utils.helpers import (
    assert_resolution,
    bind_interrupt_check,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    latent_conditionings_by_latent_sequence,
    prepare_mask_injection,
    simple_denoising_func,
    video_conditionings_by_keyframe,
    video_conditionings_by_reference_latent,
)
from .utils.media_io import encode_video
from .utils.types import PipelineComponents
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.self_refiner import create_self_refiner_handler, normalize_self_refiner_plan
from shared.utils.text_encoder_cache import TextEncoderCache

device = get_device()
_BENCH_TRANSFORMER_ENV = "WAN2GP_LTX2_BENCH_TRANSFORMER"


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class _TransformerBenchWrapper:
    def __init__(self, module, enabled: bool = False) -> None:
        self._module = module
        self._enabled = bool(enabled)
        self._cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_total_ms = 0.0
        self._cpu_calls = 0

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        if not self._enabled:
            return self._module(*args, **kwargs)
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self._module(*args, **kwargs)
            end.record()
            self._cuda_events.append((start, end))
            return out

        t0 = time.perf_counter()
        out = self._module(*args, **kwargs)
        self._cpu_total_ms += (time.perf_counter() - t0) * 1000.0
        self._cpu_calls += 1
        return out

    def consume(self) -> tuple[float, int]:
        if not self._enabled:
            return 0.0, 0
        if torch.cuda.is_available():
            if not self._cuda_events:
                return 0.0, 0
            torch.cuda.synchronize()
            total_ms = 0.0
            for start, end in self._cuda_events:
                total_ms += float(start.elapsed_time(end))
            calls = len(self._cuda_events)
            self._cuda_events.clear()
            return total_ms, calls

        total_ms = self._cpu_total_ms
        calls = self._cpu_calls
        self._cpu_total_ms = 0.0
        self._cpu_calls = 0
        return total_ms, calls


class DistilledPipeline:
    """
    Multi-stage distilled video generation pipeline.
    Stage 1 generates video at a reduced resolution, then one or more distilled
    x2 upscale/refine passes bring it to the requested output resolution.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        gemma_root: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device = device,
        fp8transformer: bool = False,
        model_device: torch.device | None = None,
        models: object | None = None,
        upscale_passes: int = 1,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.models = models
        self.upscale_passes = max(1, int(upscale_passes))

        if self.models is None:
            if checkpoint_path is None or gemma_root is None or spatial_upsampler_path is None:
                raise ValueError("checkpoint_path, gemma_root, and spatial_upsampler_path are required.")
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=model_device or device,
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root_path=gemma_root,
                loras=loras or [],
                fp8transformer=fp8transformer,
            )
        else:
            self.model_ledger = None

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.text_encoder_cache = TextEncoderCache()

    def _get_model(self, name: str):
        if self.models is not None:
            return getattr(self.models, name)
        if self.model_ledger is None:
            raise ValueError(f"Missing model source for '{name}'.")
        return getattr(self.model_ledger, name)()

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        guiding_images: list[tuple] | None = None,
        guiding_images_stage2: list[tuple] | None = None,
        alt_guidance_scale: float = 1.0,
        audio_cfg_guidance_scale: float = 1.0,
        video_conditioning: list[tuple[str, float]] | None = None,
        video_conditioning_downscale_factor: int = 1,
        latent_conditioning_stage2: torch.Tensor | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        audio_conditionings: list | None = None,
        audio_conditionings_stage2: list | None = None,
        audio_identity_guidance_scale: float = 0.0,
        callback: Callable[..., None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
        loras_slists: dict | None = None,
        text_connectors: dict | None = None,
        masking_source: dict | None = None,
        masking_strength: float | None = None,
        return_latent_slice: slice | None = None,
        self_refiner_setting: int = 0,
        self_refiner_plan: str = "",
        self_refiner_f_uncertainty: float = 0.1,
        self_refiner_certain_percentage: float = 0.999,
        self_refiner_max_plans: int = 1,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, upscale_passes=self.upscale_passes)
        alt_guidance_scale = 1.0
        negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        total_passes = self.upscale_passes + 1
        self_refiner_handlers = [None] * total_passes
        self_refiner_audio_handlers = [None] * total_passes
        if self_refiner_setting and self_refiner_setting > 0:
            plans, _ = normalize_self_refiner_plan(self_refiner_plan or "", max_plans=self_refiner_max_plans)
            self_refiner_handlers[0] = create_self_refiner_handler(
                plans[0] if plans else [],
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            self_refiner_audio_handlers[0] = create_self_refiner_handler(
                plans[0] if plans else [],
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            for pass_index in range(1, min(total_passes, len(plans))):
                if not plans[pass_index]:
                    continue
                self_refiner_handlers[pass_index] = create_self_refiner_handler(
                    plans[pass_index],
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
                self_refiner_audio_handlers[pass_index] = create_self_refiner_handler(
                    plans[pass_index],
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
        dtype = torch.bfloat16

        text_encoder = self._get_model("text_encoder")
        if enhance_prompt:
            prompt = generate_enhanced_prompt(text_encoder, prompt, images[0][0] if len(images) > 0 else None)
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(
            text_encoder, text_connectors
        )
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        contexts = self.text_encoder_cache.encode(encode_fn, [prompt, negative_prompt], device=self.device, parallel=True)

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        (video_context, audio_context), (_, audio_context_n) = contexts

        # Stage 1: Initial low resolution video generation.
        bench_transformer = _env_flag(_BENCH_TRANSFORMER_ENV, "0")
        video_encoder = self._get_model("video_encoder")
        transformer = _TransformerBenchWrapper(self._get_model("transformer"), enabled=bench_transformer)
        bind_interrupt_check(transformer, interrupt_check)
        # DISTILLED_SIGMA_VALUES = [0.421875, 0]
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        pass_no = 1
        if loras_slists is not None:
            stage_1_steps = len(stage_1_sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_1_steps,
                phase_switch_step=stage_1_steps,
                phase_switch_step2=stage_1_steps,
            )

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(stage_1_sigmas) - 1, pass_no=pass_no)

        def denoising_loop_stage1(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools: VideoLatentTools | None = None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                    alt_guidance_scale=alt_guidance_scale,
                    audio_context_n=audio_context_n,
                    audio_guidance_scale=audio_cfg_guidance_scale,
                    audio_identity_guidance_scale=audio_identity_guidance_scale,
                ),
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=1,
                transformer=transformer,
                self_refiner_handler=self_refiner_handlers[0],
                self_refiner_handler_audio=self_refiner_audio_handlers[0],
                self_refiner_generator=generator,
            )

        stage_1_scale_divisor = 2 ** self.upscale_passes
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // stage_1_scale_divisor,
            height=height // stage_1_scale_divisor,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        if guiding_images:
            stage_1_conditionings += image_conditionings_by_adding_guiding_latent(
                images=guiding_images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if video_conditioning:
            if int(video_conditioning_downscale_factor or 1) > 1:
                stage_1_conditionings += video_conditionings_by_reference_latent(
                    video_conditioning=video_conditioning,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    num_frames=num_frames,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                    downscale_factor=video_conditioning_downscale_factor,
                    tiling_config=tiling_config,
                )
            else:
                stage_1_conditionings += video_conditionings_by_keyframe(
                    video_conditioning=video_conditioning,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    num_frames=num_frames,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                    tiling_config=tiling_config,
                )

        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_1_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(stage_1_sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop_stage1,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            mask_context=mask_context,
        )
        bench_results: list[tuple[int, float, int]] = []
        if bench_transformer:
            stage1_transformer_ms, stage1_transformer_calls = transformer.consume()
            bench_results.append((1, stage1_transformer_ms, stage1_transformer_calls))
            print(
                "[WAN2GP][LTX2][bench] transformer pass1: "
                f"{stage1_transformer_ms / 1000.0:.3f}s ({stage1_transformer_calls} calls)"
            )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        current_output_shape = stage_1_output_shape
        for upscale_index in range(self.upscale_passes):
            pass_no = upscale_index + 2
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=self._get_model("spatial_upsampler"),
            )

            torch.cuda.synchronize()
            cleanup_memory()

            if loras_slists is not None:
                stage_2_steps = len(stage_2_sigmas) - 1
                update_loras_slists(
                    transformer,
                    loras_slists,
                    stage_2_steps,
                    phase_switch_step=0,
                    phase_switch_step2=stage_2_steps,
                )
            if callback is not None:
                callback(-1, None, True, override_num_inference_steps=len(stage_2_sigmas) - 1, pass_no=pass_no)

            def denoising_loop_stage_n(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: DiffusionStepProtocol,
                preview_tools: VideoLatentTools | None = None,
                mask_context=None,
                current_pass_no: int = pass_no,
                refiner_handler=self_refiner_handlers[pass_no - 1],
                refiner_audio_handler=self_refiner_audio_handlers[pass_no - 1],
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=video_context,
                        audio_context=audio_context,
                        transformer=transformer,  # noqa: F821
                        alt_guidance_scale=alt_guidance_scale,
                    ),
                    mask_context=mask_context,
                    interrupt_check=interrupt_check,
                    callback=callback,
                    preview_tools=preview_tools,
                    pass_no=current_pass_no,
                    transformer=transformer,
                    self_refiner_handler=refiner_handler,
                    self_refiner_handler_audio=refiner_audio_handler,
                    self_refiner_generator=generator,
                )

            scale_divisor = 2 ** (self.upscale_passes - upscale_index - 1)
            current_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=width // scale_divisor,
                height=height // scale_divisor,
                fps=frame_rate,
            )
            stage_2_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=current_output_shape.height,
                width=current_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
            if guiding_images_stage2:
                stage_2_conditionings += image_conditionings_by_adding_guiding_latent(
                    images=guiding_images_stage2,
                    height=current_output_shape.height,
                    width=current_output_shape.width,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                    tiling_config=tiling_config,
                )
            if latent_conditioning_stage2 is not None:
                expected_latent_height = current_output_shape.height // 32
                expected_latent_width = current_output_shape.width // 32
                if (
                    latent_conditioning_stage2.shape[3] == expected_latent_height
                    and latent_conditioning_stage2.shape[4] == expected_latent_width
                ):
                    stage_2_conditionings += latent_conditionings_by_latent_sequence(
                        latent_conditioning_stage2,
                        strength=1.0,
                        start_index=0,
                    )
            mask_context = prepare_mask_injection(
                masking_source=masking_source,
                masking_strength=masking_strength,
                output_shape=current_output_shape,
                video_encoder=video_encoder,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
                generator=mask_generator,
                num_steps=len(stage_2_sigmas) - 1,
            )
            freeze_audio_stage2 = audio_identity_guidance_scale > 0.0
            stage_2_audio_conditionings = audio_conditionings if audio_conditionings_stage2 is None else audio_conditionings_stage2
            video_state, audio_state = denoise_audio_video(
                output_shape=current_output_shape,
                conditionings=stage_2_conditionings,
                audio_conditionings=stage_2_audio_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop_stage_n,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
                noise_scale=stage_2_sigmas[0],
                audio_noise_scale=0.0 if freeze_audio_stage2 else stage_2_sigmas[0],
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=audio_state.latent,
                mask_context=mask_context,
                freeze_audio=freeze_audio_stage2,
            )
            if bench_transformer:
                stage_transformer_ms, stage_transformer_calls = transformer.consume()
                bench_results.append((pass_no, stage_transformer_ms, stage_transformer_calls))
                print(
                    f"[WAN2GP][LTX2][bench] transformer pass{pass_no}: "
                    f"{stage_transformer_ms / 1000.0:.3f}s ({stage_transformer_calls} calls)"
                )
            if video_state is None or audio_state is None:
                return None, None
            if interrupt_check is not None and interrupt_check():
                return None, None

        if bench_transformer:
            total_transformer_ms = sum(result[1] for result in bench_results)
            total_transformer_calls = sum(result[2] for result in bench_results)
            print(
                "[WAN2GP][LTX2][bench] transformer total: "
                f"{total_transformer_ms / 1000.0:.3f}s ({total_transformer_calls} calls)"
            )

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
        decoded_video = vae_decode_video_to_tensor(
            video_state.latent,
            self._get_model("video_decoder"),
            tiling_config,
            expected_frames=int(current_output_shape.frames),
            expected_height=int(current_output_shape.height),
            expected_width=int(current_output_shape.width),
            interrupt_check=interrupt_check,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self._get_model("audio_decoder"), self._get_model("vocoder")
        )
        if latent_slice is not None:
            return decoded_video, decoded_audio, latent_slice
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
