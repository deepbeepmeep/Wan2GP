import logging
from collections.abc import Callable, Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoPixelShape
from .utils import ModelLedger
from .utils.args import default_2_stage_distilled_arg_parser
from .utils.constants import (
    AUDIO_SAMPLE_RATE,
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
    image_conditionings_by_replacing_latent,
    latent_conditionings_by_latent_sequence,
    prepare_mask_injection,
    simple_denoising_func,
    video_conditionings_by_keyframe,
)
from .utils.media_io import encode_video
from .utils.types import PipelineComponents
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.text_encoder_cache import TextEncoderCache

device = get_device()


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
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
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.models = models

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
        video_conditioning: list[tuple[str, float]] | None = None,
        latent_conditioning_stage2: torch.Tensor | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        audio_conditionings: list | None = None,
        callback: Callable[..., None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
        loras_slists: dict | None = None,
        text_connectors: dict | None = None,
        masking_source: dict | None = None,
        masking_strength: float | None = None,
        return_latent_slice: slice | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
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
        contexts = self.text_encoder_cache.encode(encode_fn, [prompt], device=self.device, parallel=True)

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        video_context, audio_context = contexts[0]

        # Stage 1: Initial low resolution video generation.
        video_encoder = self._get_model("video_encoder")
        transformer = self._get_model("transformer")
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

        def denoising_loop(
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
                ),
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=pass_no,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
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
        if video_conditioning:
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
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            mask_context=mask_context,
        )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self._get_model("spatial_upsampler"),
        )

        torch.cuda.synchronize()
        cleanup_memory()

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        pass_no = 2
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
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        if latent_conditioning_stage2 is not None:
            stage_2_conditionings += latent_conditionings_by_latent_sequence(
                latent_conditioning_stage2,
                strength=1.0,
                start_index=0,
            )
        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_2_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(stage_2_sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            mask_context=mask_context,
        )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
        decoded_video = vae_decode_video(video_state.latent, self._get_model("video_decoder"), tiling_config)
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
