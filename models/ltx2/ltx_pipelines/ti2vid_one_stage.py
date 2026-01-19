import logging
from collections.abc import Callable, Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.guiders import CFGGuider
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoPixelShape
from .utils import ModelLedger
from .utils.args import default_1_stage_arg_parser
from .utils.constants import AUDIO_SAMPLE_RATE
from .utils.helpers import (
    assert_resolution,
    bind_interrupt_check,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_replacing_latent,
    prepare_mask_injection,
)
from .utils.media_io import encode_video
from .utils.types import PipelineComponents
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.text_encoder_cache import TextEncoderCache

device = get_device()


class TI2VidOneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.
    Generates video at the target resolution in a single diffusion pass with
    classifier-free guidance (CFG). Supports optional image conditioning via
    the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        fp8transformer: bool = False,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.text_encoder_cache = TextEncoderCache()

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        enhance_prompt: bool = False,
        audio_conditionings: list | None = None,
        callback: Callable[..., None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
        loras_slists: dict | None = None,
        text_connectors: dict | None = None,
        masking_source: dict | None = None,
        masking_strength: float | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        text_encoder = self.model_ledger.text_encoder()
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(
            text_encoder, text_connectors
        )
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        contexts = self.text_encoder_cache.encode(
            encode_fn,
            [prompt, negative_prompt],
            device=self.device,
            parallel=True,
        )

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        context_p, context_n = contexts
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        bind_interrupt_check(transformer, interrupt_check)
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
        if loras_slists is not None:
            stage_1_steps = len(sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_1_steps,
                phase_switch_step=stage_1_steps,
                phase_switch_step2=stage_1_steps,
            )

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(sigmas) - 1, pass_no=1)

        def first_stage_denoising_loop(
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
                denoise_fn=guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer,  # noqa: F821
                ),
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=1,
            )

        stage_1_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_1_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=None,
            generator=mask_generator,
            num_steps=len(sigmas) - 1,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            mask_context=mask_context,
        )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder())
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_1_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=1,
    )


if __name__ == "__main__":
    main()
