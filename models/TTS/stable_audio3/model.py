import json
import numpy as np
import torch
import typing as tp
from torch.nn.functional import interpolate

from models.TTS.stable_audio3.inference.audio_utils import prepare_audio, numpy_audio_to_tensor
from models.TTS.stable_audio3.inference.sampling import sample_diffusion
from models.TTS.stable_audio3.loading_utils import load_autoencoder, load_diffusion_cond
from models.TTS.stable_audio3.model_configs import ae_models, all_models
from models.TTS.stable_audio3.models.lora import (
    set_lora_strength as _set_lora_strength,
    load_and_apply_loras,
)


class StableAudioModel:
    def __init__(self, model, model_config, device, model_half):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.model_half = model_half
        self.same = self.model.pretransform
        self.dit = self.model.model
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def from_pretrained(model_name, device=None, model_half=True):
        # Load the model and any necessary components here
        if device is None and torch.cuda.is_available():
            device = "cuda"
        elif device is None and torch.backends.mps.is_available():
            device = "mps"
        elif device is None:
            device = "cpu"

        if not torch.cuda.is_available():
            if model_name in ("medium", "medium-base"):
                print(
                    f"Warning: You are loading the {model_name} model without a GPU. This model is not designed to run on cpu"
                )
            model_half = False

        if model_name not in all_models:
            raise ValueError(
                f"Unknown model '{model_name}'. Valid models: {list(all_models)}"
            )

        model_cfg = all_models[model_name]
        local_config, local_ckpt = model_cfg.resolve()
        with open(local_config) as f:
            model_config = json.load(f)

        model = load_diffusion_cond(
            model_config, local_ckpt, device=device, model_half=model_half
        )
        model.use_lora = False
        model.lora_names = []
        return StableAudioModel(model, model_config, device, model_half)

    def load_lora(self, lora_ckpt_paths):
        """Load LoRA checkpoints onto the model after construction."""
        model_type = self.model_config["model_type"]
        svd_bases_path = self.model_config.get("svd_bases_path")
        load_and_apply_loras(
            self.model, lora_ckpt_paths, model_type, svd_bases_path=svd_bases_path
        )

    def set_lora_strength(self, strength: float, lora_index: int | None = None):
        _set_lora_strength(self.model.model, strength, lora_index=lora_index)
        _set_lora_strength(self.model.conditioner, strength, lora_index=lora_index)

    @torch.inference_mode()
    def generate(
        self,
        # Simple path: pass a prompt string and duration
        prompt: str | list = None,
        negative_prompt: str | list = None,
        duration: float | list = 120,
        # Generation parameters
        steps: int = 8,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        sample_size: int = 5292032,
        truncate_output_to_duration: bool = True,
        # Low-level path: pass pre-built conditioning dicts
        conditioning: tp.Optional[tp.List[dict]] = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: tp.Optional[tp.List[dict]] = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        seed: int = -1,
        # Audio inputs
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask=None,
        inpaint_mask_start_seconds: tp.Optional[tp.Union[float, tp.List[float]]] = None,
        inpaint_mask_end_seconds: tp.Optional[tp.Union[float, tp.List[float]]] = None,
        # Schedule options
        duration_padding_sec: float = 6.0,
        apg_scale: float = 1.0,
        dist_shift=None,
        return_latents: bool = False,
        chunked_decode: tp.Optional[bool] = None,
        **sampler_kwargs,
    ) -> torch.Tensor:
        """
        Generate audio.

        Simple path:
            model.generate(prompt="...", duration=30, steps=100)

        Low-level path (pre-built conditioning):
            model.generate(conditioning=[{"prompt": "...", "seconds_total": 30}], steps=100, ...)

        Args:
            prompt: The text prompt to condition on. Ignored if conditioning dicts are provided directly.
            negative_prompt: The negative text prompt for classifier-free guidance. Ignored if negative_conditioning dicts are provided directly.
            duration: The duration of the generated audio in seconds. Only used if conditioning dicts with "seconds_total" are not provided.
            steps: The number of diffusion steps to use.
            cfg_scale: Classifier-free guidance scale
            batch_size: The batch size to use for generation.
            sample_size: The length of the audio to generate, in samples.
            truncate_output_to_duration: If True, truncate the output audio to the specified duration.
            conditioning: A dictionary of conditioning parameters to use for generation.
            conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
            negative_conditioning: A dictionary of negative conditioning parameters for classifier-free guidance.
            negative_conditioning_tensors: A dictionary of precomputed negative conditioning tensors for classifier-free guidance
            seed: The random seed to use for generation, or -1 to use a random seed.
            init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
            init_noise_level: The noise level to use when generating from an initial audio sample.
            inpaint_audio: A tuple of (sample_rate, audio) to use as the source audio for inpainting. The inpaint region will be determined by the inpaint_mask or inpaint_mask_start_seconds/inpaint_mask_end_seconds parameters.
            inpaint_mask: A prebuilt mask tensor for inpainting. Shape should be [batch_size, sample_size].
                Ignored if inpaint_mask_start_seconds/inpaint_mask_end_seconds are provided.
            inpaint_mask_start_seconds: Start of the inpaint region in seconds. Can be a float
                for a single region, or a list of floats for multiple non-contiguous regions.
            inpaint_mask_end_seconds: End of the inpaint region in seconds. Can be a float
                for a single region, or a list of floats matching inpaint_mask_start_seconds.
            duration_padding_sec: Extra seconds to add when adapting duration (default 6.0).
            apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG.
            dist_shift: Optional distribution shift override for sampling. If None, uses model.sampling_dist_shift.
            return_latents: Whether to return the latents used for generation instead of the decoded audio.
            chunked_decode: Whether to decode latents in overlapping chunks to reduce peak VRAM. True forces
                chunked decoding on, False forces it off, None (default) uses the value set in the model config.
            **sampler_kwargs: Additional keyword arguments to pass to the sampler.
        """

        device = str(self.device)

        # Build conditioning from prompt string if not provided directly
        if conditioning is None and conditioning_tensors is None:
            assert prompt is not None, "Must provide either prompt or conditioning"
            conditioning, negative_conditioning = self._build_conditioning_dicts(
                prompt, negative_prompt, duration, batch_size
            )

        # Adapt sample size based on seconds_total in conditioning
        audio_sample_size = sample_size
        if conditioning is not None:
            audio_sample_size = self._adapt_sample_size(
                conditioning,
                sample_size,
                duration_padding_sec,
            )

        # Convert audio sample size to latent size
        latent_sample_size = audio_sample_size
        if self.model.pretransform is not None:
            latent_sample_size = (
                audio_sample_size // self.model.pretransform.downsampling_ratio
            )

        # Build inpaint mask from seconds if provided
        if (
            inpaint_mask_start_seconds is not None
            and inpaint_mask_end_seconds is not None
        ):
            start_is_list = isinstance(inpaint_mask_start_seconds, list)
            end_is_list = isinstance(inpaint_mask_end_seconds, list)
            if start_is_list != end_is_list:
                raise ValueError(
                    "inpaint_mask_start_seconds and inpaint_mask_end_seconds must both be "
                    "scalars or both be lists, got "
                    f"{type(inpaint_mask_start_seconds).__name__} and "
                    f"{type(inpaint_mask_end_seconds).__name__}."
                )
            starts = (
                inpaint_mask_start_seconds
                if start_is_list
                else [inpaint_mask_start_seconds]
            )
            ends = (
                inpaint_mask_end_seconds if end_is_list else [inpaint_mask_end_seconds]
            )
            if len(starts) != len(ends):
                raise ValueError(
                    f"inpaint_mask_start_seconds and inpaint_mask_end_seconds must have the same "
                    f"length, got {len(starts)} and {len(ends)}."
                )
            inpaint_mask = torch.ones(1, audio_sample_size, device=device)
            for start_sec, end_sec in zip(starts, ends):
                mask_start_samples = min(
                    int(start_sec * self.model.sample_rate),
                    audio_sample_size,
                )
                mask_end_samples = min(
                    int(end_sec * self.model.sample_rate),
                    audio_sample_size,
                )
                inpaint_mask[:, mask_start_samples:mask_end_samples] = 0

        # If the caller passed a prebuilt mask sized to the un-adapted sample_size (or
        # anything longer than audio_sample_size), truncate to audio_sample_size so the
        # downstream nearest-neighbor interpolation preserves the mask's time-domain
        # positions instead of squashing the mask region.
        if inpaint_mask is not None and inpaint_mask.shape[-1] > audio_sample_size:
            inpaint_mask = inpaint_mask[:, :audio_sample_size]

        # Match training: when mask_padding_attention is used, random_inpaint_mask
        # zeroes the mask past real_sequence_length. Apply the
        # same convention here so the mask matches the training distribution, whether
        # it was built from seconds above or passed in by the caller.
        if inpaint_mask is not None and conditioning is not None:
            max_seconds = max(
                (c.get("seconds_total", 0.0) for c in conditioning), default=0.0
            )
            if max_seconds > 0:
                effective_audio_len = int(max_seconds * self.model.sample_rate)
                mask_len = inpaint_mask.shape[-1]
                if effective_audio_len < mask_len:
                    inpaint_mask = inpaint_mask.clone()
                    inpaint_mask[:, effective_audio_len:] = 0

        if inpaint_mask is not None:
            inpaint_mask = inpaint_mask.float()

        # Seed and noise
        seed = seed if seed != -1 else np.random.randint(0, 99999)
        torch.manual_seed(seed)
        noise = torch.randn(
            [batch_size, self.model.io_channels, latent_sample_size], device=device
        )

        # Encode conditioning
        if conditioning_tensors is None:
            conditioning_tensors = self.model.conditioner(conditioning, device)
        if (
            negative_conditioning is not None
            or negative_conditioning_tensors is not None
        ):
            if negative_conditioning_tensors is None:
                negative_conditioning_tensors = self.model.conditioner(
                    negative_conditioning, device
                )
        else:
            negative_conditioning_tensors = {}

        # Process init audio
        if init_audio is not None:
            init_audio, inpaint_mask = self._encode_audio_input(
                init_audio, audio_sample_size, inpaint_mask
            )
            init_audio = init_audio.repeat(batch_size, 1, 1)

        # Process inpaint audio
        if inpaint_audio is not None:
            inpaint_audio, inpaint_mask = self._encode_audio_input(
                inpaint_audio, audio_sample_size, inpaint_mask
            )
            inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
        else:
            if inpaint_mask is not None:
                inpaint_mask = interpolate(
                    inpaint_mask.unsqueeze(1), size=latent_sample_size, mode="nearest"
                ).squeeze(1)

        # Build inpaint mask tensor and masked input
        if inpaint_mask is None:
            mask = torch.zeros((batch_size, 1, latent_sample_size), device=device)
        else:
            mask = inpaint_mask.unsqueeze(1)
        mask = mask.to(device)

        inpaint_input = (
            inpaint_audio * mask.expand_as(inpaint_audio)
            if inpaint_audio is not None
            else torch.zeros(
                (batch_size, self.model.io_channels, latent_sample_size), device=device
            )
        )

        conditioning_tensors["inpaint_mask"] = [mask]
        conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        conditioning_inputs = self.model.get_conditioning_inputs(conditioning_tensors)

        if negative_conditioning_tensors:
            negative_conditioning_tensors["inpaint_mask"] = [mask]
            negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
            negative_conditioning_tensors = self.model.get_conditioning_inputs(
                negative_conditioning_tensors, negative=True
            )

        model_dtype = next(self.model.model.parameters()).dtype
        noise = noise.type(model_dtype)
        conditioning_inputs = {
            k: v.type(model_dtype) if v is not None else v
            for k, v in conditioning_inputs.items()
        }

        cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

        sampler_type = sampler_kwargs.pop("sampler_type", None)

        result = sample_diffusion(
            model=self.model.model,
            noise=noise,
            cond_inputs=cond_inputs,
            diffusion_objective=self.model.diffusion_objective,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_rate=self.model.sample_rate,
            pretransform=self.model.pretransform,
            mask_padding_attention=True,
            use_effective_length_for_schedule=True,
            headroom_seconds=duration_padding_sec,
            dist_shift=dist_shift
            if dist_shift is not None
            else self.model.sampling_dist_shift,
            sampler_type=sampler_type,
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=apg_scale,
            init_data=init_audio,
            init_noise_level=init_noise_level,
            decode=not return_latents,
            chunked_decode=chunked_decode,
            **sampler_kwargs,
        )

        if not return_latents:
            result = result.to(torch.float32).clamp(-1, 1)

        if not return_latents and truncate_output_to_duration:
            if isinstance(duration, (int, float)):
                max_length_samples = int(duration * self.model.sample_rate)
                result = result[:, :, :max_length_samples]
            else:
                if torch.all(torch.tensor(duration) == duration[0]):
                    max_length_samples = int(duration[0] * self.model.sample_rate)
                    result = result[:, :, :max_length_samples]
                else:
                    # Warn that we can't truncate to a single duration if the durations are different, and return the full length output
                    print(
                        "Warning: Cannot truncate output to a single duration when passing a list of different durations"
                    )

        return result

    # --- generate() helpers ---

    @staticmethod
    def _build_conditioning_dicts(prompt, negative_prompt, duration, batch_size):
        """Returns (conditioning, negative_conditioning) lists of dicts."""

        def _to_list(value, name):
            """Broadcast a scalar or validate a sequence to length batch_size."""
            if isinstance(value, (list, tuple)):
                assert len(value) == batch_size, (
                    f"Length of {name} ({len(value)}) must match batch_size ({batch_size})"
                )
                return list(value)
            return [value] * batch_size

        prompts = _to_list(prompt, "prompt")
        durations = _to_list(duration, "duration")
        conditioning = [
            {"prompt": p, "seconds_total": d} for p, d in zip(prompts, durations)
        ]

        negative_conditioning = None
        if negative_prompt is not None:
            neg_prompts = _to_list(negative_prompt, "negative_prompt")
            negative_conditioning = [
                {"prompt": p, "seconds_total": d}
                for p, d in zip(neg_prompts, durations)
            ]

        return conditioning, negative_conditioning

    def _adapt_sample_size(self, conditioning, sample_size, duration_padding_sec):
        """Returns audio_sample_size adapted from conditioning, clamped to sample_size."""
        max_seconds = 0.0
        for cond_dict in conditioning:
            if "seconds_total" in cond_dict:
                max_seconds = max(max_seconds, cond_dict["seconds_total"])

        if max_seconds <= 0:
            return sample_size

        target_audio_samples = int(
            (max_seconds + duration_padding_sec) * self.model.sample_rate
        )
        if self.model.pretransform is not None:
            ds_ratio = self.model.pretransform.downsampling_ratio
            # Round up to nearest multiple of downsampling ratio
            target_audio_samples = (
                (target_audio_samples + ds_ratio - 1) // ds_ratio
            ) * ds_ratio
            encoder_config = self.model_config["model"]["pretransform"]["config"][
                "encoder"
            ]["config"]
            chunk_size = encoder_config.get("chunk_size", 32)
            stride = encoder_config["strides"][0]  # or min(strides) if multiple
            # For chunked attention with latent space, align to chunk size after downsampling
            latent_align = chunk_size // stride
            align = ds_ratio * latent_align
            target_audio_samples = ((target_audio_samples + align - 1) // align) * align

        return min(target_audio_samples, sample_size)

    def _encode_audio_input(self, audio_input, audio_sample_size, inpaint_mask=None):
        """
        Converts a (sample_rate, audio) tuple to an encoded latent tensor.
        If model has a pretransform, encodes to latent space and downsamples inpaint_mask to match.
        Returns (encoded_audio, updated_inpaint_mask). encoded_audio is not yet repeated to batch size.
        """
        device = str(self.device)
        in_sr, audio_data = audio_input
        if isinstance(audio_data, np.ndarray):
            audio_data = numpy_audio_to_tensor(audio_data)
        io_channels = (
            self.model.pretransform.io_channels
            if self.model.pretransform is not None
            else self.model.io_channels
        )
        audio = prepare_audio(
            audio_data,
            in_sr=in_sr,
            target_sr=self.model.sample_rate,
            target_length=audio_sample_size,
            target_channels=io_channels,
            device=device,
        )
        if self.model.pretransform is not None:
            target_latent_len = audio_sample_size // self.model.pretransform.downsampling_ratio
            audio = audio.to(next(self.model.pretransform.parameters()).dtype)
            audio = self.model.pretransform.encode(audio)
            if audio.shape[-1] > target_latent_len:
                audio = audio[..., :target_latent_len]
            elif audio.shape[-1] < target_latent_len:
                audio = torch.nn.functional.pad(audio, (0, target_latent_len - audio.shape[-1]))
            if inpaint_mask is not None:
                inpaint_mask = interpolate(
                    inpaint_mask.unsqueeze(1),
                    size=target_latent_len,
                    mode="nearest",
                ).squeeze(1)
        return audio, inpaint_mask


class AutoencoderModel:
    def __init__(self, autoencoder, sample_rate, device):
        self.autoencoder = autoencoder
        self.sample_rate = sample_rate
        self.device = device

    @staticmethod
    def from_pretrained(model_name, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if not torch.cuda.is_available():
            if model_name == "same-l":
                print(
                    f"Warning: You are loading the {model_name} model without a GPU. This model is not designed to run on cpu"
                )

        if model_name not in ae_models:
            raise ValueError(
                f"Unknown autoencoder '{model_name}'. Valid models: {list(ae_models)}"
            )

        cfg = ae_models[model_name]
        local_config, local_ckpt = cfg.resolve()

        with open(local_config) as f:
            sample_rate = json.load(f)["sample_rate"]

        autoencoder = load_autoencoder(local_config, local_ckpt, device=device)
        autoencoder.eval().requires_grad_(False)

        return AutoencoderModel(autoencoder, sample_rate, device)

    @torch.inference_mode()
    def encode(self, audio, sr, chunked=False, chunk_size=128, overlap=32):
        """Encode audio to latents.

        Args:
            audio: A single waveform tensor (C, T), a list of waveform tensors,
                or a pre-batched tensor (B, C, T). Resampling, channel conversion,
                and padding are handled automatically; passing sr=ae.sample_rate
                for already-preprocessed audio skips resampling.
            sr: Sample rate of the input audio, or a list of sample rates when
                audio is a list.
            chunked: If True, encode in overlapping chunks to save memory.
            chunk_size: Chunk size in latent frames (only used when chunked=True).
            overlap: Overlap in latent frames between chunks (only used when chunked=True).

        Returns:
            Latent tensor of shape (B, latent_dim, latent_time).
        """
        if isinstance(audio, list):
            preprocessed = self.autoencoder.preprocess_audio_list_for_encoder(
                audio, in_sr_list=sr
            )
        elif isinstance(audio, torch.Tensor) and audio.dim() == 3:
            sr_list = sr if isinstance(sr, list) else [sr] * audio.shape[0]
            preprocessed = self.autoencoder.preprocess_audio_list_for_encoder(
                list(audio), in_sr_list=sr_list
            )
        else:
            preprocessed = self.autoencoder.preprocess_audio_for_encoder(
                audio, in_sr=sr
            )
        return self.autoencoder.encode_audio(
            preprocessed.to(self.device),
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    @torch.inference_mode()
    def decode(self, latents, chunked=False, chunk_size=128, overlap=32):
        """Decode latents to audio.

        Args:
            latents: Latent tensor of shape (B, latent_dim, latent_time).
            chunked: If True, decode in overlapping chunks to save memory.
            chunk_size: Chunk size in latent frames (only used when chunked=True).
            overlap: Overlap in latent frames between chunks (only used when chunked=True).

        Returns:
            Audio tensor of shape (B, channels, samples).
        """
        return self.autoencoder.decode_audio(
            latents,
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
        )

