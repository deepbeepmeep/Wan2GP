import os
import random
import re
from dataclasses import replace
from typing import Optional

import torch

from .ltx_audio_tts import LTXAudioTTSPipelineBase
from .ltx_core.components.schedulers import LTX2Scheduler
from .ltx_core.conditioning import AudioConditionByAppendedReferenceLatent
from .ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE


DRAMABOX_DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"
DRAMABOX_FPS = 25.0
DRAMABOX_DEFAULT_STEPS = 30
DRAMABOX_DEFAULT_DURATION_MULTIPLIER = 1.1
DRAMABOX_DEFAULT_REFERENCE_SECONDS = 10.0
DRAMABOX_DEFAULT_CFG_SCALE = 2.5
DRAMABOX_DEFAULT_STG_SCALE = 1.5
DRAMABOX_REFERENCE_PEAK_DB = -4.0
DRAMABOX_STG_BLOCK = 29


_LAUGH_VERBS = {
    r"\blaugh(?:s|ed|ing)?\b": 1.5,
    r"\bcackl(?:e|es|ed|ing)\b": 1.5,
    r"\bchuckl(?:e|es|ed|ing)\b": 1.0,
    r"\bgiggl(?:e|es|ed|ing)\b": 1.0,
    r"\bsnicker(?:s|ed|ing)?\b": 0.8,
    r"\bcru?el laugh\b": 1.5,
}


def _read_text_or_file(value, label: str) -> str:
    if value is None:
        return ""
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    if os.path.isfile(text) and os.path.splitext(text)[1].lower() in {".txt", ".xml"}:
        with open(text, "r", encoding="utf-8") as reader:
            return reader.read()
    return text


def _contextual_laugh_duration(text: str) -> float:
    short_mod = re.compile(r"^\s*(?:[a-z]+ly )?(?:briefly|shortly|once|quickly)", re.IGNORECASE)
    long_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:maniacally|heartily|uproariously|uncontrollably|hysterically|darkly|wickedly|evilly|loudly|long)|^\s*between phrases",
        re.IGNORECASE,
    )
    total = 0.0
    for pattern, base_duration in _LAUGH_VERBS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            context = text[match.end() : match.end() + 40]
            if short_mod.match(context):
                total += base_duration * 0.4
            elif long_mod.match(context):
                total += base_duration * 1.2
            else:
                total += base_duration

    for quoted in re.findall(r'"([^"]+)"', text) + re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text):
        for run in re.findall(r"(?:h[ae]){3,}|(?:h[ae][ \-]?){3,}", quoted, re.IGNORECASE):
            syllables = len(re.findall(r"h[ae]", run, re.IGNORECASE))
            total += 0.2 * max(syllables - 2, 0)
    return total


def _estimate_nonverbal_duration(text: str) -> float:
    patterns = {
        r"\bsighs?\b": 0.8,
        r"\bshaky breath\b": 1.0,
        r"\bbreathing deeply\b": 1.0,
        r"\bgasps?\b": 0.5,
        r"\bburps?\b": 0.5,
        r"\byawns?\b": 1.0,
        r"\bpants?\b": 0.8,
        r"\bwheezes?\b": 0.8,
        r"\bcoughs?\b": 0.8,
        r"\bsniffles?\b": 0.5,
        r"\bsnorts?\b": 0.3,
        r"\bgroans?\b": 0.8,
        r"\blong pause\b": 1.0,
        r"\bpauses? briefly\b": 0.3,
        r"\bpauses?\b": 0.5,
        r"\bsilence\b": 1.0,
        r"\blets? the .{1,20} hang\b": 1.0,
        r"\blets? .{1,20} sink in\b": 1.0,
        r"\bslams?\b": 0.5,
        r"\bclaps?\b": 0.3,
        r"\bdraws? (?:his|her|a) sword\b": 0.5,
        r"\btakes? a (?:drag|swig|sip|drink)\b": 0.5,
        r"\bwhistles?\b": 1.0,
        r"\bhums?\b": 0.8,
        r"\bmutters?\b": 1.5,
        r"\bmumbles?\b": 1.0,
        r"\bwhispers?\b": 0.0,
        r"\bclears? (?:his|her) throat\b": 0.5,
        r"\bgulps?\b": 0.5,
        r"\bswallows?\b": 0.5,
        r"\bvoice (?:breaks?|cracks?|trembles?|drops?|rises?)\b": 0.5,
        r"\bsteadies? (?:him|her)self\b": 1.0,
        r"\bcatches? (?:his|her) breath\b": 1.0,
        r"\bcomposes? (?:him|her)self\b": 0.8,
        r"\bdemeanor shifts?\b": 0.5,
        r"\bsettles? in\b": 0.5,
        r"\bleans? in\b": 0.3,
        r"\bwipes? (?:his|her) eyes\b": 0.5,
    }
    extra = 0.0
    for pattern, duration in patterns.items():
        extra += duration * len(re.findall(pattern, text, re.IGNORECASE))
    return extra + _contextual_laugh_duration(text)


def estimate_speech_duration(text: str, speed: float = 1.0) -> float:
    quotes = re.findall(r'"([^"]+)"', text)
    if not quotes:
        quotes = re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text)
        quotes = [quote for quote in quotes if len(quote.split()) > 3]
    if quotes:
        spoken = " ".join(quotes)
    elif ":" in text:
        spoken = text.split(":", 1)[1].strip()
    else:
        spoken = text

    chars_per_second = 14.0
    text_length = len(spoken)
    if text_length < 40:
        chars_per_second *= 0.6
    elif text_length < 80:
        chars_per_second *= 0.8
    chars_per_second *= speed

    duration = text_length / chars_per_second
    duration += (spoken.count(".") + spoken.count("!") + spoken.count("?")) * 0.3
    duration += _estimate_nonverbal_duration(text)
    return max(3.0, round(duration + 2.0, 1))


class DramaBoxAudioPipeline(LTXAudioTTSPipelineBase):
    def __init__(
        self,
        model_weights_path: str,
        gemma_path: str,
        audio_vae_path: str,
        vocoder_path: str,
        text_projection_path: str,
        text_connector_path: str,
        config_path: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_weights_path=model_weights_path,
            gemma_path=gemma_path,
            audio_vae_path=audio_vae_path,
            vocoder_path=vocoder_path,
            text_projection_path=text_projection_path,
            text_connector_path=text_connector_path,
            config_path=config_path,
            device=device,
            dtype=dtype,
        )

    def _encode_voice_reference(self, input_waveform, input_waveform_sample_rate, audio_guide: str | None):
        waveform, sample_rate = self._waveform_from_input(input_waveform, input_waveform_sample_rate, audio_guide)
        if waveform is None or sample_rate <= 0:
            return None
        reference_seconds = DRAMABOX_DEFAULT_REFERENCE_SECONDS
        target_samples = max(1, int(round(float(reference_seconds) * sample_rate)))
        if waveform.shape[-1] < target_samples:
            repeat = (target_samples // max(1, waveform.shape[-1])) + 1
            waveform = waveform.repeat(1, repeat)
        waveform = waveform[:, :target_samples]
        target_peak = 10 ** (DRAMABOX_REFERENCE_PEAK_DB / 20.0)
        return self._encode_reference_waveform(waveform, sample_rate, max_seconds=reference_seconds, normalize_peak=target_peak)

    @staticmethod
    def _patch_long_clip_silence_prior(audio_state):
        latent = audio_state.latent
        if latent.shape[2] <= 513:
            return audio_state
        f0, f1 = 511, 514
        span = f1 - f0
        patched = latent.clone()
        for frame in (512, 513):
            amount = (frame - f0) / span
            patched[:, :, frame, :] = (1.0 - amount) * latent[:, :, f0, :] + amount * latent[:, :, f1, :]
        return replace(audio_state, latent=patched)

    def _target_duration(self, prompt: str, duration_seconds, duration_multiplier: float) -> float:
        try:
            explicit_duration = float(duration_seconds or 0)
        except (TypeError, ValueError):
            explicit_duration = 0.0
        if explicit_duration > 0:
            return explicit_duration
        return max(1.0, round(estimate_speech_duration(prompt) * float(duration_multiplier), 1))

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str] = None,
        audio_guide: Optional[str] = None,
        *,
        alt_prompt: Optional[str] = None,
        image_start=None,
        image_end=None,
        input_frames=None,
        input_frames2=None,
        input_ref_images=None,
        input_ref_masks=None,
        input_masks=None,
        input_masks2=None,
        input_video=None,
        input_faces=None,
        input_custom=None,
        denoising_strength=None,
        masking_strength=None,
        prefix_frames_count=None,
        frame_num=None,
        batch_size=None,
        height=None,
        width=None,
        fit_into_canvas=None,
        shift=None,
        sample_solver=None,
        sampling_steps: int = DRAMABOX_DEFAULT_STEPS,
        guide_scale: float = DRAMABOX_DEFAULT_CFG_SCALE,
        guide2_scale=None,
        guide3_scale=None,
        switch_threshold=None,
        switch2_threshold=None,
        guide_phases=None,
        model_switch_phase=None,
        embedded_guidance_scale=None,
        n_prompt=None,
        seed: int = -1,
        callback=None,
        enable_RIFLEx=None,
        VAE_tile_size=None,
        joint_pass=None,
        perturbation_switch=None,
        perturbation_layers=None,
        perturbation_start=None,
        perturbation_end=None,
        apg_switch=None,
        cfg_star_switch=None,
        cfg_zero_step=None,
        alt_guide_scale=None,
        audio_cfg_scale=None,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_guide2: Optional[str] = None,
        audio_prompt_type: str = "",
        audio_proj=None,
        audio_scale=None,
        audio_context_lens=None,
        context_scale=None,
        control_scale_alt=None,
        alt_scale=None,
        motion_amplitude=None,
        model_mode_override=None,
        causal_block_size=None,
        causal_attention=None,
        fps=None,
        overlapped_latents=None,
        return_latent_slice=None,
        overlap_noise=None,
        overlap_size=None,
        color_correction_strength=None,
        conditioning_latents_size=None,
        input_video_is_hdr=None,
        lora_dir=None,
        keep_frames_parsed=None,
        model_filename=None,
        model_type=None,
        loras_slists=None,
        NAG_scale=None,
        NAG_tau=None,
        NAG_alpha=None,
        speakers_bboxes=None,
        image_mode=None,
        video_prompt_type=None,
        window_no=None,
        offloadobj=None,
        set_header_text=None,
        pre_video_frame=None,
        prefix_video=None,
        original_input_ref_images=None,
        image_refs_relative_size=None,
        outpainting_dims=None,
        face_arc_embeds=None,
        custom_settings=None,
        temperature: float = 0.0,
        window_start_frame_no=None,
        input_video_strength=None,
        self_refiner_setting=None,
        self_refiner_plan=None,
        self_refiner_f_uncertainty=None,
        self_refiner_certain_percentage=None,
        duration_seconds: Optional[float] = None,
        pause_seconds: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
        set_progress_status=None,
        loras_selected=None,
        frames_relative_positions_list=None,
        frames_to_inject=None,
        verbose_level: int = 0,
    ) -> Optional[dict]:
        self._interrupt = False
        self._early_stop = False
        prompt = _read_text_or_file(input_prompt, "Prompt").strip()
        if not prompt:
            raise ValueError("Prompt text cannot be empty for DramaBox Audio.")

        seed = random.randrange(0, 2**31) if seed is None or int(seed) < 0 else int(seed)
        duration_multiplier = self._custom_float(custom_settings, "duration_multiplier", DRAMABOX_DEFAULT_DURATION_MULTIPLIER)
        stg_scale = DRAMABOX_DEFAULT_STG_SCALE if audio_cfg_scale is None else float(audio_cfg_scale)
        rescale_scale = 0.0 if alt_scale is None else float(alt_scale)

        if set_progress_status is not None:
            set_progress_status("Encoding Voice Reference")
        ref_latent = None
        use_reference = "A" in str(audio_prompt_type or "").upper() or audio_guide is not None or input_waveform is not None
        if use_reference:
            ref_latent = self._encode_voice_reference(input_waveform, input_waveform_sample_rate, audio_guide)
            if ref_latent is None and "A" in str(audio_prompt_type or "").upper():
                raise ValueError("DramaBox Audio reference mode requires a reference audio file.")
        if self._interrupt:
            return None

        if set_progress_status is not None:
            set_progress_status("Encoding Prompt")
        cfg_scale = float(guide_scale)
        negative_prompt = _read_text_or_file(n_prompt, "Negative prompt").strip() or DRAMABOX_DEFAULT_NEGATIVE_PROMPT
        if cfg_scale > 1.0:
            audio_context, audio_context_n = self._encode_prompts([prompt, negative_prompt])
        else:
            audio_context = self._encode_prompt(prompt)
            audio_context_n = None
        if self._interrupt:
            return None

        duration = self._target_duration(prompt, duration_seconds, duration_multiplier)
        if set_header_text is not None:
            set_header_text(f"DramaBox Audio - {duration:.1f}s")

        audio_state, audio_tools = self._build_audio_state(
            duration,
            DRAMABOX_FPS,
            torch.empty(0, dtype=torch.float32, device=self.device),
            seed,
            ref_latent=ref_latent,
            reference_conditioner=AudioConditionByAppendedReferenceLatent,
        )
        sigmas = LTX2Scheduler().execute(steps=max(1, int(sampling_steps or DRAMABOX_DEFAULT_STEPS)), latent=audio_state.latent).to(self.device)
        audio_state = self._generate_audio_euler(
            audio_context,
            sigmas,
            audio_state,
            audio_tools,
            audio_context_n=audio_context_n,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            stg_blocks=[DRAMABOX_STG_BLOCK],
            rescale_scale=rescale_scale,
            callback=callback,
            set_progress_status=set_progress_status,
        )
        if audio_state is None or self._interrupt:
            return None

        audio_state = self._patch_long_clip_silence_prior(audio_state)
        audio = self._decode_audio_state(audio_state, set_progress_status=set_progress_status)
        output_audio_sampling_rate = int(getattr(self.vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE))
        return {"x": audio, "audio_sampling_rate": output_audio_sampling_rate}
