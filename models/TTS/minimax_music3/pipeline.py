# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Single-GPU MiniMax Music 3 pipeline adapted for WanGP and MMGP."""

import re
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from mmgp import offload
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3ForCausalLM

from .condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
from .minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from .minimax_music3_vocoder import MiniMaxMusic3Vocoder
from .qwen3_accelerated import MiniMaxMusic3Qwen3ForCausalLM
from .semantic_acceleration import MiniMaxMusic3SemanticAcceleration
from .transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel


_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END_TOKEN_ID = 151670
_AUDIO_CFG_TOKEN_ID = 151654
_AUDIO_CODE_OFFSET = 151675
_SEMANTIC_VOCAB_SIZE = 16384
_MAX_PROMPT_TOKENS = 5000
_MAX_AUDIO_FRAMES = 9000
_AR_CFG_SCALE = 1.5
_AR_CFG_TOP_K = 50
_AR_SAMPLING_TOP_K = 50
_CHUNK_FRAMES = 200
_CHUNK_HOP = 100
_OVERLAP_LATENT_LENGTH = 172
_CROP_LEFT_LATENT = 86
_CROP_RIGHT_LATENT = 258

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _clean_caption(caption: str) -> str:
    def rewrite_special_tag(match: re.Match) -> str:
        parts = match.group(1).strip().split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else parts[0]

    text = _SPECIAL_TAG_RE.sub(rewrite_special_tag, caption)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", "\n".join(lines), flags=re.MULTILINE)
    return re.sub(r"\n{2,}", "\n", text.replace("• ", "").replace("    ", ""))


def _normalize_lyrics(lyrics: str) -> str:
    lines = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        lines.append(match.group(1).strip() if match else line)
    text = "\n".join(lines).replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def _sample_top_k(logits: torch.Tensor, generator: torch.Generator, no_sync: bool = False) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, min(_AR_SAMPLING_TOP_K, values.shape[-1]), dim=-1).values[..., -1, None]
    values.masked_fill_(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min_(1e-12))
    if no_sync:
        noise = torch.empty_like(probs).exponential_(1, generator=generator)
        return torch.argmax(probs / noise, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


class MiniMaxMusic3Pipeline:
    frame_rate = 25.0
    latent_hop_length = 512
    sample_rate = 44100

    def __init__(self, flow_weights: str, text_encoder_weights: str, asset_paths: dict[str, str], dtype: torch.dtype, lm_decoder_engine: str = "legacy"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.lm_decoder_engine = lm_decoder_engine
        self._interrupt = False
        self._early_stop = False

        self.transformer = offload.fast_load_transformers_model(
            flow_weights,
            modelClass=MiniMaxMusic3Transformer1DModel,
            defaultConfigPath=asset_paths["transformer_config"],
            default_dtype=dtype,
            writable_tensors=False,
        )
        accelerated = lm_decoder_engine in ("cg", "vllm")
        self.text_encoder = offload.fast_load_transformers_model(
            text_encoder_weights,
            modelClass=MiniMaxMusic3Qwen3ForCausalLM if accelerated else Qwen3ForCausalLM,
            defaultConfigPath=asset_paths["text_encoder_config"],
            configKwargs={"rope_theta": 1_000_000.0, "max_position_embeddings": 16384} if accelerated else {"rope_theta": 1_000_000.0},
            default_dtype=dtype,
            writable_tensors=False,
        )
        self.rvq_depth_decoder = offload.fast_load_transformers_model(
            asset_paths["rvq_weights"],
            modelClass=MiniMaxMusic3RVQDepthDecoder,
            defaultConfigPath=asset_paths["rvq_config"],
            default_dtype=dtype,
            writable_tensors=False,
        )
        self.condition_encoder = offload.fast_load_transformers_model(
            asset_paths["condition_weights"],
            modelClass=MiniMaxMusic3ConditionEncoder,
            defaultConfigPath=asset_paths["condition_config"],
            default_dtype=None,
            writable_tensors=False,
        )
        self.vocoder = offload.fast_load_transformers_model(
            asset_paths["vocoder_weights"],
            modelClass=MiniMaxMusic3Vocoder,
            defaultConfigPath=asset_paths["vocoder_config"],
            default_dtype=None,
            writable_tensors=False,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(asset_paths["scheduler_dir"], local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(asset_paths["tokenizer_dir"], local_files_only=True, use_fast=True)
        self.sample_rate = int(self.vocoder.config.sampling_rate)
        self.frame_rate = self.condition_encoder.config.input_sampling_rate / self.condition_encoder.config.input_hop_length
        self.latent_hop_length = int(self.condition_encoder.config.output_hop_length)
        self.model = self.transformer
        self.semantic_acceleration = MiniMaxMusic3SemanticAcceleration(self.text_encoder, self.rvq_depth_decoder, lm_decoder_engine, self.device) if accelerated else None
        self._install_abort_checks()
        print(f"[MiniMax Music 3] LM Engine='{lm_decoder_engine}'")

    def _install_abort_checks(self):
        self.transformer._interrupt_check = self._abort_requested
        self.rvq_depth_decoder._interrupt_check = self._abort_requested
        self.condition_encoder._interrupt_check = self._abort_requested
        self.vocoder._interrupt_check = self._abort_requested

        def abort_after_layer(_module, _inputs, _output):
            if self._abort_requested():
                raise InterruptedError("MiniMax Music 3 generation interrupted")

        self._language_abort_handles = [layer.register_forward_hook(abort_after_layer) for layer in self.text_encoder.model.layers]

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def request_early_stop(self):
        self._early_stop = True

    def _tokenize(self, caption: str, lyrics: str) -> torch.Tensor:
        text = (
            f"{_IM_START}{_CAPTION_START}{_clean_caption(caption)}{_CAPTION_END}"
            f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
        )
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
            raise ValueError(f"The assembled MiniMax Music 3 prompt has {input_ids.shape[1]} tokens; the maximum is {_MAX_PROMPT_TOKENS}")
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
        return torch.cat((input_ids, unconditional_ids), dim=0).to(self.device)

    def _embed_audio_frame(self, frame_codes: torch.Tensor) -> torch.Tensor:
        num_codebooks = self.rvq_depth_decoder.config.num_codebooks
        embeds = self.text_encoder.model.embed_tokens(frame_codes[:, :1] + _AUDIO_CODE_OFFSET)
        offsets = (torch.arange(num_codebooks - 1, device=frame_codes.device) * self.rvq_depth_decoder.config.audio_vocab_size).unsqueeze(0)
        extra = self.rvq_depth_decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        return (embeds + extra.to(embeds.dtype)) * num_codebooks**-0.5

    def _generate_depth_codes(self, last_hidden: torch.Tensor, semantic_code: torch.Tensor, generator: torch.Generator):
        num_codebooks = self.rvq_depth_decoder.config.num_codebooks
        if self.semantic_acceleration is not None:
            self.rvq_depth_decoder.decode_embedding(last_hidden, 0)
            code_embed = self.text_encoder.model.embed_tokens(semantic_code + _AUDIO_CODE_OFFSET)
            hidden = self.rvq_depth_decoder.decode_embedding(code_embed, 1)
            codes, hidden_parts = [semantic_code], []
            for index in range(1, num_codebooks):
                hidden_parts.append(hidden[:1])
                logits = self.rvq_depth_decoder.audio_heads[index - 1](hidden)
                conditional, unconditional = logits[:1].float(), logits[1:2].float()
                code = _sample_top_k(unconditional + (conditional - unconditional) * _AR_CFG_SCALE, generator, no_sync=True).repeat(2)
                codes.append(code)
                if index < num_codebooks - 1:
                    embed = self.rvq_depth_decoder.audio_embeddings(code + (index - 1) * self.rvq_depth_decoder.config.audio_vocab_size)
                    hidden = self.rvq_depth_decoder.decode_embedding(embed, index + 1)
                if self._abort_requested():
                    raise InterruptedError("MiniMax Music 3 generation interrupted")
            return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)
        sequence = [self.rvq_depth_decoder.projection(last_hidden).unsqueeze(1)]
        code_embed = self.text_encoder.model.embed_tokens(semantic_code + _AUDIO_CODE_OFFSET)
        sequence.append(self.rvq_depth_decoder.projection(code_embed).unsqueeze(1))
        codes, hidden_parts = [semantic_code], []
        for index in range(1, num_codebooks):
            hidden = self.rvq_depth_decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = self.rvq_depth_decoder.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            code = _sample_top_k(unconditional + (conditional - unconditional) * _AR_CFG_SCALE, generator).repeat(2)
            codes.append(code)
            if index < num_codebooks - 1:
                embed = self.rvq_depth_decoder.audio_embeddings(code + (index - 1) * self.rvq_depth_decoder.config.audio_vocab_size)
                sequence.append(self.rvq_depth_decoder.projection(embed).unsqueeze(1))
            if self._abort_requested():
                raise InterruptedError("MiniMax Music 3 generation interrupted")
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def _generate_semantics(self, text_ids: torch.Tensor, duration: float, generator: torch.Generator, callback=None) -> torch.Tensor:
        max_frames = min(int(duration * self.frame_rate), _MAX_AUDIO_FRAMES)
        if max_frames < 1:
            raise ValueError(f"Duration must be at least {1 / self.frame_rate:.2f} seconds")
        if self.semantic_acceleration is None:
            text_embeds = self.text_encoder.model.embed_tokens(text_ids)
            output = self.text_encoder.model(inputs_embeds=text_embeds, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]
            logits = self.text_encoder.lm_head(last_hidden).float()
        else:
            last_hidden, logits = self.semantic_acceleration.prefill(text_ids, max_frames)
        accelerated = self.semantic_acceleration is not None
        if not accelerated:
            vocab_mask = torch.ones(self.text_encoder.config.vocab_size, dtype=torch.bool, device=self.device)
            vocab_mask[_AUDIO_CODE_OFFSET:_AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
            vocab_mask[_AUDIO_END_TOKEN_ID] = False
        frame_hiddens = []
        progress_seconds = 0
        try:
            with tqdm(total=max_frames, desc="MiniMax Music 3 semantic frames", unit="frame") as progress:
                for frame_index in range(max_frames + 1):
                    if accelerated:
                        logits = logits.float()
                    else:
                        logits = logits.float().masked_fill(vocab_mask, -float("inf"))
                    conditional, unconditional = logits[0:1], logits[1:2]
                    guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
                    threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
                    guided.masked_fill_(conditional < threshold, -float("inf"))
                    if not accelerated:
                        guided.masked_fill_(vocab_mask.unsqueeze(0), -float("inf"))
                    sampled = _sample_top_k(guided, generator, no_sync=accelerated)
                    if int(sampled.item()) == (_SEMANTIC_VOCAB_SIZE if accelerated else _AUDIO_END_TOKEN_ID):
                        break
                    semantic_code = sampled if accelerated else sampled - _AUDIO_CODE_OFFSET
                    frame_codes, depth_hidden = self._generate_depth_codes(last_hidden, semantic_code.repeat(2), generator)
                    if frame_index > 0:
                        frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1).to("cpu"))
                        progress.update(1)
                        elapsed_seconds = int(len(frame_hiddens) / self.frame_rate)
                        if callback is not None and elapsed_seconds > progress_seconds:
                            callback(step_idx=elapsed_seconds - 1, override_num_inference_steps=max(1, int(duration)), denoising_extra=f"Semantic audio: {elapsed_seconds}s/{int(duration)}s", progress_unit="seconds")
                            progress_seconds = elapsed_seconds
                        if len(frame_hiddens) >= max_frames or self._early_stop:
                            break
                    feedback = self._embed_audio_frame(frame_codes)
                    if self.semantic_acceleration is None:
                        output = self.text_encoder.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
                        past_key_values = output.past_key_values
                        last_hidden = output.last_hidden_state[:, -1]
                        logits = self.text_encoder.lm_head(last_hidden)
                    else:
                        last_hidden, logits = self.semantic_acceleration.decode(feedback)
                    if self._abort_requested():
                        raise InterruptedError("MiniMax Music 3 generation interrupted")
        finally:
            if self.semantic_acceleration is None:
                del past_key_values, output, text_embeds
        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately")
        if callback is not None:
            callback(step_idx=-1, force_refresh=True)
        frame_hiddens = torch.stack(frame_hiddens, dim=1)
        if self.semantic_acceleration is not None and not bool(torch.isfinite(frame_hiddens).all()):
            raise RuntimeError(f"MiniMax Music 3 {self.lm_decoder_engine} semantic decoder produced non-finite frame states")
        return frame_hiddens

    def _denoise(self, frame_hiddens: torch.Tensor, num_inference_steps: int, guidance_scale: float, generator: torch.Generator, callback=None):
        num_frames = frame_hiddens.shape[1]
        chunk_starts = [0] if num_frames <= _CHUNK_FRAMES else list(range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))
        previous_latent = previous_condition = None
        latent_chunks = []
        total_steps = len(chunk_starts) * num_inference_steps
        global_step = 0
        with tqdm(total=total_steps, desc="MiniMax Music 3 flow matching", unit="step") as progress:
            for chunk_index, chunk_start in enumerate(chunk_starts):
                chunk_end = min(chunk_start + _CHUNK_FRAMES, num_frames)
                condition_input = frame_hiddens[:, chunk_start:chunk_end].to(self.device, dtype=self.condition_encoder.dtype)
                condition = self.condition_encoder(condition_input).to(self.transformer.dtype)
                del condition_input
                overlap = 0
                if previous_latent is not None:
                    overlap = min(previous_latent.shape[-1], condition.shape[1])
                    condition[:, :overlap] = previous_condition[:, :overlap]
                latents = torch.randn((1, self.transformer.config.in_channels, condition.shape[1]), generator=generator, device=self.device, dtype=condition.dtype)
                noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None
                sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
                self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
                for step_index, timestep in enumerate(self.scheduler.timesteps):
                    if overlap > 0:
                        time_value = timestep.to(latents.dtype)
                        latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time_value) * noise_prompt + time_value * previous_latent[..., :overlap]
                    model_latents = latents.expand(2, -1, -1)
                    model_condition = torch.cat((condition, torch.zeros_like(condition)), dim=0)
                    model_timestep = timestep.expand(2).to(latents.dtype)
                    predictions = self.transformer(model_latents, model_timestep, model_condition, return_dict=False)[0]
                    conditional, unconditional = predictions[:1], predictions[1:2]
                    velocity = unconditional + guidance_scale * (conditional - unconditional)
                    latents = self.scheduler.step(velocity, timestep, latents, return_dict=False)[0]
                    global_step += 1
                    progress.update(1)
                    if callback is not None:
                        callback(step_idx=global_step - 1, override_num_inference_steps=total_steps, denoising_extra=f"Flow chunk {chunk_index + 1}/{len(chunk_starts)}, step {step_index + 1}/{num_inference_steps}")
                    if self._abort_requested():
                        raise InterruptedError("MiniMax Music 3 generation interrupted")
                if overlap > 0:
                    latents[..., :overlap] = previous_latent[..., :overlap]
                overlap_start = max(0, latents.shape[-1] - 2 * _OVERLAP_LATENT_LENGTH)
                overlap_end = max(overlap_start, latents.shape[-1] - _OVERLAP_LATENT_LENGTH)
                previous_latent = latents[..., overlap_start:overlap_end]
                previous_condition = condition[:, overlap_start:overlap_end]
                latent_chunks.append(latents)
        return latent_chunks

    def _decode(self, latent_chunks: list[torch.Tensor], callback=None) -> torch.Tensor:
        waveform_chunks = []
        with tqdm(total=len(latent_chunks), desc="MiniMax Music 3 vocoder", unit="chunk") as progress:
            for chunk_index, latents in enumerate(latent_chunks):
                waveform = self.vocoder(latents.to(self.vocoder.dtype))
                left = 0 if chunk_index == 0 else _CROP_LEFT_LATENT * self.latent_hop_length
                right = 0 if chunk_index == len(latent_chunks) - 1 else _CROP_RIGHT_LATENT * self.latent_hop_length
                waveform_chunks.append(waveform[..., left:waveform.shape[-1] - right].to("cpu"))
                progress.update(1)
                if callback is not None:
                    callback(step_idx=chunk_index, override_num_inference_steps=len(latent_chunks), denoising_extra=f"Vocoder chunk {chunk_index + 1}/{len(latent_chunks)}")
                if self._abort_requested():
                    raise InterruptedError("MiniMax Music 3 generation interrupted")
        return torch.cat(waveform_chunks, dim=-1).float().clamp_(-1.0, 1.0)

    @torch.inference_mode()
    def generate(
        self,
        input_prompt: str,
        *,
        alt_prompt: str,
        duration_seconds: float,
        sampling_steps: int,
        guide_scale: float,
        seed: int,
        callback=None,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        try:
            text_ids = self._tokenize(str(alt_prompt), str(input_prompt))
            frame_hiddens = self._generate_semantics(text_ids, float(duration_seconds), generator, callback)
            latent_chunks = self._denoise(frame_hiddens, int(sampling_steps), float(guide_scale), generator, callback)
            waveform = self._decode(latent_chunks, callback)
        except InterruptedError:
            return None
        return {"x": waveform, "audio_sampling_rate": self.sample_rate}

    def release(self):
        for handle in getattr(self, "_language_abort_handles", []):
            handle.remove()
        if self.semantic_acceleration is not None:
            self.semantic_acceleration.release()
        self.transformer = self.text_encoder = self.rvq_depth_decoder = None
        self.condition_encoder = self.vocoder = self.model = None
