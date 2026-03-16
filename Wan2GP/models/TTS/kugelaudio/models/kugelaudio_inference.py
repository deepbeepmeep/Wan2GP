"""KugelAudio inference model for speech generation.

This is the open-source inference implementation without optimizations.
Based on the original VibeVoice model architecture.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.utils import logging

from ..configs import KugelAudioConfig
from ..schedule.dpm_solver import DPMSolverMultistepScheduler
from .cudagraph_hooks import KugelAudioCudaGraphHooks
from .diffusion_head import KugelAudioDiffusionHead
from .kugelaudio_model import KugelAudioModel, KugelAudioPreTrainedModel
from .tokenizer import (
    KugelAudioTokenizerEncoderOutput,
    KugelAudioTokenizerStreamingCache,
)

logger = logging.get_logger(__name__)


def _get_cache_tensors(cache) -> Tuple[List, List]:
    """Get key and value cache tensors from a cache object."""
    if hasattr(cache, "layers"):
        key_cache = []
        value_cache = []
        for layer in cache.layers:
            key_cache.append(layer.keys)
            value_cache.append(layer.values)
        if key_cache and value_cache:
            return key_cache, value_cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache, cache.value_cache
    # transformers>=4.50 uses DynamicCache
    if isinstance(cache, DynamicCache):
        key_cache = []
        value_cache = []
        try:
            seq_len = len(cache)
        except TypeError:
            seq_len = 0
        for idx in range(seq_len):
            k, v = cache[idx]
            key_cache.append(k)
            value_cache.append(v)
        if key_cache and value_cache:
            return key_cache, value_cache
    raise AttributeError(f"Cannot get cache tensors from {type(cache).__name__}")


@dataclass
class KugelAudioCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class KugelAudioGenerationOutput(ModelOutput):
    """Output type for KugelAudio generation."""

    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


class KugelAudioTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.valid_token_ids] = 0
        scores = scores + mask
        return scores


class KugelAudioForConditionalGenerationInference(KugelAudioPreTrainedModel, GenerationMixin):
    """KugelAudio model for inference with speech generation capabilities."""

    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = KugelAudioModel(config)
        self._cuda_graph_hooks = KugelAudioCudaGraphHooks(self)
        self.lm_head = nn.Linear(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            bias=False,
        )
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = (
            num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps
        )

    def set_lm_decoder_engine(self, lm_decoder_engine: str | None) -> None:
        self._cuda_graph_hooks.set_lm_decoder_engine(lm_decoder_engine)

    def prepare_decode_cuda_graph(self, *, max_batch_size: int, max_cache_tokens: int) -> None:
        self._cuda_graph_hooks.prepare(max_batch_size=max_batch_size, max_cache_tokens=max_cache_tokens)

    def release_decode_cuda_graph(self) -> None:
        self._cuda_graph_hooks.release()

    def _process_speech_inputs(
        self,
        speech_tensors: Optional[torch.Tensor],
        speech_masks: Optional[torch.Tensor],
        voice_cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process speech inputs through acoustic and semantic encoders.

        Returns:
            Tuple of (acoustic_features, speech_embeds) where speech_embeds has shape
            [num_valid_frames, hidden] - already indexed by speech_masks for direct
            assignment to inputs_embeds[speech_input_mask].
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if voice_cache is not None:
            # Use pre-encoded voice features
            acoustic_mean = voice_cache["acoustic_mean"].to(device=device, dtype=dtype)
            semantic_mean = voice_cache["semantic_mean"].to(device=device, dtype=dtype)

            # Sample from acoustic distribution
            fix_std = voice_cache.get("acoustic_std", self.acoustic_tokenizer.fix_std)
            acoustic_features = acoustic_mean + fix_std * torch.randn_like(acoustic_mean)
            semantic_features = semantic_mean

            # Create speech_masks from cache dimensions (all frames valid)
            batch_size = acoustic_features.shape[0]
            seq_len = acoustic_features.shape[1]
            speech_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        elif speech_tensors is not None:
            # Encode speech through tokenizers
            with torch.no_grad():
                # Acoustic encoding
                if speech_tensors.dim() == 2:
                    speech_tensors = speech_tensors.unsqueeze(1)

                acoustic_output = self.acoustic_tokenizer.encode(speech_tensors)
                acoustic_features, _ = self.acoustic_tokenizer.sampling(acoustic_output)

                # Semantic encoding
                semantic_output = self.semantic_tokenizer.encode(speech_tensors)
                semantic_features = semantic_output.mean

            # Create speech_masks if not provided (all frames valid)
            if speech_masks is None:
                batch_size = acoustic_features.shape[0]
                seq_len = acoustic_features.shape[1]
                speech_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            # Return dummy features
            vae_dim = self.config.acoustic_vae_dim
            acoustic_features = torch.zeros(1, 1, vae_dim, device=device, dtype=dtype)
            semantic_features = torch.zeros(
                1, 1, self.config.semantic_vae_dim, device=device, dtype=dtype
            )
            speech_masks = torch.ones(1, 1, dtype=torch.bool, device=device)

        # Ensure acoustic and semantic have matching time dimensions
        acoustic_len = acoustic_features.shape[1]
        semantic_len = semantic_features.shape[1]
        if semantic_len < acoustic_len:
            pad_size = acoustic_len - semantic_len
            semantic_features = torch.nn.functional.pad(
                semantic_features, (0, 0, 0, pad_size), mode="constant", value=0
            )
        elif semantic_len > acoustic_len:
            semantic_features = semantic_features[:, :acoustic_len, :]

        # Apply scaling to acoustic features
        if not torch.isnan(self.speech_scaling_factor):
            acoustic_features = (
                acoustic_features + self.speech_bias_factor
            ) * self.speech_scaling_factor

        # Get embeddings through connectors
        acoustic_embed = self.acoustic_connector(acoustic_features)
        semantic_embed = self.semantic_connector(semantic_features)

        # Combine embeddings and index by speech_masks
        combined_embed = acoustic_embed + semantic_embed

        speech_embeds = combined_embed[speech_masks.to(device=combined_embed.device, dtype=torch.bool)]

        return acoustic_features, speech_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        voice_cache: Optional[dict] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, KugelAudioCausalLMOutputWithPast]:
        """Forward pass for the model."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Process speech inputs if provided
        if voice_cache is not None or (speech_tensors is not None and speech_masks is not None):
            _, speech_embeds = self._process_speech_inputs(
                speech_tensors.to(self.dtype) if speech_tensors is not None else None,
                speech_masks,
                voice_cache=voice_cache,
            )
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return KugelAudioCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def sample_speech_tokens(
        self,
        condition: torch.Tensor,
        neg_condition: torch.Tensor,
        cfg_scale: float = 3.0,
        abort_check=None,
    ) -> Optional[torch.Tensor]:
        """Sample speech latents using diffusion with classifier-free guidance."""
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)

        if cfg_scale == 1.0:
            # No CFG - single forward pass
            speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
            for t in self.model.noise_scheduler.timesteps:
                if abort_check is not None and abort_check():
                    return None
                eps = self.model.prediction_head(
                    speech, t.repeat(speech.shape[0]).to(speech), condition=condition
                )
                speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
            return speech

        # With CFG - run guidance on one latent batch (N), not duplicated scheduler states (2N).
        combined_condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        batch = int(condition.shape[0])
        speech = torch.randn(batch, self.config.acoustic_vae_dim).to(combined_condition)

        for t in self.model.noise_scheduler.timesteps:
            if abort_check is not None and abort_check():
                return None
            combined = torch.cat([speech, speech], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=combined_condition)
            cond_eps, uncond_eps = torch.split(eps, batch, dim=0)
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            speech = self.model.noise_scheduler.step(guided_eps, t, speech).prev_sample

        return speech

    @torch.no_grad()
    def encode_voice_prompt(
        self,
        voice_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> dict:
        """Pre-encode a voice prompt for caching."""
        device = getattr(self, "_input_device", None)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if voice_audio.dim() == 1:
            voice_audio = voice_audio.unsqueeze(0).unsqueeze(0)
        elif voice_audio.dim() == 2:
            voice_audio = voice_audio.unsqueeze(1)

        voice_audio = voice_audio.to(device=device, dtype=dtype)

        with torch.no_grad():
            acoustic_output = self.model.acoustic_tokenizer.encode(voice_audio)
            semantic_output = self.model.semantic_tokenizer.encode(voice_audio)

        return {
            "acoustic_mean": acoustic_output.mean.cpu(),
            "acoustic_std": getattr(acoustic_output, "std", self.model.acoustic_tokenizer.fix_std),
            "semantic_mean": semantic_output.mean.cpu(),
            "audio_length": voice_audio.shape[-1],
            "sample_rate": sample_rate,
        }

    @torch.no_grad()
    def generate(
        self,
        text_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        voice_prompt: Optional[torch.Tensor] = None,
        voice_cache: Optional[dict] = None,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        speech_input_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 3.0,
        max_new_tokens: int = 2048,
        tail_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        show_progress: bool = True,
        abort_check=None,
        early_stop_check=None,
        callback=None,
        progress_interval: int = 10,
        release_decode_graph_on_exit: bool = True,
        **kwargs,
    ) -> KugelAudioGenerationOutput:
        """Generate speech from text.

        Args:
            text_ids: Tokenized text input (from processor)
            input_ids: Alternative name for text_ids
            voice_prompt: Voice audio tensor for cloning (legacy, use speech_tensors instead)
            voice_cache: Pre-encoded voice features (from encode_voice_prompt)
            speech_tensors: Voice audio tensor from processor for cloning
            speech_masks: Mask indicating valid voice frames
            speech_input_mask: Boolean mask indicating where to insert voice embeddings
            cfg_scale: Classifier-free guidance scale (higher = more faithful to text)
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            show_progress: Whether to show progress bar

        Returns:
            KugelAudioGenerationOutput with sequences and speech_outputs
        """
        device = getattr(self, "_input_device", None)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Handle input_ids vs text_ids
        if text_ids is None and input_ids is not None:
            text_ids = input_ids
        if text_ids is None:
            raise ValueError("text_ids or input_ids is required")

        text_ids = text_ids.to(device)
        batch_size = text_ids.shape[0]

        # Handle legacy voice_prompt parameter
        if voice_prompt is not None and speech_tensors is None:
            speech_tensors = voice_prompt
            # Create default speech_masks if not provided
            if speech_masks is None:
                # Estimate number of frames from audio length
                audio_len = voice_prompt.shape[-1]
                num_frames = (audio_len + 3199) // 3200  # compression ratio
                speech_masks = torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)

        # Get special token IDs
        speech_start_id = getattr(self.config, "speech_start_id", None) or 151652
        speech_end_id = getattr(self.config, "speech_end_id", None) or 151653
        speech_diffusion_id = getattr(self.config, "speech_diffusion_id", None) or 151654
        eos_token_id = getattr(self.config.decoder_config, "eos_token_id", None) or 151643

        # Initialize streaming caches for tokenizers
        acoustic_cache = KugelAudioTokenizerStreamingCache()
        semantic_cache = KugelAudioTokenizerStreamingCache()

        # Initialize sequences and attention masks
        input_embedding_layer = self.model.get_input_embeddings()
        prompt_ids = text_ids
        generated_tokens: list[torch.Tensor] = []
        positive_seq_len = int(prompt_ids.shape[1])
        attention_mask = torch.ones_like(prompt_ids)

        # For CFG, create negative prompt (just speech_start token)
        max_cache_tokens = int(prompt_ids.shape[1]) + max(1, int(max_new_tokens))
        embed_dim = int(input_embedding_layer.weight.shape[1])
        negative_ids = torch.full((batch_size, max_cache_tokens), speech_start_id, dtype=torch.long, device=device)
        negative_attention_mask = torch.zeros((batch_size, max_cache_tokens), dtype=attention_mask.dtype, device=device)
        negative_attention_mask[:, 0] = 1
        negative_inputs_embeds = torch.empty((batch_size, max_cache_tokens, embed_dim), device=device, dtype=dtype)
        negative_inputs_embeds[:, :1, :] = input_embedding_layer(negative_ids[:, :1])
        negative_seq_len = 1

        # Storage for generated audio and tracking
        audio_chunks = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        tail_tokens = max(0, int(tail_tokens or 0))
        tail_remaining = torch.zeros(batch_size, dtype=torch.long, device=device)
        tail_active = torch.zeros(batch_size, dtype=torch.bool, device=device)
        tail_exhausted = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Get initial embeddings
        inputs_embeds = input_embedding_layer(prompt_ids)

        # Process voice/speech input if provided
        if speech_tensors is not None or voice_cache is not None:
            # Get speech embeddings
            if voice_cache is not None:
                _, speech_embeds = self._process_speech_inputs(
                    speech_tensors=None,
                    speech_masks=None,
                    voice_cache=voice_cache,
                )
            else:
                # Encode speech_tensors directly
                speech_tensors = speech_tensors.to(device=device, dtype=dtype)
                if speech_masks is not None:
                    speech_masks = speech_masks.to(device)
                _, speech_embeds = self._process_speech_inputs(
                    speech_tensors=speech_tensors,
                    speech_masks=speech_masks,
                    voice_cache=None,
                )

            # Insert speech embeddings at positions marked by speech_input_mask
            # speech_embeds is already flattened to [num_valid_frames, hidden] by _process_speech_inputs
            if speech_input_mask is not None:
                speech_input_mask = speech_input_mask.to(device)
                if speech_input_mask.shape[1] != inputs_embeds.shape[1]:
                    # Align mask length to input length to avoid shape mismatches.
                    if speech_input_mask.shape[1] > inputs_embeds.shape[1]:
                        speech_input_mask = speech_input_mask[:, : inputs_embeds.shape[1]]
                    else:
                        pad = inputs_embeds.shape[1] - speech_input_mask.shape[1]
                        speech_input_mask = torch.nn.functional.pad(
                            speech_input_mask, (0, pad), value=False
                        )
                if speech_input_mask.sum().item() != speech_embeds.shape[0]:
                    # Best-effort alignment if counts mismatch.
                    target = int(speech_input_mask.sum().item())
                    if speech_embeds.shape[0] > target:
                        speech_embeds = speech_embeds[:target]
                    elif speech_embeds.shape[0] < target and speech_embeds.numel() > 0:
                        pad_rows = target - speech_embeds.shape[0]
                        speech_embeds = torch.cat(
                            [speech_embeds, speech_embeds[-1:].repeat(pad_rows, 1)],
                            dim=0,
                        )
                inputs_embeds[speech_input_mask] = speech_embeds

        # Setup logits processor to constrain to valid tokens
        valid_tokens = [speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id]
        token_constraint = KugelAudioTokenConstraintProcessor(valid_tokens, device=device)

        # Initialize KV caches
        past_key_values = None
        negative_past_key_values = None
        self.prepare_decode_cuda_graph(max_batch_size=int(batch_size), max_cache_tokens=max_cache_tokens)
        graph_active = self._cuda_graph_hooks.is_active()
        use_fast_kvcache = False
        static_past_key_values = None
        static_decode_cache_position = None
        lm_attn_backend = str(
            getattr(getattr(self.model.language_model, "config", None), "_attn_implementation", None)
            or getattr(getattr(self.model.language_model, "config", None), "attn_implementation", None)
            or ""
        ).strip().lower()
        if not graph_active and device.type == "cuda" and lm_attn_backend != "flash_attention_2":
            static_past_key_values = StaticCache(
                config=self.config.decoder_config,
                max_batch_size=int(batch_size),
                max_cache_len=int(max_cache_tokens),
                device=device,
                dtype=dtype,
            )
            static_decode_cache_position = torch.empty(1, device=device, dtype=torch.long)
            use_fast_kvcache = True
        if self._cuda_graph_hooks.is_enabled():
            prepare_status = "reused" if self._cuda_graph_hooks.last_prepare_reused() else "new"
            state = "active" if graph_active else "inactive"
            if state == "inactive" or prepare_status == "new":
                print(f"[KugelAudio][cg] decode graph {state} ({prepare_status})")
        if use_fast_kvcache:
            print("[KugelAudio][kv] fast static cache active (legacy)")

        # Progress bar
        progress_iter = (
            tqdm(range(max_new_tokens), desc="KugelAudio", leave=False, mininterval=0.33)
            if show_progress
            else range(max_new_tokens)
        )

        total_steps = max(1, int(max_new_tokens))
        progress_interval = max(1, int(progress_interval or 1))
        progress_refresh_interval_s = 0.33
        last_progress_callback_ts = 0.0
        eos_token_tensor = torch.tensor(eos_token_id, device=device)
        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=total_steps,
                denoising_extra=f"0/{total_steps} tokens",
                progress_unit="tokens",
            )
            last_progress_callback_ts = time.monotonic()

        for step in progress_iter:
            if abort_check is not None and abort_check():
                if release_decode_graph_on_exit:
                    self.release_decode_cuda_graph()
                return None
            if early_stop_check is not None and early_stop_check():
                break
            if finished.all():
                break

            # Forward pass for positive (main) model
            if step == 0:
                outputs = (
                    self._cuda_graph_hooks.prefill(inputs_embeds)
                    if graph_active
                    else self(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask if not use_fast_kvcache else None,
                        past_key_values=static_past_key_values,
                        use_cache=True,
                        return_dict=True,
                        cache_position=torch.arange(positive_seq_len, device=device, dtype=torch.long)
                        if use_fast_kvcache
                        else None,
                    )
                )
                if not graph_active and not use_fast_kvcache:
                    past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                positive_last_hidden = outputs.last_hidden_state
            else:
                if graph_active:
                    decode_position = int(positive_seq_len - 1)
                    if not self._cuda_graph_hooks.has_capacity(decode_position):
                        # Static cache replay cannot compact KV; stop when max duration/token budget is reached.
                        break
                    step_logits, positive_last_hidden = self._cuda_graph_hooks.run_decode(
                        inputs_embeds[:, -1:], decode_position
                    )
                    logits = step_logits[:, -1, :]
                else:
                    outputs = self(
                        inputs_embeds=inputs_embeds[:, -1:],
                        attention_mask=attention_mask if not use_fast_kvcache else None,
                        past_key_values=past_key_values if not use_fast_kvcache else static_past_key_values,
                        use_cache=True,
                        return_dict=True,
                        cache_position=(
                            static_decode_cache_position.fill_(positive_seq_len - 1)
                            if use_fast_kvcache
                            else None
                        ),
                    )
                    if not use_fast_kvcache:
                        past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    positive_last_hidden = outputs.last_hidden_state

            # Apply token constraint
            logits = token_constraint(prompt_ids, logits)


            # If tail mode is active, prevent EOS/speech_end for those samples.
            if tail_tokens > 0 and tail_active.any():
                mask_idx = tail_active.nonzero(as_tuple=False).squeeze(-1)
                if mask_idx.numel() > 0:
                    logits[mask_idx, eos_token_id] = float("-inf")
                    logits[mask_idx, speech_end_id] = float("-inf")

            # Sample or greedy decode
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            # If EOS/speech_end is produced and tail mode is enabled, resample without them
            # so the LM never sees EOS in its past context.
            if tail_tokens > 0:
                stop_mask = ((next_tokens == eos_token_id) | (next_tokens == speech_end_id)) & ~finished
                if stop_mask.any():
                    start_tail = stop_mask & ~tail_active & ~tail_exhausted
                    if start_tail.any():
                        tail_active = tail_active | start_tail
                        tail_remaining[start_tail] = tail_tokens
                    mask_idx = stop_mask.nonzero(as_tuple=False).squeeze(-1)
                    if mask_idx.numel() > 0:
                        masked_logits = logits.clone()
                        masked_logits[mask_idx, eos_token_id] = float("-inf")
                        masked_logits[mask_idx, speech_end_id] = float("-inf")
                        if do_sample and temperature > 0:
                            masked_probs = torch.softmax(masked_logits / temperature, dim=-1)
                            resampled = torch.multinomial(masked_probs[mask_idx], num_samples=1).squeeze(-1)
                        else:
                            resampled = torch.argmax(masked_logits[mask_idx], dim=-1)
                        next_tokens[mask_idx] = resampled

            # Force finished samples to output EOS
            force_eos_mask = finished
            next_tokens = torch.where(force_eos_mask, eos_token_tensor, next_tokens)

            # Update sequences
            generated_tokens.append(next_tokens.unsqueeze(-1))
            positive_seq_len += 1
            if not graph_active and not use_fast_kvcache:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype),
                    ],
                    dim=-1,
                )

            # Check for EOS/speech_end tokens (optionally extend with tail tokens)
            eos_mask = (next_tokens == eos_token_id) & ~finished
            speech_end_mask = (next_tokens == speech_end_id) & ~finished
            stop_mask = eos_mask | speech_end_mask
            if stop_mask.any():
                if tail_tokens > 0:
                    start_tail = stop_mask & ~tail_active
                    if start_tail.any():
                        tail_active = tail_active | start_tail
                        tail_remaining[start_tail] = tail_tokens
                    stop_mask = stop_mask & ~tail_active

                if stop_mask.any():
                    finished = finished | stop_mask
                    speech_end_indices = speech_end_mask.nonzero(as_tuple=False).squeeze(-1)
                    if speech_end_indices.numel() > 0:
                        acoustic_cache.set_to_zero(speech_end_indices)
                        semantic_cache.set_to_zero(speech_end_indices)

            # Decrement tail counters once per step for active samples.
            if tail_tokens > 0 and tail_active.any():
                active_idx = tail_active.nonzero(as_tuple=False).squeeze(-1)
                if active_idx.numel() > 0:
                    tail_remaining[active_idx] = torch.clamp(
                        tail_remaining[active_idx] - 1, min=0
                    )
                    done_tail = tail_remaining == 0
                    if done_tail.any():
                        tail_exhausted = tail_exhausted | done_tail
                        finished = finished | done_tail
                    tail_active = tail_active & ~done_tail

            # Handle speech_start tokens - refresh negative model KV cache
            speech_start_mask = (next_tokens == speech_start_id) & ~finished
            if (
                speech_start_mask.any()
                and cfg_scale != 1.0
                and negative_past_key_values is not None
            ):
                speech_start_indices = speech_start_mask.nonzero(as_tuple=False).squeeze(-1)
                if speech_start_indices.dim() == 0:
                    speech_start_indices = speech_start_indices.unsqueeze(0)
                key_caches, value_caches = _get_cache_tensors(negative_past_key_values)

                for sample_idx in speech_start_indices.tolist():
                    negative_attention_mask[sample_idx, :negative_seq_len] = 0
                    negative_attention_mask[sample_idx, negative_seq_len - 1] = 1

                    for k_cache, v_cache in zip(key_caches, value_caches):
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()

                    negative_ids[sample_idx, negative_seq_len - 1] = speech_start_id

            # Prepare next input embeddings
            next_inputs_embeds = input_embedding_layer(next_tokens).unsqueeze(1)

            # Handle diffusion tokens - generate speech
            diffusion_mask = (next_tokens == speech_diffusion_id) & ~finished
            if diffusion_mask.any():
                diffusion_indices = diffusion_mask.nonzero(as_tuple=False).squeeze(-1)
                if diffusion_indices.dim() == 0:
                    diffusion_indices = diffusion_indices.unsqueeze(0)

                # Run negative forward pass for CFG
                if cfg_scale != 1.0:
                    if negative_past_key_values is None:
                        neg_outputs = self(
                            inputs_embeds=negative_inputs_embeds[:, :negative_seq_len],
                            attention_mask=negative_attention_mask[:, :negative_seq_len],
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        neg_outputs = self(
                            inputs_embeds=negative_inputs_embeds[:, negative_seq_len - 1 : negative_seq_len],
                            attention_mask=negative_attention_mask[:, :negative_seq_len],
                            past_key_values=negative_past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                    negative_past_key_values = neg_outputs.past_key_values

                    # Handle non-diffusion samples KV cache correction
                    non_diffusion_mask = ~diffusion_mask & ~finished
                    if non_diffusion_mask.any():
                        non_diffusion_indices = non_diffusion_mask.nonzero(as_tuple=False).squeeze(
                            -1
                        )
                        if non_diffusion_indices.dim() == 0:
                            non_diffusion_indices = non_diffusion_indices.unsqueeze(0)

                        key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                        for sample_idx in non_diffusion_indices.tolist():
                            start_idx = correct_cnt[sample_idx].item()
                            seq_len = int(negative_seq_len)

                            if start_idx + 1 < seq_len - 1:
                                negative_attention_mask[sample_idx, start_idx + 1 : seq_len] = (
                                    negative_attention_mask[sample_idx, start_idx : seq_len - 1].clone()
                                )
                            negative_attention_mask[sample_idx, start_idx] = 0

                            for k_cache, v_cache in zip(key_caches, value_caches):
                                if start_idx + 1 < seq_len - 1:
                                    k_cache[sample_idx, :, start_idx + 1 : seq_len, :] = k_cache[
                                        sample_idx, :, start_idx : seq_len - 1, :
                                    ].clone()
                                    v_cache[sample_idx, :, start_idx + 1 : seq_len, :] = v_cache[
                                        sample_idx, :, start_idx : seq_len - 1, :
                                    ].clone()

                            if start_idx + 1 < seq_len - 1:
                                negative_ids[sample_idx, start_idx + 1 : seq_len] = negative_ids[
                                    sample_idx, start_idx : seq_len - 1
                                ].clone()

                        correct_cnt[non_diffusion_indices] += 1

                    neg_condition = neg_outputs.last_hidden_state[diffusion_indices, -1, :]
                else:
                    neg_condition = torch.zeros(
                        diffusion_indices.shape[0],
                        self.config.decoder_config.hidden_size,
                        device=device,
                        dtype=dtype,
                    )

                # Get conditioning from last hidden state
                condition = positive_last_hidden[diffusion_indices, -1, :]

                # Sample speech latents using diffusion
                speech_latents = self.sample_speech_tokens(
                    condition, neg_condition, cfg_scale, abort_check=abort_check
                )
                if speech_latents is None:
                    if release_decode_graph_on_exit:
                        self.release_decode_cuda_graph()
                    return None

                # Unscale latents
                scaled_latent = (
                    speech_latents / self.speech_scaling_factor - self.speech_bias_factor
                )

                # Decode through acoustic tokenizer with streaming cache
                audio = self.acoustic_tokenizer.decode(
                    scaled_latent.unsqueeze(1).permute(0, 2, 1),
                    cache=acoustic_cache,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                )

                # Store audio chunks
                for i, idx in enumerate(diffusion_indices.tolist()):
                    if not finished[idx]:
                        audio_chunks[idx].append(audio[i].detach().clone())

                # Keep both acoustic + semantic conditioning for diffusion tokens.
                semantic_output = self.semantic_tokenizer.encode(
                    audio,
                    cache=semantic_cache,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                )
                semantic_features = semantic_output.mean
                acoustic_embed = self.acoustic_connector(speech_latents.unsqueeze(1))
                semantic_embed = self.semantic_connector(semantic_features)

                # Update embeddings for diffusion samples
                next_inputs_embeds[diffusion_indices] = acoustic_embed + semantic_embed

            # Positive decode only consumes the latest token after prefill.
            inputs_embeds = next_inputs_embeds

            # Update negative model
            if cfg_scale != 1.0:
                if negative_seq_len >= max_cache_tokens:
                    break
                negative_inputs_embeds[:, negative_seq_len : negative_seq_len + 1, :] = next_inputs_embeds
                negative_attention_mask[:, negative_seq_len] = 1
                negative_ids[:, negative_seq_len] = next_tokens
                negative_seq_len += 1

            if callback is not None:
                now = time.monotonic()
                should_refresh = (
                    step + 1 == total_steps
                    or (step % progress_interval == 0 and (now - last_progress_callback_ts) >= progress_refresh_interval_s)
                )
                if should_refresh:
                    callback(
                        step_idx=step,
                        override_num_inference_steps=total_steps,
                        denoising_extra=f"{step + 1}/{total_steps} tokens",
                        progress_unit="tokens",
                    )
                    last_progress_callback_ts = now

        if abort_check is not None and abort_check():
            if release_decode_graph_on_exit:
                self.release_decode_cuda_graph()
            return None

        # Concatenate audio chunks with normalization
        speech_outputs = []
        for chunks in audio_chunks:
            if chunks:
                concatenated = torch.cat(chunks, dim=-1).squeeze()
                # Normalize audio to prevent clipping
                max_val = concatenated.abs().max()
                if max_val > 1.0:
                    concatenated = concatenated * (0.95 / max_val)
                # Apply watermark to all generated audio
                # concatenated = self._apply_watermark(concatenated, sample_rate=24000)
                speech_outputs.append(concatenated.cpu())
            else:
                speech_outputs.append(None)

        if generated_tokens:
            sequences = torch.cat([prompt_ids] + generated_tokens, dim=-1)
        else:
            sequences = prompt_ids
        output = KugelAudioGenerationOutput(sequences=sequences, speech_outputs=speech_outputs)
        if release_decode_graph_on_exit:
            self.release_decode_cuda_graph()
        return output

    def _apply_watermark(self, audio: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """Apply imperceptible watermark to generated audio.

        This watermark identifies audio as generated by KugelAudio and is designed
        to be robust against various audio transformations while remaining inaudible.
        """
        try:
            import torchaudio.functional as F
            from audioseal import AudioSeal
        except ImportError:
            return audio  # Graceful fallback if audioseal not available

        device = audio.device
        dtype = audio.dtype
        original_shape = audio.shape

        # Prepare audio for watermarking (AudioSeal expects [batch, channels, samples] at 16kHz)
        if audio.dim() == 1:
            audio_for_wm = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio_for_wm = audio.unsqueeze(0)
        else:
            audio_for_wm = audio

        audio_for_wm = audio_for_wm.float()

        # Resample to 16kHz for AudioSeal
        if sample_rate != 16000:
            audio_16k = F.resample(audio_for_wm, sample_rate, 16000)
        else:
            audio_16k = audio_for_wm

        # Load watermark generator (cached after first use)
        if not hasattr(self, "_wm_generator"):
            self._wm_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
            self._wm_generator.eval()

        # Generate and apply watermark
        with torch.no_grad():
            watermark_16k = self._wm_generator.get_watermark(audio_16k.to(device), 16000)

        # Resample watermark back to original sample rate
        if sample_rate != 16000:
            watermark = F.resample(watermark_16k, 16000, sample_rate)
            # Ensure same length
            if watermark.shape[-1] != audio_for_wm.shape[-1]:
                if watermark.shape[-1] > audio_for_wm.shape[-1]:
                    watermark = watermark[..., : audio_for_wm.shape[-1]]
                else:
                    watermark = torch.nn.functional.pad(
                        watermark, (0, audio_for_wm.shape[-1] - watermark.shape[-1])
                    )
        else:
            watermark = watermark_16k

        # Add watermark to audio
        watermarked = audio_for_wm + watermark.to(audio_for_wm.device)

        # Normalize to prevent clipping
        max_val = watermarked.abs().max()
        if max_val > 1.0:
            watermarked = watermarked * (0.95 / max_val)

        # Restore original shape
        if len(original_shape) == 1:
            watermarked = watermarked.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            watermarked = watermarked.squeeze(0)

        return watermarked.to(dtype=dtype)


# Register with AutoModel
AutoModel.register(KugelAudioConfig, KugelAudioModel)
AutoModelForCausalLM.register(KugelAudioConfig, KugelAudioForConditionalGenerationInference)


__all__ = [
    "KugelAudioForConditionalGenerationInference",
    "KugelAudioCausalLMOutputWithPast",
    "KugelAudioGenerationOutput",
]
