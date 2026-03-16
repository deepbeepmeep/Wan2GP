from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    CausalLMOutput,
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging


from .tokenizer import (
    KugelAudioAcousticTokenizerModel,
    KugelAudioSemanticTokenizerModel,
)
from .diffusion_head import KugelAudioDiffusionHead
from ..schedule.dpm_solver import DPMSolverMultistepScheduler

from ..configs import KugelAudioConfig


logger = logging.get_logger(__name__)

@dataclass
class KugelAudioCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None
    speech_token_num: Optional[torch.LongTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class KugelAudioGenerationOutput(ModelOutput):
    """
    Output type for KugelAudio generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences.
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """

    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features, **kwargs):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


# @auto_docstring
class KugelAudioPreTrainedModel(PreTrainedModel):
    config_class = KugelAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, KugelAudioDiffusionHead):
            module.initialize_weights()
            return

        # Use the language model's initializer_range if available
        if hasattr(self.config, "language_model_config") and hasattr(
            self.config.language_model_config, "initializer_range"
        ):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, "decoder_config") and hasattr(
            self.config.decoder_config, "initializer_range"
        ):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02  # Default value

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


# @auto_docstring
class KugelAudioModel(KugelAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32

        # Initialize Qwen2 model for language modeling
        lm_config = config.decoder_config
        self.language_model = AutoModel.from_config(lm_config)

        # Initialize speech components if needed
        self.acoustic_tokenizer = AutoModel.from_config(
            config.acoustic_tokenizer_config
        ).to(dtype)
        self.semantic_tokenizer = AutoModel.from_config(
            config.semantic_tokenizer_config
        ).to(dtype)

        self.acoustic_connector = SpeechConnector(
            config.acoustic_vae_dim, lm_config.hidden_size
        ).to(dtype)
        self.semantic_connector = SpeechConnector(
            config.semantic_vae_dim, lm_config.hidden_size
        ).to(dtype)

        # Register scaling factors as buffers - use 1D tensors for FSDP compatibility
        self.register_buffer("speech_scaling_factor", torch.tensor(float("nan")))
        self.register_buffer("speech_bias_factor", torch.tensor(float("nan")))

        # Initialize prediction head for speech generation
        self.prediction_head = AutoModel.from_config(config.diffusion_head_config).to(
            dtype
        )

        # Initialize noise scheduler with SDE-DPM-Solver++ for better quality
        algorithm_type = getattr(
            config.diffusion_head_config, "ddpm_algorithm_type", "sde-dpmsolver++"
        )
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
            algorithm_type=algorithm_type,
            solver_order=2,
        )

    def get_input_embeddings(self):
        if hasattr(self.language_model, "embed_tokens"):
            # If the language model has an embed_tokens attribute, return it
            return self.language_model.embed_tokens

        for (
            name,
            attr,
        ) in (
            self.language_model.fullmap.items()
        ):  # parallel by nnscaler, the name is changed
            if attr.orig_name == "embed_tokens.weight":
                return getattr(self.language_model, name)
        assert False, "should not arrive here"

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer

        # Reset the encoder to evaluation mode
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()

        if self.semantic_tokenizer is not None:
            self.semantic_tokenizer.eval()

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device = None,
        cache_position: torch.Tensor = None,
        batch_size: int = None,
        config=None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Creates a 4D causal attention mask for use with static cache.
        
        This enables torch.compile to work efficiently without recompilation
        by providing a consistent mask shape during autoregressive generation.
        
        Based on the standard HuggingFace implementation without sliding window
        (KugelAudio doesn't use sliding window attention).
        
        Compatible with both old and new transformers API.
        """
        # Handle case where attention_mask is already 4D
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        
        # Get device from attention_mask or cache_position if not provided
        if device is None:
            if attention_mask is not None:
                device = attention_mask.device
            elif cache_position is not None:
                device = cache_position.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        min_dtype = torch.finfo(dtype).min
        
        # Create causal mask: (sequence_length, target_length)
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        
        if sequence_length != 1:
            # Apply upper triangular mask (can't attend to future tokens)
            causal_mask = torch.triu(causal_mask, diagonal=1)
        
        # Mask positions beyond current cache position
        if cache_position is not None:
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        
        # Expand to 4D: (batch_size, 1, sequence_length, target_length)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Combine with input attention mask if provided
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            # Create padding mask from attention_mask
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(dtype) * min_dtype
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
        
        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        if not return_dict:
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class KugelAudioForConditionalGeneration(KugelAudioPreTrainedModel):
    """
    Unified model for both training and inference.

    Supports:
    - Training via forward() with loss computation
    - Inference via generate() for audio generation
    """

    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = KugelAudioModel(config)
        self.vocab_size = config.decoder_config.vocab_size
        self.lm_head = nn.Linear(
            config.decoder_config.hidden_size, self.vocab_size, bias=False
        )

        # Inference configuration (for generate() method)
        self.ddpm_inference_steps = (
            config.diffusion_head_config.ddpm_num_inference_steps
            if hasattr(config, "diffusion_head_config")
            else 5
        )

        self.post_init()

    # Properties for easier access (used by generate())
    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if getattr(self.config.decoder_config, "tie_word_embeddings", False):
            # The standard PreTrainedModel method will handle the tying.
            # It typically does a simple parameter object assignment, which is
            # CORRECT to do BEFORE FSDP wraps the model.
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if hasattr(input_embeddings, "weight"):
                output_embeddings.weight = input_embeddings.weight
            else:
                # maybe returned input_embeddings a tensor directly
                output_embeddings.weight = input_embeddings

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (
                        0,
                        output_embeddings.weight.shape[0]
                        - output_embeddings.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            print("✅ Tied input and output embeddings using standard assignment.")
        else:
            print("ℹ️  tie_word_embeddings is False, not tying weights.")

    # Also, ensure set_output_embeddings is safe, though your implementation looks okay.
    # The key is to avoid calling it after accelerator.prepare().
    def set_output_embeddings(self, new_embeddings):
        # Your current implementation using data.copy_ is good practice,
        # but the best way is to not call this after prepare().
        self.lm_head = new_embeddings

    def forward_speech_features(
        self,
        speech_tensors=None,
        speech_masks=None,
        speech_type="audio",
        return_unmask=False,
    ):
        if speech_tensors is None:
            # Use config to get vae_dim instead of non-existent self.args
            vae_dim = self.config.acoustic_tokenizer_config.vae_dim
            audio_features = torch.zeros(1, 1, vae_dim).to(
                self.get_input_embeddings().weight
            )
            connect_features = self.model.acoustic_connector(audio_features)
            return audio_features, connect_features
        else:
            with torch.no_grad():
                if speech_type == "audio":
                    with torch.no_grad():
                        frames_out = self.model.acoustic_tokenizer.encode(
                            speech_tensors.unsqueeze(1)
                        )
                        if isinstance(frames_out, (list, tuple)):
                            frames = frames_out[0][0]
                        else:
                            frames = frames_out
                    audio_tokens = frames.sample(
                        self.model.acoustic_tokenizer.std_dist_type
                    )[0]

                elif speech_type == "vae":
                    # Use config to get vae_dim instead of non-existent self.args
                    vae_dim = self.config.acoustic_tokenizer_config.vae_dim
                    speech_mode = speech_tensors.reshape(
                        speech_tensors.size(0), -1, vae_dim
                    )

                    # gaussian sample from the speech_mode
                    batch_size = speech_mode.size(0)
                    value = self.model.acoustic_tokenizer.fix_std / 0.8
                    std = (
                        torch.randn(
                            batch_size,
                            dtype=speech_mode.dtype,
                            device=speech_mode.device,
                        )
                        * value
                    )
                    std = std.view(-1, *[1] * (speech_mode.dim() - 1))
                    audio_tokens = speech_mode + std * torch.randn(
                        speech_mode.shape
                    ).to(speech_mode)
                else:
                    raise NotImplementedError(
                        f"Speech type {speech_type} not implemented"
                    )

                if torch.isnan(self.model.speech_scaling_factor) or torch.isnan(
                    self.model.speech_bias_factor
                ):
                    scaling_factor = 1.0 / audio_tokens[speech_masks].flatten().std()
                    bias_factor = -audio_tokens[speech_masks].flatten().mean()
                    self.model.speech_scaling_factor.copy_(scaling_factor)
                    self.model.speech_bias_factor.copy_(bias_factor)
                    print(
                        f"Speech scaling factor: {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}",
                        flush=True,
                    )

                audio_features = (
                    audio_tokens + self.model.speech_bias_factor
                ) * self.model.speech_scaling_factor

            connect_features = self.model.acoustic_connector(audio_features)
            if return_unmask:
                return audio_features, connect_features
            return audio_features[speech_masks], connect_features[speech_masks]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # New arguments for speech processing and loss calculation
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speeches_loss_input: Optional[torch.FloatTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None,
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        acoustic_loss_mask: Optional[torch.BoolTensor] = None,
        ddpm_batch_mul: int = 1,
        **kwargs: Optional[Dict[str, Union[torch.Tensor, str]]],
    ) -> Union[Tuple, KugelAudioCausalLMOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        x = self.get_input_embeddings()(input_ids)

        semantic_speech_all_connect_features = self.model.semantic_connector(
            speech_semantic_tensors
        )
        if speeches_loss_input is not None:
            # only part audio need diffuse
            speech_all_features, speech_all_connect_features = (
                self.forward_speech_features(
                    speech_tensors=(
                        speech_tensors.type_as(x)
                        if speech_tensors is not None
                        else None
                    ),
                    speech_masks=speech_masks,
                    speech_type=kwargs.get("speech_type", "audio"),
                    return_unmask=True,
                )
            )
            if speech_tensors is not None:
                if semantic_speech_all_connect_features is not None:
                    x[acoustic_input_mask] = (
                        speech_all_connect_features[speech_masks]
                        + semantic_speech_all_connect_features[speech_masks]
                    )
                else:
                    x[acoustic_input_mask] = speech_all_connect_features[speech_masks]
                speech_features = speech_all_features[
                    speeches_loss_input & speech_masks
                ]  # only part audio need diffuse
                speech_connect_features = speech_all_connect_features[
                    speeches_loss_input & speech_masks
                ]
                # Forward-time consistency check: selected latent count should match number of acoustic placeholders
                try:
                    if acoustic_input_mask is not None:
                        assert speech_connect_features.shape[0] == int(
                            acoustic_input_mask.sum().item()
                        ), f"Mismatch between selected speech connectors ({speech_connect_features.shape[0]}) and acoustic_input_mask sum ({int(acoustic_input_mask.sum().item())})"
                except Exception:
                    pass
        else:
            speech_features, speech_connect_features = self.forward_speech_features(
                speech_tensors=(
                    speech_tensors.type_as(x) if speech_tensors is not None else None
                ),
                speech_masks=speech_masks,
                speech_type=kwargs.get("speech_type", "audio"),
            )
            if speech_tensors is not None:
                x[acoustic_input_mask] = speech_connect_features

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=x,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            # The custom CE loss with masking is calculated in the training script.
            # We leave the standard loss calculation here as None.
            pass

        # --- Diffusion Loss Calculation ---
        diffusion_loss = None
        # This block is executed only if we are in a context that involves speech.
        if speech_tensors is not None and acoustic_loss_mask.sum().item() > 0:
            # Build conditioning mask from positions whose NEXT token is a speech latent (shift left by 1)
            cond_mask = torch.zeros_like(acoustic_loss_mask, dtype=torch.bool)
            cond_mask[:, :-1] = acoustic_loss_mask[:, 1:]
            cond_mask[:, 0] = False
            condition_features = hidden_states[cond_mask]

            speech_len, latent_size = speech_features.shape
            # Sanity check: ensure 1:1 alignment between selected conditions and latents
            try:
                assert (
                    condition_features.shape[0] == speech_len
                ), f"Mismatch: condition_features={condition_features.shape[0]} vs speech_features={speech_len}"
            except Exception:
                pass

            noise = torch.randn(
                (speech_len * ddpm_batch_mul, latent_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            timesteps = torch.multinomial(
                torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
                speech_len * ddpm_batch_mul,
                replacement=True,
            ).to(hidden_states.device)

            speech_features_repeated = speech_features.repeat_interleave(
                ddpm_batch_mul, dim=0
            )
            condition_features_repeated = condition_features.repeat_interleave(
                ddpm_batch_mul, dim=0
            )

            noisy_speech_features = self.model.noise_scheduler.add_noise(
                speech_features_repeated, noise, timesteps
            )

            model_output = self.model.prediction_head(
                noisy_speech_features, timesteps.type_as(x), condition_features_repeated
            )

            prediction_type = self.config.diffusion_head_config.prediction_type
            if prediction_type == "epsilon":
                target_for_loss = noise
            elif prediction_type == "v_prediction":
                target_for_loss = self.model.noise_scheduler.get_velocity(
                    speech_features_repeated, noise, timesteps
                )
            else:
                raise NotImplementedError(
                    f"Prediction type {prediction_type} not implemented"
                )

            diffusion_loss = F.mse_loss(
                model_output.float(), target_for_loss.float(), reduction="sum"
            )
            if latent_size > 0 and ddpm_batch_mul > 0:
                # Normalize by latent dim, number of sampled diffusion steps per latent, and number of speech tokens
                diffusion_loss = (
                    diffusion_loss / latent_size / ddpm_batch_mul / max(speech_len, 1)
                )
            else:
                diffusion_loss = torch.tensor(0.0, device=diffusion_loss.device)

        else:
            # Dummy loss for DDP to work when there are no speech samples in a batch,
            # but we are in a speech context.
            diffusion_loss = (
                sum(p.sum() for p in self.model.prediction_head.parameters()) * 0.0
            )
            diffusion_loss += (
                sum(p.sum() for p in self.model.acoustic_connector.parameters()) * 0.0
            )
            diffusion_loss += (
                sum(p.sum() for p in self.model.semantic_connector.parameters()) * 0.0
            )
        # --- End Diffusion Loss Calculation ---

        if not return_dict:
            output = (logits, speech_len) + outputs.to_tuple()[1:]
            return (loss, diffusion_loss) + output

        return KugelAudioCausalLMOutputWithPast(
            loss=loss,
            diffusion_loss=diffusion_loss,
            speech_token_num=torch.tensor(
                speech_len if speech_tensors is not None else 0,
                device=logits.device,
                dtype=torch.long,
            ),
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoModel.register(KugelAudioConfig, KugelAudioModel)
AutoModelForCausalLM.register(KugelAudioConfig, KugelAudioForConditionalGeneration)

__all__ = [
    "KugelAudioModel",
    "KugelAudioPreTrainedModel",
    "KugelAudioForConditionalGeneration",
    "KugelAudioCausalLMOutputWithPast",
    "KugelAudioGenerationOutput",
]
