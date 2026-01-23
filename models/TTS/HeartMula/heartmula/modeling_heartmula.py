import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from .configuration_heartmula import HeartMuLaConfig
from .llama_blocks import Llama3ScaledRoPE, TransformerDecoder, build_llama_decoder


def llama3_2_3B() -> TransformerDecoder:
    return build_llama_decoder(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_300M() -> TransformerDecoder:
    return build_llama_decoder(
        vocab_size=128_256,
        num_layers=3,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_7B() -> TransformerDecoder:
    return build_llama_decoder(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_400M() -> TransformerDecoder:
    return build_llama_decoder(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-3B": llama3_2_3B,
    "llama-300M": llama3_2_300M,
    "llama-7B": llama3_2_7B,
    "llama-400M": llama3_2_400M,
}


def _prepare_transformer(model):
    if hasattr(model.tok_embeddings, "embedding_dim"):
        embed_dim = model.tok_embeddings.embedding_dim
    else:
        embed_dim = model.num_heads * model.head_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


class HeartMuLa(PreTrainedModel):
    config_class = HeartMuLaConfig

    def __init__(self, config: HeartMuLaConfig):
        super().__init__(config)

        self.config = config
        self._interrupt_check = None

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)
        self.post_init()

    def setup_caches(self, max_batch_size: int):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        try:
            self.reset_caches()
        except RuntimeError:
            pass

        self.backbone.setup_caches(max_batch_size, dtype)
        self.decoder[0].setup_caches(
            max_batch_size,
            dtype,
            decoder_max_seq_len=self.config.audio_num_codebooks,
        )

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
        )

    def move_causal_masks(self, device: torch.device) -> None:
        non_blocking = device.type == "cuda"
        if hasattr(self, "backbone_causal_mask"):
            self.backbone_causal_mask = self.backbone_causal_mask.to(
                device, non_blocking=non_blocking
            )
        if hasattr(self, "decoder_causal_mask"):
            self.decoder_causal_mask = self.decoder_causal_mask.to(
                device, non_blocking=non_blocking
            )
        for module in self.modules():
            if isinstance(module, Llama3ScaledRoPE):
                module.ensure_device(device)

    def prepare_flash(self, device: torch.device, dtype: torch.dtype) -> None:
        for module in self.modules():
            if hasattr(module, "configure_flash"):
                module.configure_flash(device, dtype)

    def _check_interrupt(self) -> bool:
        if self._interrupt_check is None:
            return False
        return bool(self._interrupt_check())

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        cfg_scale: float,
        continuous_segments: torch.Tensor = None,
        starts=None,
    ) -> torch.Tensor:
        if self._check_interrupt():
            return None
        b, _, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = None

        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_b = b // 2
            uncond_mask = torch.cat(
                [
                    torch.zeros(actual_b, dtype=torch.bool, device=tokens.device),
                    torch.ones(actual_b, dtype=torch.bool, device=tokens.device),
                ]
            )

        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2, dtype=embeds.dtype)
        if continuous_segments is not None:
            continuous_segments = self.muq_linear(continuous_segments)
            if uncond_mask is not None:
                uncond_embed = self.unconditional_text_embedding(
                    torch.zeros(1, device=h.device, dtype=torch.long)
                )
                mask_expanded = uncond_mask.view(b, 1)
                mask_expanded = mask_expanded.expand_as(continuous_segments)
                continuous_segments = torch.where(
                    mask_expanded, uncond_embed, continuous_segments
                )
            batch_indices = torch.arange(h.shape[0], device=h.device)
            h[batch_indices, starts] = continuous_segments
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        if self._check_interrupt():
            return None
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)

        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_b = b // 2
            cond_logits = c0_logits[:actual_b, :]
            uncond_logits = c0_logits[actual_b:, :]
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            c0_sample = sample_topk(guided_logits, topk, temperature)
            c0_sample = c0_sample.repeat(2, 1)
        else:
            c0_sample = sample_topk(c0_logits, topk, temperature)

        c0_embed = self._embed_audio(0, c0_sample)

        self.decoder[0].reset_caches()
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample_list = [c0_sample.clone()]
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )
        curr_h = curr_h.to(embeds.dtype)
        for i in range(1, self.config.audio_num_codebooks):
            if self._check_interrupt():
                return None
            curr_decoder_mask = None
            decoder_h = self.decoder[0](
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            )
            if self._check_interrupt():
                return None
            head_weight = self.audio_head[i - 1]
            ci_logits = torch.mm(decoder_h[:, -1, :].to(head_weight.dtype), head_weight)
            if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
                actual_b = b // 2
                cond_ci = ci_logits[:actual_b, :]
                uncond_ci = ci_logits[actual_b:, :]
                guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale

                ci_sample = sample_topk(guided_ci, topk, temperature)
                ci_sample = ci_sample.repeat(2, 1)
            else:
                ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample_list.append(ci_sample)
            curr_pos = curr_pos[:, -1:] + 1

        curr_sample = torch.cat(curr_sample_list, dim=1)

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder[0].reset_caches()

    def release_caches(self):
        # Drop KV cache tensors to free memory between generations.
        for module in self.backbone.modules():
            if hasattr(module, "kv_cache"):
                module.kv_cache = None
                if hasattr(module, "cache_enabled"):
                    module.cache_enabled = False
                if hasattr(module, "reset_flash"):
                    module.reset_flash()
        for module in self.decoder[0].modules():
            if hasattr(module, "kv_cache"):
                module.kv_cache = None
                if hasattr(module, "cache_enabled"):
                    module.cache_enabled = False
                if hasattr(module, "reset_flash"):
                    module.reset_flash()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(
        self, tokens: torch.Tensor, uncond_mask: torch.Tensor | None
    ) -> torch.Tensor:
        b, _, _ = tokens.size()
        text_tokens = tokens[:, :, -1]
        text_embeds = self.text_embeddings(text_tokens)

        if uncond_mask is not None:
            uncond_text_embed = self.unconditional_text_embedding(
                torch.zeros(1, device=text_embeds.device, dtype=torch.long)
            )
            mask_expanded = uncond_mask.view(b, 1, 1)
            mask_expanded = mask_expanded.expand_as(text_embeds)
            text_embeds = torch.where(mask_expanded, uncond_text_embed, text_embeds)

        text_embeds = text_embeds.unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1]
        audio_tokens = audio_tokens + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=audio_tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)
