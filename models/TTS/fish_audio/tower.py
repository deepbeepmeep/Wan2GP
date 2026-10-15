# Adapted from fishaudio/fish-speech, revision 1ff3c3ed2f80d3106a83909dd25b83129c3d816f.
# Copyright Fish Audio contributors. See LICENSE_FISH_AUDIO for the upstream license.
import math
from mmgp import cuda_graphs, runtime
from shared import attention_kit
from shared.llm_engines.nanovllm.layerwise import cached_attention
from .backbone import GenerationCancelled
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.types import Device

from .config import (
    AudioDecoderConfig,
    FishOmniConfig,
)
from .layers import ZeroCenteredRMSNorm

# Repetition-aware sampling of the main token, fixed by the training recipe: when
# the first draw repeats one of the last RAS_WINDOW_FRAMES main tokens, a second
# draw with these parameters is used instead.
RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5
TOWER_CHECKPOINT_PREFIXES = {
    "audio_decoder.codebook_embeddings.": "codebook_embeddings.",
    "audio_decoder_mtp_shared.layers.": "shared_layers.",
    "audio_decoder_mtp.layers.": "residual_layers.",
    "audio_decoder_mtp.heads.": "codebook_heads.",
}


def sample_top_k_top_p(
    logits: Tensor,
    temperature: float | Tensor,
    top_k: int | Tensor,
    top_p: float | Tensor,
    *,
    check_finite: bool = True,
) -> Tensor:
    """One id per row; drops the token crossing top_p, except the highest-ranked one.

    Sampling parameters may be 0-dim device tensors (CUDA graphs); check_finite
    syncs with the host.
    """
    if isinstance(temperature, Tensor):
        temperature = temperature.clamp(min=MIN_TEMPERATURE)
    else:
        temperature = max(temperature, MIN_TEMPERATURE)
    if check_finite and not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
    values, ids = torch.sort(logits.float() / temperature, descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_()
    choice = (probabilities / noise).argmax(-1, keepdim=True)
    return ids.gather(-1, choice).squeeze(-1)


class TowerAttention(nn.Module):
    """Gated attention over time; checkpoint wqkv rows are [Q | Gate | K | V]."""

    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.config = config
        q_width = config.n_head * config.head_dim
        kv_width = config.n_local_heads * config.head_dim
        linear = dict(bias=False, dtype=dtype, device=device)
        norm = dict(dtype=dtype, device=device)
        self.wq = nn.Linear(config.dim, q_width, **linear)
        self.wg = nn.Linear(config.dim, q_width, **linear)
        self.wk = nn.Linear(config.dim, kv_width, **linear)
        self.wv = nn.Linear(config.dim, kv_width, **linear)
        self.wo = nn.Linear(q_width, config.dim, **linear)
        # Always on, whatever audio_decoder_config.attention_qk_norm says: training runs
        # q/k norm in the audio decoder, but the exporter never writes that field, so
        # released Drama 3 configs carry its default false. Keep the (1 + weight) scale too:
        # exported weights are zero-centered, not pre-shifted by one.
        self.q_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps, **norm)
        self.k_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps, **norm)
        self.k_cache: Tensor | None = None
        self.v_cache: Tensor | None = None
        # CUDA-graph decode attends this many cache rows; the mask hides the rest.
        self.static_kv_len: int | None = None

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Interleaved (non-neox) partial RoPE, computed in fp32."""
        width = self.config.rotary_dim
        exponents = torch.arange(0, width, 2, device=x.device, dtype=torch.float32)
        angles = positions.float()[:, None] * self.config.rope_base ** (
            -exponents / width
        )
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        even, odd = x[..., :width:2].float(), x[..., 1:width:2].float()
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        return torch.cat((rotated.flatten(-2).to(x.dtype), x[..., width:]), dim=-1)

    def forward(self, x: Tensor, positions: Tensor, cache, kv_len: int) -> Tensor:
        cfg = self.config
        gate = self.wg(x)
        def normalize(query, key, group):
            if query is not None:
                attention_kit.rows_(query, lambda part, first, last: self.rotate(self.q_norm(part[0]), positions[first:last]).unsqueeze(0))
            if key is not None:
                attention_kit.rows_(key, lambda part, first, last: self.rotate(self.k_norm(part[0]), positions[first:last]).unsqueeze(0))
        out = attention_kit.qkv_attention([x], self.wq, self.wk, self.wv, cfg.n_head, cfg.head_dim, normalize,
            kv_heads=cfg.n_local_heads,
            attention_fn=lambda qkv, group: cached_attention(qkv, group, cache, positions, kv_len, self.engine))
        return self.wo(out.reshape(x.shape[0], -1) * gate.sigmoid())


class TowerMLP(nn.Module):
    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        linear = dict(bias=False, dtype=dtype, device=device)
        self.w1 = nn.Linear(config.dim, config.intermediate_size, **linear)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, **linear)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, **linear)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(nn.Module):
    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        args = dict(dtype=dtype, device=device)
        self.attention = TowerAttention(config, **args)
        self.attention_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.ffn_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.feed_forward = TowerMLP(config, **args)

    def forward(self, x: Tensor, positions: Tensor, cache, kv_len: int) -> Tensor:
        dtype = x.dtype
        residual = x.float() if dtype == torch.float16 else x
        residual = residual + self.attention(
            self.attention_norm(residual).to(dtype), positions, cache, kv_len
        )
        return residual + self.feed_forward(self.ffn_norm(residual).to(dtype))


class TowerLayer(nn.Module):
    """Mix a code embedding into the hidden state, then one pre-norm block."""

    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        shared: bool,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.shared = shared
        args = dict(dtype=dtype, device=device)
        # Checkpoint name of the embedding differs between shared / residual layers.
        name = "semantic_embedding" if shared else "codebook_embedding"
        self.add_module(name, nn.Embedding(config.vocab_size, config.dim, **args))
        self.enorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.hnorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False, **args)
        self.transformer_layer = TowerTransformerLayer(config, **args)
        self.final_layernorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)

    def forward(
        self,
        hidden: Tensor,
        codes: Tensor,
        semantic: Tensor,
        positions: Tensor,
        cache,
        kv_len: int,
        shared_input: Tensor | None = None,
    ) -> Tensor:
        embedding = self.semantic_embedding if self.shared else self.codebook_embedding
        codes = codes.clamp(0, embedding.num_embeddings - 1)
        e = torch.where(semantic[:, None], embedding(codes), 0.0)
        if shared_input is not None:
            e = e + shared_input
        mixed = self.eh_proj(torch.cat((self.enorm(e), self.hnorm(hidden)), dim=-1))
        out = self.final_layernorm(self.transformer_layer(mixed, positions, cache, kv_len))
        return out.to(mixed.dtype)


class FishOmniAudioDecoder(nn.Module):
    """Codebook tower: predicts codes 1..N-1 of each frame, attending over time.

    A frame row is [main token id, code 0, ..., code N-1]; code 0 is the main
    token's offset in the semantic band.
    """

    def __init__(
        self,
        config: FishOmniConfig,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: Device = None,
    ) -> None:
        super().__init__()
        cfg = self.config = config.audio_decoder
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id
        self.logit_start = config.audio_logit_rows[0]  # vocabulary id of the backbone's first logit
        args = dict(dtype=dtype, device=device)
        self.codebook_embeddings = nn.Embedding(
            cfg.vocab_size * cfg.num_codebooks, cfg.dim, **args
        )
        self.shared_layers = nn.ModuleList(
            TowerLayer(cfg, shared=True, **args)
            for _ in range(cfg.shared_enrichment_layers)
        )
        self.residual_layers = nn.ModuleList(
            TowerLayer(cfg, shared=False, **args) for _ in range(cfg.num_codebooks - 1)
        )
        self.codebook_heads = nn.ModuleList(
            nn.Linear(cfg.dim, cfg.vocab_size, bias=False, **args)
            for _ in range(cfg.num_codebooks - 1)
        )
        # Ids the main token is drawn from (eos + the semantic band) and their logit
        # rows, built lazily on the tower's device: creating them inside CUDA-graph
        # capture is illegal.
        self.main_candidates: Tensor | None = None
        self.candidate_rows: Tensor | None = None
        self.cache = []
        self.kv = None
        self.cache_capacity = 0
        self._convertWeightsFloatTo = None
        self._model_dtype = dtype
        self._cuda_graph_towers = ("shared_layers",)
        self.abort_requested = lambda: False

    @property
    def device(self) -> torch.device:
        return runtime.execution_device(self)

    def setup_caches(self, max_seq_len: int) -> None:
        """The key/value caches of all layers in one stacked tensor (one graph argument); self.cache holds each layer's views."""
        layers = len(self.shared_layers) + len(self.residual_layers)
        self.kv = torch.zeros(2, layers, 1, max_seq_len, self.config.n_local_heads, self.config.head_dim, dtype=self._model_dtype, device=self.device)
        self.cache = self.layer_caches(self.kv)
        self.cache_capacity = max_seq_len

    @staticmethod
    def layer_caches(kv):
        return [(kv[0, index], kv[1, index]) for index in range(kv.shape[1])]

    def release_cache(self):
        self.cache.clear()
        self.kv = None
        self.cache_capacity = 0
        self.main_candidates = self.candidate_rows = None

    def kv_length(self, end: int, rows: int) -> int:
        """Attended cache rows for `rows` positions ending at end - 1 (host ints: no device sync)."""
        if end > self.cache_capacity:
            raise ValueError("Codebook cache capacity exceeded")
        return min(self.cache_capacity, math.ceil(end / 256) * 256) if rows == 1 else end

    def check_abort(self):
        if self.abort_requested():
            raise GenerationCancelled()

    def is_semantic(self, tokens: Tensor) -> Tensor:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: Tensor, first: int, end: int) -> Tensor:
        """Summed embeddings of codes first..end-1 per frame."""
        runtime.ensure_loaded(self)
        if first == end:
            return torch.zeros((len(frames), self.config.dim), dtype=self._model_dtype, device=self.device)
        offsets = torch.arange(first, end, device=self.device) * self.config.vocab_size
        rows = frames[:, 1 + first : 1 + end] + offsets
        return F.embedding_bag(rows, self.codebook_embeddings.weight, mode="sum")

    def embed_backbone(self, frames: Tensor, text_embeddings: Tensor) -> Tensor:
        """Backbone input: semantic frames add codes 1..K to the token embedding."""
        frames = frames.to(self.device)
        count = self.config.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count).to(text_embeddings.device)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0]).to(text_embeddings.device)
        return torch.where(semantic[:, None], mixed, text_embeddings)

    def run_shared(
        self,
        hidden: Tensor,
        c0: Tensor,
        semantic: Tensor,
        positions: Tensor,
        frames: Tensor,
        kv_len: int,
        caches,
    ) -> Tensor:
        # The first shared layer also sees the codes the backbone did not.
        unseen = self.embedding_sum(
            frames, 1 + self.config.vq_backbone_codebooks, self.config.num_codebooks
        )
        unseen = torch.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(self.shared_layers):
            self.check_abort()
            hidden = layer(hidden, c0, semantic, positions, caches[index], kv_len, unseen if index == 0 else None)
        return hidden

    def prefill(
        self, hidden: Tensor, positions: Tensor, frames: Tensor, next_frames: Tensor, end: int
    ) -> None:
        """Teacher-force the tower cache; row p is labelled with frame p + 1. end: last position + 1."""
        semantic = self.is_semantic(next_frames[:, 0])
        kv_len = self.kv_length(end, len(positions))
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, positions, frames, kv_len, self.cache)
        for index, layer in enumerate(self.residual_layers):
            self.check_abort()
            hidden = layer(hidden, next_frames[:, index + 1], semantic, positions, self.cache[len(self.shared_layers) + index], kv_len)

    def decode_step(
        self,
        logits: Tensor,
        hidden: Tensor,
        positions: Tensor,
        frames: Tensor,
        history: Tensor,
        *,
        end: int,
        temperature: float | Tensor,
        top_k: int | Tensor,
        top_p: float | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample the frame after each row; the main token is drawn over eos + semantics.

        logits start at vocabulary id logit_start; end is the last position + 1. Returns
        the frames, a 0-dim device flag (False when any sampled logits were non-finite:
        the caller checks it with the frame, in one host sync per frame), and for the
        backbone's next step the summed embeddings of codes 1..K and the semantic flags
        (FishOmniBackbone.mix_embeddings). While the tower's weights are resident, the
        whole step (layers, heads, sampling) is one graph.
        """
        if self.main_candidates is None:  # released with the caches
            eos = torch.tensor([self.eos_id], device=self.device)
            band = torch.arange(
                self.semantic_begin, self.semantic_end + 1, device=self.device
            )
            self.main_candidates = torch.cat((eos, band)).sort().values
            self.candidate_rows = self.main_candidates - self.logit_start
        kv_len = self.kv_length(end, len(positions))
        return cuda_graphs.run(
            self, logits, hidden, positions, frames, history[-RAS_WINDOW_FRAMES:], self.kv,
            self.main_candidates, self.candidate_rows, kv_len, temperature, top_k, top_p,
            method="sample_frame", enabled=runtime.weights_resident(self),
        )

    def sample_frame(
        self,
        logits: Tensor,
        hidden: Tensor,
        positions: Tensor,
        frames: Tensor,
        history: Tensor,
        kv: Tensor,
        candidates: Tensor,
        candidate_rows: Tensor,
        kv_len: int,
        temperature: float | Tensor,
        top_k: int | Tensor,
        top_p: float | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        caches = self.layer_caches(kv)
        candidate_logits = logits[:, candidate_rows]
        finite = torch.isfinite(candidate_logits).all()
        first = candidates[
            sample_top_k_top_p(
                candidate_logits, temperature, top_k, top_p, check_finite=False
            )
        ]
        retry = candidates[
            sample_top_k_top_p(
                candidate_logits,
                RAS_TEMPERATURE,
                RAS_TOP_K,
                RAS_TOP_P,
                check_finite=False,
            )
        ]
        # Only audio tokens are resampled, as in S2: the window starts with prompt rows,
        # which hold the user turn's eos, and a stop must not be redrawn.
        repeated = (history == first[:, None]).any(-1)
        repeated &= self.is_semantic(first)
        main = torch.where(repeated, retry, first)
        c0 = (main - self.semantic_begin).clamp(0, self.config.vocab_size - 1)
        semantic = self.is_semantic(main)
        hidden = self.run_shared(hidden, c0, semantic, positions, frames, kv_len, caches)
        codes = [c0]
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads)):
            self.check_abort()
            hidden = layer(hidden, codes[-1], semantic, positions, caches[len(self.shared_layers) + index], kv_len)
            head_logits = head(hidden)
            finite &= torch.isfinite(head_logits).all()
            codes.append(
                sample_top_k_top_p(
                    head_logits, temperature, top_k, top_p, check_finite=False
                )
            )
        sampled = torch.stack((main, *codes), dim=-1)
        count = self.config.vq_backbone_codebooks
        return sampled, finite, self.embedding_sum(sampled, 1, 1 + count), semantic


