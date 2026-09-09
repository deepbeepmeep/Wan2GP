"""Fixed-shape CUDA graph runtime for MiniMax Music 3 semantic decoding."""

import math

import torch

from shared.llm_engines.cudagraph_kit import CUDAGraphRunner
from shared.llm_engines.nanovllm.utils.context import reset_context, set_context
from shared.llm_engines.nanovllm.vllm_support import probe_vllm_runtime


_VLLM_TRITON_SHAPES = {
    (2, 4096, 1024): (4, 128, 64, 4, 4),
    (2, 4096, 4096): (8, 128, 64, 4, 4),
    (2, 4096, 6144): (4, 128, 64, 4, 4),
    (2, 4096, 12288): (4, 128, 64, 4, 4),
    (2, 6144, 4096): (4, 128, 64, 4, 4),
    (2, 12288, 4096): (4, 128, 64, 4, 4),
}


class MiniMaxMusic3SemanticAcceleration:
    block_size = 256
    audio_end_token_id = 151670
    audio_code_offset = 151675
    semantic_vocab_size = 16384

    def __init__(self, language_model, rvq_depth_decoder, engine: str, device: torch.device):
        self.language_model = language_model
        self.rvq_depth_decoder = rvq_depth_decoder
        self.engine = engine
        self.device = device
        self._runner = CUDAGraphRunner("minimax_music3:qwen3_decode", self._decode_step)
        self._capacity = 0
        self._blocks_per_sequence = 0
        self._block_tables = None
        self._next_position = 0
        self._configure_kernels()

    def _configure_kernels(self):
        use_vllm = self.engine == "vllm"
        if use_vllm:
            probe = probe_vllm_runtime()
            if not probe["supported"]:
                raise RuntimeError("MiniMax Music 3 vllm mode requires both FlashAttention2 and Triton")
        from shared.kernels.quanto_int8_triton import configure_tiny_m_shape_overrides

        configure_tiny_m_shape_overrides(_VLLM_TRITON_SHAPES if use_vllm else None)
        for layer in self.language_model.model.layers:
            attention = layer.self_attn.attn
            attention.flash_attn_varlen_func = attention.flash_attn_varlen_func if use_vllm else None
            attention.flash_attn_with_kvcache = attention.flash_attn_with_kvcache if use_vllm else None
            attention.use_triton_kv_cache = use_vllm
            layer.input_layernorm.use_triton_rmsnorm = use_vllm
            layer.post_attention_layernorm.use_triton_rmsnorm = use_vllm
        self.language_model.model.norm.use_triton_rmsnorm = use_vllm
        self.rvq_depth_decoder.set_decode_engine(self.engine)
        kernels = "FlashAttention2 + Triton" if use_vllm else "SDPA"
        print(f"[MiniMax Music 3] semantic decode: CUDA graphs + {kernels}")

    def _allocate(self, capacity: int):
        if capacity <= self._capacity:
            return
        self._runner.release()
        self._capacity = int(capacity)
        self._blocks_per_sequence = math.ceil(self._capacity / self.block_size)
        total_blocks = 2 * self._blocks_per_sequence
        dtype = self.language_model.dtype
        for layer in self.language_model.model.layers:
            attention = layer.self_attn.attn
            shape = (total_blocks, self.block_size, attention.num_kv_heads, attention.head_dim)
            attention.k_cache = torch.empty(shape, device=self.device, dtype=dtype)
            attention.v_cache = torch.empty(shape, device=self.device, dtype=dtype)
        first = torch.arange(self._blocks_per_sequence, device=self.device, dtype=torch.int32)
        self._block_tables = torch.stack((first, first + self._blocks_per_sequence))

    def _slot_mapping(self, positions: torch.Tensor) -> torch.Tensor:
        sequence_offsets = torch.arange(2, device=self.device, dtype=torch.long) * self._blocks_per_sequence * self.block_size
        return sequence_offsets + positions

    def prefill(self, input_ids: torch.Tensor, max_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, prompt_length = input_ids.shape
        if batch_size != 2:
            raise ValueError(f"MiniMax Music 3 accelerated CFG expects batch size 2, got {batch_size}")
        self._allocate(prompt_length + max_frames)
        self._runner.release()
        prompt_positions = torch.arange(prompt_length, device=self.device, dtype=torch.long)
        positions = prompt_positions.repeat(2)
        slots = torch.cat((prompt_positions, prompt_positions + self._blocks_per_sequence * self.block_size))
        cu_seqlens = torch.tensor([0, prompt_length, 2 * prompt_length], device=self.device, dtype=torch.int32)
        set_context(True, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=prompt_length, max_seqlen_k=prompt_length, slot_mapping=slots)
        try:
            hidden = self.language_model(input_ids.reshape(-1), positions).view(2, prompt_length, -1)[:, -1]
        finally:
            reset_context()
        logits = self._compute_audio_logits(hidden)
        self._next_position = prompt_length
        self._prime_decode(hidden)
        self.rvq_depth_decoder.prepare_decode_graph(batch_size=2, device=self.device, dtype=hidden.dtype)
        return hidden, logits

    def _decode_step(self, inputs_embeds: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor, context_lens: torch.Tensor):
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=self._block_tables)
        hidden = self.language_model(None, positions, inputs_embeds).view(2, -1)
        return hidden, self._compute_audio_logits(hidden)

    def _compute_audio_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.language_model.lm_head.weight
        audio_logits = torch.nn.functional.linear(hidden, weight[self.audio_code_offset:self.audio_code_offset + self.semantic_vocab_size])
        end_logits = torch.nn.functional.linear(hidden, weight[self.audio_end_token_id:self.audio_end_token_id + 1])
        return torch.cat((audio_logits, end_logits), dim=-1)

    def _prime_decode(self, hidden: torch.Tensor):
        position = torch.full((2,), self._next_position, device=self.device, dtype=torch.long)
        slot_mapping = self._slot_mapping(position)
        context_lens = position.to(torch.int32) + 1
        self._decode_step(torch.zeros_like(hidden), position, slot_mapping, context_lens)
        reset_context()

    def decode(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.full((2,), self._next_position, device=self.device, dtype=torch.long)
        slot_mapping = self._slot_mapping(positions)
        context_lens = positions.to(torch.int32) + 1
        try:
            hidden, logits = self._runner.run(inputs_embeds.view(2, -1), positions, slot_mapping, context_lens)
        finally:
            reset_context()
        self._next_position += 1
        return hidden, logits

    def release(self):
        self._runner.release()
        self.rvq_depth_decoder.release_decode_graph()
        if self.engine == "vllm":
            from shared.kernels.quanto_int8_triton import configure_tiny_m_shape_overrides

            configure_tiny_m_shape_overrides()
        for layer in self.language_model.model.layers:
            attention = layer.self_attn.attn
            attention.k_cache = attention.v_cache = torch.tensor([])
        self._capacity = self._blocks_per_sequence = 0
        self._block_tables = None
        reset_context()
