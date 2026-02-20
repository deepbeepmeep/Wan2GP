import math
import gc
from collections import deque
from typing import Callable, List, Optional

import torch
from torch import nn
from tqdm import tqdm

from .attention import (
    ForwardContext,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
)
from .kv_manager import KVCacheManager, Seq


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    # @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        temperatures = temperatures.to(logits.device).clamp(min=1e-8)
        greedy_mask = temperatures < 1e-5
        temp_for_scaling = torch.where(greedy_mask, 1.0, temperatures)
        scaled_logits = logits / temp_for_scaling.unsqueeze(-1)
        probs = torch.softmax(scaled_logits, dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        sampled_tokens = probs.div_(q).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)


class AccelInferenceEngine:
    def __init__(
        self,
        model,
        lm_head,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 256,
        num_blocks: int = 128,
        use_cuda_graph: bool = True,
    ):
        """
        Args:
            model: The GPT transformer model (should have accel attention)
            lm_head: Language model head for generating logits
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            block_size: KV cache block size
            num_blocks: Total number of KV cache blocks
            use_cuda_graph: Whether to use CUDA Graph for decode optimization
        """
        self.model = model
        self.lm_head = lm_head
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._default_num_blocks = max(1, int(num_blocks))
        self.num_blocks = 1
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        self.hidden_size = (
            model.config.hidden_size
            if hasattr(model, "config")
            else head_dim * num_heads
        )
        self.kv_manager = self._new_kv_manager(self.num_blocks, device=torch.device("cpu"))
        self.kv_manager.wire_kv_cache_to_model(model)
        self.sampler = Sampler()
        self.current_sequences = []
        self.graphs = {}
        self.graph_vars = None
        self.graph_pool = None
        self.graph_captured = False
        self.graph_num_blocks = 0
        self.graph_signature = None

    def _runtime_kv_device(self) -> torch.device:
        first_param = next(self.model.parameters(), None)
        if first_param is not None and first_param.device.type == "cuda":
            return first_param.device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if first_param is not None:
            return first_param.device
        return torch.device("cpu")

    def _new_kv_manager(self, num_blocks: int, device: torch.device) -> KVCacheManager:
        return KVCacheManager(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            num_blocks=int(max(1, num_blocks)),
            dtype=torch.float16,  # Force fp16 for FlashAttention
            device=device,
        )

    @staticmethod
    def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        filtered = logits
        vocab_size = int(filtered.size(-1))
        top_k = int(top_k) if top_k is not None else 0
        top_p = float(top_p) if top_p is not None else 1.0
        if top_k > 0 and top_k < vocab_size:
            kth = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
            filtered = filtered.masked_fill(filtered < kth, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 0] = False
            remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
            filtered = filtered.masked_fill(remove_mask, float("-inf"))
        return filtered

    @staticmethod
    def _tensor_sig(tensor: Optional[torch.Tensor]):
        if tensor is None:
            return None
        return (
            int(tensor.data_ptr()),
            str(tensor.dtype),
            tuple(int(x) for x in tensor.shape),
            int(tensor.device.index if tensor.device.index is not None else -1),
        )

    def _module_first_param_sig(self, module: Optional[torch.nn.Module]):
        if module is None:
            return None
        try:
            first_param = next(module.parameters())
        except StopIteration:
            return None
        return self._tensor_sig(first_param)

    def _make_capture_signature(
        self,
        tts_mel_embedding: Optional[torch.nn.Module] = None,
        tts_text_pos_embedding: Optional[torch.nn.Module] = None,
    ):
        mel_weight = getattr(tts_mel_embedding, "weight", None) if tts_mel_embedding is not None else None
        pos_emb = None
        if tts_text_pos_embedding is not None:
            pos_emb = getattr(tts_text_pos_embedding, "emb", tts_text_pos_embedding)
        pos_weight = getattr(pos_emb, "weight", None) if pos_emb is not None else None
        return (
            self._module_first_param_sig(self.model),
            self._tensor_sig(self.kv_manager.kv_cache),
            self._tensor_sig(mel_weight),
            self._tensor_sig(pos_weight),
            int(self.num_blocks),
            int(self.block_size),
        )

    def _compute_tts_embeds(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tts_mel_embedding: Optional[torch.nn.Module] = None,
        tts_text_pos_embedding: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        if tts_mel_embedding is None or tts_text_pos_embedding is None:
            raise RuntimeError("TTS embedding modules are required for accel decode.")
        pos_emb_module = getattr(tts_text_pos_embedding, "emb", tts_text_pos_embedding)
        if not hasattr(pos_emb_module, "weight"):
            raise RuntimeError("TTS positional embedding module is missing a '.weight' tensor.")
        pos_clamped = torch.clamp(positions, min=0, max=pos_emb_module.weight.shape[0] - 1)
        mel_emb = tts_mel_embedding(input_ids)
        pos_emb = pos_emb_module(pos_clamped)
        return mel_emb + pos_emb

    def _required_blocks(self, total_tokens: int) -> int:
        # Keep one spare block to avoid edge overflows from token-length drift.
        return max(1, int(math.ceil(float(max(1, total_tokens)) / float(self.block_size))) + 1)

    def _reset_decode_graph(self):
        self.graphs = {}
        self.graph_vars = None
        self.graph_pool = None
        self.graph_captured = False
        self.graph_num_blocks = 0
        self.graph_signature = None

    def _resize_kv_cache_if_needed(self, required_blocks: int):
        target_blocks = int(max(1, required_blocks))
        target_device = self._runtime_kv_device()
        current_device = self.kv_manager.kv_cache.device
        same_device = (
            current_device.type == target_device.type
            and (current_device.type != "cuda" or current_device.index == target_device.index)
        )
        if target_blocks == int(self.num_blocks) and same_device:
            return
        self.num_blocks = target_blocks
        self.kv_manager = self._new_kv_manager(self.num_blocks, target_device)
        self.kv_manager.wire_kv_cache_to_model(self.model)
        self._reset_decode_graph()

    def release_runtime_cache(self):
        self.current_sequences = []
        reset_forward_context()
        self._reset_decode_graph()
        old_kv_manager = self.kv_manager
        self.num_blocks = 1
        self.kv_manager = self._new_kv_manager(self.num_blocks, torch.device("cpu"))
        self.kv_manager.wire_kv_cache_to_model(self.model)
        del old_kv_manager
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        gc.collect()

    def prepare_decode_graph(
        self,
        max_total_tokens: int,
        tts_mel_embedding: Optional[torch.nn.Module] = None,
        tts_text_pos_embedding: Optional[torch.nn.Module] = None,
    ):
        required_blocks = self._required_blocks(int(max_total_tokens))
        self._resize_kv_cache_if_needed(required_blocks)
        if not self.use_cuda_graph:
            return
        signature = self._make_capture_signature(
            tts_mel_embedding=tts_mel_embedding,
            tts_text_pos_embedding=tts_text_pos_embedding,
        )
        if (
            self.graph_captured
            and int(self.graph_num_blocks) == int(self.num_blocks)
            and self.graph_num_blocks >= required_blocks
            and self.graph_signature == signature
        ):
            return
        print(
            f"[CAPTURE] use_cuda_graph={self.use_cuda_graph}, graph_captured={self.graph_captured}, "
            f"graph_num_blocks={self.graph_num_blocks}, required_blocks={required_blocks}, cache_blocks={self.num_blocks}",
            flush=True,
        )
        self._reset_decode_graph()
        self._capture_cuda_graphs(
            tts_mel_embedding=tts_mel_embedding,
            tts_text_pos_embedding=tts_text_pos_embedding,
            max_num_blocks=self.num_blocks,
        )
        self.graph_captured = True
        self.graph_num_blocks = int(self.num_blocks)
        self.graph_signature = signature
        print(f"[CAPTURE] Completed! graphs={list(self.graphs.keys())}, num_blocks={self.graph_num_blocks}", flush=True)

    def _prepare_prefill(self, requests: List[Seq]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for req in requests:
            seqlen = len(req)
            input_ids.extend(req[req.num_cached_tokens :])
            positions.extend(list(range(req.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - req.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if req.block_table:
                num_cached = req.num_cached_tokens
                num_total = len(req)

                for token_idx in range(num_cached, num_total):
                    block_idx = token_idx // self.block_size
                    block_offset = token_idx % self.block_size
                    block_id = req.block_table[block_idx]
                    slot_idx = block_id * self.block_size + block_offset
                    slot_mapping.append(slot_idx)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            max_len = max(len(req.block_table) for req in requests)
            block_tables_list = []
            for req in requests:
                table = req.block_table + [-1] * (max_len - len(req.block_table))
                block_tables_list.append(table)
            block_tables = torch.tensor(
                block_tables_list, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)

        set_forward_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )

        return input_ids, positions

    def _reset_kv_allocator_state(self):
        # Keep allocated KV tensors, but reset allocator metadata to avoid
        # stale block reuse across independent generation calls.
        self.kv_manager.block_hash_to_id.clear()
        self.kv_manager.free_block_ids = deque(range(self.num_blocks))
        self.kv_manager.used_block_ids.clear()
        for block in self.kv_manager.blocks:
            block.ref_cnt = 0
            block._block_hash = None
            block.token_ids = []

    def _prepare_decode(self, requests: List[Seq]):
        if not requests:
            raise RuntimeError("FATAL: No requests provided to _prepare_decode!")

        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for req in requests:
            input_ids.append(req.last_token)

            pos = len(req) - 1
            if hasattr(self, "_tts_mode") and self._tts_mode:
                pos = pos - (self._tts_prompt_len - 1)
            positions.append(pos)

            context_lens.append(len(req))
            slot_mapping.append(
                req.block_table[-1] * self.block_size + req.last_block_num_tokens - 1
            )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        max_len = max(len(req.block_table) for req in requests)
        block_tables_list = []
        for req in requests:
            table = req.block_table + [-1] * (max_len - len(req.block_table))
            block_tables_list.append(table)
        block_tables = torch.tensor(
            block_tables_list, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        assert block_tables.dim() == 2, (
            f"block_tables must be 2D, got shape {block_tables.shape}"
        )
        assert block_tables.size(0) == len(requests), (
            f"block_tables batch size mismatch: {block_tables.size(0)} vs {len(requests)}"
        )

        set_forward_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        return input_ids, positions

    def _prepare_sample(self, requests: List[Seq], temperature: float):
        temperatures = [temperature] * len(requests)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    def _capture_cuda_graphs(
        self,
        tts_mel_embedding=None,
        tts_text_pos_embedding=None,
        max_num_blocks: Optional[int] = None,
    ):
        print("Capturing CUDA graphs for decode optimization...")
        max_bs = 8  # Support up to batch size 8
        if max_num_blocks is None:
            max_num_blocks = self.num_blocks
        max_num_blocks = max(1, min(int(max_num_blocks), int(self.num_blocks)))
        model_dtype = next(self.model.parameters()).dtype
        input_ids = torch.ones(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.ones(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device="cuda"
        )
        outputs = torch.zeros(
            max_bs, self.hidden_size, dtype=model_dtype, device="cuda"
        )
        inputs_embeds_buffer = torch.zeros(
            max_bs, self.hidden_size, dtype=model_dtype, device="cuda"
        )

        self.graph_bs = [1, 2, 4, 8]

        use_tts = tts_mel_embedding is not None and tts_text_pos_embedding is not None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            slot_mapping[:bs] = torch.arange(bs, dtype=torch.int32, device="cuda")
            context_lens[:bs] = bs + 1
            block_tables[:bs, :] = 0

            set_forward_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            # warmup
            if use_tts:
                inputs_embeds_buffer[:bs] = self._compute_tts_embeds(
                    input_ids[:bs],
                    positions[:bs],
                    tts_mel_embedding=tts_mel_embedding,
                    tts_text_pos_embedding=tts_text_pos_embedding,
                )
                out = self.model(
                    inputs_embeds=inputs_embeds_buffer[:bs].unsqueeze(1),
                    return_dict=True,
                ).last_hidden_state
            else:
                out = self.model(
                    input_ids=input_ids[:bs].unsqueeze(1), return_dict=True
                ).last_hidden_state
            outputs[:bs] = out.squeeze(1) if out.dim() == 3 else out

            with torch.cuda.graph(graph, self.graph_pool):
                if use_tts:
                    inputs_embeds_buffer[:bs] = self._compute_tts_embeds(
                        input_ids[:bs],
                        positions[:bs],
                        tts_mel_embedding=tts_mel_embedding,
                        tts_text_pos_embedding=tts_text_pos_embedding,
                    )
                    out = self.model(
                        inputs_embeds=inputs_embeds_buffer[:bs].unsqueeze(1),
                        return_dict=True,
                    ).last_hidden_state
                else:
                    out = self.model(
                        input_ids=input_ids[:bs].unsqueeze(1), return_dict=True
                    ).last_hidden_state
                outputs[:bs] = out.squeeze(1) if out.dim() == 3 else out

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_forward_context()

        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
            "inputs_embeds": inputs_embeds_buffer,
        }
        print(f"CUDA graphs captured for batch sizes: {self.graph_bs}")

    def _run_decode_with_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: ForwardContext,
        tts_mel_embedding: Optional[torch.nn.Module] = None,
        tts_text_pos_embedding: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        bs = input_ids.size(0)
        use_tts_embedding = hasattr(self, "_tts_mode") and self._tts_mode

        if not self.use_cuda_graph or not self.graphs:
            if use_tts_embedding:
                inputs_embeds = self._compute_tts_embeds(
                    input_ids,
                    positions,
                    tts_mel_embedding=tts_mel_embedding,
                    tts_text_pos_embedding=tts_text_pos_embedding,
                )
                out = self.model(
                    inputs_embeds=inputs_embeds.unsqueeze(1), return_dict=True
                ).last_hidden_state
            else:
                out = self.model(
                    input_ids=input_ids.unsqueeze(1), return_dict=True
                ).last_hidden_state
            return out.squeeze(1) if out.dim() == 3 else out

        graph_bs = next((x for x in self.graph_bs if x >= bs), None)
        if graph_bs is None:
            if use_tts_embedding:
                inputs_embeds = self._compute_tts_embeds(
                    input_ids,
                    positions,
                    tts_mel_embedding=tts_mel_embedding,
                    tts_text_pos_embedding=tts_text_pos_embedding,
                )
                out = self.model(
                    inputs_embeds=inputs_embeds.unsqueeze(1), return_dict=True
                ).last_hidden_state
            else:
                out = self.model(
                    input_ids=input_ids.unsqueeze(1), return_dict=True
                ).last_hidden_state
            return out.squeeze(1) if out.dim() == 3 else out

        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars

        if graph_vars is None:
            raise RuntimeError("Graph variables not initialized")

        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :].fill_(-1)
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
            context.block_tables
        )
        graph.replay()

        return graph_vars["outputs"][:bs]

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tts_embeddings: Optional[
            torch.Tensor
        ] = None,  # TTS: [pad][cond][text] embeddings (87 tokens, NO start_mel)
        tts_mel_embedding: Optional[torch.nn.Module] = None,  # TTS: mel_embedding layer
        tts_text_pos_embedding: Optional[
            torch.nn.Module
        ] = None,  # TTS: text_pos_embedding layer
        cg_max_total_tokens: Optional[int] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
    ) -> torch.Tensor:
        """
        Generate tokens.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            stop_tokens: List of token IDs that stop generation

        Returns:
            Generated token IDs [batch_size, total_len]
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        self._tts_mode = tts_embeddings is not None
        self._tts_prompt_len = input_ids.size(1) if self._tts_mode else 0
        self._reset_kv_allocator_state()
        prompt_tokens = (tts_embeddings.size(1) + 1) if tts_embeddings is not None else input_ids.size(1)
        required_total_tokens = int(prompt_tokens + max(1, int(max_new_tokens)))
        if cg_max_total_tokens is not None:
            required_total_tokens = max(required_total_tokens, int(cg_max_total_tokens))
        self.prepare_decode_graph(
            required_total_tokens,
            tts_mel_embedding=tts_mel_embedding,
            tts_text_pos_embedding=tts_text_pos_embedding,
        )

        if tts_embeddings is not None:
            actual_seq_len = tts_embeddings.size(1) + 1  # embeddings + start_mel_token
        else:
            actual_seq_len = input_ids.size(1)

        is_varlen_batch = (
            tts_embeddings is not None
            and attention_mask is not None
            and batch_size > 1
            and (attention_mask.sum(dim=1) != attention_mask.size(1)).any()
        )

        if is_varlen_batch:
            seq_lens = [attention_mask[i].sum().item() for i in range(batch_size)]
        else:
            seq_lens = [actual_seq_len] * batch_size

        sequences = []
        for i in range(batch_size):
            seq_len = seq_lens[i]
            token_ids = [1] * seq_len
            if tts_embeddings is not None and seq_len > 0:
                token_ids[-1] = input_ids[i, -1].item() if input_ids.size(1) > 0 else 1
            else:
                token_ids = input_ids[i].tolist()
            req = Seq(token_ids)
            self.kv_manager.allocate(req)
            sequences.append(req)

        self.current_sequences = sequences

        prefill_ids, prefill_pos = self._prepare_prefill(sequences)

        if (
            tts_embeddings is not None
            and tts_mel_embedding is not None
            and tts_text_pos_embedding is not None
        ):
            start_token_id = input_ids[0, -1] if input_ids.size(1) > 0 else 8192

            start_emb = tts_mel_embedding(
                torch.tensor([[start_token_id]], device="cuda")
            )  # [1, 1, hidden_dim]

            start_pos = torch.tensor(
                [[tts_embeddings.size(1)]], device="cuda", dtype=torch.long
            )
            pos_emb_module = getattr(tts_text_pos_embedding, "emb", tts_text_pos_embedding)
            pos_emb = pos_emb_module(start_pos)
            start_emb = start_emb + pos_emb
            start_emb = start_emb.repeat(batch_size, 1, 1)

            if is_varlen_batch:
                valid_embeddings = []
                for i in range(batch_size):
                    emb_len = seq_lens[i] - 1
                    padding_len = tts_embeddings.size(1) - emb_len
                    valid_emb = tts_embeddings[i, padding_len:].unsqueeze(
                        0
                    )  # [1, emb_len, hidden_dim]
                    valid_embeddings.append(
                        torch.cat([valid_emb, start_emb[i : i + 1]], dim=1)
                    )
                full_embeddings = torch.cat(
                    valid_embeddings, dim=1
                )  # [1, total_tokens, hidden_dim]
            else:
                full_embeddings = torch.cat(
                    [tts_embeddings, start_emb], dim=1
                )  # [batch_size, seq_len, hidden_dim]

            model_dtype = next(self.model.parameters()).dtype
            if full_embeddings.dtype != model_dtype:
                full_embeddings = full_embeddings.to(model_dtype)

            hidden_states = self.model(
                inputs_embeds=full_embeddings, return_dict=True
            ).last_hidden_state

        else:
            hidden_states = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state

        if is_varlen_batch:
            context = get_forward_context()
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            last_hidden = torch.stack(
                [hidden_states[0, cu_seqlens[i + 1] - 1] for i in range(batch_size)]
            )
        else:
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        reset_forward_context()

        if self.lm_head is not None:
            if last_hidden.dtype != next(self.lm_head.parameters()).dtype:
                last_hidden = last_hidden.to(next(self.lm_head.parameters()).dtype)
            logits = self.lm_head(last_hidden)  # [batch_size, vocab_size]
        else:
            logits = self.model.compute_logits(last_hidden)  # [batch_size, vocab_size]

        temperatures = self._prepare_sample(sequences, temperature)
        if temperature > 0:
            sampling_logits = self._filter_logits(logits, top_k=top_k, top_p=top_p)
            first_token = self.sampler(sampling_logits, temperatures)
        else:
            first_token = torch.argmax(logits, dim=-1)

        first_token_list = first_token.tolist()

        generated_tokens = [[] for _ in range(batch_size)]
        is_finished = [False] * batch_size
        token_progress = tqdm(total=int(max_new_tokens), desc="transformer_tokens", unit="tok", leave=True)
        def _should_stop_early():
            if stop_checker is None:
                return False
            try:
                return bool(stop_checker())
            except Exception:
                return False

        try:
            for i, token_id in enumerate(first_token_list):
                if stop_tokens and token_id in stop_tokens:
                    is_finished[i] = True
                    generated_tokens[i].append(token_id)
                else:
                    generated_tokens[i].append(token_id)
                    sequences[i].append_token(token_id)
                    self.kv_manager.append_to_seq(sequences[i])
            token_progress.update(1)
            stop_early = _should_stop_early()

            if all(is_finished) and not stop_early:
                for req in sequences:
                    self.kv_manager.remove_seq(req)
                self.current_sequences = []

                output_ids = []
                for i in range(batch_size):
                    full_sequence = input_ids[i].tolist() + generated_tokens[i]
                    output_ids.append(full_sequence)

                output = torch.tensor(output_ids, dtype=torch.long, device=device)
                return output

            remaining_tokens = 0 if stop_early else (max_new_tokens - 1)

            for step in range(remaining_tokens):
                if _should_stop_early():
                    break
                decode_ids, decode_pos = self._prepare_decode(sequences)

                context = get_forward_context()
                hidden_states = self._run_decode_with_graph(
                    decode_ids,
                    decode_pos,
                    context,
                    tts_mel_embedding=tts_mel_embedding,
                    tts_text_pos_embedding=tts_text_pos_embedding,
                )

                # Get logits
                if self.lm_head is not None:
                    logits = self.lm_head(hidden_states)  # [batch_size, vocab_size]
                else:
                    logits = self.model.compute_logits(
                        hidden_states
                    )  # [batch_size, vocab_size]

                reset_forward_context()

                temperatures = self._prepare_sample(sequences, temperature)
                if temperature > 0:
                    sampling_logits = self._filter_logits(logits, top_k=top_k, top_p=top_p)
                    next_token = self.sampler(sampling_logits, temperatures)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                next_token_list = next_token.tolist()

                for i, token_id in enumerate(next_token_list):
                    if is_finished[i]:
                        continue
                    elif stop_tokens and token_id in stop_tokens:
                        is_finished[i] = True
                        generated_tokens[i].append(token_id)
                    else:
                        sequences[i].append_token(token_id)
                        self.kv_manager.append_to_seq(sequences[i])
                        generated_tokens[i].append(token_id)
                token_progress.update(1)

                if all(is_finished):
                    break

            for req in sequences:
                self.kv_manager.remove_seq(req)
            self.current_sequences = []

            pad_token = stop_tokens[0] if stop_tokens else 0

            if is_varlen_batch:
                max_prompt_len = attention_mask.size(1)
                output_ids = []

                for i in range(batch_size):
                    padding_len = max_prompt_len - seq_lens[i]
                    initial_tokens = sequences[i].token_ids[
                        : sequences[i].num_prompt_tokens
                    ]
                    padded_prompt = [pad_token] * padding_len + initial_tokens
                    full_sequence = padded_prompt + generated_tokens[i]
                    output_ids.append(full_sequence)
            else:
                output_ids = [
                    sequences[i].token_ids[: sequences[i].num_prompt_tokens]
                    + generated_tokens[i]
                    for i in range(batch_size)
                ]

            max_length = max(len(seq) for seq in output_ids)
            padded_output_ids = [
                seq + [pad_token] * (max_length - len(seq)) for seq in output_ids
            ]

            output = torch.tensor(padded_output_ids, dtype=torch.long, device=device)

            assert output.size(0) == batch_size, (
                f"Output batch size mismatch: {output.size(0)} != {batch_size}"
            )

            return output
        finally:
            token_progress.close()


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    # @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
