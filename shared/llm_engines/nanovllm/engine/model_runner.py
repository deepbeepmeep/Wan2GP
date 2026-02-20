import pickle
import gc
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import sys

from ..config import Config
from .sequence import Sequence
from ..layers.sampler import Sampler
from ..utils.context import set_context, get_context, reset_context

import socket


def find_available_port(start_port: int = 2333, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: The starting port number to check
        max_attempts: Maximum number of ports to try
        
    Returns:
        An available port number
        
    Raises:
        RuntimeError: If no available port is found within max_attempts
    """
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return port
        except OSError:
            # Port is in use, try next one
            continue
    raise RuntimeError(f"Could not find an available port starting from {start_port} after {max_attempts} attempts")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], model_object=None):
        # Enable capturing scalar outputs to avoid graph breaks from Tensor.item() calls
        torch._dynamo.config.capture_scalar_outputs = True
        
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        if self.world_size > 1:
            dist_port = find_available_port()
            print(f"[debug]dist_port: {dist_port}")
            # Use gloo backend on Windows, nccl on Linux/other platforms
            backend = "gloo" if sys.platform == "win32" else "nccl"
            dist.init_process_group(backend, f"tcp://127.0.0.1:{dist_port}", world_size=self.world_size, rank=rank)
            torch.cuda.set_device(rank)
        else:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        # Use dtype instead of deprecated torch_dtype
        config_dtype = getattr(hf_config, 'dtype', getattr(hf_config, 'torch_dtype', None))

        # Validate and convert config_dtype to a valid torch floating-point dtype
        # Default to bfloat16 for CUDA (required for Flash Attention 2)
        if config_dtype is None:
            config_dtype = torch.bfloat16
        elif isinstance(config_dtype, str):
            # Convert string dtype to torch dtype
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float64': torch.float64,
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float64': torch.float64,
            }
            config_dtype = dtype_map.get(config_dtype.lower(), torch.bfloat16)
        elif not isinstance(config_dtype, torch.dtype) or not config_dtype.is_floating_point:
            # If not a valid floating-point torch dtype, default to bfloat16
            config_dtype = torch.bfloat16

        self.dtype = config_dtype  # Save for later use
        self._runtime_ready = False
        self._graph_cache = {}
        self._graph_cache_order = []
        self._logits_bias_cache = {}
        self._sampling_generator = None
        self._guard_counts = {}
        self._guard_seen_details = set()
        torch.set_default_dtype(config_dtype)
        if model_object is None:
            raise RuntimeError(
                "nanovllm now requires a preloaded MMGP model object. "
                "Pass model_object=... when creating LLM."
            )
        self.model = model_object
        self.sampler = Sampler()
        
        # Pre-allocate buffers for sampling (optimization: avoid repeated tensor creation)
        # Must be called before model execution paths that use these buffers.
        self._allocate_sample_buffers()
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def ensure_runtime_ready(self):
        if self._runtime_ready:
            return
        if self.model is None:
            raise RuntimeError("LLM model object is not available.")
        self._tie_word_embeddings_if_needed()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        self._runtime_ready = True

    def _get_tied_embeddings(self):
        if self.model is None:
            return None
        hf_config = getattr(self.config, "hf_config", None)
        if not bool(getattr(hf_config, "tie_word_embeddings", False)):
            return None
        lm_head = getattr(self.model, "lm_head", None)
        embed = getattr(self.model, "embed_tokens", None)
        if lm_head is None or embed is None:
            return None
        lm_w = getattr(lm_head, "weight", None)
        emb_w = getattr(embed, "weight", None)
        if lm_w is None or emb_w is None:
            return None
        return lm_head, embed

    def _tie_word_embeddings_if_needed(self):
        tied = self._get_tied_embeddings()
        if tied is None:
            return
        lm_head, embed = tied
        if lm_head.weight is not embed.weight:
            lm_head.register_parameter("weight", embed.weight)
        if lm_head.weight is not embed.weight:
            raise RuntimeError("Failed to retie lm_head.weight with embed_tokens.weight.")

    def reset_runtime_state(self):
        if not self._runtime_ready:
            return
        # Clear attention KV cache refs so we don't write into freed storage later.
        try:
            if self.model is not None:
                for module in self.model.modules():
                    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                        module.k_cache = module.v_cache = torch.tensor([])
        except Exception:
            pass
        if hasattr(self, "kv_cache"):
            try:
                del self.kv_cache
            except Exception:
                pass
        # CUDA graphs captured against previous model/KV pointers are unsafe after runtime reset.
        # Force recapture on next prepare to avoid stale-pointer illegal memory access.
        try:
            self.clear_graph_cache()
        except Exception:
            pass
        try:
            self.graphs = {}
            self.graph_vars = {}
            self.graph_bs = []
            self.graph_pool = None
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._logits_bias_cache.clear()
        self._sampling_generator = None
        self._runtime_ready = False
        gc.collect()

    def _get_graph_capture_signature(self):
        model_ptr = -1
        kv_ptr = -1
        try:
            first_param = next(self.model.parameters())
            if first_param.is_cuda:
                model_ptr = int(first_param.data_ptr())
        except Exception:
            pass
        try:
            if hasattr(self, "kv_cache") and torch.is_tensor(self.kv_cache) and self.kv_cache.is_cuda:
                kv_ptr = int(self.kv_cache.data_ptr())
        except Exception:
            pass
        return (model_ptr, kv_ptr, int(self.config.max_model_len), int(self.config.max_num_seqs))

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _drop_graph_cache_entry(self, cache_key):
        entry = self._graph_cache.pop(cache_key, None)
        if cache_key in self._graph_cache_order:
            self._graph_cache_order.remove(cache_key)
        if entry is None:
            return
        try:
            del entry["graphs"]
            del entry["pool"]
            del entry["vars"]
            del entry["bs"]
        except Exception:
            pass

    def clear_graph_cache(self):
        if self._graph_cache:
            for key in list(self._graph_cache.keys()):
                self._drop_graph_cache_entry(key)
            self._graph_cache.clear()
            self._graph_cache_order.clear()

    def _note_guard(self, name: str, detail: str | None = None):
        count = self._guard_counts.get(name, 0) + 1
        self._guard_counts[name] = count
        if detail:
            detail_key = (name, detail)
            if detail_key not in self._guard_seen_details:
                print(f"[nanovllm][guard] {name}: {detail}")
                self._guard_seen_details.add(detail_key)
            return
        if count == 1:
            print(f"[nanovllm][guard] {name}")

    def _get_logits_bias(self, seq: Sequence, logits: torch.Tensor):
        bias = getattr(seq, "logits_bias", None)
        if bias is None or not torch.is_tensor(bias):
            return None
        key = (id(bias), logits.device, logits.dtype)
        cached = self._logits_bias_cache.get(key)
        if cached is not None:
            return cached
        cached = bias.to(device=logits.device, dtype=logits.dtype)
        self._logits_bias_cache[key] = cached
        return cached

    def set_sampling_seed(self, seed: int | None):
        if seed is None or not torch.cuda.is_available():
            self._sampling_generator = None
            return
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(seed))
        self._sampling_generator = generator

    @staticmethod
    def _apply_logits_bias(logits_row: torch.Tensor, bias: torch.Tensor):
        logits_row.add_(bias)

    def _allocate_sample_buffers(self):
        """Pre-allocate reusable buffers for sampling to avoid repeated tensor creation."""
        max_bs = self.config.max_num_seqs
        max_tokens = self.config.max_num_batched_tokens
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        
        # Pre-allocate pinned memory buffers on CPU for fast transfer
        # Must explicitly specify device="cpu" since default device may be "cuda"
        self._cpu_temperatures = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=True)
        self._cpu_cfg_scales = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=True)
        self._cpu_top_ks = torch.zeros(max_bs, dtype=torch.int32, device="cpu", pin_memory=True)
        self._cpu_top_ps = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=True)
        self._cpu_repetition_penalties = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=True)
        
        # Pre-allocate decode buffers on CPU with pinned memory
        self._cpu_input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cpu", pin_memory=True)
        self._cpu_positions = torch.zeros(max_bs, dtype=torch.int64, device="cpu", pin_memory=True)
        self._cpu_slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cpu", pin_memory=True)
        self._cpu_context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cpu", pin_memory=True)
        
        # Pre-allocate prefill buffers on CPU with pinned memory (optimization to avoid repeated tensor creation)
        self._cpu_prefill_input_ids = torch.zeros(max_tokens, dtype=torch.int64, device="cpu", pin_memory=True)
        self._cpu_prefill_positions = torch.zeros(max_tokens, dtype=torch.int64, device="cpu", pin_memory=True)
        self._cpu_prefill_cu_seqlens = torch.zeros(max_bs + 1, dtype=torch.int32, device="cpu", pin_memory=True)
        self._cpu_prefill_slot_mapping = torch.zeros(max_tokens, dtype=torch.int32, device="cpu", pin_memory=True)
        
        # Pre-allocate block tables buffer (shared by both decode and prefill)
        self._cpu_block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        
        # Pre-allocate buffer for sequence token IDs (used in logits processor and sampler)
        # Max length is max_model_len since sequences can be that long
        self._seq_token_ids_buffer = torch.zeros(max_bs, self.config.max_model_len, dtype=torch.int64, device="cpu", pin_memory=True)

    def _release_sample_buffers(self):
        buffer_names = [
            "_cpu_temperatures",
            "_cpu_cfg_scales",
            "_cpu_top_ks",
            "_cpu_top_ps",
            "_cpu_repetition_penalties",
            "_cpu_input_ids",
            "_cpu_positions",
            "_cpu_slot_mapping",
            "_cpu_context_lens",
            "_cpu_prefill_input_ids",
            "_cpu_prefill_positions",
            "_cpu_prefill_cu_seqlens",
            "_cpu_prefill_slot_mapping",
            "_cpu_block_tables",
            "_seq_token_ids_buffer",
        ]
        for name in buffer_names:
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    pass

    def exit(self):
        try:
            self.reset_runtime_state()
        except Exception:
            pass
        self._release_sample_buffers()
        self._logits_bias_cache.clear()
        self._guard_counts.clear()
        self._guard_seen_details.clear()
        if hasattr(self, "sampler"):
            self.sampler = None
        if hasattr(self, "model"):
            self.model = None
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            if hasattr(self, "graphs"):
                del self.graphs
            if hasattr(self, "graph_pool"):
                del self.graph_pool
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize

        # Strict policy: allocate exactly the blocks required by requested runtime limits.
        required_blocks_per_seq = (config.max_model_len + self.block_size - 1) // self.block_size
        required_total_blocks = required_blocks_per_seq * max(1, config.max_num_seqs)
        config.num_kvcache_blocks = max(1, int(required_total_blocks))
        required_kv_bytes = config.num_kvcache_blocks * block_bytes
        free, total = torch.cuda.mem_get_info()
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        target_total_usage = total * config.gpu_memory_utilization
        allowed_kv_bytes = max(0, target_total_usage - current)
        if required_kv_bytes > allowed_kv_bytes:
            raise RuntimeError(
                f"Insufficient GPU memory for strict KV cache sizing under gpu_memory_utilization={config.gpu_memory_utilization:.2f}. "
                f"Required KV: {required_kv_bytes / 1024**3:.2f} GB "
                f"(blocks={config.num_kvcache_blocks}, block={block_bytes / 1024**2:.2f} MB), "
                f"Allowed by limit: {allowed_kv_bytes / 1024**3:.2f} GB, "
                f"Free: {free / 1024**3:.2f} GB, Current: {current / 1024**3:.2f} GB, "
                f"Requested max_model_len={config.max_model_len}, max_num_seqs={config.max_num_seqs}."
            )
        try:
            self.kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                device="cuda",
                dtype=self.dtype,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                free_now, total_now = torch.cuda.mem_get_info()
                raise RuntimeError(
                    f"Failed to allocate strict KV cache ({required_kv_bytes / 1024**3:.2f} GB) for "
                    f"max_model_len={config.max_model_len}, max_num_seqs={config.max_num_seqs}. "
                    f"Current free VRAM: {free_now / 1024**3:.2f} GB / {total_now / 1024**3:.2f} GB total."
                ) from exc
            raise
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup: no blocks allocated yet
                slot_mapping.extend([-1] * seqlen_q)
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Optimized decode preparation using pre-allocated buffers."""
        bs = len(seqs)
        
        # Use pre-allocated CPU buffers
        for i, seq in enumerate(seqs):
            self._cpu_input_ids[i] = seq.last_token
            self._cpu_positions[i] = len(seq) - 1
            self._cpu_context_lens[i] = len(seq)
            self._cpu_slot_mapping[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        
        # Transfer to GPU using sliced views
        input_ids = self._cpu_input_ids[:bs].cuda(non_blocking=True)
        positions = self._cpu_positions[:bs].cuda(non_blocking=True)
        slot_mapping = self._cpu_slot_mapping[:bs].cuda(non_blocking=True)
        context_lens = self._cpu_context_lens[:bs].cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_cfg_batch: bool = False):
        """Optimized sample preparation using pre-allocated buffers."""
        if is_cfg_batch:
            num_seqs = len(seqs) // 2
            target_seqs = seqs[:num_seqs]
        else:
            num_seqs = len(seqs)
            target_seqs = seqs
        
        # Fill pre-allocated CPU buffers
        top_ks_is_zero = True
        top_ps_is_one = True
        repetition_penalties_is_one = True
        for i, seq in enumerate(target_seqs):
            self._cpu_temperatures[i] = seq.temperature
            self._cpu_cfg_scales[i] = seq.cfg_scale
            self._cpu_top_ks[i] = seq.top_k if seq.top_k is not None else 0
            if seq.top_k is not None and seq.top_k > 0:
                top_ks_is_zero = False
            self._cpu_top_ps[i] = seq.top_p if seq.top_p is not None else 1.0
            if seq.top_p is not None and seq.top_p != 1.0:
                top_ps_is_one = False
            self._cpu_repetition_penalties[i] = seq.repetition_penalty if seq.repetition_penalty is not None else 1.0
            if seq.repetition_penalty is not None and seq.repetition_penalty != 1.0:
                repetition_penalties_is_one = False
        
        # Transfer to GPU using sliced views (single batched transfer)
        temperatures = self._cpu_temperatures[:num_seqs].cuda(non_blocking=True)
        cfg_scales = self._cpu_cfg_scales[:num_seqs].cuda(non_blocking=True)
        top_ks = self._cpu_top_ks[:num_seqs].cuda(non_blocking=True) if not top_ks_is_zero else None
        top_ps = self._cpu_top_ps[:num_seqs].cuda(non_blocking=True) if not top_ps_is_one else None
        repetition_penalties = self._cpu_repetition_penalties[:num_seqs].cuda(non_blocking=True) if not repetition_penalties_is_one else None
        
        return temperatures, cfg_scales, top_ks, top_ps, repetition_penalties

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            
            # Check if block_tables size exceeds pre-allocated buffer size
            # This can happen when conditional and unconditional sequences have different lengths
            # in CFG mode, causing block_tables to have more columns than expected
            max_num_blocks = self.graph_vars["block_tables"].size(1)
            if context.block_tables.size(1) > max_num_blocks:
                # Fall back to eager mode when block_tables is too large for CUDA graph
                self._note_guard(
                    "cudagraph_fallback_block_table_cols",
                    f"requested={context.block_tables.size(1)} max={max_num_blocks}",
                )
                return self.model.compute_logits(self.model(input_ids, positions))
            
            # Fix: Also check if block_tables row count matches batch size
            # Dimension mismatch can cause CUDA illegal memory access during graph replay
            if context.block_tables.size(0) != bs:
                # Fall back to eager mode when block_tables row count doesn't match batch size
                self._note_guard(
                    "cudagraph_fallback_block_table_rows",
                    f"rows={context.block_tables.size(0)} bs={bs}",
                )
                return self.model.compute_logits(self.model(input_ids, positions))
            
            # Fix: Verify slot_mapping and context_lens dimensions match batch size
            if context.slot_mapping.size(0) != bs or context.context_lens.size(0) != bs:
                # Fall back to eager mode when dimensions don't match
                self._note_guard(
                    "cudagraph_fallback_context_shape",
                    f"slot={context.slot_mapping.size(0)} ctx={context.context_lens.size(0)} bs={bs}",
                )
                return self.model.compute_logits(self.model(input_ids, positions))
            
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # Clear block_tables first to ensure no stale data from previous runs
            graph_vars["block_tables"][:bs].fill_(-1)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run model forward and sampling. For CFG sequences, batch is structured as:
        [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
        where uncond_seqi is the paired unconditional sequence of cond_seqi."""
        self.ensure_runtime_ready()
        # Check if this is a CFG batch (contains paired conditional and unconditional sequences)
        is_cfg_batch = seqs[0].cfg_scale > 1.0 and seqs[0].paired_seq is not None
        if is_cfg_batch:
            # CFG batch: seqs = [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
            num_cond = len(seqs) // 2
            cond_seqs = seqs[:num_cond]
            # uncond_seqs = seqs[num_cond:]
            
            # Prepare inputs for both conditional and unconditional (they're already in the batch)
            input_ids, positions = (self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs))
            sample_params = self.prepare_sample(seqs, is_cfg_batch=True) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = repetition_penalties = None
            
            # Run model forward (processes entire batch: cond + uncond)
            logits_all = self.run_model(input_ids, positions, is_prefill)
            reset_context()
            
            if self.rank == 0:
                # Split logits: first half is conditional, second half is unconditional
                logits_cond = logits_all[:num_cond]
                logits_uncond = logits_all[num_cond:]
                
                # Apply repetition penalty to conditional logits (before CFG)
                if repetition_penalties is not None:
                    for i, seq in enumerate(cond_seqs):
                        penalty = repetition_penalties[i].item()
                        if penalty != 1.0:
                            # Only penalize completion tokens (not prompt tokens)
                            completion_tokens = torch.tensor(seq.completion_token_ids, device=logits_cond.device)
                            if len(completion_tokens) > 0:
                                # Create token mask: mark tokens that appeared in completion
                                token_mask = torch.zeros(logits_cond.shape[1], dtype=torch.bool, device=logits_cond.device)
                                token_mask[completion_tokens] = True
                                
                                # Apply standard repetition penalty formula (matching transformers implementation):
                                # For tokens in completion: if score < 0 then score * penalty, else score / penalty
                                penalty_scores = torch.where(
                                    logits_cond[i] < 0,
                                    logits_cond[i] * penalty,
                                    logits_cond[i] / penalty
                                )
                                # Only apply penalty to tokens that appeared in completion
                                logits_cond[i] = torch.where(token_mask, penalty_scores, logits_cond[i])
                
                # Apply CFG formula: logits_cfg = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                cfg_scales_tensor = cfg_scales.unsqueeze(1)  # [num_cond, 1]
                logits_cfg = logits_uncond + cfg_scales_tensor * (logits_cond - logits_uncond)

                # Apply optional per-sequence logits bias before processors/sampling.
                for i, seq in enumerate(cond_seqs):
                    bias = self._get_logits_bias(seq, logits_cfg)
                    if bias is not None:
                        self._apply_logits_bias(logits_cfg[i], bias)
                
                # Apply logits processor for constrained decoding (if any sequence has one)
                for i, seq in enumerate(cond_seqs):
                    if seq.logits_processor is not None:
                        # Create input_ids tensor for this sequence
                        seq_input_ids = torch.tensor([seq.token_ids], device=logits_cfg.device)
                        # Apply processor to this sequence's logits
                        logits_cfg[i:i+1] = seq.logits_processor(seq_input_ids, logits_cfg[i:i+1])

                # Prepare input_ids for sampler (for repetition penalty, though we already applied it)
                # cond_input_ids = torch.tensor([seq.token_ids for seq in cond_seqs], device=logits_cfg.device)
                
                # Sample from CFG logits
                token_ids_cfg = self.sampler(
                    logits_cfg, 
                    temperatures,
                    top_ks=top_ks if top_ks is not None else None,
                    top_ps=top_ps if top_ps is not None else None,
                    repetition_penalties=None,  # Already applied above
                    generator=self._sampling_generator,
                    # input_ids=cond_input_ids,
                ).tolist()
                
                # Update logits processor state after sampling
                # NOTE: Only update for the first sequence since all sequences share the same processor
                # Updating multiple times would cause duplicate state updates (e.g., codes_count += N instead of += 1)
                if cond_seqs and cond_seqs[0].logits_processor_update_state is not None:
                    cond_seqs[0].logits_processor_update_state(token_ids_cfg[0])
                
                # Return token_ids (will be applied to both conditional and unconditional sequences)
                return token_ids_cfg
            else:
                return None
        else:
            # Normal batch (non-CFG)
            input_ids, positions = (self.prepare_prefill(seqs) if is_prefill 
                                   else self.prepare_decode(seqs))
            sample_params = self.prepare_sample(seqs, is_cfg_batch=False) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = repetition_penalties = None
            logits = self.run_model(input_ids, positions, is_prefill)
            reset_context()
            
            if self.rank == 0:
                # Apply repetition penalty to logits
                if repetition_penalties is not None:
                    for i, seq in enumerate(seqs):
                        penalty = repetition_penalties[i].item()
                        if penalty != 1.0:
                            # Only penalize completion tokens (not prompt tokens)
                            completion_tokens = torch.tensor(seq.completion_token_ids, device=logits.device)
                            if len(completion_tokens) > 0:
                                # Create token mask: mark tokens that appeared in completion
                                token_mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
                                token_mask[completion_tokens] = True
                                
                                # Apply standard repetition penalty formula (matching transformers implementation):
                                # For tokens in completion: if score < 0 then score * penalty, else score / penalty
                                penalty_scores = torch.where(
                                    logits[i] < 0,
                                    logits[i] * penalty,
                                    logits[i] / penalty
                                )
                                # Only apply penalty to tokens that appeared in completion
                                logits[i] = torch.where(token_mask, penalty_scores, logits[i])
                
                # Apply logits processor for constrained decoding (if any sequence has one)
                # Clone logits to avoid in-place update issues in inference mode
                logits = logits.clone()
                for i, seq in enumerate(seqs):
                    bias = self._get_logits_bias(seq, logits)
                    if bias is not None:
                        self._apply_logits_bias(logits[i], bias)
                for i, seq in enumerate(seqs):
                    if seq.logits_processor is not None:
                        # Create input_ids tensor for this sequence
                        seq_input_ids = torch.tensor([seq.token_ids], device=logits.device)
                        # Apply processor to this sequence's logits (clone to avoid inference mode issues)
                        processed = seq.logits_processor(seq_input_ids, logits[i:i+1].clone())
                        logits[i] = processed[0]

                # Prepare input_ids for sampler
                # seq_input_ids = torch.tensor([seq.token_ids for seq in seqs], device=logits.device)
                
                token_ids = self.sampler(
                    logits, 
                    temperatures,
                    top_ks=top_ks if top_ks is not None else None,
                    top_ps=top_ps if top_ps is not None else None,
                    repetition_penalties=None,  # Already applied above
                    generator=self._sampling_generator,
                    # input_ids=seq_input_ids,
                ).tolist()
                
                # Update logits processor state after sampling
                # NOTE: Only update for the first sequence since all sequences may share the same processor
                # (when using a single SamplingParams for batch generation)
                # Updating multiple times would cause duplicate state updates (e.g., codes_count += N instead of += 1)
                if seqs and seqs[0].logits_processor_update_state is not None:
                    seqs[0].logits_processor_update_state(token_ids[0])
                
                return token_ids
            else:
                return None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        cache_key = (config.max_model_len, config.max_num_seqs)
        model_device = torch.device("cuda") if torch.cuda.is_available() else self._get_model_device()
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            current_sig = self._get_graph_capture_signature()
            if cached.get("sig") == current_sig:
                self.graphs = cached["graphs"]
                self.graph_pool = cached["pool"]
                self.graph_vars = cached["vars"]
                self.graph_bs = cached["bs"]
                if cache_key in self._graph_cache_order:
                    self._graph_cache_order.remove(cache_key)
                self._graph_cache_order.append(cache_key)
                return
                self._drop_graph_cache_entry(cache_key)
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=model_device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=model_device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=model_device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=model_device)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_device)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, device=model_device)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        self._graph_cache[cache_key] = {
            "graphs": self.graphs,
            "pool": self.graph_pool,
            "vars": self.graph_vars,
            "bs": self.graph_bs,
            "sig": self._get_graph_capture_signature(),
        }
        if cache_key in self._graph_cache_order:
            self._graph_cache_order.remove(cache_key)
        self._graph_cache_order.append(cache_key)
        while len(self._graph_cache_order) > 5:
            old_key = self._graph_cache_order.pop(0)
            self._drop_graph_cache_entry(old_key)
