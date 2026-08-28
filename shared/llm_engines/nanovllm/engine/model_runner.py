import pickle
import gc
import math
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import sys

from ..config import Config
from .sequence import Sequence
from ..layers.sampler import Sampler, _REPETITION_INCREMENT_LIMIT, apply_sparse_repetition_penalty_
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

    _MAX_SPECULATIVE_DRAFT_TOKENS = 5

    def __init__(self, config: Config, rank: int, event: Event | list[Event], model_object=None, graph_pool_handle=None):
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
        self._repetition_token_cache = {}
        self._sampling_generator = None
        self._runtime_signature = None
        self._graph_pool_seed = graph_pool_handle
        self._guard_counts = {}
        self._guard_seen_details = set()
        self._speculative_drafts = {}
        self._speculative_pending = {}
        self.speculative_stats = self._new_speculative_stats()
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
            # In eager mode MMGP may move parameter storage between CPU/GPU after prefill.
            # That changes data_ptr() without invalidating the live KV cache or sequence state.
            # Resetting here drops the prompt context and makes legacy decode drift off-topic.
            if self.enforce_eager:
                return
            current_sig = self._get_graph_capture_signature()
            if current_sig == self._runtime_signature:
                return
            self._note_guard("runtime_reprepare_signature_change")
            self.reset_runtime_state()
        if self.model is None:
            raise RuntimeError("LLM model object is not available.")
        self._tie_word_embeddings_if_needed()
        self.allocate_kv_cache()
        self._prepare_model_sequence_state()
        if not self.enforce_eager:
            self.capture_cudagraph()
        self._runtime_ready = True
        self._runtime_signature = self._get_graph_capture_signature()

    def reset_generation_state(self):
        if self.model is None:
            return
        try:
            for module in self.model.modules():
                reset_sequence_state = getattr(module, "reset_sequence_state", None)
                if callable(reset_sequence_state):
                    reset_sequence_state()
                    continue
                if hasattr(module, "conv_state"):
                    module.conv_state = None
                if hasattr(module, "recurrent_state"):
                    module.recurrent_state = None
        except Exception:
            pass
        self._logits_bias_cache.clear()
        self._repetition_token_cache.clear()
        self._speculative_drafts.clear()
        self._speculative_pending.clear()
        self.speculative_stats = self._new_speculative_stats()
        reset_context()

    @classmethod
    def _new_speculative_stats(cls) -> dict:
        return {
            "drafted": 0,
            "accepted": 0,
            "target_passes": 0,
            "emitted_tokens": 0,
            "drafted_by_position": [0] * cls._MAX_SPECULATIVE_DRAFT_TOKENS,
            "accepted_by_position": [0] * cls._MAX_SPECULATIVE_DRAFT_TOKENS,
        }

    def _prepare_model_sequence_state(self):
        if self.model is None:
            return
        runtime_device = self._get_runtime_device()
        for module in self.model.modules():
            prepare = getattr(module, "prepare_sequence_state", None)
            if callable(prepare):
                prepare(self.config.max_num_seqs, runtime_device, self.dtype)
        mtp = getattr(self.model, "mtp", None)
        if mtp is not None and bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            mtp.prepare_cache(self.config.max_model_len, self._get_runtime_device(), self.dtype)
            self._prepare_target_speculative_state()

    def _prepare_target_speculative_state(self) -> None:
        for module in self.model.blk:
            prepare_speculative_state = getattr(module, "prepare_speculative_state", None)
            if callable(prepare_speculative_state):
                prepare_speculative_state(self._MAX_SPECULATIVE_DRAFT_TOKENS + 1)

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
                        module.k_scale = module.v_scale = torch.tensor([])
                    release_sequence_state = getattr(module, "release_sequence_state", None)
                    if callable(release_sequence_state):
                        release_sequence_state()
                        continue
                    reset_sequence_state = getattr(module, "reset_sequence_state", None)
                    if callable(reset_sequence_state):
                        reset_sequence_state()
                    else:
                        if hasattr(module, "conv_state"):
                            module.conv_state = None
                        if hasattr(module, "recurrent_state"):
                            module.recurrent_state = None
        except Exception:
            pass
        if hasattr(self, "kv_cache"):
            try:
                del self.kv_cache
            except Exception:
                pass
        if hasattr(self, "kv_cache_scales"):
            try:
                del self.kv_cache_scales
            except Exception:
                pass
        # CUDA graphs captured against previous model/KV pointers are unsafe after runtime reset.
        # Force recapture on next prepare to avoid stale-pointer illegal memory access.
        try:
            self.clear_graph_cache()
        except Exception:
            pass
        try:
            for attr_name in ("graphs", "graph_vars", "graph_bs", "graph_pool", "speculative_graphs", "speculative_graph_vars", "mtp_graph", "mtp_graph_vars", "mtp_draft_graphs", "mtp_draft_graph_vars"):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        self._logits_bias_cache.clear()
        self._repetition_token_cache.clear()
        self._speculative_drafts.clear()
        self._speculative_pending.clear()
        self._sampling_generator = None
        self._runtime_signature = None
        self._runtime_ready = False
        gc.collect()

    def _get_graph_capture_signature(self):
        # Note: this only samples the first parameter, so it can miss a partial address
        # reuse after an offloader (e.g. MMGP) evicts and reloads the model. Integrations
        # that interleave other models must explicitly call reset_runtime_state() (engine
        # release_runtime_allocations()) when their phase ends, like Chain-of-Zoom does.
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

    def _get_runtime_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return self._get_model_device()

    def _pin_memory_enabled(self) -> bool:
        return self._get_runtime_device().type == "cuda"

    def _to_runtime_device(self, tensor: torch.Tensor) -> torch.Tensor:
        device = self._get_runtime_device()
        return tensor.to(device=device, non_blocking=device.type == "cuda")

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
            entry.pop("speculative_graphs", None)
            entry.pop("speculative_vars", None)
            entry.pop("mtp_graph", None)
            entry.pop("mtp_vars", None)
            entry.pop("mtp_draft_graphs", None)
            entry.pop("mtp_draft_vars", None)
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
        if seed is None:
            self._sampling_generator = None
            return
        device = self._get_runtime_device()
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            self._sampling_generator = generator
        except Exception:
            self._sampling_generator = None

    @staticmethod
    def _apply_logits_bias(logits_row: torch.Tensor, bias: torch.Tensor):
        logits_row.add_(bias)

    def _allocate_sample_buffers(self):
        """Pre-allocate reusable buffers for sampling to avoid repeated tensor creation."""
        max_bs = self.config.max_num_seqs
        max_tokens = self.config.max_num_batched_tokens
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        pin_memory = self._pin_memory_enabled()
        
        # Pre-allocate pinned memory buffers on CPU for fast transfer
        # Must explicitly specify device="cpu" since default device may be "cuda"
        self._cpu_temperatures = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        self._cpu_cfg_scales = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        self._cpu_top_ks = torch.zeros(max_bs, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        self._cpu_top_ps = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        self._cpu_min_ps = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        self._cpu_repetition_penalties = torch.zeros(max_bs, dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        
        # Pre-allocate decode buffers on CPU with pinned memory
        self._cpu_input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cpu", pin_memory=pin_memory)
        self._cpu_positions = torch.zeros(max_bs, dtype=torch.int64, device="cpu", pin_memory=pin_memory)
        self._cpu_slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        self._cpu_context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        
        # Pre-allocate prefill buffers on CPU with pinned memory (optimization to avoid repeated tensor creation)
        self._cpu_prefill_input_ids = torch.zeros(max_tokens, dtype=torch.int64, device="cpu", pin_memory=pin_memory)
        self._cpu_prefill_positions = torch.zeros(max_tokens, dtype=torch.int64, device="cpu", pin_memory=pin_memory)
        self._cpu_prefill_cu_seqlens = torch.zeros(max_bs + 1, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        self._cpu_prefill_slot_mapping = torch.zeros(max_tokens, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        
        # Pre-allocate block tables buffer (shared by both decode and prefill)
        self._cpu_block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        
        # Pre-allocate buffer for sequence token IDs (used in logits processor and sampler)
        # Max length is max_model_len since sequences can be that long
        self._seq_token_ids_buffer = torch.zeros(max_bs, self.config.max_model_len, dtype=torch.int64, device="cpu", pin_memory=pin_memory)

    def _release_sample_buffers(self):
        buffer_names = [
            "_cpu_temperatures",
            "_cpu_cfg_scales",
            "_cpu_top_ks",
            "_cpu_top_ps",
            "_cpu_min_ps",
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
        self._repetition_token_cache.clear()
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
            if hasattr(self, "graph_vars"):
                del self.graph_vars
            if hasattr(self, "graph_bs"):
                del self.graph_bs
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
        try:
            torch.cuda.ipc_collect()
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

    def _get_kv_cache_modules(self):
        if self.model is None:
            return []
        return [module for module in self.model.modules() if hasattr(module, "k_cache") and hasattr(module, "v_cache") and not bool(getattr(module, "_exclude_paged_kv_cache", False))]

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        runtime_device = self._get_runtime_device()
        is_cuda_runtime = runtime_device.type == "cuda"
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        if config.kv_cache_int8 and head_dim % 32:
            raise RuntimeError(f"INT8 KV cache requires a head dimension divisible by 32; got {head_dim}.")
        kv_cache_modules = self._get_kv_cache_modules()
        kv_cache_layer_count = len(kv_cache_modules)
        cache_element_size = torch.empty((), dtype=torch.int8 if config.kv_cache_int8 else self.dtype).element_size()
        scale_elements = head_dim // 32 if config.kv_cache_int8 else 0
        scale_element_size = torch.float16.itemsize if config.kv_cache_int8 else 0
        block_bytes = 2 * kv_cache_layer_count * self.block_size * num_kv_heads * (head_dim * cache_element_size + scale_elements * scale_element_size)

        # Strict policy: allocate exactly the blocks required by requested runtime limits.
        required_blocks_per_seq = (config.max_model_len + self.block_size - 1) // self.block_size
        required_total_blocks = required_blocks_per_seq * max(1, config.max_num_seqs)
        config.num_kvcache_blocks = max(1, int(required_total_blocks))
        required_kv_bytes = config.num_kvcache_blocks * block_bytes
        if is_cuda_runtime:
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
                kv_cache_layer_count,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                device=runtime_device,
                dtype=torch.int8 if config.kv_cache_int8 else self.dtype,
            )
            if config.kv_cache_int8:
                self.kv_cache_scales = torch.zeros(
                    2,
                    kv_cache_layer_count,
                    config.num_kvcache_blocks,
                    self.block_size,
                    num_kv_heads,
                    scale_elements,
                    device=runtime_device,
                    dtype=torch.float16,
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                extra = ""
                if is_cuda_runtime:
                    free_now, total_now = torch.cuda.mem_get_info()
                    extra = f" Current free VRAM: {free_now / 1024**3:.2f} GB / {total_now / 1024**3:.2f} GB total."
                raise RuntimeError(
                    f"Failed to allocate strict KV cache ({required_kv_bytes / 1024**3:.2f} GB) for "
                    f"max_model_len={config.max_model_len}, max_num_seqs={config.max_num_seqs} on {runtime_device}.{extra}"
                ) from exc
            raise
        for layer_id, module in enumerate(kv_cache_modules):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            if config.kv_cache_int8:
                module.k_scale = self.kv_cache_scales[0, layer_id]
                module.v_scale = self.kv_cache_scales[1, layer_id]

    def prepare_block_tables(self, seqs: list[Sequence]):
        bs = len(seqs)
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = self._cpu_block_tables[:bs, :max_len]
        block_tables.fill_(-1)
        for row, seq in enumerate(seqs):
            if not seq.block_table:
                continue
            block_tables[row, :len(seq.block_table)] = torch.tensor(seq.block_table, dtype=torch.int32, device="cpu")
        return self._to_runtime_device(block_tables)

    def prepare_prefill(self, seqs: list[Sequence]):
        use_prompt_embeds = any(getattr(seq, "prompt_embeds", None) is not None for seq in seqs)
        if use_prompt_embeds and not all(getattr(seq, "prompt_embeds", None) is not None for seq in seqs):
            raise RuntimeError("Mixed embedded/non-embedded prefill batches are not supported.")
        input_ids = []
        positions = []
        prompt_embeds = []
        prompt_position_ids = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        has_previous_state = any(int(getattr(seq, "num_cached_tokens", 0) or 0) > 0 for seq in seqs)
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            if use_prompt_embeds:
                seq_prompt_embeds = getattr(seq, "prompt_embeds", None)
                seq_position_ids = getattr(seq, "prompt_position_ids", None)
                if seq_prompt_embeds is None or seq_position_ids is None:
                    raise RuntimeError("Embedded prefill requires both prompt_embeds and prompt_position_ids.")
                seq_prompt_embeds = seq_prompt_embeds[seq.num_cached_tokens:seqlen]
                if seq_prompt_embeds.ndim != 2 or seq_prompt_embeds.shape[0] != seqlen_q:
                    raise RuntimeError("Embedded prefill prompt_embeds shape does not match uncached prompt length.")
                if seq_position_ids.ndim == 3:
                    seq_position_ids = seq_position_ids[:, 0]
                if seq_position_ids.ndim != 2 or seq_position_ids.shape[0] != 3:
                    raise RuntimeError("Embedded prefill prompt_position_ids must have shape [3, seq_len].")
                seq_position_ids = seq_position_ids[:, seq.num_cached_tokens:seqlen]
                if seq_position_ids.shape[1] != seqlen_q:
                    raise RuntimeError("Embedded prefill prompt_position_ids shape does not match uncached prompt length.")
                prompt_embeds.append(seq_prompt_embeds)
                prompt_position_ids.append(seq_position_ids)
            else:
                positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup: no blocks allocated yet
                slot_mapping.extend([-1] * seqlen_q)
                continue
            cached_tokens = max(0, int(seq.num_cached_tokens or 0))
            cached_partial_tokens = cached_tokens % self.block_size
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i == seq.num_cached_blocks and cached_partial_tokens > 0:
                    start += cached_partial_tokens
                if i != seq.num_blocks - 1:
                    end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    end = seq.block_table[i] * self.block_size + seq.last_block_num_tokens
                if end > start:
                    slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        if use_prompt_embeds:
            input_ids = None
            positions = self._to_runtime_device(torch.cat(prompt_position_ids, dim=1).unsqueeze(1).contiguous())
            inputs_embeds = self._to_runtime_device(torch.cat(prompt_embeds, dim=0).unsqueeze(0).contiguous())
        else:
            pin_memory = self._pin_memory_enabled()
            input_ids = self._to_runtime_device(torch.tensor(input_ids, dtype=torch.int64, pin_memory=pin_memory))
            positions = self._to_runtime_device(torch.tensor(positions, dtype=torch.int64, pin_memory=pin_memory))
            inputs_embeds = None
        pin_memory = self._pin_memory_enabled()
        cu_seqlens_q = self._to_runtime_device(torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=pin_memory))
        cu_seqlens_k = self._to_runtime_device(torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=pin_memory))
        slot_mapping = self._to_runtime_device(torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=pin_memory))
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, has_previous_state=has_previous_state)
        return input_ids, positions, inputs_embeds

    @torch.inference_mode()
    def prefill_only(self, seqs: list[Sequence]) -> None:
        self.ensure_runtime_ready()
        input_ids, positions, inputs_embeds = self.prepare_prefill(seqs)
        self.run_model(input_ids, positions, True, inputs_embeds=inputs_embeds)
        for seq in seqs:
            seq.clear_prompt_data()
        reset_context()

    def prepare_decode(self, seqs: list[Sequence]):
        """Optimized decode preparation using pre-allocated buffers."""
        bs = len(seqs)
        
        # Use pre-allocated CPU buffers
        for i, seq in enumerate(seqs):
            self._cpu_input_ids[i] = seq.last_token
            self._cpu_positions[i] = len(seq) - 1 + int(getattr(seq, "position_offset", 0) or 0)
            self._cpu_context_lens[i] = len(seq)
            self._cpu_slot_mapping[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        
        # Transfer to the runtime device using sliced views
        input_ids = self._to_runtime_device(self._cpu_input_ids[:bs])
        positions = self._to_runtime_device(self._cpu_positions[:bs])
        slot_mapping = self._to_runtime_device(self._cpu_slot_mapping[:bs])
        context_lens = self._to_runtime_device(self._cpu_context_lens[:bs])
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
        min_ps_is_zero = True
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
            self._cpu_min_ps[i] = seq.min_p if seq.min_p is not None else 0.0
            if seq.min_p is not None and seq.min_p > 0.0:
                min_ps_is_zero = False
            self._cpu_repetition_penalties[i] = seq.repetition_penalty if seq.repetition_penalty is not None else 1.0
            if seq.repetition_penalty is not None and seq.repetition_penalty != 1.0:
                repetition_penalties_is_one = False
        
        # Transfer to the runtime device using sliced views (single batched transfer)
        temperatures = self._to_runtime_device(self._cpu_temperatures[:num_seqs])
        cfg_scales = self._to_runtime_device(self._cpu_cfg_scales[:num_seqs])
        top_ks = self._to_runtime_device(self._cpu_top_ks[:num_seqs]) if not top_ks_is_zero else None
        top_ps = self._to_runtime_device(self._cpu_top_ps[:num_seqs]) if not top_ps_is_one else None
        min_ps = self._to_runtime_device(self._cpu_min_ps[:num_seqs]) if not min_ps_is_zero else None
        repetition_penalties = self._cpu_repetition_penalties[:num_seqs] if not repetition_penalties_is_one else None
        
        return temperatures, cfg_scales, top_ks, top_ps, min_ps, repetition_penalties

    def _sync_repetition_token_cache(self, seq: Sequence) -> dict:
        boundary = int(seq.repetition_penalty_start)
        completion_count = int(seq.num_tokens) - boundary
        cache = self._repetition_token_cache.get(seq.seq_id)
        if cache is None or cache["boundary"] != boundary or cache["synced_tokens"] > completion_count:
            if cache is None and len(self._repetition_token_cache) >= max(1, int(self.config.max_num_seqs)) * 2:
                self._repetition_token_cache.pop(next(iter(self._repetition_token_cache)))
            if cache is None:
                cache = {"device_states": {}}
                self._repetition_token_cache[seq.seq_id] = cache
            cache.update(boundary=boundary, synced_tokens=0, seen=set(), token_ids=[])
            for state in cache["device_states"].values():
                state["count"] = 0
                state["cursor"] = 0
        new_tokens = seq.token_ids[boundary + cache["synced_tokens"]:boundary + completion_count]
        full_vocab_size = int(self.config.hf_config.vocab_size)
        for token_id in new_tokens:
            token_id = int(token_id)
            if 0 <= token_id < full_vocab_size and token_id not in cache["seen"]:
                cache["seen"].add(token_id)
                cache["token_ids"].append(token_id)
        cache["synced_tokens"] = completion_count
        return cache

    @staticmethod
    def _repetition_virtual_tokens(cache: dict, virtual_tokens, vocab_size: int) -> list[int]:
        extra_ids = []
        extra_seen = set()
        for token_id in virtual_tokens:
            token_id = int(token_id)
            if 0 <= token_id < vocab_size and token_id not in cache["seen"] and token_id not in extra_seen:
                extra_seen.add(token_id)
                extra_ids.append(token_id)
        return extra_ids

    def _apply_repetition_penalty(self, seq: Sequence, logits: torch.Tensor, penalty: float, virtual_tokens=()) -> None:
        if penalty == 1.0:
            return
        vocab_size = int(logits.shape[-1])
        cache = self._sync_repetition_token_cache(seq)
        virtual_ids = self._repetition_virtual_tokens(cache, virtual_tokens, vocab_size)
        device_key = (logits.device.type, logits.device.index, vocab_size)
        state = cache["device_states"].get(device_key)
        if state is None:
            capacity = int(self.config.max_model_len) + _REPETITION_INCREMENT_LIMIT
            state = {
                "indices": torch.empty(capacity, dtype=torch.long, device=logits.device),
                "values": torch.empty(capacity, dtype=logits.dtype, device=logits.device),
                "count": 0,
                "cursor": 0,
            }
            cache["device_states"][device_key] = state
        new_ids = [token_id for token_id in cache["token_ids"][state["cursor"]:] if token_id < vocab_size]
        state["cursor"] = len(cache["token_ids"])
        if len(new_ids) > _REPETITION_INCREMENT_LIMIT:
            valid_ids = [token_id for token_id in cache["token_ids"] if token_id < vocab_size]
            state["indices"][:len(valid_ids)].copy_(torch.tensor(valid_ids, dtype=torch.long, device=logits.device))
            state["count"] = len(valid_ids)
            new_ids = []
        apply_sparse_repetition_penalty_(logits, state["indices"], state["count"], new_ids, virtual_ids, penalty, state["values"])
        state["count"] += len(new_ids)

    @staticmethod
    def _speculative_logits_processor(seq: Sequence, predictive: bool):
        processor = seq.logits_processor
        if predictive and not seq.predictive_penalty and processor is not None and hasattr(processor, "_without_penalty"):
            processor = processor._without_penalty
        return processor

    def _apply_speculative_logit_rules(self, seq: Sequence, logits: torch.Tensor, repetition_penalty: float, virtual_tokens: list[int], vocab_size: int | None = None, predictive: bool = False) -> torch.Tensor:
        logits = logits.clone()
        logits_processor = self._speculative_logits_processor(seq, predictive)
        if predictive and not seq.predictive_penalty:
            repetition_penalty = 1.0
        if vocab_size is not None and logits_processor is not None:
            expanded_logits = logits.new_full((int(self.config.hf_config.vocab_size),), float("-inf"))
            expanded_logits[:vocab_size].copy_(logits)
            logits = expanded_logits
            vocab_size = None
        self._apply_repetition_penalty(seq, logits, repetition_penalty, virtual_tokens)
        bias = self._get_logits_bias(seq, logits.unsqueeze(0))
        if bias is not None:
            if vocab_size is not None:
                bias = bias[..., :vocab_size]
            self._apply_logits_bias(logits, bias)
        if logits_processor is not None:
            input_ids = torch.tensor([seq.token_ids + list(virtual_tokens)], dtype=torch.long, device=logits.device)
            logits = logits_processor(input_ids, logits.unsqueeze(0).clone())[0]
        return logits

    def _sample_speculative_target(self, seq: Sequence, logits: torch.Tensor, sample_params, virtual_tokens: list[int]) -> int:
        if seq.top_k == 1:
            repetition_penalties = sample_params[-1]
            penalty = 1.0 if repetition_penalties is None else float(repetition_penalties[0].item())
            logits = self._apply_speculative_logit_rules(seq, logits, penalty, virtual_tokens)
            token_id = int(torch.argmax(logits).item())
        else:
            token_id = self._sample_distribution(self._speculative_distribution(seq, logits, sample_params, virtual_tokens))
        if seq.logits_processor_update_state is not None:
            seq.logits_processor_update_state(token_id)
        return token_id

    def _speculative_distribution(self, seq: Sequence, logits: torch.Tensor, sample_params, virtual_tokens: list[int], vocab_size: int | None = None, predictive: bool = False) -> torch.Tensor:
        temperatures, _cfg_scales, _top_ks, _top_ps, _min_ps, repetition_penalties = sample_params
        penalty = 1.0 if repetition_penalties is None else float(repetition_penalties[0].item())
        logits = self._apply_speculative_logit_rules(seq, logits, penalty, virtual_tokens, vocab_size, predictive=predictive).float().div_(temperatures[0])
        top_k = int(seq.top_k) if seq.top_k is not None and 0 < int(seq.top_k) < logits.numel() else None
        top_p = float(seq.top_p) if seq.top_p is not None and 0.0 < float(seq.top_p) < 1.0 else None
        min_p = float(seq.min_p) if seq.min_p is not None and float(seq.min_p) > 0.0 else None
        if top_k is None and top_p is None and min_p is None:
            return torch.softmax(logits, dim=-1)
        if top_k is None:
            universe_logits = logits
            universe_ids = None
        else:
            threshold = torch.topk(logits, top_k).values[-1]
            universe_ids = torch.nonzero(logits >= threshold, as_tuple=False).flatten()
            universe_logits = logits[universe_ids]
        if min_p is None:
            candidate_logits = universe_logits
            candidate_ids = universe_ids
        else:
            min_p_mask = universe_logits >= universe_logits.max() + math.log(min_p)
            candidate_logits = universe_logits[min_p_mask]
            candidate_ids = torch.nonzero(min_p_mask, as_tuple=False).flatten() if universe_ids is None else universe_ids[min_p_mask]
        if top_p is not None:
            log_normalizer = torch.logsumexp(universe_logits, dim=0)
            excluded_mass = 1.0 - torch.exp(candidate_logits - log_normalizer).sum()
            order = torch.argsort(candidate_logits)
            candidate_logits = candidate_logits[order]
            candidate_ids = order if candidate_ids is None else candidate_ids[order]
            keep = excluded_mass + torch.exp(candidate_logits - log_normalizer).cumsum(dim=0) > 1.0 - top_p
            keep[-1] = True
            candidate_logits = candidate_logits[keep]
            candidate_ids = candidate_ids[keep]
        elif candidate_ids is None:
            return torch.softmax(candidate_logits, dim=-1)
        probabilities = torch.zeros_like(logits)
        probabilities[candidate_ids] = torch.softmax(candidate_logits, dim=-1)
        return probabilities

    def _sample_distribution(self, probabilities: torch.Tensor) -> int:
        noise = torch.empty_like(probabilities).exponential_(1, generator=self._sampling_generator).clamp_min_(1e-10)
        return int(probabilities.div(noise).argmax().item())

    def _commit_speculative_target_state(self, processed_tokens: int) -> None:
        for module in self.model.blk:
            commit_speculative_state = getattr(module, "commit_speculative_state", None)
            if callable(commit_speculative_state):
                commit_speculative_state(processed_tokens)

    def _store_speculative_pending(self, seq: Sequence, target_logits: torch.Tensor, hidden_states: torch.Tensor, positions: torch.Tensor) -> None:
        self._speculative_drafts.pop(seq.seq_id, None)
        self._speculative_pending[seq.seq_id] = {
            "target_logits": target_logits.detach().clone(),
            "hidden_states": hidden_states[:, -1:].detach().clone(),
            "positions": (positions[..., -1:] if positions.ndim == 3 else positions[-1:]).detach().clone(),
        }

    def _prime_mtp_context(self, seq: Sequence) -> None:
        self.model.mtp.reset_sequence_state()
        input_ids, positions, inputs_embeds = self.prepare_prefill([seq])
        hidden_states = self.model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)
        self._prepare_target_speculative_state()
        target_logits = self.model.compute_logits(hidden_states)[0]
        if len(seq) > 1:
            shifted_ids = torch.tensor(seq.token_ids[1:], dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            shifted_embeds = None if inputs_embeds is None else inputs_embeds[:, 1:]
            mtp_positions = positions[..., :-1] if positions.ndim == 3 else positions[:-1]
            self.model.mtp(shifted_ids, mtp_positions, hidden_states[:, :-1], inputs_embeds=shifted_embeds, compute_logits=False)
        self._store_speculative_pending(seq, target_logits, hidden_states, positions)
        seq.clear_prompt_data()
        reset_context()
        self.speculative_stats["target_passes"] += 1

    @torch.inference_mode()
    def prefill_mtp_only(self, seqs: list[Sequence]) -> None:
        self.ensure_runtime_ready()
        self._prime_mtp_context(seqs[0])

    @torch.inference_mode()
    def prefill_mtp_suffix(self, seqs: list[Sequence], previous_token_count: int) -> None:
        self.ensure_runtime_ready()
        seq = seqs[0]
        previous_token_count = int(previous_token_count)
        previous_pending = self._speculative_pending.pop(seq.seq_id, None)
        seq.num_cached_tokens = previous_token_count if previous_pending is not None else previous_token_count - 1
        input_ids, positions, inputs_embeds = self.prepare_prefill([seq])
        hidden_states = self.model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)
        target_logits = self.model.compute_logits(hidden_states)[0]
        suffix_ids = torch.tensor(seq.token_ids[previous_token_count:], dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        if previous_pending is None:
            mtp_hidden = hidden_states[:, :-1]
            mtp_positions = positions[:-1]
        else:
            mtp_hidden = torch.cat((previous_pending["hidden_states"], hidden_states[:, :-1]), dim=1)
            mtp_positions = torch.cat((previous_pending["positions"], positions[:-1]), dim=0)
        self.model.mtp(suffix_ids, mtp_positions, mtp_hidden, compute_logits=False)
        self._store_speculative_pending(seq, target_logits, hidden_states, positions)
        seq.clear_prompt_data()
        reset_context()
        self.speculative_stats["target_passes"] += 1

    def snapshot_speculative_state(self, seq_id: int) -> dict:
        pending = self._speculative_pending.get(seq_id)
        draft = self._speculative_drafts.get(seq_id)
        return {
            "mtp_cache": self.model.mtp.snapshot_sequence_state(),
            "draft": None if draft is None else {name: tensor.detach().to("cpu").as_subclass(torch.Tensor).clone() for name, tensor in draft.items()},
            "pending": None if pending is None else {name: tensor.detach().to("cpu").as_subclass(torch.Tensor).clone() for name, tensor in pending.items()},
        }

    def restore_speculative_state(self, seq_id: int, snapshot: dict) -> None:
        self.model.mtp.restore_sequence_state(snapshot["mtp_cache"])
        self._speculative_drafts.clear()
        self._speculative_pending.clear()
        device = self._get_runtime_device()
        draft = snapshot.get("draft")
        if draft is not None:
            self._speculative_drafts[seq_id] = {name: tensor.to(device=device) for name, tensor in draft.items()}
        pending = snapshot.get("pending")
        if pending is not None:
            self._speculative_pending[seq_id] = {name: tensor.to(device=device) for name, tensor in pending.items()}

    def _prepare_speculative_verify(self, seq: Sequence, draft_tokens: list[int] | torch.Tensor):
        verify_length = len(draft_tokens) + 1
        current_slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        if torch.is_tensor(draft_tokens):
            input_ids = torch.cat((torch.tensor([seq.last_token], dtype=torch.long, device=draft_tokens.device), draft_tokens))
        else:
            input_ids = torch.tensor([seq.last_token, *draft_tokens], dtype=torch.long, device=self._get_runtime_device())
        start_position = len(seq) - 1 + int(getattr(seq, "position_offset", 0) or 0)
        positions = torch.arange(start_position, start_position + verify_length, dtype=torch.long, device=self._get_runtime_device())
        slot_mapping = torch.arange(current_slot, current_slot + verify_length, dtype=torch.int32, device=self._get_runtime_device())
        cu_seqlens_q = torch.tensor([0, verify_length], dtype=torch.int32, device=self._get_runtime_device())
        cu_seqlens_k = torch.tensor([0, len(seq) + len(draft_tokens)], dtype=torch.int32, device=self._get_runtime_device())
        block_tables = self.prepare_block_tables([seq])
        set_context(True, cu_seqlens_q, cu_seqlens_k, verify_length, len(seq) + len(draft_tokens), slot_mapping, None, block_tables, has_previous_state=True, speculative_verify=True)
        return input_ids, positions

    def _run_mtp_forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor, inputs_embeds: torch.Tensor | None = None, compute_logits: bool = True, last_logits_only: bool = False):
        if self.enforce_eager or inputs_embeds is not None or input_ids.numel() != 1 or positions.numel() != 1 or getattr(self, "mtp_graph", None) is None:
            return self.model.mtp(input_ids, positions, hidden_states, inputs_embeds=inputs_embeds, compute_logits=compute_logits, last_logits_only=last_logits_only)
        graph_vars = self.mtp_graph_vars
        graph_vars["input_ids"].copy_(input_ids.reshape_as(graph_vars["input_ids"]))
        graph_vars["positions"].copy_(positions.reshape_as(graph_vars["positions"]))
        graph_vars["hidden_states"].copy_(hidden_states.reshape_as(graph_vars["hidden_states"]))
        self.model.mtp._cache.prepare_append()
        self.mtp_graph.replay()
        self.model.mtp._cache.advance(1)
        return graph_vars["outputs"], graph_vars["logits"]

    def _advance_mtp(self, seq: Sequence, token_ids: list[int], positions: torch.Tensor, hidden_states: torch.Tensor) -> None:
        mtp_input_ids = torch.tensor(token_ids, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        mtp_hidden, mtp_logits = self._run_mtp_forward(mtp_input_ids, positions, hidden_states, last_logits_only=True)
        self._speculative_drafts[seq.seq_id] = {"logits": mtp_logits[0, -1].clone(), "hidden_states": mtp_hidden[:, -1:].clone()}

    def _build_mtp_drafts(self, seq: Sequence, sample_params, draft_count: int, start_position: int) -> tuple[list[int] | torch.Tensor, int, list[torch.Tensor] | None]:
        draft_state = self._speculative_drafts[seq.seq_id]
        draft_vocab_size = getattr(self.model.mtp, "draft_vocab_size", None)
        cache_length = self.model.mtp.get_cache_length()
        rejection_sampling = seq.top_k != 1
        confidence_threshold = 0.0 if rejection_sampling else float(getattr(self.model, "_prompt_enhancer_speculative_confidence", 0.0))
        graph = getattr(self, "mtp_draft_graphs", {}).get(draft_count)
        predictive_processor = self._speculative_logits_processor(seq, predictive=True)
        predictive_repetition = sample_params[-1] is not None and seq.predictive_penalty
        if graph is not None and not rejection_sampling and confidence_threshold <= 0.0 and not predictive_repetition and predictive_processor is None:
            graph_vars = self.mtp_draft_graph_vars[draft_count]
            graph_vars["initial_logits"].copy_(draft_state["logits"])
            graph_vars["hidden_states"].copy_(draft_state["hidden_states"])
            graph_vars["positions"].copy_(torch.arange(start_position, start_position + draft_count - 1, dtype=torch.long, device=graph_vars["positions"].device))
            graph_vars["logits_bias"].zero_()
            bias = self._get_logits_bias(seq, draft_state["logits"].unsqueeze(0))
            if bias is not None:
                if draft_vocab_size is not None:
                    bias = bias[..., :draft_vocab_size]
                graph_vars["logits_bias"].copy_(bias)
            self.model.mtp._cache.prepare_append()
            graph.replay()
            self.model.mtp._cache.advance(draft_count - 1)
            return graph_vars["draft_tokens"], cache_length, None
        draft_tokens = []
        draft_distributions = [] if rejection_sampling else None
        draft_logits = draft_state["logits"]
        draft_hidden = draft_state["hidden_states"]
        for draft_idx in range(draft_count):
            if rejection_sampling:
                probabilities = self._speculative_distribution(seq, draft_logits, sample_params, draft_tokens, draft_vocab_size, predictive=True)
                draft_index = self._sample_distribution(probabilities)
                draft_distributions.append(probabilities)
            else:
                repetition_penalties = sample_params[-1]
                penalty = 1.0 if repetition_penalties is None else float(repetition_penalties[0].item())
                processed_logits = self._apply_speculative_logit_rules(seq, draft_logits, penalty, draft_tokens, draft_vocab_size, predictive=True)
                top_logit, top_token = torch.max(processed_logits, dim=0)
                if confidence_threshold > 0.0:
                    confidence = torch.exp(top_logit.float() - torch.logsumexp(processed_logits.float(), dim=0))
                    draft_index, confidence = torch.stack((top_token.float(), confidence)).tolist()
                    if confidence < confidence_threshold:
                        break
                else:
                    draft_index = int(top_token.item())
            draft_token = int(draft_index)
            draft_tokens.append(draft_token)
            if draft_idx + 1 == draft_count:
                break
            draft_position = torch.tensor([start_position + draft_idx], dtype=torch.long, device=draft_hidden.device)
            mtp_input_ids = torch.tensor([[draft_token]], dtype=torch.long, device=draft_hidden.device)
            draft_hidden, draft_logits = self._run_mtp_forward(mtp_input_ids, draft_position, draft_hidden, last_logits_only=True)
            draft_logits = draft_logits[0, -1]
        return draft_tokens, cache_length, draft_distributions

    def _run_native_mtp_target_token(self, seq: Sequence, sample_params) -> list[list[int]]:
        input_ids, positions = self.prepare_decode([seq])
        hidden_states = self._run_decode_hidden(input_ids, positions)
        target_token = self._sample_speculative_target(seq, self.model.compute_logits(hidden_states)[0], sample_params, [])
        self._advance_mtp(seq, [target_token], positions, hidden_states)
        reset_context()
        self.speculative_stats["target_passes"] += 1
        self.speculative_stats["emitted_tokens"] += 1
        return [[target_token]]

    def _emit_pending_mtp_token(self, seq: Sequence, sample_params) -> list[list[int]]:
        pending = self._speculative_pending.pop(seq.seq_id)
        sampled_token = self._sample_speculative_target(seq, pending["target_logits"], sample_params, [])
        self._advance_mtp(seq, [sampled_token], pending["positions"], pending["hidden_states"])
        self.speculative_stats["emitted_tokens"] += 1
        return [[sampled_token]]

    @torch.inference_mode()
    def _run_native_mtp(self, seq: Sequence, is_prefill: bool) -> list[list[int]]:
        if not self.model._prompt_enhancer_speculative_method_logged:
            sampling_method = "rejection sampling" if seq.top_k != 1 else "greedy exact-match"
            execution = "eager" if self.enforce_eager else "CUDA graph"
            draft_tokens = self.model._prompt_enhancer_speculative_sampling_tokens if seq.top_k != 1 else self.model._prompt_enhancer_speculative_tokens
            print(f"[Deepy][Speculative] method=native MTP ({sampling_method}, up to {draft_tokens} draft tokens, {execution} verification).")
            self.model._prompt_enhancer_speculative_method_logged = True
        sample_params = self.prepare_sample([seq], is_cfg_batch=False)
        if is_prefill:
            self._prime_mtp_context(seq)
        if seq.seq_id in self._speculative_pending:
            return self._emit_pending_mtp_token(seq, sample_params)

        max_emission = int(getattr(seq, "speculative_max_emission", seq.max_tokens - seq.num_completion_tokens))
        remaining_tokens = min(seq.max_tokens - seq.num_completion_tokens, max_emission)
        max_drafts = min(self._MAX_SPECULATIVE_DRAFT_TOKENS, max(1, int(getattr(self.model, "_prompt_enhancer_speculative_tokens", 1))))
        if seq.top_k != 1:
            max_drafts = min(max_drafts, max(1, int(getattr(self.model, "_prompt_enhancer_speculative_sampling_tokens", max_drafts))))
        draft_count = min(max_drafts, remaining_tokens - 1, self.block_size - seq.last_block_num_tokens - 1)
        if draft_count < 1:
            return self._run_native_mtp_target_token(seq, sample_params)

        start_position = len(seq) - 1 + int(getattr(seq, "position_offset", 0) or 0)
        draft_tokens, mtp_cache_length, draft_distributions = self._build_mtp_drafts(seq, sample_params, draft_count, start_position)
        if len(draft_tokens) == 0:
            return self._run_native_mtp_target_token(seq, sample_params)
        input_ids, positions = self._prepare_speculative_verify(seq, draft_tokens)
        hidden_states = self._run_speculative_verify_graph(input_ids, positions)
        logits = self.model.output(hidden_states)[0]
        reset_context()
        self.speculative_stats["target_passes"] += 1
        greedy_target_tokens = None
        if seq.top_k == 1 and sample_params[-1] is None and seq.logits_processor is None:
            bias = self._get_logits_bias(seq, logits)
            if bias is not None:
                logits.add_(bias)
            greedy_target_tokens = torch.argmax(logits, dim=-1).tolist()
        emitted = []
        accepted_count = 0
        speculative_stop_token_ids = set(getattr(seq, "speculative_stop_token_ids", ()))
        stopped = False
        draft_token_ids = draft_tokens.tolist() if torch.is_tensor(draft_tokens) else draft_tokens
        for draft_idx, draft_token in enumerate(draft_token_ids):
            if draft_distributions is not None:
                target_distribution = self._speculative_distribution(seq, logits[draft_idx], sample_params, emitted)
                draft_distribution = draft_distributions[draft_idx]
                acceptance = torch.clamp(target_distribution[draft_token] / draft_distribution[draft_token], max=1.0)
                accepted = bool((torch.rand((), device=logits.device, generator=self._sampling_generator) < acceptance).item())
                if accepted:
                    target_token = draft_token
                else:
                    residual = target_distribution.clone()
                    residual[:draft_distribution.numel()].sub_(draft_distribution)
                    target_token = self._sample_distribution(residual.clamp_min_(0))
                if seq.logits_processor_update_state is not None:
                    seq.logits_processor_update_state(target_token)
            else:
                target_token = greedy_target_tokens[draft_idx] if greedy_target_tokens is not None else self._sample_speculative_target(seq, logits[draft_idx], sample_params, emitted)
            emitted.append(target_token)
            self.speculative_stats["drafted"] += 1
            self.speculative_stats["drafted_by_position"][draft_idx] += 1
            if target_token != draft_token:
                break
            self.speculative_stats["accepted"] += 1
            self.speculative_stats["accepted_by_position"][draft_idx] += 1
            accepted_count += 1
            stopped = ((not seq.ignore_eos and target_token == self.config.eos) or target_token in speculative_stop_token_ids)
            if stopped:
                break
        if accepted_count == len(draft_token_ids) and not stopped:
            emitted.append(greedy_target_tokens[len(draft_token_ids)] if greedy_target_tokens is not None else self._sample_speculative_target(seq, logits[len(draft_token_ids)], sample_params, emitted))
        self._commit_speculative_target_state(len(emitted))
        if bool(getattr(self.model, "_prompt_enhancer_reuse_speculative_mtp_cache", False)) and not stopped:
            if accepted_count < len(draft_token_ids):
                self.model.mtp.truncate_cache(mtp_cache_length + accepted_count)
                commit_start = accepted_count
            else:
                self.model.mtp.truncate_cache(mtp_cache_length + len(draft_token_ids) - 1)
                commit_start = len(draft_token_ids) - 1
        else:
            self.model.mtp.truncate_cache(mtp_cache_length)
            commit_start = 0
        mtp_positions = positions[..., commit_start:len(emitted)] if positions.ndim == 3 else positions[commit_start:len(emitted)]
        self._advance_mtp(seq, emitted[commit_start:], mtp_positions, hidden_states[:, commit_start:len(emitted)])
        self.speculative_stats["emitted_tokens"] += len(emitted)
        return [emitted]

    def _replay_decode_graph_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bs = input_ids.size(0)
        context = get_context()
        graph_bs = next((x for x in self.graph_bs if x >= bs), None)
        if graph_bs is None:
            raise RuntimeError(f"CUDA graph batch size {bs} exceeds captured graph sizes {self.graph_bs} (max_num_seqs={self.config.max_num_seqs})")
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:graph_bs].zero_()
        graph_vars["positions"][:graph_bs].zero_()
        graph_vars["slot_mapping"][:graph_bs].fill_(self.config.num_kvcache_blocks * self.block_size - 1)
        graph_vars["context_lens"][:graph_bs].zero_()
        graph_vars["block_tables"][:graph_bs].fill_(-1)
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        self.graphs[graph_bs].replay()
        return graph_vars["outputs"][:bs]

    def _run_decode_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.enforce_eager:
            return self.model(input_ids=input_ids, positions=positions)
        hidden_states = self._replay_decode_graph_hidden(input_ids, positions)
        return hidden_states.unsqueeze(1) if hidden_states.ndim == 2 else hidden_states

    def _run_speculative_verify_graph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.enforce_eager:
            return self.model(input_ids=input_ids, positions=positions)
        context = get_context()
        verify_length = int(input_ids.numel())
        graph_vars = self.speculative_graph_vars[verify_length]
        graph_vars["input_ids"].copy_(input_ids)
        graph_vars["positions"].copy_(positions)
        graph_vars["slot_mapping"].copy_(context.slot_mapping)
        graph_vars["cu_seqlens_k"].copy_(context.cu_seqlens_k)
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][:, :context.block_tables.size(1)].copy_(context.block_tables)
        self.speculative_graphs[verify_length].replay()
        return graph_vars["outputs"]

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor | None, positions: torch.Tensor, is_prefill: bool, inputs_embeds: torch.Tensor | None = None):
        decode_batch_size = input_ids.size(0) if input_ids is not None else int(inputs_embeds.shape[0])
        model_kwargs = {"input_ids": input_ids, "positions": positions}
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        if is_prefill or self.enforce_eager or decode_batch_size > 512:
            return self.model.compute_logits(self.model(**model_kwargs))
        else:
            bs = decode_batch_size
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
                return self.model.compute_logits(self.model(**model_kwargs))
            
            # Fix: Also check if block_tables row count matches batch size
            # Dimension mismatch can cause CUDA illegal memory access during graph replay
            if context.block_tables.size(0) != bs:
                # Fall back to eager mode when block_tables row count doesn't match batch size
                self._note_guard(
                    "cudagraph_fallback_block_table_rows",
                    f"rows={context.block_tables.size(0)} bs={bs}",
                )
                return self.model.compute_logits(self.model(**model_kwargs))
            
            # Fix: Verify slot_mapping and context_lens dimensions match batch size
            if context.slot_mapping.size(0) != bs or context.context_lens.size(0) != bs:
                # Fall back to eager mode when dimensions don't match
                self._note_guard(
                    "cudagraph_fallback_context_shape",
                    f"slot={context.slot_mapping.size(0)} ctx={context.context_lens.size(0)} bs={bs}",
                )
                return self.model.compute_logits(self.model(**model_kwargs))
            
            return self.model.compute_logits(self._replay_decode_graph_hidden(input_ids, positions))

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run model forward and sampling. For CFG sequences, batch is structured as:
        [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
        where uncond_seqi is the paired unconditional sequence of cond_seqi."""
        self.ensure_runtime_ready()
        # Check if this is a CFG batch (contains paired conditional and unconditional sequences)
        is_cfg_batch = seqs[0].cfg_scale > 1.0 and seqs[0].paired_seq is not None
        if not is_cfg_batch and len(seqs) == 1 and getattr(self.model, "mtp", None) is not None and bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            return self._run_native_mtp(seqs[0], is_prefill)
        if is_cfg_batch:
            # CFG batch: seqs = [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
            num_cond = len(seqs) // 2
            cond_seqs = seqs[:num_cond]
            # uncond_seqs = seqs[num_cond:]
            
            # Prepare inputs for both conditional and unconditional (they're already in the batch)
            if is_prefill:
                input_ids, positions, inputs_embeds = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)
                inputs_embeds = None
            sample_params = self.prepare_sample(seqs, is_cfg_batch=True) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, min_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = min_ps = repetition_penalties = None
            
            # Run model forward (processes entire batch: cond + uncond)
            logits_all = self.run_model(input_ids, positions, is_prefill, inputs_embeds=inputs_embeds)
            if is_prefill:
                for seq in seqs:
                    seq.clear_prompt_data()
            reset_context()
            
            if self.rank == 0:
                # Split logits: first half is conditional, second half is unconditional
                logits_cond = logits_all[:num_cond].clone()
                logits_uncond = logits_all[num_cond:]
                
                # Apply repetition penalty to conditional logits (before CFG)
                if repetition_penalties is not None:
                    for i, seq in enumerate(cond_seqs):
                        penalty = repetition_penalties[i].item()
                        self._apply_repetition_penalty(seq, logits_cond[i], float(penalty))
                
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
                    min_ps=min_ps if min_ps is not None else None,
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
            if is_prefill:
                input_ids, positions, inputs_embeds = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)
                inputs_embeds = None
            sample_params = self.prepare_sample(seqs, is_cfg_batch=False) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, min_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = min_ps = repetition_penalties = None
            logits = self.run_model(input_ids, positions, is_prefill, inputs_embeds=inputs_embeds)
            if is_prefill:
                for seq in seqs:
                    seq.clear_prompt_data()
            reset_context()
            
            if self.rank == 0:
                logits = logits.clone()
                # Apply repetition penalty to logits
                if repetition_penalties is not None:
                    for i, seq in enumerate(seqs):
                        penalty = repetition_penalties[i].item()
                        self._apply_repetition_penalty(seq, logits[i], float(penalty))
                
                # Apply logits processor for constrained decoding (if any sequence has one)
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
                    min_ps=min_ps if min_ps is not None else None,
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
        if self._get_runtime_device().type != "cuda":
            self.enforce_eager = True
            return
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
                self.speculative_graphs = cached.get("speculative_graphs", {})
                self.speculative_graph_vars = cached.get("speculative_vars", {})
                self.mtp_graph = cached.get("mtp_graph")
                self.mtp_graph_vars = cached.get("mtp_vars", {})
                self.mtp_draft_graphs = cached.get("mtp_draft_graphs", {})
                self.mtp_draft_graph_vars = cached.get("mtp_draft_vars", {})
                if cache_key in self._graph_cache_order:
                    self._graph_cache_order.remove(cache_key)
                self._graph_cache_order.append(cache_key)
                return
            # signature mismatch: free the stale graphs before capturing new ones
            self._drop_graph_cache_entry(cache_key)
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=model_device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=model_device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=model_device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=model_device)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_device)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, device=model_device, dtype=self.dtype)
        base_graph_bs = [1, 2, 4, 8]
        self.graph_bs = [bs for bs in base_graph_bs if bs <= max_bs]
        if max_bs > 8:
            self.graph_bs.extend(range(16, max_bs + 1, 16))
        if max_bs not in self.graph_bs:
            self.graph_bs.append(max_bs)
            self.graph_bs.sort()
        if not self.graph_bs:
            self.graph_bs = [max_bs]
        self.graphs = {}
        self.graph_pool = self._graph_pool_seed

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
        self.speculative_graphs = {}
        self.speculative_graph_vars = {}
        self.mtp_graph = None
        self.mtp_graph_vars = {}
        self.mtp_draft_graphs = {}
        self.mtp_draft_graph_vars = {}
        if getattr(self.model, "mtp", None) is not None and bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            dummy_block = config.num_kvcache_blocks - 1
            for verify_length in reversed(range(2, self._MAX_SPECULATIVE_DRAFT_TOKENS + 2)):
                speculative_input_ids = torch.zeros(verify_length, dtype=torch.int64, device=model_device)
                speculative_positions = torch.arange(verify_length, dtype=torch.int64, device=model_device)
                speculative_slot_mapping = torch.arange(dummy_block * self.block_size, dummy_block * self.block_size + verify_length, dtype=torch.int32, device=model_device)
                speculative_cu_seqlens_q = torch.tensor([0, verify_length], dtype=torch.int32, device=model_device)
                speculative_cu_seqlens_k = torch.tensor([0, verify_length], dtype=torch.int32, device=model_device)
                speculative_block_tables = torch.full((1, max_num_blocks), -1, dtype=torch.int32, device=model_device)
                speculative_block_tables[0, 0] = dummy_block
                speculative_outputs = torch.zeros(1, verify_length, hf_config.hidden_size, device=model_device, dtype=self.dtype)
                graph = torch.cuda.CUDAGraph()
                set_context(True, speculative_cu_seqlens_q, speculative_cu_seqlens_k, verify_length, config.max_model_len, speculative_slot_mapping, None, speculative_block_tables, has_previous_state=True, speculative_verify=True)
                speculative_outputs[:] = self.model(speculative_input_ids, speculative_positions)
                with torch.cuda.graph(graph, self.graph_pool):
                    speculative_outputs[:] = self.model(speculative_input_ids, speculative_positions)
                torch.cuda.synchronize()
                reset_context()
                self.speculative_graphs[verify_length] = graph
                self.speculative_graph_vars[verify_length] = {
                    "input_ids": speculative_input_ids,
                    "positions": speculative_positions,
                    "slot_mapping": speculative_slot_mapping,
                    "cu_seqlens_k": speculative_cu_seqlens_k,
                    "block_tables": speculative_block_tables,
                    "outputs": speculative_outputs,
                }
            mtp_input_ids = torch.zeros((1, 1), dtype=torch.int64, device=model_device)
            mtp_positions = torch.zeros(1, dtype=torch.int64, device=model_device)
            mtp_hidden_states = torch.zeros((1, 1, hf_config.hidden_size), dtype=self.dtype, device=model_device)
            self.model.mtp._cache.cache_seqlens.zero_()
            mtp_outputs, mtp_logits = self.model.mtp(mtp_input_ids, mtp_positions, mtp_hidden_states, last_logits_only=True, cache_prepared=True)
            mtp_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(mtp_graph, self.graph_pool):
                mtp_outputs, mtp_logits = self.model.mtp(mtp_input_ids, mtp_positions, mtp_hidden_states, last_logits_only=True, cache_prepared=True)
            torch.cuda.synchronize()
            self.mtp_graph = mtp_graph
            self.mtp_graph_vars = {
                "input_ids": mtp_input_ids,
                "positions": mtp_positions,
                "hidden_states": mtp_hidden_states,
                "outputs": mtp_outputs,
                "logits": mtp_logits,
            }
            for draft_count in range(2, self._MAX_SPECULATIVE_DRAFT_TOKENS + 1):
                mtp_initial_logits = torch.zeros(getattr(self.model.mtp, "draft_vocab_size", hf_config.vocab_size), dtype=self.dtype, device=model_device)
                mtp_logits_bias = torch.zeros_like(mtp_initial_logits)
                mtp_draft_hidden = torch.zeros((1, 1, hf_config.hidden_size), dtype=self.dtype, device=model_device)
                mtp_draft_positions = torch.arange(draft_count - 1, dtype=torch.int64, device=model_device)
                mtp_draft_tokens = torch.zeros(draft_count, dtype=torch.int64, device=model_device)

                def run_mtp_draft_chain():
                    first_draft = torch.argmax(mtp_initial_logits + mtp_logits_bias).reshape(1)
                    mtp_draft_tokens[:1].copy_(first_draft)
                    current_hidden = mtp_draft_hidden
                    for draft_index in range(draft_count - 1):
                        current_hidden, current_logits = self.model.mtp(mtp_draft_tokens[draft_index:draft_index + 1].view(1, 1), mtp_draft_positions[draft_index:draft_index + 1], current_hidden, last_logits_only=True, cache_prepared=True)
                        self.model.mtp._cache.cache_seqlens.add_(1)
                        next_draft = torch.argmax(current_logits[0, -1] + mtp_logits_bias).reshape(1)
                        mtp_draft_tokens[draft_index + 1:draft_index + 2].copy_(next_draft)

                self.model.mtp._cache.cache_seqlens.zero_()
                run_mtp_draft_chain()
                self.model.mtp._cache.cache_seqlens.zero_()
                draft_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(draft_graph, self.graph_pool):
                    run_mtp_draft_chain()
                torch.cuda.synchronize()
                self.mtp_draft_graphs[draft_count] = draft_graph
                self.mtp_draft_graph_vars[draft_count] = {
                    "initial_logits": mtp_initial_logits,
                    "logits_bias": mtp_logits_bias,
                    "hidden_states": mtp_draft_hidden,
                    "positions": mtp_draft_positions,
                    "draft_tokens": mtp_draft_tokens,
                }
            self.model.mtp._cache.cache_seqlens.zero_()
        self._graph_cache[cache_key] = {
            "graphs": self.graphs,
            "pool": self.graph_pool,
            "vars": self.graph_vars,
            "bs": self.graph_bs,
            "speculative_graphs": self.speculative_graphs,
            "speculative_vars": self.speculative_graph_vars,
            "mtp_graph": self.mtp_graph,
            "mtp_vars": self.mtp_graph_vars,
            "mtp_draft_graphs": self.mtp_draft_graphs,
            "mtp_draft_vars": self.mtp_draft_graph_vars,
            "sig": self._get_graph_capture_signature(),
        }
        if cache_key in self._graph_cache_order:
            self._graph_cache_order.remove(cache_key)
        self._graph_cache_order.append(cache_key)
        while len(self._graph_cache_order) > 5:
            old_key = self._graph_cache_order.pop(0)
            self._drop_graph_cache_entry(old_key)
