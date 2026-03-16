from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass
class GraphKVIndexWriter:
    """Writes decode K/V values at the absolute token position provided by input_pos."""

    def write(self, k_cache: torch.Tensor, v_cache: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor, input_pos: torch.Tensor) -> None:
        index = input_pos.reshape(-1)[:1]
        if index.dtype != torch.long:
            index = index.to(dtype=torch.long)
        if index.device != k_cache.device:
            index = index.to(device=k_cache.device)
        k_cache[: k_val.shape[0]].index_copy_(1, index, k_val)
        v_cache[: v_val.shape[0]].index_copy_(1, index, v_val)


@dataclass
class GraphKVFlashState:
    """Marks a cache as using flash_attn_with_kvcache-compatible graph decode."""

    flash_kvcache: bool = True


class CUDAGraphRunner:
    """Captures and replays a fixed-shape CUDA callable."""

    def __init__(self, name: str, fn: Callable[..., object]) -> None:
        self._name = name
        self._fn = fn
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: list[torch.Tensor] = []
        self._output: object = None

    def _capture(self, *inputs: torch.Tensor) -> object:
        if not inputs:
            raise ValueError(f"[{self._name}] CUDAGraphRunner requires at least one tensor input.")
        for tensor in inputs:
            if not torch.is_tensor(tensor):
                raise TypeError(f"[{self._name}] CUDAGraphRunner accepts tensor inputs only.")
            if tensor.device.type != "cuda":
                raise ValueError(f"[{self._name}] CUDAGraphRunner requires CUDA tensors.")
        self._static_inputs = [tensor.detach().clone() for tensor in inputs]
        self._graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self._graph):
                self._output = self._fn(*self._static_inputs)
            # The capture pass records kernels but does not populate outputs for immediate use.
            # Replay once so the first caller receives valid results.
            self._graph.replay()
        except Exception:
            self.release()
            raise
        return self._output

    def _copy_inputs(self, *inputs: torch.Tensor) -> None:
        if len(inputs) != len(self._static_inputs):
            raise ValueError(f"[{self._name}] Expected {len(self._static_inputs)} inputs, got {len(inputs)}.")
        for static_tensor, tensor in zip(self._static_inputs, inputs):
            if tensor.shape != static_tensor.shape or tensor.dtype != static_tensor.dtype or tensor.device != static_tensor.device:
                raise ValueError(
                    f"[{self._name}] Static shape mismatch: expected {tuple(static_tensor.shape)} {static_tensor.dtype} on {static_tensor.device}, "
                    f"got {tuple(tensor.shape)} {tensor.dtype} on {tensor.device}."
                )
            static_tensor.copy_(tensor)

    def run(self, *inputs: torch.Tensor) -> object:
        if self._graph is None:
            return self._capture(*inputs)
        self._copy_inputs(*inputs)
        self._graph.replay()
        return self._output

    def release(self) -> None:
        self._graph = None
        self._static_inputs = []
        self._output = None


class AutoRegressiveCudaGraphKit:
    """Reusable helper to attach graph-safe KV writers and replay decode callables."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._runners: dict[str, CUDAGraphRunner] = {}
        self._attached_caches: list[object] = []
        self._attached_cache_ids: set[int] = set()

    def attach_attention_modules(self, attention_modules: Iterable[object], max_tokens: int, graph_mode: str = "index") -> int:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0.")
        if graph_mode not in ("index", "flash_kvcache"):
            raise ValueError(f"Unsupported graph_mode '{graph_mode}'.")
        final_capacity = 0
        for module in attention_modules:
            kv_cache = getattr(module, "kv_cache", None)
            if kv_cache is None:
                raise RuntimeError("KV cache must be initialized before CUDA graph attachment.")
            kv_cache.ensure_capacity(max_tokens)
            final_capacity = max(final_capacity, int(kv_cache.capacity))
            cache_id = id(kv_cache)
            if cache_id not in self._attached_cache_ids:
                if graph_mode == "flash_kvcache":
                    kv_cache.graph_state = GraphKVFlashState()
                else:
                    kv_cache.graph_state = GraphKVIndexWriter()
                self._attached_cache_ids.add(cache_id)
                self._attached_caches.append(kv_cache)
        return final_capacity

    def run(self, runner_name: str, fn: Callable[..., object], *inputs: torch.Tensor) -> object:
        runner = self._runners.get(runner_name)
        if runner is None:
            runner = CUDAGraphRunner(f"{self._name}:{runner_name}", fn)
            self._runners[runner_name] = runner
        return runner.run(*inputs)

    def release(self) -> None:
        for runner in self._runners.values():
            runner.release()
        self._runners.clear()
        for kv_cache in self._attached_caches:
            if hasattr(kv_cache, "graph_state"):
                kv_cache.graph_state = None
        self._attached_caches.clear()
        self._attached_cache_ids.clear()
