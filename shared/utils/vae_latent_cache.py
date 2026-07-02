"""
Apple Silicon-optimized cache for VAE latent tensors.

Speeds up repeat generations with the same reference/input images by
caching VAE-encoded latents on CPU (LRU eviction, configurable size).

Usage:
    vae_cache = VAELatentCache(max_size_mb=500)  # 500 MB cache
    latent = vae_cache.encode(
        encode_fn=lambda img: vae.encode([img], tile_size)[0],
        image_tensor=input_image,
        device=device,
    )
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

import torch


def _tensor_hash(t: torch.Tensor) -> str:
    """Fast deterministic hash of a tensor's content."""
    # Use first/last elements + shape + dtype + mean/std for collision resistance
    flat = t.detach().to("cpu").flatten()
    n = flat.numel()
    if n <= 16:
        sample = flat
    else:
        # Sample first 8, last 8 elements
        sample = torch.cat([flat[:8], flat[-8:]])
    # Include shape and mean for extra safety
    mean_val = float(flat.float().mean())
    std_val = float(flat.float().std()) if n > 1 else 0.0
    fingerprint = f"{sample.numpy().tobytes().hex()}_{t.shape}_{t.dtype}_{mean_val:.4f}_{std_val:.4f}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


@dataclass
class _VAECacheEntry:
    value: Any
    size_bytes: int


class VAELatentCache:
    """LRU cache for VAE-encoded latents stored on CPU.

    Designed for Apple Silicon's unified memory where CPU tensors are
    accessible by MPS without copy overhead. Cached latents are stored
    as detached CPU tensors and moved to the target device on retrieval.
    """

    def __init__(self, max_size_mb: float = 500) -> None:
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._entries: OrderedDict[Hashable, _VAECacheEntry] = OrderedDict()
        self._size_bytes = 0

    def encode(
        self,
        encode_fn: Callable[[Any], Any],
        image_tensor: torch.Tensor,
        device: torch.device | str | None = None,
        cache_key: Hashable | None = None,
    ) -> Any:
        """Encode image to VAE latent, using cache if available.

        Args:
            encode_fn: Function that takes image_tensor and returns latent.
            image_tensor: Input image tensor (B, C, H, W) or (C, H, W).
            device: Target device for the returned latent.
            cache_key: Optional custom key. Defaults to tensor hash.
        """
        if cache_key is None:
            cache_key = _tensor_hash(image_tensor)

        cached = self._entries.get(cache_key)
        if cached is not None:
            self._entries.move_to_end(cache_key)
            return self._to_device(cached.value, device)

        encoded = encode_fn(image_tensor)
        return self._store(cache_key, encoded, device)

    def _store(self, cache_key: Hashable, encoded: Any, device: torch.device | str | None) -> Any:
        cached_value = self._detach_to_cpu(encoded)
        size_bytes = self._estimate_size_bytes(cached_value)
        if size_bytes <= self.max_size_bytes:
            existing = self._entries.pop(cache_key, None)
            if existing is not None:
                self._size_bytes -= existing.size_bytes
            self._entries[cache_key] = _VAECacheEntry(cached_value, size_bytes)
            self._size_bytes += size_bytes
            self._purge_if_needed()
        return self._to_device(encoded, device)

    def _purge_if_needed(self) -> None:
        while self._entries and self._size_bytes > self.max_size_bytes:
            _, entry = self._entries.popitem(last=False)
            self._size_bytes -= entry.size_bytes

    @staticmethod
    def _estimate_size_bytes(value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(VAELatentCache._estimate_size_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(VAELatentCache._estimate_size_bytes(v) for v in value)
        return 0

    @staticmethod
    def _detach_to_cpu(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.device.type == "cpu":
                return value.detach()
            return value.detach().to("cpu")
        if isinstance(value, dict):
            return {k: VAELatentCache._detach_to_cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            items = [VAELatentCache._detach_to_cpu(v) for v in value]
            return type(value)(items)
        return value

    @staticmethod
    def _to_device(value: Any, device: torch.device | str | None) -> Any:
        if device is None:
            return value
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {k: VAELatentCache._to_device(v, device) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            items = [VAELatentCache._to_device(v, device) for v in value]
            return type(value)(items)
        return value
