"""Split parsing helpers for LoRA payloads."""

from __future__ import annotations

from typing import Any

from source.core.params.lora import LoRAConfig, LoRAEntry, LoRAStatus


def entries_from_params(params: dict[str, Any]) -> list[LoRAEntry]:
    return LoRAConfig.from_params(params).entries


def merge_additional_loras(
    entries: list[LoRAEntry],
    additional_loras: dict[str, Any],
) -> list[LoRAEntry]:
    merged = LoRAConfig(entries=list(entries)).merge(
        LoRAConfig.from_params({"additional_loras": additional_loras})
    )
    # Preserve pending state if the additional URL upgrades a local filename match.
    for entry in merged.entries:
        if entry.url and entry.status == LoRAStatus.LOCAL:
            entry.status = LoRAStatus.PENDING
    return merged.entries


__all__ = ["entries_from_params", "merge_additional_loras"]
