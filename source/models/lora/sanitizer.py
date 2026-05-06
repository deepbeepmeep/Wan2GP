"""Manifest-backed LoRA payload normalization and architecture checks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlsplit, urlunsplit

from source.models.lora.module_manifest import (
    LoRAModuleManifest,
    LoRAModuleManifestError,
    load_module_manifest,
)


LORA_EXTENSIONS = (".safetensors", ".pt", ".pth", ".bin")


@dataclass(frozen=True)
class LoRASanitizeDecision:
    original: str
    normalized: str | None
    filename: str | None
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class LoRASanitizeResult:
    loras: list[str] = field(default_factory=list)
    multipliers: list[Any] = field(default_factory=list)
    entries: list[Any] = field(default_factory=list)
    decisions: list[LoRASanitizeDecision] = field(default_factory=list)
    manifest: LoRAModuleManifest | None = None


def architecture_family(architecture: str | None) -> str | None:
    text = (architecture or "").lower()
    if not text:
        return None
    if "qwen" in text:
        return "qwen"
    if "ltx" in text or "ltxv" in text:
        return "ltx"
    if "wan" in text:
        return "wan"
    return None


def infer_lora_family(reference: str | None) -> str | None:
    text = (reference or "").lower()
    if not text:
        return None
    if "qwen" in text:
        return "qwen"
    if "ltx" in text or "ltxv" in text or "ic-lora" in text or "iclora" in text:
        return "ltx"
    if "wan" in text or "svi" in text or "lightx2v" in text or "vace" in text or "i2v-a14b" in text:
        return "wan"
    return None


def _load_manifest_for_architecture(architecture: str | None) -> LoRAModuleManifest | None:
    if not architecture_family(architecture):
        return None
    return load_module_manifest(str(architecture))


def _basename_from_url(url: str) -> str:
    path = urlsplit(url).path
    return unquote(Path(path).name)


def normalize_lora_reference(value: Any) -> tuple[str | None, str | None, str | None]:
    if not isinstance(value, str):
        return None, None, "malformed_non_string"

    raw = value.strip()
    if not raw:
        return None, None, "malformed_empty"

    if raw.startswith(("http://", "https://")):
        parts = urlsplit(raw)
        path = unquote(parts.path)
        if "/blob/" in path:
            path = path.replace("/blob/", "/resolve/", 1)
        filename = Path(path).name
        if not filename or not filename.lower().endswith(LORA_EXTENSIONS):
            return None, None, "malformed_extension"
        normalized = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
        return normalized, filename, None

    filename = os.path.basename(raw)
    if not filename or not filename.lower().endswith(LORA_EXTENSIONS):
        return None, None, "malformed_extension"
    return raw, filename, None


def _entry_reference(entry: Any) -> str | None:
    return (
        getattr(entry, "url", None)
        or getattr(entry, "local_path", None)
        or getattr(entry, "filename", None)
    )


def _decision_for_reference(
    reference: Any,
    *,
    architecture: str | None,
    seen: set[str],
) -> LoRASanitizeDecision:
    normalized, filename, reason = normalize_lora_reference(reference)
    original = str(reference) if reference is not None else ""
    if reason:
        return LoRASanitizeDecision(original, normalized, filename, False, reason)

    assert normalized is not None
    assert filename is not None

    dedupe_key = normalized.lower() if normalized.startswith(("http://", "https://")) else filename.lower()
    if dedupe_key in seen:
        return LoRASanitizeDecision(original, normalized, filename, False, "duplicate")

    arch_family = architecture_family(architecture)
    lora_family = infer_lora_family(normalized) or infer_lora_family(filename)
    if arch_family and lora_family and arch_family != lora_family:
        return LoRASanitizeDecision(
            original,
            normalized,
            filename,
            False,
            f"architecture_mismatch:{lora_family}_lora_for_{arch_family}_model",
        )

    seen.add(dedupe_key)
    return LoRASanitizeDecision(original, normalized, filename, True)


def sanitize_lora_entries(
    entries: Iterable[Any],
    *,
    architecture: str | None = None,
    task_id: str | None = None,
) -> LoRASanitizeResult:
    entry_list = list(entries)
    manifest = _load_manifest_for_architecture(architecture) if entry_list else None
    seen: set[str] = set()
    accepted: list[Any] = []
    decisions: list[LoRASanitizeDecision] = []

    for entry in entry_list:
        decision = _decision_for_reference(_entry_reference(entry), architecture=architecture, seen=seen)
        decisions.append(decision)
        if not decision.accepted:
            continue

        if getattr(entry, "url", None):
            entry.url = decision.normalized
            entry.filename = decision.filename
        elif getattr(entry, "local_path", None):
            entry.local_path = decision.normalized
            entry.filename = decision.filename
        else:
            entry.filename = decision.normalized
        accepted.append(entry)

    return LoRASanitizeResult(entries=accepted, decisions=decisions, manifest=manifest)


def sanitize_lora_values(
    loras: Sequence[Any],
    multipliers: Sequence[Any] | None = None,
    *,
    architecture: str | None = None,
    task_id: str | None = None,
) -> LoRASanitizeResult:
    lora_list = list(loras or [])
    mult_list = list(multipliers or [])
    manifest = _load_manifest_for_architecture(architecture) if lora_list else None
    seen: set[str] = set()
    accepted_loras: list[str] = []
    accepted_multipliers: list[Any] = []
    decisions: list[LoRASanitizeDecision] = []

    for index, value in enumerate(lora_list):
        decision = _decision_for_reference(value, architecture=architecture, seen=seen)
        decisions.append(decision)
        if not decision.accepted:
            continue
        assert decision.normalized is not None
        accepted_loras.append(decision.normalized)
        accepted_multipliers.append(mult_list[index] if index < len(mult_list) else 1.0)

    return LoRASanitizeResult(
        loras=accepted_loras,
        multipliers=accepted_multipliers,
        decisions=decisions,
        manifest=manifest,
    )


def sanitize_lora_payload(
    params: dict[str, Any],
    *,
    architecture: str | None = None,
    task_id: str | None = None,
) -> LoRASanitizeResult:
    loras = params.get("activated_loras") or params.get("lora_names") or []
    if isinstance(loras, str):
        loras = [item.strip() for item in loras.split(",") if item.strip()]

    multipliers = params.get("loras_multipliers") or params.get("lora_multipliers") or []
    if isinstance(multipliers, str):
        multipliers = multipliers.split()

    return sanitize_lora_values(
        list(loras),
        list(multipliers),
        architecture=architecture,
        task_id=task_id,
    )

