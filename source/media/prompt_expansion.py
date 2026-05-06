"""Worker-owned prompt expansion preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from source.runtime.wgp_bridge import create_qwen_prompt_expander


QWEN_PROMPT_EXPANSION_TASK_TYPES = frozenset(
    {
        "qwen_image",
        "qwen_image_2512",
        "qwen_image_edit",
        "qwen_image_style",
        "image_inpaint",
        "annotated_image_edit",
    }
)

QWEN_PROMPT_EXPANSION_SYSTEM_PROMPT = (
    "You expand concise image generation prompts into detailed, faithful prompts. "
    "Preserve the user's subject, intent, style, and constraints. Do not add "
    "conflicting content or policy text. Return only the expanded prompt."
)


@dataclass(frozen=True)
class PromptExpansionMetadata:
    provider: str
    task_type: str
    requested: bool
    applied: bool
    reason: str | None = None
    model_name: str | None = None
    original_prompt: str = ""
    expanded_prompt: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptExpansionResult:
    prompt: str
    metadata: PromptExpansionMetadata


def qwen_prompt_expansion_requested(params: Mapping[str, Any] | None) -> bool:
    if not params:
        return False
    value = params.get("qwen_prompt_expansion")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def expand_qwen_prompt(
    prompt: str,
    *,
    task_type: str,
    params: Mapping[str, Any] | None = None,
    expander_factory: Callable[..., Any] = create_qwen_prompt_expander,
    model_name: str | None = None,
    device: str | int = "cpu",
    seed: int = -1,
) -> PromptExpansionResult:
    """Expand a Qwen prompt when explicitly requested by worker params."""
    original_prompt = prompt or ""
    requested = qwen_prompt_expansion_requested(params)

    if not requested:
        return _unchanged(original_prompt, task_type=task_type, requested=False, reason="not_requested")
    if task_type not in QWEN_PROMPT_EXPANSION_TASK_TYPES:
        return _unchanged(original_prompt, task_type=task_type, requested=True, reason="unsupported_task_type")
    if not original_prompt.strip():
        return _unchanged(original_prompt, task_type=task_type, requested=True, reason="empty_prompt")

    selected_model = model_name or str((params or {}).get("qwen_prompt_expansion_model") or "Qwen2.5_14B")
    expander = expander_factory(model_name=selected_model, device=device, is_vl=False)
    result = expander.extend(
        prompt=original_prompt,
        system_prompt=QWEN_PROMPT_EXPANSION_SYSTEM_PROMPT,
        seed=seed,
    )
    expanded_prompt = str(getattr(result, "prompt", "") or "").strip()
    if not expanded_prompt:
        return _unchanged(original_prompt, task_type=task_type, requested=True, reason="empty_expansion")

    return PromptExpansionResult(
        prompt=expanded_prompt,
        metadata=PromptExpansionMetadata(
            provider="qwen",
            task_type=task_type,
            requested=True,
            applied=True,
            model_name=selected_model,
            original_prompt=original_prompt,
            expanded_prompt=expanded_prompt,
            raw={
                "status": getattr(result, "status", None),
                "seed": getattr(result, "seed", None),
                "message": getattr(result, "message", None),
            },
        ),
    )


def _unchanged(
    prompt: str,
    *,
    task_type: str,
    requested: bool,
    reason: str,
) -> PromptExpansionResult:
    return PromptExpansionResult(
        prompt=prompt,
        metadata=PromptExpansionMetadata(
            provider="qwen",
            task_type=task_type,
            requested=requested,
            applied=False,
            reason=reason,
            original_prompt=prompt,
            expanded_prompt=prompt,
        ),
    )
