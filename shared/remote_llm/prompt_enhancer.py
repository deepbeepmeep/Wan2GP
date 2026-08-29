from __future__ import annotations

from typing import Any, Sequence

from .images import temporary_image_paths
from .registry import create_backend


def enhance_prompt(engine: str, server_config: dict[str, Any], prompts: Sequence[str], images: Sequence[Any], *, instructions: str, max_output_tokens: int) -> list[str]:
    with temporary_image_paths(images) as image_paths:
        results = []
        for prompt in prompts:
            backend = create_backend(engine, server_config)
            try:
                results.append(backend.one_shot(str(prompt or ""), system_prompt=str(instructions or ""), images=image_paths, max_output_tokens=int(max_output_tokens)))
            finally:
                backend.close()
        return results
