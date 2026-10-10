from __future__ import annotations

from typing import Any, Sequence

from .images import temporary_image_paths
from .registry import create_backend
from shared.utils.cancellation import check_cancelled


def enhance_prompt(engine: str, server_config: dict[str, Any], prompts: Sequence[str], images: Sequence[Any], *, instructions: str, max_output_tokens: int, prompt_durations=None, enhancement_progress=None) -> list[str]:
    with temporary_image_paths(images) as image_paths:
        results = []
        for index, prompt in enumerate(prompts):
            if enhancement_progress is not None:
                enhancement_progress.prompt(index, len(prompts), max_output_tokens)
            backend = create_backend(engine, server_config)
            try:
                system_prompt = str(instructions or "")
                if prompt_durations is not None and prompt_durations[index] is not None:
                    system_prompt += f"\nThis window lasts {prompt_durations[index]:g} seconds. Keep the described action within this duration."
                def should_stop():
                    try:
                        check_cancelled()
                    except InterruptedError:
                        return True
                    return False

                check_cancelled()
                results.append(backend.run_turn(str(prompt or ""), system_prompt=system_prompt, tools=[], images=image_paths, on_event=lambda event: None, call_tool=lambda name, args: {}, should_stop=should_stop))
                check_cancelled()
            finally:
                backend.close()
        return results
