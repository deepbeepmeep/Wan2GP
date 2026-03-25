from __future__ import annotations

from typing import Any

from shared.prompt_enhancer.qwen35_vl import _prepare_multimodal_vllm_prompt


VISION_QA_SYSTEM_PROMPT = "Answer the user's question about the provided image accurately and concisely. If the answer is uncertain, say so."


def build_image_question_prompt(caption_model: Any, processor: Any, image: Any, question: str, system_prompt: str | None = None):
    question = str(question or "").strip()
    if len(question) == 0:
        raise ValueError("Vision question is empty.")
    messages = []
    system_prompt = str(system_prompt or VISION_QA_SYSTEM_PROMPT).strip()
    if len(system_prompt) > 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        return_mm_token_type_ids=True,
    )
    return _prepare_multimodal_vllm_prompt(caption_model, model_inputs)


__all__ = ["VISION_QA_SYSTEM_PROMPT", "build_image_question_prompt"]
