"""Static transition-prompt templates used by tests and callers."""

SYSTEM_PROMPT = (
    "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES "
    "focused on motion and change."
)


def build_single_transition_query(prompt_text: str) -> str:
    return (
        "Compare the left image and right image and describe the transition in THREE SENTENCES. "
        f"User prompt: {prompt_text}"
    )


def build_batch_transition_query(prompt_text: str) -> str:
    return (
        "For each pair with a green border marker, compare the left image and right image and "
        f"describe the motion transition. User prompt: {prompt_text}"
    )


__all__ = ["SYSTEM_PROMPT", "build_single_transition_query", "build_batch_transition_query"]
