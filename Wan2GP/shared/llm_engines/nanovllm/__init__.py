__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from .llm import LLM
        return LLM
    if name == "SamplingParams":
        from .sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
