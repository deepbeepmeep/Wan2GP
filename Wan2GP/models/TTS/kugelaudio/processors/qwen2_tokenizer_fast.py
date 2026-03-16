"""Local fallback for Qwen2TokenizerFast when transformers doesn't provide it."""

from transformers import PreTrainedTokenizerFast


class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """Fallback Qwen2 fast tokenizer.

    This relies on tokenizer.json/special_tokens_map.json in the model repo.
    """

    pass
