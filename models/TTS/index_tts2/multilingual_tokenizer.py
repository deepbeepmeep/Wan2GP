from __future__ import annotations

import base64
import os
from functools import lru_cache

import tiktoken
from whisper.tokenizer import Tokenizer


LANGUAGES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
)
LANGUAGE_DICT = {language: index for index, language in enumerate(LANGUAGES)}
LANGUAGE_ALIASES = {"zh-en": "zh/en", "mixed": "zh/en"}


def lang_to_token(language):
    language = LANGUAGE_ALIASES.get(str(language).lower(), str(language).lower())
    return LANGUAGE_DICT.get(language, LANGUAGE_DICT["common"])


@lru_cache(maxsize=None)
def get_encoding(model_dir, name="multilingual_zh_ja_yue_char_del", num_languages=99):
    vocab_path = os.path.join(model_dir, f"{name}.tiktoken")
    with open(vocab_path, encoding="utf-8") as handle:
        ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in handle if line)}
    special_names = [
        "<|endoftext|>", "<|startoftranscript|>",
        *[f"<|{language}|>" for language in LANGUAGES[:num_languages]],
        *[f"<|{event}|>" for event in ("ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause")],
        *[f"<|{emotion}|>" for emotion in ("HAPPY", "SAD", "ANGRY", "NEUTRAL")],
        "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{index}|>" for index in range(1, 31)],
        *[f"<|TTS/{name}|>" for name in ("B", "O", "Q", "A", "CO", "CL", "H", *[f"SP{index:02d}" for index in range(1, 14)])],
        *[f"<|{index * 0.02:.2f}|>" for index in range(1501)],
    ]
    special_tokens = {token: len(ranks) + index for index, token in enumerate(special_names)}
    return tiktoken.Encoding(name=os.path.basename(vocab_path), explicit_n_vocab=len(ranks) + len(special_tokens), pat_str=r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", mergeable_ranks=ranks, special_tokens=special_tokens)


class MultilingualTokenizer(Tokenizer):
    def tokenize(self, text, max_token_comb=4):
        token_ids = self.encode(text, allowed_special="all")
        tokens = []
        index = 0
        while index < len(token_ids):
            for length in range(1, min(max_token_comb, len(token_ids) - index) + 1):
                candidate = self.decode(token_ids[index:index + length])
                if "\ufffd" not in candidate and candidate.strip():
                    tokens.append(candidate)
                    index += length
                    break
            else:
                tokens.append(self.decode([token_ids[index]]))
                index += 1
        return tokens


@lru_cache(maxsize=None)
def get_tokenizer(model_dir):
    return MultilingualTokenizer(encoding=get_encoding(model_dir), num_languages=99, language="en", task="transcribe")
