# OpenPhonemizer English fallback for Kokoro's misaki G2P (replaces misaki.espeak.EspeakFallback, which needs GPL espeak-ng).
# Sources:
# - OpenPhonemizer (https://github.com/NeuralVox/OpenPhonemizer, commit 15ffa72, openphonemizer/__init__.py) and its
#   checkpoint openphonemizer/ckpt: Clear BSD License, Copyright (c) 2024 mrfakename, NeuralVox, OpenPhonemizer Contributors; see LICENSE.
# - DeepPhonemizer (https://github.com/as-ideas/DeepPhonemizer, commit 5dce7e2, dp/phonemizer.py, dp/model/predictor.py,
#   dp/preprocessing/text.py): MIT License, Copyright (c) 2021 Axel Springer News Media & Tech GmbH & Co. KG - Ideas Engineering; see LICENSE-DeepPhonemizer.
# - The espeak IPA -> misaki conversion (E2M table and post-processing) reimplements misaki/espeak.py EspeakFallback
#   (https://github.com/hexgrad/misaki, v0.9.4, Apache-2.0); misaki.espeak itself imports phonemizer and is never imported.
# WanGP changes: digits are read by misaki's own number reader (no num2words); accented letters are folded to ASCII instead of dropped;
# like espeak, hyphenated parts (and O'Brien) are joined and camel-case parts (DeShawn) become separate words, instead of upstream's
# acronym expansion (DIY -> D-I-Y; misaki spells capitals itself); the untied IPA the model was trained on gets espeak's '^' ties back.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import os
import re
import unicodedata

import torch
from mmgp import offload

from shared.utils import files_locator as fl
from .model import ForwardTransformer

OPENPHONEMIZER_DIR = "openphonemizer"
OPENPHONEMIZER_FILES = ["openphonemizer_fp32.safetensors", "openphonemizer_config.json", "openphonemizer_dict.json"]
PUNCTUATION = "().,:?!/–"
SPLIT_PATTERN = re.compile(f"([{re.escape(PUNCTUATION)} ])")
NUMBER_PATTERN = re.compile(r"(\d+(?:[.,]\d+)*)")
CAMEL_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])")
NAME_APOSTROPHE_PATTERN = re.compile(r"['’](?=[A-Z])")
ASCII_FOLD = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ı": "i"})

# espeak (phonemizer tie='^') ties these sequences; misaki's table relies on the ties.
VOWELS = "aeiouæɐɑɒɔəɚɛɜɝɪʊʌᵻ"
TIE_RULES = [(re.compile(pattern), repl) for pattern, repl in (
    (r"aɪ(?=[əɚ])", "a^ɪ^"), (r"([aɔe])ɪ", r"\1^ɪ"), (r"([ao])ʊ", r"\1^ʊ"), (r"əʊ", "ə^ʊ"), (r"eə", "e^ə"), (r"dʒ", "d^ʒ"), (r"tʃ", "t^ʃ"), (r"iə", "i^ə"),
    (rf"(?<![ɪi])əl(?![{VOWELS}ˈˌ])", "ə^l"),
)]
E2M = sorted({
    "ʔˌn\u0329": "ʔn", "ʔn\u0329": "ʔn",
    "a^ɪ": "I", "a^ʊ": "W",
    "d^ʒ": "ʤ",
    "e^ɪ": "A", "e": "A",
    "t^ʃ": "ʧ",
    "ɔ^ɪ": "Y",
    "ə^l": "ᵊl",
    "ʲo": "jo", "ʲə": "jə", "ʲ": "",
    "ɚ": "əɹ",
    "r": "ɹ",
    "x": "k", "ç": "k",
    "ɐ": "ə",
    "ɬ": "l",
    "\u0303": "",
}.items(), key=lambda kv: -len(kv[0]))


def espeak_to_misaki(ps: str, british: bool) -> str:
    """Untied espeak-style IPA -> Kokoro phonemes, as misaki's EspeakFallback converts espeak output (version != '2.0')."""
    for pattern, repl in TIE_RULES:
        ps = pattern.sub(repl, ps)
    for old, new in E2M:
        ps = ps.replace(old, new)
    ps = re.sub(r"(\S)\u0329", r"ᵊ\1", ps).replace(chr(809), "")
    if british:
        ps = ps.replace("e^ə", "ɛː").replace("iə", "ɪə").replace("ə^ʊ", "Q")
    else:
        ps = ps.replace("o^ʊ", "O").replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ").replace("ɪə", "iə").replace("ː", "")
    return ps.replace("o", "ɔ").replace("ɾ", "T").replace("ʔ", "t").replace("^", "")


class OpenPhonemizerFallback:
    """misaki `fallback` callable for out-of-dictionary English words: token -> (phonemes, rating).
    `lexicon` is the misaki G2P lexicon; it reads the digits of the token, OpenPhonemizer the rest."""

    def __init__(self, lexicon, device: torch.device):
        self.lexicon, self.british, self.device = lexicon, lexicon.british, torch.device(device)
        weights_path, config_path, dict_path = (fl.locate_file(os.path.join(OPENPHONEMIZER_DIR, name)) for name in OPENPHONEMIZER_FILES)
        with open(config_path, encoding="utf-8") as reader:
            config = json.load(reader)
        with open(dict_path, encoding="utf-8") as reader:
            self.words = json.load(reader)
        self.char_repeats, self.pad_index, self.start_index, self.end_index = config["char_repeats"], config["pad_index"], config["start_index"], config["end_index"]
        self.special_indices = {self.pad_index, self.start_index, self.end_index}
        self.text_index = {c: i for i, c in enumerate(config["text_tokens"]) if i not in self.special_indices}
        self.phoneme_tokens = config["phoneme_tokens"]
        with torch.device("meta"):
            model = ForwardTransformer(len(config["text_tokens"]), len(self.phoneme_tokens), **config["model"])
        offload.load_model_data(model, weights_path, writable_tensors=False, default_dtype=None)
        self.model = model.eval().requires_grad_(False).to(self.device)

    def lookup(self, word: str):
        return "" if word == "-" else word if word in PUNCTUATION + " " else self.words.get(word.lower())

    @torch.inference_mode()
    def predict(self, words: list[str]) -> list[str]:
        if not words:
            return []
        sequences = [[self.text_index[c] for c in word.lower() if c in self.text_index for _ in range(self.char_repeats)] for word in words]
        width = max(map(len, sequences))
        batch = torch.tensor([[self.start_index, *sequence, self.end_index] + [self.pad_index] * (width - len(sequence)) for sequence in sequences], dtype=torch.long, device=self.device)
        results = []
        for sequence, tokens in zip(sequences, self.model.generate(batch)):
            tokens = tokens[:tokens.index(self.end_index) + 1] if self.end_index in tokens else tokens
            results.append("".join(self.phoneme_tokens[t] for t in tokens if t not in self.special_indices) if sequence else "")
        return results

    def phonemize(self, text: str) -> str:
        """Espeak-style en-us IPA (no ties): dictionary words first, then hyphen / camel-case subwords, the model for the rest."""
        text = "".join(c for c in unicodedata.normalize("NFKD", NAME_APOSTROPHE_PATTERN.sub("-", text.translate(ASCII_FOLD))) if c.isalnum() or c in PUNCTUATION + "- ")
        parts = [part for part in SPLIT_PATTERN.split(text) if part]
        phonemes = {part: self.lookup(part) for part in parts}
        splits = {word: [sub for sub in re.split(r"(-| )", CAMEL_PATTERN.sub(" ", word)) if sub] for word, ps in phonemes.items() if ps is None}
        phonemes.update({sub: self.lookup(sub) for subs in splits.values() for sub in subs if sub not in phonemes})
        missing = [word for word, ps in phonemes.items() if ps is None and len(splits.get(word, ())) <= 1]
        phonemes.update(zip(missing, self.predict(missing)))
        return "".join(phonemes[part] if phonemes[part] is not None else "".join(phonemes[sub] for sub in splits[part]) for part in parts).strip()

    def __call__(self, token):
        ps = ""
        for i, part in enumerate(NUMBER_PATTERN.split(token.text)):
            part = (self.lexicon.get_number(part, None, True, "")[0] or "") if i % 2 else espeak_to_misaki(self.phonemize(part), self.british)
            ps += " " + part if ps and part and ps[-1] not in PUNCTUATION and part[0] not in PUNCTUATION else part
        return ps, 2
