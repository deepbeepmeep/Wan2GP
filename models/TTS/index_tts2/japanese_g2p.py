from __future__ import annotations

import re

class JapaneseG2PProcessor:
    def __init__(self):
        import fugashi

        self.tagger = fugashi.Tagger()

    def _process_segment(self, text):
        return " ".join(token.surface for token in self.tagger(text))

    def process(self, text):
        return "".join(self._process_segment(part) if part.strip() else part for part in re.split(r"( +)", text))
