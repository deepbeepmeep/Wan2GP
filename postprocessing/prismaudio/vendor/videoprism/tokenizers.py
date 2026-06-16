# Copyright 2026 VideoPrism Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenizers for text encoders."""

from collections.abc import Sequence
from typing import Protocol
import urllib.request

import sentencepiece

SentencePieceProcessor = sentencepiece.SentencePieceProcessor


class Tokenizer(Protocol):
  """Tokenizer interface."""

  def to_int(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> list[int] | list[list[int]]:
    """Tokenizes `text` into a list of integer tokens.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A list or list-of-list of tokens.
    """

  @property
  def pad_token(self) -> int:
    """Token id of padding token."""

  @property
  def eos_token(self) -> int:
    """Token id of end-of-sentence token."""

  @property
  def bos_token(self) -> int:
    """Token id of beginning-of-sentence token."""

  @property
  def vocab_size(self) -> int:
    """Returns the size of the vocabulary."""


def _read_binary(path: str) -> bytes:
  if path.startswith("gs://"):
    path = "https://storage.googleapis.com/" + path[len("gs://"):]
  if path.startswith(("http://", "https://")):
    with urllib.request.urlopen(path) as response:
      return response.read()
  with open(path, "rb") as reader:
    return reader.read()


class SentencePieceTokenizer(Tokenizer):
  """Wraps a SentencePiece model for tokenization."""

  def __init__(self, model_path):
    """Initializes the tokenizer.

    Args:
      model_path: A path to load the SentencePiece model.
    """
    self._model = SentencePieceProcessor()
    self._model.LoadFromSerializedProto(_read_binary(model_path))

  def to_int(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> list[int] | list[list[int]]:
    """Tokenizes `text` into a list of integer tokens.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A list or list-of-list of tokens.
    """

    def _single(s: str) -> list[int]:
      return (
          ([self.bos_token] if bos else [])
          + self._model.EncodeAsIds(s)
          + ([self.eos_token] if eos else [])
      )

    if isinstance(text, str):
      return _single(text)
    return list([_single(s) for s in text])

  @property
  def pad_token(self) -> int:
    """Token id of padding token."""
    return self._model.pad_id()

  @property
  def eos_token(self) -> int:
    """Token id of end-of-sentence token."""
    return self._model.eos_id()

  @property
  def bos_token(self) -> int:
    """Token id of beginning-of-sentence token."""
    return self._model.bos_id()

  @property
  def vocab_size(self) -> int:
    """Returns the size of the vocabulary."""
    return self._model.GetPieceSize()
