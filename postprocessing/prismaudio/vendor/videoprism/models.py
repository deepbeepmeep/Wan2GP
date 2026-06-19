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

"""Provides builders and loaders of VideoPrism checkpoints.

The v1 base model takes videos with shape (16, 288, 288) as inputs and outputs
embeddings with shape (batch_size, 4096, 768) which could be reshaped into
(batch_size, 16, 16, 16, 768) for spatiotemporal representations. The input
videos should be normalized in [0.0, 1.0].

Example usage:
```
from videoprism import models as vp

model_name = 'videoprism_public_v1_base'
flax_model = vp.get_model(model_name)
loaded_state = vp.load_pretrained_weights(model_name)

@jax.jit
def forward_fn(inputs):
  return flax_model.apply(loaded_state, inputs, train=False)

model_inputs = ...
outputs = forward_fn(model_inputs)
```
"""

from collections.abc import Callable, Mapping, Sequence
import functools

from flax import linen as nn
import jax
import jax.numpy as jnp
import huggingface_hub
import numpy as np
from videoprism import encoders
from videoprism import tokenizers
from videoprism import utils

K400_NUM_CLASSES: int = 400
SSV2_NUM_CLASSES: int = 174

TEXT_MAX_LEN: int = 64
TEXT_TOKENIZERS = {
    'c4_en': {
        'model_path': 'gs://t5-data/vocabs/cc_en.32000/sentencepiece.model',
        'vocab_size': 32_000,
    },
}

CHECKPOINTS = {
    # Hugging Face checkpoints (repository, filename).
    'videoprism_public_v1_base': (
        'google/videoprism-base-f16r288',
        'flax_base_f16r288_repeated.npz',
    ),
    'videoprism_public_v1_large': (
        'google/videoprism-large-f8r288',
        'flax_large_f8r288_repeated.npz',
    ),
    'videoprism_lvt_public_v1_base': (
        'google/videoprism-lvt-base-f16r288',
        'flax_lvt_base_f16r288_repeated.npz',
    ),
    'videoprism_lvt_public_v1_large': (
        'google/videoprism-lvt-large-f8r288',
        'flax_lvt_large_f8r288_repeated.npz',
    ),
}

CONFIGS = {
    'videoprism_v1_base': dict(
        patch_size=18,
        pos_emb_shape=(16, 16, 16),
        model_dim=768,
        num_spatial_layers=12,
        num_temporal_layers=4,
        num_heads=12,
        mlp_dim=3072,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_v1_large': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        model_dim=1024,
        num_spatial_layers=24,
        num_temporal_layers=4,
        num_heads=16,
        mlp_dim=4096,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_v1_giant': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        model_dim=1408,
        num_spatial_layers=40,
        num_temporal_layers=4,
        num_heads=16,
        mlp_dim=6144,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_lvt_v1_base': dict(
        patch_size=18,
        pos_emb_shape=(16, 16, 16),
        num_spatial_layers=12,
        num_temporal_layers=4,
        mlp_dim=3072,
        num_auxiliary_layers=2,
        enable_causal_atten=True,
        num_unimodal_layers=12,
        norm_policy='pre',
        model_dim=768,
        num_heads=12,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_lvt_v1_large': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        num_spatial_layers=24,
        num_temporal_layers=4,
        mlp_dim=4096,
        num_auxiliary_layers=2,
        enable_causal_atten=True,
        num_unimodal_layers=12,
        norm_policy='pre',
        model_dim=1024,
        num_heads=16,
        atten_logit_cap=50.0,
        scan=True,
    ),
    'videoprism_lvt_v1_giant': dict(
        patch_size=18,
        pos_emb_shape=(8, 16, 16),
        num_spatial_layers=40,
        num_temporal_layers=4,
        mlp_dim=6144,
        num_auxiliary_layers=2,
        enable_causal_atten=True,
        num_unimodal_layers=16,
        norm_policy='primer_hybrid',
        model_dim=1408,
        num_heads=16,
        atten_logit_cap=50.0,
        scan=True,
    ),
}


def videoprism_v1_base():
  """Builds VideoPrism v1 base model."""
  return encoders.FactorizedEncoder(**CONFIGS['videoprism_v1_base'])


def videoprism_v1_large():
  """Builds VideoPrism v1 large model."""
  return encoders.FactorizedEncoder(**CONFIGS['videoprism_v1_large'])


def videoprism_v1_giant():
  """Builds VideoPrism v1 giant model."""
  return encoders.FactorizedEncoder(**CONFIGS['videoprism_v1_giant'])


def videoprism_lvt_v1_base(text_tokenizer: str = 'c4_en'):
  """Builds VideoPrism LvT v1 base model."""
  config = CONFIGS['videoprism_lvt_v1_base']
  config['vocabulary_size'] = TEXT_TOKENIZERS[text_tokenizer]['vocab_size']
  return encoders.FactorizedVideoCLIP(**config)


def videoprism_lvt_v1_large(text_tokenizer: str = 'c4_en'):
  """Builds VideoPrism LvT v1 large model."""
  config = CONFIGS['videoprism_lvt_v1_large']
  config['vocabulary_size'] = TEXT_TOKENIZERS[text_tokenizer]['vocab_size']
  return encoders.FactorizedVideoCLIP(**config)


def videoprism_lvt_v1_giant(text_tokenizer: str = 'c4_en'):
  """Builds VideoPrism LvT v1 giant model."""
  config = CONFIGS['videoprism_lvt_v1_giant']
  config['vocabulary_size'] = TEXT_TOKENIZERS[text_tokenizer]['vocab_size']
  return encoders.FactorizedVideoCLIP(**config)


def videoprism_vc_v1_base(num_classes: int):
  """Builds VideoPrism Classification v1 base model."""
  encoder_params = CONFIGS['videoprism_v1_base']
  return encoders.FactorizedVideoClassifier(
      encoder_params=encoder_params, num_classes=num_classes
  )


def videoprism_vc_v1_large(num_classes: int):
  """Builds VideoPrism Classification v1 large model."""
  encoder_params = CONFIGS['videoprism_v1_large']
  return encoders.FactorizedVideoClassifier(
      encoder_params=encoder_params, num_classes=num_classes
  )


def videoprism_vc_v1_giant(num_classes: int):
  """Builds VideoPrism Classification v1 giant model."""
  encoder_params = CONFIGS['videoprism_v1_giant']
  return encoders.FactorizedVideoClassifier(
      encoder_params=encoder_params, num_classes=num_classes
  )


MODELS = {
    'videoprism_public_v1_base': videoprism_v1_base,
    'videoprism_public_v1_large': videoprism_v1_large,
    'videoprism_lvt_public_v1_base': functools.partial(
        videoprism_lvt_v1_base, text_tokenizer='c4_en'
    ),
    'videoprism_lvt_public_v1_large': functools.partial(
        videoprism_lvt_v1_large, text_tokenizer='c4_en'
    ),
}


def _get_model_name_by_hf_model_id(model_id: str) -> str | None:
  """Returns model name for the given Hugging Face model ID.

  Hugging Face model ID is typically the name of the repository, e.g.,
  `google/videoprism-base-f16r288`.

  Args:
    model_id: A string for the Hugging Face model ID.

  Returns:
    The model name for the given Hugging Face model ID or None if not found.
  """
  for model_name, value in CHECKPOINTS.items():
    if isinstance(value, tuple) and value[0] == model_id:
      return model_name

  return None


def has_model(
    model_name: str,
    models: Mapping[str, Callable[[], nn.Module]] | None = None,
) -> bool:
  """Returns whether the model is available."""
  models = models or MODELS
  if model_name.startswith('google/'):
    # Handle Hugging Face model ID.
    model_name = _get_model_name_by_hf_model_id(model_name)

  return model_name is not None and model_name in models


def get_model(
    model_name: str | None,
    model_fn: Callable[[], nn.Module] | None = None,
    models: Mapping[str, Callable[[], nn.Module]] | None = None,
    fprop_dtype: jax.typing.DTypeLike | None = None,
):
  """Returns VideoPrism model with the given name.

  Args:
    model_name: A string for the model name or Hugging Face model ID.
    model_fn: Optional function that returns the model.
    models: Mapping from model name to model creation function. Used with
      `model_name`. If None, use the default `MODELS`.

  Returns:
    A Flax VideoPrism model.
  """

  if model_fn is None:
    assert model_name is not None
    models = models or MODELS
    if model_name.startswith('google/'):
      # Handle Hugging Face model ID.
      model_name = _get_model_name_by_hf_model_id(model_name)
      if model_name is None:
        raise ValueError(f'Failed to find model name with `{model_name}`.')

    if model_name not in models:
      raise ValueError(f'Model `{model_name}` not found.')

    model_fn = models[model_name]

  model = model_fn()
  if fprop_dtype is not None:
    model.fprop_dtype = fprop_dtype
  return model


def load_pretrained_weights(
    model_name: str | None,
    checkpoint_path: str | None = None,
    checkpoints: Mapping[str, str | tuple[str, str]] | None = None,
):
  """Loads pretrained model weights.

  Args:
    model_name: A string for the model name or Hugging Face model ID.
    checkpoint_path: Optional path of the model checkpoint.
    checkpoints: Mapping from model name to checkpoint path. Used with
      `model_name`. If None, use the default `CHECKPOINTS`.

  Returns:
    Restored Flax model weights.
  """
  checkpoints = checkpoints or CHECKPOINTS

  if checkpoint_path is None:
    assert model_name is not None
    if model_name.startswith('google/'):
      # Handle Hugging Face model ID.
      model_name = _get_model_name_by_hf_model_id(model_name)

    repo_id, filename = checkpoints[model_name]
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=filename
    )

  variables = utils.load_checkpoint(checkpoint_path)
  return jax.tree_util.tree_map(jnp.asarray, variables)


def load_text_tokenizer(name: str) -> tokenizers.Tokenizer:
  """Loads a text tokenizer by name.

  Args:
    name: A string for the text tokenizer model name.

  Returns:
    A text tokenizer.
  """
  if name not in TEXT_TOKENIZERS:
    raise ValueError(f'Text tokenizer `{name}` not found.')

  model_path = TEXT_TOKENIZERS[name]['model_path']
  return tokenizers.SentencePieceTokenizer(model_path)


def tokenize_texts(
    tokenizer: tokenizers.Tokenizer,
    inputs: Sequence[str],
    max_length: int = TEXT_MAX_LEN,
    add_bos: bool | None = None,
    canonicalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
  """Tokenizes a batch of texts.

  Args:
    tokenizer: The tokenizer to use.
    inputs: The list of texts to tokenize.
    max_length: The maximum length of the tokenized texts.
    add_bos: Whether to add a beginning-of-sentence token. If None, the
      beginning-of-sentence token will be added if the tokenizer's bos_token is
      a non-negative integer.
    canonicalize: Whether to canonicalize the texts before tokenization.

  Returns:
    A tuple of two numpy arrays containing the padded token ids and the
    corresponding paddings, where 1 denotes padding token.
  """

  if canonicalize:
    inputs = [utils.canonicalize_text(text) for text in inputs]

  batch_ids, batch_paddings = [], []
  if add_bos is None:
    add_bos = tokenizer.bos_token >= 0

  for ids in tokenizer.to_int(inputs, bos=add_bos, eos=False):
    ids_seq_len = len(ids)
    if ids_seq_len > max_length:
      ids = ids[:max_length]

    ids = np.asarray(ids, dtype=np.int32)
    paddings = np.zeros_like(ids, dtype=np.float32)

    if ids_seq_len < max_length:
      ids = np.pad(
          ids, (0, max_length - ids_seq_len), 'constant', constant_values=0
      )
      paddings = np.pad(
          paddings,
          (0, max_length - ids_seq_len),
          'constant',
          constant_values=1.0,
      )

    batch_ids.append(ids)
    batch_paddings.append(paddings)

  return np.asarray(batch_ids), np.asarray(batch_paddings)
