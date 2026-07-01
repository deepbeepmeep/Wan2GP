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

"""Modules for video encoders."""

from collections.abc import Collection, Sequence
import dataclasses
import math
from typing import Any

import einops
import einshape
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from videoprism import layers

Array = jax.Array
Variables = nn.module.VariableDict

default_kernel_init = layers.default_kernel_init


def _contains(collection: Collection[str] | bool, key: str) -> bool:
  """Checks if a collection contains a key.

  Args:
    collection: A collection of strings or a boolean value.
    key: A string key to check.

  Returns:
    True if the collection contains the key, or if the collection is a True
    boolean. False otherwise.
  """
  return collection if isinstance(collection, bool) else key in collection


def _l2_normalize(
    x: Array, axis: int | Sequence[int] = -1, epsilon: float = 1e-12
) -> Array:
  """L2-normalizes a jax.Array along certain dimension.

  Args:
    x: An input jax.Array.
    axis: An integer or a sequence of integers for the axis to normalize.
    epsilon: A small constant for numerical stability.

  Returns:
    Normalized jax.Array.
  """
  x_dtype = x.dtype
  # Always convert embed to float32 for all precisions.
  x = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + epsilon)
  return (x / norm).astype(x_dtype)


def _image_to_patch(inputs: Array, patch_size: int) -> Array:
  """Converts an image to patches.

  Args:
    inputs: A jax.Array of shape [B, H, W, C] ,
    patch_size: An integer for dimension of a square patch.

  Returns:
    batched_patches: [B, (H * W / P^2), P^2 * C].
  """
  if len(inputs.shape) < 4:
    raise ValueError(
        f'Image should be formatted as 4D [B, H, W, C], Shape: {inputs.shape}'
    )
  height, width, channels = inputs.shape[-3:]

  if height % patch_size != 0 or width % patch_size != 0:
    raise ValueError(
        f'Image height ({height}) and width ({width}) should be multiples '
        f'of patch_size ({patch_size}).'
    )

  row_blocks = height // patch_size
  column_blocks = width // patch_size

  patches = einops.rearrange(
      inputs,
      '... (m p)(n q) c->...(m n)(p q c)',
      m=row_blocks,
      n=column_blocks,
      p=patch_size,
      q=patch_size,
      c=channels,
  )
  return patches


def _interpolate_emb_1d(emb: Array, target_emb_length: int) -> Array:
  """Interpolates a 1D positional embedding to a new shape.

  Args:
    emb: jax.Array, (1, N, D), flattened 1D positional embedding.
    target_emb_length: length of the target embedding.

  Returns:
    Flattened, interpolated embedding of shape (1, target_emb_length, D)
  """

  if len(emb.shape) > 3 or emb.shape[0] != 1:
    raise ValueError('The shape of the embedding should be (1, N, D)')

  emb_dim = emb.shape[-1]
  emb = jnp.squeeze(emb, axis=0)

  target_emb = jax.image.resize(
      emb, (target_emb_length, emb_dim), method='bilinear'
  )
  target_emb = jnp.reshape(target_emb, (1, target_emb_length, emb_dim))
  return target_emb


def _interpolate_emb_2d(
    emb: Array,
    source_emb_shape: tuple[int, int],
    target_emb_shape: tuple[int, int],
) -> Array:
  """Interpolates a 2D positional embedding to a new shape.

  Args:
    emb: A jax.Array of shape (1, H1xW1, D) for flattened 2D positional
      embedding.
    source_emb_shape: Tuple, (H1, W1), height and width of the source embedding.
    target_emb_shape: Tuple, (H2, W2), height and width of the target embedding.

  Returns:
    Flattened, interpolated embedding of shape (1, H2xW2, D)
  """

  if len(emb.shape) > 3 or emb.shape[0] != 1:
    raise ValueError('The shape of the embedding should be (1, H * W, D)')

  if emb.shape[-2] != source_emb_shape[0] * source_emb_shape[1]:
    raise ValueError('The shape of the embedding does NOT match input specs.')

  emb_dim = emb.shape[-1]
  emb = jnp.reshape(emb, (source_emb_shape[0], source_emb_shape[1], emb_dim))

  target_emb = jax.image.resize(
      emb,
      (target_emb_shape[0], target_emb_shape[1], emb_dim),
      method='bilinear',
  )
  target_emb = jnp.reshape(
      target_emb, (1, target_emb_shape[0] * target_emb_shape[1], emb_dim)
  )
  return target_emb


class Embedding(layers.Module):
  """A simple embedding layer that performs embedding lookups from ids.

  Attributes:
    num_classes: Number of tokens in the vocabulary.
    input_dim: Depth of the embedding output. This is called `input_dim` as
      opposed to the more appropriate `embedding_dim` to be compatible with
      other embedding layers defined in this file.
    lookup_style: Style of lookup, one of index or matmul.
    scale_sqrt_depth: If set to True, activations are scaled with
      sqrt(embedding_dim) in embeding lookup.
    set_nan_for_oob_id: If set to True, embeddings corresponding to
      out-of-boundaries ids will be set to NaN.
  """

  num_classes: int = 0
  input_dim: int = 0
  lookup_style: str = 'index'
  scale_sqrt_depth: bool = False
  set_nan_for_oob_id: bool = False

  @nn.compact
  def __call__(self, ids: Array) -> Array:
    """Generates a jax.Array of embedding lookup result.

    Args:
      ids: Indexes of shape [...] for embedding lookup.

    Returns:
      A jax.Array of shape [..., input_dim].
    """
    emb_var = self._cast_to_fprop_dtype(
        self.param(
            'emb_var',
            nn.initializers.normal(stddev=1.0 / math.sqrt(self.input_dim)),
            [self.num_classes, self.input_dim],
            self.dtype,
        )
    )
    if self.lookup_style == 'index':
      embs = jnp.asarray(emb_var)[(ids,)]
    elif self.lookup_style == 'matmul':
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      embs = jnp.einsum('...y,yz->...z', one_hot_ids, emb_var)
    else:
      raise ValueError(f'Unknown lookup style: `{self.lookup_style}`.')

    # Map out-of-boundary ids to NaN.
    if self.set_nan_for_oob_id:
      embs = jnp.where(ids[..., jnp.newaxis] < self.num_classes, embs, jnp.nan)

    if self.scale_sqrt_depth:
      embs *= self.input_dim**0.5

    return embs


class PositionalEmbedding(layers.Module):
  """Generates position embedding for a given 1-d sequence.

  Attributes:
    embedding_dim: Dimension of the embedding to be generated.
    min_timescale: Start of the geometric index.
    max_timescale: End of the geometric index.
  """

  embedding_dim: int = 0
  min_timescale: int = 1
  max_timescale: int = 10_000

  def __call__(self, seq_length: int) -> Array:
    """Generates a jax.Array of embedding lookup result.

    Args:
      seq_length: Sequence length of the embeddings to be generated.

    Returns:
      A jax.Array of shape [1, seq_length, embedding_dim].
    """
    position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    num_timescales = self.embedding_dim // 2
    log_timescale_increment = math.log(
        float(self.max_timescale) / float(self.min_timescale)
    ) / jnp.maximum(jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1)
    inv_timescales = self.min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = (
        position[:, :, jnp.newaxis]
        * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    embs = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1
    ).astype(self.fprop_dtype)
    # Force usage of `np` to compute static values at trace time.
    embs = jnp.pad(embs, [[0, 0], [0, 0], [0, np.mod(self.embedding_dim, 2)]])
    return embs


class TrainablePositionalEmbedding(layers.Module):
  """Generates trainable position embedding for a given 1-d sequence.

  Attributes:
    embedding_dim: Dimension of the embedding to be generated.
    max_seq_length: Max sequence length.
    lookup_style: Style of lookup, one of index or matmul.
  """

  embedding_dim: int = 0
  max_seq_length: int = 10_240
  lookup_style: str = 'matmul'

  @nn.compact
  def __call__(self, seq_length: int) -> Array:
    """Generates a jax.Array of embedding lookup result.

    Args:
      seq_length: Sequence length of the embeddings to be generated.

    Returns:
      A jax.Array of shape [1, seq_length, embedding_dim].
    """
    position = jnp.arange(seq_length, dtype=jnp.int32)[jnp.newaxis, :]
    pos_emb_var = self._cast_to_fprop_dtype(
        self.param(
            'emb_var',
            default_kernel_init,
            [self.max_seq_length, self.embedding_dim],
            self.dtype,
        )
    )
    pos_emb_var = jax.lax.slice_in_dim(pos_emb_var, 0, seq_length, axis=0)
    if self.lookup_style == 'matmul':
      one_hot_ids = jax.nn.one_hot(position, seq_length, dtype=self.fprop_dtype)
      embs = jnp.einsum('...y,yz->...z', one_hot_ids, pos_emb_var)
    else:
      raise ValueError(f'Unknown lookup style: `{self.lookup_style}`.')
    return embs


class VisionTransformer(layers.Module):
  """Vision transformer model.

  This class follows a minimalistic design pattern. Users need to configure the
  templates for the submodules themselves; this increases the generalizability
  of this class.

  Attributes:
    num_tfm_layers: Number of layers in this model.
    mlp_dim: The hidden layer dimension of FFN in Transformer layers.
    num_heads: Number of attention heads.
    xformer_has_bias: Whether to use bias.
    xformer_dropout_prob: Apply dropout at this prob at various places.
    xformer_atten_dropout_prob: Probability at which we apply dropout to the
      attention weights.
    xformer_residual_dropout_prob: Probability at which we apply dropout to the
      residual layers, such that, residual(x, y) = (x + dropout(y)).
    xformer_relu_dropout_prob: Probability at which we apply dropout to the FFN
      layers.
    atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
      positive value is specified. May not be supported by a subclass.
    norm_policy: Policy for applying normalization wrt. transformations. Options
      are: (1) "pre", applied before transformation. (2) "primer_hybrid",
        applied before and after transformation. (3) "post", applied after
        transformation. (4) "post_skip", applied after the skip connection.
    scan: Whether to use `nn.remat` and`nn.scan`.
  """

  num_tfm_layers: int = 12
  mlp_dim: int = 3072
  num_heads: int = 12
  xformer_has_bias: bool = True
  xformer_dropout_prob: float = 0.0
  xformer_atten_dropout_prob: float | None = None
  xformer_residual_dropout_prob: float | None = None
  xformer_relu_dropout_prob: float | None = None
  atten_logit_cap: float = 0.0
  norm_policy: str = 'pre'
  scan: bool = False

  @nn.compact
  def __call__(
      self, inputs: Array, paddings: Array | None = None, train: bool = False
  ) -> Array:
    """Applies the ViT model to the inputs.

    Args:
      inputs: Input tensor of shape [B, N, D], which are sequences of embeddings
        or patches.
      paddings: Optional [B, N] padding field of inputs when inputs are with [B,
        N, D].
      train: If the model is in the train mode.

    Returns:
      Output tensor of shape [B, N, D].
    """
    features = inputs
    if paddings is None:
      paddings = jnp.zeros(features.shape[:-1], dtype=features.dtype)
    features = layers.StackedTransformer(
        name='transformers_stack',
        num_layers=self.num_tfm_layers,
        hidden_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout_prob=self.xformer_dropout_prob,
        atten_dropout_prob=self.xformer_atten_dropout_prob,
        residual_dropout_prob=self.xformer_residual_dropout_prob,
        relu_dropout_prob=self.xformer_relu_dropout_prob,
        use_bias=self.xformer_has_bias,
        atten_logit_cap=self.atten_logit_cap,
        norm_policy=self.norm_policy,
        internal_enable_per_dim_scale=False,
        activation_fn=layers.gelu,
        enable_causal_atten=False,
        scan=self.scan,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(features, paddings, train=train)
    return features


class FactorizedEncoder(layers.Module):
  """Factorized encoder from the paper `ViViT: A Video Vision Transformer`.

  This is an implementation of model-2 in the paper. It applies ViT model for
  video data based on the factorized space-time encoder.

  Reference: https://arxiv.org/abs/2103.15691
  """

  patch_size: int = 18
  pos_emb_shape: tuple[int, int, int] = (16, 16, 16)
  model_dim: int = 768
  num_spatial_layers: int = 12
  num_temporal_layers: int = 4
  num_heads: int = 12
  mlp_dim: int = 3072
  atten_logit_cap: float = 0.0
  norm_policy: str = 'pre'
  scan: bool = False

  def __call__(
      self,
      inputs: Array,
      train: bool = False,
      return_intermediate: bool | Collection[str] = False,
      frame_paddings: Array | None = None,
  ) -> tuple[Array, dict[str, Array]]:
    """Computes predictions for batched inputs.

    Args:
      inputs: Input image tensor of shape [B, T, H, W, 3] (H == W).
      train: If the model is in the train mode.
      return_intermediate: A boolean for whether all intermediate features are
        returned, or a container of intermediate feature names to return.
      frame_paddings: Optional binary tensor of shape [B, T] indicating padding.
        1 denotes padding frame.

    Returns:
      embeddings: Output tensor for video embeddings of shape [B, T * N, D].
      outputs: A dictionary of additional outputs, including `spatial_features`
        (shape = [B, T * N, D]). Empty if `return_intermediate` is False or does
        not contain 'spatial_features'.
    """
    b, t, h, w, c = inputs.shape
    assert h == w
    reshaped_inputs = inputs.reshape(b * t, h, w, c)  # (B * T, H, W, C).

    # Tokenization.
    patches = _image_to_patch(reshaped_inputs, self.patch_size)
    patches_paddings = None
    if frame_paddings is not None:
      assert frame_paddings.shape == (b, t)
      reshaped_frame_paddings = frame_paddings.reshape(b * t)  # (B * T,).
      num_patches = patches.shape[1]
      patches_paddings = jnp.repeat(
          reshaped_frame_paddings[:, jnp.newaxis], num_patches, axis=-1
      )  # (B * T, num_patches).

    embeddings, outputs = self.encode_with_patches(
        patches=patches,
        image_shape=(t, h, w),
        train=train,
        return_intermediate=return_intermediate,
        patches_paddings=patches_paddings,
    )
    return embeddings, outputs

  @nn.compact
  def encode_with_patches(
      self,
      patches: Array,
      image_shape: tuple[int, int, int],
      train: bool = False,
      return_intermediate: bool | Collection[str] = False,
      patches_paddings: Array | None = None,
  ) -> tuple[Array, dict[str, Array]]:
    """Computes predictions for patches.

    Args:
      patches: Input patches tensor of shape [B * T, (H * W / P^2), P^2 * C].
      image_shape: Original image shape (T, H, W).
      train: If the model is in the train mode.
      return_intermediate: A boolean for whether all intermediate features are
        returned, or a collection of intermediate feature names to return.
      patches_paddings: Optional binary tensor of shape [B * T, (H * W / P^2)]
        indicating padding. 1 denotes padded patch.

    Returns:
      embeddings: Output tensor for video embedding sequence of shape [B, T * N,
        D].
      outputs: A dictionary of additional outputs, including `spatial_features`
        of shape [B, T * N, D]. Empty if `return_intermediate` is False or does
        not contain 'spatial_features'.
    """
    t, h, w = image_shape
    b = patches.shape[0] // t

    patches = layers.FeedForward(  # (B * T, N, D).
        name='patch_projection',
        output_dim=self.model_dim,
        activation_fn=layers.identity,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(patches)

    # Add spatial positional encoding.
    spatial_pos_emb_shape = self.pos_emb_shape[-2:]
    spatial_seq_length = np.prod(spatial_pos_emb_shape)
    spatial_pos_emb = TrainablePositionalEmbedding(
        name='spatial_pos_emb',
        embedding_dim=self.model_dim,
        max_seq_length=spatial_seq_length,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(seq_length=spatial_seq_length)
    num_row_patches = h // self.patch_size
    num_col_patches = w // self.patch_size
    if spatial_pos_emb_shape != (num_row_patches, num_col_patches):
      spatial_pos_emb = _interpolate_emb_2d(
          spatial_pos_emb,
          spatial_pos_emb_shape,
          (num_row_patches, num_col_patches),
      )
    patches += spatial_pos_emb  # (B * T, N, D).

    # Get features from the spatial encoder.
    features = VisionTransformer(  # (B * T, N, D).
        name='spatial_encoder',
        num_tfm_layers=self.num_spatial_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        atten_logit_cap=self.atten_logit_cap,
        norm_policy=self.norm_policy,
        scan=self.scan,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(patches, train=train, paddings=patches_paddings)
    features = layers.LayerNorm(
        name='spatial_ln', dtype=self.dtype, fprop_dtype=self.fprop_dtype
    )(features)
    spatial_features = features

    # Instead of mean pooling, we keep the spatial tokens.
    # Shape = (B * N, T, D).
    features = einshape.jax_einshape('(bt)nd->(bn)td', features, t=t)
    temporal_paddings = None
    if patches_paddings is not None:
      temporal_paddings = einshape.jax_einshape(
          '(bt)n->(bn)t', patches_paddings, t=t
      )  # (B * N, T).

    # Add temporal positional encoding.
    temporal_seq_length = self.pos_emb_shape[0]
    temporal_pos_emb = TrainablePositionalEmbedding(
        name='temporal_pos_emb',
        embedding_dim=self.model_dim,
        max_seq_length=temporal_seq_length,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(seq_length=temporal_seq_length)
    if temporal_seq_length != t:
      temporal_pos_emb = _interpolate_emb_1d(temporal_pos_emb, t)
    features += temporal_pos_emb

    # Get features from the temporal encoder.
    features = VisionTransformer(
        name='temporal_encoder',
        num_tfm_layers=self.num_temporal_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        atten_logit_cap=self.atten_logit_cap,
        norm_policy=self.norm_policy,
        scan=self.scan,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(features, train=train, paddings=temporal_paddings)
    features = layers.LayerNorm(
        name='temporal_ln', dtype=self.dtype, fprop_dtype=self.fprop_dtype
    )(features)
    features = einshape.jax_einshape(  # (B, T * N, D).
        '(bn)td->b(tn)d', features, b=b
    )

    embeddings, outputs = features, {}
    if _contains(return_intermediate, 'spatial_features'):
      outputs['spatial_features'] = einshape.jax_einshape(
          '(bt)nd->b(tn)d', spatial_features, t=t
      )

    return embeddings, outputs


class FactorizedVideoClassifier(layers.Module):
  """Video classifier with `FactorizedEncoder` backbone.

  Attributes:
    encoder_params: A dictionary of parameters for `FactorizedEncoder`.
    num_classes: Number of output classes.
  """

  encoder_params: dict[str, Any] = dataclasses.field(default_factory=dict)
  num_classes: int = 0

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      train: bool = False,
      return_intermediate: bool | Collection[str] = False,
      frame_paddings: Array | None = None,
  ):
    """Applies video classifier to inputs.

    Args:
      inputs: Input tensor of shape [B, T, H, W, 3].
      train: Whether the model is in the training mode.
      return_intermediate: A boolean for whether all intermediate features are
        returned, or a collection of intermediate feature names to return.
      frame_paddings: Optional binary tensor of shape [B, T] indicating padding.
        1 denotes padding frame.

    Returns:
      logits: Output tensor of shape [B, num_classes].
      outputs: A dictionary of additional outputs, including `spatial_features`
        of shape [B, T * N, D], `spatiotemporal_features` of shape [B, T * N,
        D], and `global_embeddings` of shape [B, D]. Empty if
        `return_intermediate` is False.
    """
    features, outputs = FactorizedEncoder(
        name='encoder',
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
        **self.encoder_params,
    )(
        inputs,
        train=train,
        return_intermediate=return_intermediate,
        frame_paddings=frame_paddings,
    )
    if _contains(return_intermediate, 'spatiotemporal_features'):
      outputs['spatiotemporal_features'] = features

    embeddings = layers.AttenTokenPoolingLayer(
        name='atten_pooler',
        num_heads=self.encoder_params['num_heads'],
        hidden_dim=self.encoder_params['model_dim'],
        num_queries=1,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(features, paddings=None, train=train)
    embeddings = jnp.squeeze(embeddings, axis=-2)

    if _contains(return_intermediate, 'global_embeddings'):
      outputs['global_embeddings'] = embeddings

    logits = layers.FeedForward(
        name='projection',
        output_dim=self.num_classes,
        activation_fn=layers.identity,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(embeddings)
    return logits, outputs


class TextEncoder(layers.Module):
  """CoCa-style text encoder.

  Reference: https://arxiv.org/abs/2205.01917

  Attributes:
    vocabulary_size: Vocabulary size of the text tokens.
    num_class_tokens: Number of class tokens.
    enable_causal_atten: Whether to enable causal attention.
    model_dim: The model dimension.
    num_tfm_layers: Number of layers in this model.
    mlp_dim: The hidden layer dimension of FFN in Transformer layers.
    num_heads: Number of attention heads.
    enable_per_dim_scale: Whether to ensable rescaling of attention logits with
      1/sqrt(dim) factor.
    atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
      positive value is specified. May not be supported by a subclass.
    norm_policy: Policy for applying normalization wrt. transformations. Options
      are: (1) "pre", applied before transformation. (2) "primer_hybrid",
        applied before and after transformation. (3) "post", applied after
        transformation. (4) "post_skip", applied after the skip connection.
    scan: Whether to use `nn.remat` and`nn.scan`.
  """

  vocabulary_size: int = 128
  num_class_tokens: int = 0
  enable_causal_atten: bool = True
  model_dim: int = 768
  num_layers: int = 12
  mlp_dim: int = 3072
  num_heads: int = 12
  atten_logit_cap: float = 0.0
  norm_policy: str = 'pre'
  enable_per_dim_scale: bool = False
  scan: bool = False

  @nn.compact
  def __call__(
      self, inputs: Array, paddings: Array, train: bool = False
  ) -> Array:
    """Applies the text encoder to the inputs.

    Args:
      inputs: Input tensor of shape [B, N] including sequences of token ids.
      paddings: Optional [B, N] padding field of inputs.
      train: If the model is in the train mode.

    Returns:
      Output tensor of shape [B, N, D].
    """
    batch_size, seq_length = inputs.shape

    pos_emb = PositionalEmbedding(
        name='pos_emb',
        embedding_dim=self.model_dim,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(seq_length=seq_length)
    input_emb = Embedding(
        name='token_emb',
        num_classes=self.vocabulary_size,
        input_dim=self.model_dim,
        scale_sqrt_depth=True,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(inputs)
    features = input_emb + pos_emb

    if self.num_class_tokens > 0:
      cls_emb = self._cast_to_fprop_dtype(
          self.param(
              'cls_emb',
              nn.initializers.normal(stddev=1.0 / math.sqrt(self.model_dim)),
              [1, self.num_class_tokens, self.model_dim],
              self.dtype,
          )
      )
      cls_emb = jnp.tile(cls_emb, [batch_size, 1, 1])
      cls_emb *= self.model_dim**0.5
      features = jnp.concatenate([features, cls_emb], axis=-2)

      cls_paddings = jnp.zeros(
          [batch_size, self.num_class_tokens], dtype=paddings.dtype
      )
      paddings = jnp.concatenate([paddings, cls_paddings], axis=-1)

    features = layers.StackedTransformer(
        name='unimodal_transformer',
        num_layers=self.num_layers,
        hidden_dim=self.mlp_dim,
        num_heads=self.num_heads,
        atten_logit_cap=self.atten_logit_cap,
        norm_policy=self.norm_policy,
        internal_enable_per_dim_scale=self.enable_per_dim_scale,
        activation_fn=jax.nn.relu,
        enable_causal_atten=self.enable_causal_atten,
        scan=self.scan,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(features, paddings, train=train)
    features = layers.LayerNorm(
        name='unimodal_ln', dtype=self.dtype, fprop_dtype=self.fprop_dtype
    )(features)
    return features


class FactorizedVideoCLIP(layers.Module):
  """Video CLIP model with a factorized vision encoder."""

  # Vision parameters.
  patch_size: int = 18
  pos_emb_shape: tuple[int, int, int] = (16, 16, 16)
  num_spatial_layers: int = 12
  num_temporal_layers: int = 4
  mlp_dim: int = 3072
  num_auxiliary_layers: int = 0
  # Text parameters.
  vocabulary_size: int = 128
  enable_causal_atten: bool = True
  num_unimodal_layers: int = 12
  norm_policy: str = 'pre'
  # Shared parameters.
  model_dim: int = 768
  num_heads: int = 12
  atten_logit_cap: float = 0.0
  scan: bool = False

  @nn.compact
  def __call__(
      self,
      inputs: Array | None = None,
      text_token_ids: Array | None = None,
      text_paddings: Array | None = None,
      train: bool = False,
      normalize: bool = True,
      return_intermediate: bool | Collection[str] = False,
      frame_paddings: Array | None = None,
  ) -> tuple[Array | None, Array | None, dict[str, Array]]:
    """Computes predictions for `input_batch`.

    Args:
      inputs: Input frame image tensor of shape [B, T, H, W, 3] (H == W).
      text_token_ids: Input text token id tensor of shape [B, L].
      text_paddings: Input text paddings of shape [B, L]. Required if
        `text_token_ids` is not None.
      train: If the model is in the train mode.
      normalize: Whether to normalize the output embeddings.
      return_intermediate: A boolean for whether all intermediate features are
        returned, or a collection of intermediate feature names to return.
      frame_paddings: Optional binary tensor of shape [B, T] indicating padding.
        1 denotes padding frame.

    Returns:
      video_embeddings: Output contrastive video embeddings of shape [B, D].
        None if `inputs` is None.
      text_embeddings: Output contrastive text embeddings of shape [B, D]. None
        if `text_token_ids` is None.
      outputs: A dictionary of additional outputs, including `spatial_features`
        of shape [B, T * N, D], `spatiotemporal_features` of shape [B, T * N,
        D], and `frame_embeddings` of shape [B, T, D]. Empty if
        `return_intermediate` is False or does not contain `spatial_features`.
    """
    video_embeddings, text_embeddings, outputs = None, None, {}

    if inputs is not None:
      num_frames = inputs.shape[-4]
      vision_features, vision_outputs = FactorizedEncoder(
          name='vision_encoder',
          patch_size=self.patch_size,
          pos_emb_shape=self.pos_emb_shape,
          model_dim=self.model_dim,
          num_spatial_layers=self.num_spatial_layers,
          num_temporal_layers=self.num_temporal_layers,
          num_heads=self.num_heads,
          mlp_dim=self.mlp_dim,
          atten_logit_cap=self.atten_logit_cap,
          norm_policy='pre',
          scan=self.scan,
          dtype=self.dtype,
          fprop_dtype=self.fprop_dtype,
      )(
          inputs,
          train=train,
          return_intermediate=return_intermediate,
          frame_paddings=frame_paddings,
      )
      outputs.update(vision_outputs)
      if _contains(return_intermediate, 'spatiotemporal_features'):
        outputs['spatiotemporal_features'] = vision_features

      if self.num_auxiliary_layers > 0:
        vision_features = VisionTransformer(
            name='auxiliary_encoder',
            num_tfm_layers=self.num_auxiliary_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            atten_logit_cap=self.atten_logit_cap,
            norm_policy='pre',
            scan=self.scan,
            dtype=self.dtype,
            fprop_dtype=self.fprop_dtype,
        )(vision_features, train=train)

      pooling_layer = layers.AttenTokenPoolingLayer(
          name='contrastive_vision_pooler',
          hidden_dim=self.model_dim * 4,
          num_heads=self.num_heads,
          num_queries=1,
          dtype=self.dtype,
          fprop_dtype=self.fprop_dtype,
      )
      video_embeddings = pooling_layer(vision_features, None, train=train)

      # Squeeze the query dimension in the pooler output.
      video_embeddings = jnp.squeeze(video_embeddings, axis=-2)
      if normalize:
        video_embeddings = _l2_normalize(video_embeddings, axis=-1)

      if _contains(return_intermediate, 'frame_embeddings'):
        frame_features = einshape.jax_einshape(
            'b(tn)d->(bt)nd', vision_features, t=num_frames
        )
        frame_embeddings = pooling_layer(frame_features, None, train=train)
        frame_embeddings = jnp.squeeze(frame_embeddings, axis=-2)
        frame_embeddings = einshape.jax_einshape(
            '(bt)d->btd', frame_embeddings, t=num_frames
        )
        if normalize:
          frame_embeddings = _l2_normalize(frame_embeddings, axis=-1)
        outputs['frame_embeddings'] = frame_embeddings

    if text_token_ids is not None:
      assert text_paddings is not None, 'Text paddings are required.'
      text_features = TextEncoder(
          name='text_encoder',
          vocabulary_size=self.vocabulary_size,
          num_class_tokens=1,
          enable_causal_atten=self.enable_causal_atten,
          model_dim=self.model_dim,
          num_layers=self.num_unimodal_layers,
          num_heads=self.num_heads,
          mlp_dim=self.model_dim * 4,
          atten_logit_cap=self.atten_logit_cap,
          norm_policy=self.norm_policy,
          scan=self.scan,
          dtype=self.dtype,
          fprop_dtype=self.fprop_dtype,
      )(text_token_ids, text_paddings, train=train)

      # Take the last token (i.e., class token) as the text embedding.
      text_embeddings = text_features[:, -1]
      if normalize:
        text_embeddings = _l2_normalize(text_embeddings, axis=-1)

    return video_embeddings, text_embeddings, outputs
