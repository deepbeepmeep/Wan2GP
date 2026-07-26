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

"""VideoPrism Flax layers."""

from collections.abc import Callable
import functools
import string
from typing import Any
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np

Array = jax.Array
ActivationFunc = Callable[[Array], Array]
Initializer = nn.initializers.Initializer

default_kernel_init = nn.initializers.lecun_normal()
gelu = functools.partial(jax.nn.gelu, approximate=False)


def identity(x: Array) -> Array:
  """Identity activation."""
  return x


def _get_large_negative_number(dtype: jax.typing.DTypeLike) -> Array:
  """Returns a large-magnitude negative value for the given dtype."""
  # -0.7 is a float64 in JAX. Explicit cast output to target dtype.
  if jnp.issubdtype(dtype, jnp.inexact):
    dtype_max = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtype_max = jnp.iinfo(dtype).max
  else:
    raise ValueError('Unsupported dtype for inputs.')
  return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def _apply_mask_to_logits(logits: Array, mask: Array) -> Array:
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a float32 tensor where 0 represents true and values
  below a large negative number (here set to
  _get_large_negative_number(jnp.float32) / 2) represent false. Applying the
  mask leaves the logits alone in the true case and replaces them by
  _get_large_negative_number(jnp.float32) in the false case. Previously, this
  was done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the float32 values in memory rather than
  just the predicate. This implementation avoids that problem.

  Args:
    logits: A jax.Array of logit values.
    mask: A jax.Array (float32) of mask values with the encoding described in
      the function documentation.

  Returns:
    Masked logits.
  """
  min_value = _get_large_negative_number(logits.dtype)
  return jnp.where((mask >= min_value * 0.5), logits, min_value)


def _convert_paddings_to_mask(
    paddings: Array, dtype: jax.typing.DTypeLike = jnp.float32
) -> Array:
  """Converts binary paddings to a logit mask ready to add to attention matrix.

  Args:
    paddings: A binary jax.Array of shape [B, T], with 1 denoting padding token.
    dtype: Data type of the input.

  Returns:
    A jax.Array of shape [B, 1, 1, T] ready to be added to attention logits.
  """
  attention_mask = paddings[:, jnp.newaxis, jnp.newaxis, :]
  attention_mask *= _get_large_negative_number(dtype)
  return attention_mask


def _causal_mask(input_t: Array) -> Array:
  """Computes and returns causal mask.

  Args:
    input_t: A jax.Array of shape [B, T, D].

  Returns:
    An attention_mask jax.Array of shape [1, 1, T, T]. Attention mask has
    already been converted large negative values.
  """
  assert jnp.issubdtype(input_t.dtype, jnp.floating), input_t.dtype
  large_negative_number = _get_large_negative_number(input_t.dtype)
  t = input_t.shape[-2]
  col_idx = jnp.tile(jnp.arange(t)[jnp.newaxis, :], [t, 1])
  row_idx = jnp.tile(jnp.arange(t)[:, jnp.newaxis], [1, t])
  mask = (row_idx < col_idx).astype(input_t.dtype) * large_negative_number
  return mask[jnp.newaxis, jnp.newaxis, :, :]


def _merge_masks(a: Array, b: Array) -> Array:
  """Merges two masks.

  This function merges two masks with the same shape, where the smaller value
  will be chosen at the same position. Log-scale mask is expected but 0/1 mask
  is also fine.

  Args:
    a: A jax.Array of shape [1|B, 1, 1|T, S].
    b: A jax.Array of shape [1|B, 1, 1|T, S].

  Returns:
    A jax.Array of shape [1|B, 1, 1|T, S].
  """

  def expand_t(key_mask):
    """Expands the 1D mask to the 2D mask.

    Given [[1, 1, 0, 0]], this function returns the following mask,
    1 1 0 0
    1 1 0 0
    0 0 0 0
    0 0 0 0

    Args:
      key_mask: A jax.Array of the input 1D mask.

    Returns:
      A jax.Array of the expanded 2D mask.
    """
    query_mask = jnp.transpose(key_mask, [0, 1, 3, 2])
    return jnp.minimum(query_mask, key_mask)

  if a.shape[-2] != b.shape[-2]:
    if a.shape[-2] == 1:
      a = expand_t(a)
    else:
      assert b.shape[-2] == 1
      b = expand_t(b)

  assert a.shape[-3:] == b.shape[-3:], f'a.shape={a.shape}, b.shape={b.shape}.'
  return jnp.minimum(a, b)


def compute_attention_masks_for_fprop(
    inputs: Array,
    paddings: Array,
    causal_attention: bool = False,
) -> Array:
  """Computes attention mask from inputs and paddings for fprop.

  Args:
    inputs: Input sequence jax.Array of shape [B, T, H].
    paddings: Input paddings jax.Array of shape [B, T].
    causal_attention: Boolean to apply causal masking.

  Returns:
    attention_mask: Attention mask jax.Array ready to be added to logits for
      self-attention of shape [1|B, 1, 1|T, T].
  """
  # Get paddings mask to [B, 1, 1, T].
  attention_mask = _convert_paddings_to_mask(paddings, inputs.dtype)

  # Causal mask of shape [1, 1, T, T].
  if causal_attention:
    causal_mask = _causal_mask(inputs)
    attention_mask = _merge_masks(attention_mask, causal_mask)

  return attention_mask


class Module(nn.Module):
  """Base class for layers with dtype configured.

  Attributes:
    dtype: Default dtype for all variables.
    fprop_dtype: Activations dtype to use.
  """

  dtype: jnp.dtype = jnp.float32
  fprop_dtype: jnp.dtype = jnp.float32

  @nn.nowrap
  def _cast_to_fprop_dtype(self, value: Any) -> Any:
    """Casts values to the desired dtype."""

    def _cast(x):
      if x is None:
        return None
      if self.fprop_dtype != x.dtype:
        if jnp.issubdtype(x.dtype, jnp.floating):
          return x.astype(self.fprop_dtype)
      return x

    return jax.tree_util.tree_map(_cast, value)


class LayerNorm(Module):
  """Layer normalization.

  Attributes:
    direct_scale: Whether to apply scale directly without a +1.0. Var is
      initialized to 1.0 instead when True.
    epsilon: Tiny value to guard rsqrt.
    use_scale: Whether to use a learned scaling.
    use_bias: Whether to use bias.
    reductions_in_fp32: Whether to compute mean and variance in fp32.
      Recommended for stable training on GPUs.
  """

  direct_scale: bool = False
  epsilon: float = 1e-6
  use_scale: bool = True
  use_bias: bool = True
  reductions_in_fp32: bool = False

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies layer norm to inputs.

    Args:
      inputs: A jax.Array for the inputs of shape [..., dim].

    Returns:
      A jax.Aray for the normalized inputs of the same shape.
    """
    inputs_dtype = inputs.dtype
    if self.reductions_in_fp32:
      inputs = inputs.astype(jnp.float32)
    mean = jnp.mean(inputs, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=[-1], keepdims=True)
    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.epsilon)
    if self.reductions_in_fp32:
      normed_inputs = normed_inputs.astype(inputs_dtype)

    input_dim = inputs.shape[-1]
    if self.use_scale:
      init_value = 1.0 if self.direct_scale else 0.0
      scale = self._cast_to_fprop_dtype(
          self.param(
              'scale',
              nn.initializers.constant(init_value),
              [input_dim],
              self.dtype,
          )
      )
      if not self.direct_scale:
        scale += 1.0
      normed_inputs *= scale
    if self.use_bias:
      bias = self._cast_to_fprop_dtype(
          self.param(
              'bias',
              nn.initializers.zeros_init(),
              [input_dim],
              self.dtype,
          )
      )
      normed_inputs += bias
    return normed_inputs


class FeedForward(Module):
  """Feedforward layer with activation.

  Attributes:
    output_dim: Depth of the output.
    has_bias: Adds bias weights or not.
    activation_fn: Activation function to use.
    weight_init: Initializer function for the weight matrix.
    bias_init: Initializer function for the bias.
  """

  output_dim: int = 0
  has_bias: bool = True
  activation_fn: ActivationFunc = nn.relu
  weight_init: Initializer = default_kernel_init
  bias_init: Initializer = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:

    def _promote_dtype(x, kernel, bias, dtype):
      """Promotes the dtype of the arrays to the desired dtype."""
      del dtype
      # To be compatible with other layers, we do not promote the inputs as they
      # are expected to be in the `fprop_dtype`.
      return (
          x,
          self._cast_to_fprop_dtype(kernel),
          self._cast_to_fprop_dtype(bias),
      )

    projected_inputs = nn.Dense(
        self.output_dim,
        use_bias=self.has_bias,
        kernel_init=self.weight_init,
        bias_init=self.bias_init,
        name='linear',
        param_dtype=self.dtype,
        promote_dtype=_promote_dtype,
    )(inputs)
    return self.activation_fn(projected_inputs)


class TransformerFeedForward(Module):
  """Transformer feedforward layer with residual connection and dropout.

  Attributes:
    output_dim: Depth of the output. The value of input_dim will be used when
      output_dim is 0. Must be equal to input_dim if add_skip_connection=True.
    hidden_dim: Hidden dimension of FFN.
    has_bias: Adds bias weights to Feedforward or not.
    activation_fn: Activation function to use.
    residual_dropout_prob: Residual dropout.
    relu_dropout_prob: FFN dropout.
    add_skip_connection: Whether to add residual connection.
    residual_weight: Weight of the residual connection. Output = fn(x) *
      residual_weight + x.
    norm_policy: Policy for applying normalization wrt. transformations. Options
      are: (1) "pre", applied before transformation. (2) "primer_hybrid",
        applied before and after transformation. (3) "post", applied after
        transformation, (4) "post_skip", applied after the skip connection.
  """

  output_dim: int = 0
  hidden_dim: int = 0
  has_bias: bool = True
  activation_fn: ActivationFunc = nn.relu
  residual_dropout_prob: float = 0.0
  relu_dropout_prob: float = 0.0
  add_skip_connection: bool = True
  residual_weight: float = 1.0
  norm_policy: str = 'pre'

  @nn.nowrap
  def _make_ln(self, name: str) -> LayerNorm:
    """Makes a LayerNorm module."""
    return LayerNorm(
        name=name,
        use_bias=self.has_bias,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )

  @nn.nowrap
  def _make_ffn(
      self, output_dim: int, name: str, skip_activation: bool = False
  ) -> FeedForward:
    """Makes a FeedForward module."""
    return FeedForward(
        name=name,
        output_dim=output_dim,
        has_bias=self.has_bias,
        activation_fn=identity if skip_activation else self.activation_fn,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )

  @nn.compact
  def __call__(
      self, inputs: Array, paddings: Array | None, train: bool
  ) -> Array:
    residual = inputs
    output_dim = self.output_dim
    if output_dim == 0:
      output_dim = inputs.shape[-1]
    if self.add_skip_connection and output_dim != inputs.shape[-1]:
      raise ValueError(
          'Skip connections are only supported when input_dim == output_dim '
          f'but got {self.input_dim} != {output_dim}'
      )

    # Expand paddings to last dim if not None to have shape [batch, seq_len, 1].
    if paddings is not None:
      paddings = jnp.expand_dims(paddings, axis=-1)

    if self.norm_policy == 'primer_hybrid':
      inputs = self._make_ln(name='pre_layer_norm')(inputs)
    elif self.norm_policy == 'pre':
      inputs = self._make_ln(name='layer_norm')(inputs)

    # Apply first FFN layer.
    activations = self._make_ffn(self.hidden_dim, name='ffn_layer1')(inputs)

    # Apply paddings if not None.
    if paddings is not None:
      activations *= 1.0 - paddings

    # Apply RELU dropout.
    activations = nn.Dropout(self.relu_dropout_prob, name='relu_dropout')(
        activations, deterministic=not train
    )
    # Apply second FFN layer.
    outputs = self._make_ffn(
        output_dim, name='ffn_layer2', skip_activation=True
    )(activations)

    # Apply paddings if not None.
    if paddings is not None:
      outputs *= 1.0 - paddings

    # Apply Primer normalization before dropout.
    if self.norm_policy == 'primer_hybrid':
      outputs = self._make_ln(name='post_layer_norm')(outputs)
    elif self.norm_policy == 'post':
      outputs = self._make_ln(name='layer_norm')(outputs)

    # Apply residual dropout.
    outputs = nn.Dropout(self.residual_dropout_prob, name='residual_dropout')(
        outputs, deterministic=not train
    )
    # Apply skip connection.
    if self.add_skip_connection:
      outputs = residual + outputs * self.residual_weight

    if self.norm_policy == 'post_skip':
      outputs = self._make_ln(name='layer_norm')(outputs)

    return outputs


class AttentionProjection(Module):
  """Layer that computes multi heads projection.

  This layer is expected to be used within DotProductAttention below.

  Attributes:
    output_dim: Input dimension.
    num_heads: Number of attention heads.
    dim_per_head: Size of each head.
    is_output_projection: Whether it is out projection or not. If False, we use
      "...D,DNH->...NH" for query,key,value projection. Otherwise we use
      "...NH,DNH->...D" for output projection.
    use_bias: Whether to add bias in projection or not.
  """

  output_dim: int = 0
  num_heads: int = 0
  dim_per_head: int = 0
  is_output_projection: bool = False
  use_bias: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Computes the multi headed projection for inputs.

    Args:
      inputs: A jax.Array with shape [..., num_heads, dim_per_head] if
        is_output_projection is True or [..., input_dim] otherwise.

    Returns:
      The projected jax.Array with shape [..., input_dim] if
      is_output_projection is True or [..., num_heads, dim_per_head]
      otherwise.
    """
    # Sort the available symbols to avoid nondeterminism.
    eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('DHN')))
    output_dim = (
        self.output_dim if self.is_output_projection else inputs.shape[-1]
    )
    rank = len(inputs.shape)

    hd_shape = [self.num_heads, self.dim_per_head]
    pc_shape = [output_dim] + hd_shape
    w = self._cast_to_fprop_dtype(
        self.param('w', default_kernel_init, pc_shape, self.dtype)
    )

    if self.is_output_projection:
      assert inputs.shape[-2:] == (self.num_heads, self.dim_per_head)
      batch_eqn = eqn_sym[: (rank - 2)]
      eqn = f'{batch_eqn}NH,DNH->{batch_eqn}D'
    else:
      batch_eqn = eqn_sym[: (rank - 1)] if rank else '...'
      eqn = f'{batch_eqn}D,DNH->{batch_eqn}NH'

    ret = jnp.einsum(eqn, inputs, w)
    if self.use_bias:
      b = self._cast_to_fprop_dtype(
          self.param(
              'b',
              nn.initializers.zeros_init(),
              [output_dim] if self.is_output_projection else hd_shape,
              self.dtype,
          )
      )
      ret += b
    return ret


class PerDimScale(Module):
  """A layer to scale individual dimensions of the input."""

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Returns per_dim_scale * inputs / jnp.sqrt(dim)).

    Args:
      inputs: A jax.Array with shape [..., dim].

    Returns:
      outputs: A jax.Array with shape [..., dim].
    """
    dim = inputs.shape[-1]
    per_dim_scale = self._cast_to_fprop_dtype(
        self.param(
            'per_dim_scale', nn.initializers.zeros_init(), [dim], self.dtype
        )
    )

    # 1.0/jax.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
    # can avoid unnecessary XLA op fusion mess on TPU.
    r_softplus_0 = 1.442695041
    scale = jnp.array(r_softplus_0 / np.sqrt(dim), dtype=self.fprop_dtype)
    scale *= jax.nn.softplus(per_dim_scale)
    return inputs * scale


class DotProductAttention(Module):
  """Dot-product attention with multiple attention heads.

  Attributes:
    hidden_dim: Number of hidden nodes.
    num_heads: Number of attention heads.
    dim_per_head: Dimension of each attention head. If None then dim_per_head ==
      hidden_dim // num_heads.
    atten_dropout_prob: Probability at which we apply dropout to the attention
      weights.
    use_bias: Whether to use bias for projection layers.
    internal_enable_query_scale: Internal. Enable scaling of query vector.
    internal_enable_per_dim_scale: Internal. Setting to False disables rescaling
      of attention logits with 1/sqrt(dim) factor. Some Transformer variants
      (GShard, T5) use internal_enable_per_dim_scale=False and adjust
      initialization of the linear transformations(einsums), in conjunction with
      Adafactor optimizer.
    scale_query_by_dim_per_head: whether to scale the query by dim_per_head,
      instead of default hidden_dim // num_heads (only activated when
      internal_enable_per_dim_scale = False).
    scale_logits_by_head_dims: Enables a 1/sqrt(head dim) scaling to the logits.
      This occurs prior to logit cap, if any.
    atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
      positive value is specified. May not be supported by a subclass.
    use_qk_norm: If QK norm is used.
  """

  hidden_dim: int = 0
  num_heads: int = 1
  dim_per_head: int | None = None
  atten_dropout_prob: float = 0.0
  use_bias: bool = True
  internal_enable_query_scale: bool = True
  internal_enable_per_dim_scale: bool = True
  scale_query_by_dim_per_head: bool = False
  scale_logits_by_head_dims: bool = False
  atten_logit_cap: float = 0.0
  use_qk_norm: bool = False

  def _scale_query(self, query: Array) -> Array:
    """Scales the query vector if enabled."""
    if not self.internal_enable_query_scale:
      return query
    if self.internal_enable_per_dim_scale:
      query = PerDimScale(
          name='per_dim_scale', dtype=self.dtype, fprop_dtype=self.fprop_dtype
      )(query)
    else:
      if self.scale_query_by_dim_per_head and self.dim_per_head is not None:
        dim_per_head = self.dim_per_head
      else:
        dim_per_head = self.hidden_dim // self.num_heads

      query *= dim_per_head**-0.5
    return query

  def _cap_logits(self, logits: Array) -> Array:
    """Caps the logits by p.atten_logit_cap with tanh, if enabled."""
    if not self.atten_logit_cap or self.atten_logit_cap <= 0.0:
      return logits
    cap = jnp.array(self.atten_logit_cap, dtype=self.fprop_dtype)
    # Note that since this caps the negative side as well, caller must defer the
    # pad-with-very-negative-logits logic to after this function returns.
    logits = cap * jnp.tanh(logits / cap)
    return logits

  def _atten_logits(self, query: Array, key: Array) -> Array:
    """Computes logits from query and key."""
    logits = jnp.einsum('BTNH,BSNH->BNTS', query, key)
    return logits

  def _dot_atten(
      self,
      query: Array,
      key: Array,
      value: Array,
      atten_mask: Array,
      train: bool,
  ) -> tuple[Array, Array]:
    """Main attention function.

    Args:
      query: A jax.Array of shape [B, T, N, H].
      key: A jax.Array of shape [B, S, N, H].
      value: A jax.Array of shape [B, S, N, H].
      atten_mask: A jax.Array of shape [1|B, 1, 1|T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      train: Whether the model is in the train mode.

    Returns:
      encoded: A jax.Array of shape [B, T, N, H].
      atten_probs: A jax.Array of shape [B, N, T, S].
    """
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), 'q, k, v batch dims must match.'
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), 'q, k, v num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
    # If only padding bias is supplied, then atten_mask can be [B, 1, 1, S]
    # since each target token is prohibited from attending to the same set of
    # source tokens. In this case tiling is inefficient and unnecessary.
    # If there is no padding mask, and only causal mask then the shape can be
    # [1, 1, T, S].
    assert atten_mask.ndim == 4 and atten_mask.shape[-1] == key.shape[-3]
    assert atten_mask.shape[-2] in [query.shape[-3], 1]
    assert atten_mask.shape[0] in [key.shape[0], 1]

    query = self._scale_query(query)
    logits = self._atten_logits(query, key)

    if self.scale_logits_by_head_dims:
      logits = jnp.multiply(logits, 1.0 / np.sqrt(key.shape[-1]))

    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking.
    padded_logits = _apply_mask_to_logits(logits, atten_mask)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(self.fprop_dtype)
    # Apply attention dropout.
    probs = nn.Dropout(self.atten_dropout_prob, name='atten_dropout')(
        probs, deterministic=not train
    )
    # Compute the attention context.
    encoded = jnp.einsum('BNTS,BSNH->BTNH', probs, value)
    return encoded, probs

  @nn.nowrap
  def _project_input(self, name: str, dim_per_head: int) -> AttentionProjection:
    """Builds an AttentionProjection module."""
    return AttentionProjection(
        name=name,
        num_heads=self.num_heads,
        dim_per_head=dim_per_head,
        use_bias=self.use_bias,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )

  @nn.nowrap
  def _make_ln(self, name: str) -> LayerNorm:
    """Makes a LayerNorm module."""
    return LayerNorm(
        name=name,
        use_bias=self.use_bias,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )

  @nn.compact
  def __call__(
      self,
      query_vec: Array,
      key_vec: Array,
      value_vec: Array,
      atten_mask: Array,
      train: bool,
  ) -> tuple[Array, Array]:
    """Computes the value vector given the current query output.

    Args:
      query_vec: jax.Array of shape [B, T, D].
      key_vec: jax.Array of shape [B, S, D].
      value_vec: jax.Array of shape [B, S, D].
      atten_mask: jax.Array of shape [1|B, 1, 1|T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      train: If the model is in the train mode.

    Returns:
      encoded: jax.Array of shape [B, T, D].
      atten_probs: jax.Array of shape [B, N, T, S].
    """
    dim_per_head = self.dim_per_head
    if dim_per_head is None:
      dim_per_head = self.hidden_dim // self.num_heads
      assert (
          dim_per_head * self.num_heads == self.hidden_dim
      ), f'{dim_per_head} * {self.num_heads} != {self.hidden_dim}'

    # Project inputs to key, value and query, respectively has shape
    # [B, S, N, H], [B, S, N, H], and [B, T, N, H].
    query_proj = self._project_input('query', dim_per_head)(query_vec)
    key_proj = self._project_input('key', dim_per_head)(key_vec)
    value_proj = self._project_input('value', dim_per_head)(value_vec)

    if self.use_qk_norm:
      query_proj = self._make_ln(name='layer_norm_q')(query_proj)
      key_proj = self._make_ln(name='layer_norm_k')(key_proj)

    encoded, atten_probs = self._dot_atten(
        query_proj, key_proj, value_proj, atten_mask, train=train
    )

    # Post projection. Setting is_output_projection=True to set the projection
    # direction from hidden dim to input dim. Output projection follows
    # query_input_dim.
    query_input_dim = query_vec.shape[-1]
    encoded = AttentionProjection(
        name='post',
        output_dim=query_input_dim,
        num_heads=self.num_heads,
        dim_per_head=dim_per_head,
        is_output_projection=True,
        use_bias=self.use_bias,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(encoded)
    return encoded, atten_probs


class Transformer(Module):
  """Transformer layer with multi-headed attention.

  Attributes:
    hidden_dim: Hidden dimension of FFN layer.
    num_heads: Number of heads in self-attention.
    dim_per_head: Dimension of each attention head. If None then dim_per_head ==
      hidden_dim // num_heads.
    atten_dropout_prob: Probability at which we apply dropout to the attention
      weights.
    residual_dropout_prob: Probability at which we apply dropout to the residual
      layers, such that, residual(x, y) = (x + dropout(y)).
    relu_dropout_prob: Probability at which we apply dropout to the FFN layers.
    norm_policy: Policy for applying normalization wrt. transformations. Options
      are: (1) "pre", applied before transformation. (2) "primer_hybrid",
        applied before and after transformation. (3) "post", applied after
        transformation. (4) "post_skip", applied after the skip connection.
    use_bias: Whether to use bias.
    activation_fn: Activation function to use.
    internal_enable_per_dim_scale: Internal. Setting to False disables rescaling
      of attention logits with 1/sqrt(dim) factor.
    atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
      positive value is specified. May not be supported by a subclass.
  """

  hidden_dim: int = 0
  num_heads: int = 0
  dim_per_head: int | None = None
  atten_dropout_prob: float = 0.0
  residual_dropout_prob: float = 0.0
  relu_dropout_prob: float = 0.0
  norm_policy: str = 'pre'
  use_bias: bool = True
  activation_fn: ActivationFunc = nn.relu
  internal_enable_per_dim_scale: bool = True
  atten_logit_cap: float = 0.0

  @nn.nowrap
  def _make_ln(self, name: str) -> LayerNorm:
    """Makes a LayerNorm module."""
    return LayerNorm(
        name=name,
        use_bias=self.use_bias,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      paddings: Array,
      atten_mask: Array,
      train: bool,
  ) -> Array:
    """Transformer decoder layer.

    Args:
      inputs: Input sequence jax.Array of shape [B, T, H].
      paddings: Input paddings jax.Array of shape [B, T] (only used in FFN).
      atten_mask: Self attention mask ready to add to the logits. It can be of
        shape [1|B, 1, 1|T, T] which is broadcast compatible with the
        self-attention matrix of shape [B, N, T, T]. This is assumed to have
        combined paddings, causal masking as well as segment maskings.
      train: Whether the model is in the train mode.

    Returns:
      The fflayer output with shape [B, T, D].
    """

    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self._make_ln(name='pre_layer_norm')(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self._make_ln(name='layer_norm')(inputs)
    else:
      inputs_normalized = inputs

    # Compute self-attention, key/value vectors are the input itself.
    atten_outputs, _ = DotProductAttention(
        name='self_attention',
        hidden_dim=inputs_normalized.shape[-1],
        num_heads=self.num_heads,
        dim_per_head=self.dim_per_head,
        atten_dropout_prob=self.atten_dropout_prob,
        use_bias=self.use_bias,
        internal_enable_per_dim_scale=self.internal_enable_per_dim_scale,
        atten_logit_cap=self.atten_logit_cap,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        atten_mask=atten_mask,
        train=train,
    )

    if self.norm_policy == 'primer_hybrid':
      atten_outputs = self._make_ln(name='post_layer_norm')(atten_outputs)
    elif self.norm_policy == 'post':
      atten_outputs = self._make_ln(name='layer_norm')(atten_outputs)

    # Residual dropout and connection.
    atten_outputs = nn.Dropout(
        self.residual_dropout_prob, name='residual_dropout'
    )(atten_outputs, deterministic=not train)
    atten_outputs += inputs

    if self.norm_policy == 'post_skip':
      atten_outputs = self._make_ln(name='layer_norm')(atten_outputs)

    # Apply FFN layer.
    outputs = TransformerFeedForward(
        name='ff_layer',
        hidden_dim=self.hidden_dim,
        has_bias=self.use_bias,
        activation_fn=self.activation_fn,
        residual_dropout_prob=self.residual_dropout_prob,
        relu_dropout_prob=self.relu_dropout_prob,
        norm_policy=self.norm_policy,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(atten_outputs, paddings=paddings, train=train)
    return outputs


class Repeat(nn.Module):
  """A generic repeat layer with `nn.remat` and`nn.scan`.

  Attributes:
    block_fn: The block function to repeat.
    times: The number of times to repeat block.
    checkpoint_policy: Checkpoint policy for `nn.remat`.
  """

  block_fn: Callable[..., Any]
  times: int = 0
  checkpoint_policy: str = 'nothing_saveable'

  def __call__(
      self,
      inputs: Array,
      *args: Any,
      **kwargs: Any,
  ) -> Any:
    """Forwards inputs through the block layer stack.

    Block outputs are expected to be of the same structure as inputs.

    Args:
      inputs: A NestedMap of inputs that goes through the block layer stack.
      *args: Positional args to be passed to the forward method.
      **kwargs: Keyward args to be passed to the forward method.

    Returns:
      Output from the last layer.
    """
    return self.call_with_custom_method(
        inputs,
        *args,
        main_fn=self.block_fn,
        **kwargs,
    )

  def call_with_custom_method(
      self,
      inputs: Array,
      *args: Any,
      main_fn: Callable[..., Any],
      **kwargs: Any,
  ) -> Any:
    """Similar to __call__, but allows a custom way to create a layer method."""

    def body_fn(fn, layer_inputs):
      return fn(layer_inputs, *args, **kwargs), None

    rematted_body_fn = nn.remat(
        body_fn,
        prevent_cse=False,
        policy=getattr(jax.checkpoint_policies, self.checkpoint_policy, None),
    )
    scan_fn = nn.scan(
        rematted_body_fn,
        variable_axes={'params': 0},
        split_rngs={'params': True, 'dropout': True},
        length=self.times,
    )
    outputs, _ = scan_fn(main_fn, inputs)
    return outputs


class StackedTransformer(Module):
  """A stack of Transformer layers.

  Attributes:
    num_layers: Number of layers in this stack.
    hidden_dim: The hidden layer dimension of FFN in Transformer layers.
    num_heads: Number of attention heads.
    dim_per_head: Dimension of each attention head. If None then dim_per_head ==
      model_dims // num_heads.
    dropout_prob: Apply dropout at this prob at various places.
    atten_dropout_prob: Probability at which we apply dropout to the attention
      weights.
    residual_dropout_prob: Probability at which we apply dropout to the residual
      layers, such that, residual(x, y) = (x + dropout(y)).
    relu_dropout_prob: Probability at which we apply dropout to the FFN layers.
    input_dropout_prob: Dropout probability applied to the input before any
      processing happens.
    norm_policy: Policy for applying normalization wrt. transformations. Options
      are: (1) "pre", applied before transformation. (2) "primer_hybrid",
        applied before and after transformation. (3) "post", applied after
        transformation. (4) "post_skip", applied after the skip connection.
    use_bias: Whether to use bias.
    activation_fn: Activation function to use.
    internal_enable_per_dim_scale: Internal. Setting to False disables rescaling
      of attention logits with 1/sqrt(dim) factor.
    atten_logit_cap: Cap the absolute values of logits by tanh. Enabled when a
      positive value is specified. May not be supported by a subclass.
    enable_causal_atten: Whether to enable causal attention.
    scan: Whether to use `nn.remat` and`nn.scan`.
  """

  num_layers: int = 0
  hidden_dim: int = 0
  num_heads: int = 0
  dim_per_head: int | None = None
  dropout_prob: float = 0.0
  atten_dropout_prob: float | None = None
  residual_dropout_prob: float | None = None
  relu_dropout_prob: float | None = None
  input_dropout_prob: float = 0.0
  norm_policy: str = 'pre'
  use_bias: bool = True
  activation_fn: ActivationFunc = nn.relu
  internal_enable_per_dim_scale: bool = True
  atten_logit_cap: float = 0.0
  enable_causal_atten: bool = False
  scan: bool = False

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      paddings: Array,
      train: bool,
  ) -> Array:
    """Stacked Transformer layer.

    Args:
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      train: If the model is in the train mode.

    Returns:
      Output vector with shape [B, T, D].
    """

    atten_mask = compute_attention_masks_for_fprop(
        inputs, paddings, causal_attention=self.enable_causal_atten
    )

    outputs = inputs
    if self.input_dropout_prob > 0.0:
      outputs = nn.Dropout(self.input_dropout_prob, name='input_dropout')(
          outputs, deterministic=not train
      )

    transformer_kwargs = dict(
        num_heads=self.num_heads,
        dim_per_head=self.dim_per_head,
        hidden_dim=self.hidden_dim,
        atten_dropout_prob=self.atten_dropout_prob or self.dropout_prob,
        residual_dropout_prob=self.residual_dropout_prob or self.dropout_prob,
        relu_dropout_prob=self.relu_dropout_prob or self.dropout_prob,
        norm_policy=self.norm_policy,
        use_bias=self.use_bias,
        activation_fn=self.activation_fn,
        internal_enable_per_dim_scale=self.internal_enable_per_dim_scale,
        atten_logit_cap=self.atten_logit_cap,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )
    if self.scan:
      block_fn = Transformer(name='x_layers', **transformer_kwargs)
      outputs = Repeat(block_fn=block_fn, times=self.num_layers)(
          outputs, paddings, atten_mask, train
      )
    else:
      for i in range(self.num_layers):
        outputs = Transformer(name=f'x_layers_{i}', **transformer_kwargs)(
            outputs, paddings, atten_mask, train
        )
    return outputs


class AttenTokenPoolingLayer(Module):
  """Attentional token pooling layer.

  Attributes:
    query_dim: The query dimension of attention. If None then query_dim ==
      input_dim.
    hidden_dim: The hidden layer dimension of FFN in Transformer layers.
    num_heads: Number of attention heads.
    num_queries: Number of attention queries.
    add_layer_norm: Whether to apply layer norm to the pooled tokens.
    dropout_prob: The probability of dropout on the pooled tokens.
    use_qk_norm: If QK norm is used.
    use_bias: Whether to use bias.
    internal_enable_per_dim_scale: Internal. Setting to False disables rescaling
      of attention logits with 1/sqrt(dim) factor.
  """

  query_dim: int | None = None
  hidden_dim: int = 0
  num_heads: int = 1
  num_queries: int = 1
  add_layer_norm: bool = True
  dropout_prob: float = 0.0
  use_qk_norm: bool = False
  use_bias: bool = True
  internal_enable_per_dim_scale: bool = True

  @nn.compact
  def __call__(
      self,
      tokens: Array,
      paddings: Array | None,
      train: bool,
  ) -> Array:
    """Computes the pooled tokens for inputs.

    Args:
      tokens: Input tokens of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      train: If the model is in the train mode.

    Returns:
      Output vector with shape [B, N, D].
    """
    input_dim = tokens.shape[-1]
    query_dim = self.query_dim or input_dim
    hidden_dim = self.hidden_dim if self.hidden_dim > 0 else 4 * input_dim
    batch_size, seq_length = tokens.shape[0], tokens.shape[-2]

    query = self._cast_to_fprop_dtype(
        self.param(
            'pooling_attention_query',
            default_kernel_init,
            [self.num_queries, query_dim],
            self.dtype,
        )
    )
    query = jnp.tile(query[jnp.newaxis, :, :], [batch_size, 1, 1])

    if paddings is None:
      paddings = jnp.zeros([batch_size, seq_length], dtype=tokens.dtype)

    atten_mask = _convert_paddings_to_mask(paddings, dtype=paddings.dtype)
    outputs, _ = DotProductAttention(
        name='pooling_attention',
        hidden_dim=hidden_dim,
        num_heads=self.num_heads,
        use_bias=self.use_bias,
        internal_enable_per_dim_scale=self.internal_enable_per_dim_scale,
        use_qk_norm=self.use_qk_norm,
        dtype=self.dtype,
        fprop_dtype=self.fprop_dtype,
    )(
        query,
        tokens,
        tokens,
        atten_mask=atten_mask,
        train=train,
    )

    if self.add_layer_norm:
      outputs = LayerNorm(
          name='pooling_attention_layer_norm',
          dtype=self.dtype,
          fprop_dtype=self.fprop_dtype,
      )(outputs)

    if self.dropout_prob > 0.0:
      outputs = nn.Dropout(self.dropout_prob, name='attention_dropout')(
          outputs, deterministic=not train
      )

    return outputs
