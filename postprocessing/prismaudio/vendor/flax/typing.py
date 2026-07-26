# Copyright 2024 The Flax Authors.
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
from __future__ import annotations

import abc
from collections import deque
import functools
from functools import partial
from typing import (
  Any,
  Generic,
  Optional,
  Protocol,
  TypeGuard,
  TypeVar,
  Union,
)
from collections.abc import Iterator
from collections.abc import Callable, Hashable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

import dataclasses
import jax.tree_util as jtu


# General

Array = Union[jax.Array, Any]
PRNGKey = jax.Array
RNGSequences = dict[str, PRNGKey]
Dtype = Union[jax.typing.DTypeLike, Any]
Shape = Sequence[int]
K = TypeVar('K')

class Key(Hashable, Protocol):
  def __lt__(self: K, value: K, /) -> bool:
    ...

def is_key_like(x: Any) -> TypeGuard[Key]:
  return hasattr(x, '__hash__') and hasattr(x, '__lt__')

Path = str
PathParts = tuple[Key, ...]

Leaf = Any


# Linear

PrecisionLike = Union[
  None,
  str,
  jax.lax.Precision,
  tuple[str, str],
  tuple[jax.lax.Precision, jax.lax.Precision],
]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]
EinsumT = Callable[..., Array]

PaddingLike = Union[str, int, Sequence[Union[int, tuple[int, int]]]]
LaxPadding = Union[str, Sequence[tuple[int, int]]]


# Initializers

Initializer = Union[jax.nn.initializers.Initializer, Callable[..., Any]]


# Collections

Collection = Mapping[str, Any]
MutableCollection = dict[str, Any]


# Dicts

VariableDict = Mapping[str, Collection]
FrozenVariableDict = FrozenDict[str, Collection]
MutableVariableDict = dict[str, MutableCollection]

PRNGFoldable = Union[int, str]


# Axes

T = TypeVar('T')

@dataclasses.dataclass(frozen=True)
class In(Generic[T]):
  """Specifies a variable collection should only be lifted as input."""

  axis: T

@dataclasses.dataclass(frozen=True)
class Out(Generic[T]):
  """Specifies a variable collection should only be lifted as output."""

  axis: T

Axis = Optional[int]
InOutAxis = Union[Axis, In[Axis], Out[Axis]]

ScanAxis = int
InOutScanAxis = Union[ScanAxis, In[ScanAxis], Out[ScanAxis]]

Axes = Union[int, Sequence[int]]


# SPMD

LogicalNames = tuple[Union[str, None], ...]
AxisName = str | tuple[str, ...] | None

# Maps each logical axis  to physical mesh, can be either None (replicated),
# one physical axis or a tuple of physical axes.
LogicalRules = Sequence[tuple[str, AxisName]]
ArrayPytree = Any  # pylint: disable=invalid-name
LogicalPartitionSpec = Any  # pylint: disable=invalid-name
LogicalPartitionSpecPytree = Any  # pylint: disable=invalid-name
PartitionSpecPytree = Any  # pylint: disable=invalid-name

Sharding = tuple[AxisName, ...]

A = TypeVar('A')
HA = TypeVar('HA', bound=Hashable)
HB = TypeVar('HB')


class PytreeDeque(deque[A]):
  pass


def _pytree_deque_flatten(xs: PytreeDeque, *, with_path: bool):
  if with_path:
    nodes = tuple((jtu.SequenceKey(i), x) for i, x in enumerate(xs))
    return nodes, ()
  else:
    return xs, ()


def _pytree_deque_unflatten(_, nodes):
  return PytreeDeque(nodes)


jtu.register_pytree_with_keys(
  PytreeDeque,
  partial(_pytree_deque_flatten, with_path=True),
  _pytree_deque_unflatten,
  flatten_func=partial(_pytree_deque_flatten, with_path=False),
)

class Missing:
  pass


MISSING = Missing()


def _bytes_repr(num_bytes):
  count, units = (
    (f'{num_bytes / 1e9:,.1f}', 'GB')
    if num_bytes > 1e9
    else (f'{num_bytes / 1e6:,.1f}', 'MB')
    if num_bytes > 1e6
    else (f'{num_bytes / 1e3:,.1f}', 'KB')
    if num_bytes > 1e3
    else (f'{num_bytes:,}', 'B')
  )

  return f'{count} {units}'


class ShapeDtype(Protocol):
  shape: Shape
  dtype: Dtype


def has_shape_dtype(x: Any) -> TypeGuard[ShapeDtype]:
  return hasattr(x, 'shape') and hasattr(x, 'dtype')


@dataclasses.dataclass(frozen=True, slots=True)
class SizeBytes:  # type: ignore[misc]
  size: int
  bytes: int

  @classmethod
  def from_array(cls, x: ShapeDtype):
    size = int(np.prod(x.shape))
    dtype: jnp.dtype
    if isinstance(x.dtype, str):
      dtype = jnp.dtype(x.dtype)
    else:
      dtype = x.dtype  # type: ignore
    bytes = size * dtype.itemsize  # type: ignore
    return cls(size, bytes)

  def __add__(self, other: SizeBytes):
    return type(self)(self.size + other.size, self.bytes + other.bytes)

  def __bool__(self) -> bool:
    return bool(self.size)

  def __repr__(self) -> str:
    bytes_repr = _bytes_repr(self.bytes)
    return f'{self.size:,} ({bytes_repr})'

  @classmethod
  def from_any(cls, x):
    leaves = jax.tree.leaves(x)
    size_bytes = cls(0, 0)
    for leaf in leaves:
      if has_shape_dtype(leaf):
        size_bytes += cls.from_array(leaf)

    return size_bytes


TupleArg = TypeVar('TupleArg', bound=tuple)


class PromoteDtypeFn(Protocol):
  def __call__(
    self, args: TupleArg, /, *, dtype: Any = None, inexact: bool = True
  ) -> TupleArg: ...


class HashableMapping(Mapping[HA, HB], Hashable):
  _mapping: dict[HA, HB] | Mapping[HA, HB]

  def __init__(self, mapping: Mapping[HA, HB], copy: bool = True):
    self._mapping = dict(mapping) if copy else mapping

  def __contains__(self, key: object) -> bool:
    return key in self._mapping

  def __getitem__(self, key: HA) -> HB:
    return self._mapping[key]

  def __iter__(self) -> Iterator[HA]:
    return iter(self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __hash__(self) -> int:
    # use type-aware sorting to support int keys
    def _pytree__key_sort_fn(item: tuple[Any, Any]) -> tuple[int, Any]:
      key, _ = item
      if isinstance(key, int):
        return (0, key)
      elif isinstance(key, str):
        return (1, key)
      else:
        raise ValueError(f'Unsupported key type: {type(key)!r}')

    return hash(tuple(sorted(self._mapping.items(), key=_pytree__key_sort_fn)))

  def __eq__(self, other: Any) -> bool:
    return (
      isinstance(other, HashableMapping) and self._mapping == other._mapping
    )

  def __repr__(self) -> str:
    return repr(self._mapping)

  def update(self, other: Mapping[HA, HB]) -> HashableMapping[HA, HB]:
    """Updates the mapping with another mapping."""
    mapping = dict(self._mapping)
    mapping.update(other)
    return HashableMapping(mapping, copy=False)


F = TypeVar('F', bound=Callable[..., Any])


class BaseConfigContext(abc.ABC):
  @classmethod
  @abc.abstractmethod
  def get_default(cls):
    ...

  @classmethod
  @abc.abstractmethod
  def get_stack(cls) -> list:
    ...

  def __init__(self, value, /):
    stack = self.get_stack()
    if stack:
      self.prev_value = stack[-1]
      stack[-1] = value
    else:
      self.prev_value = None
      stack.append(value)
    self.new_value = value

  @classmethod
  def current_value(cls):
    stack = cls.get_stack()
    if stack:
      return stack[-1]
    return cls.get_default()

  def __enter__(self):
    if self.prev_value is not None:
      self.get_stack().insert(-1, self.prev_value)

  def __exit__(self, exc_type, exc_value, traceback):
    self.get_stack().pop()

  def __call__(self, f: F) -> F:
    self.get_stack().pop()
    if self.prev_value is not None:
      self.get_stack().append(self.prev_value)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      self.get_stack().append(self.new_value)
      try:
        return f(*args, **kwargs)
      finally:
        self.get_stack().pop()

    return wrapper  # type: ignore[return-value]
