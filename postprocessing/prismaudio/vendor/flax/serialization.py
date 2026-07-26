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

"""Minimal state-dict registry used by the vendored VideoPrism Linen subset."""

import threading
from contextlib import contextmanager
from typing import Any

import jax

_STATE_DICT_REGISTRY: dict[Any, Any] = {}


class _ErrorContext(threading.local):
  def __init__(self):
    self.path = []


_error_context = _ErrorContext()


@contextmanager
def _record_path(name):
  try:
    _error_context.path.append(name)
    yield
  finally:
    _error_context.path.pop()


def current_path():
  return "/".join(_error_context.path)


class _NamedTuple:
  pass


def _is_namedtuple(x):
  return isinstance(x, tuple) and hasattr(x, "_fields")


def from_state_dict(target, state: dict[str, Any], name: str = "."):
  ty = _NamedTuple if _is_namedtuple(target) else type(target)
  if ty not in _STATE_DICT_REGISTRY:
    return state
  with _record_path(name):
    return _STATE_DICT_REGISTRY[ty][1](target, state)


def to_state_dict(target):
  ty = _NamedTuple if _is_namedtuple(target) else type(target)
  if ty not in _STATE_DICT_REGISTRY:
    return target
  state_dict = _STATE_DICT_REGISTRY[ty][0](target)
  if isinstance(state_dict, dict):
    for key in state_dict.keys():
      assert isinstance(key, str), "A state dict must only have string keys."
  return state_dict


def is_serializable(target):
  if not isinstance(target, type):
    target = type(target)
  return target in _STATE_DICT_REGISTRY


def register_serialization_state(ty, ty_to_state_dict, ty_from_state_dict, override=False):
  if ty in _STATE_DICT_REGISTRY and not override:
    raise ValueError(f'a serialization handler for "{ty.__name__}" is already registered')
  _STATE_DICT_REGISTRY[ty] = (ty_to_state_dict, ty_from_state_dict)


def _list_state_dict(xs: list[Any]) -> dict[str, Any]:
  return {str(i): to_state_dict(x) for i, x in enumerate(xs)}


def _restore_list(xs, state_dict: dict[str, Any]) -> list[Any]:
  if len(state_dict) != len(xs):
    raise ValueError(f"The size of the list and the state dict do not match, got {len(xs)} and {len(state_dict)} at path {current_path()}")
  return [from_state_dict(xs[i], state_dict[str(i)], name=str(i)) for i in range(len(state_dict))]


def _dict_state_dict(xs: dict[str, Any]) -> dict[str, Any]:
  str_keys = {str(k) for k in xs.keys()}
  if len(str_keys) != len(xs):
    raise ValueError(f"Dict keys do not have a unique string representation: {str_keys} vs given: {xs}")
  return {str(key): to_state_dict(value) for key, value in xs.items()}


def _restore_dict(xs, states: dict[str, Any]) -> dict[str, Any]:
  diff = set(map(str, xs.keys())).difference(states.keys())
  if diff:
    raise ValueError(f"The target dict keys and state dict keys do not match, target dict contains keys {diff} which are not present in state dict at path {current_path()}")
  return {key: from_state_dict(value, states[str(key)], name=str(key)) for key, value in xs.items()}


def _namedtuple_state_dict(nt) -> dict[str, Any]:
  return {key: to_state_dict(getattr(nt, key)) for key in nt._fields}


def _restore_namedtuple(xs, state_dict: dict[str, Any]):
  if set(state_dict.keys()) == {"name", "fields", "values"}:
    state_dict = {state_dict["fields"][str(i)]: state_dict["values"][str(i)] for i in range(len(state_dict["fields"]))}
  sd_keys = set(state_dict.keys())
  nt_keys = set(xs._fields)
  if sd_keys != nt_keys:
    raise ValueError(f"The field names of the state dict and the named tuple do not match, got {sd_keys} and {nt_keys} at path {current_path()}")
  return type(xs)(**{k: from_state_dict(getattr(xs, k), v, name=k) for k, v in state_dict.items()})


register_serialization_state(dict, _dict_state_dict, _restore_dict)
register_serialization_state(list, _list_state_dict, _restore_list)
register_serialization_state(tuple, _list_state_dict, lambda xs, state_dict: tuple(_restore_list(list(xs), state_dict)))
register_serialization_state(_NamedTuple, _namedtuple_state_dict, _restore_namedtuple)
register_serialization_state(
  jax.tree_util.Partial,
  lambda x: {"args": to_state_dict(x.args), "keywords": to_state_dict(x.keywords)},
  lambda x, sd: jax.tree_util.Partial(x.func, *from_state_dict(x.args, sd["args"]), **from_state_dict(x.keywords, sd["keywords"])),
)
