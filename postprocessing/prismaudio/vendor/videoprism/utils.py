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

"""Utility functions for checkpointing and other purposes."""

import collections
from collections.abc import Mapping, Sequence
import io
import os
import string
import urllib.request

import jax
import numpy as np


def _read_binary(path: str) -> bytes:
  if path.startswith("gs://"):
    path = "https://storage.googleapis.com/" + path[len("gs://"):]
  if path.startswith(("http://", "https://")):
    with urllib.request.urlopen(path) as response:
      return response.read()
  with open(path, "rb") as reader:
    return reader.read()


def traverse_with_names(tree, with_inner_nodes=False):
  """Traverses nested dicts and emits (leaf_name, leaf_val).

  Args:
    tree: JAX Pytree object.
    with_inner_nodes: Whether to traverse the non-leaf nodes.

  Yields:
    A pair of (leaf_name, leaf_val).
  """
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree.map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in traverse_with_names(tree[key], with_inner_nodes):
        yield (key + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  elif isinstance(tree, Sequence):
    for idx in range(len(tree)):
      for path, v in traverse_with_names(tree[idx], with_inner_nodes):
        yield (str(idx) + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  else:
    yield "", tree


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  Args:
    tree: JAX Pytree object.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree.flatten(tree)

  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)]


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  Args:
    keys: A list of keys, where '/' is used as separator between nodes.
    values: A list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  # Load the data; use local paths directly if possible:
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    # For other remote paths go via BytesIO as np.load requires seeks.
    loaded = np.load(io.BytesIO(_read_binary(fname)), allow_pickle=False)

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)


def load_checkpoint(npz):
  """Loads a jax Pytree from a npz file.

  Args:
    npz: Either path to the checkpoint file (.npz), or a dict-like.

  Returns:
    A Pytree that is the checkpoint.
  """
  if isinstance(npz, str):  # If not already loaded, then load.
    npz = npload(npz)
  keys, values = zip(*list(npz.items()))
  return recover_tree(keys, values)


def canonicalize_text(text: str) -> str:
  """Canonicalizes text.

  Canonicalization includes:
  - Replace all punctuation with a whitespace.
  - Use all lower case.
  - Leave only one whitespace between words.
  - End with a period.

  Examples:
    "Hello, World!" -> "hello world."
    "Hello,World.." -> "hello world."
    "  Hello   WORLD" -> "hello world."

  Args:
    text: A string for the input text.

  Returns:
    A string for the canonicalized text.
  """
  # Replace all punctuation with a whitespace.
  p = string.punctuation
  text = text.translate(str.maketrans(p, " " * len(p)))
  # Use all lower case.
  text = text.lower()
  # Leave only one whitespace between words.
  text = " ".join(text.split())
  # End with a period.
  text = text + "."
  return text
