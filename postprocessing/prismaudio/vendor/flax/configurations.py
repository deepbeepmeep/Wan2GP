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

"""Minimal Flax config flags used by the vendored VideoPrism Linen subset."""

from contextlib import contextmanager


class Config:
  flax_filter_frames: bool = True
  flax_profile: bool = True
  flax_preserve_adopted_names: bool = False
  flax_return_frozendict: bool = False
  flax_fix_rng_separator: bool = False

  def update(self, name: str, value):
    if not hasattr(self, name):
      raise AttributeError(f"Unknown Flax config option: {name}")
    setattr(self, name, value)

  @contextmanager
  def temp_flip_flag(self, var_name: str, var_value: bool):
    name = f"flax_{var_name}"
    old_value = getattr(self, name)
    try:
      self.update(name, var_value)
      yield
    finally:
      self.update(name, old_value)


config = Config()
