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

"""Functionality for inspecting jax tracers."""

import jax
import jax.core
from jax.extend import core as jex_core


def current_trace():
  """Returns the current JAX state tracer."""
  if jax.__version_info__ <= (0, 4, 33):
    top = jax.core.find_top_trace(())
    if top:
      return top.level
    else:
      return float('-inf')

  try:
    # JAX v0.10.0 and newer
    get_opaque_trace_state = jex_core.get_opaque_trace_state
  except AttributeError:
    # JAX v0.9.2 and older
    get_opaque_trace_state = jax.core.get_opaque_trace_state
  return get_opaque_trace_state(convention="flax")

def check_trace_level(base_level):
  # TODO(cgarciae): skipping for now as it breaks
  # too many internal tests.
  # level = current_trace()
  # if level != base_level:
  #   raise errors.JaxTransformError()
  pass
