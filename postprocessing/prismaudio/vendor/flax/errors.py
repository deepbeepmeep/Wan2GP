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

"""Compact Flax error compatibility layer for the vendored VideoPrism subset."""


class FlaxError(Exception):
  pass


class LazyInitError(FlaxError):
  def __init__(self, partial_val):
    super().__init__(f"Lazy init encountered an uncomputable value with the given inputs (shape: {partial_val}).")


class InvalidRngError(FlaxError):
  pass


class ApplyScopeInvalidVariablesTypeError(FlaxError):
  def __init__(self):
    super().__init__("The first argument passed to an apply function should be a dictionary of collections.")


class ApplyScopeInvalidVariablesStructureError(FlaxError):
  def __init__(self, variables):
    super().__init__('Expected variables to have structure {"params": ...}, but got an extra params layer.')


class ScopeParamNotFoundError(FlaxError):
  def __init__(self, param_name, scope_path):
    super().__init__(f'Could not find parameter named "{param_name}" in scope "{scope_path}".')


class ScopeCollectionNotFound(FlaxError):
  def __init__(self, col_name, var_name, scope_path):
    super().__init__(f'Tried to access "{var_name}" from collection "{col_name}" in "{scope_path}" but the collection is empty.')


class ScopeParamShapeError(FlaxError):
  def __init__(self, param_name, scope_path, value_shape, init_shape):
    super().__init__(f'For parameter "{param_name}" in "{scope_path}", expected shape {init_shape}, got existing shape {value_shape}.')


class ScopeVariableNotFoundError(FlaxError):
  def __init__(self, name, col, scope_path):
    super().__init__(f'No Variable named "{name}" for collection "{col}" exists in "{scope_path}".')


class InvalidFilterError(FlaxError):
  def __init__(self, filter_like):
    super().__init__(f'Invalid Filter: "{filter_like}"')


class InvalidScopeError(FlaxError):
  def __init__(self, scope_name):
    super().__init__(f'The scope "{scope_name}" is no longer valid.')


class ModifyScopeVariableError(FlaxError):
  def __init__(self, col, variable_name, scope_path):
    super().__init__(f'Cannot update variable "{variable_name}" in "{scope_path}" because collection "{col}" is immutable.')


class JaxTransformError(FlaxError):
  def __init__(self):
    super().__init__("JAX transforms and Flax models cannot be mixed.")


class PartitioningUnspecifiedError(FlaxError):
  def __init__(self, target):
    super().__init__(f'Trying to transform a Partitioned variable but "partition_name" is not specified in metadata_params: {target}')


class NameInUseError(FlaxError):
  def __init__(self, key_type, value, module_name):
    super().__init__(f'Could not create {key_type} "{value}" in Module {module_name}: Name in use.')


class AssignSubModuleError(FlaxError):
  def __init__(self, cls):
    super().__init__(f"Submodule {cls} must be defined in setup() or in a method wrapped in @compact.")


class SetAttributeInModuleSetupError(FlaxError):
  def __init__(self):
    super().__init__("Module construction attributes are frozen.")


class SetAttributeFrozenModuleError(FlaxError):
  def __init__(self, module_cls, attr_name, attr_val):
    super().__init__(f"Can't set {attr_name}={attr_val} for Module of type {module_cls}: Module instance is frozen outside setup().")


class ReservedModuleAttributeError(FlaxError):
  def __init__(self, annotations):
    super().__init__(f"Properties `parent` and `name` are reserved: {annotations}")


class ApplyModuleInvalidMethodError(FlaxError):
  def __init__(self, method):
    super().__init__(f"Cannot call apply(): {method} is not a valid function for apply().")


class CallCompactUnboundModuleError(FlaxError):
  def __init__(self):
    super().__init__("Can't call compact methods on unbound modules.")


class CallSetupUnboundModuleError(FlaxError):
  def __init__(self):
    super().__init__("Can't call setup on unbound modules.")


class CallUnbindOnUnboundModuleError(FlaxError):
  def __init__(self):
    super().__init__("Can't call unbind on unbound modules.")


class CallShareScopeOnUnboundModuleError(FlaxError):
  def __init__(self):
    super().__init__("Can't call share_scope on unbound modules.")


class InvalidInstanceModuleError(FlaxError):
  def __init__(self):
    super().__init__("Can only call init, init_with_output, apply, or bind methods on a Module instance.")


class IncorrectPostInitOverrideError(FlaxError):
  def __init__(self):
    super().__init__("Overrode __post_init__ without calling super().__post_init__().")


class DescriptorAttributeError(FlaxError):
  def __init__(self):
    super().__init__("Trying to access a property that references a missing attribute.")


class TransformedMethodReturnValueError(FlaxError):
  def __init__(self, name):
    super().__init__(f"Transformed module method {name} cannot return Modules or Variables.")


class TransformTargetError(FlaxError):
  def __init__(self, target):
    super().__init__(f"Linen transformations must be applied to Module classes or callables taking a Module instance; got {target}.")
