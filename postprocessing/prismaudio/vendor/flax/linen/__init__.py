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

"""The Flax Module system."""


from .activation import (
    gelu as gelu,
    one_hot as one_hot,
    relu as relu,
    softmax as softmax,
    softplus as softplus,
)
from . import initializers as initializers
from .linear import (
    Dense as Dense,
)
from . import module as module
from .module import (
    Module as Module,
    compact as compact,
    nowrap as nowrap,
)
from .stochastic import Dropout as Dropout
from .transforms import (
    remat as remat,
    scan as scan,
)
# pylint: enable=g-multiple-import
