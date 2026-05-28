# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from pathlib import Path


def get_all_models() -> OrderedDict:
    """
    Scans all YAML files in the configs directory and returns a sorted dictionary where:
    - Keys are model names (YAML filenames without the .yaml extension)
    - Values are absolute paths to the corresponding YAML files
    """
    # Get path to the configs directory within the da3 package
    # Works both in development and after pip installation
    # configs_dir = files(__package__).joinpath("configs")
    configs_dir = Path(__file__).resolve().parent / "configs"

    # Ensure path is a Path object for consistent cross-platform handling
    configs_dir = Path(configs_dir)

    model_entries = []
    # Iterate through all items in the configs directory
    for item in configs_dir.iterdir():
        # Filter for YAML files (excluding directories)
        if item.is_file() and item.suffix == ".yaml":
            # Extract model name (filename without .yaml extension)
            model_name = item.stem
            # Get absolute path (resolve() handles symlinks)
            file_abs_path = str(item.resolve())
            model_entries.append((model_name, file_abs_path))

    # Sort entries by model name and convert to OrderedDict
    sorted_entries = sorted(model_entries, key=lambda x: x[0])
    return OrderedDict(sorted_entries)


# Global registry for external imports
MODEL_REGISTRY = get_all_models()
