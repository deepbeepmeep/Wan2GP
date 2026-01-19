from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

default_checkpoints_paths = ["ckpts", "."]

_checkpoints_paths = default_checkpoints_paths

def set_checkpoints_paths(checkpoints_paths):
    global _checkpoints_paths
    _checkpoints_paths = [path.strip() for path in checkpoints_paths if len(path.strip()) > 0 ]
    if len(checkpoints_paths) == 0:
        _checkpoints_paths = default_checkpoints_paths
def get_download_location(file_name = None, force_path= None):
    if file_name is not None and os.path.isabs(file_name): return file_name
    if force_path is not None and isinstance(force_path, list) and len(force_path): force_path = force_path[0]
    if file_name is not None:
        if force_path is None:
            return os.path.join(_checkpoints_paths[0], file_name)
        else:
            return os.path.join(_checkpoints_paths[0], force_path, file_name)
    else:
        if force_path is None:
            return _checkpoints_paths[0]
        else:
            return os.path.join(_checkpoints_paths[0])

def locate_folder(folder_name, error_if_none = True):
    searched_locations = []
    if os.path.isabs(folder_name):
        if os.path.isdir(folder_name): return folder_name
        searched_locations.append(folder_name)
    else:
        for folder in _checkpoints_paths:
            path = os.path.join(folder, folder_name)
            if os.path.isdir(path):
                return path
            searched_locations.append(os.path.abspath(path))
    if error_if_none: raise Exception(f"Unable to locate folder '{folder_name}', tried {searched_locations}")    
    return None


def locate_file(file_name, create_path_if_none = False, error_if_none = True, extra_paths = None):
    if file_name.startswith("http"):
        file_name = os.path.basename(file_name)
    searched_locations = []
    if os.path.isabs(file_name):
        if os.path.isfile(file_name): return file_name
        searched_locations.append(file_name)
    else:
        for folder in _checkpoints_paths + ([] if extra_paths is None else extra_paths):
            path = os.path.join(folder, file_name)
            if os.path.isfile(path):
                return path
            searched_locations.append(os.path.abspath(path))
    
    if create_path_if_none:
        return get_download_location(file_name)
    if error_if_none: raise Exception(f"Unable to locate file '{file_name}', tried {searched_locations}")
    return None

