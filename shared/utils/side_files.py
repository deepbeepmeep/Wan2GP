# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Name and optionally save model-provided companion files."""
from pathlib import Path
import re
import json


def process_side_files(media_path, side_files, *, return_files=False):
    path = Path(media_path)
    named = {}
    for suffix, content in side_files.items():
        if not re.fullmatch(r"(?:_[\w.-]+)?\.[A-Za-z0-9]+", suffix):
            raise ValueError(f"Invalid side-file suffix: {suffix!r}")
        target = path.with_name(path.stem + suffix)
        if target == path:
            raise ValueError(f"Side file would overwrite the main media: {target.name}")
        if not isinstance(content, bytes):
            raise TypeError(f"Side file {suffix!r} must contain bytes")
        named[target.name] = content
    from shared.utils.workspace_storage import current_binding
    binding = current_binding()
    persist = binding is not None and binding[0].workspaces.media_dir(binding[1]) is not None
    if not return_files or persist:
        for filename, content in named.items():
            (path.parent / filename).write_bytes(content)
            print(f"Side file saved to Path: {(path.parent / filename).resolve()}", flush=True)
        if named:
            path.with_name(path.name + '.wangp-files.json').write_text(json.dumps(list(side_files)), encoding='utf-8')
    return named
