"""Name and optionally save model-provided companion files."""
from pathlib import Path
import re


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
    if not return_files:
        for filename, content in named.items():
            (path.parent / filename).write_bytes(content)
            print(f"Side file saved to Path: {(path.parent / filename).resolve()}", flush=True)
    return named
