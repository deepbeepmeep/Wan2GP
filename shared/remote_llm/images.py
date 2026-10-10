from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


@contextmanager
def temporary_image_paths(images: Sequence[Any]) -> Iterator[list[str]]:
    paths: list[str] = []
    temporary: list[str] = []
    try:
        for image in images:
            if image is None:
                continue
            if isinstance(image, (str, os.PathLike)) and Path(image).is_file():
                paths.append(str(Path(image).resolve()))
                continue
            suffix = ".png"
            handle = tempfile.NamedTemporaryFile(prefix="wangp-remote-image-", suffix=suffix, delete=False)
            handle.close()
            save = getattr(image, "save", None)
            if not callable(save):
                raise TypeError(f"Unsupported remote LLM image input: {type(image).__name__}")
            save(handle.name, format="PNG")
            paths.append(handle.name)
            temporary.append(handle.name)
        yield paths
    finally:
        for path in temporary:
            try:
                os.unlink(path)
            except OSError:
                pass

