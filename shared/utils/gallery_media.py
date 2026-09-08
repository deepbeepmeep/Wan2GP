"""Stable gallery identities, including IDs issued before path normalization."""

import hashlib
import os
from pathlib import Path


def gallery_media_ids(path: str, gallery: str, settings: dict | None = None, *, root: Path | None = None) -> list[str]:
    root = Path.cwd() if root is None else Path(root)
    path = str(path).strip()
    absolute = (root / path).resolve()
    paths = [str(absolute), path]
    try:
        paths.append(os.path.relpath(absolute, root))
    except ValueError:  # Windows paths on different drives have no relative spelling.
        pass
    saved_ids = (settings or {}).get("gallery_media_ids", [])
    path_ids = [f"{gallery}:" + hashlib.sha1(value.replace("\\", "/").casefold().encode("utf-8")).hexdigest()[:12] for value in paths]
    # Generation settings can be inherited by a different output file. Its source's identity cannot be inherited.
    if not set(saved_ids).intersection(path_ids):
        saved_ids = []
    return list(dict.fromkeys([*saved_ids, *path_ids]))


def disambiguate_gallery_media_ids(items: list[tuple[str, str, dict]], *, root: Path | None = None) -> list[list[str]]:
    path_ids = [gallery_media_ids(path, gallery, root=root) for path, gallery, _settings in items]
    owners = {media_id: ids[0] for ids in path_ids for media_id in ids}
    # Repair aliases copied into derived media by older session saves. A path ID belongs to that file.
    resolved = []
    for (path, gallery, settings), ids in zip(items, path_ids):
        candidates = gallery_media_ids(path, gallery, settings, root=root)
        resolved.append([media_id for media_id in candidates if owners.get(media_id, ids[0]) == ids[0]])
    return resolved
