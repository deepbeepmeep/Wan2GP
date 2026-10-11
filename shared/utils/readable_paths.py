# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Readable, portable names for persistent workspaces and sessions."""
import re
from pathlib import Path


def safe_name(name):
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')[:120].rstrip(' .') or 'Untitled'
    stem = stem.encode('utf-8')[:200].decode('utf-8', errors='ignore').rstrip(' .')
    if stem.split('.')[0].rstrip().upper() in {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$', *(f'{prefix}{i}' for prefix in ('COM', 'LPT') for i in '123456789¹²³')}:
        stem = '_' + stem
    return stem


def available_path(root, name, *, suffix='', exclude=None, reserved=()):
    """Choose a free name; callers serialize allocation and creation."""
    root = Path(root)
    used = {path.name.casefold() for path in root.iterdir() if path != exclude}
    used.update(value.casefold() for value in reserved)
    stem, number = safe_name(name), 2
    candidate = stem
    while (candidate + suffix).casefold() in used:
        candidate = f'{stem} ({number})'
        number += 1
    return root / (candidate + suffix)
