import os
from typing import Optional


def resolve_path(relpath: str, must_exist: bool = True) -> str:
    """
    Return an absolute path for a resource that may be referenced relative to:
    1) the current working directory (preferred for checkpoints/cache writes), or
    2) the installed package directory (read-only package data).

    This allows running CLIs from any CWD while still reading bundled config files.
    """
    # Try CWD first
    if os.path.isabs(relpath):
        return relpath
    cwd_path = os.path.abspath(relpath)
    if os.path.exists(cwd_path):
        return cwd_path
    # Fall back to package directory
    pkg_dir = os.path.dirname(__file__)
    pkg_path = os.path.join(pkg_dir, relpath)
    if os.path.exists(pkg_path) or not must_exist:
        return pkg_path
    # As a last resort, return CWD path even if missing (will raise on open)
    return cwd_path

