"""Paths, bundled configuration and logging for the WanGP Windows launcher.

The launcher never duplicates installation logic: it is a graphical front-end
over the repository's own ``setup.py``. This module only resolves *where*
things live and how we talk to the user about it.
"""

import json
import logging
import os
import sys

APP_DIR_NAME = "WanGP"

_DEFAULTS = {
    "app_name": "WanGP",
    "app_version": "1.0.0",
    "publisher": "Blencia",
    "repo_url": "https://github.com/blencia/Wan2Gplus.git",
    "repo_branch": "main",
    "python": {
        "series": "3.11",
        "installer_url": "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe",
        "installer_sha256": "",
        "label": "Python 3.11.9 (64-bit)",
    },
    "git": {"portable_url": "", "portable_sha256": "", "label": "Portable Git"},
    "env_type": "venv",
    "server": {"host": "127.0.0.1", "preferred_port": 7860, "startup_timeout_seconds": 900},
    "window": {"width": 1450, "height": 940, "min_width": 900, "min_height": 620},
}


def is_frozen():
    return getattr(sys, "frozen", False)


def bundle_dir():
    """Directory holding read-only resources shipped inside the executable."""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


_CHECKOUT_MARKERS = ("wgp.py", "setup.py")


def _looks_like_checkout(path):
    return all(os.path.isfile(os.path.join(path, name)) for name in _CHECKOUT_MARKERS)


def install_root():
    """The WanGP checkout: where wgp.py, setup.py and requirements.txt live.

    Frozen, the executable ships in a ``launcher-bin`` subfolder of the
    installation directory, so we walk up until the checkout markers appear.
    From source, the launcher runs out of ``launcher/`` inside the repository.
    """
    if not is_frozen():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    start = os.path.dirname(os.path.abspath(sys.executable))
    current = start
    for _ in range(4):
        if _looks_like_checkout(current):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return start


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config():
    """Bundled defaults, overridable by a JSON file next to the executable."""
    config = dict(_DEFAULTS)
    for candidate in (
        os.path.join(bundle_dir(), "launcher_config.json"),
        os.path.join(install_root(), "launcher_config.json"),
    ):
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                config = _deep_merge(config, json.load(handle))
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning("Ignoring %s: %s", candidate, exc)
    return config


def data_dir():
    """Writable per-user directory for logs and launcher state."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, APP_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def logs_dir():
    path = os.path.join(data_dir(), "logs")
    os.makedirs(path, exist_ok=True)
    return path


def tools_dir():
    """Where we drop self-contained tools (portable Git) we install ourselves."""
    path = os.path.join(data_dir(), "tools")
    os.makedirs(path, exist_ok=True)
    return path


def state_file():
    return os.path.join(data_dir(), "launcher_state.json")


def read_state():
    try:
        with open(state_file(), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def write_state(**updates):
    state = read_state()
    state.update(updates)
    try:
        with open(state_file(), "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not persist launcher state: %s", exc)


def setup_logging():
    log_path = os.path.join(logs_dir(), "launcher.log")
    handlers = [logging.FileHandler(log_path, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_path
