"""Persistence for the application's shared configuration (one server process)."""
import json
import os
import tempfile
import threading
from copy import deepcopy


config_lock = threading.RLock()


def read_config(filename):
    # Windows does not allow replacing a file while a normal Python reader holds it open.
    with config_lock:
        with open(filename, encoding="utf-8") as reader:
            return json.load(reader)


def config_snapshot(config):
    with config_lock:
        return deepcopy(config)


def write_config(config, filename):
    """Serialize writers and replace the file only after the complete JSON is ready."""
    with config_lock:
        text = json.dumps(config, indent=4)
        fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(filename)}.", suffix=".tmp", dir=os.path.dirname(os.path.abspath(filename)))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as writer:
                writer.write(text)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def update_config(config, filename, changes, *, remove=()):
    """Commit a patch without replacing the shared dictionary or unrelated values."""
    with config_lock:
        updated = dict(config)
        updated.update(changes)
        for key in remove:
            updated.pop(key, None)
        if filename:
            write_config(updated, filename)
        config.update(changes)
        for key in remove:
            config.pop(key, None)
