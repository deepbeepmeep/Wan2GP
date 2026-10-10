"""Gallery metadata loaded on access, without changing existing list consumers."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import RLock


class _Pending:
    def __init__(self, path, read):
        self.path, self.read = path, read


class MediaSettings(list):
    def __init__(self, paths=(), read=None):
        super().__init__(_Pending(path, read) for path in paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        value = super().__getitem__(index)
        if isinstance(value, _Pending):
            value = value.read(value.path)
            self[index] = value
        return value

    def __iter__(self):
        return (self[index] for index in range(len(self)))


def peek_settings(settings, index):
    """Inspect already available metadata; identity/catalog scans never probe files."""
    value = list.__getitem__(settings, index)
    return None if isinstance(value, _Pending) else value


class MediaSettingsCache:
    def __init__(self, read):
        self.lock = RLock()
        self.cached = lru_cache(maxsize=1024)(lambda path, stamp: read(path))

    def __call__(self, path):
        stat = Path(path).stat()
        with self.lock:
            return deepcopy(self.cached(path, (stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns)))
