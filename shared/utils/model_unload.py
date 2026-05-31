import threading
from contextlib import contextmanager


_lock = threading.RLock()


@contextmanager
def model_unload_guard():
    with _lock:
        yield


def wait_for_model_unload():
    with _lock:
        pass
