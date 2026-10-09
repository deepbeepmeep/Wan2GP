# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
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
