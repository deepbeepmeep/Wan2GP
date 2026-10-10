"""Cooperative cancellation scoped to one worker, including its loading and decode calls."""

from contextlib import contextmanager
from contextvars import ContextVar


_check_cancelled = ContextVar("wangp_check_cancelled", default=None)


def check_cancelled():
    callback = _check_cancelled.get()
    if callback is not None:
        callback()


@contextmanager
def cancellation_context(callback):
    token = _check_cancelled.set(callback)
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        _check_cancelled.reset(token)
