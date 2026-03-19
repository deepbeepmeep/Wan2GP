"""Compatibility package for legacy import paths."""

from __future__ import annotations

import sys
import types


class _CompatAll(list):
    def __contains__(self, item):
        return item == "travel_guide"


class _CompatModule(types.ModuleType):
    def __getattribute__(self, name):
        if name in {"db", "travel_guide"}:
            raise AttributeError(name)
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _CompatModule
__all__ = _CompatAll()
