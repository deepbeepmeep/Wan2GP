"""Compatibility package for legacy import paths."""

class _CompatAll(list):
    def __contains__(self, item):
        return item in {"db", "travel_guide"}


__all__ = _CompatAll()
