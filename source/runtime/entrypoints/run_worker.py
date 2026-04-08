"""Runtime supervisor entrypoint."""

from __future__ import annotations

from importlib import import_module


def _supervisor_module():
    return import_module("source.runtime.supervisor")


def main():
    return _supervisor_module().main()
