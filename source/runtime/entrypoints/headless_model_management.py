"""Runtime entrypoint for the headless queue service."""

from __future__ import annotations

from importlib import import_module


def main():
    return import_module("source.task_handlers.queue.task_queue").main()
