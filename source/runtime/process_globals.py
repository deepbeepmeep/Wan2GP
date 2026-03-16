"""Helpers for controlled process-global mutations used by runtime entrypoints."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

_BOOTSTRAP_STATE: dict[tuple[str | None, str], dict[str, Any]] = {}
_RUNTIME_CONTEXTS: dict[str, list["RuntimeContext"]] = {}


def _normalize_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).resolve())


def apply_process_globals(
    *,
    cwd: str | os.PathLike[str] | None = None,
    argv: list[str] | None = None,
    prepend_sys_path: str | os.PathLike[str] | None = None,
) -> None:
    """Persistently update process-global execution state."""
    if cwd is not None:
        os.chdir(_normalize_path(cwd))
    if argv is not None:
        sys.argv = list(argv)
    if prepend_sys_path is not None:
        path = _normalize_path(prepend_sys_path)
        if path is not None:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
            sys.path.insert(0, path)


@contextmanager
def temporary_process_globals(
    *,
    cwd: str | os.PathLike[str] | None = None,
    argv: list[str] | None = None,
    prepend_sys_path: str | os.PathLike[str] | None = None,
) -> Iterator[None]:
    """Temporarily apply process-global state and restore it afterwards."""
    original_cwd = os.getcwd()
    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)
    try:
        apply_process_globals(cwd=cwd, argv=argv, prepend_sys_path=prepend_sys_path)
        yield
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        sys.path[:] = original_sys_path


def get_bootstrap_controller(context_id: str | None = None) -> str | None:
    return context_id


def get_bootstrap_state(name: str, *, controller: str | None = None) -> dict[str, Any] | None:
    return _BOOTSTRAP_STATE.get((controller, name))


def reset_bootstrap_state(name: str, *, controller: str | None = None) -> None:
    _BOOTSTRAP_STATE.pop((controller, name), None)


def run_bootstrap_once(
    name: str,
    initializer: Callable[[], Any],
    *,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
    controller: str | None = None,
    context_id: str | None = None,
) -> dict[str, Any]:
    """Run *initializer* once per `(controller, name, version)` tuple."""
    state_key = (controller, name)
    state = _BOOTSTRAP_STATE.get(state_key)
    if state is not None and state.get("version") == version:
        return state

    initializer()
    new_state = {
        "name": name,
        "version": version,
        "metadata": metadata or {},
        "controller": controller,
        "context_id": context_id,
        "last_ran_at": time.time(),
    }
    _BOOTSTRAP_STATE[state_key] = new_state
    return new_state


@dataclass
class RuntimeContext:
    """Prepared runtime execution scope."""

    name: str
    wan_root: str
    default_argv: list[str]
    require_cwd: bool

    def prepare(
        self,
        *,
        argv: list[str] | None = None,
        require_cwd: bool | None = None,
    ) -> None:
        effective_require_cwd = self.require_cwd if require_cwd is None else require_cwd
        apply_process_globals(
            cwd=self.wan_root if effective_require_cwd else None,
            argv=argv or self.default_argv,
            prepend_sys_path=self.wan_root,
        )

    @contextmanager
    def scoped(
        self,
        *,
        argv: list[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        require_cwd: bool | None = None,
    ) -> Iterator[None]:
        effective_require_cwd = self.require_cwd if require_cwd is None else require_cwd
        with temporary_process_globals(
            cwd=cwd or (self.wan_root if effective_require_cwd else None),
            argv=argv or self.default_argv,
            prepend_sys_path=self.wan_root,
        ):
            yield


def get_runtime_context(
    name: str,
    *,
    wan_root: str | os.PathLike[str],
    default_argv: list[str],
    require_cwd: bool,
) -> RuntimeContext:
    context = RuntimeContext(
        name=name,
        wan_root=_normalize_path(wan_root) or "",
        default_argv=list(default_argv),
        require_cwd=require_cwd,
    )
    _RUNTIME_CONTEXTS.setdefault(name, []).append(context)
    return context


def reset_runtime_contexts(name: str | None = None) -> None:
    if name is None:
        _RUNTIME_CONTEXTS.clear()
    else:
        _RUNTIME_CONTEXTS.pop(name, None)


def ensure_runtime_ready(
    *,
    wan_root: str | os.PathLike[str],
    argv: list[str],
    require_cwd: bool,
) -> None:
    apply_process_globals(
        cwd=wan_root if require_cwd else None,
        argv=argv,
        prepend_sys_path=wan_root,
    )
