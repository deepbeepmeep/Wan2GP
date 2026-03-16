"""Bootstrap service wrappers around process-global runtime helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from source.runtime.process_globals import (
    get_bootstrap_controller,
    get_runtime_context,
    run_bootstrap_once,
)


@dataclass(frozen=True)
class RuntimeScopeSpec:
    name: str
    wan_root: str
    default_argv: list[str]
    require_cwd: bool = True


class RuntimeBootstrapService:
    def __init__(self, *, context_id: str):
        self.context_id = context_id
        self.controller = get_bootstrap_controller(context_id)

    def run_once(
        self,
        *,
        name: str,
        initializer,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return run_bootstrap_once(
            name,
            initializer,
            version=version,
            metadata=metadata,
            controller=self.controller,
            context_id=self.context_id,
        )

    def prepare_runtime(
        self,
        spec: RuntimeScopeSpec,
        *,
        argv: list[str] | None = None,
        require_cwd: bool | None = None,
    ):
        runtime_context = get_runtime_context(
            spec.name,
            wan_root=spec.wan_root,
            default_argv=spec.default_argv,
            require_cwd=spec.require_cwd,
        )
        runtime_context.prepare(argv=argv, require_cwd=require_cwd)
        return runtime_context

    @contextmanager
    def scoped_runtime(
        self,
        spec: RuntimeScopeSpec,
        *,
        argv: list[str] | None = None,
        cwd: Path | None = None,
        require_cwd: bool | None = None,
    ) -> Iterator[None]:
        runtime_context = get_runtime_context(
            spec.name,
            wan_root=spec.wan_root,
            default_argv=spec.default_argv,
            require_cwd=spec.require_cwd,
        )
        with runtime_context.scoped(argv=argv, cwd=cwd, require_cwd=require_cwd):
            yield
