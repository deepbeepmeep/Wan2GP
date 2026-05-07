"""Best-effort package bootstrap for modules loaded dynamically elsewhere."""

from __future__ import annotations

import importlib


def _import_optional_bootstrap(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception:
        # These imports register optional dynamic modules and patches. They must
        # not make lightweight route/claim tests depend on the full GPU stack.
        pass


# Explicit imports for modules used via deferred/dynamic loading.
for _module_name in (
    "source.models.wgp.wgp_patches",
    "source.models.wgp.transformers_patches",
    "source.media.video.vace_frame_utils",
    "source.media.video.hires_utils",
    "source.models.lora.lora_utils",
    "source.task_handlers.worker.heartbeat_utils",
    "source.task_handlers.worker.fatal_error_handler",
):
    _import_optional_bootstrap(_module_name)
