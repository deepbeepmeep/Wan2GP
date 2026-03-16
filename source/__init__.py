"""Best-effort package bootstrap for modules loaded dynamically elsewhere."""

# Explicit imports for modules used via deferred/dynamic loading.
# Keep them best-effort so lightweight test stubs do not fail on import.
try:
    from source.models.wgp import wgp_patches  # noqa: F401
except (ImportError, AttributeError):
    pass

try:
    from source.media.video import vace_frame_utils  # noqa: F401
except (ImportError, AttributeError):
    pass

try:
    from source.media.video import hires_utils  # noqa: F401
except (ImportError, AttributeError):
    pass

try:
    from source.models.lora import lora_utils  # noqa: F401
except (ImportError, AttributeError):
    pass

try:
    from source.task_handlers.worker import heartbeat_utils  # noqa: F401
except (ImportError, AttributeError):
    pass

try:
    from source.task_handlers.worker import fatal_error_handler  # noqa: F401
except (ImportError, AttributeError):
    pass
