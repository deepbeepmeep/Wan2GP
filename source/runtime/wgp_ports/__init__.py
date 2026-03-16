"""Port modules that isolate vendor/runtime access."""

from source.runtime.wgp_ports import runtime_registry, vendor_imports

__all__ = ["runtime_registry", "vendor_imports"]
