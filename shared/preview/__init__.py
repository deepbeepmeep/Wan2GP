"""Generation-scoped live preview primitives.

The package is intentionally import-safe: decoder weights and torch-heavy vendor
code are loaded only when Tiny VAE mode is actually used.
"""

from .types import PreviewContext, PreviewMedia, PreviewOptions

__all__ = ["PreviewContext", "PreviewMedia", "PreviewOptions"]
