# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Run the standalone WanGP MCP adapter with ``python -m shared.mcp``."""

from .server import main


if __name__ == "__main__":
    raise SystemExit(main())
