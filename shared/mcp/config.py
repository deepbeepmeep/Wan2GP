# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared MCP server configuration."""

DEFAULT_PORT = 8000


def server_port(value):
    """Validate a port supplied by the configuration UI, JSON or command line."""
    try:
        port = int(value)
        if isinstance(value, bool) or (not isinstance(value, str) and value != port) or not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("MCP Server Port must be a whole number between 1 and 65535.") from None
    return port
