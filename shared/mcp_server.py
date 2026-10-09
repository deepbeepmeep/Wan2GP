# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Compatibility entry point; the MCP implementation lives in shared.mcp."""

from shared.mcp.server import (
    build_inprocess_server, build_server, build_server_for_session, build_webui_server,
    main, parse_args, resolve_gallery_media_path, run_server,
)


if __name__ == "__main__":
    raise SystemExit(main())
