#!/usr/bin/env python
"""Entry point for PyInstaller-packaged Wan2GP API server."""

import os
import sys

# Ensure project root is on sys.path (needed when frozen)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Set the project root for the application to find its data files
os.environ.setdefault("WAN2GP_ROOT", _project_root)

import uvicorn


def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(
        "wgp_fastapi.api.routes:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
