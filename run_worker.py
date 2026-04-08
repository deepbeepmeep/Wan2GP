"""Direct launcher for the runtime supervisor entrypoint."""

from __future__ import annotations

import sys

# This stays a direct launcher so importing `run_worker` does not go through the
# root-shim deprecation machinery used by legacy entrypoints like `worker.py`.
from source.runtime.entrypoints.run_worker import main


if __name__ == "__main__":
    sys.exit(main() or 0)
