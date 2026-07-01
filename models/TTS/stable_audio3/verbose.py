"""Shared verbose flag for user-facing prints.
Defaults off. Toggle via `set_verbose(True)` (e.g. from run_gradio.py --verbose)
or by setting the environment variable `SAT_VERBOSE=1`.
Warnings and errors should continue to use plain `print()` â€” `vprint()` is only
for informational/progress messages that are noisy at default verbosity.
"""

import os

VERBOSE = os.environ.get("SAT_VERBOSE", "0") == "1"


def set_verbose(v: bool):
    global VERBOSE
    VERBOSE = bool(v)


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

