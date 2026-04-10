"""Terminal status display with growing plant animation."""

from __future__ import annotations

import os
import sys
import time

# Enable VT100 escape sequences on Windows cmd.exe
if os.name == "nt":
    os.system("")

_PLANT_STAGES = [
    # Dormant seed
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "  .  ", " ▔▔▔ "],
    # Seed cracks
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "  ,  ", " ▔▔▔ "],
    # Tiny sprout
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "  |  ", " ▔▔▔ "],
    # Sprout grows
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "  |  ", "  |  ", " ▔▔▔ "],
    # First leaf right
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "  |  ", "  |/ ", " ▔▔▔ "],
    # First leaf left
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", " \\|  ", "  |/ ", " ▔▔▔ "],
    # Growing taller
    ["     ", "     ", "     ", "     ", "     ", "     ", " \\|  ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Second leaf right
    ["     ", "     ", "     ", "     ", "     ", "     ", " \\|/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Growing more
    ["     ", "     ", "     ", "     ", "     ", " \\|/ ", "  |  ", " \\|  ", "  |/ ", " ▔▔▔ "],
    # Third branch
    ["     ", "     ", "     ", "     ", "     ", " \\|/ ", "  |  ", " \\|/ ", "  |  ", " ▔▔▔ "],
    # Taller still
    ["     ", "     ", "     ", "     ", " \\|/ ", "  |  ", " \\|/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Bud appears
    ["     ", "     ", "     ", "  .  ", " \\|/ ", "  |  ", " \\|/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Bud opens
    ["     ", "     ", "     ", "  *  ", " \\|/ ", "  |  ", " \\|/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Full bloom
    ["     ", "     ", "  ~  ", "  *  ", " \\|/ ", "  |  ", " \\|/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Bloom fades
    ["     ", "     ", "  .  ", "  |  ", " \\|/ ", "  |  ", " \\|/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Top leaves fall
    ["     ", "     ", "     ", "  |  ", "  |/ ", "  |  ", " \\|/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # More leaves fall
    ["     ", "     ", "     ", "  |  ", "  |  ", "  |  ", "  |/ ", "  |  ", "  |/ ", " ▔▔▔ "],
    # Sparse
    ["     ", "     ", "     ", "  |  ", "  |  ", "  |  ", "  |  ", "  |  ", "  |  ", " ▔▔▔ "],
    # Shrinking
    ["     ", "     ", "     ", "     ", "  |  ", "  |  ", "  |  ", "  |  ", "  |  ", " ▔▔▔ "],
    # Smaller
    ["     ", "     ", "     ", "     ", "     ", "  |  ", "  |  ", "  |  ", "  |  ", " ▔▔▔ "],
    # Retreating
    ["     ", "     ", "     ", "     ", "     ", "     ", "  |  ", "  |  ", "  |  ", " ▔▔▔ "],
    # Almost gone
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "  |  ", "  |  ", " ▔▔▔ "],
    # Stub
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "  |  ", " ▔▔▔ "],
    # Back to seed
    ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "  .  ", " ▔▔▔ "],
]

_PULSE = ["·    ", "· ·  ", "· · ·", "  · ·", "    ·", "     "]

_HEIGHT = 12  # 10 plant rows + 1 blank + 1 pulse line
_UP = f"\033[{_HEIGHT}A"
_CLEAR = "\033[K"


class WorkerStatusDisplay:
    """Animated terminal status with a growing plant."""

    def __init__(self, gpu_name: str, profile_label: str):
        self._gpu = gpu_name
        self._profile = profile_label
        self._tasks_done = 0
        self._start = time.time()
        self._tick = 0
        self._active = False

    def show_banner(self) -> None:
        """Print the startup banner once."""
        print(f"\n  ✅ Worker ready — {self._gpu}, {self._profile}\n", flush=True)
        # Print blank lines to reserve space for the display
        print("\n" * _HEIGHT, end="", flush=True)
        self._active = True

    def show_idle(self) -> None:
        """Redraw the status block. Called each poll cycle while idle."""
        if not self._active:
            return
        stage = _PLANT_STAGES[self._tick % len(_PLANT_STAGES)]
        pulse = _PULSE[self._tick % len(_PULSE)]
        self._tick += 1

        elapsed = time.time() - self._start
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        uptime = f"{h}h {m}m" if h > 0 else f"{m}m"
        tasks = f"{self._tasks_done} task{'s' if self._tasks_done != 1 else ''}"

        info_lines = {
            6: self._gpu,
            7: f"{self._profile}  ·  {uptime} up  ·  {tasks}",
            8: "waiting for tasks...",
        }

        sys.stdout.write(_UP)
        for row in range(10):
            plant = stage[row]
            info = info_lines.get(row, "")
            sys.stdout.write(f"  {info:<45s}{plant}{_CLEAR}\n")
        sys.stdout.write(f"{_CLEAR}\n")  # blank separator
        sys.stdout.write(f"  {pulse}{_CLEAR}\n")
        sys.stdout.flush()

    def on_task_start(self) -> None:
        """Clear the display before task output."""
        if not self._active:
            return
        sys.stdout.write(_UP)
        for _ in range(_HEIGHT):
            sys.stdout.write(f"{_CLEAR}\n")
        sys.stdout.write(_UP)
        sys.stdout.flush()

    def on_task_done(self) -> None:
        """Increment counter and restart plant from seed."""
        self._tasks_done += 1
        self._tick = 0
        # Re-reserve space
        print("\n" * _HEIGHT, end="", flush=True)
        self.show_idle()
