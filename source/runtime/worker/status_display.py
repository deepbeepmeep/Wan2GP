"""Terminal status display with dandelion lifecycle animation."""

from __future__ import annotations

import os
import sys
import time

# Enable VT100 escape sequences on Windows cmd.exe
if os.name == "nt":
    os.system("")

# Each entry is (frame, delay_seconds). 12 rows, 7 chars wide.
# Ground ▔▔▔ locked at row 11. Plant grows up from there.
_PLANT_STAGES = [
    # --- Spring (slow, quiet) ---
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "  ▔▔▔  "], 30),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "   .   ", "  ▔▔▔  "], 25),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "  ▔▔▔  "], 20),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "  ,|,  ", "   |   ", "  ▔▔▔  "], 20),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "  ,|,  ", "   |   ", "  ▔▔▔  "], 15),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 15),
    # --- Growing (picking up pace) ---
    (["       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 12),
    (["       ", "       ", "       ", "       ", "       ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 10),
    # --- Bud (slows for anticipation) ---
    (["       ", "       ", "       ", "       ", "   .   ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 20),
    (["       ", "       ", "       ", "       ", "  (.)  ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 20),
    # --- Bloom (medium pace) ---
    (["       ", "       ", "       ", "       ", "  {o}  ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 15),
    (["       ", "       ", "       ", "       ", " -{O}- ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 15),
    (["       ", "       ", "       ", "  \\=/  ", " -{O}- ", "  /=\\  ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 25),
    (["       ", "       ", "       ", "  \\=/  ", " -{O}- ", "  /=\\  ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 25),
    # --- Fading (slow, wistful) ---
    (["       ", "       ", "       ", "       ", " -{o}- ", "  /=\\  ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 20),
    (["       ", "       ", "       ", "       ", "  {o}  ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 15),
    (["       ", "       ", "       ", "       ", "  :o:  ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 15),
    # --- Seed head (slow build) ---
    (["       ", "       ", "       ", "   :   ", "  :O:  ", "   :   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 20),
    (["       ", "       ", "       ", "  .:.  ", "  :O:  ", "  ':'  ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 25),
    (["       ", "       ", "       ", "  .:.  ", "  :O:  ", "  ':'  ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 30),
    # --- Wind! (fast burst) ---
    (["       ", "       ", "    .  ", "  .: . ", "  :O:  ", "  ':'  ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 6),
    (["       ", "    . .", "      .", "   :   ", "  :o:  ", "   :   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 5),
    (["    . .", "      .", "       ", "       ", "   o   ", "   :   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 5),
    (["      .", "       ", "       ", "       ", "   .   ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 8),
    # --- Autumn (slowing down) ---
    (["       ", "       ", "       ", "       ", "       ", "   |   ", "   |   ", "   |   ", "   |   ", " ,,|,, ", "   |   ", "  ▔▔▔  "], 15),
    (["       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "   |   ", "   |   ", " ,.|., ", "   |   ", "  ▔▔▔  "], 20),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "   |   ", "  .|.  ", "   |   ", "  ▔▔▔  "], 25),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "   |   ", "   |   ", "  ▔▔▔  "], 20),
    # --- Winter (very slow, still) ---
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "   |   ", "  ▔▔▔  "], 30),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "   .   ", "  ▔▔▔  "], 30),
    (["       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "       ", "  ▔▔▔  "], 40),
]

_DOTS_FREE = ["GPU idle · waiting for tasks   ", "GPU idle · waiting for tasks.  ", "GPU idle · waiting for tasks.. ", "GPU idle · waiting for tasks..."]
_DOTS_ACTIVE = ["GPU ready · waiting for tasks   ", "GPU ready · waiting for tasks.  ", "GPU ready · waiting for tasks.. ", "GPU ready · waiting for tasks..."]

_ROWS = 12
_HEIGHT = _ROWS + 3  # plant rows + 3 blank below
_UP = f"\033[{_HEIGHT}A"
_CLEAR = "\033[K"


class WorkerStatusDisplay:
    """Animated terminal status with a dandelion lifecycle."""

    def __init__(self, gpu_name: str, profile_label: str):
        self._gpu = gpu_name
        self._profile = profile_label
        self._tasks_done = 0
        self._start = time.time()
        self._tick = 0
        self._active = False

    def show_banner(self) -> None:
        """Reserve space for the display."""
        print(flush=True)
        print("\n" * _HEIGHT, end="", flush=True)
        self._active = True

    def show_idle(self) -> None:
        """Redraw the status block. Called each poll cycle while idle."""
        if not self._active:
            return
        stage, _delay = _PLANT_STAGES[self._tick % len(_PLANT_STAGES)]
        dot_set = _DOTS_ACTIVE if self._tasks_done > 0 else _DOTS_FREE
        dots = dot_set[self._tick % len(dot_set)]
        self._tick += 1

        elapsed = time.time() - self._start
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        uptime = f"{h}h {m}m" if h > 0 else f"{m}m"
        tasks = f"{self._tasks_done} task{'s' if self._tasks_done != 1 else ''}"

        info_lines = {
            2: self._gpu,
            6: f"{self._profile}  ·  {uptime} up  ·  {tasks}",
            10: dots,
        }

        sys.stdout.write(_UP)
        for row in range(_ROWS):
            plant = stage[row]
            info = info_lines.get(row, "")
            sys.stdout.write(f"  {info:<43s}{plant}{_CLEAR}\n")
        sys.stdout.write(f"{_CLEAR}\n")
        sys.stdout.write(f"{_CLEAR}\n")
        sys.stdout.write(f"{_CLEAR}\n")
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
        print("\n" * _HEIGHT, end="", flush=True)
        self.show_idle()
