"""orchestrator/linux-only - not runnable on darwin-arm64 due to decord wheel gap; executor-side stub form is tests/test_wgp_patch_context_contracts.py::test_apply_all_wgp_patches_records_ltx2_runtime_fork_markers"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    if sys.platform != "linux":
        print(
            "patch_lifecycle_smoke.py is orchestrator/linux-only; "
            "skip it on darwin-arm64 and use the executor-side stub test instead.",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    wan_root = repo_root / "Wan2GP"
    if not wan_root.is_dir():
        print(f"Wan2GP root not found at {wan_root}", file=sys.stderr)
        return 1

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from source.models.wgp.orchestrator import WanOrchestrator
    from source.models.wgp.wgp_patches import clear_wgp_patch_context, get_wgp_patch_state

    previous_cwd = Path.cwd()
    clear_wgp_patch_context()

    try:
        os.chdir(wan_root)
        WanOrchestrator(str(wan_root))
        patch_state = get_wgp_patch_state()
        marker_state = patch_state.get("ltx2_runtime_fork_markers", {})
        marker_keys = [key for key in marker_state if key.startswith("model_def:ltx2")]
        if not marker_keys:
            print(
                "ltx2_runtime_fork_markers missing model_def:ltx2* entries after WanOrchestrator bootstrap",
                file=sys.stderr,
            )
            return 1

        print("patch lifecycle smoke OK")
        return 0
    except Exception as exc:  # pragma: no cover - orchestrator-only script
        print(f"patch lifecycle smoke failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        clear_wgp_patch_context()
        os.chdir(previous_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
