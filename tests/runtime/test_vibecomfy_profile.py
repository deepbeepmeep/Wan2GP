from __future__ import annotations

import ast
from pathlib import Path

import pytest

from source.runtime.vibecomfy_profile import (
    PROCESS_DEFAULT_PROFILE,
    VibeComfyProfileProtocol,
    build_memory_profile_cli_args,
    build_profile_protocol,
    resolve_memory_profile,
)


def test_process_default_profiles_emit_numeric_cli_protocol() -> None:
    for profile in (1, 2, 3, 4, 5):
        protocol = build_profile_protocol(process_default=profile)

        assert protocol == VibeComfyProfileProtocol(memory_profile=profile)
        assert protocol.to_dict() == {"memory_profile": profile}
        assert protocol.to_cli_args() == ["--memory-profile", str(profile)]
        assert build_memory_profile_cli_args(process_default=profile) == [
            "--memory-profile",
            str(profile),
        ]


def test_override_profile_takes_precedence_over_process_default() -> None:
    assert resolve_memory_profile(process_default=1, override_profile=5) == 5
    assert build_memory_profile_cli_args(process_default=1, override_profile=5) == [
        "--memory-profile",
        "5",
    ]


def test_override_default_sentinel_uses_process_default() -> None:
    assert PROCESS_DEFAULT_PROFILE == -1
    assert resolve_memory_profile(process_default=3, override_profile=-1) == 3
    assert build_memory_profile_cli_args(process_default=3, override_profile=-1) == [
        "--memory-profile",
        "3",
    ]


def test_unset_process_default_preserves_absent_profile_flag() -> None:
    assert resolve_memory_profile(process_default=None, override_profile=-1) is None
    assert resolve_memory_profile(process_default=None, override_profile=None) is None
    assert build_profile_protocol(process_default=None, override_profile=-1) is None
    assert build_memory_profile_cli_args(process_default=None, override_profile=-1) == []


@pytest.mark.parametrize("bad_default", [-1, 0, 6, True, False, "3"])
def test_process_default_must_be_numeric_profile_1_to_5(bad_default: object) -> None:
    with pytest.raises(ValueError, match="process_default"):
        resolve_memory_profile(process_default=bad_default)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_override", [0, 6, True, False, "3"])
def test_override_profile_must_be_numeric_profile_1_to_5_or_default(
    bad_override: object,
) -> None:
    with pytest.raises(ValueError, match="override_profile"):
        resolve_memory_profile(process_default=2, override_profile=bad_override)  # type: ignore[arg-type]


def test_module_imports_no_vibecomfy_or_wgp_modules() -> None:
    module_path = Path(__file__).resolve().parents[2] / "source" / "runtime" / "vibecomfy_profile.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "vibecomfy" not in imported_roots
    assert "Wan2GP" not in imported_roots
    assert "source" not in imported_roots
