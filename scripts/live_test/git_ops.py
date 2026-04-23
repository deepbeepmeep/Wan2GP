"""Local git snapshot/push/restore helpers for Variant B takeovers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROTECTED_BRANCHES = {"main", "master"}


@dataclass(frozen=True)
class LocalSnapshot:
    original_branch: str | None
    original_sha: str
    is_dirty: bool
    stash_ref: str | None


def _repo_path(submodule_path: str) -> Path:
    return Path(submodule_path).resolve()


def _git(
    repo_path: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        check=check,
        text=True,
        capture_output=True,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_stash_ref(repo_path: Path, message: str) -> str | None:
    del message
    result = _git(repo_path, "stash", "list", "-1", "--format=%gd")
    ref = result.stdout.strip()
    return ref or None


def snapshot_local_state(submodule_path: str = "reigh-worker") -> LocalSnapshot:
    repo_path = _repo_path(submodule_path)
    try:
        branch = _git(repo_path, "symbolic-ref", "--short", "HEAD").stdout.strip() or None
    except subprocess.CalledProcessError:
        branch = None

    original_sha = _git(repo_path, "rev-parse", "HEAD").stdout.strip()
    status = _git(repo_path, "status", "--porcelain").stdout
    is_dirty = bool(status.strip())
    stash_ref = None

    if is_dirty:
        message = f"live-test-{_timestamp()}"
        _git(repo_path, "stash", "push", "--include-untracked", "-m", message)
        stash_ref = _find_stash_ref(repo_path, message)
        if not stash_ref:
            raise RuntimeError("Failed to capture stash reference for dirty working tree")

    return LocalSnapshot(
        original_branch=branch,
        original_sha=original_sha,
        is_dirty=is_dirty,
        stash_ref=stash_ref,
    )


def push_working_copy_to_temp_branch(
    submodule_path: str,
    snapshot: LocalSnapshot,
) -> tuple[str, str]:
    repo_path = _repo_path(submodule_path)
    branch_name = f"live-test/{_timestamp()}-{snapshot.original_sha[:8]}"
    if branch_name in PROTECTED_BRANCHES:
        raise ValueError(f"Refusing to use protected branch name: {branch_name}")

    _git(repo_path, "checkout", "-b", branch_name)

    if snapshot.is_dirty:
        if not snapshot.stash_ref:
            raise RuntimeError("Dirty snapshot is missing stash_ref")
        _git(repo_path, "stash", "apply", snapshot.stash_ref)
        _git(repo_path, "add", "-A")
        commit_status = _git(repo_path, "status", "--porcelain").stdout.strip()
        if commit_status:
            _git(repo_path, "commit", "-m", f"live-test: {_timestamp()}")

    _git(repo_path, "push", "-u", "origin", branch_name)
    current_sha = _git(repo_path, "rev-parse", "HEAD").stdout.strip()
    return branch_name, current_sha


def restore_local_state(submodule_path: str, snapshot: LocalSnapshot) -> None:
    repo_path = _repo_path(submodule_path)
    if snapshot.original_branch:
        _git(repo_path, "checkout", snapshot.original_branch)
        _git(repo_path, "reset", "--hard", snapshot.original_sha)
    else:
        _git(repo_path, "checkout", "--detach", snapshot.original_sha)

    if snapshot.is_dirty:
        if not snapshot.stash_ref:
            raise RuntimeError("Dirty snapshot is missing stash_ref during restore")
        _git(repo_path, "stash", "pop", snapshot.stash_ref)


def cleanup_temp_branch(
    branch: str,
    *,
    preserve: bool,
    submodule_path: str = "reigh-worker",
) -> str:
    if preserve:
        return branch

    repo_path = _repo_path(submodule_path)
    _git(repo_path, "push", "origin", "--delete", branch)
    _git(repo_path, "branch", "-D", branch)
    return branch


__all__ = [
    "LocalSnapshot",
    "PROTECTED_BRANCHES",
    "cleanup_temp_branch",
    "push_working_copy_to_temp_branch",
    "restore_local_state",
    "snapshot_local_state",
]
