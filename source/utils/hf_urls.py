"""Hugging Face URL helpers."""

from __future__ import annotations

import re


def is_commit_hash(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", value or ""))


def hf_resolve_url(repo_id: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


__all__ = ["hf_resolve_url", "is_commit_hash"]
