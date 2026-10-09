# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import posixpath


def build_hf_url(repo_id, *path_parts):
    repo = (repo_id or "").strip("/")
    parts = [part.strip("/\\") for part in path_parts if part]
    path = posixpath.join(*parts) if parts else ""
    if not path:
        return f"https://huggingface.co/{repo}/resolve/main"
    return f"https://huggingface.co/{repo}/resolve/main/{path}"
