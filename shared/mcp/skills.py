# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Persistent, shared Markdown skills exposed through ordinary MCP resources."""
import hashlib
import os
import re
import tempfile
from pathlib import Path

import yaml
from filelock import FileLock
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import Resource

USER_SKILL_PREFIX = "wangp://skills/user/"
MAX_SKILL_BYTES = 128 * 1024


def skill_metadata(content, name):
    match = re.fullmatch(r"---\r?\n(.*?)\r?\n---\r?\n(.+)", content, flags=re.DOTALL)
    if match is None or not match[2].strip():
        raise ValueError("A skill needs YAML front matter between --- lines and a nonempty Markdown body.")
    try:
        metadata = yaml.safe_load(match[1])
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid skill front matter: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("name") != name:
        raise ValueError("Skill front matter must contain name matching the skill's directory name.")
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("Skill description must be nonempty text of at most 1024 characters.")
    return {"name": name, "description": description.strip()}


class SkillStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._lock = FileLock(str(self.root / ".write.lock"), timeout=10)

    def path(self, name):
        if len(name) > 64 or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None or Path(name).is_reserved():
            raise ValueError("Skill name must be a lowercase hyphenated name, at most 64 characters, not a reserved filename.")
        path = (self.root / name / "SKILL.md").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Skill path must stay inside the user-skills directory.")
        return path

    def read(self, name):
        path = self.path(name)
        with self._lock:
            with path.open("rb") as reader:
                data = reader.read(MAX_SKILL_BYTES + 1)
        if len(data) > MAX_SKILL_BYTES:
            raise ValueError("A skill cannot exceed 128 KiB.")
        content = data.decode("utf-8")
        return {**skill_metadata(content, name), "uri": USER_SKILL_PREFIX + name, "revision": hashlib.sha256(data).hexdigest(), "content": content}

    def save(self, name: str, content: str, expected_revision: str | None = None):
        path = self.path(name)
        metadata = skill_metadata(content, name)
        data = content.encode("utf-8")
        if len(data) > MAX_SKILL_BYTES:
            raise ValueError("A skill cannot exceed 128 KiB.")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            revision = self.read(name)["revision"] if path.exists() else None
            if expected_revision != revision:
                raise ValueError("Skill revision conflict. Read the skill with mcp_resource, then update using its revision; omit expected_revision only when creating a new skill.")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".skill-", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as writer:
                    writer.write(data)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {"status": "done", **metadata, "uri": USER_SKILL_PREFIX + name, "revision": hashlib.sha256(data).hexdigest()}

    def resources(self):
        records = (self.read(path.parent.name) for path in sorted(self.root.glob("*/SKILL.md")))
        return [Resource(uri=item["uri"], name=item["name"], description=item["description"], mimeType="text/markdown") for item in records]


class SkillMCP(FastMCP):
    def __init__(self, *args, skills_root, **kwargs):
        self.skills = SkillStore(skills_root)
        super().__init__(*args, **kwargs)

    async def list_resources(self):
        return [*await super().list_resources(), *self.skills.resources()]

    async def read_resource(self, uri):
        if str(uri).startswith(USER_SKILL_PREFIX):
            record = self.skills.read(str(uri)[len(USER_SKILL_PREFIX):])
            return [ReadResourceContents(content=record["content"], mime_type="text/markdown", meta={"revision": record["revision"]})]
        return await super().read_resource(uri)
