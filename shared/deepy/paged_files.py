"""Compact filesystem queries for the MCP v2 IO toolbox."""

import base64
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from shared.deepy import long_text
from shared.deepy.media_registry import detect_media_type
from shared.mcp_paging import PAGE_SIZE


def directory_entries(policy, path="", pattern="*", media_type="all"):
    if not path:
        yield from policy.roots()
        return
    if media_type not in {"all", "image", "video", "audio", "txt", "other"}:
        raise ValueError("media_type must be all, image, video, audio, txt, or other.")
    if "/" in pattern or "\\" in pattern or "**" in pattern:
        raise ValueError("list navigates one directory; use rg for recursive file search.")
    root = policy.require_read(path, directory=True)
    for entry in sorted(root.glob(pattern), key=lambda item: item.name.casefold()):
        if not policy.can_read(entry):
            continue
        kind = "directory" if entry.is_dir() else "file"
        if media_type != "all":
            detected = detect_media_type(str(entry))
            match = entry.suffix.lower() == ".txt" if media_type == "txt" else detected == "any" and entry.suffix.lower() != ".txt" if media_type == "other" else detected == media_type
            if kind == "directory" or not match:
                continue
        yield {"path": policy.virtualize_path(entry), "type": kind}


def list_directory_page(pages, policy, *, path="", pattern="*", media_type="all", limit=PAGE_SIZE, cursor=None, summary_only=False):
    """Use the same bounded, snapshot-backed listing for MCP v2 and Deepy Zero."""
    filters = {"path": path, "pattern": pattern, "media_type": media_type}
    records = None if cursor else directory_entries(policy, **filters)
    return pages.page(["io", "list", filters], records, key="entries", limit=limit, cursor=cursor, metadata={"complete": True}, summary_only=summary_only)


def rg_records(policy, arguments, metadata):
    command, paths = long_text._parse_rg_arguments(arguments, experimental=False)
    if "-L" in command:
        raise ValueError("Following filesystem links with -L is unavailable; use --files-without-match to list nonmatching files.")
    if not paths:
        mount = long_text.workspace_mount(policy)
        if mount is None:
            raise ValueError("Provide a search path after --; this server has no session workspace.")
        paths = [mount[1]]
    resolved = [policy.require_read(path) for path in paths]
    names = bool(set(command) & {"--files", "-l", "--files-with-matches", "--files-without-match"})
    text_output = bool(set(command) & {"-c", "--count", "--count-matches", "-o", "--only-matching", "-r", "--replace"}) or any(value.startswith(("--replace=", "-r")) for value in command if value.startswith("-"))
    if "--stats" in command:
        raise ValueError("Use summary_only=true for a compact search summary instead of --stats.")
    flags = ["--sort=path", "--null"] if names else ["--sort=path", "--with-filename", "--line-number", *([] if text_output else ["--json"])]
    invocation = [str(long_text.ensure_ripgrep()), "--no-config", "--color=never", *command, *flags, "--", *map(str, resolved)]
    environment = dict(os.environ)
    environment.pop("RIPGREP_CONFIG_PATH", None)
    timed_out = threading.Event()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(invocation, cwd=str(resolved[0] if resolved[0].is_dir() else resolved[0].parent), stdout=subprocess.PIPE, stderr=errors, env=environment, shell=False)

        def stop():
            timed_out.set()
            process.kill()

        timer = threading.Timer(30, stop)
        timer.daemon = True
        timer.start()
        try:
            if names:
                pending = b""
                while chunk := process.stdout.read(8192):
                    pending += chunk
                    parts = pending.split(b"\0")
                    pending = parts.pop()
                    for value in parts:
                        path = Path(value.decode("utf-8", errors="replace"))
                        if policy.can_read(path):
                            yield policy.virtualize_path(path)
                if pending:
                    metadata.update(complete=False, reason="Search ended during a filename.")
            else:
                while line := process.stdout.readline(4 * 1024 * 1024 + 1):
                    if len(line) > 4 * 1024 * 1024:
                        metadata.update(complete=False, reason="A search record exceeds 4 MiB; narrow the query.")
                        process.kill()
                        break
                    if text_output:
                        text = policy._virtualize_text(line.decode("utf-8", errors="replace").rstrip())
                        yield {"text": text[:1000], **({"excerpt_truncated": True} if len(text) > 1000 else {})}
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if timed_out.is_set():
                            break
                        raise
                    if record["type"] not in {"match", "context"}:
                        continue
                    data = record["data"]

                    def decode(value):
                        return value["text"] if "text" in value else base64.b64decode(value["bytes"]).decode("utf-8", errors="replace")

                    path = Path(decode(data["path"]))
                    if not policy.can_read(path):
                        continue
                    text = decode(data["lines"]).rstrip("\r\n")
                    result = {"path": policy.virtualize_path(path), "line": data["line_number"], "text": text[:1000]}
                    if len(text) > 1000:
                        result["excerpt_truncated"] = True
                    if record["type"] == "context":
                        result["context"] = True
                    yield result
            code = process.wait()
            if timed_out.is_set():
                metadata.update(complete=False, reason="Search timed out after 30 seconds; narrow the query.")
            elif code not in {0, 1}:
                errors.seek(0)
                metadata.update(complete=False, reason=policy._virtualize_text(errors.read(2000).decode("utf-8", errors="replace")) or f"rg exited with code {code}.")
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
