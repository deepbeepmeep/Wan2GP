"""Bounded MCP pages over session-local, immutable result snapshots."""

import base64
import hmac
import json
import secrets
import tempfile
import threading
import time


PAGE_SIZE = 20
PAGE_MAX_CHARS = 6000
SNAPSHOT_MAX_BYTES = 128 * 1024 * 1024
SNAPSHOT_TTL = 600
SNAPSHOT_COUNT = 16


class ResultPages:
    def __init__(self):
        self._snapshots = {}
        self._secret = secrets.token_bytes(32)
        self._lock = threading.RLock()

    def close(self):
        with self._lock:
            for snapshot in self._snapshots.values():
                snapshot["file"].close()
            self._snapshots.clear()

    def _cursor(self, identity, position):
        value = f"{identity}:{position}".encode()
        signature = hmac.digest(self._secret, value, "sha256")[:12]
        return base64.urlsafe_b64encode(value + b"." + signature).decode().rstrip("=")

    def _resume(self, cursor, query):
        try:
            data = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            value, signature = data[:-13], data[-12:]
            if data[-13:-12] != b"." or not hmac.compare_digest(signature, hmac.digest(self._secret, value, "sha256")[:12]):
                raise ValueError()
            identity, position = value.decode().split(":")
            snapshot = self._snapshots[identity]
            if snapshot["query"] != query:
                raise ValueError("The cursor belongs to different filters; repeat the original query.")
            return identity, int(position), snapshot
        except (KeyError, UnicodeError, ValueError) as exc:
            raise ValueError("Invalid or expired cursor, or changed filters. Repeat the original search without cursor.") from exc

    def page(self, query, records=None, *, key="items", limit=PAGE_SIZE, cursor=None, metadata=None, summary_only=False, max_chars=PAGE_MAX_CHARS):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100.")
        query = json.dumps(query, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            now = time.monotonic()
            for identity, snapshot in list(self._snapshots.items()):
                if now - snapshot["created"] > SNAPSHOT_TTL:
                    snapshot["file"].close()
                    del self._snapshots[identity]
            if cursor:
                identity, position, snapshot = self._resume(cursor, query)
            else:
                if len(self._snapshots) >= SNAPSHOT_COUNT:
                    oldest = next(iter(self._snapshots))
                    self._snapshots.pop(oldest)["file"].close()
                identity, position = secrets.token_hex(12), 0
                writer = tempfile.TemporaryFile(mode="w+b")
                count = 0
                try:
                    for record in records:
                        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        if writer.tell() + len(line) > SNAPSHOT_MAX_BYTES:
                            raise ValueError("Search snapshot exceeds 128 MiB. Narrow the search.")
                        writer.write(line)
                        count += 1
                    snapshot = {"file": writer, "query": query, "created": now, "count": count, "metadata": dict(metadata or {})}
                    self._snapshots[identity] = snapshot
                except BaseException:
                    writer.close()
                    raise
            reader = snapshot["file"]
            reader.seek(position)
            items, chars = [], 0
            if not summary_only:
                while len(items) < limit:
                    start = reader.tell()
                    line = reader.readline()
                    if not line:
                        break
                    text = line.decode("utf-8")
                    if chars + len(text) > max_chars:
                        reader.seek(start)
                        if not items:
                            raise ValueError("One result exceeds the page budget. Narrow the query or request a smaller excerpt.")
                        break
                    items.append(json.loads(text))
                    chars += len(text)
            position = reader.tell()
            has_more = bool(reader.read(1))
            result = {"status": "done", **snapshot["metadata"], key: items, "count": len(items), "has_more": has_more, "next_cursor": self._cursor(identity, position) if has_more else None}
            if summary_only:
                result["summary"] = {"stored_results": snapshot["count"], "expires_in_seconds": max(0, int(SNAPSHOT_TTL - (now - snapshot["created"])))}
            return result


def strip_schema_titles(schema):
    """Remove schema annotations, preserving properties named title and literal data."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    mappings = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
    children = {"items", "additionalProperties", "contains", "not", "if", "then", "else", "propertyNames", "unevaluatedProperties"}
    arrays = {"allOf", "anyOf", "oneOf", "prefixItems"}
    for key, value in schema.items():
        if key == "title":
            continue
        if key in mappings and isinstance(value, dict):
            value = {name: strip_schema_titles(child) for name, child in value.items()}
        elif key in children:
            value = strip_schema_titles(value)
        elif key in arrays:
            value = [strip_schema_titles(child) for child in value]
        result[key] = value
    return result
