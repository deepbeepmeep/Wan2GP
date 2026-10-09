# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Small durable MCP receipts and transfer signing key; never stores media bytes."""

import base64
import contextlib
import hashlib
import hmac
import json
import os
import sqlite3
import time
from pathlib import Path


class MCPState:
    def __init__(self, root, scope="external"):
        identity = hashlib.sha256(f"{Path(root).resolve()}\n{scope}".encode()).hexdigest()[:24]
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "WanGP" / "mcp" / identity
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / "state.sqlite3"
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, job_id TEXT UNIQUE NOT NULL, snapshot TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('signing_key', ?)", (os.urandom(32),))
            self.key = db.execute("SELECT value FROM metadata WHERE name='signing_key'").fetchone()[0]

    @contextlib.contextmanager
    def transaction(self):
        with contextlib.closing(sqlite3.connect(self.path, timeout=30)) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db

    def seal(self, record):
        payload = base64.urlsafe_b64encode(json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode()).decode().rstrip("=")
        signature = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
        return payload + "." + signature

    def unseal(self, token, kind):
        try:
            if len(token) > 16384:
                raise ValueError()
            payload, signature = token.rsplit(".", 1)
            if not hmac.compare_digest(signature, hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()):
                raise ValueError()
            record = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            if record["kind"] != kind or record["expires"] <= time.time():
                raise ValueError()
            return record
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("Transfer expired or invalid; request a new transfer.") from exc

    def save_job(self, job_id, snapshot):
        with self.transaction() as db:
            row = db.execute("SELECT snapshot FROM requests WHERE job_id=?", (job_id,)).fetchone()
            if row:
                if snapshot.get("done"):
                    snapshot = dict(snapshot, status="done")
                db.execute("UPDATE requests SET snapshot=? WHERE job_id=?", (json.dumps({**json.loads(row[0]), **snapshot}, ensure_ascii=False), job_id))

    def get_job(self, job_id):
        with self.transaction() as db:
            row = db.execute("SELECT snapshot FROM requests WHERE job_id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None


class RecoveredJob:
    def __init__(self, snapshot, session, live_owner=None):
        self.saved = snapshot
        self.session = session
        self.live_owner = live_owner

    def queue_tasks(self):
        ids = set(self.saved.get("client_ids", []))
        return [task for task in self.session.list_queue()["tasks"] if task["client_id"] in ids] if ids else []

    def snapshot(self, event_limit=0):
        result = {key: value for key, value in self.saved.items() if key not in {"resolved_settings", "owner"}}
        result.update(recovered=True, events=[])
        if not result.get("done"):
            if self.live_owner is not None and self.saved.get("owner") == self.live_owner:
                result.update(status="submitting", note="The original submission is still being admitted, or its admission outcome is uncertain. Poll get_job; this retry will not submit another job.")
                return result
            result.update(status="unknown_after_restart", note="The server restarted before a durable outcome was recorded. This request will not be resubmitted. Inspect list_queue and Gallery outputs before deciding what to do.")
            queued = self.queue_tasks()
            if queued:
                result.update(status="running" if any(task["status"] != "queued" for task in queued) else "queued", queue_tasks=queued, note="Matched the receipt's client IDs to surviving queue tasks. Poll this job or cancel it; it will not be resubmitted.")
            from .server import _compact_gallery_record, _gallery_records
            ids = set(self.saved.get("client_ids", []))
            outputs = [record for record in _gallery_records(self.session, limit=500) if record["settings"].get("client_id") in ids]
            if outputs:
                result["recovered_outputs"] = [_compact_gallery_record(record) for record in outputs]
        return result

    def cancel(self):
        tasks = self.queue_tasks()
        if not tasks and not self.saved.get("done"):
            raise ValueError("No surviving queue tasks match this receipt; inspect Gallery outputs before retrying with a new request_id.")
        for task in tasks:
            self.session.cancel_queue_task(task["queue_id"])
