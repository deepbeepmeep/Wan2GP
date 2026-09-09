from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any


_COALESCIBLE_TYPES = {"append_block_text", "stats", "status"}
_SYNC_SENTINEL = "deepy_sync_recovery"


class DeepyPublicationQueue:
    """Thread-safe Deepy-to-Gradio handoff with bounded token-event backlog."""

    def __init__(self, sync_factory: Callable[[], str], *, max_pending_events: int = 128, max_pending_bytes: int = 2 * 1024 * 1024):
        self.sync_factory = sync_factory
        self.max_pending_events = int(max_pending_events)
        self.max_pending_bytes = int(max_pending_bytes)
        self.queue: list[tuple[str, Any, dict[str, Any] | None]] = []
        self.condition = threading.Condition()
        self.session_id = ""
        self.sync_pending = False
        self.pending_bytes = 0
        self.events_received = 0
        self.events_enqueued = 0
        self.events_coalesced = 0
        self.events_invalidated = 0
        self.events_recovered = 0
        self.max_depth = 0
        self.max_bytes = 0
        self.publication_in_flight = False

    @staticmethod
    def _metadata(data: str) -> dict[str, Any] | None:
        envelope = json.loads(data)
        event = envelope.get("event")
        if not isinstance(event, dict):
            return None
        event_type = str(event.get("type", ""))
        metadata = {"type": event_type, "session_id": str(event.get("chat_session_id", "")), "pending_size": len(data)}
        if event_type in _COALESCIBLE_TYPES:
            metadata.update({"envelope": envelope, "event": event})
        return metadata

    @staticmethod
    def _size(data: Any) -> int:
        return len(data) if isinstance(data, str) else 0

    def _append(self, cmd: str, data: Any, metadata: dict[str, Any] | None) -> None:
        self.queue.append((cmd, data, metadata))
        self.pending_bytes += self._item_size(self.queue[-1])
        self.events_enqueued += 1

    def _item_size(self, item: tuple[str, Any, dict[str, Any] | None]) -> int:
        metadata = item[2]
        return int(metadata["pending_size"]) if metadata is not None else self._size(item[1])

    def _remove(self, index: int) -> tuple[str, Any, dict[str, Any] | None]:
        item = self.queue.pop(index)
        self.pending_bytes -= self._item_size(item)
        return item

    def _clear(self) -> None:
        self.events_invalidated += len(self.queue)
        self.queue.clear()
        self.pending_bytes = 0
        self.sync_pending = False

    def _coalesce_append(self, data: str, metadata: dict[str, Any]) -> bool:
        incoming = metadata["event"]
        if not self.queue:
            return False
        cmd, _previous_data, previous_metadata = self.queue[-1]
        if cmd != "chat_output" or previous_metadata is None or previous_metadata["type"] != "append_block_text":
            return False
        previous = previous_metadata["event"]
        same_block = previous.get("message_id") == incoming.get("message_id") and previous.get("block_id") == incoming.get("block_id")
        contiguous = int(previous.get("text_end", -1)) == int(incoming.get("text_start", -2))
        if not same_block or not contiguous:
            return False
        chunks = previous_metadata.setdefault("chunks", [str(previous.get("text", ""))])
        suffix = str(incoming.get("text", ""))
        chunks.append(suffix)
        previous["text"] = ""
        previous["text_end"] = incoming.get("text_end")
        previous["revision"] = incoming.get("revision")
        previous["sequence"] = incoming.get("sequence")
        previous["sequence_start"] = previous.get("sequence_start", previous.get("sequence"))
        merged_envelope = previous_metadata["envelope"]
        merged_envelope["event_id"] = metadata["envelope"].get("event_id", merged_envelope.get("event_id"))
        previous_metadata["pending_size"] += len(suffix)
        self.pending_bytes += len(suffix)
        self.queue[-1] = (cmd, None, previous_metadata)
        self.events_coalesced += 1
        return True

    def _coalesce_latest(self, event_type: str) -> None:
        for index in range(len(self.queue) - 1, -1, -1):
            cmd, _data, metadata = self.queue[index]
            if cmd != "chat_output" or metadata is None:
                break
            previous_type = metadata["type"]
            if previous_type == event_type:
                self._remove(index)
                self.events_coalesced += 1
                return
            if previous_type not in _COALESCIBLE_TYPES:
                return

    def _coalescible_count(self) -> int:
        return sum(1 for _cmd, _data, metadata in self.queue if metadata is not None and metadata["type"] in _COALESCIBLE_TYPES)

    def _schedule_sync_recovery(self) -> None:
        if self.sync_pending:
            return
        retained = []
        for item in self.queue:
            metadata = item[2]
            if metadata is not None and metadata["type"] in _COALESCIBLE_TYPES:
                self.events_invalidated += 1
                self.pending_bytes -= self._item_size(item)
            else:
                retained.append(item)
        self.queue = retained
        self.queue.append((_SYNC_SENTINEL, None, None))
        self.sync_pending = True
        self.events_recovered += 1

    def _remove_sync_sentinel(self) -> None:
        for index, item in enumerate(self.queue):
            if item[0] == _SYNC_SENTINEL:
                self.queue.pop(index)
                self.sync_pending = False
                self.events_coalesced += 1
                return

    def _record_high_water(self) -> None:
        self.max_depth = max(self.max_depth, len(self.queue))
        self.max_bytes = max(self.max_bytes, self.pending_bytes)

    def push(self, cmd: str, data: Any = None) -> None:
        with self.condition:
            self.events_received += 1
            metadata = self._metadata(data) if cmd == "chat_output" else None
            if metadata is not None:
                event_type = metadata["type"]
                incoming_session = metadata["session_id"]
                if incoming_session and self.session_id and incoming_session != self.session_id:
                    self._clear()
                if incoming_session:
                    self.session_id = incoming_session
                if event_type == "reset":
                    self._clear()
                elif event_type == "sync" and self.sync_pending:
                    self._remove_sync_sentinel()
                if self.sync_pending and event_type in _COALESCIBLE_TYPES:
                    self.events_invalidated += 1
                    return
                if event_type == "append_block_text" and self._coalesce_append(data, metadata):
                    self._record_high_water()
                    if self.pending_bytes > self.max_pending_bytes:
                        self._schedule_sync_recovery()
                    self.condition.notify()
                    return
                if event_type in {"status", "stats"}:
                    self._coalesce_latest(event_type)
            self._append(cmd, data, metadata)
            self._record_high_water()
            if metadata is not None and metadata["type"] in _COALESCIBLE_TYPES and (self._coalescible_count() > self.max_pending_events or self.pending_bytes > self.max_pending_bytes):
                self._schedule_sync_recovery()
            self.condition.notify()

    def _materialize(self, item):
        cmd, data, metadata = item
        if cmd == _SYNC_SENTINEL:
            return "chat_output", self.sync_factory()
        if metadata is not None and "chunks" in metadata:
            metadata["event"]["text"] = "".join(metadata.pop("chunks"))
            data = json.dumps(metadata["envelope"], ensure_ascii=False)
        return cmd, data

    def pop(self):
        with self.condition:
            if not self.queue:
                return None
            item = self._remove(0)
            if item[0] == _SYNC_SENTINEL:
                self.sync_pending = False
        return self._materialize(item)

    def top(self):
        with self.condition:
            if not self.queue:
                return None
            cmd, data, _metadata = self.queue[0]
            return ("chat_output", None) if cmd == _SYNC_SENTINEL or data is None else (cmd, data)

    def next(self):
        with self.condition:
            while not self.queue:
                self.condition.wait()
            item = self._remove(0)
            if item[0] == _SYNC_SENTINEL:
                self.sync_pending = False
            materialized = self._materialize(item)
            if materialized[0] == "chat_output":
                self.publication_in_flight = True
            return materialized

    def complete_publication(self) -> None:
        with self.condition:
            self.publication_in_flight = False
            self.condition.notify_all()

    def wait_for_chat_publication(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.condition:
            while self.publication_in_flight or any(item[0] in {"chat_output", _SYNC_SENTINEL} for item in self.queue):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return True

    def metrics(self) -> dict[str, int]:
        with self.condition:
            return {
                "received": self.events_received,
                "enqueued": self.events_enqueued,
                "coalesced": self.events_coalesced,
                "invalidated": self.events_invalidated,
                "sync_recoveries": self.events_recovered,
                "depth": len(self.queue),
                "pending_bytes": self.pending_bytes,
                "max_depth": self.max_depth,
                "max_pending_bytes": self.max_bytes,
            }
