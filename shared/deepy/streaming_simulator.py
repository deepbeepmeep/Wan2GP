from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from shared.deepy.filesystem import FileAccessPolicy
from shared.gradio import assistant_chat


CADENCE_PRESETS = {"Very slow": 1.0, "Normal (250 ms)": 0.25, "Fast burst": 0.0, "Backlogged": 0.01}


@dataclass
class SimulatedChatSession:
    chat_transcript: list[dict] = field(default_factory=list)
    chat_transcript_counter: int = 0
    chat_revision: int = 0
    chat_event_sequence: int = 0
    chat_session_id: str = "deepy-streaming-simulator"
    chat_status: dict | None = None
    remote_usage_stats: dict | None = None
    file_access_policy: FileAccessPolicy | None = None


class DeepyStreamingSimulator:
    def __init__(self, repo_root: str | Path):
        root = Path(repo_root).resolve()
        self.session = SimulatedChatSession(file_access_policy=FileAccessPolicy(mode="read", output_roots=(root,), root_aliases=("workspace",)))

    @staticmethod
    def _chunks(text: str, size: int) -> Iterator[str]:
        for end in range(size, len(text), size):
            yield text[:end]
        yield text

    def reset(self) -> str:
        assistant_chat.reset_session_chat(self.session)
        return assistant_chat.build_reset_event(self.session)

    def sync(self) -> str:
        return assistant_chat.build_sync_event(self.session, stats=self.session.remote_usage_stats)

    def interrupt(self) -> str:
        assistant_id = next((str(message.get("id", "")) for message in reversed(self.session.chat_transcript) if message.get("role") == "assistant"), "")
        return assistant_chat.set_message_end_badge(self.session, assistant_id, "Interrupted by simulator Stop") or self.sync()

    def events(self, *, scale: int = 1, include_recovery_glitches: bool = True) -> Iterator[str]:
        yield self.reset()
        yield assistant_chat.build_status_event("Simulator streaming…", kind="generating", session=self.session)
        user_id, payload = assistant_chat.add_user_message(self.session, "Exercise every incremental Deepy UI state, then leave the completed transcript visible.")
        yield payload
        assistant_id = assistant_chat.create_assistant_turn(self.session)

        thought = (
            "I’m checking the incremental renderer without a model.\n\n"
            "- Preserve **emphasis** and `inline_code`.\n"
            "* Keep list markers literal while this line streams: snake_case_name, __torch_function__, and *.md.\n"
            "- Keep disclosure state stable while partial tokens arrive.\n\n"
            "Request ledger without a Markdown block break:\n"
            "1. This enumeration stays aligned with the preceding text.\n"
            "2. It must not gain temporary list indentation.\n\n"
            "A proper Markdown list keeps its indentation:\n\n"
            "1. First listed item.\n"
            "2. Second listed item.\n\n"
            "```python\nfor index in range(3):\n    print(index)\n```\n\n"
            "A partial Markdown token follows: **safe while streaming, complete when finalized**."
        ) * max(1, scale)
        thought_id = ""
        for index, text in enumerate(self._chunks(thought, max(24, len(thought) // (8 * max(1, scale))))):
            thought_id, payload = assistant_chat.upsert_reasoning_block(self.session, assistant_id, thought_id, text, streaming=True)
            if payload:
                yield payload
            if index == 2:
                self.session.remote_usage_stats = {"visible": True, "text": "1,024 / 131,072 tokens · simulated", "tokens": 1024, "max_tokens": 131072}
                yield assistant_chat.build_stats_event(self.session.remote_usage_stats)
        thought_id, payload = assistant_chat.upsert_reasoning_block(self.session, assistant_id, thought_id, thought, streaming=False)
        if payload:
            yield payload

        summary = "\n\n".join(f"### Earlier item {index}\nContext paragraph {index}: " + "large stable history " * 24 for index in range(1, 26 * max(1, scale) + 1))
        summary_id = ""
        yield assistant_chat.build_status_event("Compacting earlier context…", kind="compacting", session=self.session)
        for text in self._chunks(summary, max(800, len(summary) // 12)):
            summary_id, payload = assistant_chat.upsert_context_summary(self.session, assistant_id, summary_id, text, streaming=True)
            if payload:
                yield payload
        summary_id, payload = assistant_chat.upsert_context_summary(self.session, assistant_id, summary_id, summary, streaming=False)
        if payload:
            yield payload
        yield assistant_chat.build_status_event("Continuing after compaction…", kind="generating", session=self.session)

        followup = "The large summary is complete. This later thought must remain fast and must not rerender that summary."
        followup_id = ""
        for text in self._chunks(followup, 13):
            followup_id, payload = assistant_chat.upsert_reasoning_block(self.session, assistant_id, followup_id, text, streaming=True)
            if payload:
                yield payload
        followup_id, payload = assistant_chat.upsert_reasoning_block(self.session, assistant_id, followup_id, followup, streaming=False)
        if payload:
            yield payload

        statement = (
            "Streaming stays escaped: <script>window.__unsafe_stream_executed = true</script> & <b>not raw HTML</b>.\n\n"
            "Paths: `C:\\Users\\Example\\clip.mp4`, `/tmp/example.png`, and authorized `@workspace/docs/DEEPY_INCREMENTAL_STREAMING_SPEC.md`.\n\n"
            "Links: [OpenAI](https://openai.com), ![gallery image](/wangp_api/gallery/media/visual:0123456789ab), "
            "[video](/wangp_api/gallery/media/visual:abcdef012345), and [audio](/wangp_api/gallery/media/audio:abcdef012345)."
        )
        statement_id = ""
        for text in self._chunks(statement, 37):
            statement_id, payload = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, statement_id, text, streaming=True)
            if payload:
                yield payload
        statement_id, payload = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, statement_id, statement, streaming=False)
        if payload:
            yield payload

        tool_id, payload = assistant_chat.add_tool_call(self.session, assistant_id, "list_files", {"path": "@workspace/docs", "extensions": [".md"]}, tool_label="Preparing Test File Listing", request_pending=True)
        yield payload
        yield assistant_chat.update_tool_call(self.session, assistant_id, tool_id, status="running", status_text="Running", tool_label="List Test Files", arguments={"path": "@workspace/docs", "extensions": [".md"]}, request_pending=False)
        result = {
            "status": "ok",
            "entries": [{"index": index, "name": f"reference_{index}.md", "path": "@workspace/docs/DEEPY_INCREMENTAL_STREAMING_SPEC.md", "size_bytes": 1000 + index} for index in range(24 * max(1, scale))],
            "output_file": "@workspace/docs/DEEPY_INCREMENTAL_STREAMING_SPEC.md",
        }
        yield assistant_chat.complete_tool_call(self.session, assistant_id, tool_id, result)

        second_text = "A second statement after the completed tool proves ordering across several thought/tool/statement cycles."
        second_id = ""
        for text in self._chunks(second_text, 18):
            second_id, payload = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, second_id, text, streaming=True)
            if payload:
                yield payload
        second_id, payload = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, second_id, second_text, streaming=False)
        if payload:
            yield payload

        self.session.remote_usage_stats = {"visible": True, "text": "48,320 / 131,072 tokens · 37.4 tok/s", "tokens": 48320, "max_tokens": 131072}
        yield assistant_chat.build_stats_event(self.session.remote_usage_stats)
        yield assistant_chat.build_status_event("Waiting at a steering boundary…", kind="queued", session=self.session)
        queued_id, payload = assistant_chat.add_user_message(self.session, "Queued request for Edit, Remove, and Steer controls.", queued=True, client_submission_id="simulated-queued-request")
        yield payload
        yield assistant_chat.set_message_end_badge(self.session, assistant_id, "Interrupted at steering boundary")
        yield assistant_chat.build_status_event(None, visible=False, session=self.session)

        if include_recovery_glitches:
            duplicate = assistant_chat.build_stats_event(self.session.remote_usage_stats)
            yield duplicate
            yield duplicate
            stale = json.loads(assistant_chat.build_sync_event(self.session))
            stale["event_id"] = "simulated-stale-sync"
            stale["event"]["revision"] = max(0, self.session.chat_revision - 5)
            stale["event"]["sequence"] = max(0, self.session.chat_event_sequence - 5)
            stale["event"]["sequence_start"] = stale["event"]["sequence"]
            yield json.dumps(stale, ensure_ascii=False)

            gap_text = "Intentional gap recovery must end with exactly one canonical sentence."
            gap_id, payload = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, None, gap_text[:24], streaming=True)
            if payload:
                yield payload
            gap_id, _intentionally_dropped = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, gap_id, gap_text[:42], streaming=True)
            gap_id, payload = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, gap_id, gap_text, streaming=True)
            if payload:
                yield payload
            gap_id, _finalize_dropped = assistant_chat.upsert_assistant_content_block(self.session, assistant_id, gap_id, gap_text, streaming=False)
            yield self.sync()

        yield self.sync()


def stream_simulation(simulator: DeepyStreamingSimulator, cadence: str, scale: int = 1, include_recovery_glitches: bool = True):
    delay = CADENCE_PRESETS.get(str(cadence), CADENCE_PRESETS["Normal (250 ms)"])
    for payload in simulator.events(scale=max(1, int(scale)), include_recovery_glitches=include_recovery_glitches):
        yield payload
        if delay > 0:
            time.sleep(delay)
