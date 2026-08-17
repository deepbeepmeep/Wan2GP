import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from shared.deepy.config import (
    DEEPY_COMPACTION_TYPE_DISCARD,
    DEEPY_COMPACTION_TYPE_SUMMARIZE,
    normalize_deepy_compaction_type,
    validate_deepy_compaction_config,
)
from shared.deepy.engine import AssistantEngine, AssistantSessionState


class _FakeRuntime:
    def __init__(self):
        self.sequence = SimpleNamespace(token_ids=[9, 8, 7])
        self.config = SimpleNamespace(kvcache_block_size=256)

    def prime_context(self, token_ids):
        self.sequence = SimpleNamespace(token_ids=list(token_ids))
        return self.sequence

    def extend_context(self, token_ids):
        if list(token_ids[: len(self.sequence.token_ids)]) != list(self.sequence.token_ids):
            raise RuntimeError("target does not extend prefix")
        self.sequence.token_ids = list(token_ids)
        return "chunk_prefilled"

    def snapshot_context(self):
        return {"sequence": {"token_ids": list(self.sequence.token_ids)}}

    def restore_snapshot(self, snapshot):
        self.sequence = SimpleNamespace(token_ids=list(snapshot["sequence"]["token_ids"]))

    def _get_active_sequence(self):
        return self.sequence

    def _get_live_llm(self):
        return SimpleNamespace(config=self.config)


class DeepyCompactionConfigTests(unittest.TestCase):
    def test_compaction_type_normalization(self):
        self.assertEqual(normalize_deepy_compaction_type("summarize"), DEEPY_COMPACTION_TYPE_SUMMARIZE)
        self.assertEqual(normalize_deepy_compaction_type("unknown"), DEEPY_COMPACTION_TYPE_DISCARD)

    def test_summarize_requires_32000_tokens(self):
        with self.assertRaisesRegex(ValueError, "32,000"):
            validate_deepy_compaction_config(DEEPY_COMPACTION_TYPE_SUMMARIZE, 31999)
        self.assertEqual(validate_deepy_compaction_config(DEEPY_COMPACTION_TYPE_SUMMARIZE, 32000), DEEPY_COMPACTION_TYPE_SUMMARIZE)
        self.assertEqual(validate_deepy_compaction_config(DEEPY_COMPACTION_TYPE_DISCARD, 8192), DEEPY_COMPACTION_TYPE_DISCARD)


class DeepyCompactionEngineTests(unittest.TestCase):
    def _engine(self):
        engine = AssistantEngine.__new__(AssistantEngine)
        engine.session = AssistantSessionState()
        engine.runtime = _FakeRuntime()
        engine._skip_pause_snapshot = False
        engine._run_prefill_call = lambda _tokens, callback, **_kwargs: callback()
        engine._get_context_window_tokens = lambda: 32000
        engine._current_system_prompt_signature = lambda: "system-signature"
        engine._render_system_prompt_tokens = lambda add_generation_prompt: [1, 2]
        return engine

    def test_compaction_payload_omits_transient_model_content(self):
        payload = AssistantEngine._compaction_history_payload([
            {"role": "user", "content": "visible", "model_content": "transient runtime state"},
            {"role": "assistant", "content": "result", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "content": "done", "tool_call_id": "call_1"},
        ])
        self.assertIn("visible", payload)
        self.assertNotIn("transient runtime state", payload)
        self.assertIn("call_1", payload)

    def test_last_resort_discards_tool_call_and_results_as_one_step(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "current request"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "first", "arguments": {}}},
                    {"id": "call_2", "type": "function", "function": {"name": "second", "arguments": {}}},
                ],
            },
            {"role": "tool", "content": "first result", "tool_call_id": "call_1"},
            {"role": "tool", "content": "second result", "tool_call_id": "call_2"},
            {"role": "assistant", "content": "later current-turn message"},
        ]

        reason = engine._discard_oldest_current_turn_step()

        self.assertEqual(reason, "dropped earlier current-turn assistant step (3 messages)")
        self.assertEqual(engine.session.messages, [
            {"role": "user", "content": "current request"},
            {"role": "assistant", "content": "later current-turn message"},
        ])

    def test_successful_commit_replaces_turn_snapshot_with_system_and_summary(self):
        engine = self._engine()
        prior_messages = AssistantEngine._build_compacted_summary_messages("The user selected image_4 and requested a video.")
        current_messages = [{"role": "user", "content": "Continue now."}]
        engine.session.messages = [{"role": "user", "content": "old"}, *current_messages]
        engine.session.current_turn = {"runtime_snapshot": {"old": True}, "messages_len": 1, "committed_messages_len": 2}

        def render_messages(add_generation_prompt):
            if len(engine.session.messages) == len(prior_messages):
                return [1, 2, 3, 4]
            return [1, 2, 3, 4, 5, 6]

        engine._render_messages = render_messages
        engine._render_current_turn_slice_suffix = lambda messages, add_generation_prompt: [5, 6]
        engine._commit_rewritten_history(prior_messages, current_messages, generation_reserve_tokens=1024)

        checkpoint = engine.session.current_turn
        self.assertEqual(checkpoint["runtime_snapshot"]["sequence"]["token_ids"], [1, 2, 3, 4])
        self.assertEqual(checkpoint["rendered_token_ids"], [1, 2, 3, 4])
        self.assertEqual(checkpoint["messages_len"], len(prior_messages))
        self.assertEqual(engine.session.rendered_token_ids, [1, 2, 3, 4, 5, 6])
        self.assertIsNone(engine.session.runtime_snapshot)
        self.assertEqual(engine.runtime.sequence.token_ids, [1, 2, 3, 4, 5, 6])

    def test_summary_starts_at_75_percent_and_keeps_current_turn(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2"}
        engine.session.rendered_token_ids = [9, 8, 7]
        engine.session.rendered_messages_len = 2
        engine.session.rendered_system_prompt_signature = "system-signature"
        engine.session.rendered_context_window_tokens = 32000
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(24000))
        engine._render_compaction_prompt = lambda messages: list(range(1000))
        engine._restore_or_replay_session = lambda _label: "reused"
        engine._set_status = Mock()
        engine._record_generation_metrics = Mock()
        engine._mark_history_summarized_trace = Mock()
        engine._commit_rewritten_history = Mock()
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="A compact factual summary.", stop_reason="stop_token", token_count=5))

        with patch("builtins.print") as print_mock:
            self.assertTrue(engine._maybe_summarize_context(generation_reserve_tokens=1024))
        compacted_prior, current_turn, reserve = engine._commit_rewritten_history.call_args.args
        self.assertIn("A compact factual summary.", compacted_prior[0]["content"])
        self.assertEqual(current_turn, [{"role": "user", "content": "current request"}])
        self.assertEqual(reserve, 1024)
        self.assertTrue(engine.session.current_turn["summary_compaction_attempted"])
        print_mock.assert_any_call("[Deepy] Context compacted: summarize, 24,000 -> 3 tokens, 1 completed turn summarized")

    def test_failed_summary_restores_snapshot_and_uses_discard_compaction(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2"}
        engine.session.rendered_token_ids = [9, 8, 7]
        engine.session.rendered_messages_len = 2
        engine.session.rendered_system_prompt_signature = "system-signature"
        engine.session.rendered_context_window_tokens = 32000
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(24000))
        engine._render_compaction_prompt = lambda messages: list(range(1000))
        engine._restore_or_replay_session = lambda _label: "reused"
        engine._set_status = Mock()
        engine._record_generation_metrics = Mock()
        engine._mark_summary_fallback_trace = Mock()
        engine._mark_history_trimmed_trace = Mock()
        engine._discard_prior_messages_to_trigger = Mock(return_value=[])
        engine._commit_rewritten_history = Mock()
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="unfinished", stop_reason="max_tokens", token_count=2048))

        with patch("builtins.print") as print_mock:
            self.assertTrue(engine._maybe_summarize_context(generation_reserve_tokens=1024))
        engine._mark_summary_fallback_trace.assert_called_once()
        engine._discard_prior_messages_to_trigger.assert_called_once()
        engine._commit_rewritten_history.assert_called_once_with([], [{"role": "user", "content": "current request"}], 1024)
        engine._mark_history_trimmed_trace.assert_called_once()
        self.assertEqual(engine.runtime.sequence.token_ids, [9, 8, 7])
        print_mock.assert_any_call("[Deepy] Context compacted: summarize failed; discard fallback, 24,000 -> 3 tokens, 1 oldest turn removed")


if __name__ == "__main__":
    unittest.main()
