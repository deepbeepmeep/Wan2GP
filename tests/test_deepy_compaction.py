import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from shared.llm_engines.nanovllm import SamplingParams
from shared.llm_engines.nanovllm.engine.sequence import Sequence
from shared.deepy import DEFAULT_COMPACTION_PROMPT
from shared.deepy.config import (
    DEEPY_COMPACTION_TYPE_DISCARD,
    DEEPY_COMPACTION_TYPE_SUMMARIZE,
    DEEPY_TYPE_PRIME,
    DEEPY_TYPE_DISABLED,
    DEEPY_TYPE_ZERO,
    DEEPY_KV_CACHE_QUANTIZATION_AUTO,
    deepy_mode_from_config,
    deepy_requirement_error,
    deepy_requirement_met,
    normalize_deepy_compaction_type,
    normalize_deepy_kv_cache_quantization,
    resolve_deepy_kv_cache_quantization,
    split_deepy_mode,
    validate_deepy_compaction_config,
    validate_deepy_version_config,
)
from shared.prompt_enhancer.config import PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO, normalize_prompt_enhancer_speculative_decoding, resolve_prompt_enhancer_speculative_decoding
from shared.deepy.engine import AssistantEngine, AssistantSessionState, _summarize_interrupted_committed_messages, _summary_compaction_trigger_tokens, begin_assistant_turn, build_interruption_notice, checkpoint_assistant_turn, rollback_assistant_turn
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.gradio import assistant_chat


class _FakeRuntime:
    def __init__(self):
        self.sequence = SimpleNamespace(token_ids=[9, 8, 7])
        self.config = SimpleNamespace(kvcache_block_size=256)
        self.prime_calls = 0

    def prime_context(self, token_ids):
        self.prime_calls += 1
        self.sequence = SimpleNamespace(token_ids=list(token_ids))
        return self.sequence

    def append_suffix(self, token_ids):
        self.sequence.token_ids.extend(list(token_ids))
        return "chunk_prefilled"

    def extend_context(self, token_ids):
        if list(token_ids[: len(self.sequence.token_ids)]) != list(self.sequence.token_ids):
            raise RuntimeError("target does not extend prefix")
        self.sequence.token_ids = list(token_ids)
        return "chunk_prefilled"

    def snapshot_context(self):
        return {"sequence": {"token_ids": list(self.sequence.token_ids)}}

    def restore_snapshot(self, snapshot):
        self.sequence = SimpleNamespace(token_ids=list(snapshot["sequence"]["token_ids"]))

    def restore_or_replay(self, snapshot, token_ids):
        if snapshot is not None:
            self.restore_snapshot(snapshot)
            return "restored", "exact snapshot restored"
        self.prime_context(token_ids)
        return "prefilled", "no exact runtime snapshot was available"

    def _get_active_sequence(self):
        return self.sequence

    def _get_live_llm(self):
        return SimpleNamespace(config=self.config)


class _InitiallyUnloadedFakeRuntime(_FakeRuntime):
    def __init__(self):
        super().__init__()
        self.sequence = None
        self.restore_calls = 0

    def snapshot_context(self):
        return None if self.sequence is None else {"sequence": {"token_ids": list(self.sequence.token_ids)}}

    def restore_snapshot(self, snapshot):
        self.restore_calls += 1
        super().restore_snapshot(snapshot)

    def _get_active_sequence(self):
        return self.sequence

    def _get_live_llm(self):
        if self.sequence is None:
            raise RuntimeError("Assistant runtime is not initialized.")
        return super()._get_live_llm()

    def restore_or_replay(self, snapshot, token_ids):
        if snapshot is not None:
            self.restore_snapshot(snapshot)
            return "restored", "exact snapshot restored"
        self.prime_context(token_ids)
        return "prefilled", "no exact runtime snapshot was available"


class DeepyCompactionConfigTests(unittest.TestCase):
    def test_combined_deepy_mode_maps_to_existing_config_keys(self):
        self.assertEqual(deepy_mode_from_config(0, DEEPY_TYPE_PRIME), DEEPY_TYPE_DISABLED)
        self.assertEqual(deepy_mode_from_config(1, DEEPY_TYPE_ZERO), DEEPY_TYPE_ZERO)
        self.assertEqual(deepy_mode_from_config(1, DEEPY_TYPE_PRIME), DEEPY_TYPE_PRIME)
        self.assertEqual(split_deepy_mode(DEEPY_TYPE_DISABLED), (0, DEEPY_TYPE_ZERO))
        self.assertEqual(split_deepy_mode(DEEPY_TYPE_ZERO), (1, DEEPY_TYPE_ZERO))
        self.assertEqual(split_deepy_mode(DEEPY_TYPE_PRIME), (1, DEEPY_TYPE_PRIME))

    def test_speculative_decoding_auto_uses_model_vram_thresholds(self):
        self.assertEqual(normalize_prompt_enhancer_speculative_decoding("auto"), PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO)
        self.assertEqual(resolve_prompt_enhancer_speculative_decoding(3, PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO, 48)[0], 0)
        self.assertEqual(resolve_prompt_enhancer_speculative_decoding(4, PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO, 11)[0], 0)
        self.assertEqual(resolve_prompt_enhancer_speculative_decoding(4, PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO, 12)[0], 1)
        self.assertEqual(resolve_prompt_enhancer_speculative_decoding(5, PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO, 23)[0], 0)
        self.assertEqual(resolve_prompt_enhancer_speculative_decoding(5, PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO, 24)[0], 1)
        self.assertEqual(resolve_prompt_enhancer_speculative_decoding(4, 1, 8), (1, ""))

    def test_kv_cache_auto_requires_gguf_kernels_1_0_11(self):
        self.assertEqual(normalize_deepy_kv_cache_quantization(None), DEEPY_KV_CACHE_QUANTIZATION_AUTO)
        self.assertEqual(normalize_deepy_kv_cache_quantization(""), "")
        with patch("shared.deepy.config.get_gguf_kernels_version", return_value="1.0.10"):
            self.assertEqual(resolve_deepy_kv_cache_quantization(DEEPY_KV_CACHE_QUANTIZATION_AUTO)[0], "")
        with patch("shared.deepy.config.get_gguf_kernels_version", return_value="1.0.11+torch210"):
            mode, message = resolve_deepy_kv_cache_quantization(DEEPY_KV_CACHE_QUANTIZATION_AUTO)
        self.assertEqual(mode, "int8")
        self.assertIn("1.0.11+torch210", message)

    def test_compaction_type_normalization(self):
        self.assertEqual(normalize_deepy_compaction_type("summarize"), DEEPY_COMPACTION_TYPE_SUMMARIZE)
        self.assertEqual(normalize_deepy_compaction_type("unknown"), DEEPY_COMPACTION_TYPE_DISCARD)

    def test_summarize_requires_32000_tokens(self):
        with self.assertRaisesRegex(ValueError, "32,000"):
            validate_deepy_compaction_config(DEEPY_COMPACTION_TYPE_SUMMARIZE, 31999)
        self.assertEqual(validate_deepy_compaction_config(DEEPY_COMPACTION_TYPE_SUMMARIZE, 32000), DEEPY_COMPACTION_TYPE_SUMMARIZE)
        self.assertEqual(validate_deepy_compaction_config(DEEPY_COMPACTION_TYPE_DISCARD, 8192), DEEPY_COMPACTION_TYPE_DISCARD)

    def test_summary_trigger_uses_85_percent_or_buffer_limit(self):
        self.assertEqual(_summary_compaction_trigger_tokens(32000), 27200)
        self.assertEqual(_summary_compaction_trigger_tokens(16000), 11904)

    def test_deepy_prime_requires_summarize_and_32000_tokens(self):
        self.assertEqual(validate_deepy_version_config(DEEPY_TYPE_PRIME, DEEPY_COMPACTION_TYPE_SUMMARIZE, 32000, 5), (DEEPY_TYPE_PRIME, DEEPY_COMPACTION_TYPE_SUMMARIZE, 32000))
        with self.assertRaisesRegex(ValueError, "Qwen3.8 VL 27B"):
            validate_deepy_version_config(DEEPY_TYPE_PRIME, DEEPY_COMPACTION_TYPE_SUMMARIZE, 32000, 4)
        with self.assertRaisesRegex(ValueError, "Deepy Prime requires Summarize"):
            validate_deepy_version_config(DEEPY_TYPE_PRIME, DEEPY_COMPACTION_TYPE_DISCARD, 46080, 5)
        with self.assertRaisesRegex(ValueError, "32,000"):
            validate_deepy_version_config(DEEPY_TYPE_PRIME, DEEPY_COMPACTION_TYPE_SUMMARIZE, 31999, 5)
        self.assertEqual(validate_deepy_version_config(DEEPY_TYPE_ZERO, DEEPY_COMPACTION_TYPE_DISCARD, 8192, 3), (DEEPY_TYPE_ZERO, DEEPY_COMPACTION_TYPE_DISCARD, 8192))

    def test_invalid_prime_configuration_is_unavailable_at_runtime(self):
        valid = {"enhancer_enabled": 5, "deepy_type": DEEPY_TYPE_PRIME, "deepy_compaction_type": DEEPY_COMPACTION_TYPE_SUMMARIZE, "deepy_context_tokens": 32000}
        self.assertTrue(deepy_requirement_met(valid))
        invalid = {**valid, "deepy_compaction_type": DEEPY_COMPACTION_TYPE_DISCARD}
        self.assertFalse(deepy_requirement_met(invalid))
        self.assertIn("Deepy Prime requires Summarize", deepy_requirement_error(invalid))
        wrong_model = {**valid, "enhancer_enabled": 4}
        self.assertFalse(deepy_requirement_met(wrong_model))
        self.assertIn("Qwen3.8 VL 27B", deepy_requirement_error(wrong_model))


class DeepyCompactionEngineTests(unittest.TestCase):
    def _engine(self):
        engine = AssistantEngine.__new__(AssistantEngine)
        engine.session = AssistantSessionState()
        engine.runtime = _FakeRuntime()
        engine.vram_mode = "unload"
        engine.debug_enabled = False
        engine._skip_pause_snapshot = False
        engine._skip_generation_context_sync_once = False
        engine._resume_stream_after_context_trim = False
        engine._suppress_intermediate_stream_after_context_trim = False
        engine._continued_segment_raw_text = ""
        engine._continued_segment_token_ids = []
        engine._run_prefill_call = lambda _tokens, callback, **_kwargs: callback()
        engine._get_context_window_tokens = lambda: 32000
        engine._current_system_prompt_signature = lambda: "system-signature"
        engine._current_reset_base_signature = lambda: "system-signature"
        engine._render_system_prompt_tokens = lambda add_generation_prompt: [1, 2]
        return engine

    def test_generation_reserve_uses_runtime_thinking_budget(self):
        engine = self._engine()
        engine.thinking_enabled = True
        engine._current_requested_max_new_tokens = 1024
        engine.runtime._runtime_extra_tokens = 3000
        self.assertEqual(engine._segment_generation_reserve_tokens(), 4024)

    def test_live_max_token_continuation_preserves_completion_boundary(self):
        runtime = Qwen35AssistantRuntime.__new__(Qwen35AssistantRuntime)
        sequence = Sequence([10, 11, 12], SamplingParams(max_tokens=8))
        sequence.num_prompt_tokens = 2
        runner = SimpleNamespace(call=Mock())
        runtime._get_active_sequence = lambda: sequence
        runtime._get_live_llm = lambda: SimpleNamespace(config=SimpleNamespace(max_model_len=100), model_runner=runner)
        runtime._build_sampling_params = lambda **_kwargs: (SamplingParams(max_tokens=8), {"effective_new_tokens": 8, "requested_new_tokens": 8, "effective_runtime_extra": 0, "requested_runtime_extra": 0, "available_tokens": 97})
        runtime._log = lambda _message: None

        resumed, segment_tokens = runtime.start_generation_segment(max_new_tokens=8, seed=0, do_sample=False, temperature=None, top_p=None, top_k=None, thinking_enabled=True, continue_existing_completion=True)

        self.assertIs(resumed, sequence)
        self.assertEqual(segment_tokens, 8)
        self.assertEqual(sequence.num_prompt_tokens, 2)
        self.assertEqual(sequence.num_completion_tokens, 1)
        self.assertEqual(sequence.max_tokens, 9)

    def test_loop_detection_includes_tool_actions(self):
        recent_steps = []
        calls = lambda path: [{"name": "wangp_query_file", "arguments": {"path": path}}]

        self.assertEqual(AssistantEngine._record_loop_step(recent_steps, "Let's query the remaining files.", calls("a.png")), "")
        self.assertEqual(AssistantEngine._record_loop_step(recent_steps, "Let's query the remaining files.", calls("b.png")), "")
        self.assertEqual(AssistantEngine._record_loop_step(recent_steps, "Let's query the remaining files.", calls("c.png")), "")
        self.assertEqual(AssistantEngine._record_loop_step(recent_steps, "Let's query the remaining files.", calls("d.png")), "")

        recent_steps.clear()
        self.assertEqual(AssistantEngine._record_loop_step(recent_steps, "Retrying.", calls("same.png")), "")
        self.assertEqual(AssistantEngine._record_loop_step(recent_steps, "Retrying.", calls("same.png")), "")
        self.assertIn("repeated 3 times", AssistantEngine._record_loop_step(recent_steps, "Retrying.", calls("same.png")))

    def test_compaction_memory_preserves_findings_and_prevents_repeated_work(self):
        self.assertIn("every important finding", DEFAULT_COMPACTION_PROMPT)
        self.assertIn("not to be repeated", DEFAULT_COMPACTION_PROMPT)
        messages = AssistantEngine._build_compacted_summary_messages("Completed work: generated visual:abc.")
        self.assertIn("authoritative working state", messages[0]["content"])
        self.assertIn("do not repeat completed actions", messages[0]["content"])
        self.assertIn("avoid repeating completed work", messages[1]["content"])

    def test_compaction_reuses_existing_kv_and_appends_only_instruction(self):
        engine = self._engine()
        engine.runtime.sequence.token_ids = [1, 2, 3]
        engine.runtime.tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [7, 8])
        engine.runtime.model = SimpleNamespace()
        engine.thinking_enabled = True
        engine.tool_box = SimpleNamespace(get_tool_schemas=lambda: [])
        engine._build_system_prompt = lambda log_injections=False: "system"
        checkpoint = {
            "messages_len": 2,
            "rendered_messages_len": 2,
            "rendered_token_ids": [1, 2, 3],
            "runtime_snapshot": {"sequence": {"token_ids": [1, 2, 3]}},
            "rendered_system_prompt_signature": "system-signature",
            "rendered_context_window_tokens": 32000,
        }
        prior_messages = [{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old answer"}]

        rollback_snapshot, context_tokens = engine._prepare_memory_compaction_context(prior_messages, checkpoint, 30000)

        self.assertEqual(rollback_snapshot["sequence"]["token_ids"], [1, 2, 3])
        self.assertEqual(context_tokens, 5)
        self.assertEqual(engine.runtime.sequence.token_ids, [1, 2, 3, 7, 8])

    def test_generation_sync_prefers_turn_start_before_header_replay(self):
        engine = self._engine()
        engine.session.messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "interrupted"}, {"role": "user", "content": "second"}]
        engine.session.rendered_token_ids = [1, 2]
        engine.session.rendered_messages_len = 2
        engine.session.runtime_snapshot = None
        engine.runtime._get_active_sequence = lambda: None
        engine._acquire_runtime = lambda: engine.runtime
        engine._get_context_window_tokens = lambda: 46080
        engine._segment_generation_reserve_tokens = lambda: 128
        engine._maybe_summarize_context = lambda _reserve: False
        engine._fit_rendered_messages_to_window = lambda **_kwargs: ([1, 2, 3], False)
        engine._append_target_suffix_from_live_runtime = Mock(return_value=None)
        engine._extend_context_from_preserved_base = Mock(return_value="chunk_prefilled")
        engine._sync_current_turn_context_from_turn_start_snapshot = Mock(return_value=False)
        engine._restore_or_replay_session = Mock()
        engine._remember_render_state = Mock()
        engine._log = Mock()

        engine._sync_generation_context()

        engine._sync_current_turn_context_from_turn_start_snapshot.assert_called_once_with(target_tokens=[1, 2, 3])
        engine._extend_context_from_preserved_base.assert_called_once_with([1, 2, 3])
        engine._restore_or_replay_session.assert_not_called()

    def test_preserved_base_prefill_metrics_count_only_appended_suffix(self):
        engine = self._engine()
        engine.session.reset_base_token_ids = [1, 2]
        engine.session.reset_base_snapshot = {"sequence": {"token_ids": [1, 2]}}
        engine.session.reset_base_signature = "system-signature"
        engine.session.reset_base_context_window_tokens = 32000
        prefill_counts = []

        def run_prefill(token_count, callback, **_kwargs):
            prefill_counts.append(token_count)
            return callback()

        engine._run_prefill_call = run_prefill

        self.assertEqual(engine._extend_context_from_preserved_base([1, 2, 3, 4]), "chunk_prefilled")
        self.assertEqual(prefill_counts, [2, 2])
        self.assertEqual(engine.runtime.sequence.token_ids, [1, 2, 3, 4])

    def test_interrupted_tool_stop_restores_action_snapshot_and_appends_only_delta(self):
        engine = self._engine()
        engine.runtime = _InitiallyUnloadedFakeRuntime()
        encoded_suffixes = []

        def encode(text, **_kwargs):
            encoded_suffixes.append(text)
            return [100 + len(encoded_suffixes)]

        engine.runtime.tokenizer = SimpleNamespace(encode=encode)
        engine.runtime.model = SimpleNamespace()
        engine.thinking_enabled = True
        notice = build_interruption_notice("generate the video")
        engine.session.messages = [
            {"role": "user", "content": "generate the video"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "wangp_generate", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"interrupted"}'},
            {"role": "assistant", "content": notice},
            {"role": "user", "content": "what were you doing?"},
        ]
        action_tokens = list(range(32596))
        engine.session.rendered_token_ids = action_tokens
        engine.session.rendered_messages_len = 2
        engine.session.runtime_snapshot = {"sequence": {"token_ids": action_tokens}}
        engine.session.rendered_system_prompt_signature = "system-signature"
        engine.session.rendered_context_window_tokens = 46080
        engine.session.interruption_notice = notice
        engine._acquire_runtime = lambda: engine.runtime
        engine._get_context_window_tokens = lambda: 46080
        engine._segment_generation_reserve_tokens = lambda: 128
        engine._maybe_summarize_context = lambda _reserve: False
        engine._fit_rendered_messages_to_window = Mock(side_effect=AssertionError("full canonical render must not run"))
        engine._extend_context_from_preserved_base = Mock(side_effect=AssertionError("header replay must not run"))
        engine._sync_current_turn_context_from_turn_start_snapshot = Mock(side_effect=AssertionError("turn-start replay must not run"))
        engine._emit_stats = Mock()
        engine._log = Mock()

        engine._sync_generation_context()

        self.assertEqual(engine.runtime.restore_calls, 1)
        self.assertEqual(engine.runtime.sequence.token_ids, [*action_tokens, 101, 102, 103])
        self.assertEqual(engine.session.rendered_messages_len, 5)
        self.assertEqual(engine.session.runtime_snapshot["sequence"]["token_ids"], [*action_tokens, 101, 102, 103])
        engine._extend_context_from_preserved_base.assert_not_called()
        self.assertIn("<tool_response>", encoded_suffixes[0])
        self.assertIn("previous user request was interrupted", encoded_suffixes[1])
        self.assertIn("what were you doing?", encoded_suffixes[2])

    def test_recorded_action_snapshot_survives_incomplete_thought(self):
        engine = self._engine()
        engine.session.messages = [{"role": "user", "content": "work"}]
        engine.runtime.sequence.token_ids = [1, 2, 3]
        engine._emit_stats = Mock()
        engine._log = Mock()

        engine._record_live_context("recorded")
        action_snapshot = engine.session.runtime_snapshot
        engine.runtime.sequence.token_ids.extend([4, 5, 6])

        self.assertTrue(engine._snapshot_synchronized_live_context())
        self.assertIs(engine.session.runtime_snapshot, action_snapshot)
        self.assertEqual(engine.session.runtime_snapshot["sequence"]["token_ids"], [1, 2, 3])

    def test_interrupted_resume_suffix_handles_all_safe_boundaries(self):
        notice_1 = build_interruption_notice("first request")
        notice_2 = build_interruption_notice("second request")
        cases = [
            (
                "first-pass stop",
                [{"role": "assistant", "content": "older answer"}, {"role": "user", "content": "first request"}, {"role": "assistant", "content": notice_1}, {"role": "user", "content": "new request"}],
                1,
                3,
            ),
            (
                "mid-thought stop",
                [{"role": "user", "content": "first request"}, {"role": "assistant", "content": notice_1}, {"role": "user", "content": "new request"}],
                1,
                2,
            ),
            (
                "completed multi-tool stop",
                [
                    {"role": "user", "content": "first request"},
                    {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "a", "arguments": {}}}, {"id": "b", "type": "function", "function": {"name": "b", "arguments": {}}}]},
                    {"role": "tool", "tool_call_id": "a", "content": '{"status":"done","tool":"a"}'},
                    {"role": "tool", "tool_call_id": "b", "content": '{"status":"interrupted","tool":"b"}'},
                    {"role": "assistant", "content": notice_1},
                    {"role": "user", "content": "new request"},
                ],
                2,
                3,
            ),
            (
                "repeated stops",
                [{"role": "user", "content": "first request"}, {"role": "assistant", "content": notice_1}, {"role": "user", "content": "second request"}, {"role": "assistant", "content": notice_2}, {"role": "user", "content": "third request"}],
                1,
                4,
            ),
        ]
        for label, messages, rendered_messages_len, expected_encoded_parts in cases:
            with self.subTest(label=label):
                engine = self._engine()
                encoded_suffixes = []
                engine.runtime.tokenizer = SimpleNamespace(encode=lambda text, **_kwargs: encoded_suffixes.append(text) or [100 + len(encoded_suffixes)])
                engine.runtime.model = SimpleNamespace()
                engine.runtime.sequence.token_ids = [1]
                engine.thinking_enabled = True
                engine.session.messages = messages
                engine.session.rendered_messages_len = rendered_messages_len
                engine.session.rendered_token_ids = [1]
                engine.session.rendered_system_prompt_signature = "system-signature"
                engine.session.rendered_context_window_tokens = 32000
                engine.session.interruption_notice = notice_2 if label == "repeated stops" else notice_1

                self.assertEqual(engine._append_interrupted_resume_suffix(), "chunk_prefilled")
                self.assertEqual(len(encoded_suffixes), expected_encoded_parts)
                self.assertEqual(len(engine.runtime.sequence.token_ids), 1 + expected_encoded_parts)
                if label == "completed multi-tool stop":
                    self.assertEqual(encoded_suffixes[0].count("<tool_response>"), 2)

    def test_interrupted_resume_suffix_rejects_incomplete_tool_group(self):
        engine = self._engine()
        notice = build_interruption_notice("first request")
        engine.runtime.tokenizer = SimpleNamespace(encode=lambda _text, **_kwargs: [100])
        engine.runtime.model = SimpleNamespace()
        engine.thinking_enabled = True
        engine.session.messages = [
            {"role": "user", "content": "first request"},
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "a", "arguments": {}}}]},
            {"role": "assistant", "content": notice},
            {"role": "user", "content": "new request"},
        ]
        engine.session.rendered_messages_len = 2
        engine.session.rendered_token_ids = [1]
        engine.session.rendered_system_prompt_signature = "system-signature"
        engine.session.rendered_context_window_tokens = 32000
        engine.session.interruption_notice = notice
        engine.runtime.sequence.token_ids = [1]

        self.assertIsNone(engine._append_interrupted_resume_suffix())
        self.assertEqual(engine.runtime.sequence.token_ids, [1])

    def test_summarize_mode_appends_interrupted_delta_before_near_limit_compaction(self):
        notice = build_interruption_notice("first request")

        def make_engine(compaction_type):
            engine = self._engine()
            engine.runtime.tokenizer = SimpleNamespace(encode=lambda _text, **_kwargs: [100])
            engine.runtime.model = SimpleNamespace()
            engine.runtime.sequence.token_ids = [1]
            engine.thinking_enabled = True
            engine.session.messages = [{"role": "user", "content": "first request"}, {"role": "assistant", "content": notice}, {"role": "user", "content": "new request"}]
            engine.session.rendered_messages_len = 1
            engine.session.rendered_token_ids = [1]
            engine.session.rendered_system_prompt_signature = "system-signature"
            engine.session.rendered_context_window_tokens = 3
            engine.session.interruption_notice = notice
            engine._get_context_window_tokens = lambda: 3
            engine._get_compaction_type = lambda: compaction_type
            engine._log = Mock()
            return engine

        summarize_engine = make_engine(DEEPY_COMPACTION_TYPE_SUMMARIZE)
        self.assertEqual(summarize_engine._append_interrupted_resume_suffix(generation_reserve_tokens=2), "chunk_prefilled")
        self.assertEqual(summarize_engine.runtime.sequence.token_ids, [1, 100, 100])

        discard_engine = make_engine(DEEPY_COMPACTION_TYPE_DISCARD)
        self.assertIsNone(discard_engine._append_interrupted_resume_suffix(generation_reserve_tokens=2))
        self.assertEqual(discard_engine.runtime.sequence.token_ids, [1])

    def test_interrupted_resume_rejects_changed_tool_or_thinking_signature(self):
        engine = self._engine()
        notice = build_interruption_notice("first request")
        engine.runtime.tokenizer = SimpleNamespace(encode=lambda _text, **_kwargs: [100])
        engine.runtime.model = SimpleNamespace()
        engine.thinking_enabled = True
        engine.session.messages = [{"role": "user", "content": "first request"}, {"role": "assistant", "content": notice}, {"role": "user", "content": "new request"}]
        engine.session.rendered_messages_len = 1
        engine.session.rendered_token_ids = [1]
        engine.session.rendered_system_prompt_signature = "old-tool-schema"
        engine.session.rendered_context_window_tokens = 32000
        engine.session.interruption_notice = notice
        engine.runtime.sequence.token_ids = [1]

        self.assertIsNone(engine._append_interrupted_resume_suffix())
        self.assertEqual(engine.runtime.sequence.token_ids, [1])

    def test_interrupted_multi_tool_group_keeps_prior_actions_and_uses_action_delta(self):
        session = AssistantSessionState(
            messages=[{"role": "user", "content": "older"}, {"role": "assistant", "content": "older answer"}],
            rendered_token_ids=[9],
            rendered_messages_len=2,
            runtime_snapshot={"sequence": {"token_ids": [9]}},
            rendered_system_prompt_signature="system-signature",
            rendered_context_window_tokens=32000,
        )
        begin_assistant_turn(session, "user_current", "do several actions")
        session.messages.extend([
            {"role": "user", "content": "do several actions"},
            {"role": "assistant", "tool_calls": [{"id": "prior", "type": "function", "function": {"name": "prior_tool", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "prior", "content": '{"status":"done"}'},
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "tool_a", "arguments": {}}}, {"id": "b", "type": "function", "function": {"name": "tool_b", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "a", "content": '{"status":"done"}'},
            {"role": "tool", "tool_call_id": "b", "content": '{"status":"interrupted","cancelled":true}'},
        ])
        checkpoint_assistant_turn(session)
        action_tokens = [1, 2, 3]
        session.rendered_token_ids = action_tokens
        session.rendered_messages_len = 6
        session.runtime_snapshot = {"sequence": {"token_ids": action_tokens}}

        self.assertTrue(rollback_assistant_turn(session))
        retained_calls = [call["function"]["name"] for message in session.messages for call in message.get("tool_calls", [])]
        self.assertEqual(retained_calls, ["prior_tool", "tool_a", "tool_b"])
        session.messages.append({"role": "user", "content": "what happened?"})

        engine = self._engine()
        engine.session = session
        engine.runtime = _InitiallyUnloadedFakeRuntime()
        engine.runtime.tokenizer = SimpleNamespace(encode=lambda text, **_kwargs: [100 + text.count("<tool_response>")])
        engine.runtime.model = SimpleNamespace()
        engine.thinking_enabled = True
        engine._acquire_runtime = lambda: engine.runtime
        engine._segment_generation_reserve_tokens = lambda: 128
        engine._maybe_summarize_context = lambda _reserve: False
        engine._fit_rendered_messages_to_window = Mock(side_effect=AssertionError("full canonical render must not run"))
        engine._extend_context_from_preserved_base = Mock(side_effect=AssertionError("header replay must not run"))
        engine._sync_current_turn_context_from_turn_start_snapshot = Mock(side_effect=AssertionError("turn-start replay must not run"))
        engine._emit_stats = Mock()
        engine._log = Mock()

        engine._sync_generation_context()

        self.assertEqual(engine.runtime.restore_calls, 1)
        self.assertEqual(engine.runtime.sequence.token_ids, [1, 2, 3, 102, 100, 100])
        engine._extend_context_from_preserved_base.assert_not_called()

    def test_interrupted_pause_snapshots_exact_synchronized_action_state(self):
        engine = self._engine()
        engine.session.rendered_token_ids = [1, 2, 3]
        engine.session.runtime_snapshot = None
        engine.runtime.sequence = SimpleNamespace(token_ids=[1, 2, 3])
        engine.runtime._get_active_sequence = lambda: engine.runtime.sequence
        engine.runtime.snapshot_context = Mock(return_value={"sequence": {"token_ids": [1, 2, 3]}})

        self.assertTrue(engine._snapshot_synchronized_live_context())
        self.assertEqual(engine.session.runtime_snapshot["sequence"]["token_ids"], [1, 2, 3])
        engine.runtime.snapshot_context.assert_called_once_with()

    def test_interrupted_pause_does_not_snapshot_incomplete_live_suffix(self):
        engine = self._engine()
        engine.session.rendered_token_ids = [1, 2, 3]
        engine.session.runtime_snapshot = None
        engine.runtime.sequence = SimpleNamespace(token_ids=[1, 2, 3, 4])
        engine.runtime._get_active_sequence = lambda: engine.runtime.sequence
        engine.runtime.snapshot_context = Mock()

        self.assertFalse(engine._snapshot_synchronized_live_context())
        self.assertIn("incomplete suffix", engine.session.pending_replay_reason)
        engine.runtime.snapshot_context.assert_not_called()
        self.assertEqual(engine.runtime.prime_calls, 0)

    def test_compaction_refuses_full_history_replay_without_exact_kv(self):
        engine = self._engine()
        engine.runtime.sequence.token_ids = [9]
        engine.runtime.tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [7, 8])
        engine.runtime.model = SimpleNamespace()
        engine.thinking_enabled = True
        engine.tool_box = SimpleNamespace(get_tool_schemas=lambda: [])
        engine._build_system_prompt = lambda log_injections=False: "system"
        checkpoint = {
            "messages_len": 2,
            "rendered_messages_len": 2,
            "rendered_token_ids": [1, 2, 3],
            "runtime_snapshot": None,
            "rendered_system_prompt_signature": "system-signature",
            "rendered_context_window_tokens": 32000,
        }
        prior_messages = [{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old answer"}]

        with self.assertRaisesRegex(RuntimeError, "refusing to replay the full history"):
            engine._prepare_memory_compaction_context(prior_messages, checkpoint, 30000)

        self.assertEqual(engine.runtime.sequence.token_ids, [9])
        self.assertEqual(engine.runtime.prime_calls, 0)

    def test_active_turn_compaction_appends_instruction_to_exact_live_kv(self):
        engine = self._engine()
        engine.session.messages = [{"role": "user", "content": "request"}, {"role": "assistant", "content": "completed action"}]
        engine.session.rendered_token_ids = [1, 2, 3]
        engine.session.rendered_messages_len = 2
        engine.runtime.sequence.token_ids = [1, 2, 3]
        engine.runtime.tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [7, 8])

        rollback_snapshot, context_tokens = engine._prepare_live_compaction_context(32000)

        self.assertEqual(rollback_snapshot["sequence"]["token_ids"], [1, 2, 3])
        self.assertEqual(context_tokens, 5)
        self.assertEqual(engine.runtime.sequence.token_ids, [1, 2, 3, 7, 8])
        self.assertEqual(engine.runtime.prime_calls, 0)

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

    def test_summary_fallback_trims_old_completed_steps_before_dropping_turn(self):
        engine = self._engine()
        engine._log = Mock()
        prior_messages = [
            {"role": "user", "content": "create a long video"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "discover", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "large discovery result"},
            {"role": "assistant", "content": "old reasoning"},
            {"role": "assistant", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "generate", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "latest generated media"},
            {"role": "assistant", "content": "The video is ready."},
        ]
        current_messages = [{"role": "user", "content": "make it funnier"}]
        engine._render_messages = lambda add_generation_prompt: list(range(len(engine.session.messages) * 10))

        remaining, discarded_turns, discarded_steps = engine._discard_prior_messages_to_trigger(prior_messages, current_messages, trigger_tokens=60)

        self.assertEqual(discarded_turns, 0)
        self.assertEqual(discarded_steps, 2)
        self.assertEqual(remaining, [
            {"role": "user", "content": "create a long video"},
            {"role": "assistant", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "generate", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "latest generated media"},
            {"role": "assistant", "content": "The video is ready."},
        ])

    def test_full_turn_deletion_retains_last_five_user_requests_without_answers(self):
        engine = self._engine()
        engine.session.messages = []
        for turn_no in range(1, 8):
            request = f"request {turn_no} " + ("x" * 300 if turn_no == 7 else "details")
            engine.session.messages.extend([{"role": "user", "content": request}, {"role": "assistant", "content": f"answer {turn_no}"}])
        engine.session.messages.append({"role": "user", "content": "current request"})

        for _turn_no in range(7):
            self.assertTrue(engine._discard_oldest_completed_turn())

        retained = engine.session.messages[0]
        retained_requests = retained["_deepy_removed_turn_requests"]
        self.assertEqual(retained["role"], "system")
        self.assertEqual(len(retained_requests), 5)
        self.assertTrue(retained_requests[0].startswith("request 3"))
        self.assertTrue(retained_requests[-1].startswith("request 7"))
        self.assertTrue(retained_requests[-1].endswith("..."))
        self.assertTrue(all(len(request) <= 256 for request in retained_requests))
        self.assertIn("Deepy's corresponding answers were removed", retained["content"])
        self.assertNotIn("answer 7", retained["content"])
        self.assertEqual(engine.session.messages[-1], {"role": "user", "content": "current request"})

    def test_summary_discard_fallback_retains_request_when_whole_prior_turn_must_go(self):
        engine = self._engine()
        engine._render_messages = lambda add_generation_prompt: list(range(100 if sum(message.get("role") == "user" for message in engine.session.messages) > 1 else 10))
        prior_messages = [{"role": "user", "content": "make the previous image dance"}, {"role": "assistant", "content": "large answer"}]
        current_messages = [{"role": "user", "content": "current request"}]

        remaining, discarded_turns, discarded_steps = engine._discard_prior_messages_to_trigger(prior_messages, current_messages, trigger_tokens=50)

        self.assertEqual(discarded_turns, 1)
        self.assertEqual(discarded_steps, 0)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["_deepy_removed_turn_requests"], ["make the previous image dance"])
        self.assertIn("answers were removed", remaining[0]["content"])

    def test_interrupted_turn_keeps_every_complete_tool_group(self):
        session = AssistantSessionState(messages=[{"role": "user", "content": "older"}, {"role": "assistant", "content": "older answer"}], rendered_token_ids=[1, 2], rendered_messages_len=2)
        begin_assistant_turn(session, "user_current", "turn this into a video")
        session.messages.extend([
            {"role": "user", "content": "turn this into a video"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "wangp_get_model_schema", "arguments": {"model_type": "video_fast"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"done","content":"' + ("x" * 20000) + '"}'},
        ])
        checkpoint_assistant_turn(session)

        self.assertTrue(rollback_assistant_turn(session))

        rendered_history = "\n".join(str(message.get("content", "")) for message in session.messages)
        self.assertIn("x" * 100, rendered_history)
        retained_tool_call = session.messages[3]["tool_calls"][0]["function"]
        self.assertEqual(retained_tool_call["name"], "wangp_get_model_schema")
        self.assertEqual(retained_tool_call["arguments"]["model_type"], "video_fast")
        self.assertIn("Do not continue that cancelled turn", rendered_history)

    def test_interrupted_trace_retains_template_values_and_latest_artifact(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "template_call", "type": "function", "function": {"name": "wangp_get_deepy_template_settings", "arguments": {"tool_id": "gen_video", "template": "default"}}}]},
            {"role": "tool", "tool_call_id": "template_call", "content": '{"status":"done","tool_id":"gen_video","template":"H3","general_properties_active":true,"settings":{"model_type":"minimax_h3_fl2va","num_inference_steps":8},"general_properties":{"width":832,"height":480,"num_frames":81,"seed":42}}'},
        ]
        for step_no in range(12):
            messages.append({"role": "assistant", "tool_calls": [{"id": f"read_{step_no}", "type": "function", "function": {"name": "read_tool", "arguments": {"step": step_no}}}]})
            messages.append({"role": "tool", "tool_call_id": f"read_{step_no}", "content": '{"status":"done"}'})
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "generate_call", "type": "function", "function": {"name": "wangp_generate", "arguments": {"source": {"model_type": "minimax_h3_fl2va"}}}},
                ],
            },
            {"role": "tool", "tool_call_id": "generate_call", "content": '{"status":"done","job_id":"job_9","output_file":"outputs/final.mp4"}'},
        ])

        summary = _summarize_interrupted_committed_messages(messages)

        self.assertIn("template settings retained", summary)
        self.assertIn("minimax_h3_fl2va", summary)
        self.assertIn("num_inference_steps", summary)
        self.assertIn("num_frames", summary)
        self.assertIn("job_id=job_9", summary)
        self.assertIn("output_file=outputs/final.mp4", summary)
        self.assertLessEqual(len(summary), 6000)

    def test_interrupted_turn_discards_only_incomplete_trailing_tool_group(self):
        session = AssistantSessionState(messages=[], rendered_token_ids=[1, 2], rendered_messages_len=0, runtime_snapshot={"sequence": {"token_ids": [1, 2]}})
        begin_assistant_turn(session, "user_1", "build media")
        session.messages.extend([
            {"role": "user", "content": "build media"},
            {"role": "assistant", "tool_calls": [{"id": "complete", "type": "function", "function": {"name": "read_tool", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "complete", "content": '{"status":"done"}'},
            {"role": "assistant", "tool_calls": [{"id": "incomplete", "type": "function", "function": {"name": "write_tool", "arguments": {}}}]},
        ])
        checkpoint_assistant_turn(session)

        self.assertTrue(rollback_assistant_turn(session))

        retained_calls = [tool_call["function"]["name"] for message in session.messages for tool_call in message.get("tool_calls", [])]
        self.assertEqual(retained_calls, ["read_tool"])
        self.assertEqual(session.runtime_snapshot, {"sequence": {"token_ids": [1, 2]}})
        self.assertEqual(session.pending_replay_reason, "")

    def test_interrupted_turn_preserves_exact_pre_turn_snapshot(self):
        turn_snapshot = {"sequence": {"token_ids": list(range(28000))}}
        session = AssistantSessionState(
            messages=[{"role": "user", "content": "older"}, {"role": "assistant", "content": "older answer"}],
            rendered_token_ids=list(range(28000)),
            rendered_messages_len=2,
            runtime_snapshot=turn_snapshot,
            reset_base_token_ids=[1, 2, 3],
            reset_base_snapshot={"sequence": {"token_ids": [1, 2, 3]}},
        )
        begin_assistant_turn(session, "user_current", "make a song")
        session.messages.extend([
            {"role": "user", "content": "make a song"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "wangp_generate", "arguments": {"source": "large"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"interrupted","content":"' + ("x" * 20000) + '"}'},
        ])
        checkpoint_assistant_turn(session)

        self.assertTrue(rollback_assistant_turn(session))

        self.assertEqual(session.runtime_snapshot, turn_snapshot)
        self.assertEqual(session.rendered_token_ids, list(range(28000)))
        self.assertEqual(session.rendered_messages_len, 2)
        self.assertEqual(len(session.messages), 6)

    def test_successful_commit_replaces_turn_snapshot_with_system_and_summary(self):
        engine = self._engine()
        prior_messages = AssistantEngine._build_compacted_summary_messages("The user selected image_4 and requested a video.")
        current_messages = [{"role": "user", "content": "Continue now."}]
        engine.session.messages = [{"role": "user", "content": "old"}, *current_messages]
        engine.session.current_turn = {"runtime_snapshot": {"old": True}, "messages_len": 1, "committed_messages_len": 2}
        engine.session.reset_base_token_ids = [1, 2]
        engine.session.reset_base_snapshot = {"sequence": {"token_ids": [1, 2]}}
        engine.session.reset_base_signature = "system-signature"
        engine.session.reset_base_context_window_tokens = 32000
        prefill_counts = []
        engine._run_prefill_call = lambda token_count, callback, **_kwargs: prefill_counts.append(token_count) or callback()

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
        self.assertEqual(engine.session.runtime_snapshot["sequence"]["token_ids"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(engine.runtime.sequence.token_ids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(engine.runtime.prime_calls, 0)
        self.assertEqual(prefill_counts, [2, 2, 2])

    def test_summary_starts_at_85_percent_and_keeps_current_turn(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2", "messages_len": 2, "rendered_messages_len": 2}
        engine.session.rendered_token_ids = [9, 8, 7]
        engine.session.rendered_messages_len = 2
        engine.session.rendered_system_prompt_signature = "system-signature"
        engine.session.rendered_context_window_tokens = 32000
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(28010))
        engine._render_current_turn_slice_suffix = lambda messages, add_generation_prompt: [1]
        engine._prepare_memory_compaction_context = Mock(return_value=({"sequence": {"token_ids": [9, 8, 7]}}, 1000))
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
        engine._mark_history_summarized_trace.assert_called_once_with("A compact factual summary.")
        print_mock.assert_any_call("[Deepy] Context compacted: summarize, 28,010 -> 3 tokens, 1 completed turn summarized")

    def test_completed_turn_compaction_accepts_useful_max_tokens_summary(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2", "messages_len": 2, "rendered_messages_len": 2}
        engine.session.rendered_token_ids = [9, 8, 7]
        engine.session.rendered_messages_len = 2
        engine.session.rendered_system_prompt_signature = "system-signature"
        engine.session.rendered_context_window_tokens = 32000
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(10000 if "<deepy_conversation_summary>" in str(engine.session.messages[0].get("content", "")) else 28010))
        engine._render_current_turn_slice_suffix = lambda messages, add_generation_prompt: [1]
        engine._prepare_memory_compaction_context = Mock(return_value=({"sequence": {"token_ids": [9, 8, 7]}}, 1000))
        engine._set_status = Mock()
        engine._record_generation_metrics = Mock()
        engine._mark_history_summarized_trace = Mock()
        engine._mark_summary_fallback_trace = Mock()
        engine._commit_rewritten_history = Mock()
        engine._log = Mock()
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="Useful capped summary.", stop_reason="max_tokens", token_count=2048))

        self.assertTrue(engine._maybe_summarize_context(generation_reserve_tokens=1024))

        summarized_prior, current_turn, reserve = engine._commit_rewritten_history.call_args.args
        self.assertIn("Useful capped summary.", summarized_prior[0]["content"])
        self.assertEqual(current_turn, [{"role": "user", "content": "current request"}])
        self.assertEqual(reserve, 1024)
        engine._mark_summary_fallback_trace.assert_not_called()
        self.assertTrue(any("28,010 to 10,000" in call.args[0] for call in engine._log.call_args_list))

    def test_single_active_turn_summarizes_old_steps_and_keeps_two_latest(self):
        engine = self._engine()
        engine.session.messages = [{"role": "user", "content": "build a long video"}]
        for step_no in range(1, 5):
            engine.session.messages.extend([
                {"role": "assistant", "tool_calls": [{"id": f"call_{step_no}", "type": "function", "function": {"name": f"step_{step_no}", "arguments": {}}}]},
                {"role": "tool", "tool_call_id": f"call_{step_no}", "content": f"result {step_no}"},
            ])
        engine.session.current_turn = {"user_message_id": "user_1", "messages_len": 0}
        engine.session.rendered_token_ids = list(range(28010))
        engine.session.rendered_messages_len = len(engine.session.messages)
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._prepare_live_compaction_context = Mock(return_value=({"sequence": {"token_ids": [9, 8, 7]}}, 28100))
        ui_events = []
        engine._set_status = Mock(side_effect=lambda text, kind: ui_events.append(("status", text)))
        engine._record_generation_metrics = Mock()
        engine._commit_rewritten_history = Mock()
        engine._mark_history_summarized_trace = Mock(side_effect=lambda summary: ui_events.append(("summary", summary)))
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="Older active work summarized.", stop_reason="stop_token", token_count=8))

        self.assertTrue(engine._maybe_summarize_active_turn(generation_reserve_tokens=4024))

        summarized_prior, retained_current, reserve = engine._commit_rewritten_history.call_args.args
        self.assertIn("Older active work summarized.", summarized_prior[0]["content"])
        self.assertEqual(retained_current[0], {"role": "user", "content": "build a long video"})
        self.assertEqual([message.get("tool_call_id") for message in retained_current if message.get("role") == "tool"], ["call_3", "call_4"])
        self.assertEqual(reserve, 4024)
        self.assertTrue(engine.session.current_turn["summary_compaction_attempted"])
        engine._mark_history_summarized_trace.assert_called_once_with("Older active work summarized.")
        self.assertEqual(ui_events[-2:], [("status", "Thinking..."), ("summary", "Older active work summarized.")])

    def test_active_compaction_accepts_max_tokens_when_rewritten_context_is_shorter(self):
        engine = self._engine()
        engine.session.messages = [{"role": "user", "content": "build a long workflow"}]
        for step_no in range(1, 5):
            engine.session.messages.extend([
                {"role": "assistant", "tool_calls": [{"id": f"call_{step_no}", "type": "function", "function": {"name": f"step_{step_no}", "arguments": {}}}]},
                {"role": "tool", "tool_call_id": f"call_{step_no}", "content": f"result {step_no}"},
            ])
        engine.session.current_turn = {"user_message_id": "user_1", "messages_len": 0}
        engine.session.rendered_token_ids = list(range(28010))
        engine.session.rendered_messages_len = len(engine.session.messages)
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(15000 if "<deepy_conversation_summary>" in str(engine.session.messages[0].get("content", "")) else 28010))
        engine._prepare_live_compaction_context = Mock(return_value=({"sequence": {"token_ids": [9, 8, 7]}}, 28100))
        engine._set_status = Mock()
        engine._record_generation_metrics = Mock()
        engine._commit_rewritten_history = Mock()
        engine._mark_history_summarized_trace = Mock()
        engine._mark_summary_fallback_trace = Mock()
        engine._log = Mock()
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="Useful capped summary.", stop_reason="max_tokens", token_count=2048))

        self.assertTrue(engine._maybe_summarize_active_turn(generation_reserve_tokens=4024))

        summarized_prior, retained_current, reserve = engine._commit_rewritten_history.call_args.args
        self.assertIn("Useful capped summary.", summarized_prior[0]["content"])
        self.assertEqual(reserve, 4024)
        engine._mark_summary_fallback_trace.assert_not_called()
        self.assertTrue(any("28,010 to 15,000" in call.args[0] for call in engine._log.call_args_list))

    def test_capped_compaction_rejects_shorter_result_still_above_half_window(self):
        engine = self._engine()
        engine.session.messages = [{"role": "user", "content": "original"}]
        engine._render_messages = lambda add_generation_prompt: list(range(17000))
        engine._log = Mock()
        summary_messages = engine._build_compacted_summary_messages("A capped but still oversized summary.")

        with self.assertRaisesRegex(RuntimeError, "reduce context and finish below 50%"):
            engine._accept_capped_compaction(summary_messages, [{"role": "user", "content": "current"}], before_tokens=28010, context_window_tokens=32000)

    def test_summary_notice_is_expandable_and_ordered_inside_current_turn(self):
        engine = self._engine()
        user_message_id, _event = assistant_chat.add_user_message(engine.session, "Continue the project.")
        assistant_message_id = assistant_chat.create_assistant_turn(engine.session)
        engine.session.current_turn = {"user_message_id": user_message_id, "assistant_message_id": assistant_message_id}
        engine._active_turn_id = assistant_message_id
        engine._current_status_payload = None
        engine._chat_stats_payload = Mock(return_value=None)
        engine._emit_chat_event = Mock()
        engine._log = Mock()
        engine.debug_enabled = True
        assistant_chat.add_tool_call(engine.session, assistant_message_id, "first_action", {"step": 1})

        with patch("builtins.print") as print_mock:
            engine._mark_history_summarized_trace("The user selected **video_3** and asked to preserve its soundtrack.")

        assistant_chat.add_tool_call(engine.session, assistant_message_id, "next_action", {"step": 2})
        assistant_record = assistant_chat._find_message(engine.session, assistant_message_id)
        self.assertEqual([block["type"] for block in assistant_record["blocks"]], ["tool", "context_summary", "tool"])
        self.assertEqual(len(engine.session.chat_transcript), 2)
        rendered = assistant_chat._render_message_payload(assistant_record)["html"]
        self.assertIn("wangp-assistant-chat__disclosure--context-summary", rendered)
        self.assertIn("Earlier chat history was summarized to preserve Deepy's context window.", rendered)
        self.assertIn("<strong>video_3</strong>", rendered)
        self.assertNotIn("<details open", rendered)
        print_mock.assert_any_call("[Deepy] Compaction summary begin")
        print_mock.assert_any_call("The user selected **video_3** and asked to preserve its soundtrack.")
        print_mock.assert_any_call("[Deepy] Compaction summary end")

    def test_trim_and_failed_summary_emit_no_out_of_turn_chat_cards(self):
        engine = self._engine()
        engine.session.current_turn = {"user_message_id": "user_1"}
        engine._emit_chat_event = Mock()
        engine._log = Mock()
        original_transcript = list(engine.session.chat_transcript)

        engine._mark_history_trimmed_trace()
        engine._mark_summary_fallback_trace(RuntimeError("summary failed"))

        self.assertTrue(engine.session.current_turn["history_trimmed"])
        self.assertEqual(engine.session.chat_transcript, original_transcript)
        engine._emit_chat_event.assert_not_called()

    def test_summary_restores_unloaded_runtime_before_reading_kv_configuration(self):
        engine = self._engine()
        engine.runtime = _InitiallyUnloadedFakeRuntime()
        engine.runtime.tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [7, 8])
        engine.runtime.model = SimpleNamespace()
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="A compact factual summary.", stop_reason="stop_token", token_count=5))
        engine.thinking_enabled = True
        engine.tool_box = SimpleNamespace(get_tool_schemas=lambda: [])
        engine._build_system_prompt = lambda log_injections=False: "system"
        engine.session.messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
        ]
        engine.session.current_turn = {
            "user_message_id": "user_2",
            "messages_len": 2,
            "rendered_messages_len": 2,
            "rendered_token_ids": [9, 8, 7],
            "runtime_snapshot": {"sequence": {"token_ids": [9, 8, 7]}},
            "rendered_system_prompt_signature": "system-signature",
            "rendered_context_window_tokens": 32000,
        }
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(28010))
        engine._render_current_turn_slice_suffix = lambda messages, add_generation_prompt: [1]
        engine._set_status = Mock()
        engine._record_generation_metrics = Mock()
        engine._mark_history_summarized_trace = Mock()
        engine._commit_rewritten_history = Mock()

        self.assertTrue(engine._maybe_summarize_context(generation_reserve_tokens=1024))

        self.assertEqual(engine.runtime.restore_calls, 1)
        engine._commit_rewritten_history.assert_called_once()

    def test_summarize_mode_continues_after_full_max_tokens_segment(self):
        engine = self._engine()
        engine._current_requested_max_new_tokens = 4096
        engine.thinking_enabled = True
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._recover_from_context_limit = Mock(return_value=True)
        engine._emit_stats = Mock()
        engine._log = Mock()
        engine.runtime.sequence.completion_token_ids = [11, 12]
        result = SimpleNamespace(stop_reason="max_tokens", token_count=4096)

        self.assertTrue(engine._recover_after_generation_limit(result, "partial <tool_call>", retry_no=7))

        engine._recover_from_context_limit.assert_not_called()
        self.assertTrue(engine._skip_generation_context_sync_once)
        self.assertTrue(engine._resume_stream_after_context_trim)
        self.assertEqual(engine._continued_segment_token_ids, [11, 12])
        self.assertIn("without compaction or replay", engine._log.call_args.args[0])

    def test_summarize_mode_makes_room_only_when_next_segment_will_not_fit(self):
        engine = self._engine()
        engine._current_requested_max_new_tokens = 1024
        engine.thinking_enabled = True
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._recover_from_context_limit = Mock(return_value=True)
        engine._log = Mock()
        engine.runtime.sequence.token_ids = list(range(31500))
        engine.runtime.sequence.completion_token_ids = [11, 12]
        result = SimpleNamespace(stop_reason="max_tokens", token_count=1024)

        self.assertTrue(engine._recover_after_generation_limit(result, "unfinished thought", retry_no=7))

        engine._recover_from_context_limit.assert_called_once_with("unfinished thought", 0)

    def test_context_limit_recovery_keeps_prompt_only_stop_snapshot(self):
        engine = self._engine()
        engine.session.messages = [{"role": "user", "content": "continue"}]
        engine.runtime.sequence.token_ids = [8, 9, 11, 12]
        engine.runtime.sequence.completion_token_ids = [11, 12]
        engine._fit_rendered_messages_to_window = lambda **_kwargs: ([8, 9], False)
        engine._segment_generation_reserve_tokens = lambda: 128
        engine._extend_context_from_preserved_base = lambda _tokens: None
        engine._set_status = Mock()
        engine._emit_stats = Mock()
        engine._log = Mock()

        self.assertTrue(engine._recover_from_context_limit("unfinished thought", retry_no=0))

        self.assertEqual(engine.session.rendered_token_ids, [8, 9])
        self.assertEqual(engine.session.runtime_snapshot["sequence"]["token_ids"], [8, 9])
        self.assertEqual(engine.runtime.sequence.token_ids, [8, 9, 11, 12])

    def test_live_continuation_preserves_only_current_stream_state(self):
        engine = self._engine()
        engine.runtime.model = SimpleNamespace()
        engine.thinking_enabled = True
        assistant_id = assistant_chat.create_assistant_turn(engine.session)
        assistant_chat.set_assistant_content(engine.session, assistant_id, "older user-facing text")
        engine._active_turn_id = assistant_id
        engine._resume_stream_after_context_trim = True
        engine._stream_answer_text = ""
        engine._stream_reasoning_text = "partial current thought"
        engine._stream_reasoning_block_id = "reasoning_current"
        engine._stream_thinking_unknown = False
        engine._stream_thinking_open = True

        engine._start_stream_pass()

        self.assertEqual(engine._stream_answer_text, "")
        self.assertEqual(engine._stream_reasoning_text, "partial current thought")
        self.assertEqual(engine._stream_reasoning_block_id, "reasoning_current")
        self.assertTrue(engine._stream_thinking_open)

    def test_identical_tool_calls_in_one_response_are_executed_once(self):
        generate_call = {"name": "wangp_generate", "arguments": {"source": {"model_type": "example", "seed": 7}, "wait": True}}
        calls = AssistantEngine._deduplicate_tool_calls([
            generate_call,
            {"name": "wangp_generate", "arguments": {"wait": True, "source": {"seed": 7, "model_type": "example"}}},
            {"name": "wangp_list_gallery", "arguments": {"media_type": "video"}},
        ])

        self.assertEqual(calls, [generate_call, {"name": "wangp_list_gallery", "arguments": {"media_type": "video"}}])

    def test_discard_mode_keeps_max_tokens_as_terminal_when_not_at_context_ceiling(self):
        engine = self._engine()
        engine._current_requested_max_new_tokens = 4096
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_DISCARD
        engine._recover_from_context_limit = Mock(return_value=True)
        result = SimpleNamespace(stop_reason="max_tokens", token_count=4096)

        self.assertFalse(engine._recover_after_generation_limit(result, "finished answer", retry_no=0))

        engine._recover_from_context_limit.assert_not_called()

    def test_summary_is_skipped_when_only_current_turn_causes_overflow(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
            {"role": "tool", "content": "oversized current result"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2", "messages_len": 2, "rendered_messages_len": 2}
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(50000))
        engine._render_current_turn_slice_suffix = lambda messages, add_generation_prompt: list(range(48000))
        engine._restore_or_replay_session = Mock()

        self.assertFalse(engine._maybe_summarize_context(generation_reserve_tokens=1024))
        engine._restore_or_replay_session.assert_not_called()
        self.assertNotIn("summary_compaction_attempted", engine.session.current_turn)

    def test_prior_summary_waits_until_preserved_interrupted_suffix_is_synchronized(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "interrupted request"},
            {"role": "assistant", "content": "completed work"},
            {"role": "user", "content": "new request"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2", "messages_len": 2, "rendered_messages_len": 1}
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(28010))
        engine._prepare_memory_compaction_context = Mock()

        self.assertFalse(engine._maybe_summarize_context(generation_reserve_tokens=1024))

        engine._prepare_memory_compaction_context.assert_not_called()

    def test_synchronized_preserved_prior_history_can_be_summarized_before_new_decode(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "interrupted request"},
            {"role": "assistant", "content": "completed work"},
            {"role": "user", "content": "new request"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2", "messages_len": 2}
        engine.session.rendered_token_ids = list(range(28010))
        engine.session.rendered_messages_len = 3
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._prepare_live_compaction_context = Mock(return_value=({"sequence": {"token_ids": [1, 2]}}, 28100))
        engine._set_status = Mock()
        engine._record_generation_metrics = Mock()
        engine._commit_rewritten_history = Mock()
        engine._mark_history_summarized_trace = Mock()
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="Preserved interrupted work summarized.", stop_reason="stop_token", token_count=6))

        self.assertTrue(engine._maybe_summarize_active_turn(generation_reserve_tokens=4024))

        summarized_prior, retained_current, reserve = engine._commit_rewritten_history.call_args.args
        self.assertIn("Preserved interrupted work summarized.", summarized_prior[0]["content"])
        self.assertEqual(retained_current, [{"role": "user", "content": "new request"}])
        self.assertEqual(reserve, 4024)

    def test_failed_summary_restores_snapshot_and_uses_discard_compaction(self):
        engine = self._engine()
        engine.session.messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
        ]
        engine.session.current_turn = {"user_message_id": "user_2", "messages_len": 2, "rendered_messages_len": 2}
        engine.session.rendered_token_ids = [9, 8, 7]
        engine.session.rendered_messages_len = 2
        engine.session.rendered_system_prompt_signature = "system-signature"
        engine.session.rendered_context_window_tokens = 32000
        engine._get_compaction_type = lambda: DEEPY_COMPACTION_TYPE_SUMMARIZE
        engine._render_messages = lambda add_generation_prompt: list(range(28010))
        engine._render_current_turn_slice_suffix = lambda messages, add_generation_prompt: [1]
        engine._prepare_memory_compaction_context = Mock(return_value=({"sequence": {"token_ids": [9, 8, 7]}}, 1000))
        engine._set_status = Mock()
        engine._record_generation_metrics = Mock()
        engine._mark_summary_fallback_trace = Mock()
        engine._mark_history_trimmed_trace = Mock()
        engine._discard_prior_messages_to_trigger = Mock(return_value=([], 1, 0))
        engine._commit_rewritten_history = Mock()
        engine.runtime.generate_segment = Mock(return_value=SimpleNamespace(raw_text="unfinished", stop_reason="max_tokens", token_count=2048))

        with patch("builtins.print") as print_mock:
            self.assertTrue(engine._maybe_summarize_context(generation_reserve_tokens=1024))
        engine._mark_summary_fallback_trace.assert_called_once()
        engine._discard_prior_messages_to_trigger.assert_called_once()
        engine._commit_rewritten_history.assert_called_once_with([], [{"role": "user", "content": "current request"}], 1024)
        engine._mark_history_trimmed_trace.assert_called_once()
        self.assertEqual(engine.runtime.sequence.token_ids, [9, 8, 7])
        print_mock.assert_any_call("[Deepy] Context compacted: summarize failed; discard fallback, 28,010 -> 3 tokens, 1 oldest turn removed")


if __name__ == "__main__":
    unittest.main()
