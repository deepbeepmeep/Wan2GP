from __future__ import annotations

import unittest

from shared.deepy.config import deepy_requirement_error
from shared.remote_llm.config import (
    CODEX_DEFAULT_MODEL_CATALOG,
    CLAUDE_DEFAULT_MODEL_CATALOG,
    DEEPY_ENGINE_CHOICES,
    ENGINE_CLAUDE,
    ENGINE_CODEX,
    ENGINE_LOCAL_1,
    ENGINE_LOCAL_2,
    ENGINE_OPENCODE,
    ENGINE_QWEN35_9B,
    ENGINE_QWEN38_27B,
    LLM_CONFIG_KEY,
    claude_model_choices,
    claude_reasoning_effort_choices,
    codex_model_choices,
    codex_reasoning_effort_choices,
    engine_capabilities,
    normalize_claude_model_selection,
    normalize_llm_config,
    opencode_model_choices,
    opencode_provider_choices,
    opencode_reasoning_effort_choices,
    normalize_reasoning_effort_selection,
    privacy_warning,
    resolve_role_engine,
    validate_llm_config,
)


class RemoteLLMConfigTests(unittest.TestCase):
    def test_legacy_enhancer_migrates_without_changing_local_behavior(self):
        config = normalize_llm_config({"enhancer_enabled": 4})
        self.assertEqual(config["deepy"], ENGINE_QWEN35_9B)
        self.assertEqual(resolve_role_engine({LLM_CONFIG_KEY: config}, "prompt_enhancer"), ENGINE_QWEN35_9B)

    def test_remote_role_cannot_save_without_deepy_prime(self):
        config = {LLM_CONFIG_KEY: {"deepy": ENGINE_CODEX}}
        for enabled, deepy_type in ((0, "prime"), (1, "zero")):
            with self.subTest(enabled=enabled, deepy_type=deepy_type):
                with self.assertRaisesRegex(ValueError, "requires Deepy Prime"):
                    validate_llm_config(config, deepy_enabled=enabled, deepy_type=deepy_type)

    def test_one_engine_drives_deepy_prompt_enhancer_and_visual_inspection(self):
        config = {LLM_CONFIG_KEY: {"deepy": ENGINE_CODEX, "prompt_enhancer": "qwen35_9b", "visual_inspector": "disabled"}}
        self.assertEqual({resolve_role_engine(config, role) for role in ("deepy", "prompt_enhancer", "visual_inspector")}, {ENGINE_CODEX})

    def test_small_prompt_enhancers_remain_in_shared_engine_choices(self):
        values = {value for _label, value in DEEPY_ENGINE_CHOICES}
        self.assertIn(ENGINE_LOCAL_1, values)
        self.assertIn(ENGINE_LOCAL_2, values)

    def test_remote_prime_config_keeps_only_non_secret_profile_fields(self):
        config = {
            LLM_CONFIG_KEY: {
                "deepy": ENGINE_CODEX,
                "profiles": {ENGINE_CODEX: {"executable": "codex-custom", "model": "gpt-x", "password": "do-not-store", "token": "do-not-store"}},
            }
        }
        normalized = validate_llm_config(config, deepy_enabled=1, deepy_type="prime")
        self.assertEqual(normalized["profiles"][ENGINE_CODEX]["executable"], "codex-custom")
        self.assertNotIn("password", normalized["profiles"][ENGINE_CODEX])
        self.assertNotIn("token", normalized["profiles"][ENGINE_CODEX])

    def test_codex_model_catalog_is_cached_independently_from_selection(self):
        catalog = [{"model": "gpt-current", "display_name": "GPT Current", "is_default": True, "default_reasoning_effort": "medium", "reasoning_efforts": ["low", "medium", "high"]}]
        config = normalize_llm_config({LLM_CONFIG_KEY: {"profiles": {ENGINE_CODEX: {"model": "gpt-selected", "model_catalog": catalog}}}})
        self.assertEqual(config["profiles"][ENGINE_CODEX]["model"], "gpt-selected")
        self.assertEqual(config["profiles"][ENGINE_CODEX]["model_catalog"], catalog)
        self.assertEqual(codex_model_choices(catalog, "gpt-selected")[-1], ("gpt-selected (currently selected)", "gpt-selected"))
        self.assertEqual(codex_reasoning_effort_choices(catalog, "gpt-current", "high"), [("Automatic (model default)", ""), ("Low", "low"), ("Medium (model default)", "medium"), ("High", "high")])

    def test_codex_model_catalog_has_current_bundled_defaults(self):
        self.assertEqual(CODEX_DEFAULT_MODEL_CATALOG[0]["model"], "gpt-5.6-sol")
        self.assertIn("gpt-5.3-codex-spark", {entry["model"] for entry in CODEX_DEFAULT_MODEL_CATALOG})

    def test_claude_model_catalog_and_effort_are_cached_independently(self):
        catalog = [{"model": "sonnet", "display_name": "Sonnet", "is_default": False, "default_reasoning_effort": "high", "reasoning_efforts": ["low", "high"]}]
        config = normalize_llm_config({LLM_CONFIG_KEY: {"profiles": {ENGINE_CLAUDE: {"model": "custom-claude", "reasoning_effort": "high", "model_catalog": catalog}}}})
        self.assertEqual(config["profiles"][ENGINE_CLAUDE]["model"], "custom-claude")
        self.assertEqual(config["profiles"][ENGINE_CLAUDE]["reasoning_effort"], "high")
        self.assertEqual(config["profiles"][ENGINE_CLAUDE]["model_catalog"], catalog)
        self.assertEqual(claude_model_choices(catalog, "custom-claude")[-1], ("custom-claude (currently selected)", "custom-claude"))
        self.assertEqual(claude_reasoning_effort_choices(catalog, "sonnet", "high"), [("Automatic (model default)", ""), ("Low", "low"), ("High (model default)", "high")])

    def test_claude_model_catalog_has_current_documented_aliases(self):
        aliases = {entry["model"] for entry in CLAUDE_DEFAULT_MODEL_CATALOG}
        self.assertTrue({"best", "fable", "sonnet", "opus", "haiku", "sonnet[1m]", "opus[1m]", "opusplan"}.issubset(aliases))

    def test_claude_automatic_dropdown_labels_are_not_model_ids(self):
        config = normalize_llm_config({LLM_CONFIG_KEY: {"profiles": {ENGINE_CLAUDE: {"model": "Automatic (Claude account default)", "reasoning_effort": "Automatic (model default)"}}}})
        self.assertEqual(config["profiles"][ENGINE_CLAUDE]["model"], "")
        self.assertEqual(config["profiles"][ENGINE_CLAUDE]["reasoning_effort"], "")
        self.assertEqual(normalize_claude_model_selection("Automatic (Claude account default)"), "")
        self.assertEqual(normalize_reasoning_effort_selection("Automatic (model default)"), "")

    def test_opencode_catalog_drives_provider_model_variant_and_context_metadata(self):
        catalog = [
            {"provider": "openai", "provider_name": "OpenAI", "model": "gpt-codex", "display_name": "GPT Codex", "is_default": True, "context_window": 200000, "reasoning_efforts": ["low", "high"]},
            {"provider": "local", "provider_name": "Local", "model": "qwen", "display_name": "Qwen", "is_default": False, "context_window": 32768, "reasoning_efforts": []},
        ]
        config = normalize_llm_config({LLM_CONFIG_KEY: {"profiles": {ENGINE_OPENCODE: {"provider": "openai", "model": "gpt-codex", "reasoning_effort": "high", "config": '{"share":"disabled"}', "model_catalog": catalog}}}})
        profile = config["profiles"][ENGINE_OPENCODE]
        self.assertEqual(profile["config"], '{"share":"disabled"}')
        self.assertEqual(profile["model_catalog"], catalog)
        self.assertEqual(opencode_provider_choices(catalog, "openai"), [("Automatic (OpenCode default)", ""), ("OpenAI", "openai"), ("Local", "local")])
        self.assertEqual(opencode_model_choices(catalog, "openai", "gpt-codex"), [("Automatic (provider default)", ""), ("GPT Codex (default)", "gpt-codex")])
        self.assertEqual(opencode_reasoning_effort_choices(catalog, "openai", "gpt-codex", "high"), [("Automatic (model default)", ""), ("Low", "low"), ("High", "high")])

    def test_remote_deepy_satisfies_local_model_requirement_only_for_prime(self):
        config = {"deepy_enabled": 1, "deepy_type": "prime", LLM_CONFIG_KEY: {"deepy": ENGINE_CODEX}}
        self.assertEqual(deepy_requirement_error(config), "")
        config["deepy_type"] = "zero"
        self.assertIn("require Deepy Prime", deepy_requirement_error(config))

    def test_canonical_local_engine_overrides_legacy_loader_value_for_validation(self):
        config = {
            "enhancer_enabled": 3,
            "deepy_enabled": 1,
            "deepy_type": "prime",
            "deepy_compaction_type": "summarize",
            "deepy_context_tokens": 32000,
            LLM_CONFIG_KEY: {"deepy": ENGINE_QWEN38_27B},
        }
        self.assertEqual(deepy_requirement_error(config), "")

    def test_privacy_warning_and_local_opencode_classification(self):
        remote = {LLM_CONFIG_KEY: {"deepy": ENGINE_CODEX}}
        self.assertIn("**Internet connection required", privacy_warning(remote))
        local = {LLM_CONFIG_KEY: {"deepy": ENGINE_OPENCODE, "profiles": {ENGINE_OPENCODE: {"base_url": "http://127.0.0.1:4096", "provider": "ollama"}}}}
        capabilities = engine_capabilities(local, ENGINE_OPENCODE)
        self.assertFalse(capabilities.requires_internet)
        self.assertFalse(capabilities.data_leaves_machine)
        self.assertEqual(privacy_warning(local), "")


if __name__ == "__main__":
    unittest.main()
