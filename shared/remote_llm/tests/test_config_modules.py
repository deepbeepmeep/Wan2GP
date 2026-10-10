from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shared.remote_llm.claude_config import claude_profile_from_values
from shared.remote_llm.codex_config import codex_profile_from_values
from shared.remote_llm.config import ENGINE_OPENCODE, LLM_CONFIG_KEY
from shared.remote_llm.config_ui_common import save_model_catalog
from shared.remote_llm.opencode_config import opencode_profile_from_values, validate_opencode_profile


class RemoteConfigModuleTests(unittest.TestCase):
    def test_engine_profiles_normalize_ui_values(self):
        self.assertEqual(codex_profile_from_values("  codex.exe  ", "Automatic (Codex recommended default)", "Automatic (model default)", ["cached"]), {"executable": "codex.exe", "model": "", "reasoning_effort": "", "model_catalog": ["cached"]})
        self.assertEqual(claude_profile_from_values("  claude.exe  ", "Automatic (Claude account default)", "high", ["cached"]), {"executable": "claude.exe", "model": "", "reasoning_effort": "high", "model_catalog": ["cached"]})
        self.assertEqual(opencode_profile_from_values(" opencode.exe ", "http://127.0.0.1:4096/", "openai", "gpt", "low", "  {}  ", ["cached"]), {"executable": "opencode.exe", "base_url": "http://127.0.0.1:4096", "provider": "openai", "model": "gpt", "reasoning_effort": "low", "config": "{}", "model_catalog": ["cached"]})

    def test_opencode_validation_is_engine_owned(self):
        with self.assertRaisesRegex(ValueError, "absolute http"):
            validate_opencode_profile({"executable": "opencode", "base_url": "localhost:4096"})

    def test_catalog_persistence_keeps_selection_independent(self):
        server_config = {LLM_CONFIG_KEY: {"deepy": ENGINE_OPENCODE, "profiles": {ENGINE_OPENCODE: {"provider": "openai", "model": "selected"}}}}
        catalog = [{"provider": "openai", "model": "new", "display_name": "New"}]
        with tempfile.TemporaryDirectory() as root:
            filename = str(Path(root, "wgp_config.json"))
            save_model_catalog(server_config, filename, ENGINE_OPENCODE, catalog)
            saved = json.loads(Path(filename).read_text(encoding="utf-8"))
        profile = saved[LLM_CONFIG_KEY]["profiles"][ENGINE_OPENCODE]
        self.assertEqual(profile["model"], "selected")
        self.assertEqual(profile["model_catalog"][0]["model"], "new")


if __name__ == "__main__":
    unittest.main()
