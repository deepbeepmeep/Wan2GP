import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from shared.api import WanGPSession
from shared.api_webui import GradioWanGPSession
from shared.utils.loras_mutipliers import merge_loras_settings


ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPE = "ltx2_25_22B"
FAST_CHOICE = "ltx2_25_dev_accelerators/Two-Stage Dev DistilledLoRA (8+3 Steps).json"
QUALITY_CHOICE = "ltx2_25_dev_accelerators/Two-Stage Dev HQ Res2S (15+3 Steps).json"
VBVR_CHOICE = "ltx2_presets/VBVR LoRA - Video Reasoning.json"
DISTILLED_LORA = "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/ltx-2.5-22b-distilled-lora-450_bf16.safetensors"
VBVR_LORA = "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/loras/Ltx2.3-Licon-VBVR-I2V-96000-R32.safetensors"


class _FakeRuntimeModule:
    def __init__(self, accelerator_choices, preset_choices, defaults=None):
        self.groups = {
            "accelerator_profiles": [str(choice) for choice in accelerator_choices],
            "preset_settings": [str(choice) for choice in preset_choices],
        }
        self.defaults = copy.deepcopy(defaults or {
            "sample_solver": "default_solver",
            "num_inference_steps": 99,
            "default_only": "kept",
            "config": {"source": "default"},
            "nested": {"values": []},
            "activated_loras": ["default.safetensors"],
            "loras_multipliers": "0.25",
        })
        self.fix_calls = []

    def get_model_def(self, model_type):
        return {"name": "LTX"} if model_type == MODEL_TYPE else None

    def get_default_settings(self, model_type):
        if model_type != MODEL_TYPE:
            raise ValueError(model_type)
        return copy.deepcopy(self.defaults)

    def _get_builtin_lset_groups(self, model_type):
        if model_type != MODEL_TYPE:
            return []
        return [(group_id, group_id, choices) for group_id, choices in self.groups.items()]

    @staticmethod
    def _builtin_lset_file_path(choice):
        path = Path(choice)
        return path if path.is_absolute() else ROOT / "profiles" / path

    @staticmethod
    def are_model_types_compatible(declared, requested):
        return requested == MODEL_TYPE and declared in (MODEL_TYPE, "compatible_alias")

    merge_loras_settings = staticmethod(merge_loras_settings)

    def fix_settings(self, model_type, settings, min_settings_version=0):
        self.fix_calls.append((model_type, min_settings_version))
        settings["normalized"] = True


def _session(accelerator_choices=(), preset_choices=(), defaults=None):
    module = _FakeRuntimeModule(accelerator_choices, preset_choices, defaults=defaults)
    runtime = SimpleNamespace(root=ROOT, module=module)
    session = WanGPSession(root=ROOT, console_output=False)
    session._ensure_runtime = lambda: runtime
    return session, module


class AcceleratorProfileResolutionTests(unittest.TestCase):
    def test_resolves_ltx_profiles_with_defaults_loras_and_isolated_results(self):
        session, module = _session([FAST_CHOICE, QUALITY_CHOICE], [VBVR_CHOICE])

        fast = session.resolve_profiles(MODEL_TYPE, accelerator_profile_id="ltx2_25_two_stage_distilled_8_3")
        quality = session.resolve_profiles(MODEL_TYPE, accelerator_profile_id="ltx2_25_two_stage_hq_res2s_15_3")
        preset = session.resolve_profiles(MODEL_TYPE, preset_profile_id="ltx2_vbvr_video_reasoning")
        combined = session.resolve_profiles(
            MODEL_TYPE,
            accelerator_profile_id="ltx2_25_two_stage_distilled_8_3",
            preset_profile_id="ltx2_vbvr_video_reasoning",
        )

        self.assertEqual(fast["sample_solver"], "distilled_8_steps_ancestral")
        self.assertEqual(fast["num_inference_steps"], 8)
        self.assertEqual(fast["guidance_phases"], 2)
        self.assertEqual(fast["activated_loras"], [DISTILLED_LORA, "default.safetensors"])
        self.assertEqual(fast["loras_multipliers"], "0.5|0.25")
        self.assertEqual(quality["sample_solver"], "res2s")
        self.assertEqual(quality["num_inference_steps"], 15)
        self.assertEqual(quality["guidance_phases"], 2)
        self.assertEqual(
            (quality["guidance_scale"], quality["audio_guidance_scale"], quality["alt_guidance_scale"]),
            (3.0, 7.0, 3.0),
        )
        self.assertEqual(quality["alt_scale"], 0.45)
        self.assertEqual(quality["activated_loras"], [DISTILLED_LORA, "default.safetensors"])
        self.assertEqual(quality["loras_multipliers"], "0.5|0.25")
        self.assertEqual(preset["activated_loras"], [VBVR_LORA])
        self.assertEqual(preset["loras_multipliers"], "1")
        self.assertEqual(combined["sample_solver"], "distilled_8_steps_ancestral")
        self.assertEqual(combined["activated_loras"], [DISTILLED_LORA, VBVR_LORA])
        self.assertEqual(combined["loras_multipliers"], "0.5|1")
        for settings in (fast, quality, preset, combined):
            self.assertEqual(settings["model_type"], MODEL_TYPE)
            self.assertEqual(settings["settings_version"], 2.56)
            self.assertEqual(settings["default_only"], "kept")
            self.assertEqual(settings["config"], {"source": "default"})
            self.assertTrue(settings["normalized"])
            self.assertNotIn("profile_id", settings)
            self.assertNotIn("help", settings)
        self.assertEqual(module.fix_calls, [(MODEL_TYPE, 2.38)] * 5)

        fast["nested"]["values"].append("changed")
        again = session.resolve_profiles(MODEL_TYPE, accelerator_profile_id="ltx2_25_two_stage_distilled_8_3")
        self.assertEqual(again["nested"]["values"], [])
        self.assertEqual(module.defaults["nested"]["values"], [])

    def test_validation_legacy_and_model_type_contracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)

            inherited = temp / "legacy.json"
            inherited.write_text(json.dumps({"activated_loras": ["profile.safetensors"], "settings_version": 2.56}), encoding="utf-8")
            session, _ = _session(preset_choices=[inherited])
            resolved = session.resolve_profiles(MODEL_TYPE, preset_profile_id="legacy.json")
            self.assertEqual(resolved["model_type"], MODEL_TYPE)
            self.assertEqual(resolved["activated_loras"], ["profile.safetensors"])
            self.assertEqual(resolved["loras_multipliers"], "1")

            compatible = temp / "compatible.json"
            compatible.write_text(json.dumps({"profile_id": "compatible", "model_type": "compatible_alias"}), encoding="utf-8")
            session, _ = _session([compatible])
            self.assertEqual(session.resolve_profiles(MODEL_TYPE, accelerator_profile_id="compatible")["model_type"], MODEL_TYPE)

            incompatible = temp / "incompatible.json"
            incompatible.write_text(json.dumps({"profile_id": "incompatible", "model_type": "other_model"}), encoding="utf-8")
            session, _ = _session([incompatible])
            with self.assertRaisesRegex(ValueError, "incompatible model_type"):
                session.resolve_profiles(MODEL_TYPE, accelerator_profile_id="incompatible")

            duplicate_a = temp / "duplicate-a.json"
            duplicate_b = temp / "duplicate-b.json"
            duplicate_a.write_text(json.dumps({"profile_id": "duplicate"}), encoding="utf-8")
            duplicate_b.write_text(json.dumps({"profile_id": "duplicate"}), encoding="utf-8")
            session, _ = _session([duplicate_a, duplicate_b])
            with self.assertRaisesRegex(ValueError, "Duplicate accelerator profile_id"):
                session.resolve_profiles(MODEL_TYPE, accelerator_profile_id="duplicate")

            invalid = temp / "invalid.json"
            invalid.write_text("{not-json", encoding="utf-8")
            session, _ = _session(preset_choices=[invalid])
            with self.assertRaisesRegex(ValueError, "Invalid preset profile JSON"):
                session.resolve_profiles(MODEL_TYPE, preset_profile_id="invalid.json")

            first = temp / "first" / "ambiguous.json"
            second = temp / "second" / "ambiguous.json"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            session, _ = _session(preset_choices=[first, second])
            with self.assertRaisesRegex(ValueError, "Ambiguous legacy"):
                session.resolve_profiles(MODEL_TYPE, preset_profile_id="ambiguous.json")

            session, _ = _session(preset_choices=[inherited])
            with self.assertRaisesRegex(ValueError, "Unknown model_type"):
                session.resolve_profiles("unknown_model", preset_profile_id="legacy.json")
            with self.assertRaisesRegex(ValueError, "Unknown accelerator_profile_id"):
                session.resolve_profiles(MODEL_TYPE, accelerator_profile_id="unknown")
            with self.assertRaisesRegex(ValueError, "At least one profile ID"):
                session.resolve_profiles(MODEL_TYPE)
            for unsafe in ("../legacy.json", "C:\\tmp\\legacy.json", "/tmp/legacy.json", "\\\\server\\share\\legacy.json"):
                with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, "not a filesystem path"):
                    session.resolve_profiles(MODEL_TYPE, preset_profile_id=unsafe)

    def test_webui_proxy_and_run_path_remain_direct_delegations(self):
        backend = Mock()
        backend.resolve_profiles.return_value = {"model_type": MODEL_TYPE}
        adapter = GradioWanGPSession(init_fn=lambda **_: backend)
        adapter._ensure_session = lambda: backend
        self.assertEqual(
            adapter.resolve_profiles(MODEL_TYPE, accelerator_profile_id="accelerator", preset_profile_id="preset"),
            {"model_type": MODEL_TYPE},
        )
        backend.resolve_profiles.assert_called_once_with(
            MODEL_TYPE,
            accelerator_profile_id="accelerator",
            preset_profile_id="preset",
        )
        self.assertFalse(hasattr(WanGPSession, "resolve_accelerator_profile"))
        self.assertFalse(hasattr(GradioWanGPSession, "resolve_accelerator_profile"))

        session = WanGPSession(root=ROOT, console_output=False)
        job = Mock()
        job.result.return_value = "unchanged"
        session.submit = Mock(return_value=job)
        source = Path("existing-profile.json")
        self.assertEqual(session.run(source), "unchanged")
        session.submit.assert_called_once_with(source, callbacks=None)
        job.result.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
