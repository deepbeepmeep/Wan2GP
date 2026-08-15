import ast
import json
import unittest
from pathlib import Path


def _load_variant_selection():
    source_path = Path("postprocessing/ltx2_upsampler/runtime.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"MODEL_TYPES", "LORA_KEYS", "LORA_MULTIPLIERS"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.Assign) and node.targets[0].id in names)
        or (isinstance(node, ast.FunctionDef) and node.name == "lora_urls")
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source_path, "exec"), namespace)
    return namespace


class LTXUpsamplerRuntimeTests(unittest.TestCase):
    def test_variants_load_dev_base_with_matching_distilled_and_upscaler_loras(self):
        runtime = _load_variant_selection()
        expected = {
            "ltx23": ("ltx2_22B", ("distilled23", "upscaler23"), (0.5, 1.0)),
            "ltx25": ("ltx2_25_22B", ("distilled25", "upscaler25"), (1.0, 1.0)),
        }
        model_defs = {
            "ltx2_22B": {
                "ltx2_lora_distilled_1_1": "distilled23",
                "ltx2_lora_pixel_spatial_upscaler": "upscaler23",
            },
            "ltx2_25_22B": {
                "ltx2_lora_distilled": "distilled25",
                "ltx2_lora_pixel_spatial_upscaler": "upscaler25",
            },
        }
        wgp = type("WGP", (), {"get_model_def": staticmethod(model_defs.__getitem__)})()

        for variant, (model_type, loras, multipliers) in expected.items():
            with self.subTest(variant=variant):
                self.assertEqual(runtime["MODEL_TYPES"][variant], model_type)
                self.assertEqual(runtime["lora_urls"](wgp, variant), loras)
                self.assertEqual(runtime["LORA_MULTIPLIERS"][variant], multipliers)

    def test_full_ltx25_distilled_checkpoint_definition_remains_available(self):
        model_def = json.loads(Path("defaults/ltx2_25_22B_distilled.json").read_text(encoding="utf-8"))["model"]

        self.assertEqual(model_def["architecture"], "ltx2_25_22B")
        self.assertEqual(model_def["ltx2_pipeline"], "distilled")
        self.assertTrue(all("distilled_diffusion_model" in url for url in model_def["URLs"]))


if __name__ == "__main__":
    unittest.main()
