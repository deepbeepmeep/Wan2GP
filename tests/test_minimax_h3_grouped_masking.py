import unittest
from unittest.mock import patch

import torch
from mmgp import offload

from models.minimax_h3.constants import H3_MASK_MODE_DEFAULT, H3_MASK_MODE_GROUPED_ROWS, H3_MASK_MODE_SETTING, h3_grouped_masking_enabled
from models.minimax_h3.minimax_h3_handler import FL2VA_ARCHITECTURE, family_handler
from models.minimax_h3.pipeline import _build_outpainting_mask, _masking_step_mask, _resize_video_mask, _set_grouped_video_rows, _snap_video_mask_to_patch_cells
from models.minimax_h3.transformer import MiniMaxH3Model, VISUAL_COND_TIMESTEP, _grouped_video_timestep_rows
from shared.attention import attention_shared_state
from shared.gradio.magic_mask import video_mask_area_visible, video_mask_dropdown_visible


def _tiny_model():
    torch.manual_seed(7)
    return MiniMaxH3Model(hidden_size=8, num_layers=1, token_refiner_num_layers=0, num_attention_heads=1,
                          attention_head_dim=8, ffn_hidden_size=16, latents_dim=4, audio_latents_dim=4,
                          patch_size=(1, 2, 2), text_dim=8, timestep_input_dim=8, time_embed_hidden_size=8,
                          time_embed_dim=4, rope_inv_freq_len=1, ffn_chunk_size=0,
                          dtype=torch.float32, device="cpu").eval()


def _inputs():
    generator = torch.Generator().manual_seed(11)
    video = torch.randn((1, 4, 2, 4, 4), generator=generator)
    audio = torch.randn((1, 4, 2, 2), generator=generator)
    context = torch.randn((1, 1, 8), generator=generator)
    payload = {"text_token_tags": torch.tensor([[0]]), "fps": 24, "attention_sparsity": 1.0}
    return video, audio, context, payload


class MiniMaxH3GroupedMaskingTests(unittest.TestCase):
    def setUp(self):
        self.attention_state = attention_shared_state("sdpa")
        self.attention_state.__enter__()
        self.addCleanup(self.attention_state.__exit__, None, None, None)

    def test_any_pixel_coverage_activates_the_complete_h3_cell(self):
        mask = torch.zeros((1, 1, 32, 32))
        mask[..., 31, 31] = 1.0

        latent_mask = _resize_video_mask(mask, (1, 2, 2), clip_length=1, temporal_ratio=1)
        actual = _snap_video_mask_to_patch_cells(latent_mask)

        torch.testing.assert_close(actual, torch.ones_like(actual))

    def test_temporal_mask_max_pools_every_frame_covered_by_each_latent(self):
        mask = torch.zeros((1, 34, 2, 2))
        mask[:, 0, 0, 0] = 1.0
        mask[:, 1, 0, 1] = 1.0
        mask[:, 4, 1, 0] = 1.0
        mask[:, 5, 1, 1] = 1.0
        mask[:, 8, 0, 0] = 1.0
        mask[:, 17, 1, 0] = 1.0
        mask[:, 18, 0, 0] = 1.0
        mask[:, 21, 0, 1] = 1.0

        actual = _resize_video_mask(mask, (10, 2, 2), clip_length=17, temporal_ratio=4)

        expected = torch.zeros((1, 1, 10, 2, 2))
        expected[:, :, 0, 0, 0] = 1.0
        expected[:, :, 1, 0, 1] = 1.0
        expected[:, :, 1, 1, 0] = 1.0
        expected[:, :, 2, 1, 1] = 1.0
        expected[:, :, 2, 0, 0] = 1.0
        expected[:, :, 5, 1, 0] = 1.0
        expected[:, :, 6, 0, 0] = 1.0
        expected[:, :, 6, 0, 1] = 1.0
        torch.testing.assert_close(actual, expected)

    def test_temporal_mask_pooling_includes_the_short_final_interval(self):
        mask = torch.zeros((1, 107, 1, 1))
        mask[:, 106] = 1.0

        actual = _resize_video_mask(mask, (32, 1, 1), clip_length=17, temporal_ratio=4)

        self.assertEqual(tuple(actual.shape), (1, 1, 32, 1, 1))
        torch.testing.assert_close(actual[:, :, :-1], torch.zeros_like(actual[:, :, :-1]))
        torch.testing.assert_close(actual[:, :, -1], torch.ones_like(actual[:, :, -1]))

    def test_mask_is_snapped_to_exact_patch_cells_without_extra_dilation(self):
        mask = torch.zeros((1, 1, 1, 8, 8))
        mask[..., 2, 2] = 1.0

        actual = _snap_video_mask_to_patch_cells(mask)

        expected = torch.zeros_like(mask)
        expected[..., 2:4, 2:4] = 1.0
        torch.testing.assert_close(actual, expected)

    def test_grouping_places_fixed_rows_first_and_builds_an_inverse(self):
        mask = torch.tensor([[[[[0.0, 0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0, 1.0],
                                [1.0, 1.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0, 0.0]]]]])
        payload = {}

        self.assertTrue(_set_grouped_video_rows(payload, mask, (1, 4, 4), torch.device("cpu")))

        torch.testing.assert_close(payload["target_video_order"], torch.tensor([0, 3, 1, 2]))
        torch.testing.assert_close(payload["target_video_inverse_order"], torch.tensor([0, 2, 3, 1]))
        self.assertEqual(payload["target_video_fixed_rows"], 2)

    def test_zero_length_masking_window_has_no_active_spatial_mask(self):
        mask = torch.ones((1, 1, 1, 2, 2))

        self.assertIsNone(_masking_step_mask(mask, step=0, denoising_start_step=0, mask_end_step=0))
        self.assertIs(_masking_step_mask(mask, step=1, denoising_start_step=1, mask_end_step=2), mask)
        self.assertIsNone(_masking_step_mask(mask, step=2, denoising_start_step=1, mask_end_step=2))

    def test_grouped_timestep_uses_two_scalar_adaln_segments(self):
        timestep = torch.tensor([0.2, 0.4])
        indices = torch.tensor([0, 1, 0, 0, 0, 0])

        timestep, indices, head_rows, block_rows = _grouped_video_timestep_rows(timestep, indices, 2, 4, 2, True)

        fixed_row = int((timestep == VISUAL_COND_TIMESTEP).nonzero()[0])
        current_row = int(indices[2])
        self.assertEqual(head_rows, [(0, 2, fixed_row), (2, 4, current_row)])
        self.assertEqual(block_rows, [(2, 4, fixed_row * 3), (4, 6, current_row * 3)])

    @torch.inference_mode()
    def test_inactive_grouping_is_permutation_equivalent_and_keeps_canonical_rope(self):
        model = _tiny_model()
        video, audio, context, native_payload = _inputs()
        sigma = torch.tensor([0.7])
        native_video, native_audio = model(video.clone(), audio.clone(), sigma, sigma, context, native_payload)
        grouped_payload = {key: value for key, value in native_payload.items() if key not in ("layout", "layout_signature", "rope")}
        order = torch.tensor([1, 4, 6, 0, 2, 3, 5, 7])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        grouped_payload.update(target_video_order=order, target_video_inverse_order=inverse,
                               target_video_fixed_rows=3, target_video_mask_active=False)

        grouped_video, grouped_audio = model(video.clone(), audio.clone(), sigma, sigma, context, grouped_payload)

        torch.testing.assert_close(grouped_video, native_video, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(grouped_audio, native_audio, rtol=1e-5, atol=1e-6)
        video_start = grouped_payload["layout"].sequence_length - order.numel()
        torch.testing.assert_close(grouped_payload["rope"][:, video_start:],
                                   native_payload["rope"][:, video_start:].index_select(1, order))

    @torch.inference_mode()
    def test_active_grouping_is_invariant_to_order_within_each_timestep_group(self):
        model = _tiny_model()
        video, audio, context, base_payload = _inputs()

        def run(order):
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel())
            payload = dict(base_payload, target_video_order=order, target_video_inverse_order=inverse,
                           target_video_fixed_rows=3, target_video_mask_active=True)
            return model(video.clone(), audio.clone(), torch.tensor([0.7]), torch.tensor([0.7]), context, payload)

        first_video, first_audio = run(torch.tensor([1, 4, 6, 0, 2, 3, 5, 7]))
        second_video, second_audio = run(torch.tensor([6, 4, 1, 7, 5, 3, 2, 0]))

        torch.testing.assert_close(first_video, second_video, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(first_audio, second_audio, rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_grouped_masking_rejects_sol_before_kernel_validation(self):
        model = _tiny_model()
        video, audio, context, payload = _inputs()
        order = torch.arange(8)
        payload.update(target_video_order=order, target_video_inverse_order=order,
                       target_video_fixed_rows=4, target_video_mask_active=True)

        with patch.dict(offload.shared_state, {"_attention": "sol"}):
            with self.assertRaisesRegex(ValueError, "not compatible with Sol Attention"):
                model(video, audio, torch.tensor([0.7]), torch.tensor([0.7]), context, payload)

    @torch.inference_mode()
    def test_active_grouped_masking_runs_with_dense_attention(self):
        model = _tiny_model()
        video, audio, context, payload = _inputs()
        order = torch.tensor([1, 4, 6, 0, 2, 3, 5, 7])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        payload.update(target_video_order=order, target_video_inverse_order=inverse,
                       target_video_fixed_rows=3, target_video_mask_active=True)

        output_video, output_audio = model(video, audio, torch.tensor([0.7]), torch.tensor([0.7]), context, payload)

        self.assertEqual(tuple(output_video.shape), tuple(video.shape))
        self.assertEqual(tuple(output_audio.shape), tuple(audio.shape))
        self.assertTrue(torch.isfinite(output_video).all())
        self.assertTrue(torch.isfinite(output_audio).all())

    def test_handler_exposes_outpainting_and_rejects_masked_sol(self):
        model_def = family_handler.query_model_def(FL2VA_ARCHITECTURE, {})
        self.assertEqual(model_def["video_guide_outpainting"], [0])
        self.assertEqual(model_def["outpainting_quantize_margins"], 32)
        inputs = {"sliding_window_overlap": 0, "override_attention": "sol", "video_prompt_type": "GVA",
                  "video_mask": object(), "video_guide_outpainting": "", "video_guide_outpainting_ratio": "",
                  "custom_settings": {H3_MASK_MODE_SETTING: H3_MASK_MODE_GROUPED_ROWS}}

        error = family_handler.validate_generative_settings(FL2VA_ARCHITECTURE, model_def, inputs)

        self.assertIn("not compatible with Sol Attention", error)
        inputs.update(audio_prompt_type="", custom_settings={H3_MASK_MODE_SETTING: H3_MASK_MODE_DEFAULT})
        self.assertIsNone(family_handler.validate_generative_settings(FL2VA_ARCHITECTURE, model_def, inputs))

    def test_handler_exposes_g_only_mask_mode_with_legacy_default(self):
        model_def = family_handler.query_model_def(FL2VA_ARCHITECTURE, {})
        setting = model_def["custom_settings"][0]

        self.assertEqual(setting["id"], H3_MASK_MODE_SETTING)
        self.assertEqual(setting["default"], H3_MASK_MODE_DEFAULT)
        self.assertEqual(setting["video_prompt_type"], "G")
        self.assertEqual([label for label, _ in setting["choices"]], [
            "Shared Timestep [same denoising timestep for fixed and editable latent rows]",
            "Grouped Rows [conditioning timestep for fixed rows; denoising timestep for editable rows]",
        ])
        self.assertEqual([value for _, value in setting["choices"]], ["shared_timestep", "grouped_rows"])

    def test_missing_mask_mode_uses_shared_timestep(self):
        self.assertFalse(h3_grouped_masking_enabled(None))
        self.assertFalse(h3_grouped_masking_enabled({}))

    def test_ref2va_generic_control_exposes_masking_without_enabling_it_for_references(self):
        model_def = family_handler.query_model_def("minimax_h3_ref2va", {})
        self.assertIn(("Provide Generic Control Video", "GV"), model_def["guide_custom_choices"]["choices"])
        self.assertIn(("Use One Reference Video", "V-U"), model_def["guide_custom_choices"]["choices"])
        self.assertIn(("Use Two Reference Videos", "V+-U"), model_def["guide_custom_choices"]["choices"])
        self.assertEqual(model_def["mask_preprocessing"]["selection"], ["", "A", "NA"])
        mask_preprocessing = model_def["mask_preprocessing"]

        self.assertTrue(video_mask_dropdown_visible(mask_preprocessing, "GV"))
        self.assertTrue(video_mask_area_visible("GVA"))
        self.assertFalse(video_mask_dropdown_visible(mask_preprocessing, "VU"))
        self.assertFalse(video_mask_area_visible("VAU"))

    def test_old_ref2va_generic_control_settings_receive_the_g_flag(self):
        settings = {"video_prompt_type": "VA"}

        family_handler.fix_settings("minimax_h3_ref2va", 2.75, {}, settings)

        self.assertEqual(settings["video_prompt_type"], "GVA")

    def test_old_ref2va_reference_settings_receive_the_u_flag(self):
        for old_value, expected in (("V-", "V-U"), ("V+-", "V+-U")):
            with self.subTest(old_value=old_value):
                settings = {"video_prompt_type": old_value}

                family_handler.fix_settings("minimax_h3_ref2va", 2.75, {}, settings)

                self.assertEqual(settings["video_prompt_type"], expected)

    def test_ref2va_generic_control_is_not_validated_as_a_reference_video(self):
        model_def = family_handler.query_model_def("minimax_h3_ref2va", {})
        inputs = {"sliding_window_overlap": 18, "override_attention": "sdpa", "video_prompt_type": "GVA",
                  "video_mask": object(), "video_guide_outpainting": "", "video_guide_outpainting_ratio": "",
                  "resolution": "832x480", "audio_prompt_type": "", "image_refs": [],
                  "video_guide": "not-a-reference-video", "video_guide2": None, "audio_guide": None, "audio_guide2": None}

        self.assertIsNone(family_handler.validate_generative_settings("minimax_h3_ref2va", model_def, inputs))

    def test_outpainting_builds_an_editable_margin_mask(self):
        video = torch.zeros((3, 5, 64, 96))

        mask = _build_outpainting_mask(video, [0, 0, 50, 50])

        self.assertEqual(tuple(mask.shape), (1, 5, 64, 96))
        torch.testing.assert_close(mask[..., 32:64], torch.zeros_like(mask[..., 32:64]))
        torch.testing.assert_close(mask[..., :32], torch.ones_like(mask[..., :32]))
        torch.testing.assert_close(mask[..., 64:], torch.ones_like(mask[..., 64:]))


if __name__ == "__main__":
    unittest.main()
