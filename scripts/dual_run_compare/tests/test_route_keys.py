from __future__ import annotations

from scripts.dual_run_compare.route_keys import (
    cohort_e_route_key,
    direct_route_key,
    edit_route_key,
    model_family_from_model_name,
    route_key_from_payload,
    slug,
)


def test_direct_cohort_a_b_aliases() -> None:
    assert direct_route_key("z-image") == "z_image_turbo"
    assert direct_route_key("qwen-image-2512") == "qwen_image_2512"
    assert direct_route_key("optimised-t2i") == "wan_2_2_t2i"
    assert direct_route_key("image_inpaint") == "image_inpaint"


def test_cohort_b_edit_dimensions() -> None:
    assert (
        edit_route_key("qwen_image_edit", edit_variant="qwen-edit-2511", profile=3)
        == "qwen_image_edit__variant-qwen_edit_2511__profile-3"
    )
    assert (
        edit_route_key(
            "qwen_image_style",
            edit_variant="style-reference",
            style_reference_case="image prompt style",
            profile="profile 2",
        )
        == "qwen_image_style__variant-style_reference__style_reference-image_prompt_style__profile-profile_2"
    )
    assert (
        edit_route_key("image_inpaint", edit_variant="mask", mask_case="alpha mask", profile=5)
        == "image_inpaint__variant-mask__mask-alpha_mask__profile-5"
    )
    assert (
        edit_route_key("annotated_image_edit", edit_variant="annotation", annotation_case="drawn arrows")
        == "annotated_image_edit__variant-annotation__annotation-drawn_arrows"
    )
    assert (
        route_key_from_payload(
            {
                "task_type": "image_inpaint",
                "edit_variant": "mask",
                "mask_case": "binary upload",
                "profile": "profile 5",
            }
        )
        == "image_inpaint__variant-mask__mask-binary_upload__profile-profile_5"
    )


def test_cohort_e_dimensional_key_shape() -> None:
    key = cohort_e_route_key(
        task_type="individual_travel_segment",
        model_name="wan_2_2_i2v_lightning_baseline_2_2_2",
        guidance_kind="none",
        continuity_case="first_last",
        profile="default",
    )
    assert (
        key
        == "individual_travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default"
    )


def test_slug_normalization_and_determinism() -> None:
    assert slug("  LTX Anchor + Audio  ") == "ltx_anchor_plus_audio"
    first = cohort_e_route_key(
        task_type="Travel Segment",
        model_family="Wan22 VACE",
        guidance_kind="VACE",
        continuity_case="Video Source",
        profile="Default",
    )
    second = cohort_e_route_key(
        task_type="Travel Segment",
        model_family="Wan22 VACE",
        guidance_kind="VACE",
        continuity_case="Video Source",
        profile="Default",
    )
    assert first == second
    assert first == "travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default"


def test_model_family_i2v_is_not_vace() -> None:
    assert model_family_from_model_name("wan_2_2_i2v_lightning_baseline_2_2_2") == "wan22_i2v"
    assert model_family_from_model_name("wan_2_2_vace_lightning_baseline_2_2_2") == "wan22_vace"
    assert "vace" not in cohort_e_route_key(
        task_type="travel_segment",
        model_name="wan_2_2_i2v_lightning_baseline_2_2_2",
    )


def test_ltx_model_family_mapping() -> None:
    assert model_family_from_model_name("ltx2_22B") == "ltx2"
    assert model_family_from_model_name("ltx2_22B_distilled_1_1") == "ltx2_distilled"
