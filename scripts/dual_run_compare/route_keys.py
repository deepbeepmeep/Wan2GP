from __future__ import annotations

import re
from typing import Any, Mapping


DIRECT_ROUTE_ALIASES: dict[str, str] = {
    "z_image": "z_image_turbo",
    "z_image_turbo": "z_image_turbo",
    "z_image_turbo_i2i": "z_image_turbo_i2i",
    "qwen_image": "qwen_image",
    "qwen_image_2512": "qwen_image_2512",
    "optimised_t2i": "wan_2_2_t2i",
    "wan_2_2_t2i": "wan_2_2_t2i",
    "qwen_image_edit": "qwen_image_edit",
    "qwen_image_style": "qwen_image_style",
    "image_inpaint": "image_inpaint",
    "annotated_image_edit": "annotated_image_edit",
}

EDIT_VARIANT_ALIASES: dict[str, str] = {
    "qwen-edit": "qwen_edit_default",
    "qwen_edit": "qwen_edit_default",
    "qwen_edit_default": "qwen_edit_default",
    "qwen-edit-2509": "qwen_edit_2509",
    "qwen_edit_2509": "qwen_edit_2509",
    "qwen-edit-2511": "qwen_edit_2511",
    "qwen_edit_2511": "qwen_edit_2511",
    "style-reference": "style_reference",
    "style_reference": "style_reference",
    "mask": "mask",
    "annotation": "annotation",
}

COHORT_B_EDIT_DIMENSION_FIELDS: tuple[str, ...] = (
    "edit_variant",
    "qwen_edit_model",
    "mask_case",
    "mask_type",
    "annotation_case",
    "annotation_type",
    "style_reference_case",
    "style_reference_type",
    "profile",
    "wgp_profile",
)


def slug(value: Any) -> str:
    """Normalize route dimensions into stable lowercase token fragments."""

    text = str(value or "none").strip().lower()
    text = text.replace("+", "_plus_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "none"


def direct_route_key(task_type_or_alias: str) -> str:
    """Canonical direct Cohort A/B product route key.

    Direct routes key only by production task type because the selector can safely
    flip these routes without inspecting model/guidance dimensions.
    """

    key = slug(task_type_or_alias)
    return DIRECT_ROUTE_ALIASES.get(key, key)


def edit_route_key(
    task_type_or_alias: str,
    *,
    edit_variant: str | None = None,
    mask_case: str | None = None,
    annotation_case: str | None = None,
    style_reference_case: str | None = None,
    profile: str | int | None = None,
) -> str:
    """Canonical dimensional key for Cohort B edit variants when dimensions exist."""

    task_type = direct_route_key(task_type_or_alias)
    parts = [task_type]
    if edit_variant:
        variant = EDIT_VARIANT_ALIASES.get(slug(edit_variant), slug(edit_variant))
        parts.append(f"variant-{variant}")
    if mask_case:
        parts.append(f"mask-{slug(mask_case)}")
    if annotation_case:
        parts.append(f"annotation-{slug(annotation_case)}")
    if style_reference_case:
        parts.append(f"style_reference-{slug(style_reference_case)}")
    if profile is not None:
        parts.append(f"profile-{slug(profile)}")
    return "__".join(parts)


def model_family_from_model_name(model_name: str | None) -> str:
    """Map audited worker model ids to route-key model families.

    The Wan 2.2 I2V baseline contains "lightning" and "baseline" but is not a
    VACE model. Only model ids containing the actual "vace" token map to VACE.
    """

    normalized = slug(model_name)
    if not normalized or normalized == "none":
        return "unknown"
    if "wan_2_2" in normalized or "wan22" in normalized:
        return "wan22_vace" if "vace" in normalized else "wan22_i2v"
    if "ltx2" in normalized:
        return "ltx2_distilled" if "distilled" in normalized else "ltx2"
    if "qwen" in normalized:
        return "qwen"
    if "z_image" in normalized:
        return "z_image"
    return normalized


def cohort_e_route_key(
    *,
    task_type: str,
    model_name: str | None = None,
    model_family: str | None = None,
    guidance_kind: str | None = "none",
    guidance_mode: str | None = None,
    continuity_case: str | None = "first_last",
    profile: str | int | None = "default",
) -> str:
    """Canonical dimensional Cohort E route key.

    Cohort E keys always include task type, model family, guidance kind,
    continuity case, and profile because parent/child route propagation depends
    on these dimensions.
    """

    family = slug(model_family or model_family_from_model_name(model_name))
    guidance = cohort_e_guidance_key(guidance_kind=guidance_kind, guidance_mode=guidance_mode)
    return "__".join(
        [
            slug(task_type),
            f"model-{family}",
            f"guidance-{slug(guidance)}",
            f"continuity-{slug(continuity_case)}",
            f"profile-{slug(profile)}",
        ]
    )


def cohort_e_guidance_key(
    *,
    guidance_kind: str | None = "none",
    guidance_mode: str | None = None,
) -> str:
    """Return the route guidance dimension, preserving mode where it affects routing."""

    kind = slug(guidance_kind)
    mode = slug(guidance_mode)
    if kind in {"vace", "ltx_control"} and mode != "none":
        return f"{kind}_{mode}"
    return kind


def route_key_from_payload(payload: Mapping[str, Any]) -> str:
    """Best-effort canonical route key from a task payload or fixture mapping."""

    task_type = str(payload.get("task_type") or payload.get("type") or "")
    cohort = payload.get("cohort")
    travel_guidance = payload.get("travel_guidance")
    travel_guidance_kind = travel_guidance.get("kind") if isinstance(travel_guidance, Mapping) else None
    travel_guidance_mode = travel_guidance.get("mode") if isinstance(travel_guidance, Mapping) else None
    if cohort == "E" or task_type in {
        "travel_orchestrator",
        "travel_segment",
        "individual_travel_segment",
        "join_clips_segment",
        "join_clips_orchestrator",
        "join_final_stitch",
        "travel_stitch",
        "edit_video_orchestrator",
    }:
        return cohort_e_route_key(
            task_type=task_type,
            model_name=payload.get("model_name") or payload.get("model"),
            model_family=payload.get("model_family"),
            guidance_kind=(
                payload.get("guidance_kind")
                or payload.get("travel_guidance_kind")
                or travel_guidance_kind
                or "none"
            ),
            guidance_mode=(
                payload.get("guidance_mode")
                or payload.get("travel_guidance_mode")
                or travel_guidance_mode
            ),
            continuity_case=payload.get("continuity_case") or "first_last",
            profile=payload.get("profile") or payload.get("wgp_profile") or "default",
        )
    if any(payload.get(field) for field in COHORT_B_EDIT_DIMENSION_FIELDS):
        return edit_route_key(
            task_type,
            edit_variant=payload.get("edit_variant") or payload.get("qwen_edit_model"),
            mask_case=payload.get("mask_case") or payload.get("mask_type"),
            annotation_case=payload.get("annotation_case") or payload.get("annotation_type"),
            style_reference_case=payload.get("style_reference_case") or payload.get("style_reference_type"),
            profile=payload.get("profile") or payload.get("wgp_profile"),
        )
    return direct_route_key(task_type)
