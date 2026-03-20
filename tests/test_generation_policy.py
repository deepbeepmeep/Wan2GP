"""Regression tests for semantic continuation flags on GenerationPolicy."""

from __future__ import annotations

from source.core.params.generation_policy import GenerationPolicy


def _payload_for(strategy: str, *, model_type: str = "i2v") -> dict[str, object]:
    return {
        "model_type": model_type,
        "continuation_config": {
            "strategy": strategy,
            "overlap_frames": 4,
        },
    }


def test_prefix_video_source_sets_semantic_flag():
    policy = GenerationPolicy.from_payload(_payload_for("prefix_video_source"))

    assert policy.continuation.enabled is True
    assert policy.continuation.requires_video_source is True
    assert policy.continuation.uses_svi_latent_chaining is False


def test_svi_latent_chaining_sets_semantic_flag():
    policy = GenerationPolicy.from_payload(_payload_for("svi_latent_chaining"))

    assert policy.continuation.enabled is True
    assert policy.continuation.uses_svi_latent_chaining is True
    assert policy.continuation.requires_video_source is True


def test_guide_overlap_masked_keeps_existing_semantics_and_new_flags_false():
    policy = GenerationPolicy.from_payload(_payload_for("guide_overlap_masked", model_type="vace"))

    assert policy.continuation.enabled is True
    assert policy.continuation.uses_guide_for_overlap is True
    assert policy.continuation.uses_mask_video is True
    assert policy.continuation.requires_video_source is False
    assert policy.continuation.uses_svi_latent_chaining is False


def test_disabled_continuation_keeps_all_flags_false():
    policy = GenerationPolicy.from_payload({"model_type": "i2v"})

    assert policy.continuation.enabled is False
    assert policy.continuation.strategy == "none"
    assert policy.continuation.requires_video_source is False
    assert policy.continuation.uses_svi_latent_chaining is False
    assert policy.continuation.uses_guide_for_overlap is False


def test_legacy_vace_fallback_uses_overlap_guide_only():
    policy = GenerationPolicy.from_payload({"model_type": "vace"})

    assert policy.continuation.enabled is True
    assert policy.continuation.strategy == "guide_overlap_masked"
    assert policy.continuation.uses_guide_for_overlap is True
    assert policy.continuation.requires_video_source is False
    assert policy.continuation.uses_svi_latent_chaining is False
