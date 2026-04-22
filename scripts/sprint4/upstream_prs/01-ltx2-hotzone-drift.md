# LTX2 Hot-Zone Drift vs Upstream

## Title
LTX2 Hot-Zone Drift vs Upstream

## Summary
Audit status: no PR-ready additive sync remains in the current submodule for this theme. Use this draft as a maintainer review packet before opening any upstream work, and treat the three symbol-level items called out in `README.md` as investigation follow-ups rather than ready PRs.

## Affected Files
- `Wan2GP/models/ltx2/ltx2.py`
- `Wan2GP/models/ltx2/ltx2_handler.py`
- `Wan2GP/models/ltx2/ltx_core/components/diffusion_steps.py`
- `Wan2GP/models/ltx2/ltx_core/components/guiders.py`
- `Wan2GP/models/ltx2/ltx_core/conditioning/__init__.py`
- `Wan2GP/models/ltx2/ltx_core/conditioning/types/__init__.py`
- `Wan2GP/models/ltx2/ltx_core/conditioning/types/keyframe_cond.py`
- `Wan2GP/models/ltx2/ltx_core/conditioning/types/latent_cond.py`
- `Wan2GP/models/ltx2/ltx_core/conditioning/types/reference_video_cond.py`
- `Wan2GP/models/ltx2/ltx_core/model/transformer/attention.py`
- `Wan2GP/models/ltx2/ltx_core/model/transformer/modality.py`
- `Wan2GP/models/ltx2/ltx_core/model/transformer/rope.py`
- `Wan2GP/models/ltx2/ltx_core/model/transformer/transformer.py`
- `Wan2GP/models/ltx2/ltx_core/model/transformer/transformer_args.py`
- `Wan2GP/models/ltx2/ltx_core/types.py`
- `Wan2GP/models/ltx2/ltx_core/utils.py`
- `Wan2GP/models/ltx2/ltx_pipelines/distilled.py`
- `Wan2GP/models/ltx2/ltx_pipelines/ti2vid_one_stage.py`
- `Wan2GP/models/ltx2/ltx_pipelines/ti2vid_two_stages.py`
- `Wan2GP/models/ltx2/ltx_pipelines/utils/helpers.py`

## Verification
- `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`
- Maintainer-directed upstream smoke against `deepbeepmeep/Wan2GP` after a fresh diff review for this theme.

## Review Pointer
- Re-check the matching `docs/wan2gp-triage.csv` rows for this literal theme string against current upstream head before filing anything.
