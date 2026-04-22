# Wan Runtime Drift vs Upstream

## Title
Wan Runtime Drift vs Upstream

## Summary
Audit status: no PR-ready additive sync remains in the current submodule for this theme. Use this draft as a maintainer review packet before opening any upstream work, and treat the three symbol-level items called out in `README.md` as investigation follow-ups rather than ready PRs.

## Affected Files
- `Wan2GP/models/wan/__init__.py`
- `Wan2GP/models/wan/animate/face_blocks.py`
- `Wan2GP/models/wan/any2video.py`
- `Wan2GP/models/wan/diffusion_forcing.py`
- `Wan2GP/models/wan/modules/model.py`
- `Wan2GP/models/wan/modules/t5.py`
- `Wan2GP/models/wan/multitalk/attention.py`
- `Wan2GP/models/wan/multitalk/multitalk.py`
- `Wan2GP/models/wan/multitalk/multitalk_utils.py`
- `Wan2GP/models/wan/ovi_fusion_engine.py`
- `Wan2GP/models/wan/ovi_handler.py`
- `Wan2GP/models/wan/scail/model_scail.py`
- `Wan2GP/models/wan/wan_handler.py`

## Verification
- `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`
- Maintainer-directed upstream smoke against `deepbeepmeep/Wan2GP` after a fresh diff review for this theme.

## Review Pointer
- Re-check the matching `docs/wan2gp-triage.csv` rows for this literal theme string against current upstream head before filing anything.
