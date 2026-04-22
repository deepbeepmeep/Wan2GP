# Shared Runtime Drift vs Upstream

## Title
Shared Runtime Drift vs Upstream

## Summary
Audit status: no PR-ready additive sync remains in the current submodule for this theme. Use this draft as a maintainer review packet before opening any upstream work, and treat the three symbol-level items called out in `README.md` as investigation follow-ups rather than ready PRs.

## Affected Files
- `Wan2GP/shared/api.py`
- `Wan2GP/shared/attention.py`
- `Wan2GP/shared/kernels/quanto_int8_inject.py`
- `Wan2GP/shared/model_dropdowns.py`
- `Wan2GP/shared/prompt_enhancer/qwen35_text.py`
- `Wan2GP/shared/prompt_enhancer/qwen35_vl.py`
- `Wan2GP/shared/qtypes/gguf.py`
- `Wan2GP/shared/qtypes/nvfp4.py`
- `Wan2GP/shared/sage2_core.py`

## Verification
- `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`
- Maintainer-directed upstream smoke against `deepbeepmeep/Wan2GP` after a fresh diff review for this theme.

## Review Pointer
- Re-check the matching `docs/wan2gp-triage.csv` rows for this literal theme string against current upstream head before filing anything.
