# Preset and Config Drift vs Upstream

## Title
Preset and Config Drift vs Upstream

## Summary
Audit status: no PR-ready additive sync remains in the current submodule for this theme. Use this draft as a maintainer review packet before opening any upstream work, and treat the three symbol-level items called out in `README.md` as investigation follow-ups rather than ready PRs.

## Affected Files
- `Wan2GP/defaults/i2v_2_2_Enhanced_Lightning_v2.json`
- `Wan2GP/defaults/index_tts2.json`
- `Wan2GP/defaults/kugelaudio_0_open.json`
- `Wan2GP/defaults/ltx2_19B.json`
- `Wan2GP/defaults/ltx2_19B_nvfp4.json`
- `Wan2GP/defaults/ltx2_22B.json`
- `Wan2GP/defaults/ltx2_22B_distilled.json`
- `Wan2GP/defaults/ltx2_22B_distilled_gguf_q4_k_m.json`
- `Wan2GP/defaults/ltx2_22B_distilled_gguf_q6_k.json`
- `Wan2GP/defaults/ltx2_22B_distilled_gguf_q8_0.json`
- `Wan2GP/defaults/ltx2_distilled.json`
- `Wan2GP/defaults/ltx2_distilled_gguf_q4_k_m.json`
- `Wan2GP/defaults/ltx2_distilled_gguf_q6_k.json`
- `Wan2GP/defaults/ltx2_distilled_gguf_q8_0.json`
- `Wan2GP/defaults/qwen3_tts_base.json`

## Verification
- `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`
- Maintainer-directed upstream smoke against `deepbeepmeep/Wan2GP` after a fresh diff review for this theme.

## Review Pointer
- Re-check the matching `docs/wan2gp-triage.csv` rows for this literal theme string against current upstream head before filing anything.
