# TTS Implementation Drift vs Upstream

## Title
TTS Implementation Drift vs Upstream

## Summary
Audit status: no PR-ready additive sync remains in the current submodule for this theme. Use this draft as a maintainer review packet before opening any upstream work, and treat the three symbol-level items called out in `README.md` as investigation follow-ups rather than ready PRs.

## Affected Files
- `Wan2GP/models/TTS/ace_step15/models/modeling_acestep_v15_turbo.py`
- `Wan2GP/models/TTS/ace_step15/pipeline_ace_step15.py`
- `Wan2GP/models/TTS/ace_step_handler.py`
- `Wan2GP/models/TTS/chatterbox_handler.py`
- `Wan2GP/models/TTS/heartmula_handler.py`
- `Wan2GP/models/TTS/index_tts2/accel/accel_engine.py`
- `Wan2GP/models/TTS/index_tts2/pipeline.py`
- `Wan2GP/models/TTS/index_tts2_handler.py`
- `Wan2GP/models/TTS/kugelaudio_handler.py`
- `Wan2GP/models/TTS/qwen3_handler.py`
- `Wan2GP/models/TTS/yue_handler.py`

## Verification
- `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`
- Maintainer-directed upstream smoke against `deepbeepmeep/Wan2GP` after a fresh diff review for this theme.

## Review Pointer
- Re-check the matching `docs/wan2gp-triage.csv` rows for this literal theme string against current upstream head before filing anything.
