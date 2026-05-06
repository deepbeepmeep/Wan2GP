# Sprint 3.5 Wan 2.2 VACE Dry-Run Report

- Decision: `FALL-BACK`
- Decision date: `2026-05-06`
- Selected route key: `travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default`
- Threshold YAML version: `0B-2026-05-05`
- Fixture manifest: `scripts/dual_run_compare/fixtures/sprint35/manifest.json`
- Ready template id: `wanvideo_wrapper_22_14b_vace_cocktail_dry_run`
- Ready template path: `../vibecomfy/ready_templates/video/wanvideo_wrapper_22_14b_vace_cocktail_dry_run.py`

## Gate Decision

`PROCEED` is allowed only if every required Section 11 video threshold is green. This dry run does not meet that bar because the live VibeComfy candidate failed Comfy validation before producing a video, so the required candidate video observations are missing. Missing required observations are not green threshold results.

The WGP 49-frame reference was produced successfully. The VibeComfy queue-validation failure is treated as legitimate Sprint 3.5 feasibility evidence, not as an infrastructure pause. The fallback is scoped to Wan-family VACE travel/join routes; non-Wan migration work continues.

## Artifact Status

| Artifact | Path / id | Status |
| --- | --- | --- |
| WGP 49-frame reference video | `scripts/dual_run_compare/artifacts/sprint35/wgp_reference/20260506T044342Z/outputs/2026-05-06-04h54m05s_seed12345_A steady cinematic travel shot through a glass-roof train station at golden hour, camera gliding for.mp4` | Produced on RunPod pod `eferweepqfsu5h`; ffprobe reports `832x480`, `16 fps`, `49` frames, duration `3.062500s`, SHA-256 `66d0b7a2470666c62a9c5bd6b9aa619ef9c27bab2b3a12924cd8312f4c793824`. |
| WGP run metadata | `scripts/dual_run_compare/artifacts/sprint35/wgp_reference/20260506T044342Z/run_metadata.json` and `scripts/dual_run_compare/artifacts/sprint35/wgp_reference/20260506T043642Z/runpod_metadata.json` | `status=success`; route, model, seed, resolution, fps, and 49-frame target match the fixture. |
| VibeComfy 49-frame candidate video | not produced | Live RunPod candidate failed Comfy workflow validation before queue execution produced an output. |
| VibeComfy runtime metadata | `scripts/dual_run_compare/artifacts/sprint35/vibecomfy_candidate/20260506T053546Z/run_metadata.json`, `scripts/dual_run_compare/artifacts/sprint35/vibecomfy_candidate/20260506T053546Z/exception.txt`, `scripts/dual_run_compare/artifacts/sprint35/vibecomfy_candidate/20260506T050302Z/runpod_metadata.json` | Pod `j4u2fcet62cnux`; `status=terminated_after_candidate_queue_failure`; no Comfy prompt id and no selected video. |
| VibeComfy model staging notes | `scripts/dual_run_compare/artifacts/sprint35/vibecomfy_candidate/staging/model_staging.json` | Staged the Sprint 3.5 fixture, dry-run template, WanVideoWrapper custom-node surface, and required Wan VACE model assets. |
| Shared fixture bundle | `scripts/dual_run_compare/fixtures/sprint35/` | Present and locally validated. |
| Local dry-run template | `../vibecomfy/ready_templates/video/wanvideo_wrapper_22_14b_vace_cocktail_dry_run.py` | Present and locally validated. |

## Run Scope

- WGP RunPod pod id: `eferweepqfsu5h`
- WGP run label: `20260506T043642Z`
- WGP output run dir: `scripts/dual_run_compare/artifacts/sprint35/wgp_reference/20260506T044342Z/`
- VibeComfy RunPod pod id: `j4u2fcet62cnux`
- VibeComfy run label: `20260506T050302Z`
- VibeComfy candidate metadata run dir: `scripts/dual_run_compare/artifacts/sprint35/vibecomfy_candidate/20260506T053546Z/`
- Focused VibeComfy matrix scope: `sprint35_wan_vace_dry_run`
- Focused ready template: `wanvideo_wrapper_22_14b_vace_cocktail_dry_run`
- Fixture source: `scripts/dual_run_compare/fixtures/sprint35/manifest.json`
- Comfy prompt id: not available; validation failed before prompt execution.
- RunPod command/run id: WGP used the focused Sprint 3.5 WGP RunPod wrapper retry with `tar --no-same-owner`; VibeComfy used the focused Sprint 3.5 candidate run from `run_sprint35_vibecomfy_candidate.py` staged in the downloaded artifact archive.

## Model And Schedule Evidence

WGP default evidence comes from `Wan2GP/defaults/wan_2_2_vace_lightning_baseline_2_2_2.json`, the Sprint 3.5 fixture manifest, and the local WGP Euler scheduling path:

| Field | Value |
| --- | --- |
| Model/default id | `wan_2_2_vace_lightning_baseline_2_2_2` |
| Model name | `Wan2.2 Vace Fun Cocktail Lightning 14B (3-Phase) 2-2-2 Steps - Baseline (Lightning Only)` |
| Architecture | `vace_14B` |
| Seed | `12345` |
| Resolution | `832x480` |
| FPS | `16` |
| Frame target | `49` |
| Solver | `euler` |
| Flow shift | `5` |
| Total inference steps | `6` |
| Guidance phases | `3` |
| Guidance scales | `3 / 1 / 1` |
| Switch thresholds | `883 / 558` |
| Model switch phase | `2` |
| Derived phase allocation | `2 / 2 / 2` steps |
| Explicit sampler topology | `HIGH -> HIGH -> LOW` |

The local dry-run template compiles to two `WanVideoModelLoader` nodes and three chained `WanVideoSampler` nodes. The first two samplers use the HIGH model, the final sampler uses the LOW model, and the LOW sampler feeds `WanVideoDecode`.

## Metric Gate

The required video metrics were loaded through `Thresholds.load(strict=True)` and evaluated through `compare_route_observations(..., required_metric_keys=...)` in T8. `compare_video_artifacts()` could not compute a numeric pair because the VibeComfy candidate artifact is missing.

| Metric | Threshold source | Observation | Gate status |
| --- | --- | --- | --- |
| `video_frame_count` | `migration-thresholds.yaml` exact match | missing candidate observation | RED: required observation missing |
| `video_phash_mean` | `migration-thresholds.yaml` max `0.08` normalized Hamming | missing candidate observation | RED: required observation missing |
| `video_phash_p95` | `migration-thresholds.yaml` max `0.12` normalized Hamming | missing candidate observation | RED: required observation missing |
| `video_duration_ms` | `migration-thresholds.yaml` tolerance `50 ms` | missing candidate observation | RED: required observation missing |
| `video_fps` | `migration-thresholds.yaml` exact match | missing candidate observation | RED: required observation missing |

`video_audio_duration_ms` is not part of this Sprint 3.5 hard gate because the selected segment fixture has no applicable audio observation. Runtime, VRAM, and human visual notes are context-only; they cannot override the required video threshold result.

## Local Validation

Completed local validation before this report:

```text
cd reigh-worker && python -m scripts.dual_run_compare.check_thresholds --strict
cd vibecomfy && python -m vibecomfy.cli validate ready_templates/video/wanvideo_wrapper_22_14b_vace_cocktail_dry_run.py
cd vibecomfy && python -m pytest -q tests/test_ready_templates.py
cd vibecomfy && python -m pytest -q tests/test_runpod_matrix.py
```

Additional T8 threshold-path probe:

```text
compare_video_artifacts(wgp_reference_video, vibecomfy_candidate_video)
Thresholds.load(strict=True)
compare_route_observations(thresholds, route_key, observations, required_metric_keys=[
  "video_frame_count",
  "video_phash_mean",
  "video_phash_p95",
  "video_duration_ms",
  "video_fps",
])
```

Result: the strict threshold path found missing required observations for all five required video metric keys because the VibeComfy candidate video was not produced. The report decision is therefore `FALL-BACK`.
