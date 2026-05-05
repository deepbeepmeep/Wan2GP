# WGP Self Repeatability Report: wgp-self-repeat-0b-2026-05-05-deferral

- Threshold version: `0B-2026-05-05`
- Mode: `deferral`
- Created at: `2026-05-05T22:08:45.533342+00:00`
- Route count: 14

## Commands
- `python scripts/live_test/main.py --variant fresh --wgp-profile 3 --timeout-image 3600 --timeout-travel-segment 1800 --timeout-travel-orchestrator 3600`
- `python scripts/live_test/main.py --variant fresh --wgp-profile 3 --timeout-image 3600 --timeout-travel-segment 1800 --timeout-travel-orchestrator 3600`

## Routes
- `z_image_turbo`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `z_image_turbo_i2i`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `qwen_image`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `qwen_image_2512`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `wan_2_2_t2i`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `qwen_image_edit`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `qwen_image_style`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `image_inpaint`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `annotated_image_edit`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `individual_travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)
- `travel_segment__model-ltx2_distilled__guidance-ltx_anchor__continuity-video_source__profile-default`: not_evaluated_no_metric_observations (calibration: `deferred_pending_sprint_0c_disk`)

## Deferral
- Blocker: Paired WGP-vs-WGP calibration was not required as a Sprint 0B stop-the-line gate after the user correction that WGP is already trusted. Live RunPod infrastructure was separately verified by the agent on 2026-05-05 with pod `f9s5vqk15gux9d`: launch, SSH readiness, GPU visibility, storage health, pod listing, termination, and post-terminate absence all passed. This report remains a route-keyed deferral because it contains no paired WGP metric observations.
- Next action: In Sprint 0C or the first sprint that consumes measured WGP drift, run paired WGP self-repeatability only if fresh calibration is needed for threshold promotion; otherwise keep WGP as the trusted control and replace `deferred_pending_sprint_0c_disk` statuses only when measured reports contain metric observations.
