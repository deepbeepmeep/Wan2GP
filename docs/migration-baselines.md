# Migration Baselines — Sprint 0A Kickoff

> Author: Sprint 0A executor  
> Date: 2026-05-05  
> Status: Draft — pending human review (U3)

---

## 1. RayWorker-Owned USED Task Inventory

Validated against three sources of truth:
- **RayWorker dispatch:** `reigh-worker/source/task_handlers/tasks/task_registry.py:1442-1511` (specialized handlers) + `task_registry.py:1437-1438` (direct queue path via `DIRECT_QUEUE_TASK_TYPES`)
- **RayWorker catalog:** `reigh-worker/source/task_handlers/tasks/task_types.py:82-117` (`TASK_TYPE_TO_MODEL`) and `task_types.py:120-138` (`TASK_TYPE_CATALOG`)
- **Resolver registry:** `reigh-app/supabase/functions/create-task/resolvers/registry.ts:16-30` (maps task family → resolver)

### 1.1 USED-IN-APP (12 types)

Task types that have a live resolver entry in `registry.ts` and are emitted by active app code paths.

| # | task_type | Resolver family | Dispatch path | Direct queue? | Status | Notes |
|---|-----------|------------------|---------------|---------------|--------|-------|
| 1 | `wan_2_2_t2i` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable** | Forces `video_length=1`; Wan 2.2 T2I via WGP. |
| 2 | `qwen_image` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable** | Text-to-image Qwen path. |
| 3 | `qwen_image_2512` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable / VibeComfy proven** | Qwen 2512 model; live worker proof via `image/qwen_image_2512` on 2026-05-07. |
| 4 | `z_image_turbo` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable** | Z Image Turbo via WGP. |
| 5 | `z_image_turbo_i2i` | `z_image_turbo_i2i` | Direct queue → `_handle_direct_queue_task` | YES | **runnable** | Z Image img2img via WGP. |
| 6 | `qwen_image_edit` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable / VibeComfy proven** | Empty prompt allowed; live worker proof via `edit/qwen_image_edit` on 2026-05-07. |
| 7 | `qwen_image_style` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable / VibeComfy proven** | Rewrites prompt/model during conversion; live worker proof via `edit/qwen_image_edit` on 2026-05-07. |
| 8 | `image_inpaint` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable / VibeComfy proven** | Empty prompt allowed; worker creates masked composite before `edit/qwen_image_edit`. |
| 9 | `annotated_image_edit` | `image_generation` | Direct queue → `_handle_direct_queue_task` | YES | **runnable / VibeComfy proven** | Empty prompt allowed; worker creates annotated/masked composite before `edit/qwen_image_edit`. |
| 10 | `travel_orchestrator` | `travel_between_images` | Specialized handler → `handle_travel_orchestrator_task` | NO | **runnable** | Orchestrates travel segment children. |
| 11 | `individual_travel_segment` | `individual_travel_segment` | Specialized handler → `_load_handler_callable("individual_travel_segment")` | NO | **runnable** | Standalone segment via queue seam. |
| 12 | `travel_stitch` | (child of orchestrator) | Specialized handler → `handle_travel_stitch_task` | NO | **runnable** | ffmpeg/media stitcher; no WGP execution. |

### 1.2 USED-INDIRECTLY (3 types)

Task types that are not emitted by a resolver directly but are created as children by orchestrator task types (`travel_orchestrator`, `join_clips_orchestrator`).

| # | task_type | Dispatch path | Status | Notes |
|---|-----------|---------------|--------|-------|
| 13 | `travel_segment` | Specialized handler → `_load_handler_callable("travel_segment")` via queue seam | **blocked** | Wan-family tasks blocked on Sprint 4 NEW template (`ready_templates/video/wanvideo_wrapper_22_14b_vace_cocktail.py` — see §1A row 409); LTX family unblocks earlier. Created as child of `travel_orchestrator`. |
| 14 | `join_clips_segment` | Specialized handler → `handle_join_clips_task` via queue seam | **blocked** | Wan-family tasks blocked on Sprint 4 NEW template (same vace cocktail template as `travel_segment`). Non-Wan stitch is ffmpeg-only. Created as child of `join_clips_orchestrator`. |
| 15 | `join_final_stitch` | Specialized handler → `handle_join_final_stitch` | **blocked** | ffmpeg-only finalization; no WGP/VibeComfy execution. Blocked because it consumes `join_clips_segment` outputs — its upstream is blocked. Created as child of `join_clips_orchestrator`. |

### 1.3 USED-NON-RAYWORKER (routed via API orchestrator)

These task types appear in the resolver registry but execute via the API orchestrator (`reigh-worker-orchestrator`), not the RayWorker. See Section 2 for the detailed route inventory.

| # | task_type | Resolver family | Runtime |
|---|-----------|------------------|---------|
| 16 | `video_enhance` | `video_enhance` | API orchestrator (fal.ai) |
| 17 | `image_upscale` | `image_upscale` | API orchestrator (fal.ai) — see §2.1 for hyphen/underscore note |

### 1.4 VibeComfy Proof Update — 2026-05-07

After the original Sprint 0A baseline, the worker promoted and live-tested the
Qwen ready-template group through the production-shaped VibeComfy queue path:

- `qwen_image_2512` -> `image/qwen_image_2512`
- `qwen_image_edit` -> `edit/qwen_image_edit`
- `qwen_image_style` -> `edit/qwen_image_edit`
- `image_inpaint` -> `edit/qwen_image_edit` with a worker-created mask composite
- `annotated_image_edit` -> `edit/qwen_image_edit` with a worker-created
  annotated/masked composite

Live RunPod report:
`scripts/live_test/runs/20260507T065745Z/report.md`, pod `v7ozyfck7hqhbb`,
`5/5 passed`.

Remaining direct-route gaps are deliberate:

- `qwen_image` stays WGP-only because VibeComfy has `qwen_image_2512`, not a
  proven plain Qwen Image template. Aliasing these would violate the threshold
  policy.
- `z_image_turbo_i2i` stays WGP-only because the VibeComfy corpus has Z-Image
  text-to-image and Flux edit templates, but no proven Z-Image Turbo img2img
  parity template.
- `wan_2_2_t2i` stays WGP-only because the available VibeComfy Wan templates
  do not match the production single-frame Wan 2.2 T2I contract.
- `travel_segment` and `join_clips_segment` stay blocked for VibeComfy because
  the required Wan 2.2 VACE cocktail source template is not present in the
  VibeComfy ready-template source tree.

---

## 2. Non-RayWorker Route Inventory

Routes that bypass the RayWorker entirely and execute via the API orchestrator (`reigh-worker-orchestrator`). Verified against `api_orchestrator/task_handlers.py:22-44`.

| task_type | Current runtime | Owner | Executor handler | External API | Preserve or move? | Rationale | Completion handler | Billing path |
|-----------|-----------------|-------|------------------|--------------|--------------------|-----------|--------------------|-------------|
| `video_enhance` | API orchestrator | Peter O'Malley, interim product owner | `fal.py:364` — `handle_video_enhance` | fal.ai | **Preserve** | Stable external API; no VibeComfy template equivalent exists; migrating would degrade quality. | `complete_task/generation-handlers.ts` | `complete_task/billing.ts` |
| `image-upscale` | API orchestrator | Peter O'Malley, interim product owner | `fal.py:20` — `handle_image_upscale` | fal.ai | **Preserve** | Stable external API; note hyphen/underscore mismatch (see §2.1). | `complete_task/generation-handlers.ts` | `complete_task/billing.ts` |
| `animate_character` | API orchestrator | Peter O'Malley, interim product owner | `wavespeed.py:356` — `handle_animate_character` | Wavespeed | **Preserve** | Stable external API; character animation via Wavespeed API. | `complete_task/generation-handlers.ts` | `complete_task/billing.ts` |
| `flux_klein_edit` | API orchestrator | Peter O'Malley, interim product owner | `fal.py:276` — `handle_flux_klein_edit` | fal.ai | **Preserve** | Stable external API; Flux-based Klein edit via fal.ai. | `complete_task/generation-handlers.ts` | `complete_task/billing.ts` |

All four routes confirmed active in `reigh-worker-orchestrator/api_orchestrator/task_handlers.py`:
- `image-upscale` at line 30
- `video_enhance` at line 35
- `flux_klein_edit` at line 36
- `animate_character` at line 26

### 2.1 image-upscale Hyphen/Underscore Mismatch

There is a known inconsistency between the resolver-emitted task type and the DB registry:

| Component | Value | Source |
|-----------|-------|--------|
| Resolver (`imageUpscale.ts:80`) | `image-upscale` (hyphen) | `reigh-app/supabase/functions/create-task/resolvers/imageUpscale.ts` |
| DB migration `20251021000002` | `image_upscale` (underscore) at line 17 | Original `task_types` INSERT (`ON CONFLICT (name) DO NOTHING`). Note: the SQL comment on line 1 says "image-upscale" but the value inserted is `'image_upscale'`. |
| DB migration `20260204191820` | `image-upscale` (hyphen) at line 18 | `UPDATE task_types SET variant_type = 'upscaled' WHERE name = 'image-upscale'` — would no-op if only underscore row exists |
| API orchestrator handler | `image-upscale` (hyphen) at line 30 | `reigh-worker-orchestrator/api_orchestrator/task_handlers.py` |
| `get_task_run_type()` | Matches `task_types.name = p_task_type` | Falls back to `'gpu'` when no row matches |

**Claim-path consequence:** `get_task_run_type()` matches `task_types.name` against the resolver-emitted `p_task_type`. If the DB has both rows (underscore and hyphen), the resolver (hyphen) matches the correct handler. If the DB has only the underscore row, the resolver-emitted `image-upscale` (hyphen) won't match any active row, `get_task_run_type` returns `'gpu'` fallback, and API workers passing `p_run_type='api'` cannot claim these tasks.

**DB state:** The U2 verification (completed in T13) confirmed both rows exist in the production DB — `image_upscale` (underscore, `run_type: 'gpu'`) and `image-upscale` (hyphen, `run_type: 'api'`). This means the current system works correctly: the resolver emits hyphen, it matches the API-typed row, and API workers claim the task. However, the coexistence is fragile — if the hyphen row is ever deleted or deactivated, the claim path silently breaks.

**Recommendation:** Normalize to one consistent form (`image-upscale` with hyphen) across resolver, DB, and handler registry. The underscore row should be deactivated or removed after confirming no claim paths depend on it.

---

## 3. Per-USED-RayWorker-Task Contract Skeleton

Cross-referenced against `docs/migration-vibecomfy.md §1` (lines 248-285) and `§1A` (lines 379-416).

Each contract entry covers: payload shape, timeout, polling cadence, output fields, product effects, billing, duplicate completion idempotency, partial orchestrator failure.

### Cohort A: Image Generation

#### `qwen_image`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `imageGeneration.ts` — `prompt`, `model_name`, `resolution`, optional `reference_image_urls`, `strength`, `reference_mode`, `loras` |
| Timeout | 3600s (`max_wait_time = 3600` at `task_registry.py:1565`) |
| Polling cadence | 2s interval (`time.sleep(2)` at `task_registry.py:1576`) |
| Output fields | Single image path (WGP output) |
| Product effects | Image generation → `complete_task/generation-handlers.ts` → variant creation or standalone generation |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` idempotency_key; `task_claim.py` recovery |
| Orchestrator failure | N/A — direct queue task, no orchestrator children |

#### `qwen_image_style`
Same as `qwen_image`; additional prompt/model rewrite during conversion (`task_types.py` notes).

#### `qwen_image_2512`
Same as `qwen_image`; uses `qwen_image_2512_20B` model.

#### `z_image_turbo`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `imageGeneration.ts` — `prompt`, `model_name`, `resolution` |
| Timeout | 3600s |
| Polling cadence | 2s interval |
| Output fields | Single image path (WGP output); single-frame/image defaults in `task_conversion.py` |
| Product effects | Image generation → variant creation |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | N/A |

#### `z_image_turbo_i2i`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `zImageTurboI2I.ts` — `image_url`, `prompt`, `model_name`, `resolution` |
| Timeout | 3600s |
| Polling cadence | 2s interval |
| Output fields | Single image path; downloads input image to local temp for WGP |
| Product effects | Image-to-image variant creation |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | N/A |

#### `wan_2_2_t2i`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `imageGeneration.ts` — `prompt`, `model_name` (defaults to `wan_2_2_i2v_lightning_baseline_2_2_2`), `resolution` |
| Timeout | 3600s (`max_wait_time = 3600` at `task_registry.py:1565`) |
| Polling cadence | 2s interval (`time.sleep(2)` at `task_registry.py:1576`) |
| Output fields | Single image path; forces `video_length=1` (`task_registry.py:1554-1555`) |
| Product effects | Image generation via Wan T2I |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | N/A |

### Cohort B: Image Edit

#### `qwen_image_edit`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `imageGeneration.ts` — `prompt` (optional), `reference_image_urls`, `strength`, `model_name` |
| Timeout | 3600s |
| Polling cadence | 2s interval |
| Output fields | Single image path; empty prompt allowed |
| Product effects | Image edit → variant creation |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | N/A |

#### `image_inpaint`
Same as `qwen_image_edit`; requires mask handling in payload.

#### `annotated_image_edit`
Same as `qwen_image_edit`; annotation baked onto source in pre-process.

### Cohort E: Orchestration

#### `travel_orchestrator`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `travelBetweenImages.ts` — `orchestrator_details` including `input_image_paths_resolved`, `base_prompts_expanded`, `segment_frames_expanded`, `frame_overlap_expanded`, `model_name`, `model_type`, `seed_base`, etc. |
| Timeout | Orchestrator-managed; per-segment child timeout via queue (3600s per child) |
| Polling cadence | Orchestrator polls child completion via task status; 2s child polling |
| Output fields | Final travel video path or orchestrating status |
| Product effects | Creates `travel_segment` children; `complete_task/generation-handlers.ts`; `orchestrator.ts` |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` idempotency_key; `task_claim.py` recovery |
| Orchestrator failure | Partial failure handled by `orchestrator.gate.test.ts`; children may complete independently |

#### `individual_travel_segment`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `individualTravelSegment.ts` — per-segment params: `input_image_paths_resolved`, `base_prompts_expanded`, `segment_frames_expanded`, `frame_overlap_expanded`, `model_name`, `model_type`, `seed_base` |
| Timeout | 1800s via queue (`max_wait_time = 1800` at `task_registry.py:1346`) |
| Polling cadence | 2s interval via `_handle_travel_segment_via_queue_impl` (`wait_interval = 2` at `task_registry.py:1347`) |
| Output fields | Segment video path |
| Product effects | Standalone segment generation (lightbox regen path) |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | N/A — standalone; no orchestrator dependency |

#### `travel_stitch`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Child of `travel_orchestrator` — receives segment output paths |
| Timeout | Handler-managed (ffmpeg stitch time) |
| Polling cadence | N/A — synchronous handler |
| Output fields | Stitched final video path |
| Product effects | Final video assembly |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | Runs after all segments complete; failure = no final video |

#### `travel_segment` (USED-INDIRECTLY)
| Contract column | Value |
|-----------------|-------|
| Payload shape | Created by `travel_orchestrator` — per-segment params with `orchestrator_task_id_ref`, `segment_index`, `start_image`/`end_image` URLs |
| Timeout | 1800s via queue (`max_wait_time = 1800` at `task_registry.py:1346`) |
| Polling cadence | 2s interval via `_handle_travel_segment_via_queue_impl` (`wait_interval = 2` at `task_registry.py:1347`; `time.sleep(wait_interval)` at line 1394) |
| Output fields | Segment video path; SVI latent tail upload if active |
| Product effects | Feeds into orchestrator completion counting |
| Billing | Billed via parent orchestrator |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | Child of orchestrator; orchestrator self-detects completion |

#### `join_clips_orchestrator`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `joinClips.ts` — clip URLs, transition params |
| Timeout | Orchestrator-managed |
| Polling cadence | Orchestrator polls children |
| Output fields | Final joined video or orchestrating status |
| Product effects | Creates `join_clips_segment` and optionally `join_final_stitch` children |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | Children may complete independently |

#### `join_clips_segment` (USED-INDIRECTLY)
| Contract column | Value |
|-----------------|-------|
| Payload shape | Created by `join_clips_orchestrator` — per-segment transition params |
| Timeout | 1800s via queue default (`max_wait_time = 1800` at `join/generation.py:772`) |
| Polling cadence | 2s interval (`wait_interval = 2` at `join/generation.py`; `time.sleep(wait_interval)` at line 1224) |
| Output fields | Transition/joined video path |
| Product effects | Feeds into orchestrator completion |
| Billing | Billed via parent orchestrator |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | Child of orchestrator |

#### `join_final_stitch` (USED-INDIRECTLY)
| Contract column | Value |
|-----------------|-------|
| Payload shape | Created by `join_clips_orchestrator` — segment output paths |
| Timeout | Handler-managed (ffmpeg stitch time) |
| Polling cadence | N/A — synchronous handler |
| Output fields | Final stitched video path |
| Product effects | Final video assembly |
| Billing | Billed via parent orchestrator |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | Runs after all segments complete |

#### `edit_video_orchestrator`
| Contract column | Value |
|-----------------|-------|
| Payload shape | Resolver: `editVideoOrchestrator.ts` — edit specifications |
| Timeout | Orchestrator-managed |
| Polling cadence | Orchestrator polls children |
| Output fields | Final edited video or orchestrating status |
| Product effects | Creates child join/regen tasks |
| Billing | `complete_task/billing.ts` |
| Duplicate completion | Idempotency via `complete_task` |
| Orchestrator failure | Children may complete independently |

---

## Cross-Reference Validation

- **§1 coverage (lines 248-285):** All 36 task types catalogued. 12 USED-IN-APP, 3 USED-INDIRECTLY, 2 USED-NON-RAYWORKER, 19 UNUSED.
- **§1A coverage (lines 379-416):** Cohort A (5 types: `z_image_turbo`, `z_image_turbo_i2i`, `qwen_image`, `qwen_image_2512`, `wan_2_2_t2i`), Cohort B (3 types: `qwen_image_edit`, `image_inpaint`, `annotated_image_edit`), Cohort E (9 types including orchestrators and indirect children).
- **Total USED count:** 17 (12 USED-IN-APP + 3 USED-INDIRECTLY + 2 USED-NON-RAYWORKER). This matches the §1A authoritative count of 17 in-scope types.
