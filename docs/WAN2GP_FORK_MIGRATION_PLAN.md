# Wan2GP Fork Migration — Design Document

## 1. Executive Summary
`reigh-worker/Wan2GP/` is currently a flat vendored snapshot with no `.git` directory, so it cannot be cleanly rebased onto `deepbeepmeep/Wan2GP` even though it is roughly 50 days and about 9 minor versions behind. The hottest drift zone is `Wan2GP/models/ltx2/`, where 22 files differ and the working diff is roughly 4303 lines.

This migration plan is intentionally a planning artifact, not an implementation batch. It breaks the migration into 4 sprints of 2 weeks each so the team can inventory drift first, cut over to a maintainable fork-plus-patch model second, resync the risky `models/ltx2/` surface third, and finish with sustainability controls last.

The non-negotiable constraint is that no functionality may be lost during any sprint. The plan therefore treats these as protected behaviors that must survive with explicit verification: IC-LoRA auto-injection for pose, depth, canny, and cameraman modes; the newly landed `ltx_anchor` kind; the existing `ltx_control` pixel cross-fade guide-video path; the `ltx_hybrid` route; `clear_conditioning` keyframe-token stripping before VAE decode; the existing 6-file travel test suite; and the import-path contract that supports both dotted `Wan2GP.*` callers and bare-module callers.

## Migration Status

**Migration complete — 2026-04-22.** All four sprints landed on `banodoco/reigh-worker main` and the submodule now tracks `banodoco/Wan2GP @ 181bb71a` (fork branch `reigh-sprint-3`).

| Sprint | Status | Commit | Date |
| --- | --- | --- | --- |
| Sprint 1 — Triage & Fork Cut | ✅ Done | `9656cc90` | 2026-04-21 |
| Sprint 2 — Submodule Cut-over, Low-Risk Resync, Doc Updates | ✅ Done | `253e9fba` | 2026-04-22 |
| Sprint 3 — `models/ltx2/` Resync + Cameraman/Dedup Tests | ✅ Done | `550320a3` | 2026-04-22 |
| Sprint 4 — Patch Consolidation, Upstream PRs, Sustainability | ✅ Done | `90e05f2d` | 2026-04-22 |

Rollback anchors: tags `wan2gp-mig-sprint-{1,2,3,4}-baseline` (baseline-N = state *before* sprint N began). Tag `wan2gp-mig-complete` points at the Sprint 4 commit (`90e05f2d`).

An independent regression audit (codex) confirmed the migration landed additively: `Wan2GP/wgp.py` byte-identical pre→post, all 97 pre-existing `models/ltx2/*.py` files SHA-identical, zero pre-existing file contents modified. Only `.gitignore` differs in content. See §10 Audit Findings for details.

## 2. Current-State Inventory

### 2a. Vendored tree facts
`reigh-worker/Wan2GP/` is currently a plain copied tree rather than a repository checkout, so there is no `.git` metadata to rebase, cherry-pick, or diff-history against. The vendored changelog currently tops out at `WanGP v10.01` dated January 1, 2026, while upstream context for this plan is `v10.951` dated February 19, 2026.

The hottest resync zone is `Wan2GP/models/ltx2/`, where the working comparison indicates 22 differing files and about 4303 lines of diff. That is large enough that the migration cannot safely be framed as a single blind replacement.

The repo also has a non-trivial import and runtime surface already coupled to the current mount path. The approved background for this plan treats that as a 26+ call-site contract, which is why the migration must preserve both filesystem location and import semantics rather than just swapping in a dependency reference.

### 2b. Genuinely missing upstream items
The current vendored tree is missing several upstream features that are desired outcomes of the migration rather than already-carried drift. Confirmed missing items include `Wan2GP/models/ltx2/ltx_pipelines/utils/res2s.py` and `Wan2GP/defaults/ltx2_22B_distilled_1_1.json`.

Other upstream-side capabilities that this design document should plan to absorb during the migration include `_apply_gamma_to_media`, `LTX2_OUTPAINT_GAMMA`, `LTX2_DISABLE_STAGE2_WITH_CONTROL_VIDEO`, and the newer prompt-enhancer improvements. These belong in the migration inventory because they are feature gaps, not just cleanup opportunities.

### 2c. Already-vendored drift-only items
`Wan2GP/shared/utils/self_refiner.py` is already vendored in the current tree. It should therefore be treated as a drift candidate that may need reconciliation with upstream head, not as a missing upstream backfill.

That distinction matters for sprint design: Sprint 2 can absorb `self_refiner` drift as a low-risk resync task, while `res2s.py` and `ltx2_22B_distilled_1_1.json` remain true additions.

### 2d. Patch seam and lifecycle
The repo already has an established patch seam for Wan2GP-specific runtime behavior. The module-load trigger is `source/__init__.py:6`, which ensures `source.models.wgp.wgp_patches` is importable during bootstrap.

Actual patch application happens later in the runtime lifecycle at `source/models/wgp/orchestrator.py:265-266`, where `apply_all_wgp_patches(wgp, self.wan_root)` is called after the Wan2GP module is loaded and before orchestration proceeds. The migration plan should preserve this existing seam instead of introducing a second patch package.

### 2e. banodoco-carry items
Some behaviors are not generic upstream drift and must be treated as load-bearing banodoco carry items until they are either upstreamed or deliberately relocated. The clearest examples are `clear_conditioning` in `Wan2GP/models/ltx2/ltx_core/tools.py:66-79`, the `_IC_LORA_BY_MODE` registry in `source/core/params/travel_guidance.py`, and the `ltx_anchor`-related orchestrator rewiring around `_build_segment_anchor_guidance_config`, PATH A skip behavior, and segment-local anchor remapping.

These items are the reason the migration cannot be framed as “replace vendor tree with upstream head.” Each one needs an explicit carry-location decision later in the plan.

### 2f. Pre-existing test coverage gaps
The current test suite does not yet pin the cameraman-specific override branch or the task-registry dedup branch. T1 confirmed that current collected cameraman coverage is limited to `test_parse_ltx_control_cameraman` and `test_needs_ic_lora[cameraman-True]`, and neither one reaches `get_ic_lora_entry()`.

That leaves the cameraman override branch at `source/core/params/travel_guidance.py:310-313` and the dedup logic at `source/task_handlers/tasks/task_registry.py:964-968` unexercised by current tests. Sprint 3 therefore needs new first-class test deliverables rather than relying on `-k` selectors over the existing suite.

### 2g. Stale-doc inventory
T1 identified existing documentation that still describes `Wan2GP/` as a flat vendored tree prior to Sprint 2. Confirmed narrow-sweep hits are `README.md:39`, `STRUCTURE.md:22`, and `docs/KIJAI_SVI_IMPLEMENTATION.md:202`.

The broader sweep also found `STRUCTURE.md:70`, which still describes `Wan2GP/` as “Upstream video generation engine (vendored, do-not-edit-in-place).” Sprint 2 therefore needs a doc-sweep gate that covers both the narrow regex and the broader `Wan2GP` plus `vendor` pass, followed by manual reviewer confirmation because the regex is heuristic.

### 2h. Invocation contract
The repo’s test and worker invocation contract is `uv run --python 3.10 pytest …`, consistent with the locked Python 3.10 environment in `uv.lock` and the worker run examples in `docs/KIJAI_SVI_IMPLEMENTATION.md:195`. This document should treat that as the canonical pytest invocation form for every later verification gate.

Plain `pytest` is not sufficient in this repo because it can fail with `ModuleNotFoundError: source` when the expected environment contract is not active. The migration plan therefore needs to standardize on the `uv run --python 3.10` prefix anywhere it names pytest.

### 2i. Path-contract verification technique
The path contract has two caller classes that must both survive the migration. First, `ensure_wan2gp_on_path()` inserts the absolute `Wan2GP/` directory itself into `sys.path`, not the repo root. Second, the more common dotted `Wan2GP.*` callers can work when either the repo root or `Wan2GP/` is on `sys.path`, so they are necessary but not sufficient proof that the mount contract remains correct.

The stricter contract is driven by the bare-module callers in `vendor_imports.py:76-88`: `models.qwen.qwen_handler`, `shared.utils.loras_mutipliers`, and `models.qwen.qwen_main`. Those class-b callers require the absolute `Wan2GP/` insertion specifically, which is why the Sprint 2 smoke must assert `os.path.abspath(str(get_wan2gp_path())) in [os.path.abspath(p) for p in sys.path]` with no repo-root fallback.

Bare-import reachability should be verified only through filesystem proxies such as `os.path.isfile('Wan2GP/models/qwen/qwen_handler.py')`, `os.path.isfile('Wan2GP/models/qwen/qwen_main.py')`, and `os.path.isfile('Wan2GP/shared/utils/loras_mutipliers.py')`. `find_spec` and real imports on `shared.utils.*` are intentionally avoided because `Wan2GP/shared/utils/__init__.py` eagerly imports SciPy-sensitive solver modules. Full end-to-end validation of the bare-module callers should then come from the existing test suite exercising `vendor_imports.py` through normal codepaths.

## 3. Fork Architecture Decisions
The migration will use the hybrid architecture defined by SD-001: `banodoco/Wan2GP` becomes the long-running fork, and `reigh-worker/Wan2GP/` becomes a git submodule pinned to that fork rather than a flat copied tree. That choice keeps the existing filesystem contract stable while making future rebases, drift review, and upstream comparison mechanically possible again.

Patch ownership is split deliberately rather than informally. SD-002 defines the required triage buckets so every deviation is classified as accept-upstream, upstreamable, banodoco-fork-or-banodoco-patch, or cruft; SD-003 stages the resync by subsystem so low-risk `shared/` and `postprocessing/` work lands before the hot `models/ltx2/` zone; and SD-004 keeps LTX 2.3 distilled 1.1 default promotion out of the migration path so the migration is not coupled to a defaults experiment.

Feature exposure is also staged. SD-005 keeps `res_2s` and `self_refiner` behind off-by-default flags until the fork relationship is stable, and SD-006 keeps any post-gen detailer work outside this initiative so the migration remains focused on source-of-truth and drift control instead of product expansion.

Sustainability is handled as part of the architecture, not as follow-up cleanup. SD-007 sets a monthly rebase cadence with event-triggered syncs on upstream minor-version bumps, while SD-008 adds scheduled drift detection in CI so the repository does not slide back into another opaque vendor snapshot. Together, SD-001 through SD-008 define a maintainable fork-plus-patch-layer model rather than a one-time resync.

## 4. Sprint Plan (4 × 2 weeks)
### ✅ Sprint 1 — Triage & Fork Cut (Weeks 1-2)
#### Goals
- Fork `banodoco/Wan2GP` at upstream HEAD and establish it as the future source of truth.
- Build `docs/wan2gp-triage.csv` and classify every differing file into the SD-002 buckets.
- Pre-classify `clear_conditioning` as bucket `(c)` before any hot-zone resync begins.

#### Deliverables
- `banodoco/Wan2GP` fork created and available for migration work.
- `docs/wan2gp-triage.csv` with rows covering the 22-file `models/ltx2/` hot zone and every differing file elsewhere.

#### Verification Gates
- `docs/wan2gp-triage.csv` contains at least one row for every differing file and explicit bucket decisions for all 22 `models/ltx2/` files.
- The `clear_conditioning` row remains pre-classified as bucket `(c)` with an explicit carry location and verification entry.

#### Rollback Plan
- Record tag `wan2gp-mig-sprint-1-baseline` before opening the fork workstream.
- No `reigh-worker` code changes are planned in this sprint; if the fork cut is rejected, abandon the fork workstream and leave `reigh-worker` at `wan2gp-mig-sprint-1-baseline`.

#### Functionality-Preservation Checks
- Zero `reigh-worker` code changes means zero risk to IC-LoRA behavior, `ltx_anchor`, `ltx_control`, `ltx_hybrid`, `clear_conditioning`, and the path/import contract during Sprint 1.
- The only outputs are the fork setup and the triage artifact, so every protected behavior is preserved by construction.

### ✅ Sprint 2 — Submodule Cut-over, Low-Risk Resync, Doc Updates (Weeks 3-4)
#### Goals
- Replace the flat `Wan2GP/` tree with a git submodule at the same mount path.
- Resync `shared/` and `postprocessing/`, absorb `self_refiner` drift, and introduce `res2s.py` behind an off-by-default flag per SD-005.
- Update all stale docs identified in T1.

#### Deliverables
- `.gitmodules` plus the submodule mounted at `reigh-worker/Wan2GP/`.
- `Wan2GP/shared/utils/self_refiner.py` aligned to fork HEAD.
- `Wan2GP/models/ltx2/ltx_pipelines/utils/res2s.py` present and gated off by default.
- `README.md`, `STRUCTURE.md`, and `docs/KIJAI_SVI_IMPLEMENTATION.md` updated to remove pre-Sprint-2 vendored-tree wording.

#### Verification Gates
- Strict path-contract smoke:
  > ```bash
  > python -c "import os, sys; from source.core.runtime_paths import ensure_wan2gp_on_path, get_wan2gp_path; ensure_wan2gp_on_path(); assert os.path.abspath(str(get_wan2gp_path())) in [os.path.abspath(p) for p in sys.path]; assert os.path.isfile('Wan2GP/models/qwen/qwen_handler.py'); assert os.path.isfile('Wan2GP/models/qwen/qwen_main.py'); assert os.path.isfile('Wan2GP/shared/utils/loras_mutipliers.py'); assert os.path.isfile('Wan2GP/shared/utils/self_refiner.py'); assert os.path.isfile('Wan2GP/models/ltx2/ltx_core/tools.py'); print('path-contract smoke OK')"
  > ```
- Full 6-file suite:
  `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py`
- LTX-2 `self_refiner` smoke against the pre-sprint behavioral baseline.
- Doc-sweep gate: run the two regex sweeps specified in Sprint 2's brief (gate C). Both must return zero hits across `docs/`, `README.md`, and `STRUCTURE.md`. Regex literals are intentionally omitted here to avoid self-matching the gate commands.

#### Rollback Plan
- Record tag `wan2gp-mig-sprint-2-baseline` before submodule cut-over.
- Roll back with `git submodule deinit` plus `git reset --hard wan2gp-mig-sprint-2-baseline`.

#### Functionality-Preservation Checks
- No `models/ltx2/` files are touched in Sprint 2.
- The LTX-2 `self_refiner` smoke remains green.
- The strict path-contract smoke remains green and proves that `Wan2GP/` is on `sys.path` and all three bare-import targets still exist.
- The doc sweeps plus reviewer pass show no stale wording remains in approved locations.

### ✅ Sprint 3 — `models/ltx2/` Resync + Write Cameraman/Dedup Tests (Weeks 5-6)
#### Goals
- Resync the 22-file `models/ltx2/` hot zone while explicitly carrying `clear_conditioning`.
- Add `ltx2_22B_distilled_1_1.json` as available-but-not-default per SD-004.
- Reconcile upstream `guide_phases=1` behavior and write the missing cameraman and dedup coverage as first-class deliverables.

#### Deliverables
- `models/ltx2/` resynced against the fork.
- `clear_conditioning` preserved as byte-identical on a saved latent fixture.
- New file `tests/test_travel_guidance_ic_lora_override.py` containing `test_get_ic_lora_entry_cameraman_returns_cseti_url`, which builds `TravelGuidanceConfig(kind="ltx_control", mode="cameraman", ...)`, calls `get_ic_lora_entry()`, asserts `path == "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1/resolve/main/LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors"`, `name == "ic-lora-cameraman (auto-injected)"`, and strength matches config, and `test_get_ic_lora_entry_pose_falls_back_to_union_control`, which uses `mode="pose"` and asserts the path equals the `_IC_LORA_UNION_CONTROL` filename.
- New file `tests/test_task_registry_ic_lora_dedup.py` containing `test_dedup_updates_strength_when_basename_matches`, which seeds `segment_loras=[{"path": "…/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", "strength": 0.4}]`, drives auto-inject with the same basename at strength `0.8`, and asserts length `1` plus `strength == 0.8`, and `test_dedup_appends_when_no_match`, which seeds an empty list and asserts the injected entry appears once.

#### Verification Gates
- New IC-LoRA tests:
  `uv run --python 3.10 pytest tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`
- Full 6-file suite:
  `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py`
- `clear_conditioning` byte-compare against a saved latent fixture.
- PATH A skip gate:
  `uv run --python 3.10 pytest tests/test_travel_orchestrator_terminal_gating.py`
- 20 anchor tests:
  `uv run --python 3.10 pytest tests/test_travel_guidance_config.py`
- No `pytest -k` selectors involving `cameraman` or `ic_lora and cameraman` are used anywhere in the sprint.

#### Rollback Plan
- Record tag `wan2gp-mig-sprint-3-baseline` before the `models/ltx2/` resync.
- Revert the submodule pointer to the Sprint 2 state and run `git reset --hard wan2gp-mig-sprint-3-baseline`.

#### Functionality-Preservation Checks
- The cameraman override branch is pinned by `test_get_ic_lora_entry_cameraman_returns_cseti_url`.
- The pose fallback branch is pinned by `test_get_ic_lora_entry_pose_falls_back_to_union_control`.
- Dedup is pinned by `test_dedup_updates_strength_when_basename_matches` and `test_dedup_appends_when_no_match`.
- `clear_conditioning` remains byte-identical.
- `ltx_anchor` PATH A skip remains intact.
- `ltx_control` pixel cross-fade remains intact.
- `ltx_hybrid` remains unchanged.
- The 26+ `Wan2GP.*` call sites and the bare-module callers still resolve after the hot-zone resync.

### ✅ Sprint 4 — Patch Consolidation, Upstream PRs, Sustainability (Weeks 7-8)
#### Goals
- Extend `source/models/wgp/wgp_patches.py` using the existing runtime-patch primitives.
- Open upstream PRs for bucket `(b)` items.
- Land drift CI per SD-008 and publish the rebase runbook required by SD-007.

#### Deliverables
- `source/models/wgp/wgp_patches.py` extended using the existing `_register_patch_application`, `begin_runtime_model_patch`, and `rollback_runtime_model_patch` primitives.
- Draft upstream PRs for upstreamable hunks.
- `.github/workflows/wan2gp-drift.yml` merged.
- `docs/wan2gp-rebase-runbook.md` published.

#### Verification Gates
- Drift CI runs once successfully.
- Full regression suite:
  `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`
- Patch-lifecycle smoke drives bootstrap to `source/models/wgp/orchestrator.py:265-266` via `apply_all_wgp_patches(wgp, self.wan_root)` and only then inspects `get_wgp_patch_state()`.

#### Rollback Plan
- Record tag `wan2gp-mig-sprint-4-baseline` before patch consolidation lands.
- Because the Sprint 4 additions are isolated to `source/models/wgp/wgp_patches.py`, roll back with `git reset --hard wan2gp-mig-sprint-4-baseline` or disable the runtime patch context through `_PATCH_CONTEXT_ROLLBACKS`.

#### Functionality-Preservation Checks
- Re-run the full Sprint 3 regression matrix.
- Confirm that patch-state verification sees a single registry rather than split runtime patch state.

## 5. Functionality Preservation Matrix
| Behavior | Defining location | Sprint verifying it | Verification step |
| --- | --- | --- | --- |
| IC-LoRA union-control fallback for pose/depth/canny | `source/core/params/travel_guidance.py:314` plus `source/task_handlers/tasks/task_registry.py:955-987` | Sprint 3 | `tests/test_travel_guidance_ic_lora_override.py::test_get_ic_lora_entry_pose_falls_back_to_union_control` |
| IC-LoRA cameraman override | `source/core/params/travel_guidance.py:310-313` plus `_IC_LORA_BY_MODE` at `source/core/params/travel_guidance.py:32-37` | Sprint 3 | `tests/test_travel_guidance_ic_lora_override.py::test_get_ic_lora_entry_cameraman_returns_cseti_url` |
| IC-LoRA dedup | `source/task_handlers/tasks/task_registry.py:961-968` | Sprint 3 | `tests/test_task_registry_ic_lora_dedup.py::test_dedup_updates_strength_when_basename_matches` plus `tests/test_task_registry_ic_lora_dedup.py::test_dedup_appends_when_no_match` |
| Pixel cross-fade guide video for `ltx_control` | `source/media/video/travel_guide.py` | Sprint 3 | `uv run --python 3.10 pytest tests/test_ltx_hybrid_travel.py` plus manual smoke |
| `ltx_anchor` kind survives the resync | `source/task_handlers/travel/orchestrator.py:134,231,1272-1289,2115,2126` | Sprint 3 | `uv run --python 3.10 pytest tests/test_travel_guidance_config.py` |
| `clear_conditioning` keyframe strip before VAE decode | `Wan2GP/models/ltx2/ltx_core/tools.py:66-79` | Sprint 3 | Byte-compare against saved latent fixture |
| PATH A skip for `ltx_anchor` | `source/task_handlers/travel/orchestrator.py:1272-1289` | Sprint 3 | `uv run --python 3.10 pytest tests/test_travel_orchestrator_terminal_gating.py` |
| `_build_segment_anchor_guidance_config` rename remains intact | `source/task_handlers/travel/orchestrator.py:178` | Sprint 3 | `uv run --python 3.10 pytest tests/test_travel_guidance_config.py` |
| Dotted `Wan2GP.*` import root for class-a callers | 26+ call sites using `Wan2GP.*` | Sprint 2 | Strict path smoke asserts `get_wan2gp_path()` absolute path on `sys.path` |
| BARE-MODULE IMPORT CONTRACT for class-b callers | `source/runtime/wgp_ports/vendor_imports.py:76-88`, specifically `models.qwen.qwen_handler` at `:77`, `shared.utils.loras_mutipliers` at `:82`, and `models.qwen.qwen_main` at `:87` | Sprint 2 | Strict path-contract smoke asserts the absolute `Wan2GP/` path on `sys.path` plus filesystem existence of `Wan2GP/models/qwen/qwen_handler.py`, `Wan2GP/models/qwen/qwen_main.py`, and `Wan2GP/shared/utils/loras_mutipliers.py`; filesystem proxy only because `Wan2GP/shared/utils/__init__.py:1-3` eagerly imports SciPy |
| `self_refiner` drift upgrade | `Wan2GP/shared/utils/self_refiner.py` | Sprint 2 | LTX-2 smoke test |
| `wgp_patches` patch-state integrity after orchestrator bootstrap | `source/models/wgp/wgp_patches.py` plus `source/models/wgp/orchestrator.py:265-266` | Sprint 4 | Drive bootstrap, then inspect `get_wgp_patch_state()` |
| Stale-doc wording is fully removed from approved surfaces | `README.md`, `STRUCTURE.md`, `docs/KIJAI_SVI_IMPLEMENTATION.md:202`, plus broader-sweep hits | Sprint 2 | Narrow and broad regex sweeps plus reviewer manual pass |
| 6-file travel regression suite remains green throughout the migration | `tests/` | Sprints 1-4 | `uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py` |

## 6. Risk Register
| Risk | Sprint | Prob | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| Triage misclassifies a banodoco-specific hunk as safe upstream drift | S1 | M | H | Spot-check hot-zone rows and require a second-pass review on `docs/wan2gp-triage.csv` before Sprint 2 starts |
| Silent bare-module caller break: `ensure_wan2gp_on_path()` regresses to insert the repo root, so dotted `Wan2GP.*` callers still work while `vendor_imports.py:76-88` callers break | S2 | M | H | Strict path smoke asserts `get_wan2gp_path()` absolute path on `sys.path` with no repo-root fallback, plus filesystem assertions for `Wan2GP/models/qwen/qwen_handler.py`, `Wan2GP/models/qwen/qwen_main.py`, and `Wan2GP/shared/utils/loras_mutipliers.py` |
| Submodule mount breaks `Wan2GP.*` imports or the `Wan2GP/` filesystem path contract | S2 | M | H | Enforce the SD-001 mount-path invariant and re-run the strict path-contract smoke after cut-over |
| `self_refiner` version bump silently changes LTX-2 behavior | S2 | M | M | Compare against the pre-sprint behavioral baseline using the planned LTX-2 smoke |
| Stale doc wording survives the cut-over and misleads later maintainers | S2 | L | M | Run both narrow and broad regex sweeps plus a reviewer manual pass, explicitly acknowledging the regex sweep is heuristic |
| Verification commands fail the repo environment contract and produce misleading results | Any | M | H | Require `uv run --python 3.10 pytest` for every named pytest invocation |
| SciPy trigger via `shared.utils.*` smoke: a verification command imports or `find_spec`s `shared.utils.*` and trips `Wan2GP/shared/utils/__init__.py:1-3` | S2 | L | M | Keep the path smoke limited to filesystem checks and `sys.path` assertions only; never use `import` or `find_spec` on `shared.utils.*` |
| `clear_conditioning` is silently reverted during the `models/ltx2/` resync | S3 | M | H | Preserve it explicitly as bucket `(c)` carry and gate Sprint 3 on a byte-compare latent fixture |
| IC-LoRA registry collides with upstream union-control preload behavior | S3 | H | M | Pin both branches with the new override and fallback tests before accepting the hot-zone resync |
| Cameraman override or dedup regression slips through because the current suite never exercised those branches | S3 | H | H | Write `tests/test_travel_guidance_ic_lora_override.py` and `tests/test_task_registry_ic_lora_dedup.py` as first-class Sprint 3 deliverables |
| Parallel patch package or split patch registry emerges during consolidation | S4 | M | M | Keep all overrides in `source/models/wgp/wgp_patches.py` per SD-001 and assert a single registry in the patch-state smoke |
| Patch smoke is anchored at the wrong lifecycle point and reports false confidence | S4 | M | M | Drive bootstrap to `source/models/wgp/orchestrator.py:265-266` before inspecting patch state |
| Monthly rebase cadence slips and drift starts accumulating again | Ongoing | M | M | Use SD-008 weekly drift CI to surface missed cadence before drift becomes opaque again |

## 7. Success Criteria
### Migration complete
- The `reigh-worker/Wan2GP/` submodule pointer is at upstream HEAD or within the monthly SD-007 cadence window.
- The 6 named regression suites plus the new Sprint 3 IC-LoRA tests all pass under `uv run --python 3.10 pytest`.
- Drift CI is green.
- The post-bootstrap `get_wgp_patch_state()` smoke passes.
- The strict path-contract smoke passes, including the bare-import target filesystem assertions.

### Migration in progress
- The submodule is mounted at `reigh-worker/Wan2GP/`.
- A subset of subsystems has been synced, but each completed checkpoint still has green tests.
- The doc sweeps plus reviewer pass are green.
- The strict path smoke is green at the current checkpoint.

### Regression observed
- Any row in Section 5 fails its named verification step.
- Any named regression suite fails.
- The path-contract smoke fails either because `get_wan2gp_path()` is not on `sys.path` exactly as required or because any bare-import target filesystem assertion fails.
- The doc sweep surfaces stale wording that is not explicitly approved for the current sprint.

## 8. Non-Goals
- Flipping `ltx_anchor` to default
- Real-GPU empirical validation of the new defaults
- Upscaler-pass anchor re-injection
- Identity-drift fixes

## 9. Settled Decisions
### Decision 1
- `id`: `SD-001`
- `load_bearing`: `true`
- `decision`: Use a hybrid fork shape: maintain `banodoco/Wan2GP` as the long-running fork and consume it in `reigh-worker` as a git submodule mounted at `reigh-worker/Wan2GP/`.
  Path/import contract sub-clause:
  - The submodule mount path stays exactly `reigh-worker/Wan2GP/`; no relocation or rename.
  - `ensure_wan2gp_on_path()` must continue inserting the absolute `Wan2GP/` directory itself into `sys.path`, not the repo root or a parent/sibling path.
  - The bare-module callers in `source/runtime/wgp_ports/vendor_imports.py:76-88` must remain valid, explicitly including `models.qwen.qwen_handler`, `shared.utils.loras_mutipliers`, and `models.qwen.qwen_main`.
  - The dotted `Wan2GP.*` callers remain supported as the other caller class.
  - Deviations that do not belong in the fork stay in the existing `source/models/wgp/wgp_patches.py` seam, which is module-loaded via `source/__init__.py:6` and applied by `apply_all_wgp_patches(wgp, self.wan_root)` at `source/models/wgp/orchestrator.py:265-266`.
  - No parallel runtime-side Wan2GP patch package is introduced.
- `rationale`: This preserves the current runtime contract while restoring maintainability. The stable mount path protects the import surface already used by both dotted `Wan2GP.*` imports and the bare-module import sites, and the existing patch seam prevents fork-only changes from being scattered into a second, conflicting patch registry.

### Decision 2
- `id`: `SD-002`
- `load_bearing`: `true`
- `decision`: Every deviation between the current vendored tree and upstream is triaged into one of four buckets: `(a) accept-upstream`, `(b) upstreamable`, `(c) banodoco-fork-or-banodoco-patch`, `(d) cruft`. The canonical artifact is `docs/wan2gp-triage.csv` with columns `path,hunk,bucket,carry_location,owner,verification`.
  Illustrative sample row:

  ```csv
  path,hunk,bucket,carry_location,owner,verification
  Wan2GP/models/ltx2/ltx_core/tools.py,"clear_conditioning() strips appended keyframe tokens before VAE decode",banodoco-fork-or-banodoco-patch,banodoco/Wan2GP or source/models/wgp/wgp_patches.py,worker-runtime,byte-compare saved latent fixture
  ```
- `rationale`: The migration risk is concentrated in drift classification, not in git mechanics alone. A forced bucket per hunk makes carry decisions reviewable, gives Sprint 1 a concrete deliverable, and prevents load-bearing customizations like `clear_conditioning` from being lost inside an undifferentiated merge.

### Decision 3
- `id`: `SD-003`
- `load_bearing`: `true`
- `decision`: Resync by subsystem, not all at once: `shared/` plus `postprocessing/` first, `models/ltx2/` second, and defaults/configs last.
- `rationale`: The repo already depends on fragile `models/ltx2/` behavior, including `ltx_anchor`, IC-LoRA handling, and `clear_conditioning`. Staging the migration lowers blast radius, lets the team validate path/import and doc updates before touching the 22-file hot zone, and keeps rollback boundaries understandable.

### Decision 4
- `id`: `SD-004`
- `load_bearing`: `false`
- `decision`: Hold LTX 2.3 distilled 1.1 default promotion for a separate A/B sprint outside the migration, even if `ltx2_22B_distilled_1_1.json` is imported during the resync.
- `rationale`: Default promotion is a product-choice experiment, not a prerequisite for establishing the fork relationship. Separating it from the migration keeps “source sync” and “new default behavior” from failing together or obscuring regressions.

### Decision 5
- `id`: `SD-005`
- `load_bearing`: `false`
- `decision`: Introduce `res_2s` and refreshed `self_refiner` support behind off-by-default flags during Sprints 2 and 3, with any default promotion deferred until upstream behavior and local verification converge.
- `rationale`: Both features are desirable, but neither is required to complete the vendor-to-fork transition safely. Flagging them keeps the migration forward-moving without forcing immediate behavior changes in the travel pipeline.

### Decision 6
- `id`: `SD-006`
- `load_bearing`: `false`
- `decision`: Keep post-gen detailer work out of this migration plan and treat it as a separate initiative after the fork-plus-patch relationship is stable.
- `rationale`: The user preference already points toward a task-handler implementation rather than inline travel-pipeline detailer logic. Folding that feature into the migration would expand scope, confuse verification ownership, and make it harder to distinguish migration regressions from brand-new behavior.

### Decision 7
- `id`: `SD-007`
- `load_bearing`: `true`
- `decision`: Rebase the fork monthly, with additional event-triggered syncs whenever upstream ships a new minor version that touches relevant subsystems.
- `rationale`: Weekly rebases would create churn without matching the repo’s current staffing and risk profile, while ad hoc syncs are what created the current opaque drift. Monthly cadence plus event triggers is frequent enough to keep diffs reviewable and infrequent enough to be realistic.

### Decision 8
- `id`: `SD-008`
- `load_bearing`: `true`
- `decision`: Add `.github/workflows/wan2gp-drift.yml` as a weekly scheduled CI job that compares the pinned submodule pointer against upstream HEAD and reports drift for review.
- `rationale`: Once the vendor tree becomes a submodule, drift becomes measurable. A scheduled CI check converts that measurability into an operating habit, so the repo does not return to a silent `.git`-less copy that is only audited after major divergence has already accumulated.

## 10. Post-Migration Audit Findings

Independent codex audits ran at end-of-migration. Headline results:

**Regression audit (pre-migration vs post-Sprint-3 submodule):** Migration landed as purely additive.
- `Wan2GP/wgp.py` (12,054 lines) byte-identical; sha256 `d8aab33f6d835fd87658446ea8da89fa57a1ca90685394fe0b7bfd5a3219a800`
- All 97 pre-existing `models/ltx2/*.py` files SHA-identical; 2 pure additions (`prompt_enhancer.py`, `ltx_pipelines/utils/res2s.py`)
- Zero pre-existing file contents modified across `shared/`, `postprocessing/`, `preprocessing/`, `plugins/`, `defaults/`, `models/`
- Only `.gitignore` differs in content
- All 17 `wgp.*` symbols referenced by `source/models/wgp/{orchestrator,wgp_patches}.py` resolve
- The `22d82ca6` class of silent-loss risk (upstream sync rewriting a feature file) is provably impossible here

**Upstream-PR inventory audit:** §2b is effectively obsolete.
- `res2s.py`, `ltx2_22B_distilled_1_1.json`, prompt-enhancer, `self_refiner` all already present in the submodule (absorbed as additive sync, not fork-only carries)
- `_apply_gamma_to_media`, `LTX2_OUTPAINT_GAMMA`, `LTX2_DISABLE_STAGE2_WITH_CONTROL_VIDEO` appear only in this document — no code occurrence (planning placeholders)
- Sprint 4's `scripts/sprint4/upstream_prs/*.md` drafts should be reclassified as triage/investigation notes, not filed as upstream PRs. 0 of the §2b items are currently PR-ready.

**Adversarial review of Sprint 3:** One CRITICAL-severity finding landed as followup.
- IC-LoRA dedup at `source/task_handlers/tasks/task_registry.py:867-875` only matches exact `basename()` equality. Case-variant filenames and URL-decorated paths bypass the check and can silently double-apply a LoRA. Tracked as a post-migration followup.

**Post-migration risk register (independent):** HIGH-severity items are all in the monkeypatch layer, not in the migration surface.
- Qwen patch failures are non-fatal (`wgp_patches.py:554-569`) — green bootstrap can hide broken Qwen lane
- LoRA cache key ignores `lora_multi` (`wgp_patches.py:429-447`) — strength changes silently ignored
- LoRA tolerance strip silently neutralizes new-format keys (`wgp_patches.py:315-381`)
- Drift CI correctly exempts `Wan2GP/wgp_config.json` (per audit flag), since worker boot rewrites it via `source/models/wgp/orchestrator.py:198-220`

See `scripts/sprint4/upstream_prs/` (triage notes) and `docs/wan2gp-rebase-runbook.md` §8 for followup work.
