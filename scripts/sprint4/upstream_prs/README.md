# Sprint 4 Upstream Draft Inventory

Drafts only - no `gh pr create` is invoked. GitHub PR creation is orchestrator-owned and is not performed from this repository batch.

The approved Sprint 4 artifact is kept as a 14-theme inventory derived literally from `docs/wan2gp-triage.csv`. The current audit also found that the older "missing upstream items" list is stale: the current submodule already contains the additive sync files that originally motivated this deliverable, so these documents are review packets rather than PR-ready bodies. Treat the three unresolved symbols below as investigation tickets, not PR-ready changes.

Remaining investigation-only symbols from the audit:
- `_apply_gamma_to_media`
- `LTX2_OUTPAINT_GAMMA`
- `LTX2_DISABLE_STAGE2_WITH_CONTROL_VIDEO`

CSV theme map:
- `01-ltx2-hotzone-drift.md` -> `ltx2 hot-zone implementation drift vs upstream`
- `02-wan-runtime-drift.md` -> `Wan runtime drift vs upstream`
- `03-shared-runtime-drift.md` -> `shared runtime drift vs upstream`
- `04-shared-utility-drift.md` -> `shared utility drift vs upstream`
- `05-model-implementation-drift.md` -> `model implementation drift vs upstream`
- `06-tts-implementation-drift.md` -> `TTS implementation drift vs upstream`
- `07-llm-engine-drift.md` -> `LLM engine drift vs upstream`
- `08-gradio-ui-drift.md` -> `Gradio UI drift vs upstream`
- `09-plugin-integration-drift.md` -> `plugin integration drift vs upstream`
- `10-preset-config-drift.md` -> `preset/config drift vs upstream`
- `11-documentation-drift.md` -> `documentation drift vs upstream`
- `12-installer-update-script-drift.md` -> `installer/update script drift vs upstream`
- `13-project-metadata-drift.md` -> `project metadata drift vs upstream`
- `14-content-drift.md` -> `content drift vs upstream`
