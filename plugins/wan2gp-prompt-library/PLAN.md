# Prompt Library/Injector Plugin — Implementation Plan

## Goals
- Store reusable prompts with categories/tags and quick search.
- Inject selected prompts into the current Gen page without copy/paste.
- Support both “Advanced Prompt” and “Wizard Prompt” modes.

## Key Integration Points
- Request core components:
  - `state`, `main_tabs`, `refresh_form_trigger`, `save_form_trigger`
  - `prompt` (advanced textbox), `wizard_prompt` (wizard textbox)
  - `wizard_prompt_activated_var` (which mode is active)
- Insert UI near prompt area using `insert_after("prompt", ...)` and/or add a new top-level “Prompt Library” tab for full management.

## Relevant Code References
- Prompt textboxes:
  - wgp.py:8673 → `prompt = gr.Textbox(...)`
  - wgp.py:8690 → `wizard_prompt = gr.Textbox(...)`
- Mode flag and events around prompt updates:
  - wgp.py:8691 → `wizard_prompt_activated_var = gr.Text(...)`
  - wgp.py:9390–9579 → event chain wiring that validates/saves prompt
- Plugin API:
  - shared/utils/plugins.py:78 → `class WAN2GPPlugin`
  - shared/utils/plugins.py:141 → `insert_after(target_component_id, new_component_constructor)`
  - shared/utils/plugins.py:427 → component injection and setup pipeline
- Tabs + component plumbing:
  - wgp.py:10594–10603 → `generate_video_tab` returns `generator_tab_components`
  - wgp.py:10692 → `app.setup_ui_tabs(...)`

## Data Model and Storage
- File: `plugins/wan2gp-prompt-library/library.json`
- JSON schema:
  - `version: string`
  - `items: Array<{ id, title, body, tags[], category, createdAt, updatedAt, favorite, modelTypes[] }>`
  - `collections: Array<{ id, name, description, itemIds[] }>`
  - Optional: `settings: { showInlinePanel: boolean, defaultInsertMode: "replace"|"append" }`
- Scoping:
  - Store globally; optionally filter by current model type (`state["model_type"]`).

## UI Composition
- New tab: “Prompt Library”
  - Search box, tag filters, category dropdown.
  - List/grid with preview; actions: Insert (Replace/Append), Edit, Duplicate, Favorite, Delete.
  - Import/Export JSON.
- Inline panel on Gen page (injected with `insert_after("prompt", ...)`)
  - Compact: search, recent/favorites, and quick “Insert” selector.
  - Mode-aware: if `wizard_prompt_activated_var` is true → target `wizard_prompt`, else → `prompt`.
  - Buttons:
    - Insert (Replace)
    - Insert (Append)
    - Apply + Go to Video Tab (uses `goto_video_tab` if needed)

## Event Wiring (No copy/paste)
- Backend flow (preferred for reliability):
  - `Insert` button handler:
    - Inputs: selected library item, `wizard_prompt_activated_var`, `prompt`, `wizard_prompt`
    - Logic:
      - Determine target textbox by `wizard_prompt_activated_var`
      - Return new string (replace or append) to appropriate output (`prompt` or `wizard_prompt`)
    - Then chain:
      - `.then(fn=self.goto_video_tab, inputs=[self.state], outputs=[self.main_tabs])` (optional)
      - Optionally trigger `save_form_trigger` or rely on normal validation chain
- Optional front-end enhancements:
  - `add_custom_js` to support:
    - Keyboard shortcut to open/insert
    - Cursor-position insert (advanced; otherwise append at end)
  - Hidden input/trigger pattern (as in Guides/Motion Designer) only if needed.

## Persistence and Settings
- Read `library.json` on setup; create if missing.
- All create/update/delete actions re-write atomic JSON.
- Optional: persist plugin preferences in `library.json` under `settings`.

## Search and Filtering
- Simple in-memory search and tag/category filters.
- Per-model-type filter using `state["model_type"]` so users see relevant prompts.

## Safeguards and UX
- Confirm on replace (toggleable).
- “Undo last insert” local cache (keep previous `prompt`/`wizard_prompt` for session).
- Validate that an item body is non-empty; show `gr.Info` if not.

## Extendability
- Negative prompt library: optional second tab/section with the same flows.
- Templates with variables
  - Support `{{var}}` placeholders; prompt users for values before insert.
  - Optionally integrate `shared.utils.prompt_parser` for macros later.

## Deliverables
- Folder `plugins/wan2gp-prompt-library/`
  - `__init__.py`
  - `plugin.py` (implements `WAN2GPPlugin`)
  - `library.json` (created on first run)
  - `README.md` (usage)
  - `PLAN.md` (this document)
- MVP features:
  - New tab + inline panel
  - Add/edit/delete prompts
  - Search/tag/category
  - Insert Replace/Append into correct textbox
  - Import/export library

## Step‑By‑Step Implementation
1) Scaffolding
- Create plugin package, metadata, and `setup_ui()`
- Request components: `state`, `main_tabs`, `prompt`, `wizard_prompt`, `wizard_prompt_activated_var`, `refresh_form_trigger`, `save_form_trigger`
2) Storage
- Implement `LibraryStore` class to load/save JSON; handle migrations by `version`
3) Tab UI
- Build full management UI and wire events to store
4) Inline Panel
- `insert_after("prompt", ...)` to create compact panel
- Wire Insert buttons to backend handlers returning updated prompt strings
5) Integration polish
- Optionally call `goto_video_tab` after insert
- Respect validation chains (don’t break existing flows)
6) Import/Export
- File upload/download handlers for JSON
7) Optional JS
- Add keyboard shortcut and small UX improvements via `add_custom_js`
8) Testing
- Sanity: plugin appears, prompts persist, insert works in both modes
- Confirm behavior with `refresh_form_trigger` updates and after model switches

## Open Questions/Assumptions
- Cursor-position insert: feasible via frontend JS; MVP defaults to append/replace.
- Discoverable anchors: `insert_after("prompt", ...)` is supported through components dict (not elem_id), per PluginManager’s `run_component_insertion_and_setup` logic.

## File References
- shared/utils/plugins.py:78
- shared/utils/plugins.py:141
- shared/utils/plugins.py:427
- wgp.py:8673
- wgp.py:8690
- wgp.py:8691
- wgp.py:9390
- wgp.py:10594
