# WanGP workflows

This guide is loaded on demand. Discover toolbox contracts for exact arguments and defaults. Returned contracts, capabilities and settings describe the active model and tools; these workflow examples explain how to combine them.

For a specific fact in a known document, use `mcp_resource(uri="...", query="topic or exact setting")`; `query="*"` lists its section headings. Read a returned heading with `section`. A full document or large section is read in bounded pages; follow `next_call` until `has_more=false` when complete coverage is requested. `limit` and `cursor` also apply to document reads and searches. IO `rg` searches accessible filesystem paths; `mcp_resource` searches resource URIs.

## Templates and generation

A named template or exact model takes priority. A named family requires choosing a compatible workflow and variant. Otherwise identify each generation step's category from the current request and retrieve its configured default template, including intermediate assets. Choose beyond it only for a missing capability, a requested declared speciality, or an explicit model-selection/comparison request. Use gen_video_with_refs for video conditioned on subject identity, appearance or motion from reference images or videos; use gen_video for text or start/end-frame video, and gen_video_with_speech for an existing speech clip. When references and speech are both required, start with gen_video_with_refs and verify the selected model supports the required audio behavior. Defaults belong to categories, not the last model used. Retain previous choices for continuation or revision of the same generation, not unrelated requests. Keep choices scoped to their task and category through compaction. Verify support for each supplied media kind. Standing Speed/Model Size preferences and generic “high quality” do not by themselves replace templates.

Use `wangp_models(query="<name>")` for a named model/family. Search specialities with `action="search", arguments={"filters":{"main_output":"image"},"specialities":["infographics"]}`; add strict family or required `capabilities` filters where needed. Known speciality terms need no catalog read; `action="specialities", arguments={}` lists current names/aliases. Exact speciality matches win; only if none exist are partial matches returned with missing and word-match evidence. Never relax required inputs. No speciality match may still permit the configured template.

Search results rank relevance first, then the number of standing preference matches. Prime uses current Speed/Model Size controls; standalone MCP is neutral. Request-specific overrides use `preferences` with `speed` (`fast`, `standard`, `any`) and `size` (`smaller`, `larger`, `any`). Explicit user choices win. Relative size is not a GB/VRAM measurement or quality score. Distinguish opening frames from identity/appearance references before choosing an architecture.

For the selected non-template model, read defaults and needed guidance through `wangp_model`: `definition` with `property="infos"` explains input combinations; `property="prompt_infos"` explains prompting. Independent reads can share one response. Query capabilities only for missing requirements or limits. Definition strings may be previews: request the property for its full value. Apply model defaults plus any selected accelerator, or the chosen template settings, then applicable `general_properties`, then explicit user overrides. Preserve flags for the chosen input modes and profile constraints. On input errors, recheck category and mode; never duplicate or invent media merely to satisfy inherited settings. Do not fetch pristine defaults after a template already supplies the needed values.

Shared setting meanings are in `wangp://docs/settings`: query one exact key, such as `image_mode`, or read a known section directly. The model definition contains model-specific declarations, not a dictionary of all setting meanings. A fixed normal `image_mode` may be omitted for an image-only model; keep the returned template rather than trying to reconstruct hidden defaults.

Pass the prepared settings directly with `wangp_generate(action="generate", arguments={"source": settings})`; read its contract only for missing options. `model_type`, media inputs and mode flags select the workflow. Preserve model flags and values not explicitly overridden. Keep prompt enhancement off unless requested. Generation and post-processing calls return their results when complete.

For Fast and `accelerated="profiles"`, list the selected model's `saved_settings`, read the `recommended` profile ID, and apply its complete settings delta over model defaults. Profiles sort by descending `profile_priority`; a unique positive maximum is recommended. If none is recommended, consult model help instead of choosing alphabetically. Native acceleration already uses accelerated defaults: do not automatically add another profile. Standard/No preference does not automatically add acceleration. An explicitly requested profile wins if compatible. Recheck constraints when overriding steps, resolution or mode. Do not list LoRAs separately when the profile supplies identifiers and multipliers; `profile_priority` is metadata, not a generation setting.

Two existing workflows remain valid: one generation call containing a batch of settings/tasks, or multiple independent tool calls emitted together. A batch remains one queue submission. Multiple calls keep separate results and call IDs; Deepy executes them in order before the next local LLM pass. Do not force separate requests into a batch or split an existing batch merely for tool presentation.

## Media selection and inputs

Image-to-video is a broad workflow. A start image anchors the opening frame, including its scene and composition. Reference images guide subject identity or appearance according to the model's mode without fixing the opening frame; injected frames occupy specified positions. Template responses include the model's supported `media_inputs`; these roles do not say which inputs are already populated. Broad model search uses `main_output="video", inputs="image"`; the `image_to_video` capability specifically means opening-frame support, while `reference_images` means image reference conditioning. `reference_videos` and `media_inputs.video.reference` identify video references, distinct from continuation and spatial control. Read model input guidance for reference/control modes and video inputs such as `video_guide` and `video_guide2`.

For selected media, call `wangp_list_gallery(action="list", arguments={"selected_only":true})`; omit the filter for current/remembered media. Returned `media_id` values are inputs to generation, post-processing and media utilities. Generation's media fields also accept authorized returned paths, including `output_file` or `generated_files`; use these directly when a result supplies paths. Display labels and abbreviated filenames are not paths. Do not list the Gallery just to convert a usable path to an ID. Use the media toolbox's `media_settings` action for a previous generation's full settings; IO `info` provides missing physical metadata, with `path` for one reference or `paths` for 1–20 references in input order.

For a named subfolder, try its path directly under the known root, such as IO `list` with `path="@outputs/selection"`. If a recursive search is needed, `rg --files -g '**/selection/**' -- @outputs` finds files inside matching folders; `-g '*selection*'` matches basenames. Reuse exact paths from the result for subsequent inspection.

Read model capabilities and, when necessary, `wangp://docs/settings/prompt-flags`. WanGP infers compatible start/end/continuation/reference/audio flags when media is provided without flags. Do not guess unsupported flags:

- Start/end images use `image_start` / `image_end` and supported `S` / `E` image flags.
- Reference images use `image_refs=[media_id_or_authorized_path]` with the complete template `video_prompt_type` containing `I`, such as `KI`. Preserve that mode for the same input use. Query `image_ref_choices` only when choosing a different reference role; its `K` meaning is model-specific. Single/multiple-reference limits still apply.
- Background, injected frames, controls and masks require the declared `K`, `F`, `V` or `A` modes and associated inputs.
- Video inputs use `video_source`, `video_guide` or `video_mask` according to capabilities.
- Audio prompt input does not imply audio output; verify output capabilities separately.

## Post-processing and media utilities

Discover post-processing with a media type (`image`, `video`, `audio`), Gallery ID or authorized path. A type suffices to discover actions and contracts; execution requires actual media. Action without arguments describes; an arguments object executes. Disabled processors are omitted.

The media toolbox inspects, compares, transcribes, extracts, trims, resizes/crops, mutes, replaces audio, composes side by side and merges media. Inspect Media handles explicitly selected visuals; Inspect Video samples a time range. Extract frames first only when saved Gallery images are needed. Outputs follow the normal Gallery publication path; `add_to_gallery` supports authorized existing files.

For a requested named processor, discover with the known media type before generation when availability affects the plan. Read that processor's contract, then reuse it with the generated media ID/path. Do not search the generation model catalog for a post-processor or generate an intermediate copy just to inspect its options.

## Speech and chained media

Apply the model-selection policy above to these workflows, including explicit model choices and requested specialities. For a video with spoken words supplied as text, start with the configured `gen_video_with_refs` template when using subject references, otherwise `gen_video`, and check audio-output capabilities. If the selected model supports native speech, keep the exact spoken words in its video prompt and follow model-specific `prompt_guidance`; a separate TTS job is unnecessary. Text-to-video support also avoids a preliminary start image unless the user requests one or the workflow needs it.

When the user supplies or explicitly requests a separate speech clip, use `gen_video_with_speech` with that existing/generated audio and the requested start image. If the request also needs subject references, follow the `gen_video_with_refs` routing above and verify the audio mode. A requested portrait and speech sample are independent assets: their generation calls may share a response, but the final video depends on both results. For a new voice use `gen_speech_from_description`; use `gen_speech_from_sample` only with an actual voice sample. Preserve each model's input mapping from its capabilities/settings.

An audio input is not proof of lip sync or motion conditioning. For specialized editing models, read their declared audio modes and required reference preparation; some use audio as the output soundtrack. Resolve an unsupported requested behavior before producing dependent assets.

## Extraction by spoken words

Resolve the selected video once. Use `transcribe_media` directly on it with word timestamps to locate the requested phrase; no audio extraction or visual inspection is needed just to find spoken words. Pass the matching start time to `extract_video`. When only a starting phrase is specified, use the original end unless context specifies an endpoint. If the phrase is absent or ambiguous, resolve that specific uncertainty instead of guessing a timestamp.

## Visual verification

When verification is requested, inspect the produced image using `inspect_media` with the user's actual criteria: content, readable labels, spelling, composition, anatomy and artifacts. For a large infographic, inspect the whole scene first and zoom into text or suspicious regions when the first pass cannot establish correctness. Use `bbox` for targeted inspection instead of creating crop files solely for review. Each `media_inputs` image or video frame can have its own `bbox`; omitted boxes use the shared `bbox` or the full visual. Fix observed flaws and inspect the affected regions again; do not repeatedly regenerate a satisfactory image or claim perfect verification beyond what was visible.

## Long video and sliding windows

Choose the workflow before generating assets. For a fully planned sequence, prefer **one planned sliding-window video generation**. Repeated continuation is for deciding the next scene after reviewing the latest clip; generating each planned shot separately and extracting its last frame is not the planned end-frame method.

Read capabilities with `wangp_model(model_type=<chosen model>, action="capabilities", arguments={})`. `metadata.frames_maximum` limits one window, not total video length when `sliding_window` is supported. Keep `sliding_window_size` at or below that limit. `video_length` accepts the total as a seconds string, e.g. `"60s"`; WanGP converts and normalizes frames. Set numeric `force_fps` only for requested FPS, otherwise keep the model default. Preserve template/standing dimensions unless overridden. Follow returned `prompt_guidance` for model-specific prompt structure.

When `metadata.media_inputs.image` includes `end`, prepare anchors before the video:

1. For End Frames without separate identity or motion references, start with the configured `gen_video` template and verify end-frame support. Reference inputs used to edit anchors do not change the video category. Plan each window's action, transition and end-frame composition. Generate one master image and prepare anchor edits using the selection policy above; the normal choices are the configured `gen_image` and `edit_image` templates.
2. Edit the **same master image independently** for every planned end frame, preserving identity and clothing while changing pose, camera and environment. These edits can share a batch or be emitted as independent calls together, once the master exists. Do not chain edits from previous edited outputs.
3. Pass the master in `image_start` and the ordered anchors in `image_end` as a list of media IDs or authorized output paths, one per window. Use supported `image_prompt_type="SE"` (`E` alone without a start image). For the anchor edits, pass `image_refs=[master_media_id_or_path]` and retain the edit template's complete reference `video_prompt_type`, such as `KI`.
4. Execute `wangp_generate(action="generate", arguments={"source": settings})` with one video settings object containing total `video_length`, the window plan and anchors. Each window's motion must lead toward its corresponding anchor. WanGP joins its sliding windows into one output.

Set `multi_prompts_gen_type="W"` for one non-empty line per window, or `"PW"` for one paragraph per window. With `PW`, keep labeled sections within a window adjacent with single newlines and exactly one blank line between complete windows. Verify paragraph count equals planned windows and `image_end` count.

Window prefixes can set contributed duration (`[/duration=5s]`, `[/duration=121]`, `[/duration=20%]`) and overlap (`[/overlap=9]`; `[/overlap]` restores the default). Duration commands define the window schedule and predicted total; their sum must match the requested duration. Keep each generated window, including overlap, within the model limit. Use `[/new_shot]` or `[/overlap=0]` only for an intended hard cut supported by text-to-video; preserve overlap for a continuous shot. Read `wangp://docs/processing`, section `During Generation: Temporal And Spatial Processing > Long Videos Workflows`, only for additional scheduling and transition details.

If end frames are unsupported but `injected_frames` is declared, intermediate anchors use `image_refs`, `frames_positions` and a declared `video_prompt_type` containing `F`. If the chosen model cannot satisfy the planned method, resolve that limitation before generating assets; do not silently substitute a different method or model.

For interactive continuation with declared support, pass the latest video in `video_source` with `image_prompt_type` containing `V`. Limit each new portion to one window. The result already contains its source and continuation: use that combined output next and do not merge the source twice. Repeated encoding can reduce quality.

## Prompt files and long documents

Use the session workspace for working text. IO `append_text` creates missing UTF-8 files or appends literal text; include intended newlines. IO `edit` replaces an exact passage in an existing file: search for enough context to identify one occurrence and preserve whitespace/line endings. `write_text` provides exclusive creation or explicit complete overwrite.

These IO actions use `path`; `rg` takes a `command` string. Keep action at the top level and its parameters inside `arguments`, for example `wangp_io(action="rg", arguments={"command":"-n -F needle -- @workspace/story.md"})`. Their contracts include complete call examples. A successful edit's `replacements` and content hash confirm the write; file size can remain unchanged. For chapter appends, use the returned heading receipt to check the expected new heading and uniqueness without an extra search; investigate mismatches.

For long generation prompts, read `wangp://skills/long-generation-prompts`, prepare a text file, then pass an exact prompt reference such as `@file("@workspace/prompt.txt")`. WanGP snapshots authorized UTF-8 contents, preserving blank-line window boundaries. Prompt files require read permission, not write permission, and have a 4 MiB limit.

For long stories, read `wangp://skills/long-story-writing` before outlining. Keep the manuscript, outline and compact canon in workspace files. Track chronology, character knowledge, evidence, causal dependencies and unresolved promises during drafting; verify them against relevant manuscript passages. Reuse facts in active context and group related searches. Discover IO actions once and reuse their contracts. Skill examples describe action arguments inside `wangp_io`; they are not separate top-level tools.

## File search and pagination

IO `list` navigates one directory or lists roots; IO `rg` searches filenames or content using its discovered ripgrep syntax. Prefer targeted or names-only searches. Use `info` for metadata and ranged `read_text` for complete lines when an excerpt is truncated.

Results contain `count`, `has_more` and `next_cursor`; retain the query filters and pass the cursor to continue. Pages come from stored snapshots, expire after ten minutes, and may be evicted by newer searches. After expiry, start a new search instead of treating its first page as a continuation. `summary_only` reports the stored count without loading the collection. Interrupted/failed searches report `complete=false` even if matches were found.

Small template catalogs return the usage, configured default and template names once. Only an incomplete catalog includes `next_call`: repeat the same toolbox with that object to continue. Distinct display labels appear separately; identical labels are omitted.

Prefer the session workspace for experiments, drafts and any files you expect to edit; Prime manages it freely. Outputs and their subfolders allow new files only when write access is enabled. Existing output files cannot be modified, appended to, overwritten, deleted, renamed or moved, whether or not they appear in Gallery. Other folders follow configured R/RW permissions; deleting or moving a source outside outputs removes its vanished path from Gallery. In final answers, reference Gallery IDs or exact authorized paths; WanGP creates links.
