"""Progressive MCP v2 presentation over the shared WanGP operations."""

import copy
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from mcp.server.fastmcp.utilities.func_metadata import func_metadata

from shared.mcp_paging import PAGE_SIZE, ResultPages, strip_schema_titles


RESOURCE_ALIASES = {"wangp://guides/v2": "wangp://guides/workflows"}


PAGING = {
    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": PAGE_SIZE},
    "cursor": {"type": ["string", "null"], "default": None},
    "summary_only": {"type": "boolean", "default": False},
}
PAGING_HELP = "Pages have count/volume limits. Continue with unchanged filters and cursor=next_cursor; cursors expire after 10 minutes or eviction. summary_only returns the stored count without items."
DEEPY_USAGES = {
    "gen_image": "Create images from text",
    "edit_image": "Edit images using reference images",
    "gen_video": "Generate video from a prompt, including dialogue with audio-capable models",
    "gen_video_with_speech": "Generate video driven by an existing speech clip",
    "gen_video_with_refs": "Generate video conditioned on reference images or videos for subjects, appearance or motion; supported media depend on the template",
    "gen_song": "Generate music or songs",
    "gen_speech_from_description": "Generate speech using a text-described voice",
    "gen_speech_from_sample": "Generate speech using a voice sample",
}
MEDIA_DISCOVERY_DESCRIPTIONS = {
    "create_color_frame": "Create a solid-color Gallery image.",
    "extract_audio": "Extract audio, optionally a time range.",
    "extract_video": "Extract a video segment.",
    "inspect_media": "Answer a required question about images via media_id/media_ids or exact video frames via media_inputs, with optional crops; no frame extraction needed.",
    "inspect_video": "Inspect a video time range via media_id using automatically sampled frames.",
    "merge_videos": "Join two videos end to end.",
    "transcribe_media": "Transcribe speech with timestamps.",
}


def action_def(description, properties=None, required=()):
    return {"description": description, "parameters": {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}}


def public_media_result(result, media_records=()):
    """Expose one Gallery identity; keep restoration aliases in internal state."""
    from shared.utils.gallery_media import gallery_media_ids

    identities = {}
    for record in media_records:
        ids = gallery_media_ids(record["path"], "audio" if record["media_type"] == "audio" else "visual", record.get("settings"))
        identities.update({value: ids[0] for value in [record["media_id"], *ids]})
    private = {"gallery_media_ids", "deepy_media_id", "deepy_media_fingerprint", "deepy_session_id"}

    def clean(value, key=""):
        if isinstance(value, dict):
            return {name: clean(child, name) for name, child in value.items() if name not in private}
        if isinstance(value, list):
            return [clean(child, key) for child in value]
        if isinstance(value, str) and key.endswith(("media_id", "media_ids")):
            return identities.get(value, value)
        return value

    return clean(result)


def function_def(function, description=None, exclude=()):
    schema = strip_schema_titles(func_metadata(function, skip_names=list(exclude), structured_output=False).arg_model.model_json_schema())
    for name in exclude:
        schema["properties"].pop(name, None)
    schema["required"] = [name for name in schema.get("required", []) if name not in exclude]
    schema["additionalProperties"] = False
    return {"description": description or function.__doc__.strip(), "parameters": schema}


def paginated(definition, *, continuation_only=False):
    definition = copy.deepcopy(definition)
    definition["parameters"]["properties"].update(copy.deepcopy(PAGING))
    if continuation_only:
        definition["_continuation_only"] = True
    else:
        definition["limitations"] = PAGING_HELP
    return definition


def invocation(actions, action, arguments):
    if action is None:
        if arguments is not None:
            raise ValueError('arguments requires an action at the top level, beside arguments: {"action":"<action>","arguments":{...}}. Omit both to discover actions.')
        compact = {name: value.get("_summary", value["description"].split(". ", 1)[0].rstrip(".") + ".") for name, value in actions.items()}
        for name, value in actions.items():
            if not value["parameters"]["properties"]:
                compact[name] += " Execute with arguments={}."
        return {"status": "discovery", "actions": compact, "usage": "Describe: action + arguments=null. Execute: action + arguments={...}."}
    if action not in actions:
        raise ValueError(f"Unknown or unavailable action: {action}. Omit action to discover available actions.")
    definition = actions[action]
    if arguments is None:
        return {"status": "schema", "action": {"name": action, **action_contract(definition)}}
    errors = sorted(Draft202012Validator(definition["parameters"]).iter_errors(arguments), key=lambda error: str(list(error.path)))
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.path)) or "arguments"
        if action == "inspect_media" and error.validator == "oneOf":
            raise ValueError("inspect_media requires exactly one of media_id (one visual), media_ids (a list), or media_inputs (selected frames), inside arguments alongside question.")
        if action == "inspect_media" and "bbox" in error.path:
            raise ValueError(f"{path}: {error.message}. bbox=[x_min,y_min,x_max,y_max], integers 0..1000 relative to the full image; x_max>x_min and y_max>y_min.")
        required = ", ".join(definition["parameters"].get("required", [])) or "none"
        if "example" in definition:
            raise ValueError(f"{path}: {error.message}. Required arguments: {required}. Call structure: {json.dumps(definition['example'], ensure_ascii=False)}")
        raise ValueError(f"{path}: {error.message}. Required arguments: {required}. To read the contract, repeat this call with action={action!r} and arguments=null (the entire object, not a nested value).")
    return None


def action_contract(definition):
    result = {key: value for key, value in definition.items() if key not in {"_continuation_only", "_summary"}}
    result["parameters"] = strip_schema_titles(definition["parameters"])
    if definition.get("_continuation_only"):
        for key in PAGING:
            result["parameters"]["properties"].pop(key, None)
    return result


def register_v2(mcp, session, operations, jobs, policy, get_toolbox, *, downloads=False, allow_async=False, defer_wait=False, deepy_help=False):
    from shared import mcp_server as core
    from shared.deepy import filesystem, long_text
    from shared.deepy.paged_files import directory_entries, rg_records
    from shared.model_selection import normalize_preferences, rank_models, speciality_catalog

    pages = ResultPages()
    mcp._wangp_pages = pages
    mcp._wangp_jobs = jobs
    mcp._wangp_allow_async = allow_async

    def collection(query, records, args, key="items", metadata=None):
        return pages.page(query, records, key=key, limit=args.get("limit", PAGE_SIZE), cursor=args.get("cursor"), metadata=metadata, summary_only=args.get("summary_only", False))

    def known_model(model_type):
        if session.get_model_metadata(model_type) is None:
            raise ValueError(f"Unknown model_type: {model_type}")

    model_filters = {name: {"type": "string"} for name in ("family", "base_model_type", "finetune", "model_type", "main_output", "inputs", "name")}
    model_filters["inputs"]["description"] = "Accepted media kind. For broad image-conditioned video discovery combine inputs='image' with main_output='video'; check media_inputs for the supported image roles."
    model_filters["capabilities"] = {"type": "array", "items": {"type": "string"}, "description": "Require every named capability. image_to_video means opening-frame support; reference_images means image reference conditioning; reference_videos means video reference conditioning, distinct from continuation or spatial control. Other examples: audio_output, video_continuation. Never relaxed by speciality matching."}
    filter_schema = {"type": "object", "properties": model_filters, "additionalProperties": False}
    search_def = paginated(action_def("Find models by name, strict capabilities or specialities, ranked by speed/size preferences. query is a literal case-insensitive substring of name, ID, family or description; filters stay strict. specialities matches all named strengths or aliases; only when no exact matches exist, returns partial matches with missing/word-match evidence. Results sort by speciality relevance, then matching speed/size preferences (both, one, neither). Prime inherits its live preferences; standalone MCP is neutral. Override preferences for this request only, using 'any' for no preference. accelerated='native' uses defaults; 'profiles' offers accelerator profiles; 'none' has none. size='lighter'/'large' is relative, not GB or a quality score.", {"query": {"type": "string", "default": ""}, "filters": filter_schema, "specialities": {"type": "array", "items": {"type": "string", "minLength": 1}}, "preferences": {"type": "object", "properties": {"speed": {"type": "string", "enum": ["fast", "standard", "any"]}, "size": {"type": "string", "enum": ["smaller", "larger", "any"]}}, "additionalProperties": False}}))
    specialities_def = paginated(action_def("List declared model specialities and aliases, optionally filtered by model capabilities. Use returned names in search.specialities. Known terms can be searched directly; no catalog read is required.", {"filters": filter_schema}))

    @mcp.tool()
    def wangp_models(action: str | None = None, arguments: dict[str, Any] | None = None, query: str | None = None) -> dict[str, Any]:
        """Find models by name (query alone), or discover search filters, speciality matching and speed/size preferences. action='specialities', arguments={} lists declared strengths when terminology is unknown."""
        if query is not None:
            if action is not None or arguments is not None:
                raise ValueError("Pass query alone, or use action='search' with query inside arguments alongside advanced filters.")
            action, arguments = "search", {"query": query}
        response = invocation({"search": search_def, "specialities": specialities_def}, action, arguments)
        if response is not None:
            return response
        filters = dict(arguments.get("filters", {}))
        required = filters.pop("capabilities", [])
        filters["query"] = arguments.get("query", "") or None
        specialities = arguments.get("specialities", [])
        preferences = {}
        if deepy_help and not arguments.get("cursor"):
            from shared.deepy.ui_settings import normalize_assistant_tool_ui_settings
            settings = normalize_assistant_tool_ui_settings(**get_toolbox().session.tool_ui_settings)
            preferences = {"speed": settings["model_speed"], "size": settings["model_size"]}
        preferences = normalize_preferences({**preferences, **arguments.get("preferences", {})})
        records, metadata = None, None
        if not arguments.get("cursor"):
            records = [record for record in session.list_model_metadata(include_selection=True, **filters) if all(record["capabilities"].get(key, False) for key in required)]
            if action == "specialities":
                records = speciality_catalog(records)
            else:
                records, match = rank_models(records, specialities, preferences)
                metadata = {"match": match, "preferences": preferences}
                if match == "none" and specialities:
                    metadata["note"] = "No declared speciality matches within the required filters. A capable default template may still meet the request."
                records = [core._compact_deepy_model_metadata(item) for item in records]
        result = collection(["models", action, filters, required, specialities, arguments.get("preferences", {})], records, arguments, "specialities" if action == "specialities" else "models", metadata=metadata)
        if result["has_more"]:
            result["next_call"] = {"action": action, "arguments": {**arguments, "cursor": result["next_cursor"]}}
        return result

    model_actions = {
        "capabilities": action_def("Read the model's capabilities, parameter usage and generation limits."),
        "definition": action_def("Read model declarations: property='infos' explains inputs and their combinations; 'prompt_infos' explains prompting. Shared setting meanings: wangp://docs/settings. Omit property only to explore declarations. Root strings longer than 256 characters are previews; request the indicated property for its full value.", {"property": {"type": "string"}}),
        "defaults": action_def("Read pristine generation defaults for this model; saved UI settings are separate."),
        "saved_settings": paginated(action_def("List saved settings, accelerator profiles and presets, or read one by setting_id. Accelerators sort by descending profile_priority; a unique highest positive priority is recommended. For fast generation on accelerated='profiles', read that profile and apply its settings over model defaults. No recommended flag means consult model help, not filename order. Native acceleration needs no automatic extra profile.", {"setting_id": {"type": "string"}})),
        "loras": paginated(action_def("Find locally available LoRAs for this model. name accepts case-insensitive * and ? globs; returned identifiers go in activated_loras with loras_multipliers.", {"name": {"type": "string"}})),
    }

    @mcp.tool()
    def wangp_model(model_type: str, action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Explore a chosen model's capabilities, limits, definition, defaults, saved settings and LoRAs."""
        known_model(model_type)
        response = invocation(model_actions, action, arguments)
        if response is not None:
            return response
        if action in {"capabilities", "definition", "defaults"}:
            if action == "definition":
                try:
                    return core._mcp_model_definition(session.get_model_def(model_type), property_name=arguments.get("property"), string_limit=256, deepy_help=deepy_help)
                except KeyError as error:
                    if arguments.get("property") in {"defaults", "capabilities"}:
                        action = arguments["property"]
                    else:
                        call = {"model_type": model_type, "action": "definition", "arguments": {}}
                        raise ValueError(f"{error.args[0]} for {model_type}. Read available properties: call wangp_model with {json.dumps(call, ensure_ascii=False)}. Shared setting meanings: wangp://docs/settings.") from error
            result = operations["wangp_model"](model_type, view="schema" if action == "capabilities" else action)
            if action == "capabilities":
                result["metadata"].update(session.get_model_selection_metadata(model_type))
                model_def = session.get_model_def(model_type)
                for property_name, route in (("infos", "input_guidance"), ("prompt_infos", "prompt_guidance")):
                    help_text = model_def.get(f"deepy_{property_name}", model_def.get(property_name)) if deepy_help else model_def.get(property_name)
                    if help_text:
                        result[route] = {"action": "definition", "arguments": {"property": property_name}}
            return result
        if action == "saved_settings" and "setting_id" in arguments:
            if arguments.get("cursor"):
                raise ValueError("A single setting_id cannot be combined with a cursor.")
            return operations["wangp_model_settings"](model_type, arguments["setting_id"])
        name = arguments.get("name")
        result = None if arguments.get("cursor") else operations["wangp_list_loras"](model_type, name) if action == "loras" else operations["wangp_model_settings"](model_type)
        key = "loras" if action == "loras" else "settings"
        return collection(["model", model_type, action, name], None if result is None else result[key], arguments, key)

    tool_ids = list(DEEPY_USAGES)
    template_actions = {
        "deepy_template_settings": action_def('Read default settings and supported media_inputs directly with arguments={"tool_id":"<chosen tool_id>","template":"default"}; no template listing is needed. tool_id identifies a Deepy usage from tool_ids in discovery. Both fields are required; template may instead be a returned name. media_inputs describes supported model roles, not populated settings: image start/end anchor frames, reference conditions appearance, injected_frames inserts frames at specified positions; video reference guides appearance or motion rather than continuing a clip. Use model help for mode-specific input combinations. Apply settings, then returned general_properties, then explicit user overrides; query capabilities only for missing limits.', {"tool_id": {"type": "string", "enum": tool_ids}, "template": {"type": "string", "minLength": 1}}, ("tool_id", "template")),
        "deepy_templates": paginated(action_def("List named templates only to choose an alternative for one tool_id. Choose the Deepy usage from tool_ids in discovery. Returns its default_template and template names; labels appear only when different. Large lists return next_call to continue with unchanged filters.", {"tool_id": {"type": "string", "enum": tool_ids}}, ("tool_id",)), continuation_only=True),
    }

    @mcp.tool()
    def wangp_deepy_templates(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Discover Deepy recipes for image, video, voice and music, including configured defaults and their generation settings."""
        response = invocation(template_actions, action, arguments)
        if response is not None:
            if action is None:
                response["tool_ids"] = DEEPY_USAGES
            return response
        if action == "deepy_template_settings":
            result = operations["wangp_get_deepy_template_settings"](**arguments)
            result["media_inputs"] = core._compact_deepy_model_metadata(session.get_model_metadata(result["settings"]["model_type"]))["media_inputs"]
            return result
        tool_id = arguments["tool_id"]
        records, metadata = None, None
        if not arguments.get("cursor"):
            group = operations["wangp_list_deepy_templates"](tool_id)[0]
            metadata = {"tool_id": tool_id, "default_template": group["default_template"]}
            records = ({"template": item["template"], **({"label": item["label"]} if item["label"] != item["template"] else {})} for item in group["templates"])
        page = collection(["deepy_templates", tool_id], records, arguments, metadata=metadata)
        result = {"status": "done", "tool_id": tool_id, "default_template": page["default_template"], "deepy_templates": [item["template"] for item in page["items"]]}
        labels = {item["template"]: item["label"] for item in page["items"] if "label" in item}
        if labels:
            result["labels"] = labels
        if page["has_more"]:
            result["next_call"] = {"action": action, "arguments": {"tool_id": tool_id, "cursor": page["next_cursor"], **({"limit": arguments["limit"]} if "limit" in arguments else {})}}
        if "summary" in page:
            result["summary"] = page["summary"]
        return result

    gallery_def = paginated(action_def("List current and remembered Gallery media, alternating image/video then audio, newest-first within each gallery. Optionally filter by media_type or only live selections. Returned media_id values identify inputs to other tools.", {"media_type": {"type": "string", "enum": ["all", "image", "video", "audio"], "default": "all"}, "selected_only": {"type": "boolean", "default": False}}))

    @mcp.tool()
    def wangp_list_gallery(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Find current and remembered images, videos and audio, including the user's live Gallery selections."""
        response = invocation({"list": gallery_def}, action, arguments)
        if response is not None:
            return response
        kind, selected = arguments.get("media_type", "all"), arguments.get("selected_only", False)
        records = None
        if not arguments.get("cursor"):
            core._gallery_records(session, media_type=kind, limit=500)
            with core._GALLERY_LOCK:
                snapshot = copy.deepcopy(list(core._gallery_history(session).values()))
            ordered = [record for record in reversed(snapshot) if (kind == "all" or record["media_type"] == kind) and (not selected or record["selected"])]
            if kind == "all":
                visual = (record for record in ordered if record['gallery'] == 'visual')
                audio = (record for record in ordered if record['gallery'] == 'audio')
                ordered = (record for pair in zip_longest(visual, audio) for record in pair if record is not None)
            records = (core._compact_gallery_record(record) for record in ordered)
        return collection(["gallery", kind, selected], records, arguments, "media")

    def io_definitions():
        actions = {item["name"]: {"description": item["description"], "parameters": copy.deepcopy(item["parameters"])} for item in filesystem.available_io_actions(policy, downloads)}
        for name in ("search_text", "write_artifact_text"):
            actions.pop(name, None)
        actions["info"] = action_def("Read physical file, directory or media metadata using path, or batch up to 20 paths. Each accepts an authorized path or Gallery media ID. Supply exactly one of path or paths; batch results appear in files in input order. Use exact returned paths/IDs, not display labels or shortened filenames.", {"path": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20}})
        actions["info"]["parameters"]["oneOf"] = [{"required": ["path"]}, {"required": ["paths"]}]
        if policy.read_enabled:
            actions["list"] = paginated(action_def('Navigate one authorized directory or list accessible roots. For a named subfolder, try its path directly, e.g. path="@outputs/selection"; no need to enumerate its parent. Use rg for recursive search; detailed metadata is available through info.', {"path": {"type": "string", "default": ""}, "pattern": {"type": "string", "default": "*"}, "media_type": {"type": "string", "enum": ["all", "image", "video", "audio", "txt", "other"], "default": "all"}}))
            actions["rg"] = paginated(action_def("Search filenames or text with ripgrep. command contains options without the rg executable, then -- and authorized paths; omit paths to search the workspace. Filenames: --files -g '*.txt' -- @outputs (no text pattern). Content: -n -F -e 'needle' -e 'other' -- @workspace (one positional pattern or repeated -e). -l lists files containing matching text and needs a pattern. To find files inside a named folder use -g '**/selection/**', not -g '*selection*'; -i affects text matching, not glob casing. -m limits matches per file, not the page. Hidden/ignored files require explicit rg options. Excerpts are bounded; use read_text for complete lines.", {"command": {"type": "string"}}, ("command",)))
            actions["rg"]["_summary"] = "Search via arguments.command: --files -g '*.txt' -- @outputs lists filenames; -n -F 'needle' -- @workspace searches text."
        if policy.write_enabled:
            actions["edit"] = function_def(long_text.edit_text, "Replace exact literal text in an existing UTF-8 file. old_string must occur once unless replace_all=true; whitespace and line endings are literal. A successful result reports replacements and a content hash; equal file size does not mean equal content.", exclude=("policy",))
            actions["append_text"] = function_def(long_text.append_text, "Create a missing UTF-8 file or append exact literal text to an existing file. Parent folders are created; no newline is added implicitly. Appends containing Markdown headings return a compact heading receipt for boundary and duplicate checks.", exclude=("policy", "include_structure"))
            for name in ("edit", "append_text"):
                schema = actions[name]["parameters"]
                schema["properties"]["path"] = schema["properties"].pop("file_path")
                schema["required"] = ["path" if key == "file_path" else key for key in schema["required"]]
            actions["write_text"]["parameters"]["properties"]["mode"]["enum"] = ["create", "overwrite"]
            actions["write_text"]["description"] = "Write complete text using exclusive creation or explicit overwrite. Use append_text to append; existing parent folders are required."
            actions["zip"]["parameters"]["properties"]["sources"] = {"type": "array", "items": {"type": "string"}, "minItems": 1}
        for definition in actions.values():
            definition["parameters"]["additionalProperties"] = False
        examples = {
            "info": {"paths": ["@outputs/clip.mp4", "@outputs/voice.wav"]},
            "rg": {"command": '-n -F "needle" -- @workspace/story.md'},
            "read_text": {"path": "@workspace/story.md"},
            "write_text": {"path": "@workspace/outline.md", "text": "# Outline\n"},
            "append_text": {"path": "@workspace/story.md", "text": "\n\n## Chapter Two\n\nChapter text.\n"},
            "edit": {"path": "@workspace/story.md", "old_string": "exact existing text", "new_string": "replacement text"},
        }
        for name, arguments in examples.items():
            if name in actions:
                actions[name]["example"] = {"action": name, "arguments": arguments}
        return actions

    io_actions = io_definitions()

    @mcp.tool()
    def wangp_io(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Navigate authorized folders, search files or text, read and edit documents, manage files and archives, and inspect physical metadata."""
        if arguments is not None:
            arguments = dict(arguments)
            aliases = {"file_path": "path"} if action in io_actions and "path" in io_actions[action]["parameters"]["properties"] else {}
            if action == "info":
                aliases["source"] = "path"
            if action == "rg":
                aliases["arguments"] = "command"
            for alias, name in aliases.items():
                if alias in arguments:
                    value = arguments.pop(alias)
                    if name in arguments and arguments[name] != value:
                        raise ValueError(f"{action}: {name} and its legacy alias {alias} must not specify different values.")
                    arguments[name] = value
        response = invocation(io_actions, action, arguments)
        if response is not None:
            if action is None:
                response["roots"] = policy.roots()
                from shared.deepy.prime_filesystem import PrimeFileAccessPolicy
                if isinstance(policy, PrimeFileAccessPolicy):
                    response["file_rules"] = "Prefer the session workspace for experiments and edits. Outputs allow new files only with write access; existing output files cannot be changed, deleted or moved. Other folders follow configured R/RW access."
            return response
        args = dict(arguments)
        if action in {"list", "rg"}:
            filters = {name: definition["default"] for name, definition in io_actions[action]["parameters"]["properties"].items() if name not in PAGING and "default" in definition}
            filters.update({name: value for name, value in args.items() if name not in PAGING})
            metadata = {"complete": True}
            records = None
            if not args.get("cursor"):
                records = directory_entries(policy, **filters) if action == "list" else rg_records(policy, filters["command"], metadata)
            return collection(["io", action, filters], records, args, "entries" if action == "list" else "matches", metadata)
        with core._GALLERY_LOCK:
            if action == "info":
                if "paths" in args:
                    files = [core._run_io_action(session, policy, action, {"source": path}, downloads) for path in args["paths"]]
                    return {"status": "error" if any(item["status"] == "error" for item in files) else "done", "files": files}
                args["source"] = args.pop("path")
            if action in {"move", "delete"}:
                from shared.deepy.prime_filesystem import PrimeFileAccessPolicy
                if isinstance(policy, PrimeFileAccessPolicy):
                    source = policy.require_mutation(args["source" if action == "move" else "path"])
                    try:
                        return core._run_io_action(session, policy, action, args, downloads)
                    finally:
                        toolbox = get_toolbox()
                        if core._prune_deleted_gallery_files(session, source, toolbox.session):
                            toolbox.send_cmd("output", None)
            if action == "edit":
                return long_text.edit_text(policy, file_path=args.pop("path"), **args)
            if action == "append_text":
                return long_text.append_text(policy, file_path=args.pop("path"), include_structure=True, **args)
            return core._run_io_action(session, policy, action, args, downloads)

    media_settings_def = function_def(operations["wangp_get_media_settings"], "Read the settings that generated a media file. Supply exactly one of media_id or an authorized path.")

    @mcp.tool()
    def wangp_toolbox(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Inspect or compare media, transcribe, extract, trim, assemble and transform media, or retrieve their generation settings."""
        actions = {item["name"]: {"description": item["description"], "parameters": {**item["parameters"], "additionalProperties": False}} for item in core._toolbox_discovery(get_toolbox(), policy.read_enabled) if item["name"] not in {"search_doc", "load_doc_section", "get_media_details"}}
        actions["media_settings"] = media_settings_def
        for name, summary in MEDIA_DISCOVERY_DESCRIPTIONS.items():
            if name in actions:
                actions[name]["_summary"] = summary
        if "inspect_media" in actions:
            properties = actions["inspect_media"]["parameters"]["properties"]
            actions["inspect_media"]["parameters"]["oneOf"] = [{"required": [name]} for name in ("media_id", "media_ids", "media_inputs")]
            properties["media_inputs"]["items"]["properties"]["bbox"] = {key: value for key, value in properties["bbox"].items() if key != "description"}
            actions["inspect_media"]["description"] += (
                " Choose exactly one input form: media_id for one visual, media_ids for a list, or media_inputs for per-video time_seconds/frame_no."
                " Inspect video frames directly; use extract_image only to save a frame."
                " Each media_inputs item may specify bbox=[x_min,y_min,x_max,y_max] (integers 0..1000), overriding the shared bbox; omitted boxes use the shared bbox or the full visual. Cropping precedes resizing."
            )
            actions["inspect_media"]["example"] = {"action": "inspect_media", "arguments": {"media_id": "visual:IMAGE", "question": "Describe this image."}}
        if "extract_image" in actions:
            actions["extract_image"]["description"] += " frame_no is zero-based (0 is the first frame); use frame_no or time_seconds, not both."
        if action == "inspect_media" and isinstance(arguments, dict) and "media" in arguments:
            arguments = dict(arguments)
            media = arguments.pop("media")
            if not isinstance(media, (str, list)):
                raise ValueError("inspect_media: media must be one reference string or a list of reference strings inside arguments.")
            name = "media_ids" if isinstance(media, list) else "media_id"
            if name in arguments and arguments[name] != media:
                raise ValueError(f"inspect_media: media and {name} have different values. Supply one input.")
            arguments[name] = media
        response = invocation(actions, action, arguments)
        if response is not None:
            return response
        if action == "media_settings":
            return public_media_result(operations["wangp_get_media_settings"](**arguments), get_toolbox().session.media_registry)
        return public_media_result(operations["wangp_toolbox"](action, arguments), get_toolbox().session.media_registry)

    wait_properties = {
        "wait": {"type": "boolean", "default": True, **({} if allow_async else {"const": True})},
        "timeout_s": {"type": ["number", "null"], "minimum": 0, "default": None},
        "event_limit": {"type": "integer", "minimum": 0, "maximum": 500, "default": 0},
    }
    wait_help = "wait=true waits for completion; timeout_s bounds waiting without cancelling the job. " + ("wait=false returns a job for asynchronous tracking." if allow_async else "Asynchronous execution is disabled; wait=false is rejected before submission.")
    if defer_wait and not allow_async:
        wait_properties = {"event_limit": wait_properties["event_limit"]}
        wait_help = "WanGP manages waiting and returns the completed result."

    def finish(initial, args):
        if defer_wait or not args.get("wait", True):
            return public_media_result(initial)
        job = jobs.get(initial["job_id"])
        try:
            job.job.result(timeout=args.get("timeout_s"))
        except TimeoutError:
            result = job.snapshot(event_limit=args.get("event_limit", 0))
            result.update(status="timeout", waiting_timed_out=True)
            return public_media_result(result)
        return public_media_result(job.snapshot(event_limit=args.get("event_limit", 0)))

    generate_def = action_def("Generate image, video or audio from prepared settings, a task wrapping settings in params or settings, a task list or a manifest with tasks. Each generation settings object requires model_type (legacy base_model_type is accepted); edit_* post-processing tasks need no model. Model selection and supplied media inputs are checked before the batch is submitted once, preserving task order and settings. Inputs unsupported by the model or inactive in the selected mode return an error. source may contain @file(\"@workspace/prompt.txt\") in a prompt field: WanGP reads and snapshots that authorized UTF-8 file, preserving blank lines. " + wait_help, {"source": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}, "minItems": 1}]}, **wait_properties}, ("source",))
    generate_def["_summary"] = 'Generate from prepared settings with arguments={"source":{...settings}}; source also accepts a task list or tasks manifest. Waits for completion by default; read the contract only for additional options.'

    @mcp.tool()
    def wangp_generate(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate images, video or audio from prepared settings and obtain the result."""
        response = invocation({"generate": generate_def}, action, arguments)
        if response is not None:
            return response
        return finish(operations["wangp_generate"](arguments["source"], wait=False, event_limit=arguments.get("event_limit", 0)), arguments)

    @mcp.tool()
    def wangp_postprocess(media: str, action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Discover and apply compatible image, video or audio treatments, including upscaling, refinement, interpolation and sound processing."""
        from postprocessing import catalog

        concrete = media not in {"image", "video", "audio"}
        source = core._resolve_mcp_media_reference(session, media, "media", policy.read_enabled, policy) if concrete else None
        kind = core._mcp_media_type(source) if concrete else media
        if kind not in {"image", "video", "audio"}:
            raise ValueError("media must be image, video, audio, a Gallery media ID or an authorized media path.")
        processes = catalog.query_processes(kind)
        actions = {}
        for process in core._mcp_postprocessing_processes(kind, policy.read_enabled, processes):
            if process.get("status") == "disabled":
                continue
            properties, required = {}, []
            for parameter in process.get("parameters", ()):
                name = parameter["name"]
                properties[name] = {key: value for key, value in parameter.items() if key in {"type", "description", "default", "enum", "minimum", "maximum", "items"}}
                for old, new in (("min", "minimum"), ("max", "maximum")):
                    if old in parameter:
                        properties[name][new] = parameter[old]
                if parameter.get("required") and "default" not in parameter:
                    required.append(name)
            actions[process["id"]] = action_def(process["description"], {**properties, **wait_properties}, required)
            actions[process["id"]]["limitations"] = "Execution requires a concrete media ID or authorized path. " + wait_help
        response = invocation(actions, action, arguments)
        if response is not None:
            response["media_type"] = kind
            return response
        if not concrete:
            raise ValueError("Execution requires a concrete Gallery media ID or authorized path, not a media type.")
        parameters = {key: value for key, value in arguments.items() if key not in wait_properties}
        initial = operations["wangp_postprocess"](media_id=media if core._is_media_id(media) else None, path=None if core._is_media_id(media) else media, process=action, parameters=parameters)
        return finish(initial, arguments)

    session_operations = {name: operations[f"wangp_{name}"] for name in ("get_job", "cancel_job", "notify")}
    for name in ("create_gallery_upload", "create_gallery_download"):
        if f"wangp_{name}" in operations:
            session_operations[name] = operations[f"wangp_{name}"]
    session_actions = {name: function_def(function) for name, function in session_operations.items()}
    session_actions["get_job"]["description"] = "Read the state or result of a generation or post-processing job."
    session_actions["cancel_job"]["description"] = "Request cancellation of a generation or post-processing job."
    session_actions["notify"]["description"] = "Send a notification through configured destinations; notifications are independent of job completion."

    @mcp.tool()
    def wangp_session(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Track or cancel generation and post-processing jobs, send configured notifications, and transfer Gallery media when supported."""
        response = invocation(session_actions, action, arguments)
        if response is not None:
            return response
        return public_media_result(session_operations[action](**arguments))

    for tool in mcp._tool_manager.list_tools():
        tool.fn_metadata.arg_model.model_config["extra"] = "forbid"
        tool.fn_metadata.arg_model.model_rebuild(force=True)
        tool.parameters = strip_schema_titles(tool.fn_metadata.arg_model.model_json_schema())

    @mcp.resource("wangp://guides/workflows", name="workflows", description="WanGP workflows: templates and profiles, media inputs, speech and chained media, visual verification, extraction, long video, prompt files and editing.", mime_type="text/markdown")
    def workflow_guide() -> str:
        return (Path(__file__).parent / "deepy" / "prime_guide.md").read_text(encoding="utf-8")

    @mcp.resource("wangp://guides/long-video", name="long_video", title="Long video and planned end frames", description="Long video workflow: first/start frame, master image, independent edits, planned end-frame anchors, sliding windows, multi-shot continuity and video continuation.", mime_type="text/markdown")
    def long_video_guide() -> str:
        return core._read_markdown_section(Path(__file__).parent / "deepy" / "prime_guide.md", "## Long video and sliding windows", "## Prompt files and long documents")

    @mcp._mcp_server.read_resource()
    async def read_resource(uri):
        return await mcp.read_resource(RESOURCE_ALIASES.get(str(uri), uri))
