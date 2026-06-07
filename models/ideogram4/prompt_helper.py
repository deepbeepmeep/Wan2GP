import html
import json


DEFAULT_POPUP_DIMS = (78, 100)


def _popup_dims(model_def):
    dims = (model_def or {}).get("prompt_helper_popup_dims", DEFAULT_POPUP_DIMS)
    if not isinstance(dims, (list, tuple)) or len(dims) < 2:
        return DEFAULT_POPUP_DIMS
    try:
        width, height = int(dims[0]), int(dims[1])
    except (TypeError, ValueError):
        return DEFAULT_POPUP_DIMS
    return max(35, min(width, 96)), max(35, min(height, 100))


def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
    width, height = _popup_dims(model_def)
    config = {
        "promptTarget": prompt_elem_id,
        "resolutionTarget": resolution_elem_id,
        "promptId": prompt_id,
        "modelType": model_type,
    }
    config = html.escape(json.dumps(config), quote=True)
    popup_id_html = html.escape(popup_id, quote=True)
    return f"""
<div id="{popup_id_html}" class="wangp-model-info-popup wangp-prompt-helper-popup ideogram4-prompt-helper-popup" role="dialog" aria-label="Ideogram 4 Prompt Helper" data-wangp-model-info-popup data-wangp-prompt-helper-popup="1" hidden style="width:min({width}vw,calc(100vw - 24px));height:min({height}vh,calc(100vh - 12px));">
  <div class="wangp-model-info-card ideogram4-prompt-helper" data-ideogram4-prompt-helper data-ideogram4-config="{config}">
    <div class="wangp-model-info-titlebar" data-wangp-model-info-drag>
      <div class="wangp-model-info-heading">Ideogram 4 Prompt Helper</div>
      <button type="button" class="wangp-model-info-close" aria-label="Close prompt helper" data-wangp-model-info-close>&times;</button>
    </div>
    <div class="wangp-model-info-content ideogram4-prompt-helper-content" data-ideogram4-prompt-helper-content></div>
  </div>
</div>
"""


def get_prompt_helper_css():
    return """
.ideogram4-prompt-helper-popup .wangp-model-info-content {
    padding: 12px;
}
.ideogram4-prompt-helper {
    --ideogram4-canvas-ratio: 1 / 1;
}
.ideogram4-prompt-helper-content {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
}
.ideogram4-prompt-helper-content.is-splitting {
    cursor: row-resize;
    user-select: none;
}
.ideogram4-helper-canvas-column {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    gap: 8px;
}
.ideogram4-helper-actions {
    display: flex;
    flex: 0 0 auto;
    gap: 7px;
    align-items: center;
    justify-content: center;
}
.ideogram4-helper-actions button {
    width: 30px;
    height: 30px;
    min-width: 30px;
    min-height: 30px;
    padding: 0;
    border-radius: 6px;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16));
    background: var(--button-secondary-background-fill, #f6fafc);
    color: var(--button-secondary-text-color, #174a67);
    cursor: pointer;
    font-size: 15px;
    line-height: 1;
}
.ideogram4-helper-actions button:disabled {
    cursor: default;
    opacity: 0.42;
}
.ideogram4-helper-actions button[data-ideogram4-action="apply"] {
    width: auto;
    min-width: 104px;
    padding: 0 10px;
}
.ideogram4-helper-actions button[data-ideogram4-action="delete"] {
    margin-left: auto;
}
.ideogram4-helper-resolution {
    margin-left: 4px;
    color: var(--body-text-color-subdued, #5d7787);
    font-size: 0.82rem;
    white-space: nowrap;
}
.ideogram4-helper-workspace {
    display: grid;
    grid-template-columns: minmax(220px, 1fr) minmax(160px, 240px);
    flex: 0 0 auto;
    min-height: 0;
    gap: 12px;
}
.ideogram4-helper-splitter {
    position: relative;
    flex: 0 0 14px;
    min-height: 14px;
    margin: 6px 0;
    border: 0;
    padding: 0;
    background: transparent;
    cursor: row-resize;
}
.ideogram4-helper-splitter::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    height: 3px;
    transform: translateY(-50%);
    border-radius: 999px;
    background: var(--border-color-primary, rgba(17, 84, 118, 0.46));
}
.ideogram4-helper-splitter::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    width: 56px;
    height: 9px;
    box-sizing: border-box;
    transform: translate(-50%, -50%);
    border-top: 2px solid var(--border-color-primary, rgba(17, 84, 118, 0.58));
    border-bottom: 2px solid var(--border-color-primary, rgba(17, 84, 118, 0.58));
    border-radius: 999px;
    background: var(--background-fill-primary, #fff);
}
.ideogram4-helper-splitter:focus-visible::after {
    outline: 2px solid var(--button-primary-border-color, #156082);
    outline-offset: 3px;
}
.ideogram4-helper-canvas-wrap {
    display: flex;
    flex: 1 1 auto;
    align-items: center;
    justify-content: center;
    min-width: 0;
    min-height: 0;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16));
    border-radius: 8px;
    background: #f8fbfd;
    overflow: hidden;
}
.ideogram4-helper-canvas-shell {
    position: relative;
    aspect-ratio: var(--ideogram4-canvas-ratio);
    max-width: 100%;
    max-height: 100%;
}
.ideogram4-helper-canvas-shell canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: crosshair;
}
.ideogram4-helper-box-editor {
    position: absolute;
    display: none;
    z-index: 3;
    box-sizing: border-box;
    min-width: 42px;
    min-height: 28px;
    padding: 4px 6px;
    border: 1px solid rgba(122, 82, 12, 0.65);
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.88);
    color: var(--body-text-color, #174a67);
    font: 700 0.84rem/1.2 sans-serif;
    resize: none;
    outline: none;
}
.ideogram4-helper-list {
    min-height: 0;
    overflow: auto;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16));
    border-radius: 8px;
    background: var(--background-fill-secondary, #f4f9fc);
}
.ideogram4-helper-list-row {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 6px;
    align-items: center;
    width: 100%;
    padding: 7px 9px;
    box-sizing: border-box;
    border-bottom: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.10));
}
.ideogram4-helper-list-row.selected {
    background: rgba(16, 86, 121, 0.12);
}
.ideogram4-helper-list-row button {
    width: 28px;
    height: 28px;
    min-width: 28px;
    min-height: 28px;
    padding: 0;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16));
    border-radius: 999px;
    background: var(--button-secondary-background-fill, #f6fafc);
    color: var(--body-text-color, #174a67);
    cursor: pointer;
    font-size: 0.8rem;
    line-height: 1;
}
.ideogram4-helper-list-row textarea {
    width: 100%;
    min-width: 0;
    min-height: 42px;
    box-sizing: border-box;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 7px;
    background: transparent;
    color: var(--body-text-color, #174a67);
    font: inherit;
    resize: vertical;
}
.ideogram4-helper-list-row.selected textarea,
.ideogram4-helper-list-row textarea:focus {
    border-color: var(--border-color-primary, rgba(17, 84, 118, 0.16));
    background: var(--input-background-fill, #fff);
    outline: none;
}
.ideogram4-helper-details {
    flex: 1 1 auto;
    min-height: 0;
    max-height: none;
    overflow: auto;
    padding-right: 4px;
}
.ideogram4-helper-fields,
.ideogram4-helper-selected {
    display: grid;
    grid-template-columns: repeat(2, minmax(150px, 1fr));
    gap: 9px 10px;
}
.ideogram4-helper-selected {
    margin-top: 10px;
}
.ideogram4-helper-fields label,
.ideogram4-helper-selected label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 0.82rem;
    font-weight: 700;
}
.ideogram4-helper-fields input,
.ideogram4-helper-fields textarea,
.ideogram4-helper-selected input,
.ideogram4-helper-selected textarea,
.ideogram4-helper-selected select {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16));
    border-radius: 6px;
    padding: 6px 8px;
    background: var(--input-background-fill, #fff);
    color: var(--body-text-color, #174a67);
    font: inherit;
}
.ideogram4-helper-fields textarea,
.ideogram4-helper-selected textarea {
    min-height: 50px;
    resize: vertical;
}
.ideogram4-helper-extras {
    grid-column: 1 / -1;
}
.ideogram4-helper-extras textarea {
    min-height: 68px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.78rem;
}
.ideogram4-helper-status {
    color: var(--body-text-color-subdued, #5d7787);
    font-size: 0.82rem;
}
.ideogram4-helper-status.error {
    color: #b42318;
    font-weight: 700;
}
@media (max-width: 760px) {
    .ideogram4-helper-workspace,
    .ideogram4-helper-fields,
    .ideogram4-helper-selected {
        grid-template-columns: 1fr;
    }
}
"""


def get_prompt_helper_javascript():
    return """
window.wangpIdeogram4PromptHelper = window.wangpIdeogram4PromptHelper || {};
(function(helper) {
    const ROOT_KEYS = ["aspect_ratio", "high_level_description", "style_description", "compositional_deconstruction"];
    const STYLE_KEYS = ["aesthetics", "lighting", "medium", "art_style", "photo", "color_palette"];
    const COMP_KEYS = ["background", "elements"];
    const ELEMENT_KEYS = ["type", "bbox", "text", "desc", "color_palette"];

    helper.clamp = function(value, min, max) {
        return Math.max(min, Math.min(max, value));
    };
    helper.isObject = function(value) {
        return value && typeof value === "object" && !Array.isArray(value);
    };
    helper.copy = function(value) {
        return helper.isObject(value) || Array.isArray(value) ? JSON.parse(JSON.stringify(value)) : value;
    };
    helper.escapeHtml = function(value) {
        return String(value || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    };
    helper.parsePalette = function(value, limit) {
        if (Array.isArray(value)) return value.map((item) => String(item || "").trim().toUpperCase()).filter(Boolean).slice(0, limit);
        return String(value || "").split(",").map((item) => item.trim().toUpperCase()).filter(Boolean).slice(0, limit);
    };
    helper.paletteText = function(value) {
        return Array.isArray(value) ? value.join(", ") : String(value || "");
    };
    helper.defaultDoc = function() {
        return {
            aspect_ratio: "",
            high_level_description: "",
            style_description: { aesthetics: "", lighting: "", medium: "graphic_design", art_style: "", color_palette: [] },
            compositional_deconstruction: { background: "", elements: [] }
        };
    };
    helper.extractPromptComments = function(raw) {
        const comments = [];
        const jsonLines = [];
        String(raw || "").split(/\\r?\\n/).forEach((line) => {
            if (line.trimStart().startsWith("#")) comments.push(line);
            else jsonLines.push(line);
        });
        return { comments, jsonText: jsonLines.join("\\n").trim() };
    };
    helper.normalizePromptJsonText = function(raw) {
        const text = String(raw || "").trim();
        const notes = [];
        if (!text) return { jsonText: "", notes };
        const start = text.indexOf("{");
        if (start < 0) return { jsonText: text, notes };
        if (text.slice(0, start).trim()) notes.push("ignored text before JSON");
        const openCount = (text.match(/{/g) || []).length;
        const closeCount = (text.match(/}/g) || []).length;
        if (openCount === closeCount) {
            const end = text.lastIndexOf("}");
            if (text.slice(end + 1).trim()) notes.push("ignored text after JSON");
            return { jsonText: text.slice(start, end + 1).trim(), notes };
        }
        if (openCount > closeCount) {
            const missing = openCount - closeCount;
            notes.push(`added ${missing} missing closing ${missing === 1 ? "brace" : "braces"}`);
            return { jsonText: text.slice(start).trimEnd() + "}".repeat(missing), notes };
        }
        return { jsonText: text.slice(start).trim(), notes };
    };
    helper.parsePromptJson = function(raw) {
        const normalized = helper.normalizePromptJsonText(raw);
        if (!normalized.jsonText) return { doc: helper.defaultDoc(), notes: normalized.notes };
        return { doc: JSON.parse(normalized.jsonText), notes: normalized.notes };
    };
    helper.withPromptComments = function(jsonText, comments) {
        return comments && comments.length ? comments.join("\\n") + "\\n" + jsonText : jsonText;
    };
    helper.stringifyDoc = function(doc) {
        return JSON.stringify(doc, null, 2).replace(/"bbox": \\[\\n\\s*([0-9.-]+),\\n\\s*([0-9.-]+),\\n\\s*([0-9.-]+),\\n\\s*([0-9.-]+)\\n\\s*\\]/g, '"bbox": [$1, $2, $3, $4]');
    };
    helper.extractExtras = function(source, knownKeys) {
        const out = {};
        if (!helper.isObject(source)) return out;
        Object.keys(source).forEach((key) => {
            if (!knownKeys.includes(key)) out[key] = helper.copy(source[key]);
        });
        return out;
    };
    helper.hasExtras = function(value) {
        if (Array.isArray(value)) return value.some(helper.hasExtras);
        return helper.isObject(value) && Object.keys(value).length > 0;
    };
    helper.parseResolutionText = function(text) {
        const match = String(text || "").match(/(\\d{2,5})\\s*x\\s*(\\d{2,5})/i);
        return match ? { width: Math.max(1, Number(match[1])), height: Math.max(1, Number(match[2])), label: `${match[1]}x${match[2]}` } : null;
    };
    helper.resolutionFromChoice = function(text, choices) {
        const needle = String(text || "").trim();
        if (!needle || !Array.isArray(choices)) return null;
        for (const choice of choices) {
            const label = Array.isArray(choice) ? choice[0] : choice;
            const value = Array.isArray(choice) ? choice[1] : choice;
            if (String(label || "").trim() === needle || String(value || "").trim() === needle) return helper.parseResolutionText(value) || helper.parseResolutionText(label);
        }
        return null;
    };
    helper.readResolution = function(targetId) {
        const target = document.getElementById(targetId || "");
        const component = (window.gradio_config?.components || []).find((item) => item.props?.elem_id === targetId);
        const choices = component?.props?.choices || [];
        const values = [];
        if (target) {
            target.querySelectorAll("input, select, textarea, button").forEach((node) => values.push(node.value || node.textContent || ""));
            values.push(target.getAttribute("data-value") || "", target.textContent || "");
        }
        for (const value of values) {
            const resolved = helper.parseResolutionText(value) || helper.resolutionFromChoice(value, choices);
            if (resolved) return resolved;
        }
        return helper.parseResolutionText(component?.props?.value) || { width: 1000, height: 1000, label: "1000x1000" };
    };
    helper.buildContent = function(content) {
        content.innerHTML = `
            <div class="ideogram4-helper-workspace">
                <div class="ideogram4-helper-canvas-column">
                    <div class="ideogram4-helper-canvas-wrap">
                        <div class="ideogram4-helper-canvas-shell">
                            <canvas width="1000" height="1000" aria-label="Ideogram bbox canvas"></canvas>
                            <textarea class="ideogram4-helper-box-editor" aria-label="Selected box text"></textarea>
                        </div>
                    </div>
                    <div class="ideogram4-helper-actions">
                        <button type="button" data-ideogram4-action="undo" title="Undo" aria-label="Undo">&#8630;</button>
                        <button type="button" data-ideogram4-action="redo" title="Redo" aria-label="Redo">&#8631;</button>
                        <span class="ideogram4-helper-resolution" data-ideogram4-resolution></span>
                        <button type="button" data-ideogram4-action="apply" title="Save changes" aria-label="Save changes">Save Changes</button>
                        <button type="button" data-ideogram4-action="delete" title="Delete selected box" aria-label="Delete selected box">&#128465;</button>
                        <button type="button" data-ideogram4-action="clear" title="Clear all" aria-label="Clear all">&#9003;</button>
                    </div>
                </div>
                <div class="ideogram4-helper-list" aria-label="Prompt elements"></div>
            </div>
            <div class="ideogram4-helper-splitter" role="separator" aria-orientation="horizontal" aria-label="Resize canvas and properties panels" tabindex="0"></div>
            <div class="ideogram4-helper-details">
                <div class="ideogram4-helper-selected">
                    <label>Selected type<select data-ideogram4-field="type"><option value="obj">obj</option><option value="text">text</option></select></label>
                    <label>Selected description<textarea data-ideogram4-field="desc"></textarea></label>
                    <label>Selected palette<input data-ideogram4-field="element_palette" type="text" placeholder="#FFFFFF, #111111"></label>
                    <div class="ideogram4-helper-status" data-ideogram4-status></div>
                </div>
                <div class="ideogram4-helper-fields">
                    <label>Aspect ratio<input data-ideogram4-field="aspect_ratio" type="text" placeholder="4:3"></label>
                    <label>High level description<textarea data-ideogram4-field="high"></textarea></label>
                    <label>Background<textarea data-ideogram4-field="background"></textarea></label>
                    <label>Aesthetics<input data-ideogram4-field="aesthetics" type="text"></label>
                    <label>Lighting<input data-ideogram4-field="lighting" type="text"></label>
                    <label>Medium<input data-ideogram4-field="medium" type="text" placeholder="graphic_design"></label>
                    <label>Photo details<input data-ideogram4-field="photo" type="text" placeholder="35mm, f/1.4"></label>
                    <label>Art style<input data-ideogram4-field="art_style" type="text" placeholder="flat vector design"></label>
                    <label>Palette<input data-ideogram4-field="style_palette" type="text" placeholder="#FFFFFF, #111111"></label>
                    <label class="ideogram4-helper-extras">Custom JSON leftovers<textarea data-ideogram4-field="extras"></textarea></label>
                </div>
            </div>`;
    };
    helper.init = function(card) {
        if (!card) return null;
        if (card._ideogram4Helper) return card._ideogram4Helper;
        const content = card.querySelector("[data-ideogram4-prompt-helper-content]");
        if (!content) return null;
        helper.buildContent(content);

        const config = JSON.parse(card.getAttribute("data-ideogram4-config") || "{}");
        const canvas = card.querySelector("canvas");
        const ctx = canvas.getContext("2d");
        const workspace = card.querySelector(".ideogram4-helper-workspace");
        const splitter = card.querySelector(".ideogram4-helper-splitter");
        const canvasWrap = card.querySelector(".ideogram4-helper-canvas-wrap");
        const canvasShell = card.querySelector(".ideogram4-helper-canvas-shell");
        const boxEditor = card.querySelector(".ideogram4-helper-box-editor");
        const list = card.querySelector(".ideogram4-helper-list");
        const status = card.querySelector("[data-ideogram4-status]");
        const resolutionLabel = card.querySelector("[data-ideogram4-resolution]");
        const defaultTextDesc = "clear, legible typography contained within the bounding box";
        const fields = {};
        card.querySelectorAll("[data-ideogram4-field]").forEach((field) => fields[field.getAttribute("data-ideogram4-field")] = field);
        const state = { doc: helper.defaultDoc(), elements: [], elementExtras: [], unboxedElements: [], promptComments: [], selected: -1, drawing: null, resizing: null, moving: null, splitting: null, syncing: false, ratio: 1, splitRatio: 0.64, newType: "obj", dirty: false, changedSinceOpen: false };
        const actionButtons = {};
        card.querySelectorAll("[data-ideogram4-action]").forEach((button) => actionButtons[button.getAttribute("data-ideogram4-action")] = button);
        const history = { undo: [], redo: [], clean: "", pending: false };
        const historyFields = ["aspect_ratio", "high", "background", "aesthetics", "lighting", "medium", "photo", "art_style", "style_palette", "extras"];

        function promptField() {
            const target = document.getElementById(config.promptTarget || "");
            return target?.querySelector("textarea") || target?.querySelector("input") || null;
        }
        function markDirty() {
            if (state.syncing) return;
            state.dirty = true;
            state.changedSinceOpen = true;
        }
        function snapshotState() {
            const values = {};
            historyFields.forEach((name) => values[name] = fields[name]?.value || "");
            return {
                values,
                elements: helper.copy(state.elements),
                elementExtras: helper.copy(state.elementExtras),
                unboxedElements: helper.copy(state.unboxedElements),
                extraRoot: helper.copy(state.extraRoot || {}),
                extraStyle: helper.copy(state.extraStyle || {}),
                extraComp: helper.copy(state.extraComp || {}),
                promptComments: state.promptComments.slice(),
                selected: state.selected,
                newType: state.newType
            };
        }
        function snapshotKey(snapshot) {
            const key = Object.assign({}, snapshot);
            delete key.selected;
            return JSON.stringify(key);
        }
        function currentSnapshotKey() {
            return snapshotKey(snapshotState());
        }
        function updateDirty() {
            state.dirty = !!history.clean && currentSnapshotKey() !== history.clean;
        }
        function updateHistoryButtons() {
            if (actionButtons.undo) actionButtons.undo.disabled = history.undo.length < 2;
            if (actionButtons.redo) actionButtons.redo.disabled = history.redo.length < 1;
        }
        function resetHistory() {
            const snapshot = snapshotState();
            history.undo = [snapshot];
            history.redo = [];
            history.clean = snapshotKey(snapshot);
            history.pending = false;
            state.dirty = false;
            state.changedSinceOpen = false;
            updateHistoryButtons();
        }
        function recordHistory() {
            if (state.syncing) return;
            history.pending = false;
            const snapshot = snapshotState();
            const key = snapshotKey(snapshot);
            if (!history.undo.length || snapshotKey(history.undo[history.undo.length - 1]) !== key) {
                history.undo.push(snapshot);
                if (history.undo.length > 50) history.undo.shift();
            }
            history.redo = [];
            updateDirty();
            updateHistoryButtons();
        }
        function markPendingHistory() {
            if (state.syncing) return;
            history.pending = true;
            markDirty();
        }
        function commitPendingHistory() {
            if (history.pending) recordHistory();
            else updateDirty();
        }
        function restoreSnapshot(snapshot) {
            if (!snapshot) return;
            state.syncing = true;
            history.pending = false;
            historyFields.forEach((name) => {
                if (fields[name]) fields[name].value = snapshot.values?.[name] || "";
            });
            state.elements = helper.copy(snapshot.elements || []);
            state.elementExtras = helper.copy(snapshot.elementExtras || []);
            state.unboxedElements = helper.copy(snapshot.unboxedElements || []);
            state.extraRoot = helper.copy(snapshot.extraRoot || {});
            state.extraStyle = helper.copy(snapshot.extraStyle || {});
            state.extraComp = helper.copy(snapshot.extraComp || {});
            state.promptComments = (snapshot.promptComments || []).slice();
            state.selected = Math.min(snapshot.selected ?? -1, state.elements.length - 1);
            state.newType = typeValue(snapshot.newType);
            state.syncing = false;
            refresh();
            updateDirty();
            updateHistoryButtons();
        }
        function undoChange() {
            commitPendingHistory();
            if (history.undo.length < 2) return;
            history.redo.push(history.undo.pop());
            restoreSnapshot(history.undo[history.undo.length - 1]);
            state.changedSinceOpen = true;
        }
        function redoChange() {
            commitPendingHistory();
            if (!history.redo.length) return;
            const snapshot = history.redo.pop();
            history.undo.push(snapshot);
            restoreSnapshot(snapshot);
            state.changedSinceOpen = true;
        }
        function setStatus(text, error) {
            if (!status) return;
            status.textContent = text || "";
            status.classList.toggle("error", !!error);
        }
        function stripTrailingLineBreaks(value) {
            return String(value || "").replace(/[\\r\\n]+$/g, "");
        }
        function point(event) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: helper.clamp(Math.round((event.clientX - rect.left) * 1000 / rect.width), 0, 1000),
                y: helper.clamp(Math.round((event.clientY - rect.top) * 1000 / rect.height), 0, 1000)
            };
        }
        function bboxFromRect(rect) {
            return [
                helper.clamp(Math.min(rect.y1, rect.y2), 0, 1000),
                helper.clamp(Math.min(rect.x1, rect.x2), 0, 1000),
                helper.clamp(Math.max(rect.y1, rect.y2), 0, 1000),
                helper.clamp(Math.max(rect.x1, rect.x2), 0, 1000)
            ];
        }
        function hitBox(x, y) {
            for (let i = state.elements.length - 1; i >= 0; i--) {
                const bbox = state.elements[i].bbox || [];
                if (bbox.length === 4 && x >= bbox[1] && x <= bbox[3] && y >= bbox[0] && y <= bbox[2]) return i;
            }
            return -1;
        }
        function resizeTolerance() {
            const rect = canvas.getBoundingClientRect();
            return Math.max(10, Math.round(10 * 1000 / Math.max(rect.width, rect.height, 1)));
        }
        function resizeMode(bbox, x, y, tolerance) {
            const nearTop = Math.abs(y - bbox[0]) <= tolerance;
            const nearLeft = Math.abs(x - bbox[1]) <= tolerance;
            const nearBottom = Math.abs(y - bbox[2]) <= tolerance;
            const nearRight = Math.abs(x - bbox[3]) <= tolerance;
            const inX = x >= bbox[1] - tolerance && x <= bbox[3] + tolerance;
            const inY = y >= bbox[0] - tolerance && y <= bbox[2] + tolerance;
            if (nearTop && nearLeft) return "nw";
            if (nearTop && nearRight) return "ne";
            if (nearBottom && nearLeft) return "sw";
            if (nearBottom && nearRight) return "se";
            if (nearTop && inX) return "n";
            if (nearBottom && inX) return "s";
            if (nearLeft && inY) return "w";
            if (nearRight && inY) return "e";
            return "";
        }
        function hitResizeHandle(x, y) {
            const tolerance = resizeTolerance();
            const selected = state.elements[state.selected];
            if (selected) {
                const mode = resizeMode(selected.bbox, x, y, tolerance);
                if (mode) return { index: state.selected, mode };
            }
            for (let i = state.elements.length - 1; i >= 0; i--) {
                if (i === state.selected) continue;
                const mode = resizeMode(state.elements[i].bbox || [], x, y, tolerance);
                if (mode) return { index: i, mode };
            }
            return null;
        }
        function resizeCursor(mode) {
            return { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize" }[mode] || "";
        }
        function typeValue(value) {
            return value === "text" ? "text" : "obj";
        }
        function isText(element) {
            return typeValue(element?.type) === "text";
        }
        function updateResize(point) {
            const element = state.elements[state.resizing?.index];
            if (!element) return;
            const bbox = element.bbox.slice();
            const mode = state.resizing.mode;
            if (mode.includes("n")) bbox[0] = helper.clamp(Math.min(point.y, bbox[2] - 8), 0, 1000);
            if (mode.includes("s")) bbox[2] = helper.clamp(Math.max(point.y, bbox[0] + 8), 0, 1000);
            if (mode.includes("w")) bbox[1] = helper.clamp(Math.min(point.x, bbox[3] - 8), 0, 1000);
            if (mode.includes("e")) bbox[3] = helper.clamp(Math.max(point.x, bbox[1] + 8), 0, 1000);
            element.bbox = bbox;
            markDirty();
            draw();
            syncEditor();
        }
        function startMove(index, point) {
            const element = state.elements[index];
            if (!element) return;
            state.selected = -1;
            boxEditor.style.display = "none";
            state.moving = { index, startX: point.x, startY: point.y, bbox: element.bbox.slice(), moved: false };
            canvas.style.cursor = "move";
        }
        function updateMove(point) {
            const move = state.moving;
            const element = state.elements[move?.index];
            if (!element) return;
            const dx = point.x - move.startX, dy = point.y - move.startY;
            if (!move.moved && Math.abs(dx) < 3 && Math.abs(dy) < 3) return;
            move.moved = true;
            const height = move.bbox[2] - move.bbox[0], width = move.bbox[3] - move.bbox[1];
            const y1 = helper.clamp(move.bbox[0] + dy, 0, 1000 - height);
            const x1 = helper.clamp(move.bbox[1] + dx, 0, 1000 - width);
            element.bbox = [y1, x1, y1 + height, x1 + width];
            markDirty();
            draw();
            updateBoxEditor(false);
        }
        function wrapTextLines(text, maxWidth) {
            const lines = [];
            String(text || "").split(/\\r?\\n/).forEach((sourceLine) => {
                const words = sourceLine.split(/\\s+/).filter(Boolean);
                if (!words.length) {
                    lines.push("");
                    return;
                }
                let line = "";
                words.forEach((word) => {
                    const next = line ? line + " " + word : word;
                    if (ctx.measureText(next).width <= maxWidth || !line) {
                        line = next;
                    } else {
                        lines.push(line);
                        line = word;
                    }
                });
                lines.push(line);
            });
            return lines;
        }
        function drawMultilineText(text, x, y, w, h) {
            const padding = 10;
            const lineHeight = 30;
            const maxWidth = Math.max(20, w - padding * 2);
            const maxLines = Math.max(1, Math.floor((h - padding * 2) / lineHeight));
            const lines = wrapTextLines(text, maxWidth).slice(0, maxLines);
            ctx.save();
            ctx.beginPath();
            ctx.rect(x + padding, y + padding, Math.max(1, w - padding * 2), Math.max(1, h - padding * 2));
            ctx.clip();
            lines.forEach((line, index) => ctx.fillText(line, x + padding, y + padding + 22 + index * lineHeight));
            ctx.restore();
        }
        function drawBox(bbox, selected, label) {
            const x = bbox[1], y = bbox[0], w = bbox[3] - bbox[1], h = bbox[2] - bbox[0];
            ctx.save();
            ctx.strokeStyle = selected ? "#D18A00" : "#156082";
            ctx.lineWidth = selected ? 5 : 3;
            ctx.fillStyle = selected ? "rgba(209,138,0,0.13)" : "rgba(21,96,130,0.10)";
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = selected ? "#7A520C" : "#0F4967";
            ctx.font = "24px sans-serif";
            drawMultilineText(label, x, y, w, h);
            if (selected) {
                const points = [[bbox[1], bbox[0]], [bbox[3], bbox[0]], [bbox[1], bbox[2]], [bbox[3], bbox[2]], [(bbox[1] + bbox[3]) / 2, bbox[0]], [(bbox[1] + bbox[3]) / 2, bbox[2]], [bbox[1], (bbox[0] + bbox[2]) / 2], [bbox[3], (bbox[0] + bbox[2]) / 2]];
                ctx.fillStyle = "#fff";
                ctx.strokeStyle = "#D18A00";
                ctx.lineWidth = 2;
                points.forEach(([px, py]) => {
                    ctx.fillRect(px - 8, py - 8, 16, 16);
                    ctx.strokeRect(px - 8, py - 8, 16, 16);
                });
            }
            ctx.restore();
        }
        function draw() {
            ctx.clearRect(0, 0, 1000, 1000);
            ctx.fillStyle = "#F8FBFD";
            ctx.fillRect(0, 0, 1000, 1000);
            ctx.strokeStyle = "rgba(17,84,118,0.13)";
            ctx.lineWidth = 1;
            for (let i = 100; i < 1000; i += 100) {
                ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 1000); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(1000, i); ctx.stroke();
            }
            state.elements.forEach((element, index) => drawBox(element.bbox, index === state.selected, isText(element) ? (element.text || "text") : (element.desc || "obj")));
            if (state.drawing) drawBox(bboxFromRect(state.drawing), true, "new");
        }
        function resizeCanvas() {
            const maxW = Math.max(1, canvasWrap.clientWidth);
            const maxH = Math.max(1, canvasWrap.clientHeight);
            let width = maxW, height = maxW / state.ratio;
            if (height > maxH) {
                height = maxH;
                width = maxH * state.ratio;
            }
            canvasShell.style.width = Math.floor(width) + "px";
            canvasShell.style.height = Math.floor(height) + "px";
            updateBoxEditor(false);
        }
        function splitMetrics() {
            const style = getComputedStyle(splitter);
            const splitterSpace = splitter.offsetHeight + parseFloat(style.marginTop || 0) + parseFloat(style.marginBottom || 0);
            const available = Math.max(1, content.clientHeight - splitterSpace);
            const minTop = Math.min(180, Math.max(100, Math.round(available * 0.28)));
            const minBottom = Math.min(120, Math.max(70, Math.round(available * 0.20)));
            return { available, minTop, maxTop: Math.max(minTop, available - minBottom), splitterSpace };
        }
        function applySplitTop(top) {
            const metrics = splitMetrics();
            const height = helper.clamp(Math.round(top), metrics.minTop, metrics.maxTop);
            workspace.style.flexBasis = height + "px";
            workspace.style.height = height + "px";
            resizeCanvas();
            return height;
        }
        function layoutPanels() {
            applySplitTop(splitMetrics().available * state.splitRatio);
        }
        function updateSplitFromEvent(event) {
            const metrics = splitMetrics();
            const split = state.splitting;
            const height = applySplitTop((split?.startTop || 0) + event.clientY - (split?.startY || event.clientY));
            state.splitRatio = height / metrics.available;
        }
        function updateResolution() {
            const res = helper.readResolution(config.resolutionTarget);
            state.ratio = res.width / res.height;
            card.style.setProperty("--ideogram4-canvas-ratio", `${res.width} / ${res.height}`);
            if (resolutionLabel) resolutionLabel.textContent = res.label;
            requestAnimationFrame(layoutPanels);
        }
        function renderList() {
            list.innerHTML = state.elements.map((element, index) => {
                const label = isText(element) ? (element.text || "text") : (element.desc || "obj");
                return `<div class="ideogram4-helper-list-row ${index === state.selected ? "selected" : ""}" data-index="${index}"><button type="button" data-index="${index}" title="Select box ${index + 1}" aria-label="Select box ${index + 1}">${index + 1}</button><textarea data-index="${index}" placeholder="${isText(element) ? "text" : "description"}">${helper.escapeHtml(label)}</textarea></div>`;
            }).join("");
        }
        function updateListSelection() {
            list.querySelectorAll(".ideogram4-helper-list-row").forEach((row) => row.classList.toggle("selected", Number(row.getAttribute("data-index")) === state.selected));
        }
        function selectElement(index, rebuildList) {
            state.selected = index;
            draw();
            if (rebuildList) renderList();
            else updateListSelection();
            syncEditor();
        }
        function updateElementText(index, value) {
            const element = state.elements[index];
            if (!element) return;
            value = stripTrailingLineBreaks(value);
            if (isText(element)) element.text = value;
            else element.desc = value;
            markPendingHistory();
            state.selected = index;
            fields.desc.value = element.desc || "";
            draw();
            updateListSelection();
            updateBoxEditor(false);
        }
        function boxEditorValue(element) {
            return element ? stripTrailingLineBreaks(isText(element) ? element.text || "" : element.desc || "") : "";
        }
        function updateBoxEditor(focus) {
            const element = state.elements[state.selected];
            if (!element || state.drawing || state.resizing || state.moving) {
                boxEditor.style.display = "none";
                return;
            }
            const shellW = Math.max(1, canvasShell.clientWidth);
            const shellH = Math.max(1, canvasShell.clientHeight);
            const x = Math.round(element.bbox[1] * shellW / 1000);
            const y = Math.round(element.bbox[0] * shellH / 1000);
            const w = Math.max(42, Math.round((element.bbox[3] - element.bbox[1]) * shellW / 1000));
            const h = Math.max(28, Math.round((element.bbox[2] - element.bbox[0]) * shellH / 1000));
            boxEditor.style.left = x + "px";
            boxEditor.style.top = y + "px";
            boxEditor.style.width = Math.min(w, shellW - x) + "px";
            boxEditor.style.height = Math.min(h, shellH - y) + "px";
            boxEditor.value = boxEditorValue(element);
            boxEditor.placeholder = isText(element) ? "text" : "description";
            boxEditor.style.display = "block";
            if (focus) setTimeout(() => {
                boxEditor.focus();
                const end = boxEditor.value.length;
                boxEditor.setSelectionRange(end, end);
            }, 0);
        }
        function syncEditor() {
            state.syncing = true;
            const element = state.elements[state.selected];
            const disabled = !element;
            ["type", "desc", "element_palette"].forEach((name) => fields[name].disabled = disabled);
            if (element) {
                fields.type.value = typeValue(element.type);
                fields.desc.value = element.desc || "";
                fields.element_palette.value = helper.paletteText(element.color_palette);
            } else {
                fields.type.value = state.newType;
                fields.desc.value = "";
                fields.element_palette.value = "";
            }
            fields.desc.closest("label").style.display = element && isText(element) ? "" : "none";
            updateBoxEditor(false);
            state.syncing = false;
        }
        function refresh() {
            draw();
            renderList();
            syncEditor();
        }
        function normalizeElement(source) {
            const type = typeValue(source.type);
            const bbox = Array.isArray(source.bbox) && source.bbox.length === 4 ? source.bbox.map((n) => helper.clamp(Math.round(Number(n) || 0), 0, 1000)) : [100, 100, 300, 500];
            const out = { type, bbox, desc: stripTrailingLineBreaks(source.desc) };
            if (type === "text") out.text = stripTrailingLineBreaks(source.text);
            const palette = helper.parsePalette(source.color_palette, 5);
            if (palette.length) out.color_palette = palette;
            return out;
        }
        function compactExtras(extras) {
            const out = {};
            if (helper.hasExtras(extras.root)) out.root = extras.root;
            if (helper.hasExtras(extras.style_description)) out.style_description = extras.style_description;
            if (helper.hasExtras(extras.compositional_deconstruction)) out.compositional_deconstruction = extras.compositional_deconstruction;
            if (helper.hasExtras(extras.elements)) out.elements = extras.elements;
            return out;
        }
        function showExtras() {
            const payload = compactExtras({ root: state.extraRoot, style_description: state.extraStyle, compositional_deconstruction: state.extraComp, elements: state.elementExtras });
            fields.extras.value = Object.keys(payload).length ? JSON.stringify(payload, null, 2) : "";
        }
        function resetFields() {
            fields.aspect_ratio.value = "";
            fields.high.value = "";
            fields.background.value = "";
            fields.aesthetics.value = "";
            fields.lighting.value = "";
            fields.medium.value = "graphic_design";
            fields.photo.value = "";
            fields.art_style.value = "";
            fields.style_palette.value = "";
            fields.extras.value = "";
        }
        function readExtras() {
            const raw = fields.extras.value.trim();
            if (!raw) return { root: {}, style_description: {}, compositional_deconstruction: {}, elements: [] };
            try {
                const parsed = JSON.parse(raw);
                return {
                    root: helper.isObject(parsed.root) ? parsed.root : {},
                    style_description: helper.isObject(parsed.style_description) ? parsed.style_description : {},
                    compositional_deconstruction: helper.isObject(parsed.compositional_deconstruction) ? parsed.compositional_deconstruction : {},
                    elements: Array.isArray(parsed.elements) ? parsed.elements.map((item) => helper.isObject(item) ? item : {}) : []
                };
            } catch (err) {
                setStatus(`Custom JSON leftovers error: ${err.message}`, true);
                return null;
            }
        }
        function loadDocFromPrompt() {
            const field = promptField();
            const rawPrompt = field ? field.value : "";
            const parsedPrompt = helper.extractPromptComments(rawPrompt);
            const raw = parsedPrompt.jsonText;
            let doc = helper.defaultDoc();
            let promptNotes = [];
            state.dirty = false;
            state.promptComments = parsedPrompt.comments;
            setStatus("");
            if (raw) {
                try {
                    const parsedDoc = helper.parsePromptJson(raw);
                    doc = parsedDoc.doc;
                    promptNotes = parsedDoc.notes || [];
                } catch (err) {
                    state.doc = helper.defaultDoc();
                    state.elements = [];
                    state.elementExtras = [];
                    state.unboxedElements = [];
                    state.promptComments = parsedPrompt.comments;
                    state.selected = -1;
                    resetFields();
                    refresh();
                    resetHistory();
                    setStatus(`Invalid JSON prompt: ${err.message}`, true);
                    return;
                }
                if (!helper.isObject(doc)) {
                    state.doc = helper.defaultDoc();
                    state.elements = [];
                    state.elementExtras = [];
                    state.unboxedElements = [];
                    state.promptComments = parsedPrompt.comments;
                    state.selected = -1;
                    resetFields();
                    refresh();
                    resetHistory();
                    setStatus("Invalid JSON prompt: root value must be an object.", true);
                    return;
                }
            }
            const style = helper.isObject(doc.style_description) ? doc.style_description : {};
            const comp = helper.isObject(doc.compositional_deconstruction) ? doc.compositional_deconstruction : {};
            const sourceElements = Array.isArray(comp.elements) ? comp.elements : [];
            state.doc = doc;
            state.extraRoot = helper.extractExtras(doc, ROOT_KEYS);
            state.extraStyle = helper.extractExtras(style, STYLE_KEYS);
            state.extraComp = helper.extractExtras(comp, COMP_KEYS);
            state.elements = [];
            state.elementExtras = [];
            state.unboxedElements = [];
            sourceElements.forEach((element) => {
                if (helper.isObject(element) && Array.isArray(element.bbox) && element.bbox.length === 4) {
                    state.elements.push(normalizeElement(element));
                    state.elementExtras.push(helper.extractExtras(element, ELEMENT_KEYS));
                } else {
                    state.unboxedElements.push(helper.copy(element));
                }
            });
            fields.aspect_ratio.value = doc.aspect_ratio || "";
            fields.high.value = doc.high_level_description || "";
            fields.background.value = comp.background || "";
            fields.aesthetics.value = style.aesthetics || "";
            fields.lighting.value = style.lighting || "";
            fields.medium.value = style.medium || "graphic_design";
            fields.photo.value = style.photo || "";
            fields.art_style.value = style.art_style || "";
            fields.style_palette.value = helper.paletteText(style.color_palette);
            showExtras();
            state.selected = state.elements.length ? 0 : -1;
            refresh();
            resetHistory();
            if (state.unboxedElements.length) setStatus(`Preserved ${state.unboxedElements.length} unboxed element(s).`, true);
            else if (fields.extras.value.trim()) setStatus("Custom JSON leftovers are editable below.");
            else if (promptNotes.length) setStatus(`Loaded JSON prompt (${promptNotes.join(", ")}).`);
        }
        function buildElement(source, extra) {
            const type = typeValue(source.type);
            const out = Object.assign({}, helper.isObject(extra) ? extra : {});
            out.type = type;
            out.bbox = source.bbox;
            if (type === "text") out.text = stripTrailingLineBreaks(source.text);
            out.desc = stripTrailingLineBreaks(source.desc);
            const palette = helper.parsePalette(source.color_palette, 5);
            if (palette.length) out.color_palette = palette;
            return out;
        }
        function buildDoc() {
            const extras = readExtras();
            if (!extras) return null;
            const doc = {};
            if (fields.aspect_ratio.value.trim()) doc.aspect_ratio = fields.aspect_ratio.value.trim();
            Object.assign(doc, extras.root);
            if (fields.high.value.trim()) doc.high_level_description = fields.high.value.trim();
            const style = Object.assign({}, extras.style_description);
            const palette = helper.parsePalette(fields.style_palette.value, 16);
            if (fields.aesthetics.value.trim()) style.aesthetics = fields.aesthetics.value.trim();
            if (fields.lighting.value.trim()) style.lighting = fields.lighting.value.trim();
            if (fields.photo.value.trim()) {
                style.photo = fields.photo.value.trim();
                style.medium = fields.medium.value.trim() || "photograph";
            } else {
                style.medium = fields.medium.value.trim() || "graphic_design";
                if (fields.art_style.value.trim()) style.art_style = fields.art_style.value.trim();
            }
            if (palette.length) style.color_palette = palette;
            if (Object.keys(style).length) doc.style_description = style;
            const comp = Object.assign({}, extras.compositional_deconstruction);
            comp.background = fields.background.value.trim();
            comp.elements = state.elements.map((element, index) => buildElement(element, extras.elements[index])).concat(state.unboxedElements);
            doc.compositional_deconstruction = comp;
            return doc;
        }
        function applyPrompt() {
            commitPendingHistory();
            const field = promptField();
            const doc = buildDoc();
            if (!doc) return false;
            if (!field) {
                setStatus("Prompt textbox was not found.", true);
                return false;
            }
            const value = helper.withPromptComments(helper.stringifyDoc(doc), state.promptComments);
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
            if (setter && field instanceof HTMLTextAreaElement) setter.call(field, value);
            else field.value = value;
            field.dispatchEvent(new Event("input", { bubbles: true }));
            field.dispatchEvent(new Event("change", { bubbles: true }));
            state.dirty = false;
            resetHistory();
            setStatus("Applied JSON prompt.");
            const popup = card.closest("[data-wangp-model-info-popup], .wangp-model-info-popup");
            if (popup) popup.hidden = true;
            return true;
        }
        function deleteSelected() {
            commitPendingHistory();
            if (state.selected < 0) return;
            state.elements.splice(state.selected, 1);
            state.elementExtras.splice(state.selected, 1);
            state.selected = Math.min(state.selected, state.elements.length - 1);
            markDirty();
            showExtras();
            refresh();
            recordHistory();
        }
        function clearHelper() {
            commitPendingHistory();
            window.wangpConfirm({
                title: "Clear Prompt Helper",
                message: "Clear the canvas, boxes, and editable prompt properties?",
                confirmText: "Clear",
                cancelText: "Keep",
                danger: true
            }).then((confirmed) => {
                if (!confirmed) return;
                state.doc = helper.defaultDoc();
                state.elements = [];
                state.elementExtras = [];
                state.unboxedElements = [];
                state.extraRoot = {};
                state.extraStyle = {};
                state.extraComp = {};
                state.selected = -1;
                state.newType = "obj";
                resetFields();
                markDirty();
                refresh();
                recordHistory();
                history.clean = currentSnapshotKey();
                state.dirty = false;
                state.changedSinceOpen = false;
                updateHistoryButtons();
                setStatus("Cleared prompt helper.");
            });
        }
        function focusSelectedBox() {
            updateBoxEditor(true);
        }
        function unselectElement() {
            if (state.selected < 0) return;
            state.selected = -1;
            boxEditor.style.display = "none";
            draw();
            renderList();
            syncEditor();
        }

        canvas.addEventListener("mousedown", (event) => {
            const p = point(event);
            const resizeHit = hitResizeHandle(p.x, p.y);
            if (resizeHit) {
                state.selected = resizeHit.index;
                state.resizing = resizeHit;
                boxEditor.style.display = "none";
                draw();
                renderList();
                syncEditor();
                canvas.style.cursor = resizeCursor(resizeHit.mode);
                return;
            }
            const hit = hitBox(p.x, p.y);
            if (hit >= 0) {
                startMove(hit, p);
                draw();
                renderList();
                syncEditor();
                return;
            }
            state.drawing = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
            boxEditor.style.display = "none";
            draw();
        });
        function handlePointerMove(event) {
            const p = point(event);
            if (state.resizing) {
                updateResize(p);
                return;
            }
            if (state.moving) {
                updateMove(p);
                return;
            }
            if (!state.drawing) {
                const resizeHit = hitResizeHandle(p.x, p.y);
                canvas.style.cursor = resizeHit ? resizeCursor(resizeHit.mode) : (hitBox(p.x, p.y) >= 0 ? "move" : "crosshair");
                return;
            }
            state.drawing.x2 = p.x;
            state.drawing.y2 = p.y;
            draw();
        }
        canvas.addEventListener("mousemove", handlePointerMove);
        window.addEventListener("mousemove", (event) => {
            if (state.resizing || state.moving || state.drawing) handlePointerMove(event);
        });
        window.addEventListener("mouseup", () => {
            if (state.resizing) {
                state.resizing = null;
                canvas.style.cursor = "crosshair";
                refresh();
                recordHistory();
                return;
            }
            if (state.moving) {
                const move = state.moving;
                const moved = move.moved;
                state.moving = null;
                canvas.style.cursor = "crosshair";
                if (moved) {
                    state.selected = -1;
                    refresh();
                    recordHistory();
                } else {
                    state.selected = move.index;
                    refresh();
                    focusSelectedBox();
                }
                return;
            }
            if (!state.drawing) return;
            const bbox = bboxFromRect(state.drawing);
            state.drawing = null;
            if (bbox[2] - bbox[0] >= 8 && bbox[3] - bbox[1] >= 8) {
                const type = state.newType;
                const element = type === "text" ? { type, bbox, text: "Text", desc: defaultTextDesc } : { type, bbox, desc: "Object" };
                state.elements.push(element);
                state.elementExtras.push({});
                state.selected = state.elements.length - 1;
                markDirty();
            } else {
                state.selected = -1;
            }
            showExtras();
            refresh();
            recordHistory();
            if (state.selected >= 0) focusSelectedBox();
        });
        boxEditor.addEventListener("mousedown", (event) => {
            event.stopPropagation();
        });
        boxEditor.addEventListener("input", () => {
            const element = state.elements[state.selected];
            if (!element) return;
            const value = stripTrailingLineBreaks(boxEditor.value);
            if (boxEditor.value !== value) boxEditor.value = value;
            if (isText(element)) {
                element.text = value;
            } else {
                element.desc = value;
            }
            fields.desc.value = element.desc || "";
            markPendingHistory();
            draw();
            renderList();
        });
        boxEditor.addEventListener("change", commitPendingHistory);
        boxEditor.addEventListener("blur", commitPendingHistory);
        list.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-index]");
            if (!button) return;
            selectElement(Number(button.getAttribute("data-index")), true);
        });
        list.addEventListener("focusin", (event) => {
            const input = event.target.closest("textarea[data-index]");
            if (!input) return;
            selectElement(Number(input.getAttribute("data-index")), false);
        });
        list.addEventListener("input", (event) => {
            const input = event.target.closest("textarea[data-index]");
            if (!input) return;
            const value = stripTrailingLineBreaks(input.value);
            if (input.value !== value) input.value = value;
            updateElementText(Number(input.getAttribute("data-index")), value);
        });
        list.addEventListener("change", (event) => {
            if (event.target.closest("textarea[data-index]")) commitPendingHistory();
        });
        list.addEventListener("focusout", (event) => {
            if (event.target.closest("textarea[data-index]")) commitPendingHistory();
        });
        document.addEventListener("mousedown", (event) => {
            if (event.button !== 0 || card.closest("[data-wangp-model-info-popup], .wangp-model-info-popup")?.hidden) return;
            if (event.target.closest(".ideogram4-helper-canvas-shell, .ideogram4-helper-list, .ideogram4-helper-selected, .ideogram4-helper-actions")) return;
            unselectElement();
        });
        splitter.addEventListener("pointerdown", (event) => {
            if (event.button !== 0) return;
            state.splitting = { pointerId: event.pointerId, startY: event.clientY, startTop: workspace.getBoundingClientRect().height };
            content.classList.add("is-splitting");
            splitter.setPointerCapture?.(event.pointerId);
            event.preventDefault();
        });
        window.addEventListener("pointermove", (event) => {
            if (!state.splitting || state.splitting.pointerId !== event.pointerId) return;
            updateSplitFromEvent(event);
            event.preventDefault();
        });
        function finishSplit(event) {
            if (!state.splitting || state.splitting.pointerId !== event.pointerId) return;
            state.splitting = null;
            content.classList.remove("is-splitting");
        }
        window.addEventListener("pointerup", finishSplit);
        window.addEventListener("pointercancel", finishSplit);
        splitter.addEventListener("keydown", (event) => {
            const step = event.shiftKey ? 0.08 : 0.03;
            if (event.key === "ArrowUp") state.splitRatio -= step;
            else if (event.key === "ArrowDown") state.splitRatio += step;
            else if (event.key === "Home") state.splitRatio = 0;
            else if (event.key === "End") state.splitRatio = 1;
            else return;
            state.splitRatio = helper.clamp(state.splitRatio, 0.1, 0.9);
            layoutPanels();
            event.preventDefault();
        });
        actionButtons.undo.addEventListener("click", undoChange);
        actionButtons.redo.addEventListener("click", redoChange);
        actionButtons.clear.addEventListener("click", clearHelper);
        actionButtons.apply.addEventListener("click", applyPrompt);
        actionButtons.delete.addEventListener("click", deleteSelected);
        ["aspect_ratio", "high", "background", "aesthetics", "lighting", "medium", "photo", "art_style", "style_palette", "extras"].forEach((name) => {
            fields[name].addEventListener("input", () => {
                markPendingHistory();
                setStatus("");
            });
            fields[name].addEventListener("change", commitPendingHistory);
            fields[name].addEventListener("blur", commitPendingHistory);
        });
        function updateSelectedProperties() {
            if (state.syncing || state.selected < 0) return;
            const element = state.elements[state.selected];
            const previousType = typeValue(element.type);
            element.type = typeValue(fields.type.value);
            state.newType = element.type;
            if (isText(element)) {
                if (!element.text) element.text = "Text";
                if (previousType !== "text" && (!element.desc || element.desc === "Object")) fields.desc.value = defaultTextDesc;
            } else if (previousType === "text" && (!element.desc || element.desc === defaultTextDesc)) {
                fields.desc.value = "Object";
            } else if (!isText(element)) {
                element.desc = "Object";
            }
            element.desc = fields.desc.value;
            element.color_palette = helper.parsePalette(fields.element_palette.value, 5);
            markPendingHistory();
            refresh();
        }
        ["type", "desc", "element_palette"].forEach((name) => {
            fields[name].addEventListener("input", updateSelectedProperties);
            fields[name].addEventListener("change", () => {
                updateSelectedProperties();
                commitPendingHistory();
            });
            fields[name].addEventListener("blur", commitPendingHistory);
        });
        document.addEventListener("change", (event) => {
            const target = document.getElementById(config.resolutionTarget || "");
            if (target && target.contains(event.target)) updateResolution();
        });
        window.addEventListener("resize", layoutPanels);
        window.visualViewport?.addEventListener("resize", layoutPanels);
        if (window.ResizeObserver) {
            const resizeObserver = new ResizeObserver(layoutPanels);
            resizeObserver.observe(card);
            resizeObserver.observe(content);
        }
        const popup = card.closest("[data-wangp-model-info-popup], .wangp-model-info-popup");
        popup?.addEventListener("wangp:model-info-before-close", (event) => {
            commitPendingHistory();
            updateDirty();
            if (!state.changedSinceOpen || !state.dirty) return;
            event.preventDefault();
            window.wangpConfirm({
                title: "Discard Changes?",
                message: "Close the prompt helper and discard your unapplied edits?",
                buttons: [
                    { text: "Keep Editing", value: "keep", cancel: true },
                    { text: "Discard", value: "discard", danger: true },
                    { text: "Save Changes", value: "save", primary: true, autofocus: true, action: applyPrompt }
                ]
            }).then((confirmed) => {
                if (confirmed !== "discard") return;
                state.dirty = false;
                state.changedSinceOpen = false;
                history.clean = snapshotKey(snapshotState());
                popup.hidden = true;
            });
        });
        const api = { open: () => { updateResolution(); loadDocFromPrompt(); requestAnimationFrame(layoutPanels); requestAnimationFrame(layoutPanels); } };
        card._ideogram4Helper = api;
        return api;
    };
    helper.open = function(card) {
        const api = helper.init(card);
        if (api) api.open();
    };
    if (!helper.boundOpenEvent) {
        document.addEventListener("wangp:model-info-opened", (event) => {
            const popup = event.target?.closest?.(".ideogram4-prompt-helper-popup");
            const card = popup?.querySelector("[data-ideogram4-prompt-helper]");
            if (card) helper.open(card);
        });
        helper.boundOpenEvent = true;
    }
})(window.wangpIdeogram4PromptHelper);
"""
