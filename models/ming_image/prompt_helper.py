"""Canvas editor for Ming Image's structured JSON prompt."""

import html
import json


def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
    dims = (model_def or {}).get("prompt_helper_popup_dims", (86, 94))
    try:
        width, height = int(dims[0]), int(dims[1])
    except (TypeError, ValueError, IndexError):
        width, height = 86, 94
    width, height = max(35, min(width, 96)), max(35, min(height, 100))
    config = html.escape(json.dumps({
        "promptTarget": prompt_elem_id,
        "resolutionTarget": resolution_elem_id,
        "modelType": model_type,
    }), quote=True)
    popup_id = html.escape(popup_id, quote=True)
    return f"""
<div id="{popup_id}" class="wangp-model-info-popup wangp-prompt-helper-popup ming-prompt-helper-popup" role="dialog" aria-label="Ming Image Prompt Helper" data-wangp-model-info-popup data-wangp-prompt-helper-popup="1" hidden style="width:min({width}vw,calc(100vw - 24px));height:min({height}vh,calc(100vh - 12px));">
  <div class="wangp-model-info-card ming-helper" data-ming-prompt-helper data-ming-config="{config}">
    <div class="wangp-model-info-titlebar" data-wangp-model-info-drag>
      <div class="wangp-model-info-heading">Ming Image JSON Prompt Helper</div>
      <button type="button" class="wangp-model-info-close" aria-label="Close prompt helper" data-wangp-model-info-close>&times;</button>
    </div>
    <div class="wangp-model-info-content ming-helper-content">
      <div class="ming-helper-toolbar">
        <button type="button" data-ming-action="add">Add Layer</button>
        <button type="button" data-ming-action="back">Move Back</button>
        <button type="button" data-ming-action="front">Move Forward</button>
        <button type="button" data-ming-action="delete">Delete</button>
        <button type="button" data-ming-action="undo">Undo</button>
        <button type="button" data-ming-action="redo">Redo</button>
        <span class="ming-helper-resolution" data-ming-resolution></span>
        <button type="button" data-ming-action="save">Save JSON Prompt</button>
      </div>
      <div class="ming-helper-raw" data-ming-raw hidden>
        <label for="{popup_id}-raw-json">The prompt could not be parsed. Edit its JSON here, then try again.</label>
        <textarea id="{popup_id}-raw-json" data-ming-raw-input spellcheck="false"></textarea>
        <button type="button" data-ming-action="retry">Try Parsing JSON</button>
      </div>
      <div class="ming-helper-status" data-ming-status role="status"></div>
      <div class="ming-helper-grid">
        <div class="ming-helper-visual">
          <div class="ming-helper-canvas-wrap"><canvas width="1000" height="1000" aria-label="Ming layer layout canvas"></canvas></div>
          <div class="ming-helper-hint">Drag on empty canvas or Alt-drag anywhere to add a layer. Drag a layer to move it; drag its bottom-right corner to resize it. Layers are listed back to front.</div>
          <div class="ming-helper-list" aria-label="Prompt layers"></div>
        </div>
        <div class="ming-helper-fields">
          <h3>Canvas Settings</h3>
          <label>Aspect Ratio<input data-ming-field="aspect_ratio" type="text" readonly></label>
          <label>Ambient Lighting<textarea data-ming-field="ambient_lighting"></textarea></label>
          <label>Image Style<textarea data-ming-field="image_style"></textarea></label>
          <h3>Selected Layer</h3>
          <label>Description<textarea data-ming-field="description"></textarea></label>
          <label>Coordinates<input data-ming-field="coordinates" type="text" placeholder="cx: 0.500, cy: 0.500, w: 0.400, h: 0.200"></label>
          <label>Hierarchy And Relation<textarea data-ming-field="hierarchy_and_relation"></textarea></label>
          <label>Color Specs<input data-ming-field="color_specs" type="text" placeholder="#14213D, #FFFFFF"></label>
        </div>
      </div>
    </div>
  </div>
</div>
"""


def get_prompt_helper_css():
    return """
.ming-prompt-helper-popup .wangp-model-info-content { padding: 12px; }
.ming-helper-content { display: flex; flex-direction: column; gap: 10px; min-height: 0; height: 100%; overflow: auto; box-sizing: border-box; }
.ming-helper-toolbar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.ming-helper-toolbar button { border: 1px solid var(--border-color-primary, #9cabb5); border-radius: 6px; padding: 6px 10px; background: var(--button-secondary-background-fill, #f5f8fa); color: var(--body-text-color, #173b50); cursor: pointer; }
.ming-helper-toolbar button:disabled { opacity: .45; cursor: default; }
.ming-helper-toolbar button[data-ming-action="save"] { margin-left: auto; background: var(--button-primary-background-fill, #176b92); color: var(--button-primary-text-color, white); }
.ming-helper-resolution { color: var(--body-text-color-subdued, #607785); font-size: .86rem; }
.ming-helper-raw { display: flex; flex-direction: column; gap: 7px; min-height: 0; flex: 1; }
.ming-helper-raw[hidden], .ming-helper-grid[hidden] { display: none; }
.ming-helper-raw label { font-size: .9rem; }
.ming-helper-raw textarea { flex: 1; min-height: 180px; width: 100%; box-sizing: border-box; resize: vertical; padding: 8px; border: 1px solid var(--border-color-primary, #b6c5ce); border-radius: 5px; background: var(--input-background-fill, white); color: var(--body-text-color, #173b50); font: .85rem/1.4 monospace; }
.ming-helper-raw button { align-self: flex-start; border: 1px solid var(--border-color-primary, #9cabb5); border-radius: 6px; padding: 6px 10px; background: var(--button-primary-background-fill, #176b92); color: var(--button-primary-text-color, white); cursor: pointer; }
.ming-helper-grid { display: grid; grid-template-columns: minmax(290px, 1.2fr) minmax(270px, 1fr); gap: 12px; min-height: 0; flex: 1; }
.ming-helper-visual, .ming-helper-fields { min-height: 0; overflow: auto; }
.ming-helper-visual { display: flex; flex-direction: column; gap: 8px; }
.ming-helper-canvas-wrap { display: flex; align-items: center; justify-content: center; min-height: 270px; height: min(52vh, 580px); overflow: hidden; border: 1px solid var(--border-color-primary, #b6c5ce); border-radius: 7px; background: #e9eef1; }
.ming-helper-canvas-wrap canvas { display: block; max-width: 100%; max-height: 100%; width: auto; height: auto; touch-action: none; cursor: crosshair; background: #fff; }
.ming-helper-hint { font-size: .82rem; color: var(--body-text-color-subdued, #607785); }
.ming-helper-list { display: flex; flex-direction: column; gap: 4px; overflow: auto; min-height: 60px; }
.ming-helper-list button { display: block; width: 100%; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 7px 9px; border: 1px solid var(--border-color-primary, #b6c5ce); border-radius: 5px; background: var(--background-fill-secondary, #f3f7f9); color: var(--body-text-color, #173b50); cursor: pointer; }
.ming-helper-list button[aria-selected="true"] { border-color: var(--button-primary-background-fill, #176b92); background: rgba(23, 107, 146, .13); }
.ming-helper-fields { display: flex; flex-direction: column; gap: 7px; }
.ming-helper-fields h3 { margin: 8px 0 0; font-size: .94rem; }
.ming-helper-fields label { display: flex; flex-direction: column; gap: 3px; font-size: .86rem; }
.ming-helper-fields input, .ming-helper-fields textarea { width: 100%; box-sizing: border-box; border: 1px solid var(--border-color-primary, #b6c5ce); border-radius: 5px; padding: 6px 8px; background: var(--input-background-fill, white); color: var(--body-text-color, #173b50); font: inherit; }
.ming-helper-fields textarea { min-height: 56px; resize: vertical; }
.ming-helper-fields textarea[data-ming-field="description"] { min-height: 105px; }
.ming-helper-fields input[readonly] { opacity: .7; }
.ming-helper-status { min-height: 1.3em; font-size: .85rem; }
.ming-helper-status.error { color: var(--error-text-color, #a41e2b); }
@media (max-width: 800px) { .ming-helper-grid { grid-template-columns: 1fr; } .ming-helper-canvas-wrap { height: 42vh; } }
"""


def get_prompt_helper_javascript():
    return r"""
(() => {
    const helper = window.wangpMingPromptHelper = window.wangpMingPromptHelper || {};
    if (helper.bound) return;
    helper.bound = true;
    const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
    const copy = (value) => JSON.parse(JSON.stringify(value));
    const COORDS = /^cx:\s*([+-]?\d*\.?\d+)\s*,\s*cy:\s*([+-]?\d*\.?\d+)\s*,\s*w:\s*([+-]?\d*\.?\d+)\s*,\s*h:\s*([+-]?\d*\.?\d+)\s*$/i;
    const HEX = /^#[0-9A-Fa-f]{6}$/;
    const COMMENT_LINE = /^\s*\\?#/;
    // Center and size are each rounded to 0.001, so an edge can drift by 0.00075.
    const COORD_EPS = 0.001;
    helper.stripCommentLines = (value) => String(value || "")
        .split(/\r?\n/).filter((line) => !COMMENT_LINE.test(line)).join("\n");
    helper.collectComments = (value) => {
        const comments = [];
        let contentLines = 0;
        for (const line of String(value || "").split(/\r?\n/)) {
            if (COMMENT_LINE.test(line)) comments.push({ line, before: contentLines });
            else if (line.trim()) contentLines++;
        }
        return { comments, contentLines };
    };
    helper.restoreComments = (json, layout) => {
        if (!layout?.comments?.length) return json;
        const lines = json.split("\n");
        const buckets = Array.from({ length: lines.length + 1 }, () => []);
        for (const comment of layout.comments) {
            const at = comment.before > 0 && comment.before >= layout.contentLines
                ? lines.length : clamp(comment.before, 0, lines.length);
            buckets[at].push(comment.line.replace(/^(\s*)\\(?=#)/, "$1"));
        }
        const output = [];
        for (let index = 0; index <= lines.length; index++) {
            output.push(...buckets[index]);
            if (index < lines.length) output.push(lines[index]);
        }
        return output.join("\n");
    };

    helper.parseCoordinates = (value) => {
        const match = String(value || "").match(COORDS);
        if (!match) return null;
        const [cx, cy, w, h] = match.slice(1).map(Number);
        if (![cx, cy, w, h].every(Number.isFinite) || w <= 0 || h <= 0 ||
            cx - w / 2 < -COORD_EPS || cx + w / 2 > 1 + COORD_EPS ||
            cy - h / 2 < -COORD_EPS || cy + h / 2 > 1 + COORD_EPS) return null;
        return { cx, cy, w, h };
    };
    helper.formatCoordinates = ({cx, cy, w, h}) =>
        `cx: ${cx.toFixed(3)}, cy: ${cy.toFixed(3)}, w: ${w.toFixed(3)}, h: ${h.toFixed(3)}`;
    helper.repairCoordinates = (value) => {
        const match = String(value || "").match(COORDS);
        if (!match) return null;
        let [cx, cy, w, h] = match.slice(1).map(Number);
        if (![cx, cy, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
        w = clamp(w, .02, 1); h = clamp(h, .02, 1);
        cx = clamp(cx, w / 2, 1 - w / 2);
        cy = clamp(cy, h / 2, 1 - h / 2);
        return helper.formatCoordinates({cx, cy, w, h});
    };
    helper.parseColors = (value) => {
        const values = String(value || "").split(",").map((part) => part.trim()).filter(Boolean);
        return values.every((color) => HEX.test(color)) ? values.map((color) => color.toUpperCase()) : null;
    };
    helper.defaultDoc = () => ({
        canvas_settings: { aspect_ratio: "", ambient_lighting: "", image_style: "" },
        layers: []
    });
    helper.normalizeDoc = (source) => {
        if (!source || typeof source !== "object" || Array.isArray(source) ||
            !source.canvas_settings || !Array.isArray(source.layers)) throw new Error("Expected canvas_settings and layers.");
        const canvas = source.canvas_settings;
        return {
            canvas_settings: {
                aspect_ratio: String(canvas.aspect_ratio || ""),
                ambient_lighting: String(canvas.ambient_lighting || ""),
                image_style: String(canvas.image_style || "")
            },
            layers: source.layers.map((layer) => ({
                description: String(layer?.description || ""),
                coordinates: String(layer?.coordinates || ""),
                hierarchy_and_relation: String(layer?.hierarchy_and_relation || ""),
                color_specs: Array.isArray(layer?.color_specs) ? layer.color_specs.map(String) : []
            }))
        };
    };
    helper.parseJsonPrompt = (value) => {
        let raw = String(value || "").trim();
        if (/^```(?:json)?\s*\n/i.test(raw) && /\n\s*```\s*$/.test(raw))
            raw = raw.replace(/^```(?:json)?\s*\n/i, "").replace(/\n\s*```\s*$/, "").trim();
        // Markdown-escaped underscores are not legal JSON escapes.
        const escapedUnderscores = raw.includes("\\_");
        if (escapedUnderscores) raw = raw.replace(/\\_/g, "_");
        function completeJson(candidate) {
            try { return { parsed: JSON.parse(candidate), completedBrackets: false }; }
            catch (originalError) {
                const stack = [];
                let quoted = false, escaped = false, mismatched = false;
                for (const char of candidate) {
                    if (quoted) {
                        if (escaped) escaped = false;
                        else if (char === "\\") escaped = true;
                        else if (char === '"') quoted = false;
                    } else if (char === '"') quoted = true;
                    else if (char === "{" || char === "[") stack.push(char);
                    else if (char === "}" || char === "]") {
                        if (stack.pop() !== (char === "}" ? "{" : "[")) mismatched = true;
                    }
                }
                if (quoted || mismatched || !stack.length) throw originalError;
                const closing = stack.reverse().map((open) => open === "{" ? "}" : "]").join("");
                try { return { parsed: JSON.parse(candidate + closing), completedBrackets: true }; }
                catch (_) { throw originalError; }
            }
        }
        const candidates = [{ text: raw, normalizedQuotes: false }];
        // Chat/Markdown copies can double the backslash that escapes a quote in a JSON string.
        const normalized = raw.replace(/\\\\(?=")/g, "\\");
        if (normalized !== raw) candidates.push({ text: normalized, normalizedQuotes: true });
        let originalError;
        for (const candidate of candidates) {
            try {
                const result = completeJson(candidate.text);
                return { doc: helper.normalizeDoc(result.parsed), completedBrackets: result.completedBrackets,
                    escapedUnderscores, normalizedQuotes: candidate.normalizedQuotes };
            } catch (error) { if (!originalError) originalError = error; }
        }
        throw originalError;
    };
    function gcd(a, b) { while (b) [a, b] = [b, a % b]; return a || 1; }
    function parseResolution(value) {
        const match = String(value || "").match(/(\d{2,5})\s*[x×]\s*(\d{2,5})/i);
        return match ? { width: Number(match[1]), height: Number(match[2]) } : null;
    }
    function readResolution(targetId) {
        const target = document.getElementById(targetId || "");
        const component = (window.gradio_config?.components || []).find((item) => item.props?.elem_id === targetId);
        const choices = component?.props?.choices || [];
        const candidates = [];
        if (target) {
            target.querySelectorAll("input, select, textarea, button").forEach((node) => candidates.push(node.value || node.textContent || ""));
            candidates.push(target.getAttribute("data-value") || "", target.textContent || "");
        }
        candidates.push(component?.props?.value || "");
        for (const candidate of candidates) {
            const direct = parseResolution(candidate);
            if (direct) return direct;
            for (const choice of choices) {
                const label = Array.isArray(choice) ? choice[0] : choice;
                const value = Array.isArray(choice) ? choice[1] : choice;
                if (String(candidate).trim() === String(label).trim() || String(candidate).trim() === String(value).trim()) {
                    const found = parseResolution(value) || parseResolution(label);
                    if (found) return found;
                }
            }
        }
        return { width: 2048, height: 2048 };
    }
    function aspectText(resolution) {
        const divisor = gcd(resolution.width, resolution.height);
        return `${resolution.width / divisor}:${resolution.height / divisor}, ${resolution.width} × ${resolution.height} px`;
    }
    function init(card) {
        if (card._mingPromptHelper) return card._mingPromptHelper;
        const config = JSON.parse(card.getAttribute("data-ming-config") || "{}");
        const popup = card.closest("[data-wangp-model-info-popup]");
        const canvas = card.querySelector("canvas");
        const ctx = canvas.getContext("2d");
        const list = card.querySelector("[data-ming-list], .ming-helper-list");
        const status = card.querySelector("[data-ming-status]");
        const rawPanel = card.querySelector("[data-ming-raw]");
        const rawInput = card.querySelector("[data-ming-raw-input]");
        const grid = card.querySelector(".ming-helper-grid");
        const resolutionLabel = card.querySelector("[data-ming-resolution]");
        const fields = Object.fromEntries(Array.from(card.querySelectorAll("[data-ming-field]"))
            .map((node) => [node.dataset.mingField, node]));
        const buttons = Object.fromEntries(Array.from(card.querySelectorAll("[data-ming-action]"))
            .map((node) => [node.dataset.mingAction, node]));
        const state = { doc: helper.defaultDoc(), selected: -1, resolution: { width: 2048, height: 2048 },
            gesture: null, preview: null, invalid: false, snapshots: [], cursor: -1,
            baselineDoc: "", baselinePrompt: "", parsedRawSource: null, commentLayout: null };
        function promptField() {
            const target = document.getElementById(config.promptTarget || "");
            return target?.querySelector("textarea") || target?.querySelector("input") || null;
        }
        function setStatus(message, error=false) {
            status.textContent = message || "";
            status.classList.toggle("error", !!error);
        }
        function showRaw(source) {
            rawInput.value = source;
            rawPanel.hidden = false;
            grid.hidden = true;
        }
        function showCanvas() {
            rawPanel.hidden = true;
            grid.hidden = false;
        }
        function hasUnsavedChanges() {
            if (!rawPanel.hidden) return rawInput.value !== state.baselinePrompt;
            return (state.parsedRawSource !== null && state.parsedRawSource !== state.baselinePrompt) ||
                JSON.stringify(state.doc) !== state.baselineDoc;
        }
        function selectedLayer() { return state.doc.layers[state.selected] || null; }
        function refreshList() {
            list.replaceChildren();
            state.doc.layers.forEach((layer, index) => {
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = `${index + 1}. ${layer.description || "New visible group"}`;
                button.title = layer.description || "New visible group";
                button.setAttribute("aria-selected", index === state.selected ? "true" : "false");
                button.addEventListener("click", () => select(index));
                list.appendChild(button);
            });
            buttons.back.disabled = state.selected <= 0;
            buttons.front.disabled = state.selected < 0 || state.selected >= state.doc.layers.length - 1;
            buttons.delete.disabled = state.selected < 0;
            buttons.undo.disabled = state.cursor <= 0;
            buttons.redo.disabled = state.cursor >= state.snapshots.length - 1;
        }
        function refreshFields() {
            const canvasSettings = state.doc.canvas_settings;
            fields.aspect_ratio.value = canvasSettings.aspect_ratio;
            fields.ambient_lighting.value = canvasSettings.ambient_lighting;
            fields.image_style.value = canvasSettings.image_style;
            const layer = selectedLayer();
            for (const key of ["description", "coordinates", "hierarchy_and_relation", "color_specs"]) {
                fields[key].disabled = !layer;
                fields[key].value = layer ? (key === "color_specs" ? layer.color_specs.join(", ") : layer[key]) : "";
            }
        }
        function draw() {
            const width = canvas.width, height = canvas.height;
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, width, height);
            ctx.strokeStyle = "#e4e9ed";
            ctx.lineWidth = 1;
            for (let step = 1; step < 4; step++) {
                ctx.beginPath(); ctx.moveTo(width * step / 4, 0); ctx.lineTo(width * step / 4, height); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, height * step / 4); ctx.lineTo(width, height * step / 4); ctx.stroke();
            }
            state.doc.layers.forEach((layer, index) => {
                const box = helper.parseCoordinates(layer.coordinates);
                if (!box) return;
                const x = (box.cx - box.w / 2) * width, y = (box.cy - box.h / 2) * height;
                const w = box.w * width, h = box.h * height;
                ctx.fillStyle = index === state.selected ? "rgba(19, 111, 151, .24)" : "rgba(52, 127, 164, .10)";
                ctx.fillRect(x, y, w, h);
                ctx.strokeStyle = index === state.selected ? "#0b729e" : "#5a8298";
                ctx.lineWidth = index === state.selected ? 3 : 2;
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = "#0b425e";
                ctx.font = "bold 15px sans-serif";
                ctx.fillText(`Layer ${index + 1}`, x + 6, y + 19, Math.max(20, w - 10));
                if (index === state.selected) {
                    ctx.fillStyle = "#0b729e";
                    ctx.fillRect(x + w - 8, y + h - 8, 12, 12);
                }
            });
            if (state.preview) {
                const box = state.preview;
                ctx.strokeStyle = "#0b729e"; ctx.lineWidth = 2;
                ctx.strokeRect(box.x * width, box.y * height, box.w * width, box.h * height);
            }
        }
        function refresh() { refreshList(); refreshFields(); draw(); }
        function select(index) { state.selected = index; refresh(); }
        function checkpoint() {
            const snapshot = JSON.stringify({ doc: state.doc, selected: state.selected });
            if (state.snapshots[state.cursor] === snapshot) return;
            state.snapshots = state.snapshots.slice(0, state.cursor + 1);
            state.snapshots.push(snapshot);
            state.cursor = state.snapshots.length - 1;
            refreshList();
        }
        function restore(delta) {
            const cursor = state.cursor + delta;
            if (cursor < 0 || cursor >= state.snapshots.length) return;
            state.cursor = cursor;
            const snapshot = JSON.parse(state.snapshots[cursor]);
            state.doc = snapshot.doc;
            state.selected = snapshot.selected;
            refresh();
        }
        function resetHistory() {
            state.snapshots = [JSON.stringify({ doc: state.doc, selected: state.selected })];
            state.cursor = 0;
            state.baselineDoc = JSON.stringify(state.doc);
            state.baselinePrompt = String(promptField()?.value || "");
            state.parsedRawSource = null;
            refreshList();
        }
        function updateResolution() {
            state.resolution = readResolution(config.resolutionTarget);
            state.doc.canvas_settings.aspect_ratio = aspectText(state.resolution);
            canvas.width = 1000;
            canvas.height = Math.round(1000 * state.resolution.height / state.resolution.width);
            resolutionLabel.textContent = `${state.resolution.width}×${state.resolution.height}`;
            refreshFields(); draw();
        }
        function loadSource(source, fromRaw=false) {
            state.commentLayout = helper.collectComments(source);
            const raw = helper.stripCommentLines(source).trim();
            const commentsRemoved = source.split(/\r?\n/).filter((line) => COMMENT_LINE.test(line)).length;
            state.invalid = false;
            state.doc = helper.defaultDoc();
            state.selected = -1;
            showCanvas();
            const notes = [];
            let invalidBoxes = 0;
            let repaired = false;
            if (commentsRemoved) notes.push(`Preserved ${commentsRemoved} comment line${commentsRemoved === 1 ? "" : "s"}; ignored for JSON parsing.`);
            if (raw.startsWith("{") || raw.startsWith("[") || raw.startsWith("```")) {
                try {
                    const parsed = helper.parseJsonPrompt(raw);
                    state.doc = parsed.doc;
                    if (parsed.completedBrackets) { notes.push("Completed missing closing JSON brackets."); repaired = true; }
                    if (parsed.escapedUnderscores) { notes.push("Removed Markdown escapes from JSON keys."); repaired = true; }
                    if (parsed.normalizedQuotes) { notes.push("Normalized escaped quotes in JSON descriptions."); repaired = true; }
                    let adjusted = 0;
                    for (const layer of state.doc.layers) {
                        if (helper.parseCoordinates(layer.coordinates)) continue;
                        const repaired = helper.repairCoordinates(layer.coordinates);
                        if (repaired && helper.parseCoordinates(repaired)) {
                            layer.coordinates = repaired;
                            adjusted++;
                        } else invalidBoxes++;
                    }
                    if (adjusted) { notes.push(`Moved ${adjusted} layer box${adjusted === 1 ? "" : "es"} inside the canvas.`); repaired = true; }
                    if (invalidBoxes) notes.push(`${invalidBoxes} layer box${invalidBoxes === 1 ? "" : "es"} need valid coordinates before saving.`);
                    state.selected = state.doc.layers.length ? 0 : -1;
                } catch (error) {
                    state.invalid = true;
                    showRaw(source);
                    setStatus(`Invalid JSON prompt: ${error.message}`, true);
                }
            } else if (raw) {
                state.doc.layers.push({ description: raw,
                    coordinates: "cx: 0.500, cy: 0.500, w: 1.000, h: 1.000",
                    hierarchy_and_relation: "Full-canvas visible group.", color_specs: [] });
                state.selected = 0;
            }
            updateResolution();
            refresh();
            resetHistory();
            if (fromRaw) state.parsedRawSource = source;
            if (!state.invalid) {
                if (notes.length) setStatus(`${notes.join(" ")}${repaired && !invalidBoxes ? " Save to apply repairs." : ""}`, invalidBoxes > 0);
                else setStatus("");
            }
        }
        function load() { loadSource(String(promptField()?.value || "")); }
        function validate() {
            if (state.invalid) throw new Error("Fix the invalid JSON prompt before saving, or replace it with a valid prompt.");
            if (!state.doc.layers.length) throw new Error("Add at least one visible layer.");
            const settings = state.doc.canvas_settings;
            if (!settings.ambient_lighting.trim() || !settings.image_style.trim())
                throw new Error("Describe ambient lighting and image style.");
            for (const [index, layer] of state.doc.layers.entries()) {
                if (!layer.description.trim()) throw new Error(`Layer ${index + 1} needs a description.`);
                if (!helper.parseCoordinates(layer.coordinates)) throw new Error(`Layer ${index + 1} needs valid normalized coordinates inside the canvas.`);
                if (!Array.isArray(layer.color_specs) || layer.color_specs.some((color) => !HEX.test(color)))
                    throw new Error(`Layer ${index + 1} needs six-digit hex color specs.`);
            }
        }
        function save() {
            try { validate(); } catch (error) { setStatus(error.message, true); return; }
            const field = promptField();
            if (!field) { setStatus("Prompt field was not found.", true); return; }
            const value = helper.restoreComments(JSON.stringify(state.doc, null, 2), state.commentLayout);
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
            if (setter && field instanceof window.HTMLTextAreaElement) setter.call(field, value);
            else field.value = value;
            field.dispatchEvent(new Event("input", { bubbles: true }));
            field.dispatchEvent(new Event("change", { bubbles: true }));
            resetHistory();
            setStatus("Applied JSON prompt.");
            if (popup) popup.hidden = true;
        }
        function point(event) {
            const rect = canvas.getBoundingClientRect();
            return { x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
                y: clamp((event.clientY - rect.top) / rect.height, 0, 1) };
        }
        function hit(point) {
            for (let index = state.doc.layers.length - 1; index >= 0; index--) {
                const box = helper.parseCoordinates(state.doc.layers[index].coordinates);
                if (!box) continue;
                const left = box.cx - box.w / 2, right = box.cx + box.w / 2;
                const top = box.cy - box.h / 2, bottom = box.cy + box.h / 2;
                if (point.x >= left && point.x <= right && point.y >= top && point.y <= bottom)
                    return { index, box, resize: Math.abs(point.x - right) < .025 && Math.abs(point.y - bottom) < .025 };
            }
            return null;
        }
        canvas.addEventListener("pointerdown", (event) => {
            const start = point(event), found = hit(start);
            if (found && !event.altKey) {
                select(found.index);
                state.gesture = { kind: found.resize ? "resize" : "move", start, box: found.box, index: found.index };
            } else state.gesture = { kind: "draw", start };
            canvas.setPointerCapture(event.pointerId);
            event.preventDefault();
        });
        canvas.addEventListener("pointermove", (event) => {
            const gesture = state.gesture;
            if (!gesture) return;
            const now = point(event);
            if (gesture.kind === "draw") {
                state.preview = { x: Math.min(now.x, gesture.start.x), y: Math.min(now.y, gesture.start.y),
                    w: Math.abs(now.x - gesture.start.x), h: Math.abs(now.y - gesture.start.y) };
            } else {
                const box = gesture.box;
                let cx = box.cx, cy = box.cy, w = box.w, h = box.h;
                if (gesture.kind === "move") {
                    cx = clamp(box.cx + now.x - gesture.start.x, box.w / 2, 1 - box.w / 2);
                    cy = clamp(box.cy + now.y - gesture.start.y, box.h / 2, 1 - box.h / 2);
                } else {
                    const left = box.cx - box.w / 2, top = box.cy - box.h / 2;
                    w = clamp(now.x - left, .02, 1 - left);
                    h = clamp(now.y - top, .02, 1 - top);
                    cx = left + w / 2; cy = top + h / 2;
                }
                state.doc.layers[gesture.index].coordinates = helper.formatCoordinates({cx, cy, w, h});
                fields.coordinates.value = state.doc.layers[gesture.index].coordinates;
            }
            draw();
        });
        function finishGesture(event) {
            const gesture = state.gesture;
            if (!gesture) return;
            if (gesture.kind === "draw" && state.preview?.w >= .02 && state.preview?.h >= .02) {
                const box = state.preview;
                const relation = state.doc.layers.length ? "In front of the previous visible groups." : "Backmost visible group.";
                state.doc.layers.push({ description: "", coordinates: helper.formatCoordinates({
                    cx: box.x + box.w / 2, cy: box.y + box.h / 2, w: box.w, h: box.h }),
                    hierarchy_and_relation: relation, color_specs: [] });
                state.selected = state.doc.layers.length - 1;
            }
            state.gesture = null; state.preview = null;
            if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
            checkpoint(); refresh();
        }
        canvas.addEventListener("pointerup", finishGesture);
        canvas.addEventListener("pointercancel", finishGesture);
        for (const key of ["ambient_lighting", "image_style", "description", "coordinates", "hierarchy_and_relation", "color_specs"]) {
            fields[key].addEventListener("input", () => {
                const layer = selectedLayer();
                if (key === "ambient_lighting" || key === "image_style") state.doc.canvas_settings[key] = fields[key].value;
                else if (layer && key !== "color_specs") layer[key] = fields[key].value;
                else if (layer) layer.color_specs = fields[key].value.split(",").map((part) => part.trim()).filter(Boolean);
                state.invalid = false;
                if (key === "description") refreshList();
                if (key === "coordinates") draw();
                setStatus("");
            });
            fields[key].addEventListener("change", checkpoint);
            fields[key].addEventListener("blur", checkpoint);
        }
        buttons.add.addEventListener("click", () => {
            if (state.invalid) showCanvas();
            state.doc.layers.push({ description: "", coordinates: "cx: 0.500, cy: 0.500, w: 0.400, h: 0.250",
                hierarchy_and_relation: state.doc.layers.length ? "In front of the previous visible groups." : "Backmost visible group.", color_specs: [] });
            state.selected = state.doc.layers.length - 1;
            state.invalid = false; checkpoint(); refresh(); fields.description.focus();
        });
        buttons.back.addEventListener("click", () => {
            const index = state.selected;
            if (index <= 0) return;
            [state.doc.layers[index - 1], state.doc.layers[index]] = [state.doc.layers[index], state.doc.layers[index - 1]];
            state.selected--; checkpoint(); refresh();
        });
        buttons.front.addEventListener("click", () => {
            const index = state.selected;
            if (index < 0 || index >= state.doc.layers.length - 1) return;
            [state.doc.layers[index + 1], state.doc.layers[index]] = [state.doc.layers[index], state.doc.layers[index + 1]];
            state.selected++; checkpoint(); refresh();
        });
        buttons.delete.addEventListener("click", () => {
            if (state.selected < 0) return;
            state.doc.layers.splice(state.selected, 1);
            state.selected = Math.min(state.selected, state.doc.layers.length - 1);
            checkpoint(); refresh();
        });
        buttons.undo.addEventListener("click", () => restore(-1));
        buttons.redo.addEventListener("click", () => restore(1));
        buttons.retry.addEventListener("click", () => loadSource(rawInput.value, true));
        buttons.save.addEventListener("click", save);
        document.addEventListener("change", (event) => {
            const target = document.getElementById(config.resolutionTarget || "");
            if (target?.contains(event.target)) { updateResolution(); checkpoint(); }
        });
        popup?.addEventListener("wangp:model-info-before-close", (event) => {
            if (!hasUnsavedChanges()) return;
            event.preventDefault();
            const discard = () => { popup.hidden = true; };
            if (window.wangpConfirm) {
                window.wangpConfirm({ title: "Discard Changes?", message: "Close the prompt helper and discard unapplied edits?",
                    buttons: [{ text: "Keep Editing", value: "keep", cancel: true },
                        { text: "Discard", value: "discard", danger: true },
                        { text: "Save Changes", value: "save", primary: true, action: save }] })
                    .then((choice) => { if (choice === "discard") discard(); });
            } else if (window.confirm("Discard unapplied prompt changes?")) discard();
        });
        const api = { open: load };
        card._mingPromptHelper = api;
        return api;
    }
    document.addEventListener("wangp:model-info-opened", (event) => {
        const popup = event.target?.closest?.(".ming-prompt-helper-popup");
        const card = popup?.querySelector("[data-ming-prompt-helper]");
        if (card) init(card).open();
    });
})();
"""
