// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
    if (window.wangpPromptLayout) return;
    const helper = window.wangpPromptLayout = {};
    const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
    const COMMENT_LINE = /^\s*\\?#/;
    helper.stripCommentLines = (value, commentLine=COMMENT_LINE) => String(value || "")
        .split(/\r?\n/).filter((line) => !commentLine.test(line)).join("\n");
    helper.collectComments = (value, commentLine=COMMENT_LINE) => {
        const comments = [];
        let contentLines = 0;
        for (const line of String(value || "").split(/\r?\n/)) {
            if (commentLine.test(line)) comments.push({ line, before: contentLines });
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
    helper.readResolution = readResolution;
})();
