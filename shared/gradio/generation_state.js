// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
    const forms = new Set();
    const operations = new Map();
    let current = null;
    function refresh(id) {
        if (current) window.__wangpAssistantChatNS.setBridgeValue('#' + id + ' textarea, #' + id + ' input', JSON.stringify(current));
    }
    function receive(value, reset = false) {
        if (!value) return;
        if (!reset && current && (value.instance !== current.instance || value.revision <= current.revision)) return;
        if (reset && current && value.instance !== current.instance) {
            // A restarted server cannot finish an old page-local operation.
            // Restore only controls whose progress was active in this page.
            for (const [id, component] of operations) {
                if (document.querySelector('#' + component + ' .wangp-progress:not(.finished)')) {
                    window.__wangpAssistantChatNS.setBridgeValue('#' + id + ' textarea', value.instance);
                }
            }
        }
        current = value;
        for (const id of forms) refresh(id);
    }
    function register(id) {
        forms.add(id);
        refresh(id);
    }
    function updates() {
        // A queued Gradio event must apply the newest stream state, including
        // after reconnect, rather than replay its earlier textbox payload.
        if (!current) return Array.from({length: 3}, () => ({__type__: 'update'}));
        return [
            current.html === null ? {__type__: 'update'} : {__type__: 'update', value: current.html, visible: true},
            {__type__: 'update', visible: !current.paused, interactive: !current.pending},
            {__type__: 'update', visible: current.paused},
        ];
    }
    function registerOperation(id, component) {
        operations.set(id, component);
    }
    window.WanGPGenerationState = {receive, register, registerOperation, updates};
})();
