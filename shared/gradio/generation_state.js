// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
    const forms = new Set();
    let current = null;
    function refresh(id) {
        if (current) window.__wangpAssistantChatNS.setBridgeValue('#' + id + ' textarea, #' + id + ' input', JSON.stringify(current));
    }
    function receive(value, reset = false) {
        if (!value) return;
        if (!reset && current && (value.instance !== current.instance || value.revision <= current.revision)) return;
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
    window.WanGPGenerationState = {receive, register, updates};
})();
