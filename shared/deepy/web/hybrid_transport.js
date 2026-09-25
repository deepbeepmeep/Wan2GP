(() => {
  const WAC = window.__wangpAssistantChatNS;
  // Gradio's hidden textbox can still contain an earlier session snapshot.
  // Hybrid chat has one ordered source of truth: the shared event stream.
  WAC.readEventSource = () => {};
  WAC.handleEventNodeMutation = () => {};
  WAC.observeEventSourceValue = () => {};
  clearInterval(WAC.pollTimer);
  WAC.pollTimer = null;
  let started = false;
  function boot() {
    const marker = document.querySelector('[data-deepy-hybrid]');
    const galleryTabs = document.querySelector('#wangp-gallery-tabs [role=tablist]');
    if (started || !marker || !galleryTabs) return;
    started = true;
    observer.disconnect();
    const legacyInput = WAC.eventSource();
    if (legacyInput?.__wangpAssistantEventValueObserved) delete legacyInput.value;
    let galleryRevision = -1, settingsRevision = -1, previewRevision = -1, generating = false, noticeTimer = null, unauthorized = false;
    let restoration = null, restoredId = null;
    const base = marker.dataset.deepyHybrid;
    const openApp = document.createElement('a');
    openApp.href = base; openApp.target = '_blank'; openApp.rel = 'noopener';
    openApp.className = 'chat__web-link'; openApp.textContent = 'Web app ↗';
    openApp.title = 'Continue this conversation in the Web app';
    document.querySelector('#assistant_chat_settings_panel .chat__template-modal-titlebar')?.append(openApp);
    function click(id) { document.querySelector('#' + id + ' button, button#' + id)?.click(); }
    let errorSequence = 0;
    function showError(data) {
      // Let Gradio render its native popup, including duration/title/visibility.
      // A distinct input value also delivers repeated errors with identical text.
      WAC.setBridgeValue('#deepy_hybrid_error textarea', JSON.stringify([++errorSequence, typeof data === 'string' ? {message: data} : data]));
    }
    function notice(text, persistent = false) {
      let node = document.getElementById('deepy-hybrid-notice');
      if (!node) {node = document.createElement('div'); node.id = 'deepy-hybrid-notice'; node.setAttribute('role', 'status'); node.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:10000;max-width:90vw;box-sizing:border-box;background:#fff0f1;color:#a72424;border:1px solid #e6a7ab;box-shadow:0 3px 12px #70222b20;padding:10px 16px;border-radius:12px;overflow-wrap:anywhere'; document.body.append(node);}
      node.textContent = text; node.hidden = !text;
      clearTimeout(noticeTimer);
      if (text && !persistent) noticeTimer = setTimeout(() => {node.hidden = true;}, 7000);
      if (unauthorized) {
        const link = openApp.cloneNode(true); link.textContent = 'Sign in';
        node.append(' ', link);
      }
    }
    function refresh(view) {
      const gallery = view.gallery !== galleryRevision, settings = view.settings !== settingsRevision;
      galleryRevision = view.gallery; settingsRevision = view.settings;
      if (gallery) {click('deepy_hybrid_gallery_sync'); viewer.invalidate();}
      if (settings) click('deepy_hybrid_settings_sync');
      if (view.preview !== previewRevision) {previewRevision = view.preview; click('deepy_hybrid_preview_sync');}
    }
    function catalog(state) {
      WAC.consumePayload({type: 'session_catalog', sessions: state.sessions, active_session_id: state.active_session_id, multi_session_enabled: state.multi_session});
    }
    function restoring(value) {
      restoration = value;
      const pending = value && (value.pending || restoredId !== value.id);
      WAC.setRestoration(pending ? value : null);
      // Readiness is per browser: retain loading until Gradio applies its gallery outputs.
      if (pending && !value.pending) click('deepy_hybrid_gallery_sync');
    }
    WAC.galleryRestored = (value, appliedRevision) => {
      // A selection can discard the response after its host revision was seen.
      // Complete that refresh even if no further host notification is coming.
      if (appliedRevision < galleryRevision) click('deepy_hybrid_gallery_sync');
      if (value && !value.pending && value.id === restoration?.id && !restoration.pending) {
        restoredId = value.id;
        WAC.setRestoration(null);
      }
    };
    function progress(data) {
      if (data && !generating) WAC.setBridgeValue('#wangp_main_status_trigger textarea, #wangp_main_status_trigger input', String(Date.now()));
      generating = !!data;
    }
    const transport = new DeepyConnection(base, {
      snapshot: state => {WAC.applyDisplaySettings(state.display_settings); renderWorkspaces(state.workspaces, true); restoring(state.restoration); WAC.consumePayload(state.chat); catalog(state); galleryRevision = settingsRevision = previewRevision = -1; refresh(state.host_view); progress(state.progress); window.WanGPFormSync.refresh(state.forms);},
      event: event => {
        if (event.type === 'snapshot') {WAC.applyDisplaySettings(event.data.display_settings); renderWorkspaces(event.data.workspaces); restoring(event.data.restoration); WAC.consumePayload(event.data.chat); catalog(event.data); refresh(event.data.host_view); progress(event.data.progress); window.WanGPFormSync.refresh(event.data.forms);}
        else if (event.type === 'workspaces') {renderWorkspaces(event.data); viewer.invalidate();}
        else if (event.type === 'forms') window.WanGPFormSync.refresh(event.data);
        else if (event.type === 'display_settings') WAC.applyDisplaySettings(event.data);
        else if (event.type === 'voice_config') WAC.setVoiceMode?.(event.data.mode);
        else if (event.type === 'restoration') restoring(event.data);
        else if (event.type === 'chat') WAC.consumePayload(event.data);
        else if (event.type === 'host_view') refresh(event.data);
        else if (event.type === 'sessions') catalog(event.data);
        else if (event.type === 'progress') progress(event.data);
        else if (event.type === 'error') showError(event.data);
      },
      connection: connected => {if (connected) unauthorized = false; notice(connected ? '' : unauthorized ? 'Sign in to the Web app, then return here.' : 'Connection to server lost. Reconnecting…', true);},
      unauthorized: () => {unauthorized = true; notice('Sign in to the Web app, then return here.', true);},
    });
    WAC.saveDisplaySettings = values => transport.request('display-settings', values);
    WAC.requestCanonicalSync = () => {transport.request('state').then(state => WAC.consumePayload(state.chat)).catch(error => showError(error.notification || error.message)); return true;};
    WAC.submitRequest = async (text, id, steering = false) => {
      WAC.clearRequestInput(text);
      try {await WAC.waitForChatUpload(); WAC.acceptRestorationSubmission(id, await transport.request('messages', {text, submission_id: id, steering}));}
      catch (error) {WAC.dropOptimisticSubmit(id); if (!WAC.requestInput().value) WAC.setRequestInputValue(text); showError(error.notification || error.message);}
    };
    WAC.queueBusyRequest = (text, id) => {WAC.submitRequest(text, id); return true;};
    WAC.steerRequest = (text, id) => {WAC.submitRequest(text, id, true); return true;};
    const control = (action, payload = {}) => transport.request('control', {action, payload}).catch(error => {
      if (action === 'resume') WAC.requestCanonicalSync();
      showError(error.notification || error.message);
    });
    // The native Gradio info event stacks messages without reusing the error bridge's callback.
    const workspaceInfo = message => galleryTabs.dispatchEvent(new CustomEvent('gradio', {bubbles: true, detail: {event: 'info', data: message}}));
    const workspaces = new WanGPWorkspacePicker(galleryTabs, (path, payload) => transport.request(path, payload), notice, {info: workspaceInfo});
    let sessionWorkspaces = null;
    function syncSessionWorkspaces() {
      const host = document.querySelector('#deepy-session-workspaces');
      if (!host) return;
      if (sessionWorkspaces?.node.parentNode !== host) {
        sessionWorkspaces?.dialog.remove();
        sessionWorkspaces = new WanGPWorkspacePicker(host, (path, payload) => transport.request(path, payload), notice, {deepy: true, management: false});
      }
      sessionWorkspaces.render(workspaces.state, true);
    }
    // Gradio mounts this field when its Sessions tab becomes visible.
    new MutationObserver(syncSessionWorkspaces).observe(document.querySelector('#assistant_chat_settings_panel'), {childList: true, subtree: true});
    function renderWorkspaces(state, reset = false) {workspaces.render(state, reset); syncSessionWorkspaces();}
    const viewer = new WanGPWorkspaceViewer(workspaces, transport);
    WAC.queuedRequestAction = (action, message_id, text) => {control('queued', {action, message_id, text}); return true;};
    WAC.stopBridgeTargets = () => [{click: () => control('stop')}];
    WAC.pauseBridgeTargets = () => [{click: () => control('pause')}];
    WAC.resumeSelectedSession = trigger => {
      const picker = trigger.closest('.chat__session-picker').querySelector('select');
      WAC.setStatus({visible: true, kind: 'session_loading', text: 'Loading Session ' + picker.selectedOptions[0].textContent});
      control('resume', {id: picker.value});
      return true;
    };
    WAC.prefillResumedSession = () => {};
    window.addEventListener('online', () => transport.connect());
    window.addEventListener('focus', () => {if (!transport.source) transport.connect();});
    window.addEventListener('pagehide', () => transport.close());
    window.addEventListener('pageshow', event => {if (event.persisted) transport.connect();});
    WAC.mountChatUpload((path, body) => transport.request(path, body), notice);
    transport.connect();
  }
  const observer = new MutationObserver(boot);
  observer.observe(document.body, {childList: true, subtree: true});
  boot();
})();
