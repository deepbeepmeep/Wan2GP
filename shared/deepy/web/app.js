(() => {
  const WAC = window.__wangpAssistantChatNS = window.__wangpAssistantChatNS || {};
  let source = null, reconnectTimer = null, noticeTimer = null, connecting = false;
  let currentTab = 'chat', chatScrollState = null, lastAnswer = '', dismissedAnswer = '', hasSnapshot = false;
  let settingsSessionId = null;
  let backgroundChatState = null;
  let mediaScrollState = null;
  let loadingSession = null, generationProgress = null;
  const $ = selector => document.querySelector(selector);
  function notice(text) {
    const node = $('#app-notice'); node.textContent = text; node.hidden = !text;
    clearTimeout(noticeTimer); if (text) noticeTimer = setTimeout(() => {node.hidden = true;}, 7000);
  }
  function connection(connected) {
    $('#connection').hidden = connected;
    $('#connection').dataset.state = connected ? 'connected' : 'disconnected';
  }
  async function api(path, body) {
    const response = await fetch('deepy_api/' + path, body === undefined ? {} : body instanceof FormData ? {method: 'POST', body} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const result = await response.json();
    if (!response.ok) {
      if (response.status === 401 && !$('#login-dialog').open) $('#login-dialog').showModal();
      throw new Error(typeof result.detail === 'string' ? result.detail : 'Request failed.');
    }
    return result;
  }
  function progress(data) {
    generationProgress = data;
    const sessionStatus = loadingSession || WAC.state.status;
    const loading = sessionStatus?.kind === 'session_loading' && currentTab !== 'chat';
    if (loading) data = {type: 'status', data: sessionStatus.text, aborting: false};
    $('#abort-generation').hidden = loading;
    const scrollState = currentTab === 'chat' ? WAC.captureAutoscrollState() : null;
    const node = $('#generation-progress'); node.hidden = !data;
    if (!data) { if (scrollState) WAC.scheduleComposerLayout(scrollState); return; }
    const bar = $('#generation-bar');
    const position = data.type === 'progress' ? data.data[0] : null;
    $('#generation-status').textContent = data.type === 'progress' ? data.data[1] : data.data;
    $('#generation-count').textContent = '';
    if (Array.isArray(position) && position[1] > 0) {
      const [step, total] = position;
      bar.max = total; bar.value = step;
      $('#generation-count').textContent = `${step} / ${total} ${data.data[3] || 'steps'} · ${Math.round(100 * step / total)}%`;
    } else if (typeof position === 'number' && position > 0) {
      bar.max = 1; bar.value = position;
      $('#generation-count').textContent = Math.round(position * 100) + '%';
    } else bar.removeAttribute('value');
    $('#abort-generation').disabled = data.aborting;
    $('#abort-generation span').textContent = data.aborting ? 'Aborting…' : 'Abort';
    if (scrollState) WAC.scheduleComposerLayout(scrollState);
  }
  function tab(name) {
    if (currentTab === 'chat' && name !== 'chat') chatScrollState = WAC.captureAutoscrollState();
    currentTab = name;
    if (name === 'chat') $('#answer-notice').hidden = true;
    document.querySelectorAll('.view').forEach(node => {node.hidden = node.id !== name + '-view';});
    document.querySelectorAll('[data-tab]').forEach(node => node.setAttribute('aria-selected', String(node.dataset.tab === name)));
    if (name === 'chat') { WAC.setDockOpen(true, false); WAC.scheduleComposerLayout(chatScrollState); chatScrollState = null; }
    if (name === 'visual' || name === 'audio') scrollGallerySelection($('#' + name + '-view'));
    if (name === 'settings' && settingsSessionId !== WAC.chatSessionId) loadSettings();
    progress(generationProgress);
  }
  function scrollGallerySelection(view) {
    if (view.hidden || !view.dataset.pendingMedia) return;
    requestAnimationFrame(() => {
      if (view.hidden) return;
      const card = [...view.querySelectorAll('.media-card')].find(node => node.dataset.mediaId === view.dataset.pendingMedia);
      if (card) card.scrollIntoView({block: 'nearest', behavior: 'smooth'});
      delete view.dataset.pendingMedia;
    });
  }
  function installGalleryDivider(divider) {
    const view = divider.closest('.gallery-view');
    const storageKey = 'deepy-gallery-split-' + view.id;
    const toggle = divider.querySelector('.gallery-details-toggle');
    const collapse = closed => {
      view.classList.toggle('details-collapsed', closed);
      toggle.setAttribute('aria-expanded', String(!closed));
      toggle.title = closed ? 'Show media information' : 'Hide media information';
      toggle.setAttribute('aria-label', toggle.title);
      localStorage.setItem(storageKey + '-collapsed', String(closed));
    };
    toggle.onclick = () => collapse(!view.classList.contains('details-collapsed'));
    if (localStorage.getItem(storageKey + '-collapsed') === 'true') collapse(true);
    const resize = value => {
      const percent = Math.max(20, Math.min(80, value));
      view.style.setProperty('--gallery-split', percent + '%');
      divider.setAttribute('aria-valuenow', String(Math.round(percent)));
    };
    const saved = localStorage.getItem(storageKey);
    if (saved !== null && Number.isFinite(Number(saved))) resize(Number(saved));
    divider.onpointerdown = event => {
      if (event.button !== 0 || event.target.closest('button')) return;
      if (view.classList.contains('details-collapsed')) collapse(false);
      event.preventDefault(); divider.setPointerCapture(event.pointerId); view.classList.add('is-resizing');
    };
    divider.onpointermove = event => {
      if (!divider.hasPointerCapture(event.pointerId)) return;
      const box = view.getBoundingClientRect(); resize(100 * (event.clientY - box.top) / box.height);
    };
    divider.onlostpointercapture = () => {view.classList.remove('is-resizing'); localStorage.setItem(storageKey, divider.getAttribute('aria-valuenow'));};
    divider.onpointerup = event => { if (divider.hasPointerCapture(event.pointerId)) divider.releasePointerCapture(event.pointerId); };
    divider.onkeydown = event => {
      if (event.target === toggle) return;
      const current = Number(divider.getAttribute('aria-valuenow'));
      const next = {ArrowUp: current - 5, ArrowDown: current + 5, Home: 20, End: 80}[event.key];
      if (next === undefined) return;
      event.preventDefault(); collapse(false); resize(next); localStorage.setItem(storageKey, divider.getAttribute('aria-valuenow'));
    };
  }
  function renderSettings(form) {
    function field(def, container) {
      const label = document.createElement('label'); label.textContent = def.label;
      const input = document.createElement(def.choices ? 'select' : 'input'); input.name = def.key;
      if (def.choices) for (const [text, value] of def.choices) { const option = document.createElement('option'); option.value = String(value); option.textContent = text; input.append(option); }
      else { input.type = 'number'; input.min = def.minimum; input.max = def.maximum; input.step = def.step; input.required = true; }
      input.value = String(form.values[def.key]); label.append(input); container.append(label);
    }
    for (const id of ['property-mode', 'property-fields', 'template-fields']) $('#' + id).replaceChildren();
    field({key: 'use_template_properties', label: 'Default Dimensions / Durations / Seed', choices: form.property_modes}, $('#property-mode'));
    for (const def of form.properties) field(def, $('#property-fields'));
    for (const def of form.templates) field(def, $('#template-fields'));
    const mode = $('#property-mode select');
    mode.onchange = () => { for (const input of $('#property-fields').querySelectorAll('input')) input.disabled = mode.value === 'true'; };
    mode.onchange(); $('#save-settings').disabled = false;
  }
  async function loadSettings() {
    $('#settings-status').textContent = 'Loading…'; $('#save-settings').disabled = true;
    try { renderSettings(await api('settings')); settingsSessionId = WAC.chatSessionId; $('#settings-status').textContent = ''; }
    catch (error) { $('#settings-status').textContent = error.message; }
  }
  function consumeChat(payload, notify = true) {
    const wasLoading = WAC.state.status?.kind === 'session_loading';
    WAC.consumePayload(payload);
    if (loadingSession) WAC.setStatus(loadingSession);
    if (wasLoading || WAC.state.status?.kind === 'session_loading') progress(generationProgress);
    const messageId = [...WAC.state.order].reverse().find(id => WAC.state.messages[id].role === 'assistant');
    const message = messageId && WAC.messageNode(messageId);
    const text = message ? [...message.querySelectorAll('[data-block-type="markdown"]')].map(node => node.textContent).join(' ').replace(/\s+/g, ' ').trim() : '';
    const answerId = WAC.chatSessionId + ':' + messageId;
    const preview = text.slice(0, 160) + (text.length > 160 ? '…' : '');
    const signature = answerId + ':' + preview;
    if (!notify || !text) $('#answer-notice').hidden = true;
    else if (currentTab !== 'chat' && signature !== lastAnswer && answerId !== dismissedAnswer) {
      $('#answer-title').textContent = $('#deepy-name').textContent + ' replied';
      $('#answer-preview').textContent = preview;
      $('#answer-notice').dataset.answerId = answerId;
      $('#answer-notice').hidden = false;
    }
    lastAnswer = signature;
  }
  function openMedia(item) {
    const viewer = $('#media-viewer');
    if (currentTab === 'chat') mediaScrollState = {sessionId: WAC.chatSessionId, scroll: WAC.captureAutoscrollState()};
    const media = viewer.querySelector(item.kind === 'image' ? 'img' : item.kind);
    for (const node of viewer.querySelectorAll('img, video, audio')) node.hidden = node !== media;
    if (item.kind === 'video') media.poster = item.poster;
    media.src = item.url;
    media.setAttribute('aria-label', item.name);
    if (item.kind === 'image') media.alt = item.name;
    viewer.showModal();
  }
  WAC.openAttachment = link => {
    const kind = link.dataset.mediaKind;
    if (!['image', 'video', 'audio'].includes(kind)) return false;
    openMedia({kind, url: link.href, name: link.querySelector('.chat__attachment-title').textContent, poster: link.querySelector('img')?.src || ''});
    return true;
  };
  function gallery(items) {
    for (const [scope, audio] of [['visual', false], ['audio', true]]) {
      const container = $('#' + scope + '-gallery');
      const wanted = items.filter(item => (item.kind === 'audio') === audio);
      const previous = [...container.querySelectorAll('.media-card')];
      const followedLast = !previous.length || previous[previous.length - 1].classList.contains('selected');
      const view = $('#' + scope + '-view');
      const info = $('#' + scope + '-info');
      const selected = wanted.find(item => item.selected);
      const mediaId = selected ? selected.id : '';
      if (view.dataset.pendingMedia !== mediaId) delete view.dataset.pendingMedia;
      if (mediaId && followedLast && !previous.some(card => card.dataset.mediaId === mediaId)) view.dataset.pendingMedia = mediaId;
      if (info.dataset.mediaId !== mediaId) {
        info.dataset.mediaId = mediaId;
        const documentHtml = html => '<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'"><style>body{margin:0;color:#24485d;font:13px/1.5 system-ui}table{width:100%;table-layout:fixed}td{overflow-wrap:anywhere}td:first-child{width:28%;white-space:normal!important}#video_info td{font-size:12px}b{font-weight:500}.copy-swap__full{display:none}.copy-swap:focus .copy-swap__trunc{display:none}.copy-swap:focus .copy-swap__full{display:inline}</style>' + html;
        info.srcdoc = documentHtml(mediaId ? 'Loading media information…' : 'Select media to view its information.');
        if (mediaId) api('media/' + encodeURIComponent(mediaId) + '/info').then(result => {
          if (info.dataset.mediaId === mediaId) info.srcdoc = documentHtml(result.html);
        }).catch(error => { if (info.dataset.mediaId === mediaId) {info.srcdoc = documentHtml('Could not load media information.'); notice(error.message);} });
      }
      const ids = new Set(wanted.map(item => item.id));
      for (const card of container.querySelectorAll('[data-media-id]')) if (!ids.has(card.dataset.mediaId)) card.remove();
      container.querySelector('.gallery-empty')?.remove();
      for (const item of wanted) {
        let card = [...container.children].find(node => node.dataset.mediaId === item.id);
        if (!card) {
          card = document.createElement('article'); card.className = 'media-card'; card.dataset.mediaId = item.id;
          card.tabIndex = 0; card.setAttribute('aria-label', item.name);
          card.onclick = async event => {
            if (event.target.closest('a, button')) return;
            try { const result = await api('media/' + encodeURIComponent(item.id) + '/select', {}); gallery(result.gallery); }
            catch (error) { notice(error.message); }
          };
          card.onkeydown = event => { if (event.target === card && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); card.click(); } };
          const media = document.createElement(item.kind === 'image' ? 'img' : item.kind);
          if (item.kind === 'image') {
            media.alt = item.name; media.loading = 'lazy';
          }
          else { media.controls = true; media.preload = 'metadata'; if (item.kind === 'video') { media.playsInline = true; media.poster = item.poster; } }
          media.src = item.url;
          const title = document.createElement('div'); title.className = 'media-name'; title.textContent = item.name;
          const actions = document.createElement('div'); actions.className = 'media-actions';
          if (item.kind === 'image') {
            const expand = document.createElement('button'); expand.type = 'button'; expand.title = 'View full screen'; expand.setAttribute('aria-label', 'View full screen: ' + item.name);
            expand.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 3H3v6M15 3h6v6M3 15v6h6M21 15v6h-6"/></svg>';
            expand.onclick = () => openMedia(item); actions.append(expand);
          }
          const download = document.createElement('a'); download.href = item.url + '?download=true'; download.download = item.name; download.title = 'Download'; download.setAttribute('aria-label', 'Download ' + item.name);
          download.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5"/></svg>';
          actions.append(download); card.append(media, title, actions); container.append(card);
        }
        if (item.kind === 'video' && card.querySelector('video').poster !== item.poster) card.querySelector('video').poster = item.poster;
        card.classList.toggle('selected', item.selected);
        card.setAttribute('aria-current', String(item.selected));
      }
      if (!wanted.length) { const empty = document.createElement('div'); empty.className = 'gallery-empty'; empty.textContent = 'Your imports and creations will appear here.'; container.append(empty); }
      scrollGallerySelection(view);
    }
  }
  function snapshot(state, notify = false) {
    $('#deepy-name').textContent = state.deepy_type === 'prime' ? 'Deepy Prime' : 'Deepy Zero';
    WAC.multiSessionEnabled = state.multi_session;
    consumeChat(state.chat, notify); gallery(state.gallery); progress(state.progress);
    WAC.activeSessionId = state.active_session_id;
    const picker = $('#saved-session'); picker.replaceChildren();
    if (!WAC.activeSessionId) { const current = document.createElement('option'); current.value = ''; current.textContent = 'Current conversation'; current.disabled = true; picker.append(current); }
    for (const item of state.sessions) { const option = document.createElement('option'); option.value = item.id; option.textContent = item.title; picker.append(option); }
    picker.value = WAC.activeSessionId;
    $('.session-bar').hidden = !state.multi_session;
    WAC.setQueuedEditButtonLabels(!!WAC.queuedEditMessageId);
  }
  async function connect() {
    if (connecting) return;
    connecting = true;
    clearTimeout(reconnectTimer);
    source?.close();
    try {
      const state = await api('state'); snapshot(state, hasSnapshot); hasSnapshot = true;
      if (backgroundChatState && currentTab === 'chat' && backgroundChatState.sessionId === WAC.chatSessionId) {
        WAC.applyAutoscrollState(backgroundChatState.scroll);
        WAC.scheduleComposerLayout(backgroundChatState.scroll);
      }
      backgroundChatState = null;
      $('#login-dialog').close(); connection(true);
      source = new EventSource('deepy_api/events?after=' + state.cursor);
      source.onmessage = event => {
        const message = JSON.parse(event.data);
        if (message.type === 'snapshot') snapshot(message.data, true);
        else if (message.type === 'chat') consumeChat(message.data);
        else if (message.type === 'gallery') gallery(message.data);
        else if (message.type === 'progress') progress(message.data);
        else if (message.type === 'error') notice(message.data);
      };
      source.onerror = () => { source.close(); connection(false); reconnectTimer = setTimeout(connect, 1500); };
    } catch (error) {
      connection(false);
      if (!$('#login-dialog').open) reconnectTimer = setTimeout(connect, 2500);
    } finally { connecting = false; }
  }
  async function control(action, payload = {}) {
    const changingSession = action === 'resume' || action === 'reset';
    if (action === 'resume' && (!payload.id || payload.id === WAC.activeSessionId)) return;
    const generating = !!generationProgress;
    if (changingSession && (generating || WAC.isAssistantBusy())) {
      $('#saved-session').value = WAC.activeSessionId;
      notice(generating ? 'A generation is in progress. Wait for it to finish before changing conversation.' : 'Deepy is active in this conversation. Wait for it to finish before changing conversation.');
      return;
    }
    if (action === 'resume') {
      const title = [...$('#saved-session').options].find(option => option.value === payload.id).textContent;
      loadingSession = {visible: true, kind: 'session_loading', text: 'Loading Session ' + title};
      WAC.setStatus(loadingSession); progress(generationProgress);
    }
    try { const state = await api('control', {action, payload}); if (action === 'resume') loadingSession = null; snapshot(state); }
    catch (error) {
      if (action === 'resume') { loadingSession = null; WAC.setStatus(null); progress(generationProgress); }
      if (changingSession) $('#saved-session').value = WAC.activeSessionId;
      notice(error.message);
    }
  }
  WAC.submitRequest = async (text, submissionId, steering = false) => {
    try {
      await api('messages', {text, submission_id: submissionId, steering});
      WAC.clearRequestInput(text);
      if (window.matchMedia('(pointer: coarse)').matches && !WAC.requestInput().value && $('#assistant_chat_controls').contains(document.activeElement)) document.activeElement.blur();
    }
    catch (error) { WAC.dropOptimisticSubmit(submissionId); notice(error.message); }
  };
  WAC.queueBusyRequest = (text, id) => {WAC.submitRequest(text, id); return true;};
  WAC.steerRequest = (text, id) => {WAC.submitRequest(text, id, true); return true;};
  WAC.queuedRequestAction = (action, message_id, text) => {control('queued', {action, message_id, text}); return true;};
  WAC.stopBridgeTargets = () => [{click: () => control('stop')}];
  WAC.pauseBridgeTargets = () => [{click: () => control('pause')}];
  WAC.requestCanonicalSync = () => {api('state').then(state => snapshot(state, true)).catch(error => notice(error.message)); return true;};
  WAC.resumeSelectedSession = trigger => {const picker = trigger.closest('.chat__session-picker').querySelector('select'); control('resume', {id: picker.value});};
  WAC.prefillResumedSession = () => {};
  WAC.handleEventNodeMutation = () => {};
  WAC.readEventSource = () => {};
  WAC.installEventBridge = () => {window.addEventListener('resize', () => WAC.scheduleComposerLayout());};
  document.addEventListener('DOMContentLoaded', () => {
    const request = WAC.requestInput();
    const resizeRequest = () => {
      const scrollState = WAC.captureAutoscrollState();
      request.style.height = 'auto';
      request.style.height = request.scrollHeight + 'px';
      WAC.scheduleComposerLayout(scrollState);
    };
    request.addEventListener('input', resizeRequest);
    window.addEventListener('resize', resizeRequest);
    const composer = $('#assistant_chat_controls');
    composer.addEventListener('pointerdown', event => { if (event.button === 0 && event.target.closest('button')) event.preventDefault(); });
    new ResizeObserver(() => {
      if (!composer.offsetHeight) return;
      const scrollState = WAC.composerResizeScrollState || WAC.captureAutoscrollState();
      WAC.panel().style.setProperty('--deepy-composer-height', composer.offsetHeight + 'px');
      WAC.scheduleComposerLayout(scrollState);
    }).observe(composer);
    resizeRequest();
    document.querySelectorAll('.gallery-divider').forEach(installGalleryDivider);
    const settingsTabs = [...document.querySelectorAll('[data-settings-tab]')];
    for (const button of settingsTabs) {
      button.onclick = () => {
        for (const item of settingsTabs) { const active = item === button; item.setAttribute('aria-selected', String(active)); item.tabIndex = active ? 0 : -1; $('#' + item.dataset.settingsTab + '-settings').hidden = !active; }
      };
      button.onkeydown = event => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {event.preventDefault(); const next = settingsTabs.find(item => item !== button); next.click(); next.focus();} };
    }
    $('#deepy-settings-form').oninput = () => {$('#settings-status').textContent = 'Unsaved changes';};
    $('#deepy-settings-form').onsubmit = async event => {
      event.preventDefault(); $('#save-settings').disabled = true;
      const values = {};
      for (const input of $('#deepy-settings-form').querySelectorAll('[name]')) values[input.name] = input.name === 'use_template_properties' ? input.value === 'true' : input.type === 'number' ? Number(input.value) : input.value;
      try { renderSettings(await api('settings', values)); $('#settings-status').textContent = 'Settings saved'; }
      catch (error) { $('#settings-status').textContent = error.message; }
      finally { $('#save-settings').disabled = false; }
    };
    $('#abort-generation').onclick = async () => {
      $('#abort-generation').disabled = true;
      try { const state = await api('control', {action: 'abort', payload: {}}); progress(state.progress); }
      catch (error) { $('#abort-generation').disabled = false; notice(error.message); }
    };
    $('#read-answer').onclick = () => {chatScrollState = {atBottom: true, top: 0}; tab('chat');};
    $('#dismiss-answer').onclick = () => {dismissedAnswer = $('#answer-notice').dataset.answerId; $('#answer-notice').hidden = true;};
    $('#close-media-viewer').onclick = () => $('#media-viewer').close();
    $('#media-viewer').addEventListener('close', () => {
      for (const media of $('#media-viewer').querySelectorAll('img, video, audio')) {
        if (media.tagName !== 'IMG') media.pause();
        media.removeAttribute('src');
        if (media.tagName !== 'IMG') media.load();
      }
      if (mediaScrollState && currentTab === 'chat' && mediaScrollState.sessionId === WAC.chatSessionId) WAC.scheduleComposerLayout(mediaScrollState.scroll);
      mediaScrollState = null;
    });
    document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => tab(button.dataset.tab));
    $('#assistant_chat_reset_button').onclick = () => control('reset');
    $('#saved-session').onchange = () => control('resume', {id: $('#saved-session').value});
    $('#login-form').onsubmit = async event => {event.preventDefault(); try {await api('login', {token: $('#access-key').value}); $('#access-key').value = ''; $('#login-dialog').close(); connect();} catch (error) {$('#login-error').textContent = error.message;}};
    $('#login-dialog').addEventListener('cancel', event => event.preventDefault());
    for (const input of document.querySelectorAll('input[type=file]')) input.onchange = async () => {
      input.disabled = true;
      try { for (const file of input.files) {notice('Uploading ' + file.name + '…'); const form = new FormData(); form.append('file', file); const result = await api('media', form); gallery(result.gallery);} notice('Import complete.'); }
      catch (error) {notice(error.message);} finally {input.disabled = false; input.value = '';}
    };
    const rememberBackgroundScroll = () => {
      if (currentTab === 'chat' && !backgroundChatState) backgroundChatState = {sessionId: WAC.chatSessionId, scroll: WAC.captureAutoscrollState()};
    };
    const viewport = $('meta[name=viewport]');
    const viewportContent = viewport.content;
    let resumeFrame = 0;
    const resume = () => {
      if (document.hidden) return;
      rememberBackgroundScroll();
      if (!window.matchMedia('(pointer: coarse)').matches) { connect(); return; }
      cancelAnimationFrame(resumeFrame);
      // WebKit can discard the mobile viewport on resume (bug 262207).
      // Reapply it across layout frames before restoring the chat scroll position.
      viewport.content = viewportContent.replace('viewport-fit=cover', 'viewport-fit=contain');
      resumeFrame = requestAnimationFrame(() => {
        viewport.content = viewportContent;
        resumeFrame = requestAnimationFrame(() => { resumeFrame = 0; if (!document.hidden) connect(); });
      });
    };
    document.addEventListener('visibilitychange', () => { if (document.hidden) rememberBackgroundScroll(); else resume(); });
    for (const type of ['gesturestart', 'gesturechange']) document.addEventListener(type, event => event.preventDefault(), {passive: false});
    for (const type of ['contextmenu', 'dragstart']) document.addEventListener(type, event => {
      if (window.matchMedia('(pointer: coarse)').matches && event.target.closest('.media-card img, .media-actions a, #media-viewer img')) event.preventDefault();
    });
    window.addEventListener('online', connect);
    window.addEventListener('offline', () => { source?.close(); clearTimeout(reconnectTimer); connection(false); });
    window.addEventListener('pagehide', () => {rememberBackgroundScroll(); source?.close(); clearTimeout(reconnectTimer);});
    window.addEventListener('pageshow', event => {if (event.persisted) resume();});
    window.addEventListener('blur', rememberBackgroundScroll);
    window.addEventListener('focus', resume);
    tab('chat'); connect();
  });
})();
