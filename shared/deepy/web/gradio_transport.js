(() => {
const WAC = window.__wangpAssistantChatNS = window.__wangpAssistantChatNS || {};
WAC.stopBridgeTargets = function () {
  const wrapper = document.querySelector('#assistant_chat_stop_bridge');
  if (!wrapper) return [];
  const targets = [wrapper];
  const button = wrapper.querySelector('button');
  if (button) targets.unshift(button);
  return targets.filter((target, index, items) => !!target && items.indexOf(target) === index);
};

WAC.pauseBridgeTargets = function () {
  const wrapper = document.querySelector('#assistant_chat_pause_bridge');
  if (!wrapper) return [];
  const targets = [wrapper];
  const button = wrapper.querySelector('button');
  if (button) targets.unshift(button);
  return targets.filter((target, index, items) => !!target && items.indexOf(target) === index);
};

WAC.requestCanonicalSync = function () {
  const button = document.querySelector('#assistant_chat_sync_button button, #assistant_chat_sync_button');
  if (!button || typeof button.click !== 'function') return false;
  button.click();
  return true;
};

WAC.queueBusyRequest = function (text, submissionId) {
  const input = document.querySelector('#assistant_chat_busy_queue_input textarea, #assistant_chat_busy_queue_input input');
  const button = document.querySelector('#assistant_chat_busy_queue_button button, #assistant_chat_busy_queue_button');
  if (!input || !button) return false;
  WAC.setBridgeValue('#assistant_chat_busy_queue_input textarea, #assistant_chat_busy_queue_input input', text);
  WAC.setBridgeValue('#assistant_chat_busy_queue_submission_id textarea, #assistant_chat_busy_queue_submission_id input', submissionId);
  if (typeof button.click === 'function') button.click();
  return true;
};

WAC.steerRequest = function (text, submissionId) {
  const button = document.querySelector('#assistant_chat_steer_button button, #assistant_chat_steer_button');
  if (!button) return false;
  WAC.setBridgeValue('#assistant_chat_steer_input textarea, #assistant_chat_steer_input input', text);
  WAC.setBridgeValue('#assistant_chat_steer_submission_id textarea, #assistant_chat_steer_submission_id input', submissionId);
  if (typeof button.click === 'function') button.click();
  return true;
};

WAC.queuedRequestAction = function (action, messageId, text) {
  const button = document.querySelector('#assistant_chat_queued_action_button button, #assistant_chat_queued_action_button');
  if (!button) return false;
  const payload = JSON.stringify({ action: String(action || ''), message_id: String(messageId || ''), text: String(text || '') });
  WAC.setBridgeValue('#assistant_chat_queued_action_input textarea, #assistant_chat_queued_action_input input', payload);
  if (typeof button.click === 'function') button.click();
  return true;
};

WAC.resumeSelectedSession = function (trigger) {
  const container = trigger && trigger.closest ? trigger.closest('.chat__session-picker') : null;
  const picker = container ? container.querySelector('[data-wac-session-picker]') : null;
  const storageId = String(picker && picker.value || '').trim();
  const bridgeHost = document.getElementById('assistant_chat_welcome_session_button');
  const bridge = bridgeHost && bridgeHost.matches && bridgeHost.matches('button') ? bridgeHost : bridgeHost && bridgeHost.querySelector('button');
  if (!storageId || !bridge || typeof bridge.click !== 'function') return false;
  if (!WAC.setBridgeValue('#assistant_chat_welcome_session_input textarea, #assistant_chat_welcome_session_input input', storageId)) return false;
  window.setTimeout(() => bridge.click(), 0);
  return true;
};

WAC.prefillResumedSession = function (requestId) {
  const normalizedRequestId = String(requestId || '').trim();
  if (!normalizedRequestId || normalizedRequestId === WAC.lastSessionResumeRequestId) return false;
  WAC.lastSessionResumeRequestId = normalizedRequestId;
  window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
    const bridgeHost = document.getElementById('assistant_chat_session_prefill_button');
    const bridge = bridgeHost && bridgeHost.matches && bridgeHost.matches('button') ? bridgeHost : bridgeHost && bridgeHost.querySelector('button');
    if (bridge && typeof bridge.click === 'function') bridge.click();
  }));
  return true;
};

WAC.eventSource = function () {
  return document.querySelector('#assistant_chat_event textarea, #assistant_chat_event input');
};

WAC.readEventSource = function () {
  const node = WAC.eventSource();
  if (!node) return;
  const value = typeof node.value === 'string' ? node.value.trim() : '';
  if (!value) return;
  WAC.consumePayload(value);
};

WAC.observeEventSourceValue = function (node) {
  if (!node || node.__wangpAssistantEventValueObserved) return;
  let prototype = node;
  let descriptor = null;
  while ((prototype = Object.getPrototypeOf(prototype))) {
    descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
    if (descriptor && typeof descriptor.get === 'function' && typeof descriptor.set === 'function') break;
  }
  if (!descriptor) return;
  try {
    Object.defineProperty(node, 'value', {
      configurable: true,
      enumerable: descriptor.enumerable,
      get() { return descriptor.get.call(this); },
      set(value) {
        descriptor.set.call(this, value);
        WAC.consumePayload(String(descriptor.get.call(this) || '').trim());
      },
    });
    Object.defineProperty(node, '__wangpAssistantEventValueObserved', { configurable: true, value: true });
  } catch (_error) {}
};

WAC.handleEventNodeMutation = function () {
  const node = WAC.eventSource();
  if (!node) return;
  WAC.observeEventSourceValue(node);
  if (node === WAC.eventNode) return;
  if (WAC.eventNode && WAC.eventNodeHandler) {
    WAC.eventNode.removeEventListener('input', WAC.eventNodeHandler, true);
    WAC.eventNode.removeEventListener('change', WAC.eventNodeHandler, true);
  }
  WAC.eventNode = node;
  const handler = function () { WAC.readEventSource(); };
  WAC.eventNodeHandler = handler;
  node.addEventListener('input', handler, true);
  node.addEventListener('change', handler, true);
  setTimeout(handler, 0);
};

WAC.installEventBridge = function () {
  WAC.handleEventNodeMutation();
  WAC.syncDisclosureBridge();
  WAC.syncScrollBridge();
  if (!WAC.pollTimer) WAC.pollTimer = window.setInterval(() => { WAC.readEventSource(); }, 250);
  window.addEventListener('focus', () => { WAC.readEventSource(); }, { passive: true });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) WAC.readEventSource();
  });
  window.addEventListener('resize', () => {
    // CSS viewport constraints have already changed by the time resize fires.
    const scrollState = WAC.lastScrollState;
    WAC.resetComposerLayout();
    WAC.syncDockLayout();
    window.requestAnimationFrame(() => {
      WAC.syncComposerLayout();
      WAC.applyAutoscrollState(scrollState);
    });
  }, { passive: true });
};

})();
