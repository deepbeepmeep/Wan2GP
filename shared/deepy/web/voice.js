(() => {
  const WAC = window.__wangpAssistantChatNS;
  const smartphone = navigator.userAgentData?.mobile || /iPhone|iPod|Android.*Mobile/i.test(navigator.userAgent);
  if (document.body.hasAttribute('data-deepy-app') && smartphone) return;
  if (WAC.voiceInstalled) return;
  WAC.voiceInstalled = true;
  let recorder = null, stream = null, chunks = [], timer = null, noticeTimer = null, starting = false;
  const icon = '<svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/></svg>';
  function release() {
    clearTimeout(timer);
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  function status(button, text, recording = false) {
    button.title = text;
    button.setAttribute('aria-label', text);
    button.setAttribute('aria-pressed', String(recording));
    button.classList.toggle('is-recording', recording);
  }
  async function toggle(button) {
    if (starting) return;
    if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      const httpsPort = document.body.dataset.deepyHttpsPort;
      const url = new URL(window.location.href);
      url.protocol = 'https:';
      if (httpsPort) url.port = httpsPort;
      WAC.voiceNotice('Open Deepy using trusted HTTPS to record a voice message.', {httpsUrl: httpsPort ? url.href : null});
      return;
    }
    if (!window.MediaRecorder) { WAC.voiceNotice('Audio recording is unavailable in this browser.'); return; }
    starting = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus'].find(type => MediaRecorder.isTypeSupported(type));
      recorder = mimeType ? new MediaRecorder(stream, {mimeType}) : new MediaRecorder(stream);
      chunks = [];
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => { release(); button.disabled = false; status(button, 'Record a voice message'); WAC.voiceNotice('Audio recording failed.'); };
      recorder.onstop = async () => {
        const type = recorder.mimeType;
        release();
        button.disabled = true;
        status(button, 'Transcribing…');
        button.classList.add('is-transcribing');
        const request = document.getElementById('assistant_chat_request');
        request.setAttribute('aria-busy', 'true');
        WAC.voiceNotice('Transcribing your voice message…', {persistent: true});
        try {
          const statusResponse = await fetch('deepy_api/voice');
          if (!statusResponse.ok) throw new Error('Unable to check Whisper availability.');
          const voiceStatus = await statusResponse.json();
          if (voiceStatus.download_required) WAC.voiceNotice('Preparing your transcription… Whisper files will download for this first use.', {persistent: true});
          const form = new FormData();
          const suffix = type.includes('mp4') ? 'mp4' : type.includes('ogg') ? 'ogg' : 'webm';
          form.append('file', new Blob(chunks, {type}), `voice.${suffix}`);
          const response = await fetch('deepy_api/transcribe', {method: 'POST', body: form});
          const result = await response.json();
          if (!response.ok) throw new Error(result.detail || 'Transcription failed.');
          const input = WAC.requestInput();
          WAC.setRequestInputValue([input.value.trim(), result.text.trim()].filter(Boolean).join('\n'));
          input.focus();
          WAC.scheduleComposerLayout();
          WAC.voiceNotice(result.text.trim() ? '' : 'No speech detected.');
        } catch (error) { WAC.voiceNotice(error.message); }
        finally { chunks = []; recorder = null; button.disabled = false; button.classList.remove('is-transcribing'); request.removeAttribute('aria-busy'); status(button, 'Record a voice message'); }
      };
      recorder.start();
      status(button, 'Stop recording and transcribe', true);
      WAC.voiceNotice('Recording… tap the microphone to stop.', {persistent: true});
      timer = setTimeout(() => { if (recorder?.state === 'recording') recorder.stop(); }, 180000);
    } catch (error) { release(); WAC.voiceNotice(error.message); }
    finally { starting = false; }
  }
  function positionNotice() {
    const notice = document.getElementById('assistant_chat_voice_notice');
    if (!notice || notice.hidden) return;
    const box = WAC.requestInput().getBoundingClientRect();
    notice.style.left = box.left + 'px';
    notice.style.maxWidth = box.width + 'px';
    notice.style.top = Math.max(8, box.top - notice.offsetHeight - 8) + 'px';
  }
  WAC.voiceNotice = (text, {persistent = false, httpsUrl = null} = {}) => {
    const request = document.getElementById('assistant_chat_request');
    if (!request) return;
    let notice = document.getElementById('assistant_chat_voice_notice');
    if (!notice) { notice = document.createElement('div'); notice.id = 'assistant_chat_voice_notice'; notice.setAttribute('role', 'status'); document.body.append(notice); }
    notice.textContent = text;
    notice.hidden = !text;
    clearTimeout(noticeTimer);
    if (text) {
      if (httpsUrl) { const link = document.createElement('a'); link.href = httpsUrl; link.textContent = 'Open HTTPS'; notice.append(link); }
      const close = document.createElement('button'); close.type = 'button'; close.textContent = '×'; close.setAttribute('aria-label', 'Dismiss microphone notice'); close.onclick = () => WAC.voiceNotice(''); notice.append(close);
      if (!persistent) noticeTimer = setTimeout(() => WAC.voiceNotice(''), 6000);
    }
    positionNotice();
  };
  function mount() {
    const ask = document.getElementById('assistant_chat_ask_button');
    if (!ask || document.getElementById('assistant_chat_microphone')) return;
    const button = document.createElement('button');
    button.type = 'button'; button.id = 'assistant_chat_microphone'; button.innerHTML = icon;
    status(button, 'Record a voice message');
    button.addEventListener('click', () => toggle(button));
    ask.before(button);
  }
  new MutationObserver(mount).observe(document.body, {childList: true, subtree: true});
  window.addEventListener('resize', positionNotice);
  window.addEventListener('pagehide', () => { if (recorder) recorder.onstop = null; release(); });
  mount();
})();
