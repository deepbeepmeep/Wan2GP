/* Only user gestures write gallery selection; Gradio refresh events are read-only. */
(() => {
  function view() {
    const value = document.querySelector('#wangp-gallery-view textarea')?.value;
    return value ? JSON.parse(value) : null;
  }
  let sequence = 0;
  function select(source, index = null) {
    const state = view();
    if (!state) return;
    const paths = state[source];
    if (index !== null && (index < 0 || index >= paths.length)) return;
    window.__wangpAssistantChatNS.setBridgeValue('#wangp-gallery-interaction textarea', JSON.stringify({workspace: state.workspace, source, index: index === null ? null : index + state[source + '_offset'], path: index === null ? null : paths[index], sequence: ++sequence}));
  }
  window.WanGPGallerySelection = {audio: index => {const state = view(); if (state) select('audio', Number(index) - state.audio_offset);}};
  function indexOf(thumb) {
    return [...thumb.parentElement.querySelectorAll('.thumbnail-item')].indexOf(thumb);
  }
  document.addEventListener('click', event => {
    const tab = event.target.closest('#wangp-gallery-tabs [role=tab]');
    if (tab) {
      select([...tab.parentElement.querySelectorAll('[role=tab]')].indexOf(tab) === 0 ? 'video' : 'audio');
      return;
    }
    const thumb = event.target.closest('#gallery .thumbnail-item');
    if (thumb) {select('video', indexOf(thumb)); return;}
    if (event.target.closest('#gallery .preview')) {
      requestAnimationFrame(() => {
        const selected = document.querySelector('#gallery .thumbnail-item.selected');
        if (selected && indexOf(selected) !== view().selected) select('video', indexOf(selected));
      });
    }
  }, true);
  document.addEventListener('keydown', event => {
    const tab = event.target.closest('#wangp-gallery-tabs [role=tab]');
    if (tab && ['ArrowLeft', 'ArrowRight'].includes(event.code)) {
      select([...tab.parentElement.querySelectorAll('[role=tab]')].indexOf(tab) === 0 ? 'audio' : 'video');
      return;
    }
    if (!event.target.closest('#gallery') || !['ArrowLeft', 'ArrowRight'].includes(event.code)) return;
    const selected = document.querySelector('#gallery .thumbnail-item.selected'), state = view();
    if (selected && state.video.length) select('video', (indexOf(selected) + (event.code === 'ArrowRight' ? 1 : -1) + state.video.length) % state.video.length);
  }, true);
})();
