// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* Only user gestures write gallery selection; Gradio refresh events are read-only. */
(() => {
  function view() {
    const value = document.querySelector('#wangp-gallery-view textarea')?.value;
    return value ? JSON.parse(value) : null;
  }
  let sequence = 0, pendingView = null, acceptedView = null;
  function select(source, index = null) {
    const state = view();
    if (!state) return;
    const paths = state[source];
    if (index !== null && (index < 0 || index >= paths.length)) return;
    pendingView = state;
    window.__wangpAssistantChatNS.setBridgeValue('#wangp-gallery-interaction textarea', JSON.stringify({session: state.session, workspace: state.workspace, source, index: index === null ? null : index + state[source + '_offset'], path: index === null ? null : paths[index], sequence: ++sequence}));
  }
  // A gallery tab (or its entry in Gradio's overflow menu, which has no tab attributes) is identified by its label.
  function tabSource(tab, step = 0) {
    const sources = view()?.sources || [{id: 'video', label: ''}];
    const index = Math.max(0, sources.findIndex(source => source.label.trim() === tab.textContent.trim()));
    return sources[(index + step + sources.length) % sources.length].id;
  }
  window.WanGPGallerySelection = {
    // A strip gallery thumbnail click: index in the visible window of gallery `source`.
    select: (source, index) => {const state = view(); if (state) select(source, Number(index) - state[source + '_offset']);},
    acceptView(value) {
      const state = JSON.parse(value);
      const current = acceptedView || view();
      // Network completion order is not publication order. Revision counters
      // belong to a Gradio session and can restart with a new server.
      if (state.session === current?.session && state.revision < current.revision) return false;
      const accept = state.session !== pendingView?.session || state.workspace !== pendingView?.workspace || state.gallery_sequence >= sequence;
      // Several responses may share one Gradio flush, before its DOM updates.
      if (accept) acceptedView = state;
      return accept;
    }
  };
  function indexOf(thumb) {
    return [...thumb.parentElement.querySelectorAll('.thumbnail-item')].indexOf(thumb);
  }
  document.addEventListener('click', event => {
    const tab = event.target.closest('#wangp-gallery-tabs [role=tab], #wangp-gallery-tabs .overflow-dropdown button');
    if (tab) {
      select(tabSource(tab));
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
      select(tabSource(tab, event.code === 'ArrowRight' ? 1 : -1));
      return;
    }
    if (!event.target.closest('#gallery') || !['ArrowLeft', 'ArrowRight'].includes(event.code)) return;
    const selected = document.querySelector('#gallery .thumbnail-item.selected'), state = view();
    if (selected && state.video.length) select('video', (indexOf(selected) + (event.code === 'ArrowRight' ? 1 : -1) + state.video.length) % state.video.length);
  }, true);
})();
