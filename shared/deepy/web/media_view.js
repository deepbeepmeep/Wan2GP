/* Shared presentation for gallery media and the existing Python property formatter. */
(() => {
  const documentHtml = html => '<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'"><style>body{margin:0;color:#24485d;font:13px/1.5 system-ui}table{width:100%;table-layout:fixed}td{overflow-wrap:anywhere}td:first-child{width:28%;white-space:normal!important}#video_info td{font-size:12px}b{font-weight:500}.copy-swap__full{display:none}.copy-swap:focus .copy-swap__trunc{display:none}.copy-swap:focus .copy-swap__full{display:inline}</style>' + html;
  function media(item) {
    const node = document.createElement(item.kind === 'image' ? 'img' : item.kind);
    node.setAttribute('aria-label', item.name);
    if (item.kind === 'image') {node.alt = item.name; node.loading = 'lazy';}
    else {node.controls = true; node.preload = 'metadata'; if (item.kind === 'video') {node.playsInline = true; if (item.poster) node.poster = item.poster;}}
    node.src = item.url;
    return node;
  }
  function thumbnail(item) {
    if (item.kind === 'audio') {
      const node = document.createElement('div'); node.className = 'media-audio-thumbnail';
      node.textContent = '♫'; node.setAttribute('aria-label', 'Audio');
      return node;
    }
    const node = document.createElement('img');
    node.alt = item.name; node.loading = 'lazy'; node.decoding = 'async';
    node.dataset.mediaKind = item.kind;
    node.src = item.thumbnail;
    return node;
  }
  function playButton(play) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'media-play';
    button.title = 'Play video'; button.setAttribute('aria-label', button.title);
    // Keep native scrolling and dragging; a moved/cancelled touch is not a tap.
    let pointer = null;
    button.onpointerdown = event => {pointer = {x: event.clientX, y: event.clientY, moved: false};};
    button.onpointermove = event => {if (pointer && Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 8) pointer.moved = true;};
    button.onpointercancel = () => {if (pointer) pointer.moved = true;};
    button.onclick = event => {
      event.stopPropagation();
      const moved = pointer?.moved; pointer = null;
      if (event.detail && moved) return;
      play();
    };
    return button;
  }
  window.WanGPMediaView = {documentHtml, media, thumbnail, playButton};
})();
