// The backend can see HTTP after a proxy terminates browser HTTPS. Gradio's
// embedded root then points the browser at mixed-content API and queue URLs.
(() => {
  const config = window.gradio_config;
  if (!config?.root) return;
  let root;
  try { root = new URL(config.root, location.href); }
  catch { return; }
  if (root.host !== location.host || root.protocol !== 'http:' || location.protocol !== 'https:') return;
  root.protocol = 'https:';
  config.root = root.href.replace(/\/$/, '');
})();
