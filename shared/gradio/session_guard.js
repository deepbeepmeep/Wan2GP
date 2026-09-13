(() => {
    const config = window.gradio_config;
    const api = new URL(config.root + config.api_prefix + '/', location.href);
    const signature = __WANGP_UI_SIGNATURE__;
    const fetch = window.fetch.bind(window);
    window.fetch = async (input, options) => {
        const url = new URL(input instanceof Request ? input.url : input, location.href);
        if (url.origin !== api.origin || !url.pathname.startsWith(api.pathname)) return fetch(input, options);
        if (window.__wangpGradioStale) throw new DOMException('Reload this page after the interface change.', 'AbortError');
        const headers = new Headers(options?.headers || (input instanceof Request ? input.headers : undefined));
        // Keep the header name so older pages are also rejected safely once.
        headers.set('X-WanGP-App-Id', signature);
        const response = await fetch(input, {...options, headers});
        if (response.headers.get('X-WanGP-Reload') === '1' && !window.__wangpGradioStale) {
            window.__wangpGradioStale = true;
            const notice = document.createElement('div');
            notice.id = 'wangp-server-restarted';
            notice.setAttribute('role', 'alert');
            notice.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:10001;padding:12px 18px;background:#fff3df;color:#553600;border:1px solid #bb8534;border-radius:10px;max-width:90vw';
            notice.textContent = 'The interface changed. Reload this page before using its controls. Your unsaved text remains here until you reload. ';
            const reload = document.createElement('button');
            reload.type = 'button';
            reload.textContent = 'Reload page';
            reload.onclick = () => location.reload();
            notice.append(reload);
            document.body.append(notice);
        }
        return response;
    };
})();
