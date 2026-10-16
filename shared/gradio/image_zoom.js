/* Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt) */
// Called by the image component's mount/update/destroy hooks and modal load.
// Only the displayed <img> is transformed; no canvas, copied pixels or cache.
window.WanGPImageZoom ||= (() => {
    const viewers = new WeakMap();
    const original = src => {
        const url = new URL(src, document.baseURI);
        url.searchParams.delete('__wangp_gallery_preview');
        return url.href;
    };
    function mount(image, options = {}) {
        const viewport = options.viewport || image.closest('.media-button, .modal-image-viewport');
        if (!viewport || !options.viewport && !image.matches('[data-testid="detailed-image"], .modal-image')) return;
        // Svelte reapplies the image's class attribute on source/fullscreen updates.
        image.classList.add('wangp-zoom-image');
        const known = viewers.get(image);
        if (known) {
            if ((options.source || original(image.src)) !== known.source) {
                known.source = options.source || original(image.src);
                known.preview = image.src;
                known.reset();
            } else if (!known.fullscreen && known.scale === 1) {
                known.preview = image.src;
            }
            return;
        }
        const host = viewport.closest('.preview, .modal-content-wrapper') || viewport;
        const fullscreenRoot = viewport.closest('.gallery-container') || viewport;
        const toolbar = host.querySelector(':scope > .icon-button-wrapper, :scope > .wv-preview-actions');
        const template = toolbar?.querySelector('button');
        const controls = document.createElement('div');
        // Reuse the neighboring actions' theme, spacing and icon dimensions.
        controls.className = toolbar?.className || '';
        controls.classList.remove('top-panel', 'hide-top-corner', 'display-top-corner');
        controls.classList.add('wangp-image-zoom-controls');
        const state = {source: options.source || original(image.src), preview: image.src, fullscreenRoot, fullscreen: fullscreenRoot === document.fullscreenElement, scale: 1, x: 0, y: 0, pointer: null, suppressClick: false, maxX: 0, maxY: 0};
        viewers.set(image, state);
        const buttons = [ ['Zoom In', 'M12 5v14M5 12h14', 1.5], ['Zoom Out', 'M5 12h14', 1 / 1.5] ].map(([label, path, factor]) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.setAttribute('aria-label', label);
            button.className = template?.className || '';
            button.style.cssText = template?.style.cssText || '';
            const icon = document.createElement('div');
            icon.className = template?.querySelector('div')?.className || '';
            icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="${path}"/></svg>`;
            button.append(icon);
            button.addEventListener('click', event => {
                event.preventDefault();
                event.stopPropagation();
                zoom(factor);
            });
            controls.append(button);
            return button;
        });
        (toolbar || host).append(controls);
        viewport.classList.add('wangp-zoom-viewport');
        function zoom(factor) {
            const scale = Math.max(1, Math.min(16, state.scale * factor));
            state.x *= scale / state.scale;
            state.y *= scale / state.scale;
            state.scale = scale;
            // Replace this image's source, never decode a second image.
            const src = scale > 1 || state.fullscreen ? state.source : state.preview;
            if (image.src !== src) image.src = src;
            layout();
        }
        function wheel(event) {
            if ((!state.fullscreen && !viewport.matches('#media-viewer, .modal-image-viewport')) || !event.deltaY || event.ctrlKey) return;
            event.preventDefault();
            event.stopPropagation();
            if (state.pointer !== null) return;
            const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientHeight : 1);
            zoom(Math.exp(-Math.max(-100, Math.min(100, delta)) * Math.log(1.5) / 100));
        }
        function paint() {
            state.x = Math.max(-state.maxX, Math.min(state.maxX, state.x));
            state.y = Math.max(-state.maxY, Math.min(state.maxY, state.y));
            image.style.transform = state.scale === 1 ? '' : `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
        }
        function layout() {
            const fit = Math.min(image.clientWidth / image.naturalWidth, image.clientHeight / image.naturalHeight);
            state.maxX = Math.max(0, (image.naturalWidth * fit * state.scale - viewport.clientWidth) / 2) || 0;
            state.maxY = Math.max(0, (image.naturalHeight * fit * state.scale - viewport.clientHeight) / 2) || 0;
            viewport.classList.toggle('wangp-zoomed', state.scale > 1);
            buttons[0].disabled = state.scale === 16;
            buttons[1].disabled = state.scale === 1;
            buttons.forEach(button => button.title = `${button.getAttribute('aria-label')} (${Math.round(state.scale * 100)}%)`);
            paint();
        }
        function end(event) {
            if (state.pointer === null) return;
            // A held press must not become a fullscreen click on release.
            if (event) state.suppressClick ||= event.type === 'pointercancel' || event.timeStamp - state.downTime >= 300;
            const pointer = state.pointer;
            state.pointer = null;
            if (image.hasPointerCapture(pointer)) image.releasePointerCapture(pointer);
            viewport.classList.remove('wangp-panning');
        }
        function down(event) {
            if (event.button !== 0 || event.target !== image || state.pointer !== null) return;
            event.preventDefault();
            state.suppressClick = false;
            state.pointer = event.pointerId;
            state.downTime = event.timeStamp;
            state.downX = event.clientX;
            state.downY = event.clientY;
            state.startX = event.clientX - state.x;
            state.startY = event.clientY - state.y;
            image.setPointerCapture(event.pointerId);
        }
        function move(event) {
            if (event.pointerId !== state.pointer) return;
            if (!state.suppressClick && Math.hypot(event.clientX - state.downX, event.clientY - state.downY) < 4) return;
            state.suppressClick = true;
            viewport.classList.toggle('wangp-panning', state.scale > 1);
            state.x = event.clientX - state.startX;
            state.y = event.clientY - state.startY;
            paint();
        }
        function click(event) {
            if (event.target !== image && event.target !== viewport) return;
            if (!state.suppressClick) return;
            state.suppressClick = false;
            event.preventDefault();
            event.stopImmediatePropagation();
        }
        function drag(event) { event.preventDefault(); }
        viewport.addEventListener('pointerdown', down);
        viewport.addEventListener('pointermove', move);
        viewport.addEventListener('pointerup', end);
        viewport.addEventListener('pointercancel', end);
        viewport.addEventListener('lostpointercapture', end);
        viewport.addEventListener('click', click, true);
        viewport.addEventListener('wheel', wheel, {passive: false});
        image.addEventListener('dragstart', drag);
        image.addEventListener('load', layout);
        state.layout = layout;
        state.reset = () => {
            end();
            state.scale = 1;
            state.x = state.y = 0;
            state.suppressClick = false;
            layout();
        };
        state.destroy = () => {
            end();
            viewport.removeEventListener('pointerdown', down);
            viewport.removeEventListener('pointermove', move);
            viewport.removeEventListener('pointerup', end);
            viewport.removeEventListener('pointercancel', end);
            viewport.removeEventListener('lostpointercapture', end);
            viewport.removeEventListener('click', click, true);
            viewport.removeEventListener('wheel', wheel);
            image.removeEventListener('dragstart', drag);
            image.removeEventListener('load', layout);
            image.style.transform = '';
            image.classList.remove('wangp-zoom-image');
            viewport.classList.remove('wangp-zoom-viewport', 'wangp-zoomed');
            controls.remove();
            viewers.delete(image);
        };
        layout();
    }
    // Query only mounted viewers on real viewport changes; nothing retains
    // removed image nodes, and there is no work scheduled while idle.
    window.addEventListener('resize', () => {
        document.querySelectorAll('.wangp-zoom-image').forEach(image => viewers.get(image).layout());
    });
    document.addEventListener('fullscreenchange', () => {
        document.querySelectorAll('.wangp-zoom-image').forEach(image => {
            const state = viewers.get(image);
            const fullscreen = state.fullscreenRoot === document.fullscreenElement;
            if (state.fullscreen === fullscreen) return;
            state.fullscreen = fullscreen;
            state.reset();
            const src = fullscreen ? state.source : state.preview;
            if (image.src !== src) image.src = src;
        });
    });
    return {mount, destroy(image) { viewers.get(image)?.destroy(); }};
})();
