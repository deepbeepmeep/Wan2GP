import {I as WangpFrameIconButton} from "./IconButton-DSs-aBrk.js";

// Reuse Gradio's actual icon buttons, including their theme and hover styles.
function wangpGallerySave(video, state) {
    const preview = video.closest(".preview");
    const menu = document.createElement("div");
    menu.className = "icon-button-wrapper svelte-9lsba8 wangp-video-actions no-border";
    menu.setAttribute("role", "group");
    menu.setAttribute("aria-label", "Frame Actions");
    const controls = [];
    for (const [action, label, title, drawing] of [
        ["add", "Add Frame to Image Gallery", "Add Current Frame to Image Gallery", '<path d="M14 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v9M3 17l6-6 5 5M15 19h8m-4-4v8"/><circle cx="15" cy="8" r="1.5"/>'],
        ["save", "Save Image", "Save Current Frame as PNG - Original Resolution", '<path d="M12 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v7M3 17l6-6 4 4M18 14v8m-3-3 3 3 3-3"/><circle cx="15" cy="8" r="1.5"/>']
    ]) {
        const control = new WangpFrameIconButton({target: menu, props: {label}});
        controls.push(control);
        const button = menu.lastElementChild;
        button.dataset.action = action;
        button.title = title;
        button.querySelector('div').innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${drawing}</svg>`;
    }
    const status = document.createElement('span');
    status.setAttribute('role', 'status');
    menu.appendChild(status);
    const src = video.src;
    const filename = decodeURIComponent(new URL(src).pathname).split(/[\\/]/).pop();
    const stem = filename.replace(/\.[^.]+$/, "");
    let disposed = false;
    // Anchor to the video controls, not the taller gallery/thumbnail area.
    // Gradio fullscreen resizes the video independently of the preview.
    const positionActions = () => {
        const bounds = video.getBoundingClientRect();
        const parent = menu.offsetParent.getBoundingClientRect();
        menu.style.left = `${bounds.right - parent.left - 24}px`;
        menu.style.top = `${bounds.bottom - parent.top - 48}px`;
    };
    const resizeObserver = new ResizeObserver(positionActions);
    state.saveCleanup = () => {
        disposed = true;
        resizeObserver.disconnect();
        document.removeEventListener('fullscreenchange', positionActions);
        controls.forEach(control => control.$destroy());
        menu.remove();
    };
    menu.addEventListener("keydown", event => {
        event.stopPropagation();
    });
    menu.addEventListener("click", async event => {
        event.stopPropagation();
        const button = event.target.closest("button[data-action]");
        if (!button) return;
        status.textContent = "";
        const canvas = document.createElement("canvas");
        try {
            if (video.readyState < 2 || video.seeking || video.currentSrc !== src) {
                throw new Error("Wait for the Video Frame to Load, Then Try Again.");
            }
            const time = video.currentTime;
            if (button.dataset.action === "add") {
                const request = document.querySelector('#wangp-gallery-add-frame textarea');
                const view = document.querySelector('#wangp-gallery-view textarea');
                const thumbnails = [...video.closest('.gallery-container').querySelectorAll('.thumbnail-small')];
                const index = thumbnails.findIndex(item => item.classList.contains('selected'));
                if (!request || !view?.value || index < 0) throw new Error("Open a Video in the Output Gallery First.");
                const gallery = JSON.parse(view.value);
                const payload = {workspace: gallery.workspace, path: gallery.video[index], time, id: Date.now()};
                Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(request, JSON.stringify(payload));
                request.dispatchEvent(new Event('input', {bubbles: true}));
                return;
            }
            button.disabled = true;
            status.textContent = "Saving Frame…";
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext("2d").drawImage(video, 0, 0);
            const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
            if (disposed) return;
            if (!blob) throw new Error("Could Not Encode Frame.");
            const url = URL.createObjectURL(blob);
            const download = document.createElement("a");
            download.href = url;
            download.download = `${stem}_frame_${time.toFixed(3)}s.png`;
            document.body.appendChild(download);
            download.click();
            download.remove();
            setTimeout(() => URL.revokeObjectURL(url), 60000);
            status.textContent = "";
        } catch (error) {
            if (!disposed) status.textContent = error.message;
        } finally {
            canvas.width = canvas.height = 0;
            button.disabled = false;
        }
    });
    // Reference galleries still offer downloads, but cannot append output media.
    menu.querySelector('[data-action="add"]').hidden = !video.closest('#gallery');
    preview.appendChild(menu);
    positionActions();
    resizeObserver.observe(video);
    resizeObserver.observe(preview);
    document.addEventListener('fullscreenchange', positionActions);
}
