function() {
    console.log("[WanGP] main JS initialized");
    let generationLayoutFrame = 0;
    function scheduleGenerationReferences() {
        if (generationLayoutFrame) return;
        generationLayoutFrame = requestAnimationFrame(() => {
            generationLayoutFrame = 0;
            const grids = [...document.querySelectorAll('.wangp-generation-info .generation-references')];
            // Always measure the one-column layout; the two-column prompt wraps
            // differently and must not feed back into the next decision.
            grids.forEach(grid => grid.classList.remove('generation-references-two-columns'));
            const twoColumns = grids.map(grid => {
                if (grid.childElementCount < 2 || !grid.getBoundingClientRect().width) return false;
                const prompt = grid.closest('tr').querySelector('.generation-prompt-cell');
                const first = prompt.firstElementChild, last = prompt.lastElementChild;
                const promptHeight = first ? last.getBoundingClientRect().bottom - first.getBoundingClientRect().top : 0;
                return grid.getBoundingClientRect().height > promptHeight;
            });
            grids.forEach((grid, index) => grid.classList.toggle('generation-references-two-columns', twoColumns[index]));
        });
    }
    document.addEventListener('animationstart', event => {
        if (event.animationName === 'wangp-generation-references-ready') scheduleGenerationReferences();
    });
    window.addEventListener('resize', scheduleGenerationReferences, {passive: true});
    document.fonts.addEventListener('loadingdone', scheduleGenerationReferences);
    scheduleGenerationReferences();
    // CSS positions the bars; this only distinguishes floating from landed.
    let actionFrame = 0;
    const actionResize = new ResizeObserver(scheduleActionAppearance);
    const observedActions = new Set();
    function scheduleActionAppearance() {
        if (!actionFrame) actionFrame = requestAnimationFrame(updateActionAppearance);
    }
    function updateActionAppearance() {
        actionFrame = 0;
        const bars = document.querySelectorAll('.wangp-settings-actions');
        for (const bar of bars) {
            if (!observedActions.has(bar)) {
                observedActions.add(bar);
                actionResize.observe(bar);
                actionResize.observe(bar.parentElement);
            }
            const rect = bar.getBoundingClientRect();
            if (!rect.width || !rect.height) {
                bar.classList.remove('is-floating');
                continue;
            }
            // The previous visible sibling stays in normal flow while the bar
            // sticks, so it marks the bar's landing position without a spacer.
            let previous = bar.previousElementSibling;
            while (previous && !previous.getBoundingClientRect().height) previous = previous.previousElementSibling;
            const parentStyle = getComputedStyle(bar.parentElement);
            const naturalTop = previous
                ? previous.getBoundingClientRect().bottom + (parseFloat(parentStyle.rowGap) || 0)
                : bar.parentElement.getBoundingClientRect().top + (parseFloat(parentStyle.paddingTop) || 0);
            bar.classList.toggle('is-floating', rect.top < naturalTop - 1);
        }
        for (const bar of observedActions) {
            if (!bar.isConnected) {
                actionResize.unobserve(bar);
                observedActions.delete(bar);
            }
        }
    }
    document.addEventListener('scroll', scheduleActionAppearance, {capture: true, passive: true});
    window.addEventListener('resize', scheduleActionAppearance, {passive: true});
    new MutationObserver(scheduleActionAppearance).observe(document.querySelector('gradio-app'), {
        childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'hidden']
    });
    scheduleActionAppearance();
    document.addEventListener('click', (event) => {
        const closeButton = event.target.closest('button[aria-label="Close"]');
        if (!closeButton || !closeButton.closest('.amg-remove-selected')) return;
        const preview = closeButton.closest('.preview');
        if (!preview || !preview.parentElement?.classList.contains('gallery-container')) return;
        const gallery = closeButton.closest('.amg-remove-selected');
        const removeButton = gallery.querySelector('.amg-remove-button button, button.amg-remove-button');
        if (!removeButton) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        removeButton.click();
    }, true);
    window.updateAndTrigger = function(action) {
        const hiddenTextbox = document.querySelector('#queue_action_input textarea');
        const hiddenButton = document.querySelector('#queue_action_trigger');
        if (hiddenTextbox && hiddenButton) {
            hiddenTextbox.value = action;
            hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
            hiddenButton.click();
        } else {
            console.error("Could not find hidden queue action elements.");
        }
    };

    window.scrollToQueueTop = function() {
        const container = document.querySelector('#queue-scroll-container');
        if (container) container.scrollTop = 0;
    };
    window.scrollToQueueBottom = function() {
        const container = document.querySelector('#queue-scroll-container');
        if (container) container.scrollTop = container.scrollHeight;
    };

    window.showImageModal = function(action) {
        const hiddenTextbox = document.querySelector('#modal_action_input textarea');
        const hiddenButton = document.querySelector('#modal_action_trigger');
        if (hiddenTextbox && hiddenButton) {
            hiddenTextbox.value = action;
            hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
            hiddenButton.click();
        }
    };
    window.closeImageModal = function() {
        const closeButton = document.querySelector('#modal_close_trigger_btn');
        if (closeButton) closeButton.click();
    };

    window.WanGPDownloads = {
        trigger(payload) {
            if (!payload) {
                console.log("[WanGP] No download payload received.");
                return "";
            }
            let data = payload;
            if (typeof payload === "string") {
                try {
                    data = JSON.parse(payload);
                } catch (e) {
                    console.error("[WanGP] Invalid download payload:", e);
                    return "";
                }
            }
            if (!data?.url) {
                console.log("[WanGP] Download payload does not contain a URL.");
                return "";
            }
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = data.url;
            a.download = data.filename || "";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            return "";
        }
    };

    let lastSelectedVideoTime = null;
    let selectedVideoTimeTimer = null;

    function pushSelectedVideoTime(currentTime) {
        const hiddenTextbox = document.querySelector('#selected_video_time_input textarea, #selected_video_time_input input');
        if (!hiddenTextbox) return;
        hiddenTextbox.value = String(currentTime);
        hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
        hiddenTextbox.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function scheduleSelectedVideoTimeUpdate(video, immediate = false) {
        if (!(video instanceof HTMLVideoElement) || !video.closest('#gallery')) return;
        const currentTime = Math.max(0, Math.round(Number(video.currentTime || 0) * 1000) / 1000);
        if (!Number.isFinite(currentTime)) return;
        const commit = () => {
            if (lastSelectedVideoTime === currentTime) return;
            lastSelectedVideoTime = currentTime;
            pushSelectedVideoTime(currentTime);
        };
        if (immediate) {
            clearTimeout(selectedVideoTimeTimer);
            commit();
            return;
        }
        clearTimeout(selectedVideoTimeTimer);
        selectedVideoTimeTimer = setTimeout(commit, 180);
    }

    function handleGalleryVideoEvent(event, immediate = false) {
        const video = event?.target;
        if (!(video instanceof HTMLVideoElement) || !video.closest('#gallery')) return;
        scheduleSelectedVideoTimeUpdate(video, immediate);
    }

    let draggedItem = null;

    function attachDelegatedDragAndDrop(container) {
        if (container.dataset.dndDelegated) return; // Listeners already attached
        container.dataset.dndDelegated = 'true';

        container.addEventListener('dragstart', (e) => {
            const row = e.target.closest('.draggable-row');
            if (!row || e.target.closest('.action-button') || e.target.closest('.hover-image')) {
                if (row) e.preventDefault(); // Prevent dragging if it's on a button/image
                return;
            }
            draggedItem = row;
            setTimeout(() => draggedItem.classList.add('dragging'), 0);
        });

        container.addEventListener('dragend', () => {
            if (draggedItem) {
                draggedItem.classList.remove('dragging');
            }
            draggedItem = null;
            document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(el => {
                el.classList.remove('drag-over-top', 'drag-over-bottom');
            });
        });

        container.addEventListener('dragover', (e) => {
            e.preventDefault();
            const targetRow = e.target.closest('.draggable-row');

            document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(el => {
                el.classList.remove('drag-over-top', 'drag-over-bottom');
            });

            if (targetRow && draggedItem && targetRow !== draggedItem) {
                const rect = targetRow.getBoundingClientRect();
                const midpoint = rect.top + rect.height / 2;

                if (e.clientY < midpoint) {
                    targetRow.classList.add('drag-over-top');
                } else {
                    targetRow.classList.add('drag-over-bottom');
                }
            }
        });

        container.addEventListener('drop', (e) => {
            e.preventDefault();
            const targetRow = e.target.closest('.draggable-row');
            if (!draggedItem || !targetRow || targetRow === draggedItem) return;

            const oldIndex = draggedItem.dataset.index;
            let newIndex = parseInt(targetRow.dataset.index);

            if (targetRow.classList.contains('drag-over-bottom')) {
                newIndex++;
            }

            if (oldIndex != newIndex) {
               const action = `move_${oldIndex}_to_${newIndex}`;
               window.updateAndTrigger(action);
            }
        });
    }

    const observer = new MutationObserver((mutations, obs) => {
        const container = document.querySelector('#queue_html_container');
        if (container) {
            attachDelegatedDragAndDrop(container);
            obs.disconnect();
        }
    });

    const targetNode = document.querySelector('gradio-app');
    if (targetNode) {
        observer.observe(targetNode, { childList: true, subtree: true });
        ['loadedmetadata', 'seeked', 'pause', 'click'].forEach((eventName) => {
            document.addEventListener(eventName, (event) => handleGalleryVideoEvent(event, true), true);
        });
        document.addEventListener('timeupdate', (event) => handleGalleryVideoEvent(event, false), true);
    }

    const hit = n => n?.id === "img_editor" || n?.classList?.contains("wheel-pass");
    addEventListener("wheel", e => {
        const path = e.composedPath?.() || (() => { let a=[],n=e.target; for(;n;n=n.parentNode||n.host) a.push(n); return a; })();
        if (path.some(hit)) e.stopImmediatePropagation();
    }, { capture: true, passive: true });
