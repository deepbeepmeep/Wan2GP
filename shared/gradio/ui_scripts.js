/* Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt) */ function() {
    console.log("[WanGP] main JS initialized");
    const settingsSeparatorRoots = new Set();
    let settingsSeparatorFrame = 0;
    function scheduleSettingsSeparators(root) {
        settingsSeparatorRoots.add(root);
        if (settingsSeparatorFrame) return;
        settingsSeparatorFrame = requestAnimationFrame(() => {
            settingsSeparatorFrame = 0;
            for (const root of settingsSeparatorRoots) {
                for (const row of root.querySelectorAll('[data-testid="dropdown-option"]')) {
                    const label = row.getAttribute('aria-label');
                    const match = label.match(/^─+(.+?)─+$/);
                    let caption = row.querySelector('.wangp-settings-separator-title');
                    if (match) {
                        row.closest('.block').classList.add('wangp-settings-dropdown');
                        if (!caption) {
                            // Keep Svelte's text node alive when filtering reuses an option.
                            const text = [...row.childNodes].find(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim() === label);
                            caption = document.createElement('span');
                            caption.className = 'wangp-settings-separator-title';
                            text.before(caption);
                            caption.append(text);
                        }
                        caption.firstChild.textContent = match[1].trim();
                    } else if (caption) {
                        caption.firstChild.textContent = label;
                        caption.replaceWith(...caption.childNodes);
                    }
                }
            }
            settingsSeparatorRoots.clear();
        });
    }
    // One-shot mount events also cover lists replaced by a server-side update.
    document.addEventListener('animationstart', event => {
        if (event.animationName === 'wangp-settings-separator-ready') scheduleSettingsSeparators(event.target.parentElement);
    });
    for (const type of ['focusin', 'pointerdown', 'input', 'keydown']) {
        document.addEventListener(type, event => {
            const dropdown = event.target.closest('.block');
            if (dropdown?.querySelector('input[role="listbox"]')) scheduleSettingsSeparators(dropdown);
        }, {passive: true});
    }
    window.wangpRefreshSettingsTools = function() {
        for (const row of document.querySelectorAll('.wangp-settings-file-actions')) {
            const reset = row.querySelector('.wangp-reset-settings');
            const exportButton = row.querySelector('.wangp-settings-export');
            // Keep keyboard focus order aligned with the displayed actions.
            if (reset.nextElementSibling !== exportButton) exportButton.before(reset);
            row.classList.toggle('is-compact', row.clientWidth < 580);
            for (const button of row.querySelectorAll('.wangp-settings-default, .wangp-settings-export')) {
                if (!button.hasAttribute('aria-label')) button.setAttribute('aria-label', button.textContent.trim());
            }
        }
        for (const button of document.querySelectorAll('.wangp-lora-download-row > button')) {
            button.setAttribute('aria-label', 'Download');
        }
    };
    window.wangpRefreshSettingsTools();
    let inputModeFrame = 0;
    const inputModeHelp = {"": "Text Prompt", S: "Start with an Image", V: "Continue an Uploaded Video", L: "Continue the Last Generated Video"};
    const inputModeShortLabels = {"": "Text", S: "Start image", V: "Continue Video", L: "Last Video"};
    function closeInputMenu(dropdown, restoreFocus = false) {
        dropdown.lastElementChild.hidden = true;
        dropdown.firstElementChild.setAttribute('aria-expanded', 'false');
        if (restoreFocus) dropdown.firstElementChild.focus({preventScroll: true});
    }
    function openInputMenu(dropdown) {
        dropdown.lastElementChild.hidden = false;
        dropdown.firstElementChild.setAttribute('aria-expanded', 'true');
        (dropdown.querySelector('[aria-selected="true"]:not(:disabled)') || dropdown.querySelector('[role="option"]:not(:disabled)')).focus({preventScroll: true});
    }
    window.wangpRefreshInputModes = function() {
        if (inputModeFrame) return;
        inputModeFrame = requestAnimationFrame(() => {
            inputModeFrame = 0;
            window.wangpRefreshSettingsTools();
            for (const radio of document.querySelectorAll('.wangp-input-modes')) {
                if (!radio.getBoundingClientRect().width) continue;
                const inputs = [...radio.querySelectorAll('input[type="radio"]')];
                const segments = inputs[0].closest('.wrap');
                for (const input of inputs) {
                    const label = input.closest('label');
                    // Keep Gradio's caption node intact so model changes can update it.
                    const full = label.querySelector('span');
                    let short = label.querySelector('.wangp-mode-short');
                    if (!short) {
                        full.classList.add('wangp-mode-full');
                        full.setAttribute('aria-hidden', 'true');
                        short = document.createElement('span');
                        short.className = 'wangp-mode-short';
                        short.setAttribute('aria-hidden', 'true');
                        label.append(short);
                    }
                    if (short.textContent !== inputModeShortLabels[input.value]) short.textContent = inputModeShortLabels[input.value];
                    if (input.getAttribute('aria-label') !== full.textContent) input.setAttribute('aria-label', full.textContent);
                }
                let dropdown = radio.querySelector('.wangp-input-dropdown');
                if (!dropdown) {
                    dropdown = document.createElement('div');
                    dropdown.className = 'wangp-input-dropdown';
                    const trigger = document.createElement('button');
                    trigger.type = 'button';
                    trigger.className = 'wangp-input-trigger';
                    trigger.setAttribute('aria-haspopup', 'listbox');
                    trigger.setAttribute('aria-expanded', 'false');
                    const menu = document.createElement('div');
                    menu.className = 'wangp-input-menu';
                    menu.id = radio.id + '-input-menu';
                    menu.setAttribute('role', 'listbox');
                    menu.setAttribute('aria-label', 'Video Input Mode');
                    menu.hidden = true;
                    trigger.setAttribute('aria-controls', menu.id);
                    dropdown.append(trigger, menu);
                    radio.append(dropdown);
                }
                const trigger = dropdown.firstElementChild;
                const menu = dropdown.lastElementChild;
                const choices = inputs.map(input => [input.value, input.closest('label').querySelector('.wangp-mode-full').textContent, input.disabled]);
                const signature = JSON.stringify(choices);
                if (dropdown.dataset.choices !== signature) {
                    menu.replaceChildren(...choices.map(([value, text, disabled]) => {
                        const option = document.createElement('button');
                        option.type = 'button';
                        option.setAttribute('role', 'option');
                        option.tabIndex = -1;
                        option.dataset.value = value;
                        option.textContent = text;
                        option.disabled = disabled;
                        return option;
                    }));
                    dropdown.dataset.choices = signature;
                }
                for (const input of inputs) {
                    const label = input.closest('label');
                    if (label.dataset.wangpInputTooltip !== inputModeHelp[input.value]) label.dataset.wangpInputTooltip = inputModeHelp[input.value];
                }
                const selected = inputs.find(input => input.checked);
                const value = selected ? selected.value : '';
                const caption = choices.find(choice => choice[0] === value)[1];
                if (trigger.textContent !== caption) trigger.textContent = caption;
                trigger.setAttribute('aria-label', 'Video Input Mode: ' + caption);
                trigger.disabled = inputs.every(input => input.disabled);
                if (trigger.dataset.wangpInputTooltip !== inputModeHelp[value]) trigger.dataset.wangpInputTooltip = inputModeHelp[value];
                for (const option of menu.children) {
                    const checked = String(option.dataset.value === value);
                    if (option.getAttribute('aria-selected') !== checked) option.setAttribute('aria-selected', checked);
                }
                // Measure text and padding, excluding the extra width shared by expanded buttons.
                // These sizes remain available while the segments are hidden.
                const segmentStyle = getComputedStyle(segments);
                let requiredWidth = parseFloat(segmentStyle.borderLeftWidth) + parseFloat(segmentStyle.borderRightWidth);
                const labelSizes = inputs.map(input => {
                    const label = input.closest('label');
                    const style = getComputedStyle(label);
                    const fullWidth = label.querySelector('.wangp-mode-full').getBoundingClientRect().width;
                    requiredWidth += fullWidth + parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
                    return {label, savedWidth: fullWidth - label.querySelector('.wangp-mode-short').getBoundingClientRect().width};
                });
                for (const {label, savedWidth} of labelSizes.reverse()) {
                    const shorten = savedWidth > 0 && requiredWidth > radio.clientWidth;
                    label.classList.toggle('is-short', shorten);
                    if (shorten) requiredWidth -= savedWidth;
                }
                const compact = requiredWidth > radio.clientWidth;
                if (radio.classList.contains('is-compact') !== compact) {
                    const focused = document.activeElement;
                    radio.classList.toggle('is-compact', compact);
                    closeInputMenu(dropdown);
                    if (compact && segments.contains(focused)) trigger.focus({preventScroll: true});
                    else if (!compact && dropdown.contains(focused) && selected) selected.focus({preventScroll: true});
                }
            }
        });
    };
    document.addEventListener('change', event => {
        if (event.target.closest('.wangp-input-mode-row')) window.wangpRefreshInputModes();
    });
    document.addEventListener('click', event => {
        const dropdown = event.target.closest('.wangp-input-dropdown');
        for (const other of document.querySelectorAll('.wangp-input-dropdown:has([aria-expanded="true"])')) {
            if (other !== dropdown) closeInputMenu(other);
        }
        if (dropdown) {
            const option = event.target.closest('[role="option"]');
            if (option) {
                const radio = dropdown.closest('.wangp-input-modes');
                [...radio.querySelectorAll('input[type="radio"]')].find(input => input.value === option.dataset.value).click();
                closeInputMenu(dropdown, true);
            } else if (event.target.closest('.wangp-input-trigger')) {
                if (dropdown.lastElementChild.hidden) openInputMenu(dropdown);
                else closeInputMenu(dropdown);
            }
        }
        if (event.target.closest('[role="tab"]')) window.wangpRefreshInputModes();
    });
    document.addEventListener('keydown', event => {
        const dropdown = event.target.closest('.wangp-input-dropdown');
        if (!dropdown) return;
        if (event.key === 'Escape') {
            closeInputMenu(dropdown, true);
            event.preventDefault();
        } else if (event.key === 'Tab') {
            closeInputMenu(dropdown, true);
        } else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            if (dropdown.lastElementChild.hidden) openInputMenu(dropdown);
            else {
                const options = [...dropdown.querySelectorAll('[role="option"]:not(:disabled)')];
                const index = options.indexOf(document.activeElement);
                const next = event.key === 'Home' ? 0 : event.key === 'End' ? options.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length;
                options[next].focus({preventScroll: true});
            }
        }
    });
    window.addEventListener('resize', window.wangpRefreshInputModes, {passive: true});
    document.fonts.addEventListener('loadingdone', window.wangpRefreshInputModes);
    window.wangpRefreshInputModes();
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
    const observedActionNodes = new Set();
    function scheduleActionAppearance() {
        if (!actionFrame) actionFrame = requestAnimationFrame(updateActionAppearance);
    }
    function updateActionAppearance() {
        actionFrame = 0;
        const bars = document.querySelectorAll('.wangp-settings-actions');
        const nodes = new Set([...bars].flatMap(bar => [bar, bar.parentElement]));
        for (const node of observedActionNodes) {
            if (!nodes.has(node)) {
                actionResize.unobserve(node);
                observedActionNodes.delete(node);
            }
        }
        for (const node of nodes) {
            if (!observedActionNodes.has(node)) {
                actionResize.observe(node);
                observedActionNodes.add(node);
            }
        }
        for (const bar of bars) {
            const rect = bar.getBoundingClientRect();
            if (!rect.width || !rect.height) {
                bar.classList.toggle('is-floating', false);
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
    }
    document.addEventListener('scroll', event => {
        for (const node of observedActionNodes) {
            if (event.target.contains(node)) {
                scheduleActionAppearance();
                break;
            }
        }
    }, {capture: true, passive: true});
    window.addEventListener('resize', scheduleActionAppearance, {passive: true});
    // Form rebuilds register their new bars; resize handles visibility and geometry.
    window.wangpRefreshActionBars = scheduleActionAppearance;
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
