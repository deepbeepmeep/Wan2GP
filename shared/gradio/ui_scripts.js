function() {
    console.log("[WanGP] main JS initialized");
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
    }

    const hit = n => n?.id === "img_editor" || n?.classList?.contains("wheel-pass");
    addEventListener("wheel", e => {
        const path = e.composedPath?.() || (() => { let a=[],n=e.target; for(;n;n=n.parentNode||n.host) a.push(n); return a; })();
        if (path.some(hit)) e.stopImmediatePropagation();
    }, { capture: true, passive: true });
