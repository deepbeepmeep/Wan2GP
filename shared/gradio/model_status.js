// Render-only: retain Gradio's original labels, model IDs and event handling.
function wangpModelLabel(node, text, input = false) {
    if (!node.closest('#model_base_types_list, #model_list')) return text;
    const match = /^([⬛🟦🟨])\s/u.exec(text || '');
    const host = input ? node.parentElement : node;
    if (!match) {
        delete host.dataset.wangpAvailability;
        node.removeAttribute('aria-description');
        return text;
    }
    const status = {'⬛': 'missing', '🟦': 'available', '🟨': 'partial'}[match[1]];
    host.dataset.wangpAvailability = status;
    node.setAttribute('aria-description', {missing: 'Not installed', available: 'Available', partial: 'Partially available'}[status]);
    return text.slice(match[0].length);
}

function wangpModelInput(node, text) {
    const display = wangpModelLabel(node, text, true);
    if (node.value !== (display ?? '')) me(node, display);
}
