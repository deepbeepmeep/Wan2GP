<svelte:options accessors={true} />

<script>
	import { onDestroy, onMount, tick } from "svelte";
	import TreeNodes from "./TreeNodes.svelte";

	export let elem_id = "";
	export let elem_classes = [];
	export let visible = true;
	export let value = [];
	export let hierarchy = { folders: [], items: [] };
	export let height = 10;
	export let label = "Hierarchy Selector";
	export let info = undefined;
	export let show_label = true;
	export let container = true;
	export let scale = null;
	export let min_width = undefined;
	export let interactive = true;
	export let gradio;

	let rootEl;
	let inputEl;
	let panelEl;
	let open = false;
	let focused = false;
	let expanded = new Set();
	let draggedIndex = null;
	let dragOverIndex = null;
	let itemLabels = {};
	let panelStyle = "";
	const rowHeight = 32;
	const panelPadding = 8;
	const panelGap = 6;
	const viewportPadding = 8;

	$: selectedValue = normalizeValue(value);
	$: normalizedHierarchy = normalizeHierarchy(hierarchy);
	$: itemLabels = collectLabels(normalizedHierarchy);
	$: panelRows = Math.max(10, Number(height) || 10);
	$: panelHeight = panelRows * rowHeight + panelPadding;
	$: classes = [
		"hierarchy-selector",
		container ? "hierarchy-selector-container" : "",
		focused ? "hierarchy-selector-focused" : "",
		...(Array.isArray(elem_classes) ? elem_classes : [])
	].filter(Boolean).join(" ");
	$: style = [
		scale !== null ? `flex-grow:${scale};` : "",
		min_width !== undefined ? `min-width:${min_width}px;` : ""
	].join("");

	function normalizeValue(next) {
		if (Array.isArray(next)) return next.map((item) => String(item));
		if (next === null || next === undefined || next === "") return [];
		return [String(next)];
	}

	function normalizeHierarchy(next) {
		return { folders: sortFolders(next?.folders || []), items: sortItems(next?.items || []) };
	}

	function sortFolders(folders) {
		return folders
			.map((folder) => ({ ...folder, folders: sortFolders(folder.folders || []), items: sortItems(folder.items || []) }))
			.sort((a, b) => labelForFolder(a).localeCompare(labelForFolder(b), undefined, { sensitivity: "base" }));
	}

	function sortItems(items) {
		return items
			.map((item) => ({ ...item }))
			.sort((a, b) => labelForItem(a).localeCompare(labelForItem(b), undefined, { sensitivity: "base" }));
	}

	function labelForFolder(folder) {
		return String(folder.name || folder.path || "");
	}

	function labelForItem(item) {
		return String(item.name || item.path || item.value || "");
	}

	function selectedLabelForItem(item) {
		return String(item.path || item.name || item.value || "");
	}

	function valueForItem(item) {
		return String(item.value || item.path || item.name || "");
	}

	function collectLabels(root) {
		const labels = {};
		function visit(folder) {
			for (const item of folder.items || []) labels[valueForItem(item)] = selectedLabelForItem(item);
			for (const child of folder.folders || []) visit(child);
		}
		visit(root);
		return labels;
	}

	function displayValue(selected) {
		return itemLabels[selected] || stripSuffix(selected);
	}

	function stripSuffix(selected) {
		return selected.replace(/\\/g, "/").replace(/\.[^/.]+$/, "");
	}

	function dispatch(next, event = "change", data = undefined) {
		value = normalizeValue(next);
		gradio?.dispatch?.("input");
		if (event === "select") gradio?.dispatch?.("select", data);
		gradio?.dispatch?.("change");
		tick().then(updatePanelPosition);
	}

	function toggleItem(item) {
		if (!interactive) return;
		const itemValue = valueForItem(item);
		if (selectedValue.includes(itemValue)) dispatch(selectedValue.filter((selected) => selected !== itemValue), "select", { value: itemValue, selected: false });
		else dispatch([...selectedValue, itemValue], "select", { value: itemValue, selected: true });
	}

	function removeValue(index) {
		if (!interactive) return;
		const next = selectedValue.filter((_, pos) => pos !== index);
		dispatch(next, "select", { value: selectedValue[index], selected: false });
	}

	function toggleFolder(path) {
		if (expanded.has(path)) expanded.delete(path);
		else expanded.add(path);
		expanded = new Set(expanded);
	}

	function updatePanelPosition() {
		if (!inputEl || !open) return;
		const rect = inputEl.getBoundingClientRect();
		const preferredHeight = panelHeight;
		const spaceAbove = rect.top - viewportPadding - panelGap;
		const spaceBelow = window.innerHeight - rect.bottom - viewportPadding - panelGap;
		const placeBelow = spaceAbove < preferredHeight && spaceBelow > spaceAbove;
		const availableHeight = Math.max(rowHeight * 4, placeBelow ? spaceBelow : spaceAbove);
		const actualHeight = Math.min(preferredHeight, availableHeight);
		const top = placeBelow ? rect.bottom + panelGap : Math.max(viewportPadding, rect.top - actualHeight - panelGap);
		const left = Math.max(viewportPadding, rect.left);
		const width = Math.max(240, Math.min(rect.width, window.innerWidth - left - viewportPadding));
		panelStyle = `top:${Math.round(top)}px;left:${Math.round(left)}px;width:${Math.round(width)}px;height:${Math.round(actualHeight)}px;`;
	}

	function openPanel() {
		if (!interactive) return;
		open = true;
		focused = true;
		gradio?.dispatch?.("focus");
		tick().then(() => {
			inputEl?.focus();
			updatePanelPosition();
		});
	}

	function closePanel() {
		if (!open && !focused) return;
		open = false;
		focused = false;
		gradio?.dispatch?.("blur");
	}

	function onDocumentPointerDown(event) {
		if (rootEl?.contains(event.target) || panelEl?.contains(event.target)) return;
		closePanel();
	}

	function onDragStart(index, event) {
		if (!interactive) return;
		draggedIndex = index;
		event.dataTransfer?.setData("text/plain", String(index));
		if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
	}

	function onDragEnd() {
		draggedIndex = null;
		dragOverIndex = null;
	}

	function onDrop(index, event) {
		event.preventDefault();
		if (draggedIndex === null || draggedIndex === index) return;
		const next = [...selectedValue];
		const [item] = next.splice(draggedIndex, 1);
		next.splice(index, 0, item);
		draggedIndex = null;
		dragOverIndex = null;
		dispatch(next);
	}

	function onInputKeydown(event) {
		if (event.key === "Escape") closePanel();
		else if (event.key === "Enter" || event.key === " " || event.key === "ArrowDown") {
			event.preventDefault();
			openPanel();
		}
	}

	function portal(node) {
		document.body.appendChild(node);
		return {
			destroy() {
				node.remove();
			}
		};
	}

	export function get_value() {
		return normalizeValue(value);
	}

	$: if (open && selectedValue) tick().then(updatePanelPosition);

	onMount(() => {
		document.addEventListener("pointerdown", onDocumentPointerDown, true);
		window.addEventListener("resize", updatePanelPosition);
		window.addEventListener("scroll", updatePanelPosition, true);
	});
	onDestroy(() => {
		document.removeEventListener("pointerdown", onDocumentPointerDown, true);
		window.removeEventListener("resize", updatePanelPosition);
		window.removeEventListener("scroll", updatePanelPosition, true);
	});
</script>

{#if visible}
	<div id={elem_id} bind:this={rootEl} class={classes} style={style}>
		<div class="hierarchy-selector-field">
			{#if show_label && label}
				<span class="hierarchy-selector-label">{label}</span>
			{/if}
			<div
				class="hierarchy-selector-input"
				class:hierarchy-selector-disabled={!interactive}
				role="button"
				tabindex={interactive ? 0 : -1}
				aria-haspopup="tree"
				aria-expanded={open}
				bind:this={inputEl}
				on:click={openPanel}
				on:keydown={onInputKeydown}
			>
				<div class="hierarchy-selector-chips">
					{#if selectedValue.length === 0}
						<span class="hierarchy-selector-placeholder">{label}</span>
					{/if}
					{#each selectedValue as selected, index}
						<span
							class="hierarchy-selector-chip"
							class:hierarchy-selector-chip-dragging={draggedIndex === index}
							class:hierarchy-selector-chip-over={dragOverIndex === index}
							role="listitem"
							draggable={interactive}
							on:dragstart={(event) => onDragStart(index, event)}
							on:dragend={onDragEnd}
							on:dragover={(event) => {
								event.preventDefault();
								dragOverIndex = index;
							}}
							on:dragleave={() => dragOverIndex = null}
							on:drop={(event) => onDrop(index, event)}
						>
							<span class="hierarchy-selector-chip-text">{displayValue(selected)}</span>
							<button type="button" class="hierarchy-selector-remove" aria-label="Remove" on:click|stopPropagation={() => removeValue(index)}>x</button>
						</span>
					{/each}
				</div>
			</div>
			{#if open}
				<div bind:this={panelEl} use:portal class="hierarchy-selector-panel" style={panelStyle}>
					<TreeNodes folders={normalizedHierarchy.folders || []} items={normalizedHierarchy.items || []} depth={0} {expanded} value={selectedValue} {toggleItem} {toggleFolder} {valueForItem} {labelForItem} />
				</div>
			{/if}
		</div>
		{#if info}
			<div class="hierarchy-selector-info">{info}</div>
		{/if}
	</div>
{/if}

<style>
	.hierarchy-selector {
		position: relative;
		box-sizing: border-box;
		width: 100%;
		font: inherit;
	}

	.hierarchy-selector-container {
		position: relative;
		border: var(--block-border-width) solid var(--block-border-color);
		border-radius: var(--block-radius);
		background: var(--block-background-fill);
		box-shadow: var(--block-shadow);
		padding: 0;
		line-height: var(--line-sm);
	}

	.hierarchy-selector-field {
		position: relative;
		padding: var(--block-padding);
	}

	.hierarchy-selector-label {
		display: inline-block;
		border: var(--block-title-border-width, 0) solid var(--block-title-border-color, transparent);
		border-radius: var(--block-title-radius);
		background: var(--block-title-background-fill);
		padding: var(--block-title-padding);
		color: var(--block-title-text-color);
		font-family: var(--font);
		font-size: var(--block-title-text-size);
		font-weight: var(--block-title-text-weight);
		line-height: var(--line-sm);
		margin: 0 0 8px 0;
		cursor: default;
	}

	.hierarchy-selector-input {
		display: flex;
		align-items: center;
		min-height: 42px;
		border: var(--input-border-width) solid var(--input-border-color);
		border-radius: var(--input-radius);
		background: var(--input-background-fill);
		box-shadow: var(--input-shadow);
		padding: 5px 6px;
		cursor: text;
		transition: border-color 120ms ease, box-shadow 120ms ease, background-color 120ms ease;
	}

	.hierarchy-selector-focused .hierarchy-selector-input {
		border-color: var(--input-border-color-focus);
		background: var(--input-background-fill-focus);
		box-shadow: var(--input-shadow-focus);
	}

	.hierarchy-selector-disabled {
		cursor: default;
		opacity: 0.6;
	}

	.hierarchy-selector-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 5px;
		width: 100%;
		min-width: 0;
	}

	.hierarchy-selector-placeholder {
		display: inline-flex;
		align-items: center;
		min-height: 26px;
		color: var(--body-text-color-subdued);
		font-size: var(--text-sm);
		line-height: 20px;
		padding: 2px 4px;
	}

	.hierarchy-selector-chip {
		display: inline-flex;
		align-items: center;
		max-width: 100%;
		border: var(--checkbox-border-width, 1px) solid var(--checkbox-label-border-color, var(--border-color-primary));
		border-radius: var(--button-small-radius);
		background: var(--checkbox-label-background-fill);
		color: var(--body-text-color);
		font-size: var(--text-sm);
		line-height: 22px;
		overflow: hidden;
		cursor: grab;
		transition: border-color 120ms ease, background-color 120ms ease, opacity 120ms ease, transform 120ms ease;
	}

	.hierarchy-selector-chip::before {
		content: "";
		flex: 0 0 auto;
		width: 7px;
		height: 17px;
		margin-left: 6px;
		opacity: 0.45;
		background-image: radial-gradient(currentColor 1px, transparent 1px);
		background-size: 3px 4px;
		background-position: center;
	}

	.hierarchy-selector-chip-dragging {
		opacity: 0.55;
		cursor: grabbing;
	}

	.hierarchy-selector-chip-over {
		border-color: var(--color-accent);
		background: var(--input-background-fill);
	}

	.hierarchy-selector-chip-text {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		padding: 2px 7px;
	}

	.hierarchy-selector-remove {
		border: none;
		min-width: 26px;
		align-self: stretch;
		padding: 2px 7px;
		background: transparent;
		color: var(--body-text-color-subdued);
		cursor: pointer;
		font-weight: 600;
	}

	.hierarchy-selector-remove:hover {
		color: var(--body-text-color);
		background: transparent;
	}

	.hierarchy-selector-info {
		margin-top: var(--spacing-sm);
		color: var(--body-text-color-subdued);
		font-size: var(--text-sm);
	}

	.hierarchy-selector-panel {
		position: fixed;
		box-sizing: border-box;
		z-index: 2000;
		border: var(--block-border-width) solid var(--block-border-color);
		border-radius: var(--block-radius);
		background: var(--block-background-fill);
		box-shadow: var(--shadow-drop-lg);
		overflow-x: hidden;
		overflow-y: auto;
		padding: 4px;
		scrollbar-width: thin;
	}
</style>
