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
	export let display_mode = "file";
	export let breadcrumb_separator = " ";
	export let sort_hierarchy = true;
	export let search_empty_label = "No matches";
	export let show_placeholder = false;
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
	let searchInputEl;
	let panelEl;
	let panelContentEl;
	let open = false;
	let focused = false;
	let expanded = new Set();
	let searchQuery = "";
	let draggedIndex = null;
	let dragOverIndex = null;
	let itemLabels = {};
	let panelStyle = "";
	const rowHeight = 28;
	const panelPadding = 8;
	const panelGap = 6;
	const viewportPadding = 8;
	const uniqueId = `hierarchy-selector-${Math.random().toString(36).slice(2)}`;

	$: selectedValue = normalizeValue(value);
	$: normalizedHierarchy = normalizeHierarchy(hierarchy, sort_hierarchy);
	$: breadcrumbMode = display_mode === "breadcrumb";
	$: itemLabels = collectLabels(normalizedHierarchy, breadcrumbMode, breadcrumb_separator);
	$: flatItems = flattenItems(normalizedHierarchy, breadcrumbMode, breadcrumb_separator);
	$: searchTerm = searchQuery.trim().toLowerCase();
	$: searchMode = searchTerm.length > 0;
	$: searchResults = searchMode ? filterItems(flatItems, searchTerm) : [];
	$: panelId = elem_id ? `${elem_id}-panel` : `${uniqueId}-panel`;
	$: heightRows = Number(height);
	$: autoPanelHeight = heightRows === 0;
	$: panelRows = autoPanelHeight ? 0 : Math.max(10, heightRows || 10);
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

	function normalizeHierarchy(next, shouldSort) {
		const root = { folders: next?.folders || [], items: next?.items || [] };
		return shouldSort ? { folders: sortFolders(root.folders), items: sortItems(root.items) } : cloneHierarchy(root);
	}

	function cloneHierarchy(node) {
		return {
			...node,
			folders: (node.folders || []).map((folder) => cloneHierarchy(folder)),
			items: (node.items || []).map((item) => ({ ...item }))
		};
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
		return displayLabelForItem(item);
	}

	function valueForItem(item) {
		return String(item.value || item.path || item.name || "");
	}

	function collectLabels(root, _breadcrumbMode, _breadcrumbSeparator) {
		const labels = {};
		function visit(folder, folderPath = "") {
			for (const item of folder.items || []) labels[valueForItem(item)] = displayLabelForItem(item, folderPath);
			for (const child of folder.folders || []) visit(child, folderPathForFolder(child, folderPath));
		}
		visit(root);
		return labels;
	}

	function folderPathForFolder(folder, parentPath = "") {
		const explicit = String(folder.path || "");
		if (explicit) return explicit;
		const name = labelForFolder(folder);
		return parentPath && name ? `${parentPath}/${name}` : name;
	}

	function pathForItem(item, folderPath = "") {
		const explicit = String(item.path || "");
		if (explicit) return explicit;
		const name = labelForItem(item);
		return folderPath && name ? `${folderPath}/${name}` : name;
	}

	function pathParts(path) {
		return String(path || "").split("/").map((part) => part.trim()).filter(Boolean);
	}

	function formatBreadcrumb(path) {
		const parts = pathParts(path);
		return parts.length ? parts.join(breadcrumb_separator) : String(path || "");
	}

	function displayLabelForItem(item, folderPath = "") {
		const itemPath = pathForItem(item, folderPath) || String(item.value || "");
		return breadcrumbMode ? formatBreadcrumb(itemPath) : itemPath;
	}

	function searchTextForItem(item, folderPath = "") {
		return breadcrumbMode ? displayLabelForItem(item, folderPath) : labelForItem(item);
	}

	function flattenItems(root, _breadcrumbMode, _breadcrumbSeparator) {
		const items = [];
		function visit(folder, folderPath = "") {
			for (const item of folder.items || []) {
				const itemPath = pathForItem(item, folderPath);
				const slash = itemPath.lastIndexOf("/");
				const parentPath = slash > -1 ? itemPath.slice(0, slash) : folderPath;
				items.push({ ...item, search_name: labelForItem(item), search_path: parentPath, search_text: searchTextForItem(item, folderPath), search_display: displayLabelForItem(item, folderPath) });
			}
			for (const child of folder.folders || []) visit(child, folderPathForFolder(child, folderPath));
		}
		visit(root);
		return items;
	}

	function filterItems(items, term) {
		return items
			.map((item) => {
				const name = String(item.search_name || labelForItem(item));
				const path = String(item.search_path || "");
				const text = String(item.search_text || name);
				const display = String(item.search_display || name);
				return { item, index: text.toLowerCase().indexOf(term), name: display, path };
			})
			.filter((match) => match.index > -1)
			.sort((a, b) => a.index - b.index || a.name.localeCompare(b.name, undefined, { sensitivity: "base" }) || a.path.localeCompare(b.path, undefined, { sensitivity: "base" }))
			.map((match) => match.item);
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

	function clearValues() {
		if (!interactive || selectedValue.length === 0) return;
		const previous = selectedValue;
		searchQuery = "";
		dispatch([], "select", { value: previous, selected: false });
	}

	function toggleFolder(path) {
		if (expanded.has(path)) expanded.delete(path);
		else expanded.add(path);
		expanded = new Set(expanded);
		tick().then(updatePanelPosition);
	}

	function updatePanelPosition() {
		if (!inputEl || !open) return;
		const rect = inputEl.getBoundingClientRect();
		const contentHeight = Math.max(rowHeight + panelPadding, panelContentEl ? Math.ceil(panelContentEl.scrollHeight + panelPadding) : rowHeight + panelPadding);
		const preferredHeight = autoPanelHeight ? contentHeight : panelHeight;
		const spaceAbove = rect.top - viewportPadding - panelGap;
		const spaceBelow = window.innerHeight - rect.bottom - viewportPadding - panelGap;
		const placeBelow = spaceAbove < preferredHeight && spaceBelow > spaceAbove;
		const availableHeight = Math.max(rowHeight + panelPadding, placeBelow ? spaceBelow : spaceAbove);
		const actualHeight = Math.min(preferredHeight, availableHeight);
		const top = placeBelow ? rect.bottom + panelGap : Math.max(viewportPadding, rect.top - actualHeight - panelGap);
		const left = Math.max(viewportPadding, rect.left);
		const width = Math.max(240, Math.min(rect.width, window.innerWidth - left - viewportPadding));
		panelStyle = `top:${Math.round(top)}px;left:${Math.round(left)}px;width:${Math.round(width)}px;height:${Math.round(actualHeight)}px;`;
	}

	function openPanel() {
		if (!interactive) return;
		const wasFocused = focused;
		open = true;
		focused = true;
		if (!wasFocused) gradio?.dispatch?.("focus");
		tick().then(() => {
			searchInputEl?.focus();
			updatePanelPosition();
		});
	}

	function closePanel() {
		if (!open && !focused) return;
		open = false;
		focused = false;
		searchQuery = "";
		gradio?.dispatch?.("blur");
	}

	function onSearchFocus() {
		openPanel();
	}

	function onSearchInput(event) {
		searchQuery = event.currentTarget.value;
		openPanel();
	}

	function onInputPointerDown(event) {
		if (!interactive || event.target === searchInputEl || event.target.closest?.("button") || event.target.closest?.(".hierarchy-selector-chip")) return;
		event.preventDefault();
		openPanel();
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
		if (event.key === "Escape") {
			event.preventDefault();
			if (searchQuery) searchQuery = "";
			else closePanel();
		} else if (event.key === "Enter" && searchMode && searchResults.length) {
			event.preventDefault();
			toggleItem(searchResults[0]);
		} else if (event.key === "ArrowDown") {
			event.preventDefault();
			openPanel();
		} else if (event.key === "Backspace" && !searchQuery && selectedValue.length) {
			removeValue(selectedValue.length - 1);
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

	$: if (open && (selectedValue || searchQuery || expanded || normalizedHierarchy || searchResults || autoPanelHeight || panelHeight)) tick().then(updatePanelPosition);

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
				role="combobox"
				tabindex={interactive ? 0 : -1}
				aria-haspopup="tree"
				aria-expanded={open ? "true" : "false"}
				aria-controls={panelId}
				bind:this={inputEl}
				on:mousedown={onInputPointerDown}
				on:keydown={onInputKeydown}
			>
				<div class="hierarchy-selector-chips">
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
					<input
						bind:this={searchInputEl}
						bind:value={searchQuery}
						class="hierarchy-selector-search-input"
						type="text"
						autocomplete="off"
						spellcheck="false"
						disabled={!interactive}
						tabindex={interactive ? 0 : -1}
						placeholder={show_placeholder && selectedValue.length === 0 ? label : ""}
						aria-label={label}
						on:focus={onSearchFocus}
						on:input={onSearchInput}
						on:keydown|stopPropagation={onInputKeydown}
					/>
				</div>
				{#if selectedValue.length > 0}
					<button type="button" class="hierarchy-selector-clear" aria-label="Clear selection" on:click|stopPropagation={clearValues}>x</button>
				{/if}
			</div>
			{#if open}
				<div id={panelId} bind:this={panelEl} use:portal class="hierarchy-selector-panel" style={panelStyle}>
					<div bind:this={panelContentEl} class="hierarchy-selector-panel-content">
						{#if searchMode}
							{#if searchResults.length}
								{#each searchResults as item (valueForItem(item))}
									{@const selected = selectedValue.includes(valueForItem(item))}
									<div
										class="hierarchy-search-row"
										class:hierarchy-search-row-selected={selected}
										title={item.search_display || selectedLabelForItem(item)}
										role="button"
										tabindex="0"
										aria-pressed={selected}
										on:click={() => toggleItem(item)}
										on:keydown={(event) => {
											if (event.key === "Enter" || event.key === " ") {
												event.preventDefault();
												toggleItem(item);
											}
										}}
									>
										<span class="hierarchy-search-spacer"></span>
										<svg class="hierarchy-search-icon" viewBox="0 0 20 20" aria-hidden="true">
											<path d="M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z" />
											<path d="M11.25 2.95V7.3h4.3" />
										</svg>
										{#if breadcrumbMode}
											<span class="hierarchy-search-label hierarchy-search-name">{item.search_display}</span>
										{:else}
											<span class="hierarchy-search-label">
												<span class="hierarchy-search-name">{labelForItem(item)}</span>
												{#if item.search_path}
													<span class="hierarchy-search-path"> [{item.search_path}]</span>
												{/if}
											</span>
										{/if}
									</div>
								{/each}
							{:else}
								<div class="hierarchy-search-empty">{search_empty_label}</div>
							{/if}
						{:else}
							<TreeNodes folders={normalizedHierarchy.folders || []} items={normalizedHierarchy.items || []} depth={0} {expanded} value={selectedValue} {toggleItem} {toggleFolder} {valueForItem} {labelForItem} />
						{/if}
					</div>
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
		gap: 4px;
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

	.hierarchy-selector-search-input {
		flex: 1 1 96px;
		min-width: 70px;
		min-height: 26px;
		border: 0;
		outline: 0;
		background: transparent;
		color: var(--body-text-color);
		font: inherit;
		font-size: var(--text-sm);
		line-height: 20px;
		padding: 2px 4px;
	}

	.hierarchy-selector-search-input::placeholder {
		color: var(--body-text-color-subdued);
		opacity: 1;
	}

	.hierarchy-selector-clear {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex: 0 0 auto;
		width: 26px;
		height: 26px;
		border: none;
		border-radius: var(--button-small-radius);
		background: transparent;
		color: var(--body-text-color-subdued);
		cursor: pointer;
		font-size: var(--text-sm);
		font-weight: 600;
		line-height: 1;
		padding: 0;
	}

	.hierarchy-selector-clear:hover {
		color: var(--body-text-color);
		background: var(--background-fill-secondary);
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

	.hierarchy-selector-panel-content {
		min-width: 0;
		min-height: 28px;
	}

	.hierarchy-search-row {
		display: grid;
		grid-template-columns: 16px 20px minmax(0, 1fr);
		column-gap: 7px;
		align-items: center;
		min-height: 28px;
		box-sizing: border-box;
		border-radius: var(--radius-sm);
		color: var(--body-text-color);
		font-size: var(--text-md, var(--text-sm));
		line-height: 1.12;
		user-select: none;
		padding: 0 4px 0 6px;
		cursor: pointer;
		transition: background-color 120ms ease, color 120ms ease;
	}

	.hierarchy-search-row:hover {
		background: var(--background-fill-secondary);
	}

	.hierarchy-search-spacer {
		width: 16px;
		height: 16px;
	}

	.hierarchy-search-icon {
		width: 17px;
		height: 18px;
		color: var(--body-text-color-subdued);
		stroke: currentColor;
		stroke-width: 1.5;
		stroke-linecap: round;
		stroke-linejoin: round;
		fill: none;
	}

	.hierarchy-search-label {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.hierarchy-search-path {
		color: var(--body-text-color-subdued);
		font-style: italic;
	}

	.hierarchy-search-row-selected .hierarchy-search-name {
		font-weight: 700;
	}

	.hierarchy-search-empty {
		display: flex;
		align-items: center;
		min-height: 28px;
		padding: 0 10px;
		color: var(--body-text-color-subdued);
		font-size: var(--text-md, var(--text-sm));
		line-height: 1.12;
	}
</style>
