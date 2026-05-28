<script>
	export let folders = [];
	export let items = [];
	export let depth = 0;
	export let expanded;
	export let value;
	export let toggleItem;
	export let toggleFolder;
	export let valueForItem;
	export let labelForItem;

	function folderLabel(folder) {
		return String(folder.name || folder.path || "");
	}

	function folderPath(folder) {
		return String(folder.path || folder.name || "");
	}
</script>

{#each folders as folder (folderPath(folder))}
	{@const path = folderPath(folder)}
	<div
		class="hierarchy-row hierarchy-folder"
		style:padding-left={`${depth * 18 + 6}px`}
		role="button"
		tabindex="0"
		title={folderLabel(folder)}
		on:click={() => toggleFolder(path)}
		on:keydown={(event) => {
			if (event.key === "Enter" || event.key === " ") {
				event.preventDefault();
				toggleFolder(path);
			}
		}}
	>
		<svg class="hierarchy-twist" class:hierarchy-twist-open={expanded.has(path)} viewBox="0 0 16 16" aria-hidden="true">
			<path d="M6 4.5L10 8l-4 3.5" />
		</svg>
		<svg class="hierarchy-icon hierarchy-folder-icon" viewBox="0 0 20 20" aria-hidden="true">
			<path d="M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z" />
			<path d="M2.25 7.95V5.6c0-.55.45-1 1-1h4.5l1.35 1.65" />
		</svg>
		<span class="hierarchy-name">{folderLabel(folder)}</span>
	</div>
	{#if expanded.has(path)}
		<svelte:self
			folders={folder.folders || []}
			items={folder.items || []}
			depth={depth + 1}
			{expanded}
			{value}
			{toggleItem}
			{toggleFolder}
			{valueForItem}
			{labelForItem}
		/>
	{/if}
{/each}

{#each items as item (valueForItem(item))}
	{@const selected = value.includes(valueForItem(item))}
	<div
		class="hierarchy-row hierarchy-item"
		class:hierarchy-item-selected={selected}
		style:padding-left={`${depth * 18 + 6}px`}
		title={labelForItem(item)}
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
		<span class="hierarchy-twist-spacer"></span>
		<svg class="hierarchy-icon hierarchy-item-icon" viewBox="0 0 20 20" aria-hidden="true">
			<path d="M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z" />
			<path d="M11.25 2.95V7.3h4.3" />
		</svg>
		<span class="hierarchy-name">{labelForItem(item)}</span>
	</div>
{/each}

<style>
	.hierarchy-row {
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
		padding-right: 4px;
		transition: background-color 120ms ease, color 120ms ease, opacity 120ms ease;
	}

	.hierarchy-row:hover {
		background: var(--background-fill-secondary);
	}

	.hierarchy-folder {
		cursor: pointer;
		font-style: italic;
		font-weight: 400;
	}

	.hierarchy-twist {
		width: 16px;
		height: 16px;
		color: var(--body-text-color-subdued);
		stroke: currentColor;
		stroke-width: 1.8;
		stroke-linecap: round;
		stroke-linejoin: round;
		fill: none;
		transition: transform 120ms ease;
	}

	.hierarchy-twist-open {
		transform: rotate(90deg);
	}

	.hierarchy-twist-spacer {
		width: 16px;
		height: 16px;
	}

	.hierarchy-icon {
		width: 18px;
		height: 18px;
		color: var(--body-text-color-subdued);
		stroke: currentColor;
		stroke-width: 1.5;
		stroke-linecap: round;
		stroke-linejoin: round;
		fill: none;
	}

	.hierarchy-item-icon {
		width: 17px;
	}

	.hierarchy-name {
		min-width: 0;
		flex: 1 1 auto;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.hierarchy-item {
		cursor: pointer;
	}

	.hierarchy-item-selected {
		color: var(--body-text-color);
	}

	.hierarchy-item-selected .hierarchy-name {
		font-weight: 700;
	}
</style>
