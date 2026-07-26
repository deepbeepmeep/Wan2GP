<script lang="ts" context="module">
	export interface EditorData {
		id?: string | null;
		background: FileData | null;
		layers: FileData[] | null;
		composite: FileData | null;
	}

	export interface ImageBlobs {
		id?: string | null;
		background: FileData | null;
		layers: FileData[];
		composite: FileData | null;
	}
</script>

<script lang="ts">
	import { createEventDispatcher } from "svelte";
	import { type I18nFormatter } from "@gradio/utils";
	import { type FileData, type Client } from "@gradio/client";
	import { type CommandNode } from "./shared/utils/commands";
	import ImageEditor from "./shared/ImageEditor.svelte";
	// import Layers from "./layers/Layers.svelte";
	import { type Brush as IBrush, type Eraser } from "./shared/brush/types";
	// import { type Eraser } from "./tools/Brush.svelte";
	import { type Tool as ToolbarTool } from "./shared/Toolbar.svelte";

	// import { Tools, Crop, Brush, Sources } from "./tools";
	import { BlockLabel } from "@gradio/atoms";
	import { Image as ImageIcon } from "@gradio/icons";
	import { inject } from "./shared/utils/parse_placeholder";
	// import Sources from "./shared/image/Sources.svelte";
	import {
		type LayerOptions,
		type Transform,
		type Source,
		type WebcamOptions
	} from "./shared/types";

	export let brush: IBrush;
	export let eraser: Eraser;
	export let sources: Source[];
	export let i18n: I18nFormatter;
	export let root: string;
	export let label: string | undefined = undefined;
	export let show_label: boolean;
	export let changeable = false;
	export let theme_mode: "dark" | "light";

	export let layers: FileData[];
	export let composite: FileData | null;
	export let background: FileData | null;
	export let value_id: string | null = null;

	export let layer_options: LayerOptions;
	export let transforms: Transform[];
	export let accept_blobs: (a: any) => Promise<void>;

	export let canvas_size: [number, number];
	export let fixed_canvas = false;
	export let realtime: boolean;
	export let upload: Client["upload"];
	export let is_dragging: boolean;
	export let placeholder: string | undefined = undefined;
	export let border_region: number;
	export let full_history: CommandNode | null = null;
	export let webcam_options: WebcamOptions;
	export let show_download_button = false;
	export let value_is_output = false;

	const dispatch = createEventDispatcher<{
		clear?: never;
		upload?: never;
		change?: never;
	}>();

	let editor: ImageEditor;
	let has_drawn = false;
	let last_data: ImageBlobs = empty_data();
	let last_value_id: string | null = null;
	let last_value_server_seen_at = 0;
	const instance_id = Math.random().toString(36).substring(2);

	function empty_data(): ImageBlobs {
		return { id: null, background: null, layers: [], composite: null };
	}

	function is_not_null(o: Blob | null): o is Blob {
		return !!o && o.size > 0;
	}

	$: if (background_image) dispatch("upload");

	function data_from_props(
		next_id: string | null,
		next_background: FileData | null,
		next_layers: FileData[] | null,
		next_composite: FileData | null
	): ImageBlobs {
		const layer_list = next_layers || [];
		return {
			id: next_id,
			background: next_background,
			layers: layer_list,
			composite: next_composite
		};
	}

	let get_data_inflight: Promise<ImageBlobs | { id: string }> | null = null;

	export async function get_data(): Promise<ImageBlobs | { id: string }> {
		while (get_data_inflight) {
			try {
				await get_data_inflight;
			} catch {
				// The next loop iteration should read the current editor state again.
			}
		}
		const read = read_data();
		get_data_inflight = read;
		try {
			return await read;
		} finally {
			if (get_data_inflight === read) {
				get_data_inflight = null;
			}
		}
	}

	function blob_file(blob: Blob, name: string): File {
		return new File([blob], name, { type: blob.type || "image/png" });
	}

	type DirtyState = { background: boolean; layers: boolean; composite: boolean; base_id?: string | null };

	async function store_blob(id: string, type: string, blob: Blob, index: number | null): Promise<void> {
		await accept_blobs({
			binary: true,
			data: { file: blob_file(blob, `${type}.png`), id, type, index, instance_id }
		});
	}

	async function store_meta(id: string, dirty: DirtyState): Promise<void> {
		await store_blob(
			id,
			"wangp_meta",
			new Blob([JSON.stringify({ ...dirty, base_id: last_value_id })], { type: "application/json" }),
			null
		);
	}

	async function store_blobs(blobs: { background: Blob | null; layers: (Blob | null)[]; composite: Blob | null }, dirty: DirtyState): Promise<{ id: string }> {
		const id = Math.random().toString(36).substring(2);
		const uploads: Promise<void>[] = [store_meta(id, dirty)];
		if (dirty.background && blobs.background) uploads.push(store_blob(id, "background", blobs.background, null));
		if (dirty.layers) uploads.push(...blobs.layers.filter(is_not_null).map((layer, i) => store_blob(id, "layer", layer, i)));
		if (dirty.composite && blobs.composite) uploads.push(store_blob(id, "composite", blobs.composite, null));
		await Promise.all(uploads);
		last_value_server_seen_at = Date.now();
		return { id };
	}

	function cached_value_survived_connection(): boolean {
		const connection_lost_at = typeof window === "undefined" ? 0 : (window as any).__wangp_gradio_connection_lost_at || 0;
		return !connection_lost_at || last_value_server_seen_at > connection_lost_at;
	}

	async function read_data(): Promise<ImageBlobs | { id: string }> {
		if (editor?.is_empty?.()) {
			last_data = empty_data();
			last_value_id = null;
			last_value_server_seen_at = 0;
			return last_data;
		}
		if (editor?.is_export_deferred?.()) return last_data;
		const dirty = editor.get_dirty_state();
		const force_full_value = Boolean(last_value_id) && !dirty.background && !dirty.layers && !dirty.composite && !cached_value_survived_connection();
		if (!dirty.background && !dirty.layers && !dirty.composite && !force_full_value) {
			if (last_value_id) return { id: last_value_id };
			const id = Math.random().toString(36).substring(2);
			await store_meta(id, dirty);
			last_value_server_seen_at = Date.now();
			return { id };
		}
		if (force_full_value) console.warn("WanGPImageEditor reconnect detected; get_value will return a full image/mask payload.");
		const export_dirty = force_full_value ? { background: true, layers: true, composite: false } : dirty;
		let blobs;
		try {
			blobs = await editor.get_blobs(export_dirty);
		} catch (e) {
			last_data = empty_data();
			return last_data;
		}
		const data = await store_blobs(blobs, export_dirty);
		editor.mark_value_clean();
		last_value_id = data.id;
		return data;
	}
	$: last_data = data_from_props(
		value_id,
		background,
		layers || [],
		composite
	);
	$: {
		last_value_id = last_data.background || last_data.layers.length || last_data.composite ? last_data.id || null : null;
	}

	let background_image = false;
	let history = false;

	function nextframe(): Promise<void> {
		return new Promise((resolve) => setTimeout(() => resolve(), 30));
	}

	let uploading = false;
	let pending = false;
	async function handle_change(e: CustomEvent<Blob | any>): Promise<void> {
		if (!realtime) return;
		if (editor?.is_export_deferred?.()) return;
		if (uploading) {
			pending = true;
			return;
		}
		uploading = true;
		await nextframe();
		dispatch("change");
		await nextframe();
		uploading = false;
		if (pending) {
			pending = false;
			uploading = false;
			handle_change(e);
		}
	}

	$: [heading, paragraph] = placeholder ? inject(placeholder) : [false, false];

	let current_tool: ToolbarTool;
</script>

<BlockLabel
	{show_label}
	Icon={ImageIcon}
	label={label || i18n("image.image")}
/>
<ImageEditor
	{transforms}
	{composite}
	{layers}
	{background}
	on:history
	{canvas_size}
	bind:this={editor}
	{changeable}
	on:save
	on:change={handle_change}
	on:clear={() => dispatch("clear")}
	on:download_error
	{sources}
	{full_history}
	bind:background_image
	bind:current_tool
	brush_options={brush}
	eraser_options={eraser}
	{fixed_canvas}
	{border_region}
	{layer_options}
	{i18n}
	{root}
	{upload}
	bind:is_dragging
	bind:has_drawn
	{webcam_options}
	{show_download_button}
	{theme_mode}
	{value_is_output}
>
	{#if !background_image && current_tool === "image" && !has_drawn}
		<div class="empty wrap">
			{#if sources && sources.length}
				{#if heading || paragraph}
					{#if heading}
						<h2>{heading}</h2>
					{/if}
					{#if paragraph}
						<p>{paragraph}</p>
					{/if}
				{:else}
					<div>Upload an image</div>
				{/if}
			{/if}

			{#if sources && sources.length && brush && !placeholder}
				<div class="or">or</div>
			{/if}
			{#if brush && !placeholder}
				<div>select the draw tool to start</div>
			{/if}
		</div>
	{/if}
</ImageEditor>

<style>
	h2 {
		font-size: var(--text-xl);
	}

	p,
	h2 {
		white-space: pre-line;
	}

	.empty {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		position: absolute;
		height: 100%;
		width: 100%;
		left: 0;
		right: 0;
		margin: auto;
		z-index: var(--layer-1);
		text-align: center;
		color: var(--color-grey-500) !important;
		cursor: pointer;
	}

	.wrap {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		line-height: var(--line-md);
		font-size: var(--text-md);
	}

	.or {
		color: var(--body-text-color-subdued);
	}
</style>
