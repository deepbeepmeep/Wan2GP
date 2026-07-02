<script lang="ts" context="module">
	import type { Tool, Subtool } from "./Toolbar.svelte";

	export const EDITOR_KEY = Symbol("editor");
	export type context_type = "bg" | "layers" | "crop" | "draw" | "erase";
</script>

<script lang="ts">
	import { onMount, createEventDispatcher, tick } from "svelte";
	import Toolbar, { type Tool as ToolbarTool } from "./Toolbar.svelte";
	import { CropTool } from "./crop/crop";
	import { ResizeTool } from "./resize/resize";
	import { Webcam } from "@gradio/image";
	import type { I18nFormatter } from "@gradio/utils";
	import type { Client, FileData } from "@gradio/client";
	import tinycolor, { type ColorInput } from "tinycolor2";
	import { ZoomTool } from "./zoom/zoom";
	import { type CommandManager, type CommandNode } from "./utils/commands";
	import { ImageEditor } from "./core/editor";
	import { type Brush, type Eraser } from "./brush/types";
	import { BrushTool } from "./brush/brush";
	import { create_drag } from "@gradio/upload";
	import SecondaryToolbar from "./SecondaryToolbar.svelte";
	import { Check } from "@gradio/icons";
	import type { LayerOptions, Source, Transform, WebcamOptions } from "./types";

	import { type ImageBlobs } from "./types";
	import Controls from "./Controls.svelte";
	import IconButton from "./IconButton.svelte";

	const { drag, open_file_upload } = create_drag();
	const dispatch = createEventDispatcher<{
		clear?: never;
		save: void;
		change: void;
		history: CommandManager["current_history"];
		upload: void;
		input: void;
		download_error: string;
	}>();

	export const antialias = true;
	export const full_history: CommandNode | null = null;

	export let changeable = false;
	export let sources: Source[] = ["upload", "webcam", "clipboard"];
	export let transforms: Transform[] = ["crop", "resize"];
	export let canvas_size: [number, number];
	export let is_dragging = false;
	export let background_image = false;
	export let brush_options: Brush;
	export let eraser_options: Eraser;
	export let fixed_canvas = false;
	export let root: string;
	export let i18n: I18nFormatter;
	export let upload: Client["upload"];
	export let composite: FileData | null;
	export let layers: FileData[];
	export let background: FileData | null;
	export let border_region = 0;
	export let layer_options: LayerOptions;
	export let current_tool: ToolbarTool;
	export let webcam_options: WebcamOptions;
	export let show_download_button = false;
	export let theme_mode: "dark" | "light";
	export let value_is_output = false;

	let pixi_target: HTMLDivElement;
	let pixi_target_crop: HTMLDivElement;

	$: if (layer_options) {
		if (check_if_should_init()) {
			editor.set_layer_options(layer_options);
			refresh_tools();
		}
	}

	function refresh_tools(): void {
		editor.set_tool(current_tool);
		editor.set_subtool(current_subtool);
	}

	function check_if_should_init(): boolean {
		return layer_options && editor && ready;
	}

	export let has_drawn = false;
	let background_dirty = false;
	let layers_dirty = false;

	export function get_dirty_state(): { background: boolean; layers: boolean; composite: boolean } {
		return {
			background: background_dirty,
			layers: layers_dirty,
			composite: false
		};
	}

	export function mark_value_clean(): void {
		background_dirty = false;
		layers_dirty = false;
		editor.wangp_value_dirty = false;
		crop.wangp_value_dirty = false;
	}

	/**
	 * Gets the image blobs from the editor
	 * @returns {Promise<ImageBlobs>} Object containing background, layers, and composite image blobs
	 */
	function canvas_blob(canvas: HTMLCanvasElement): Promise<Blob> {
		return new Promise((resolve, reject) => {
			canvas.toBlob((blob) => {
				if (blob) resolve(blob);
				else reject(new Error("Could not export empty image editor mask."));
			}, "image/png");
		});
	}

	async function empty_mask_blob(width: number, height: number): Promise<Blob> {
		const canvas = document.createElement("canvas");
		canvas.width = Math.max(1, Math.round(width));
		canvas.height = Math.max(1, Math.round(height));
		const ctx = canvas.getContext("2d");
		if (!ctx) throw new Error("Could not create 2D canvas context for empty image editor mask.");
		ctx.fillStyle = "#000";
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		return canvas_blob(canvas);
	}

	type DirtyState = { background: boolean; layers: boolean; composite: boolean };

	export async function get_blobs(dirty: DirtyState = { background: true, layers: true, composite: true }): Promise<ImageBlobs> {
		if (!editor) return { background: null, layers: [], composite: null };
		if (is_empty())
			return { background: null, layers: [], composite: null };
		if (is_export_deferred()) return { background: null, layers: [], composite: null };
		const source_image = editor.wangp_source_image;
		if (
			source_image &&
			editor.wangp_source_image_clean &&
			layer_options.disabled &&
			background_image &&
			!has_drawn
		) {
			const next_layers = dirty.layers
				? layers?.length
					? (await editor.get_blobs({ background: false, layers: true, composite: false })).layers
					: [await empty_mask_blob(editor.width, editor.height)]
				: [];
			return {
				background: dirty.background ? source_image : null,
				layers: next_layers,
				composite: dirty.composite ? source_image : null
			};
		}
		const blobs = await editor.get_blobs(dirty);
		return blobs;
	}

	export function is_export_deferred(): boolean {
		return current_tool === "image" && current_subtool === "crop";
	}

	export function is_empty(): boolean {
		return !editor || editor.wangp_editor_empty || (!background_image && !has_drawn);
	}

	let editor: ImageEditor;
	let clearing_value: Promise<void> | null = null;

	export async function clear_value(
		force = false,
		notify = false,
		is_current = () => true
	): Promise<void> {
		if (clearing_value) await clearing_value;
		if (!editor || !crop || !ready || !is_current()) return;
		if (
			!force &&
			editor.wangp_editor_empty &&
			crop.wangp_editor_empty &&
			!background_image &&
			!has_drawn
		) {
			editor.wangp_value_dirty = false;
			crop.wangp_value_dirty = false;
			return;
		}
		const clear = (async () => {
			await editor.reset_canvas(notify);
			if (!is_current()) return;
			await crop.reset_canvas(notify);
			if (!is_current()) return;
			editor.wangp_set_empty(true);
			crop.wangp_set_empty(true);
			handle_tool_change({ tool: "image" });
			background_image = false;
			has_drawn = false;
			released_value_key = null;
			background_dirty = false;
			layers_dirty = false;
			editor.wangp_value_dirty = false;
			crop.wangp_value_dirty = false;
		})();
		clearing_value = clear;
		try {
			await clear;
		} finally {
			if (clearing_value === clear) clearing_value = null;
		}
	}

	/**
	 * Adds an image to the editor
	 * @param {Blob | File} image - The image to add
	 */
	export function add_image(image: Blob | File): void {
		synced_value_key = "";
		released_value_key = null;
		background_dirty = true;
		layers_dirty = true;
		editor.wangp_set_empty(false);
		editor.add_image({ image });
	}

	let pending_bg: Promise<void>;
	/**
	 * Adds an image to the editor from a URL
	 * @param {string | FileData} source - The URL of the image or a FileData object
	 * @returns {Promise<void>}
	 */
	export async function add_image_from_url(
		source:
			| string
			| {
					url: string;
					meta: {
						_type: string;
					};
			  }
			| any,
		dispatch_events = true,
		is_current = () => true
	): Promise<void> {
		if (!editor || !source || !check_if_should_init() || !is_current()) return;
		let url: string;

		// Handle different source types
		if (typeof source === "string") {
			url = source;
		} else if (source?.meta?._type === "gradio.FileData" && source?.url) {
			url = source.url;
		} else {
			console.warn("Invalid source provided to add_image_from_url:", source);
			return;
		}

		try {
			editor.wangp_set_empty(false);
			crop.wangp_set_empty(false);
			editor.wangp_resize_to_element();
			crop.wangp_resize_to_element();
			pending_bg = editor.add_image_from_url(url, is_current, dispatch_events);
			let pending_crop = crop.add_image_from_url(url, is_current, dispatch_events);
			await Promise.all([pending_bg, pending_crop]);
			if (!is_current()) return;
			editor.wangp_value_dirty = false;
			crop.wangp_value_dirty = false;
			crop.wangp_reset_crop_tool();
			crop.set_tool("image");
			crop.set_subtool("crop");
			background_image = true;
			if (dispatch_events) {
				dispatch("upload");
				dispatch("input");
			}
		} catch (error) {
			console.error("Error adding image from URL:", error);
		}
	}

	/**
	 * Adds a new layer with an image loaded from a URL
	 * @param {string | FileData} source - The URL of the image or a FileData object
	 * @returns {Promise<string | null>} - The ID of the created layer, or null if failed
	 */
	export async function add_layers_from_url(
		source: FileData[] | any,
		dispatch_events = true,
		is_current = () => true
	): Promise<void> {
		if (!editor || !source.length || !check_if_should_init() || !is_current()) return;

		// Handle different source types
		if (
			Array.isArray(source) &&
			source.every((item) => item?.meta?._type === "gradio.FileData")
		) {
			try {
				await pending_bg;
				if (!is_current()) return;

				await editor.add_layers_from_url(
					source.map((item) => item.url),
					is_current,
					dispatch_events
				);
				if (!is_current()) return;
				editor.wangp_value_dirty = false;
				if (dispatch_events) {
					dispatch("change");
					dispatch("input");
				}
			} catch (error) {
				console.error("Error adding layer from URL:", error);
			}
		}
	}

	async function clear_mask_layers(is_current = () => true): Promise<void> {
		await editor.add_layers_from_url([], is_current, false);
		if (!is_current()) return;
		synced_value_key = "";
		released_value_key = null;
		has_drawn = false;
		layers_dirty = true;
	}

	let brush: BrushTool;
	let zoom: ZoomTool;
	let zoom_level = 1;
	let ready = false;
	let mounted = false;
	let min_zoom = true;
	let editor_unsubscribers: (() => void)[] = [];

	let last_dimensions = { width: 0, height: 0 };

	/**
	 * Handles visibility changes and resets zoom if dimensions have changed
	 */
	async function handle_visibility_change(): Promise<void> {
		if (!editor || !ready || !zoom) return;
		await tick();
		const is_visible = pixi_target.offsetParent !== null;
		if (editor.wangp_editor_empty) return;

		if (is_visible) {
			const current_dimensions = pixi_target.getBoundingClientRect();
			editor.wangp_resize_to_element();
			crop?.wangp_resize_to_element();
			if (current_tool === "draw" || current_tool === "erase") {
				editor.set_tool(current_tool);
			}

			if (
				current_dimensions.width !== last_dimensions.width ||
				current_dimensions.height !== last_dimensions.height
			) {
				// Use set_zoom with "fit" to reset to appropriate zoom level
				zoom.set_zoom("fit");

				// Update the last known dimensions
				last_dimensions = {
					width: current_dimensions.width,
					height: current_dimensions.height
				};
			}
			if (released_value_key === current_value_key) {
				await sync_value_from_props(
					current_value_key,
					background,
					layers || [],
					composite,
					false,
					true
				);
			}
		} else {
			if (!editor.wangp_editor_empty && synced_value_key) {
				released_value_key = synced_value_key;
			}
			editor.wangp_release_surface();
			crop?.wangp_release_surface();
			if (current_subtool === "crop") {
				(crop?.tools.get("crop") as any)?.wangp_cancel_active_drag?.();
			}
		}
	}

	onMount(() => {
		let intersection_observer: IntersectionObserver;
		let resize_observer: ResizeObserver;
		let destroyed = false;
		init_image_editor().then(() => {
			if (destroyed) {
				editor?.destroy();
				crop?.destroy();
				return;
			}
			mounted = true;
			intersection_observer = new IntersectionObserver(() => {
				handle_visibility_change();
			});

			resize_observer = new ResizeObserver(() => {
				handle_visibility_change();
			});

			intersection_observer.observe(pixi_target);
			resize_observer.observe(pixi_target);
		});

		// Set up mutation observer to detect visibility changes

		return () => {
			destroyed = true;
			if (intersection_observer) {
				intersection_observer.disconnect();
			}
			if (resize_observer) {
				resize_observer.disconnect();
			}
			for (const unsubscribe of editor_unsubscribers.splice(0)) {
				unsubscribe();
			}
			if (editor) {
				editor.destroy();
			}
			if (crop) {
				crop.destroy();
			}
		};
	});

	let crop: ImageEditor;
	let crop_zoom: ZoomTool;
	async function init_image_editor(): Promise<void> {
		brush = new BrushTool();
		zoom = new ZoomTool();
		editor = new ImageEditor({
			target_element: pixi_target,
			width: canvas_size[0],
			height: canvas_size[1],
			tools: ["image", zoom, new ResizeTool(), brush],
			fixed_canvas,
			border_region,
			layer_options,
			theme_mode
		});

		brush.on("change", () => {
			if (!clearing_value && !editor.wangp_editor_empty) {
				synced_value_key = "";
				released_value_key = null;
				has_drawn = true;
				layers_dirty = true;
			}
		});

		crop_zoom = new ZoomTool();

		crop = new ImageEditor({
			target_element: pixi_target_crop,
			width: canvas_size[0],
			height: canvas_size[1],
			tools: ["image", crop_zoom, new CropTool()],
			dark: true,
			fixed_canvas: false,
			border_region: 0,
			pad_bottom: 40
		});

		editor_unsubscribers.push(editor.scale.subscribe((_scale) => {
			zoom_level = _scale;
		}));

		editor_unsubscribers.push(editor.min_zoom.subscribe((is_min_zoom) => {
			min_zoom = is_min_zoom;
		}));

		editor_unsubscribers.push(editor.dimensions.subscribe((dimensions) => {
			// Store dimensions for later comparison
			last_dimensions = { ...dimensions };
		}));

		await Promise.all([editor.ready, crop.ready]).then(() => {
			handle_tool_change({ tool: "image" });
			ready = true;
			editor.wangp_set_empty(true);
			crop.wangp_set_empty(true);
			if (sources.length > 0) {
				handle_tool_change({ tool: "image" });
			} else {
				handle_tool_change({ tool: "draw" });
			}
			crop.set_subtool("crop");
		});

		editor.on("change", async () => {
			if (current_tool === "image" && current_subtool === "size") {
				await clear_mask_layers();
			}
			dispatch("change");
		});

	}

	function file_key(file: FileData | null): string | null {
		if (!file) return null;
		return file.url || (file as any).path || (file as any).orig_name || null;
	}

	function value_key(
		next_background: FileData | null,
		next_layers: FileData[],
		next_composite: FileData | null
	): string {
		return JSON.stringify([
			file_key(next_background),
			next_layers.map(file_key),
			file_key(next_composite)
		]);
	}

	function empty_value(
		next_background: FileData | null,
		next_layers: FileData[],
		next_composite: FileData | null
	): boolean {
		return !next_background && next_layers.length === 0 && !next_composite;
	}

	function release_surfaces_if_hidden(): void {
		if (pixi_target?.offsetParent === null) {
			if (!editor.wangp_editor_empty && synced_value_key) {
				released_value_key = synced_value_key;
			}
			editor.wangp_release_surface();
			crop.wangp_release_surface();
		}
	}

	let synced_value_key = "";
	let released_value_key: string | null = null;
	let value_sync_serial = 0;
	async function sync_value_from_props(
		next_key: string,
		next_background: FileData | null,
		next_layers: FileData[],
		next_composite: FileData | null,
		force_empty = false,
		force_restore = false
	): Promise<void> {
		const is_empty_value = empty_value(next_background, next_layers, next_composite);
		if (
			!force_restore &&
			next_key === synced_value_key &&
			!(force_empty && is_empty_value)
		) return;

		const sync_id = ++value_sync_serial;
		const is_current = () => sync_id === value_sync_serial;

		if (is_empty_value) {
			await tick();
			if (!is_current()) return;
			await clear_value(true, false, is_current);
			if (!is_current()) return;
			synced_value_key = next_key;
			released_value_key = null;
			return;
		}

		await clear_value(true, false, is_current);
		if (!is_current()) return;
		await add_image_from_url(next_background || next_composite, false, is_current);
		if (!is_current()) return;
		await add_layers_from_url(next_layers, false, is_current);
		if (!is_current()) return;
		editor.wangp_value_dirty = false;
		crop.wangp_value_dirty = false;
		background_dirty = false;
		layers_dirty = false;
		synced_value_key = next_key;
		released_value_key = null;
		handle_tool_change({ tool: "draw" });
		release_surfaces_if_hidden();
	}

	$: current_value_key = value_key(background, layers || [], composite);
	$: if (editor && ready) {
		sync_value_from_props(
			current_value_key,
			background,
			layers || [],
			composite,
			value_is_output
		);
	}

	let was_crop_active = false;
	$: {
		const crop_active = current_tool === "image" && current_subtool === "crop";
		if (crop_active && !was_crop_active) {
			tick().then(() => crop_zoom?.set_zoom("fit"));
		}
		was_crop_active = crop_active;
	}

	// function resize_canvas(width: number, height: number): void {
	// 	if (!editor) return;
	// 	if (mounted && ready) {
	// 		editor.resize(width, height);
	// 	}
	// }

	/**
	 * Handles file uploads
	 * @param {File[]} files - The uploaded files
	 */
	async function handle_files(
		files: File[] | Blob[] | File | Blob | null
	): Promise<void> {
		if (files == null) return;
		if (!sources.includes("upload")) return;
		const _file = Array.isArray(files) ? files[0] : files;
		background_dirty = true;
		layers_dirty = true;
		released_value_key = null;
		editor.wangp_set_empty(false);
		crop.wangp_set_empty(false);
		editor.wangp_resize_to_element();
		crop.wangp_resize_to_element();
		await editor.reset_canvas(false);
		await crop.reset_canvas(false);
		editor.wangp_set_empty(false);
		crop.wangp_set_empty(false);
		await editor.add_image({ image: _file });
		await crop.add_image({ image: _file });
		synced_value_key = "";
		crop.reset();
		crop.wangp_reset_crop_tool();
		background_image = true;
		handle_tool_change({ tool: "draw" });
		dispatch("upload");
		dispatch("input");
		dispatch("change");
	}

	/**
	 * Handles tool change events
	 * @param {{ tool: ToolbarTool }} param0 - Object containing the selected tool
	 */
	function handle_tool_change({ tool }: { tool: ToolbarTool }): void {
		editor.set_tool(tool);
		current_tool = tool;

		if (tool === "image") {
			crop.set_tool("image");
			crop.set_subtool("crop");
			if (!background_image) {
				crop.wangp_reset_crop_tool();
			}
		} else {
			editor.wangp_set_empty(false);
			crop.wangp_set_empty(false);
			editor.wangp_resize_to_element();
			crop.wangp_resize_to_element();
		}
	}

	/**
	 * Handles subtool change events
	 * @param {{ tool: ToolbarTool, subtool: Subtool }} param0 - Object containing the selected tool and subtool
	 */
	function handle_subtool_change({
		tool,
		subtool
	}: {
		tool: ToolbarTool;
		subtool: Subtool | null;
	}): void {
		editor.set_subtool(subtool);
		current_subtool = subtool;

		if (subtool === null) {
			return;
		}

		if (tool === "draw") {
			if (subtool === "size") {
				brush_size_visible = true;
			} else if (subtool === "color") {
				brush_color_visible = true;
			}
		}

		if (tool === "erase" && subtool === "size") {
			eraser_size_visible = true;
		}

		if (tool === "image" && subtool === "paste") {
			process_clipboard();
		}

		if (tool === "image" && subtool === "upload") {
			tick().then(() => {
				disable_click = false;
				open_file_upload();
			});
		}
	}

	let eraser_size_visible = false;
	let selected_color: ColorInput | string;
	let selected_size: number;
	let selected_opacity = 1;
	let selected_eraser_size: number;

	$: {
		if (brush_options) {
			update_brush_options();
		}

		if (eraser_options) {
			update_eraser_options();
		}
	}

	function update_brush_options(): void {
		const default_color =
			brush_options.default_color === "auto"
				? brush_options.colors[0]
				: brush_options.default_color;

		// color is already a tuple [color, opacity]
		if (Array.isArray(default_color)) {
			selected_color = default_color[0];
			selected_opacity = default_color[1];
		} else {
			selected_color = default_color;

			// color is a string, check if it has opacity info
			const color = tinycolor(default_color);
			if (color.getAlpha() < 1) {
				selected_opacity = color.getAlpha();
			} else {
				selected_opacity = 1;
			}
		}

		selected_size =
			typeof brush_options.default_size === "number"
				? brush_options.default_size
				: 25;
	}

	function update_eraser_options(): void {
		selected_eraser_size =
			eraser_options.default_size === "auto" ? 25 : eraser_options.default_size;
	}

	let brush_size_visible = false;

	let brush_color_visible = false;

	$: brush?.set_brush_color(
		(() => {
			let color_value;
			if (selected_color === "auto") {
				const default_color =
					brush_options.colors.find((color) =>
						Array.isArray(color)
							? color[0] === brush_options.default_color
							: color === brush_options.default_color
					) || brush_options.colors[0];

				color_value = Array.isArray(default_color)
					? default_color[0]
					: default_color;
			} else {
				color_value = selected_color;
			}
			return color_value;
		})()
	);

	// Type-safe brush size handling
	$: brush?.set_brush_size(
		typeof selected_size === "number" ? selected_size : 25
	);

	$: brush?.set_eraser_size(
		typeof selected_eraser_size === "number" ? selected_eraser_size : 25
	);

	$: disable_click =
		(current_tool && current_tool !== "image") ||
		(current_tool === "image" && background_image) ||
		(current_tool === "image" && current_subtool === "webcam") ||
		!sources.includes("upload");

	let current_subtool: Subtool | null = null;
	let preview = false;
	$: brush?.preview_brush(preview);
	$: brush?.set_brush_opacity(selected_opacity);

	function handle_zoom_change(zoom_level: number | "fit"): void {
		zoom.set_zoom(zoom_level);
	}

	function zoom_in_out(direction: "in" | "out"): void {
		zoom.set_zoom(
			direction === "in"
				? zoom_level + (zoom_level < 1 ? 0.1 : zoom_level * 0.1)
				: zoom_level - (zoom_level < 1 ? 0.1 : zoom_level * 0.1)
		);
	}

	async function process_clipboard(): Promise<void> {
		const items = await navigator.clipboard.read();

		for (let i = 0; i < items.length; i++) {
			const type = items[i].types.find((t) => t.startsWith("image/"));
			if (type) {
				const blob = await items[i].getType(type);

				handle_files(blob);
			}
		}
	}

	function handle_capture(e: CustomEvent): void {
		if (e.detail !== null) {
			handle_files(e.detail as Blob);
		}
		handle_subtool_change({ tool: current_tool, subtool: null });
	}

	function handle_save(): void {
		dispatch("save");
	}

	async function handle_crop_confirm(): Promise<void> {
		const { image } = await crop.get_crop_bounds();
		if (!image) return;

		await editor.add_image({
			image,
			resize: false
		});
		await clear_mask_layers();
		synced_value_key = "";
		released_value_key = null;
		background_dirty = true;
		layers_dirty = true;
		handle_subtool_change({ tool: "image", subtool: null });
		dispatch("change");
		dispatch("input");
	}

	async function handle_download(): Promise<void> {
		const blobs = await editor.get_blobs();

		const blob = blobs.composite;
		if (!blob) {
			dispatch("download_error", "Unable to generate image to download.");
			return;
		}
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		link.download = "image.png";
		link.click();
		URL.revokeObjectURL(url);
	}
</script>

<div
	data-testid="image"
	class="image-container"
	class:dark-bg={current_subtool === "crop"}
	use:drag={{
		on_drag_change: (dragging) => (is_dragging = dragging),
		on_files: handle_files,
		accepted_types: "image/*",
		disable_click: disable_click
	}}
	aria-label={"Click to upload or drop files"}
	aria-dropeffect="copy"
>
	{#if ready}
		{#if current_subtool !== "crop"}
			<Controls
				{changeable}
				on:set_zoom={(e) => handle_zoom_change(e.detail)}
				on:zoom_in={() => zoom_in_out("in")}
				on:zoom_out={() => zoom_in_out("out")}
				{min_zoom}
				current_zoom={zoom_level}
				on:remove_image={async () => {
					await clear_value(true, false);
					dispatch("clear");
				}}
				tool={current_tool}
				can_save={true}
				on:save={handle_save}
				on:pan={(e) => {
					handle_tool_change({ tool: "pan" });
				}}
				enable_download={show_download_button}
				on:download={() => handle_download()}
			/>
		{/if}

		{#if current_subtool !== "crop"}
			<Toolbar
				{sources}
				{transforms}
				background={background_image}
				on:tool_change={(e) => handle_tool_change(e.detail)}
				on:subtool_change={(e) => handle_subtool_change(e.detail)}
				show_brush_size={brush_size_visible}
				show_brush_color={brush_color_visible}
				show_eraser_size={eraser_size_visible}
				{brush_options}
				{eraser_options}
				bind:selected_color
				bind:selected_size
				bind:selected_eraser_size
				bind:selected_opacity
				bind:preview
				tool={current_tool}
				subtool={current_subtool}
			/>
		{/if}

		{#if current_tool === "image" && current_subtool === "webcam"}
			<div class="modal">
				<div class="modal-inner">
					<Webcam
						{upload}
						{root}
						on:capture={handle_capture}
						on:error
						on:drag
						streaming={false}
						mode="image"
						include_audio={false}
						{i18n}
						mirror_webcam={webcam_options.mirror}
						webcam_constraints={webcam_options.constraints}
					/>
				</div>
			</div>
		{/if}

		{#if current_subtool !== "crop" && !layer_options.disabled}
			<SecondaryToolbar
				enable_additional_layers={layer_options.allow_additional_layers}
				layers={editor.layers}
				on:new_layer={() => {
					editor.add_layer();
				}}
				on:change_layer={(e) => {
					editor.set_layer(e.detail);
					if (current_tool === "draw") {
						handle_tool_change({ tool: "draw" });
					}
				}}
				on:move_layer={(e) => {
					editor.move_layer(e.detail.id, e.detail.direction);
				}}
				on:delete_layer={(e) => {
					editor.delete_layer(e.detail);
				}}
				on:toggle_layer_visibility={(e) => {
					editor.toggle_layer_visibility(e.detail);
				}}
			/>
		{/if}
	{/if}
	<div
		class="pixi-target"
		class:visible={current_subtool !== "crop"}
		bind:this={pixi_target}
	></div>
	<div
		class="pixi-target-crop"
		class:visible={current_subtool === "crop"}
		bind:this={pixi_target_crop}
	></div>

	{#if current_subtool === "crop"}
		<div class="crop-confirm-button">
			<IconButton
				Icon={Check}
				label="Confirm crop"
				show_label={true}
				size="large"
				padded={true}
				color="white"
				background="var(--color-green-500)"
				label_position="right"
				on:click={handle_crop_confirm}
			/>
		</div>
	{/if}
	<slot></slot>
</div>

<style>
	.image-container {
		display: flex;
		height: 100%;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		max-height: 100%;
		border-radius: var(--radius-sm);
	}

	:global(.wangp-empty-image-editor) {
		background: var(--block-background-fill, #ffffff) !important;
		min-height: 100% !important;
	}

	:global(.wangp-empty-image-editor canvas) {
		opacity: 0 !important;
		pointer-events: none !important;
	}

	:global(.wangp-empty-image-editor .pixi-target),
	:global(.wangp-empty-image-editor .pixi-target-crop) {
		opacity: 0 !important;
		pointer-events: none !important;
	}

	.pixi-target {
		width: 100%;
		height: 100%;
		position: absolute;
		top: 0;
		left: 0;
		z-index: 1;
		display: block;
		opacity: 0;
		pointer-events: none;
		border-radius: var(--radius-sm);
	}

	.pixi-target-crop {
		width: 100%;
		height: 100%;
		position: absolute;
		top: 0;
		left: 0;
		z-index: 2;
		display: block;
		opacity: 0;
		pointer-events: none;
		border-radius: var(--radius-sm);
	}

	.visible {
		opacity: 1;
		pointer-events: auto;
	}

	.pixi-target {
		width: 100%;
		height: 100%;
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		overflow: hidden;
	}

	.modal {
		position: absolute;
		height: 100%;
		width: 100%;
		left: 0;
		right: 0;
		margin: auto;
		z-index: var(--layer-top);
		display: flex;
		align-items: center;
	}

	.modal-inner {
		height: 100%;
		width: 100%;
		background: var(--block-background-fill);
	}

	.dark-bg {
		background: #333;
	}

	.crop-confirm-button {
		position: absolute;
		bottom: 8px;
		left: 0;
		right: 0;
		margin: auto;
		z-index: var(--layer-top);
		display: flex;
		align-items: center;
		justify-content: center;
	}
</style>
