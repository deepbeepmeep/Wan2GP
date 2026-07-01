<svelte:options accessors={true} immutable={true} />

<script lang="ts">
	import type { Brush, Eraser } from "./shared/brush/types";
	import type { EditorData, ImageBlobs } from "./InteractiveImageEditor.svelte";

	import { FileData } from "@gradio/client";

	import type { Gradio, SelectData } from "@gradio/utils";
	import { BaseStaticImage as StaticImage } from "@gradio/image";
	import InteractiveImageEditor from "./InteractiveImageEditor.svelte";
	import { Block } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";
	import type { LoadingStatus } from "@gradio/statustracker";
	import { afterUpdate, tick } from "svelte";
	import type {
		LayerOptions,
		Transform,
		Source,
		WebcamOptions
	} from "./shared/types";

	export let elem_id = "";
	export let elem_classes: string[] = [];
	export let visible = true;
	export let value: EditorData | null = {
		background: null,
		layers: [],
		composite: null
	};
	export let label: string;
	export let show_label: boolean;
	export let show_download_button: boolean;
	export let root: string;
	export let value_is_output = false;

	export let height = 350;
	export let width: number | undefined;

	export let _selectable = false;
	export let container = true;
	export let scale: number | null = null;
	export let min_width: number | undefined = undefined;
	export let loading_status: LoadingStatus;
	export let show_share_button = false;
	export let sources: Source[] = [];
	export let interactive: boolean;
	export let placeholder: string | undefined;
	export let brush: Brush;
	export let eraser: Eraser;
	export let transforms: Transform[] = [];
	export let layers: LayerOptions;
	export let attached_events: string[] = [];
	export let server: {
		accept_blobs: (a: any) => Promise<void>;
	};
	export let canvas_size: [number, number];
	export let fixed_canvas = false;
	export let show_fullscreen_button = true;
	export let full_history: any = null;
	export let gradio: Gradio<{
		change: never;
		error: string;
		input: never;
		edit: never;
		drag: never;
		apply: never;
		upload: never;
		clear: never;
		select: SelectData;
		share: ShareData;
		clear_status: LoadingStatus;
	}>;
	export let border_region = 0;
	export let webcam_options: WebcamOptions;
	export let theme_mode: "dark" | "light";

	let editor_instance: InteractiveImageEditor;
	const patch_id = "wangp-image-editor-20260701-connection-loss-export-log-32";
	const connection_lost_at_key = "__wangp_gradio_connection_lost_at";
	const connection_lost_logged_key = "__wangp_gradio_connection_lost_logged";

	function mark_gradio_connection_lost(event?: any): void {
		if (event?.name === "AbortError") return;
		(window as any)[connection_lost_at_key] = Date.now();
		if (!(window as any)[connection_lost_logged_key]) {
			(window as any)[connection_lost_logged_key] = true;
			console.warn("WanGPImageEditor reconnect detected; get_value will return a full image/mask payload when needed.");
		}
	}

	function wrap_stream_errors(stream: any): any {
		if (!stream || stream.__wangp_connection_loss_patched) return stream;
		stream.__wangp_connection_loss_patched = true;
		try {
			stream.addEventListener("error", mark_gradio_connection_lost);
		} catch {}
		let onerror = stream.onerror;
		try {
			Object.defineProperty(stream, "onerror", {
				configurable: true,
				get() {
					return onerror;
				},
				set(handler) {
					onerror = typeof handler === "function"
						? function (this: any, event: any) {
								mark_gradio_connection_lost(event);
								return handler.call(this, event);
							}
						: handler;
				}
			});
			stream.onerror = onerror || mark_gradio_connection_lost;
		} catch {
			stream.onerror = function (this: any, event: any) {
				mark_gradio_connection_lost(event);
				return onerror?.call(this, event);
			};
		}
		return stream;
	}

	function install_gradio_connection_loss_patch(client: any): void {
		if (!client || client.__wangp_connection_loss_patch_installed || typeof client.stream !== "function") return;
		const stream = client.stream;
		client.stream = function (this: any, ...args: any[]) {
			return wrap_stream_errors(stream.apply(this, args));
		};
		client.__wangp_connection_loss_patch_installed = true;
		wrap_stream_errors(client.heartbeat_event);
		wrap_stream_errors(client.stream_instance);
	}

	if (typeof window !== "undefined") {
		(window as any).__WANGP_IMAGE_EDITOR_VENDOR_PATCH = {
			id: patch_id,
			cache_busted_component_id: true,
			crop_export_deferred: true,
			brush_reactivated_after_visibility: true,
			value_prop_is_source_of_truth: true,
			get_value_is_pure_read: true,
			clear_resets_history: true,
			empty_export_is_canonical_payload: true,
			output_none_forces_prop_sync: true,
			empty_frame_is_rendered: true,
			gpu_surface_released_when_hidden_or_empty: true,
			restores_surface_before_add_image: true,
			restores_hidden_value_surface: true,
			releases_hidden_value_surface_after_load: true,
			fit_uses_live_dimensions: true,
			refits_after_released_surface_restore: true,
			export_defers_surface_release: true,
			get_data_serializes_exports: true,
			clean_source_bypasses_webgl_export: true,
			clean_source_keeps_empty_mask: true,
			memory_value_transport: true,
			dirty_meta_sends_base_id: true,
			restores_released_value_on_show: false,
			preserves_editor_state_on_hidden_show: true,
			ignores_output_value_echo: true,
			cancels_stale_empty_value_sync: true,
			reuses_server_value_cache_ids: true,
			imported_layers_clear_transparent: true,
			imported_layers_use_logical_size: true,
			clears_mask_on_crop_resize: true,
			empty_mask_layer_keeps_active_layer: true,
			records_gradio_connection_loss_timestamp: true
		};
	}

	$: install_gradio_connection_loss_patch(gradio?.client);

	export async function get_value(): Promise<ImageBlobs | { id: string }> {
		return editor_instance.get_data();
	}

	let is_dragging: boolean;
	$: value && handle_change(value_is_output);
	const is_browser = typeof window !== "undefined";
	const raf = is_browser
		? window.requestAnimationFrame
		: (cb: (...args: any[]) => void) => cb();

	function wait_for_next_frame(): Promise<void> {
		return new Promise((resolve) => {
			raf(() => raf(() => resolve()));
		});
	}

	async function handle_change(from_output: boolean): Promise<void> {
		await wait_for_next_frame();
		if (from_output) return;

		if (
			value &&
			(value.background || value.layers?.length || value.composite)
		) {
			gradio.dispatch("change");
		}
	}

	function handle_save(): void {
		gradio.dispatch("apply");
	}

	function handle_history_change(): void {
		gradio.dispatch("change");
		if (!value_is_output) {
			gradio.dispatch("input");
			tick().then((_) => (value_is_output = false));
		}
	}

	afterUpdate(() => {
		value_is_output = false;
	});

	function handle_clear(): void {
		value = { background: null, layers: [], composite: null };
		full_history = null;
		gradio.dispatch("clear");
		gradio.dispatch("input");
		gradio.dispatch("change", value);
	}

	$: has_value = value?.background || value?.layers?.length || value?.composite;

	$: normalised_background = value?.background
		? new FileData(value.background)
		: null;
	$: normalised_composite = value?.composite
		? new FileData(value.composite)
		: null;
	$: normalised_layers =
		value?.layers?.map((layer) => new FileData(layer)) || [];
</script>

{#if !interactive}
	<Block
		{visible}
		variant={"solid"}
		border_mode={is_dragging ? "focus" : "base"}
		padding={false}
		{elem_id}
		{elem_classes}
		{height}
		{width}
		allow_overflow={true}
		overflow_behavior="visible"
		{container}
		{scale}
		{min_width}
	>
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
		<StaticImage
			on:select={({ detail }) => gradio.dispatch("select", detail)}
			on:share={({ detail }) => gradio.dispatch("share", detail)}
			on:error={({ detail }) => gradio.dispatch("error", detail)}
			value={value?.composite || null}
			{label}
			{show_label}
			{show_download_button}
			selectable={_selectable}
			{show_share_button}
			i18n={gradio.i18n}
			{show_fullscreen_button}
		/>
	</Block>
{:else}
	<Block
		{visible}
		variant={has_value ? "solid" : "dashed"}
		border_mode={is_dragging ? "focus" : "base"}
		padding={false}
		{elem_id}
		{elem_classes}
		{height}
		{width}
		allow_overflow={true}
		overflow_behavior="visible"
		{container}
		{scale}
		{min_width}
	>
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>

		<InteractiveImageEditor
			{border_region}
			on:history={(e) => (full_history = e.detail)}
			bind:is_dragging
			{canvas_size}
			on:change={() => handle_history_change()}
			layers={normalised_layers}
			composite={normalised_composite}
			background={normalised_background}
			value_id={value?.id || null}
			bind:this={editor_instance}
			{root}
			{sources}
			{label}
			{show_label}
			{fixed_canvas}
			on:save={(e) => handle_save()}
			on:edit={() => gradio.dispatch("edit")}
			on:clear={handle_clear}
			on:drag={({ detail }) => (is_dragging = detail)}
			on:upload={() => gradio.dispatch("upload")}
			on:share={({ detail }) => gradio.dispatch("share", detail)}
			on:error={({ detail }) => {
				loading_status = loading_status || {};
				loading_status.status = "error";
				gradio.dispatch("error", detail);
			}}
			on:error
			{brush}
			{eraser}
			changeable={attached_events.includes("apply")}
			realtime={attached_events.includes("change") ||
				attached_events.includes("input")}
			i18n={gradio.i18n}
			{transforms}
			layer_options={layers}
			accept_blobs={server.accept_blobs}
			upload={(...args) => gradio.client.upload(...args)}
			{placeholder}
			{full_history}
			{webcam_options}
			{show_download_button}
			{theme_mode}
			{value_is_output}
			on:download_error={(e) => gradio.dispatch("error", e.detail)}
		></InteractiveImageEditor>
	</Block>
{/if}
