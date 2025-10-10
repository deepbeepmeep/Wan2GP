import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import re
from PIL import Image
import gc

from .gallery_utils import get_thumbnails_in_batch_windows

class GalleryPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "File Gallery"
        self.version = "1.0.0"
        self.description = "Adds a Gallery tab that allows you to view metadata of all files in your output folders, join video frames with a single click, and more"

    def setup_ui(self):
        self.add_tab(
            tab_id="gallery_tab",
            label="Gallery",
            component_constructor=self.create_gallery_ui,
            position=1
        )
        self.request_global("server_config")
        self.request_global("has_video_file_extension")
        self.request_global("has_image_file_extension")
        self.request_global("get_settings_from_file")
        self.request_global("get_video_info")
        self.request_global("extract_audio_tracks")
        self.request_global("get_file_creation_date")
        self.request_global("get_video_frame")
        self.request_global("are_model_types_compatible")
        self.request_global("get_model_def")
        self.request_global("get_default_settings")
        self.request_global("add_to_sequence")
        self.request_global("set_model_settings")
        self.request_global("generate_dropdown_model_list")
        self.request_global("get_unique_id")
        self.request_global("args")
        self.request_component("main")
        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("model_family")
        self.request_component("model_choice")
        self.request_component("refresh_form_trigger")
        self.request_component("image_start")
        self.request_component("image_end")
        self.request_component("image_prompt_type")
        self.request_component("image_start_row")
        self.request_component("image_end_row")
        self.request_component("image_prompt_type_radio")
        self.request_component("image_prompt_type_endcheckbox")
        self.request_component("plugin_data")
        self.register_data_hook("before_metadata_save", self.add_merge_info_to_metadata)
        
    def create_gallery_ui(self):
        css = """
            #gallery-layout { display: flex; gap: 16px; min-height: 75vh; }
            #gallery-container { flex: 3; overflow-y: auto; border: 1px solid #e0e0e0; padding: 10px; background-color: #f9f9f9; border-radius: 8px; }
            #metadata-panel-container { flex: 1; overflow-y: auto; border: 1px solid #e0e0e0; padding: 15px; background-color: #ffffff; border-radius: 8px; }
            .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 16px; }
            .gallery-item { position: relative; cursor: pointer; border: 2px solid transparent; border-radius: 8px; overflow: hidden; aspect-ratio: 4 / 5; display: flex; flex-direction: column; background-color: #ffffff; transition: all 0.2s ease-in-out; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .gallery-item:hover { border-color: #a0a0a0; transform: translateY(-2px); }
            .gallery-item.selected { border-color: var(--primary-500); box-shadow: 0 0 0 3px var(--primary-200); }
            .gallery-item-thumbnail { flex-grow: 1; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; overflow: hidden; }
            .gallery-item-thumbnail img, .gallery-item-thumbnail video { width: 100%; height: 100%; object-fit: contain; }
            .gallery-item-name { padding: 4px 8px; font-size: 12px; text-align: center; background-color: #f8f9fa; white-space: normal; word-break: break-word; border-top: 1px solid #ddd; min-height: 3.2em; display: flex; align-items: center; justify-content: center; }
            .metadata-content { font-family: monospace; font-size: 13px; line-height: 1.6; word-wrap: break-word; }
            .metadata-content b { color: var(--primary-500); }
            .metadata-content hr { border: 0; border-top: 1px solid #e0e0e0; margin: 8px 0; }
            .metadata-content .placeholder { color: #999; text-align: center; margin-top: 20px; font-style: italic; }
            #video_info, #video_info TR, #video_info TD { background-color: transparent; color: inherit; padding: 4px; border:0px !important; font-size:12px; }
        """
        js = """
            function() {
                window.selectGalleryItem = function(event, element) {
                    const gallery = element.closest('.gallery-grid');
                    const selectedFilesInput = document.querySelector('#selected-files-backend textarea');
                    if (!gallery || !selectedFilesInput) { return; }
                    if (!event.ctrlKey && !event.metaKey) {
                        gallery.querySelectorAll('.gallery-item.selected').forEach(el => {
                            if (el !== element) el.classList.remove('selected');
                        });
                    }
                    element.classList.toggle('selected');
                    const selectedItems = Array.from(gallery.querySelectorAll('.gallery-item.selected'));
                    const selectedPaths = selectedItems.map(el => el.dataset.path);
                    selectedFilesInput.value = selectedPaths.join(',');
                    selectedFilesInput.dispatchEvent(new Event('input', { bubbles: true }));
                };
                
                function setupVideoFrameSeeker(containerId, sliderId, fps) {
                    const container = document.querySelector(`#${containerId}`);
                    const sliderContainer = document.querySelector(`#${sliderId}`);
                    if (!container || !sliderContainer) return;

                    const video = container.querySelector('video');
                    if (!video) return;

                    let frameTime = (fps > 0) ? 1 / fps : 0;
                    let isSeekingFromSlider = false;
                    let debounceTimer;

                    function updateVideoToFrame(frameNumber) {
                        if (frameTime === 0 || !isFinite(video.duration)) return;
                        const maxFrame = Math.floor(video.duration * fps);
                        const clampedFrame = Math.max(1, Math.min(frameNumber, maxFrame || 1));
                        const targetTime = (clampedFrame - 1) * frameTime;
                        if (Math.abs(video.currentTime - targetTime) > frameTime / 2) {
                            video.currentTime = targetTime;
                        }
                    }
                    
                    video.addEventListener('loadedmetadata', () => {
                        const sliderInput = sliderContainer.querySelector('input[type="range"]');
                        if (sliderInput) setTimeout(() => updateVideoToFrame(parseInt(sliderInput.value, 10)), 100);
                    }, { once: true });

                    video.addEventListener('timeupdate', () => {
                        const sliderInput = sliderContainer.querySelector('input[type="range"]');
                        if (!isSeekingFromSlider && frameTime > 0 && sliderInput) {
                            const currentFrame = Math.round(video.currentTime / frameTime) + 1;
                            if (sliderInput.value != currentFrame) {
                                sliderInput.value = currentFrame;
                                const numberInput = sliderContainer.querySelector('input[type="number"]');
                                if (numberInput) numberInput.value = currentFrame;
                            }
                        }
                    });

                    const handleSliderInput = () => {
                        const sliderInput = sliderContainer.querySelector('input[type="range"]');
                        if (sliderInput) {
                            isSeekingFromSlider = true;
                            const frameNumber = parseInt(sliderInput.value, 10);
                            clearTimeout(debounceTimer);
                            debounceTimer = setTimeout(() => {
                                updateVideoToFrame(frameNumber);
                            }, 50);
                        }
                    };
                    
                    const handleInteractionEnd = () => {
                        setTimeout(() => { isSeekingFromSlider = false; }, 150);
                    };
                    
                    sliderContainer.addEventListener('input', handleSliderInput);
                    sliderContainer.addEventListener('mouseup', handleInteractionEnd);
                    sliderContainer.addEventListener('touchend', handleInteractionEnd);
                }

                const observer = new MutationObserver((mutationsList) => {
                    for(const mutation of mutationsList) {
                        if (mutation.type === 'childList') {
                            document.querySelectorAll('.video-joiner-player').forEach(player => {
                                if (!player.dataset.initialized) {
                                    const containerId = player.id;
                                    const { sliderId, fps } = player.dataset;
                                    if (containerId && sliderId && !isNaN(parseFloat(fps))) {
                                        setupVideoFrameSeeker(containerId, sliderId, parseFloat(fps));
                                        player.dataset.initialized = 'true';
                                    }
                                }
                            });
                        }
                    }
                });
                observer.observe(document.body, { childList: true, subtree: true });
            }
        """
        with gr.Blocks() as gallery_blocks:
            gr.HTML(value=f"<style>{css}</style>")
            gallery_blocks.load(fn=None, js=js)
            with gr.Column():
                with gr.Row():
                    self.refresh_gallery_files_btn = gr.Button("Refresh Files")
                with gr.Row(elem_id="gallery-layout"):
                    self.gallery_html_output = gr.HTML(value="<div class='gallery-grid'><p class='placeholder'>Click 'Refresh Files' to load gallery.</p></div>", elem_id="gallery-container")
                    with gr.Column(elem_id="metadata-panel-container"):
                        self.send_to_generator_settings_btn = gr.Button("Use Settings in Generator", interactive=False, visible=False)
                        self.join_videos_btn = gr.Button("Join 2 Selected Videos", interactive=False, visible=False)
                        self.recreate_join_btn = gr.Button("Recreate Join From This Video", visible=False, interactive=False)
                        with gr.Row(visible=False) as self.frame_preview_row:
                            self.first_frame_preview = gr.Image(label="First Frame", interactive=False, height=150)
                            self.last_frame_preview = gr.Image(label="Last Frame", interactive=False, height=150)
                        self.metadata_panel_output = gr.HTML(value="<div class='metadata-content'><p class='placeholder'>Select a file to view its metadata.</p></div>")
                        with gr.Column(visible=False) as self.merge_info_display:
                            gr.Markdown("--- \n #### Merged From")
                            with gr.Row():
                                with gr.Column():
                                    self.merge_source1_prompt = gr.Markdown(elem_classes="metadata-content")
                                    self.merge_source1_image = gr.Image(interactive=False, show_label=False)
                                with gr.Column():
                                    self.merge_source2_prompt = gr.Markdown(elem_classes="metadata-content")
                                    self.merge_source2_image = gr.Image(interactive=False, show_label=False)
                        with gr.Column(visible=False) as self.join_interface:
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### Video 1 (Provides End Frame)")
                                    self.video1_preview = gr.HTML(label="Video 1 Preview")
                                    self.video1_frame_slider = gr.Slider(label="Frame Number", minimum=1, maximum=100, step=1, interactive=True, elem_id="video1_frame_slider")
                                    self.video1_path = gr.Text(visible=False)
                                    self.video1_info = gr.HTML(label="Video 1 Info")
                                with gr.Column():
                                    gr.Markdown("#### Video 2 (Provides Start Frame)")
                                    self.video2_preview = gr.HTML(label="Video 2 Preview")
                                    self.video2_frame_slider = gr.Slider(label="Frame Number", minimum=1, maximum=100, step=1, interactive=True, elem_id="video2_frame_slider")
                                    self.video2_path = gr.Text(visible=False)
                                    self.video2_info = gr.HTML(label="Video 2 Info")
                            with gr.Row():
                                self.send_to_generator_btn = gr.Button("Send Frames to Generator", variant="primary")
                                self.cancel_join_btn = gr.Button("Cancel")
                self.selected_files_for_backend = gr.Text(label="Selected Files", visible=False, elem_id="selected-files-backend")
                self.path_for_settings_loader = gr.Text(label="Path for Settings Loader", visible=False)
        return gallery_blocks

    def list_output_files_as_html(self, current_state):
        save_path = self.server_config.get("save_path", "outputs")
        image_save_path = self.server_config.get("image_save_path", "outputs")
        paths = {save_path, image_save_path}
        all_files = []
        for path in paths:
            if os.path.isdir(path) and os.path.exists(path):
                valid_files = [f for f in os.listdir(path) if self.has_video_file_extension(f) or self.has_image_file_extension(f)]
                all_files.extend([os.path.join(path, f) for f in valid_files])
        all_files.sort(key=os.path.getctime, reverse=True)
        thumbnails_dict = get_thumbnails_in_batch_windows([os.path.abspath(f) for f in all_files])

        items_html = ""
        for f in all_files:
            abs_f = os.path.abspath(f)
            safe_path = f.replace("'", "\\'")
            basename = os.path.basename(f)
            display_name = basename
            match = re.search(r'_seed\d+_(.+)\.(mp4|jpg|jpeg|png|webp)$', basename, re.IGNORECASE)
            if match: display_name = match.group(1)
            is_video = self.has_video_file_extension(f)
            base64_thumb = thumbnails_dict.get(abs_f)
            thumbnail_html = f'<img src="data:image/jpeg;base64,{base64_thumb}" alt="thumb">' if base64_thumb else \
                             f'<video muted preload="metadata" src="/gradio_api/file={f}#t=0.5"></video>' if is_video else \
                             f'<img src="/gradio_api/file={f}" alt="thumb">'
            items_html += f"""<div class="gallery-item" data-path='{safe_path}' onclick="selectGalleryItem(event, this)">
                <div class="gallery-item-thumbnail">{thumbnail_html}</div><div class="gallery-item-name" title="{basename}">{display_name}</div></div>"""
        
        full_html = f"<div class='gallery-grid'>{items_html}</div>"
        clear_metadata_html = "<div class='metadata-content'><p class='placeholder'>Select a file to view its metadata.</p></div>"
        return {
            self.gallery_html_output: full_html,
            self.selected_files_for_backend: "",
            self.metadata_panel_output: clear_metadata_html,
            self.join_videos_btn: gr.Button(visible=False),
            self.recreate_join_btn: gr.Button(visible=False),
            self.send_to_generator_settings_btn: gr.Button(visible=False),
            self.frame_preview_row: gr.Row(visible=False),
            self.first_frame_preview: gr.Image(value=None),
            self.last_frame_preview: gr.Image(value=None),
            self.join_interface: gr.Column(visible=False),
            self.merge_info_display: gr.Column(visible=False)
        }

    def add_merge_info_to_metadata(self, configs, plugin_data, **kwargs):
        if plugin_data and "merge_info" in plugin_data:
            configs["merge_info"] = plugin_data["merge_info"]
        return configs

    def get_video_info_html(self, current_state, file_path):
        configs, _ = self.get_settings_from_file(current_state, file_path, False, False, False)
        values, labels = [os.path.basename(file_path)], ["File Name"]
        misc_values, misc_labels, pp_values, pp_labels = [], [], [], []
        is_image = self.has_image_file_extension(file_path)
        if is_image: width, height = Image.open(file_path).size; frames_count = fps = 1; nb_audio_tracks = 0
        else: fps, width, height, frames_count = self.get_video_info(file_path); nb_audio_tracks = self.extract_audio_tracks(file_path, query_only=True)
        if configs:
            video_model_name = configs.get("type", "Unknown model").split(" - ")[-1]
            misc_values.append(video_model_name); misc_labels.append("Model")
            if configs.get("temporal_upsampling"): pp_values.append(configs["temporal_upsampling"]); pp_labels.append("Upsampling")
            if configs.get("film_grain_intensity", 0) > 0: pp_values.append(f"Intensity={configs['film_grain_intensity']}, Saturation={configs['film_grain_saturation']}"); pp_labels.append("Film Grain")
        if configs is None or "seed" not in configs:
            values.extend(misc_values); labels.extend(misc_labels)
            creation_date = str(self.get_file_creation_date(file_path)); values.append(creation_date[:creation_date.rfind('.')]); labels.append("Creation Date")
            if is_image: values.append(f"{width}x{height}"); labels.append("Resolution")
            else: values.extend([f"{width}x{height}", f"{frames_count} frames (duration={frames_count/fps:.1f}s, fps={round(fps)})"]); labels.extend(["Resolution", "Frames"])
            if nb_audio_tracks > 0: values.append(nb_audio_tracks); labels.append("Nb Audio Tracks")
            values.extend(pp_values); labels.extend(pp_labels)
        else:
            values.extend(misc_values); labels.extend(misc_labels); values.append(configs.get("prompt", "")[:1024]); labels.append("Text Prompt")
            values.extend([f"{configs.get('resolution', '')} (real: {width}x{height})", configs.get('video_length', 0), configs.get('seed', -1), configs.get('guidance_scale', 'N/A'), configs.get('num_inference_steps', 'N/A')])
            labels.extend(["Resolution", "Video Length", "Seed", "Guidance (CFG)", "Num Inference steps"])
        rows = [f"<TR><TD style='text-align: right; vertical-align: top; width:1%; white-space:nowrap;'>{l}</TD><TD><B>{v}</B></TD></TR>" for l, v in zip(labels, values) if v is not None]
        return f"<TABLE ID=video_info WIDTH=100%>{''.join(rows)}</TABLE>"

    def update_metadata_panel_and_buttons(self, selection_str, current_state):
        file_paths = selection_str.split(',') if selection_str else []
        video_files = [f for f in file_paths if self.has_video_file_extension(f)]
        
        updates = {
            self.join_videos_btn: gr.Button(visible=len(video_files) == 2 and len(file_paths) == 2, interactive=True),
            self.recreate_join_btn: gr.Button(visible=False),
            self.send_to_generator_settings_btn: gr.Button(visible=False),
            self.path_for_settings_loader: "",
            self.frame_preview_row: gr.Row(visible=False),
            self.first_frame_preview: gr.Image(value=None),
            self.last_frame_preview: gr.Image(value=None, visible=True),
            self.join_interface: gr.Column(visible=False),
            self.merge_info_display: gr.Column(visible=False),
            self.metadata_panel_output: gr.HTML(value="<div class='metadata-content'><p class='placeholder'>Select a file to view its metadata.</p></div>", visible=True),
            self.merge_source1_prompt: gr.Markdown(value=""),
            self.merge_source1_image: gr.Image(value=None),
            self.merge_source2_prompt: gr.Markdown(value=""),
            self.merge_source2_image: gr.Image(value=None)
        }

        if len(file_paths) == 1:
            file_path = file_paths[0]
            updates[self.path_for_settings_loader] = file_path
            configs, _ = self.get_settings_from_file(current_state, file_path, False, False, False)
            updates[self.send_to_generator_settings_btn] = gr.Button(visible=True, interactive=bool(configs))
            updates[self.metadata_panel_output] = gr.HTML(value=self.get_video_info_html(current_state, file_path), visible=True)

            if configs and "merge_info" in configs:
                merge_info = configs["merge_info"]
                save_path = self.server_config.get("save_path", "outputs")
                image_save_path = self.server_config.get("image_save_path", "outputs")
                vid1_rel, vid2_rel = merge_info['source_video_1']['path'], merge_info['source_video_2']['path']
                vid1_abs = next((p for p in [os.path.join(save_path, vid1_rel), os.path.join(image_save_path, vid1_rel)] if os.path.exists(p)), None)
                vid2_abs = next((p for p in [os.path.join(save_path, vid2_rel), os.path.join(image_save_path, vid2_rel)] if os.path.exists(p)), None)

                if vid1_abs and vid2_abs:
                    updates[self.recreate_join_btn] = gr.Button(visible=True, interactive=True)
                    updates[self.merge_info_display] = gr.Column(visible=True)
                    f1_num, f2_num = merge_info['source_video_1']['frame_used'], merge_info['source_video_2']['frame_used']
                    f1_pil, f2_pil = self.get_video_frame(vid1_abs, f1_num - 1, return_PIL=True), self.get_video_frame(vid2_abs, f2_num - 1, return_PIL=True)
                    c1, _ = self.get_settings_from_file(current_state, vid1_abs, False, False, False)
                    c2, _ = self.get_settings_from_file(current_state, vid2_abs, False, False, False)
                    p1, p2 = (c1.get('prompt', 'N/A') if c1 else 'N/A'), (c2.get('prompt', 'N/A') if c2 else 'N/A')
                    
                    updates[self.merge_source1_prompt] = f"<b>{vid1_rel} (Frame {f1_num})</b><br>{p1[:100] + '...' if len(p1) > 100 else p1}"
                    updates[self.merge_source1_image] = f1_pil
                    updates[self.merge_source2_prompt] = f"<b>{vid2_rel} (Frame {f2_num})</b><br>{p2[:100] + '...' if len(p2) > 100 else p2}"
                    updates[self.merge_source2_image] = f2_pil
                else:
                    updates[self.frame_preview_row] = gr.Row(visible=True) # Fallback to default preview
            
            else: # Not a merge file, show normal preview
                updates[self.frame_preview_row] = gr.Row(visible=True)
                if self.has_video_file_extension(file_path):
                    first_frame_pil = self.get_video_frame(file_path, 0, return_PIL=True)
                    _, _, _, frame_count = self.get_video_info(file_path)
                    last_frame_pil = self.get_video_frame(file_path, frame_count - 1, return_PIL=True) if frame_count > 1 else first_frame_pil
                    updates[self.first_frame_preview], updates[self.last_frame_preview] = gr.Image(value=first_frame_pil, label="First Frame"), gr.Image(value=last_frame_pil, visible=True)
                elif self.has_image_file_extension(file_path):
                    updates[self.first_frame_preview], updates[self.last_frame_preview] = gr.Image(value=Image.open(file_path), label="Image Preview"), gr.Image(visible=False)

        elif len(file_paths) > 1:
            updates[self.metadata_panel_output] = gr.HTML(value=f"<div class='metadata-content'><p>{len(file_paths)} items selected.</p></div>", visible=True)
        
        return updates

    def load_settings_and_frames_from_gallery(self, current_state, file_path):
        if not file_path: gr.Warning("No file selected."); return gr.update(), gr.update(), gr.update(), gr.update()
        configs, _ = self.get_settings_from_file(current_state, file_path, True, True, True)
        if not configs: gr.Info("No settings found."); return gr.update(), gr.update(), gr.update(), gr.update()
        current_model_type = current_state["model_type"]
        target_model_type = configs.get("model_type", current_model_type)
        if self.are_model_types_compatible(target_model_type, current_model_type): target_model_type = current_model_type
        configs["model_type"] = target_model_type
        first_frame, last_frame = None, None
        if self.has_video_file_extension(file_path):
            first_frame = self.get_video_frame(file_path, 0, return_PIL=True)
            _, _, _, frame_count = self.get_video_info(file_path)
            if frame_count > 1: last_frame = self.get_video_frame(file_path, frame_count - 1, return_PIL=True)
        elif self.has_image_file_extension(file_path): first_frame = Image.open(file_path)
        allowed_prompts = self.get_model_def(target_model_type).get("image_prompt_types_allowed", "")
        configs = {**self.get_default_settings(target_model_type), **configs}
        if first_frame:
            updated_prompts = self.add_to_sequence(configs.get("image_prompt_type", ""), "S") if "S" in allowed_prompts else configs.get("image_prompt_type", "")
            configs["image_start"] = [(first_frame, "First Frame")]
            if last_frame and "E" in allowed_prompts:
                updated_prompts = self.add_to_sequence(updated_prompts, "E")
                configs["image_end"] = [(last_frame, "Last Frame")]
            configs["image_prompt_type"] = updated_prompts
        self.set_model_settings(current_state, target_model_type, configs)
        gr.Info(f"Settings from '{os.path.basename(file_path)}' sent to generator.")
        mf, mc = (gr.update(), gr.update()) if target_model_type == current_model_type else self.generate_dropdown_model_list(target_model_type)
        return mf, mc, gr.update(selected="video_gen"), self.get_unique_id()

    def show_join_interface(self, selection_str, current_state):
        video_files = [f for f in selection_str.split(',') if self.has_video_file_extension(f)] if selection_str else []
        if len(video_files) != 2: gr.Warning("Please select exactly two videos."); return {}
        return self.recreate_join_interface(video_files, current_state)
    
    def recreate_join_interface(self, file_info, current_state):
        if isinstance(file_info, str): # Called from single merged video
            configs, _ = self.get_settings_from_file(current_state, file_info, False, False, False)
            if not (configs and "merge_info" in configs):
                gr.Warning("Could not find merge info in the selected file.")
                return {}
            merge_info = configs["merge_info"]
            save_path = self.server_config.get("save_path", "outputs")
            image_save_path = self.server_config.get("image_save_path", "outputs")
            vid1_rel, vid2_rel = merge_info['source_video_1']['path'], merge_info['source_video_2']['path']
            vid1_abs = next((p for p in [os.path.join(save_path, vid1_rel), os.path.join(image_save_path, vid1_rel)] if os.path.exists(p)), None)
            vid2_abs = next((p for p in [os.path.join(save_path, vid2_rel), os.path.join(image_save_path, vid2_rel)] if os.path.exists(p)), None)
            if not (vid1_abs and vid2_abs):
                gr.Warning("One or both source videos for merging could not be found.")
                return {}
            vid1_path, vid2_path = vid1_abs, vid2_abs
            frame1_num, frame2_num = merge_info['source_video_1']['frame_used'], merge_info['source_video_2']['frame_used']
        elif isinstance(file_info, list) and len(file_info) == 2: # Called from 2 selected videos
            vid1_path, vid2_path = file_info[0], file_info[1]
            _, _, _, v1_frames = self.get_video_info(vid1_path)
            frame1_num, frame2_num = v1_frames, 1
        else:
            return {}
        
        server_port_val = int(self.args.server_port) if self.args.server_port != 0 else 7860
        server_name_val = self.args.server_name if self.args.server_name and self.args.server_name != "0.0.0.0" else "127.0.0.1"
        base_url = f"http://{server_name_val}:{server_port_val}"
        v1_fps, _, _, v1_frames = self.get_video_info(vid1_path)
        v2_fps, _, _, v2_frames = self.get_video_info(vid2_path)
        
        def create_player(container_id, slider_id, path, fps): 
            return f'<div id="{container_id}" class="video-joiner-player" data-slider-id="{slider_id}" data-fps="{fps}"><video src="{base_url}/gradio_api/file={path}" style="width:100%;" controls muted preload="metadata"></video></div>'

        player1_html = create_player("video1_player_container", "video1_frame_slider", vid1_path, v1_fps)
        player2_html = create_player("video2_player_container", "video2_frame_slider", vid2_path, v2_fps)
        
        return { 
            self.join_interface: gr.Column(visible=True), 
            self.frame_preview_row: gr.Row(visible=False), 
            self.merge_info_display: gr.Column(visible=False),
            self.metadata_panel_output: gr.HTML(visible=False),
            self.send_to_generator_settings_btn: gr.Button(visible=False),
            self.join_videos_btn: gr.Button(visible=False),
            self.recreate_join_btn: gr.Button(visible=False),
            self.video1_preview: gr.HTML(value=player1_html), 
            self.video2_preview: gr.HTML(value=player2_html), 
            self.video1_path: vid1_path, 
            self.video2_path: vid2_path, 
            self.video1_frame_slider: gr.Slider(maximum=v1_frames, value=frame1_num), 
            self.video2_frame_slider: gr.Slider(maximum=v2_frames, value=frame2_num), 
            self.video1_info: self.get_video_info_html(current_state, vid1_path), 
            self.video2_info: self.get_video_info_html(current_state, vid2_path) 
        }

    def send_selected_frames_to_generator(self, vid1_path, frame1_num, vid2_path, frame2_num, current_image_prompt_type):
        frame1 = self.get_video_frame(vid1_path, int(frame1_num) - 1, return_PIL=True)
        frame2 = self.get_video_frame(vid2_path, int(frame2_num) - 1, return_PIL=True)
        gr.Info("Frames sent to Video Generator.")
        updated_image_prompt_type = self.add_to_sequence(current_image_prompt_type, "SE")
        merge_info = {
            "source_video_1": {
                "path": os.path.basename(vid1_path),
                "frame_used": int(frame1_num)
            },
            "source_video_2": {
                "path": os.path.basename(vid2_path),
                "frame_used": int(frame2_num)
            }
        }
        return {
            self.image_start: [(frame1, "Start Frame")],
            self.image_end: [(frame2, "End Frame")],
            self.main_tabs: gr.Tabs(selected="video_gen"),
            self.join_interface: gr.Column(visible=False),
            self.image_prompt_type: updated_image_prompt_type,
            self.image_start_row: gr.Row(visible=True),
            self.image_end_row: gr.Row(visible=True),
            self.image_prompt_type_radio: gr.Radio(value="S"),
            self.image_prompt_type_endcheckbox: gr.Checkbox(value=True),
            self.plugin_data: {"merge_info": merge_info}
        }

    def post_ui_setup(self, components: dict):
        self.main.load(fn=self.list_output_files_as_html, inputs=[self.state], outputs=[
            self.gallery_html_output, self.selected_files_for_backend, self.metadata_panel_output, 
            self.join_videos_btn, self.send_to_generator_settings_btn, self.frame_preview_row, 
            self.first_frame_preview, self.last_frame_preview, self.join_interface, 
            self.recreate_join_btn, self.merge_info_display
        ])
        self.refresh_gallery_files_btn.click(fn=self.list_output_files_as_html, inputs=[self.state], outputs=[
            self.gallery_html_output, self.selected_files_for_backend, self.metadata_panel_output, 
            self.join_videos_btn, self.send_to_generator_settings_btn, self.frame_preview_row, 
            self.first_frame_preview, self.last_frame_preview, self.join_interface,
            self.recreate_join_btn, self.merge_info_display
        ])
        self.selected_files_for_backend.change(fn=self.update_metadata_panel_and_buttons, 
            inputs=[self.selected_files_for_backend, self.state], 
            outputs=[
                self.join_videos_btn, self.send_to_generator_settings_btn, self.metadata_panel_output, 
                self.path_for_settings_loader, self.frame_preview_row, self.first_frame_preview, 
                self.last_frame_preview, self.join_interface, self.recreate_join_btn, self.merge_info_display,
                self.merge_source1_prompt, self.merge_source1_image, self.merge_source2_prompt, self.merge_source2_image
            ], show_progress="hidden"
        )
        self.join_videos_btn.click(fn=self.show_join_interface, inputs=[self.selected_files_for_backend, self.state], outputs=[
            self.join_interface, self.frame_preview_row, self.merge_info_display, self.metadata_panel_output,
            self.send_to_generator_settings_btn, self.join_videos_btn, self.recreate_join_btn,
            self.video1_preview, self.video2_preview,
            self.video1_path, self.video2_path, self.video1_frame_slider, self.video2_frame_slider,
            self.video1_info, self.video2_info
        ])
        self.recreate_join_btn.click(
            fn=self.recreate_join_interface, inputs=[self.path_for_settings_loader, self.state], outputs=[
            self.join_interface, self.frame_preview_row, self.merge_info_display, self.metadata_panel_output,
            self.send_to_generator_settings_btn, self.join_videos_btn, self.recreate_join_btn,
            self.video1_preview, self.video2_preview,
            self.video1_path, self.video2_path, self.video1_frame_slider, self.video2_frame_slider,
            self.video1_info, self.video2_info
        ])
        self.send_to_generator_settings_btn.click(fn=self.load_settings_and_frames_from_gallery, inputs=[self.state, self.path_for_settings_loader], outputs=[self.model_family, self.model_choice, self.main_tabs, self.refresh_form_trigger], show_progress="hidden")
        self.send_to_generator_btn.click(fn=self.send_selected_frames_to_generator, inputs=[self.video1_path, self.video1_frame_slider, self.video2_path, self.video2_frame_slider, self.image_prompt_type], outputs=[self.image_start, self.image_end, self.main_tabs, self.join_interface, self.image_prompt_type, self.image_start_row, self.image_end_row, self.image_prompt_type_radio, self.image_prompt_type_endcheckbox, self.plugin_data])
        self.cancel_join_btn.click(
            fn=self.update_metadata_panel_and_buttons,
            inputs=[self.selected_files_for_backend, self.state],
            outputs=[
                self.join_videos_btn, self.send_to_generator_settings_btn, self.metadata_panel_output,
                self.path_for_settings_loader, self.frame_preview_row, self.first_frame_preview,
                self.last_frame_preview, self.join_interface, self.recreate_join_btn, self.merge_info_display,
                self.merge_source1_prompt, self.merge_source1_image, self.merge_source2_prompt, self.merge_source2_image
            ]
        )
        
        return {}