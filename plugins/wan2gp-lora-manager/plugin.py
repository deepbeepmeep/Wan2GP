import gradio as gr
import os
import json
import hashlib
import urllib.request
import time
from datetime import datetime
from shared.utils.plugins import WAN2GPPlugin

class LoraManagerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "LoRA Manager"
        self.version = "1.1.0"
        self.description = "Multi-LoRA management with rich CivitAI integration."
        self.lora_root = "loras" 

        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.plugin_dir, "lora_db.json")
        self.lora_metadata = {}

    def setup_ui(self):
        self.request_global("get_lora_dir")
        self.request_global("get_state_model_type")
        self.request_global("model_types") 
        self.request_global("get_model_name") 

        self.request_component("state")
        self.request_component("prompt") 
        self.request_component("loras_choices")
        self.request_component("main_tabs")

        self.load_json_db()
        self.on_tab_outputs = [] 

        self.add_tab(
            tab_id="lora_manager_tab",
            label="LoRA Manager",
            component_constructor=self.create_manager_ui,
            position=2
        )

    def generate_hash(self, file_path):
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def fetch_civitai_data(self, file_path):
        try:
            file_hash = self.generate_hash(file_path)
            
            url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Wan2GP-Plugin'})
            
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if 'error' in data:
                return None, f"CivitAI Error: {data.get('error')}"
                
            return data, None
        except Exception as e:
            return None, str(e)

    def get_sidecar_json_path(self, lora_path):
        return os.path.splitext(lora_path)[0] + ".json"

    def format_date(self, date_str):
        if not date_str: return "N/A"
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except:
            return date_str

    def _fetch_and_process_single_lora(self, full_path, key):
        data, err = self.fetch_civitai_data(full_path)
        if err:
            return False, err

        dest = self.get_sidecar_json_path(full_path)
        try:
            with open(dest, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            return False, f"JSON save failed: {e}"

        trained_words = data.get('trainedWords', [])
        prompt_updated = False
        
        current_prompt = self.lora_metadata.get(key, {}).get("prompt", "")
        
        if trained_words and not current_prompt:
            new_prompt = ", ".join(trained_words)
            
            if key not in self.lora_metadata:
                self.lora_metadata[key] = {}
            
            self.lora_metadata[key]["prompt"] = new_prompt
            
            try:
                with open(self.db_path, 'w', encoding='utf-8') as f:
                    json.dump(self.lora_metadata, f, indent=4)
                prompt_updated = True
            except Exception as e:
                return True, f"Metadata updated, but DB save failed: {e}"

        msg = "Metadata updated."
        if prompt_updated:
            msg += " Default prompt set from triggers."
            
        return True, msg

    def batch_update_metadata(self, state, category, current_files, progress=gr.Progress()):
        if not current_files:
            gr.Warning("No files to update.")
            return gr.update(), gr.update()

        self.lora_root = self.discover_lora_root(state)
        
        updated_count = 0
        error_count = 0
        
        for i, item_name in enumerate(progress.tqdm(current_files, desc="Updating Metadata")):
            full_path = ""
            key = ""
            
            if category == "All LoRAs":
                full_path = os.path.join(self.lora_root, item_name)
                key = item_name.replace("\\", "/")
            else:
                full_path = os.path.join(self.lora_root, category, item_name)
                key = os.path.join(category, item_name).replace("\\", "/")

            if os.path.exists(full_path):
                success, msg = self._fetch_and_process_single_lora(full_path, key)
                if success:
                    updated_count += 1
                else:
                    print(f"Failed to update {item_name}: {msg}")
                    error_count += 1

            time.sleep(0.2)

        gr.Info(f"Batch Update Complete. Updated: {updated_count}, Failed/Skipped: {error_count}")
        return self.refresh_trigger.value + 1

    def create_manager_ui(self):
        self.is_initialized = gr.State(False)
        self.refresh_trigger = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Library")
                
                self.category_dropdown = gr.Dropdown(
                    label="Category",
                    choices=[],
                    value=None,
                    interactive=True
                )
                
                self.lora_list = gr.CheckboxGroup(
                    choices=[],
                    label="Available LoRAs",
                    info="Select LoRAs to view details and inject.",
                    interactive=True,
                    elem_classes="lora-checkbox-list"
                )
                
                with gr.Row():
                    self.refresh_btn = gr.Button("🔄 Refresh List", size="sm")
                    self.update_all_btn = gr.Button("🔄 Update All (Visible)", size="sm", variant="secondary")

            with gr.Column(scale=2):

                with gr.Group():
                    gr.Markdown("### 🛠️ Settings")
                    self.auto_fetch_chk = gr.Checkbox(
                        label="Auto-fetch metadata from CivitAI (if missing)",
                        value=False,
                        interactive=True
                    )

                @gr.render(inputs=[self.lora_list, self.refresh_trigger, self.auto_fetch_chk], triggers=[self.lora_list.change, self.refresh_trigger.change])
                def render_lora_cards(selected_items, _, auto_fetch):
                    if not selected_items:
                        gr.Markdown("### 📝 Details")
                        gr.Markdown("*Select a LoRA from the list on the left to view details and edit prompts.*")
                        return

                    gr.Markdown(f"### 📝 Selected Details ({len(selected_items)})")
                    
                    for lora_name in selected_items:
                        key, full_path = self.resolve_path(lora_name)
                        current_prompt = self.lora_metadata.get(key, {}).get("prompt", "")

                        json_path = self.get_sidecar_json_path(full_path)
                        civitai_data = None

                        if not os.path.exists(json_path) and auto_fetch:
                            gr.Info(f"Auto-fetching metadata for {lora_name}...")
                            success, msg = self._fetch_and_process_single_lora(full_path, key)
                            if success:
                                current_prompt = self.lora_metadata.get(key, {}).get("prompt", "")
                            else:
                                print(f"Auto-fetch warning: {msg}")

                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    civitai_data = json.load(f)
                            except: pass

                        with gr.Group():
                            with gr.Row(elem_classes="lora-card-header"):
                                gr.Markdown(f"#### 🏷️ {os.path.basename(lora_name)}")
                            
                            if "All LoRAs" in str(self.category_dropdown.value):
                                gr.Markdown(f"*(Folder: {os.path.dirname(lora_name)})*")

                            if civitai_data:
                                images_list = civitai_data.get('images', [])
                                if images_list:
                                    img_urls = [img.get('url') for img in images_list]
                                    gr.Gallery(value=img_urls, label="Preview Images", columns=4, height=250, object_fit="contain")

                                with gr.Row():
                                    model_name = civitai_data.get('model', {}).get('name', 'Unknown Model')
                                    version_name = civitai_data.get('name', 'Unknown Version')
                                    base_model = civitai_data.get('baseModel', 'Unknown Base')
                                    
                                    stats = civitai_data.get('stats', {})
                                    downloads = stats.get('downloadCount', 0)
                                    thumbs = stats.get('thumbsUpCount', 0)
                                    nsfw_level = civitai_data.get('nsfwLevel', 'N/A')
                                    
                                    stats_md = f"""
                                    **Model:** {model_name} ({version_name})  
                                    **Base:** {base_model}  
                                    **Downloads:** {downloads:,} | **👍** {thumbs:,} | **NSFW Level:** {nsfw_level}
                                    """
                                    gr.Markdown(stats_md)

                                    with gr.Column(scale=0):
                                        pub_date = self.format_date(civitai_data.get('publishedAt'))
                                        upd_date = self.format_date(civitai_data.get('updatedAt'))
                                        gr.Markdown(f"**Published:** {pub_date}\n**Updated:** {upd_date}")

                                        model_id = civitai_data.get('modelId')
                                        version_id = civitai_data.get('id')
                                        if model_id and version_id:
                                            link = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
                                            gr.Button("🔗 View on CivitAI", link=link, size="sm")

                                        update_btn = gr.Button("🔄 Update Info", size="sm", variant="secondary")
                                        def perform_update(fpath=full_path, k=key):
                                            s, m = self._fetch_and_process_single_lora(fpath, k)
                                            if s: gr.Info(m)
                                            else: gr.Warning(m)
                                            return 1
                                        update_btn.click(fn=perform_update, inputs=None, outputs=[self.refresh_trigger])

                                trained_words = civitai_data.get('trainedWords', [])
                                if trained_words:
                                    t_str = ", ".join(trained_words)
                                    gr.Markdown(f"**Trigger Words:** `{t_str}`")

                                desc_html = civitai_data.get('description', '')
                                if desc_html:
                                    with gr.Accordion("Description", open=False):
                                        gr.HTML(desc_html)

                            else:
                                with gr.Row():
                                    gr.Markdown("*No metadata found locally.*")
                                    fetch_btn = gr.Button("🌐 Fetch Info from CivitAI", size="sm", variant="secondary")
                                    
                                    def perform_manual_fetch(fpath=full_path, k=key):
                                        s, m = self._fetch_and_process_single_lora(fpath, k)
                                        if s: gr.Info(m)
                                        else: gr.Warning(m)
                                        return 1

                                    fetch_btn.click(fn=perform_manual_fetch, inputs=None, outputs=[self.refresh_trigger])

                            key_state = gr.State(key)
                            
                            prompt_input = gr.TextArea(
                                value=current_prompt,
                                label="Default Trigger / Prompt",
                                placeholder="Enter trigger words or prompt here...",
                                lines=2,
                                interactive=True
                            )
                            
                            save_btn = gr.Button("💾 Save Prompt", size="sm", variant="secondary")

                            save_btn.click(
                                fn=self.save_metadata,
                                inputs=[key_state, prompt_input],
                                outputs=None
                            )
                        gr.Markdown("---")

                with gr.Column(visible=False) as self.actions_panel:
                    gr.Markdown("### ⚙️ Injection Settings")
                    with gr.Row():
                        self.prompt_mode = gr.Radio(
                            choices=["Append", "Overwrite"],
                            value="Append",
                            label="Prompt Mode",
                            interactive=True
                        )
                        self.lora_mode = gr.Radio(
                            choices=["Append", "Overwrite"],
                            value="Append",
                            label="LoRA List Mode",
                            interactive=True
                        )
                    
                    self.use_btn = gr.Button("✨ Send to Generator", variant="primary")

                    with gr.Column(visible=False) as self.conflict_panel:
                        gr.Markdown("---")
                        gr.Markdown("#### ⚠️ Conflict Resolution")
                        gr.Markdown("Multiple prompts detected. How should they be handled?")
                        self.prompt_choice = gr.Radio(
                            choices=[],
                            label="Choose Strategy",
                            interactive=True
                        )
                        with gr.Row():
                            self.confirm_inject_btn = gr.Button("Confirm", variant="stop")
                            self.cancel_inject_btn = gr.Button("Cancel", variant="secondary")

        self.on_tab_outputs = [self.is_initialized, self.category_dropdown, self.lora_list]

        def toggle_actions(selected):
            return gr.update(visible=bool(selected))

        def reset_conflict_panel():
            return gr.update(visible=False), gr.update(value=None)

        self.lora_list.change(
            fn=toggle_actions,
            inputs=[self.lora_list],
            outputs=[self.actions_panel]
        ).then(
            fn=reset_conflict_panel,
            inputs=None,
            outputs=[self.conflict_panel, self.prompt_choice]
        )

        self.category_dropdown.change(
            fn=self.update_list_by_category,
            inputs=[self.category_dropdown],
            outputs=[self.lora_list]
        )

        self.refresh_btn.click(
            fn=self.refresh_button_click,
            inputs=[self.state, self.category_dropdown],
            outputs=[self.category_dropdown, self.lora_list]
        )

        self.update_all_btn.click(
            fn=self.batch_update_metadata,
            inputs=[self.state, self.category_dropdown, self.lora_list],
            outputs=[self.refresh_trigger]
        )

        self.use_btn.click(
            fn=self.prepare_injection,
            inputs=[self.lora_list, self.prompt_mode, self.lora_mode],
            outputs=[self.conflict_panel, self.prompt_choice, self.prompt, self.loras_choices, self.main_tabs]
        )

        self.confirm_inject_btn.click(
            fn=self.finalize_injection,
            inputs=[self.lora_list, self.prompt_choice, self.prompt_mode, self.lora_mode, self.prompt, self.loras_choices],
            outputs=[self.prompt, self.loras_choices, self.main_tabs, self.conflict_panel]
        )

        self.cancel_inject_btn.click(
            fn=reset_conflict_panel,
            inputs=None,
            outputs=[self.conflict_panel, self.prompt_choice]
        )

    def on_tab_select(self, state):
        return self.handle_tab_load(state)

    def handle_tab_load(self, state):
        if getattr(self, 'has_loaded_once', False):
            return gr.update(), gr.update(), gr.update()
        
        self.has_loaded_once = True
        is_init, dd_update, list_update = self.force_refresh(state, None)
        return is_init, dd_update, list_update

    def load_json_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.lora_metadata = json.load(f)
            except:
                self.lora_metadata = {}
        else:
            self.lora_metadata = {}

    def resolve_path(self, item_name):
        is_recursive = os.path.sep in item_name or "/" in item_name
        
        if is_recursive:
            full_path = os.path.join(self.lora_root, item_name)
            key = item_name
        else:
            full_path = ""
            key = ""
            for root, _, files in os.walk(self.lora_root):
                if item_name in files:
                    full_path = os.path.join(root, item_name)
                    key = os.path.join(os.path.relpath(root, self.lora_root), item_name)
                    break
        
        key = key.replace("\\", "/") 
        return key, full_path

    def save_metadata(self, key, prompt):
        if not key: return
        key = key.replace("\\", "/") 
        
        if key not in self.lora_metadata: self.lora_metadata[key] = {}
        self.lora_metadata[key]["prompt"] = prompt
        
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.lora_metadata, f, indent=4)
            gr.Info(f"Saved prompt!")
        except Exception as e:
            gr.Error(f"Save error: {e}")

    def discover_lora_root(self, state):
        model_type = self.get_state_model_type(state)
        try:
            specific_dir = self.get_lora_dir(model_type)
            if specific_dir and os.path.isdir(specific_dir):
                return os.path.dirname(specific_dir)
        except:
            pass
        return "loras"

    def build_category_map(self):
        folder_to_models = {}
        
        if hasattr(self, 'model_types') and self.model_types:
            for mtype in self.model_types:
                try:
                    path = self.get_lora_dir(mtype)
                    if path:
                        folder = os.path.basename(path)
                        dummy_list = [""]
                        pretty_name = self.get_model_name(mtype, dummy_list)
                        
                        if folder not in folder_to_models:
                            folder_to_models[folder] = []
                        
                        if pretty_name not in folder_to_models[folder]:
                            folder_to_models[folder].append(pretty_name)
                except:
                    continue
        
        display_map = {}
        for folder, models in folder_to_models.items():
            if not models:
                display_map[folder] = folder
            else:
                model_str = ", ".join(models[:2])
                if len(models) > 2: model_str += ", ..."
                display_map[folder] = f"{folder} ({model_str})"
        
        return display_map

    def force_refresh(self, state, current_selection):
        self.lora_root = self.discover_lora_root(state)
        display_map = self.build_category_map()

        folder_choices = []
        if os.path.isdir(self.lora_root):
            subdirs = [
                d for d in os.listdir(self.lora_root)
                if os.path.isdir(os.path.join(self.lora_root, d))
                and not d.startswith('.') and not d.startswith('__')
            ]

            for d in sorted(subdirs):
                label = display_map.get(d, d)
                folder_choices.append((label, d))

        choices = [("All LoRAs", "All LoRAs")] + folder_choices

        selected_val = current_selection
        valid_values = [c[1] for c in choices]

        if not selected_val or selected_val not in valid_values:
            current_model_type = self.get_state_model_type(state)
            try:
                target_dir = self.get_lora_dir(current_model_type)
                target_folder = os.path.basename(target_dir)

                if target_folder in valid_values:
                    selected_val = target_folder
                else:
                    selected_val = "All LoRAs"
            except:
                selected_val = "All LoRAs"

        list_update = self.update_list_by_category(selected_val)
        dropdown_update = gr.update(choices=choices, value=selected_val)
        return True, dropdown_update, list_update

    def refresh_button_click(self, state, current_selection):
        """Wrapper for refresh button - returns only dropdown and list updates (2 values)."""
        _, dropdown_update, list_update = self.force_refresh(state, current_selection)
        return dropdown_update, list_update

    def update_list_by_category(self, category):
        files = []
        if not category or not os.path.isdir(self.lora_root):
            return gr.update(choices=[])

        if category == "All LoRAs":
            for root, dirs, f_names in os.walk(self.lora_root):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                rel_root = os.path.relpath(root, self.lora_root)
                if rel_root == ".": rel_root = ""
                
                for f in f_names:
                    if f.endswith(".safetensors") or f.endswith(".sft"):
                        if rel_root:
                            files.append(os.path.join(rel_root, f))
                        else:
                            files.append(f)
        else:
            target_dir = os.path.join(self.lora_root, category)
            if os.path.isdir(target_dir):
                for f in os.listdir(target_dir):
                    if f.endswith(".safetensors") or f.endswith(".sft"):
                        files.append(f)
        
        files.sort()
        return gr.update(choices=files, value=[], label=f"Files in {category}")

    def prepare_injection(self, selected_loras, prompt_mode, lora_mode):
        if not selected_loras:
            gr.Warning("No LoRAs selected.")
            return gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update()

        prompts = []
        for l in selected_loras:
            key, _ = self.resolve_path(l)
            p = self.lora_metadata.get(key, {}).get("prompt", "")
            if p: prompts.append((os.path.basename(l), p))

        if len(selected_loras) == 1:
            p_text = prompts[0][1] if prompts else ""
            new_prompt, new_choices, tab_upd = self._perform_inject(selected_loras, p_text, prompt_mode, lora_mode)
            return gr.update(visible=False), gr.update(), new_prompt, new_choices, tab_upd

        if not prompts:
             new_prompt, new_choices, tab_upd = self._perform_inject(selected_loras, "", prompt_mode, lora_mode)
             return gr.update(visible=False), gr.update(), new_prompt, new_choices, tab_upd

        choices = []
        combined = []
        for name, p in prompts:
            choices.append((f"Use {name} prompt only", p))
            if p: combined.append(p)
        
        combined_str = ", ".join(combined)
        if combined_str:
            choices.append(("Combine All Prompts", combined_str))
        
        choices.append(("Don't add prompt (LoRAs only)", ""))
        
        return (
            gr.update(visible=True),
            gr.update(choices=choices, value=combined_str if combined_str else ""),
            gr.update(), gr.update(), gr.update()
        )

    def finalize_injection(self, selected_loras, prompt_choice, prompt_mode, lora_mode, current_prompt, current_loras):
        new_prompt, new_choices, tab_update = self._perform_inject(
            selected_loras, 
            prompt_choice, 
            prompt_mode,
            lora_mode,
            current_prompt,
            current_loras
        )
        return new_prompt, new_choices, tab_update, gr.update(visible=False)

    def _perform_inject(self, selected_loras_list, prompt_text, prompt_mode, lora_mode, current_ui_prompt="", current_ui_loras=None):
        if prompt_mode == "Overwrite":
            new_prompt = prompt_text
        else:
            new_prompt = current_ui_prompt or ""
            if prompt_text:
                if new_prompt:
                    new_prompt += "\n" + prompt_text
                else:
                    new_prompt = prompt_text

        if current_ui_loras is None: current_ui_loras = []
        if not isinstance(current_ui_loras, list): current_ui_loras = []
        
        if lora_mode == "Overwrite":
            final_loras = []
        else:
            final_loras = current_ui_loras.copy()
        
        for l in selected_loras_list:
            base = os.path.basename(l)
            if base not in final_loras:
                final_loras.append(base)

        gr.Info(f"Injected {len(selected_loras_list)} LoRAs")
        return new_prompt, final_loras, gr.Tabs(selected="video_gen")
