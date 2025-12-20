"""
Prompt Library Plugin for Wan2GP
Save, organize, and reuse prompts with their generation settings
"""

import gradio as gr
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from shared.utils.plugins import WAN2GPPlugin
from .library import PromptLibrary
from .templates import initialize_library_with_templates


class PromptLibraryPlugin(WAN2GPPlugin):
    """Plugin for managing and organizing prompt templates"""

    def __init__(self):
        super().__init__()
        self.name = "Prompt Library"
        self.version = "1.0.0"
        self.description = "Save and organize prompt templates with settings"
        self.library = PromptLibrary()

        # Initialize with templates if library is empty
        if self._is_library_empty():
            initialize_library_with_templates(self.library)

        # UI state
        self.selected_prompt_id = None
        self.selected_collection = "favorites"

    def _is_library_empty(self) -> bool:
        """Check if library has no prompts"""
        for collection in self.library.data["collections"].values():
            if collection["prompts"]:
                return False
        return True

    def setup_ui(self):
        """Setup plugin UI and request components"""
        # Request access to main UI components
        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("refresh_form_trigger")

        # Request global functions
        self.request_global("get_current_model_settings")
        self.request_global("server_config")

        # Register data hook to track prompt usage
        self.register_data_hook('before_metadata_save', self.track_prompt_usage)

        # Add the library tab
        self.add_tab(
            tab_id="prompt_library",
            label="📚 Prompt Library",
            component_constructor=self._build_ui,
        )

    def _build_ui(self):
        """Build the main UI for the prompt library"""
        with gr.Row():
            # Left panel - Collections
            with gr.Column(scale=1):
                gr.Markdown("### Collections")
                collection_choices = [name for name, _ in self.library.get_collection_names()]
                self.collection_radio = gr.Radio(
                    choices=collection_choices,
                    value=collection_choices[0] if collection_choices else None,
                    show_label=False,
                    container=False
                )

                with gr.Row():
                    self.new_collection_btn = gr.Button("➕ New", size="sm", scale=1)
                    self.delete_collection_btn = gr.Button("🗑️ Delete", size="sm", scale=1, variant="stop")

                # New collection creation group (hidden by default)
                with gr.Group(visible=False) as self.new_collection_group:
                    self.new_collection_name = gr.Textbox(label="New Collection Name", placeholder="My Collection")
                    with gr.Row():
                        self.confirm_new_collection_btn = gr.Button("Create", size="sm", variant="primary")
                        self.cancel_new_collection_btn = gr.Button("Cancel", size="sm")

            # Right panel - Prompts
            with gr.Column(scale=3):
                # Search and filter bar
                with gr.Row():
                    self.search_box = gr.Textbox(
                        placeholder="🔍 Search prompts...",
                        show_label=False,
                        scale=3
                    )
                    self.tag_filter = gr.Dropdown(
                        choices=self.library.get_all_tags(),
                        multiselect=True,
                        label="Filter by tags",
                        scale=2
                    )

                # Prompt gallery (HTML cards)
                self.prompt_gallery = gr.HTML(
                    value=self._render_prompt_gallery("favorites"),
                    elem_id="prompt_library_gallery"
                )

                # Selected prompt details
                with gr.Group(visible=False) as self.prompt_details_group:
                    gr.Markdown("### Selected Prompt")

                    with gr.Row():
                        self.prompt_name_display = gr.Textbox(
                            label="Name",
                            interactive=False,
                            scale=3
                        )
                        self.prompt_favorite_btn = gr.Button("⭐ Favorite", size="sm", scale=1)

                    self.prompt_text_display = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        interactive=False
                    )

                    self.negative_prompt_display = gr.Textbox(
                        label="Negative Prompt",
                        lines=2,
                        interactive=False
                    )

                    self.prompt_tags_display = gr.Textbox(
                        label="Tags",
                        interactive=False
                    )

                    # Variable substitution (shown only if prompt has variables)
                    with gr.Row(visible=False) as self.variable_row:
                        self.variable_inputs = gr.Textbox(
                            label="Fill in variables (format: key=value, key2=value2)",
                            placeholder="e.g., location=mountains, character=warrior",
                            info="Replace {variables} in the prompt with your values"
                        )

                    # Action buttons
                    with gr.Row():
                        self.use_prompt_btn = gr.Button(
                            "📝 Use Prompt Only",
                            variant="secondary",
                            scale=1
                        )
                        self.use_with_settings_btn = gr.Button(
                            "⚙️ Use with Settings",
                            variant="primary",
                            scale=1
                        )
                        self.edit_prompt_btn = gr.Button("✏️ Edit", scale=1)
                        self.delete_prompt_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)

        # Save current prompt section
        with gr.Accordion("💾 Create / Save Prompt", open=False):
            with gr.Row():
                self.fetch_settings_btn = gr.Button("📥 Fetch from Main Tab", size="sm")
                gr.Markdown("Populate fields from current generation settings")

            with gr.Row():
                self.save_name = gr.Textbox(
                    label="Prompt Name",
                    placeholder="My awesome prompt",
                    scale=2
                )
                self.save_collection = gr.Dropdown(
                    choices=[coll_id for _, coll_id in self.library.get_collection_names()],
                    label="Collection",
                    value="favorites",
                    scale=1
                )

            self.save_prompt_input = gr.Textbox(
                label="Prompt",
                lines=3,
                placeholder="Enter prompt here..."
            )

            self.save_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                placeholder="Enter negative prompt here..."
            )

            self.save_tags = gr.Textbox(
                label="Tags (comma-separated)",
                placeholder="cinematic, landscape, aerial"
            )

            self.save_include_settings = gr.Checkbox(
                label="Save current generation settings (model, resolution, LoRAs, etc.)",
                value=True
            )

            with gr.Row():
                self.save_btn = gr.Button("💾 Save to Library", variant="primary")
                self.save_status = gr.Markdown("")

        # Import/Export section
        with gr.Accordion("📤 Import/Export Collections", open=False):
            with gr.Row():
                self.import_file = gr.File(
                    label="Import Collection JSON",
                    file_types=[".json"]
                )
                self.import_merge_checkbox = gr.Checkbox(
                    label="Merge with existing (instead of replace)",
                    value=True
                )
                self.import_btn = gr.Button("📥 Import")

            with gr.Row():
                self.export_collection_choice = gr.Dropdown(
                    choices=[coll_id for _, coll_id in self.library.get_collection_names()],
                    label="Collection to Export",
                    value="favorites"
                )
                self.export_btn = gr.Button("📤 Export Collection")

            self.import_export_status = gr.Markdown("")

        # Hidden state for selected prompt ID
        self.selected_prompt_state = gr.State(value=None)

        # Wire up events
        self._setup_events()

        # Add custom CSS for prompt cards
        self._add_custom_css()

    def _setup_events(self):
        """Wire up all UI event handlers"""
        # Collection selection
        self.collection_radio.change(
            fn=self._on_collection_change,
            inputs=[self.collection_radio, self.search_box, self.tag_filter],
            outputs=[self.prompt_gallery, self.prompt_details_group, self.selected_prompt_state]
        )

        # Search and filter
        self.search_box.change(
            fn=self._on_search_or_filter_change,
            inputs=[self.collection_radio, self.search_box, self.tag_filter],
            outputs=[self.prompt_gallery, self.prompt_details_group, self.selected_prompt_state]
        )

        self.tag_filter.change(
            fn=self._on_search_or_filter_change,
            inputs=[self.collection_radio, self.search_box, self.tag_filter],
            outputs=[self.prompt_gallery, self.prompt_details_group, self.selected_prompt_state]
        )

        # Prompt card click (handled via JavaScript callback in HTML)
        # The prompt ID is passed via the custom JS event

        # Use prompt buttons
        self.use_prompt_btn.click(
            fn=self._use_prompt_only,
            inputs=[self.state, self.selected_prompt_state, self.variable_inputs],
            outputs=[self.main_tabs, self.save_status]
        )

        self.use_with_settings_btn.click(
            fn=self._use_with_settings,
            inputs=[self.state, self.selected_prompt_state, self.variable_inputs],
            outputs=[self.refresh_form_trigger, self.main_tabs, self.save_status]
        )

        # Edit prompt
        self.edit_prompt_btn.click(
            fn=self._show_edit_dialog,
            inputs=[self.selected_prompt_state],
            outputs=[
                self.prompt_name_display,
                self.prompt_text_display,
                self.negative_prompt_display,
                self.prompt_tags_display
            ]
        )

        # Delete prompt
        self.delete_prompt_btn.click(
            fn=self._delete_prompt,
            inputs=[self.selected_prompt_state, self.collection_radio],
            outputs=[self.prompt_gallery, self.prompt_details_group, self.save_status]
        )

        # Favorite button
        self.prompt_favorite_btn.click(
            fn=self._toggle_favorite,
            inputs=[self.selected_prompt_state],
            outputs=[self.save_status, self.prompt_favorite_btn]
        )

        # Fetch settings
        self.fetch_settings_btn.click(
            fn=self._fetch_current_settings,
            inputs=[self.state],
            outputs=[self.save_prompt_input, self.save_negative_prompt_input, self.save_status]
        )

        # Save current prompt
        self.save_btn.click(
            fn=self._save_current_prompt,
            inputs=[
                self.state,
                self.save_name,
                self.save_collection,
                self.save_prompt_input,
                self.save_negative_prompt_input,
                self.save_tags,
                self.save_include_settings
            ],
            outputs=[self.save_status, self.prompt_gallery, self.tag_filter]
        )

        # New collection UI
        self.new_collection_btn.click(
            fn=lambda: gr.Group(visible=True),
            outputs=[self.new_collection_group]
        )

        self.cancel_new_collection_btn.click(
            fn=lambda: gr.Group(visible=False),
            outputs=[self.new_collection_group]
        )

        self.confirm_new_collection_btn.click(
            fn=self._create_new_collection,
            inputs=[self.new_collection_name],
            outputs=[
                self.collection_radio,
                self.save_collection,
                self.save_status,
                self.new_collection_group,
                self.new_collection_name
            ]
        )

        # Delete collection
        self.delete_collection_btn.click(
            fn=self._delete_collection,
            inputs=[self.collection_radio],
            outputs=[self.collection_radio, self.prompt_gallery, self.save_status]
        )

        # Import/Export
        self.import_btn.click(
            fn=self._import_collection,
            inputs=[self.import_file, self.import_merge_checkbox],
            outputs=[self.import_export_status, self.collection_radio, self.prompt_gallery]
        )

        self.export_btn.click(
            fn=self._export_collection,
            inputs=[self.export_collection_choice],
            outputs=[self.import_export_status]
        )

    def _render_prompt_gallery(self, collection_display_name: str, search: str = "", tags: List[str] = None) -> str:
        """Render prompt cards as HTML

        Args:
            collection_display_name: Display name of collection (with icon)
            search: Search filter
            tags: Tag filters

        Returns:
            HTML string
        """
        # Extract collection ID from display name
        collection_id = self._extract_collection_id(collection_display_name)
        if not collection_id:
            return "<p>Collection not found</p>"

        prompts = self.library.get_prompts_in_collection(collection_id, search, tags)

        if not prompts:
            return "<div style='text-align: center; padding: 40px; color: #888;'>No prompts found. Add some prompts to get started!</div>"

        html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px;'>"

        for prompt in prompts:
            is_favorite = self.library.is_in_favorites(prompt["id"])
            favorite_icon = "⭐" if is_favorite else "☆"

            # Truncate prompt text for display
            prompt_text = prompt["prompt"]
            if len(prompt_text) > 100:
                prompt_text = prompt_text[:100] + "..."

            # Format tags
            tag_html = ""
            for tag in prompt.get("tags", [])[:3]:  # Show max 3 tags
                tag_html += f'<span style="background: #e3f2fd; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 4px;">{tag}</span>'

            # Format usage stats
            use_count = prompt.get("use_count", 0)
            stats = f"Used {use_count}x"

            # Variables indicator
            variables_html = ""
            if prompt.get("variables"):
                var_count = len(prompt["variables"])
                variables_html = f'<span style="color: #ff9800; font-size: 12px;">📝 {var_count} variable{"s" if var_count > 1 else ""}</span>'

            html += f"""
            <div onclick="selectPrompt('{prompt['id']}')"
                 style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; cursor: pointer;
                        transition: all 0.2s; background: white;"
                 onmouseover="this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)'; this.style.transform='translateY(-2px)'"
                 onmouseout="this.style.boxShadow='none'; this.style.transform='translateY(0)'">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                    <strong style="font-size: 16px;">{prompt['name']}</strong>
                    <span style="font-size: 20px;">{favorite_icon}</span>
                </div>
                <p style="color: #666; font-size: 14px; margin: 8px 0;">{prompt_text}</p>
                <div style="margin: 8px 0;">{tag_html}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                    <span style="color: #888; font-size: 12px;">{stats}</span>
                    {variables_html}
                </div>
            </div>
            """

        html += "</div>"

        # Add JavaScript for card selection
        html += """
        <script>
        function selectPrompt(promptId) {
            // Store selected prompt ID
            window.selectedPromptId = promptId;

            // Trigger Gradio update
            const event = new CustomEvent('promptSelected', { detail: { promptId: promptId } });
            window.dispatchEvent(event);

            console.log('Selected prompt:', promptId);
        }
        </script>
        """

        return html

    def _extract_collection_id(self, display_name: str) -> Optional[str]:
        """Extract collection ID from display name"""
        for display, coll_id in self.library.get_collection_names():
            if display == display_name:
                return coll_id
        return None

    def _on_collection_change(self, collection_display: str, search: str, tags: List[str]) -> Tuple:
        """Handle collection selection change"""
        self.selected_collection = self._extract_collection_id(collection_display)
        gallery_html = self._render_prompt_gallery(collection_display, search, tags)
        return gallery_html, gr.update(visible=False), None

    def _on_search_or_filter_change(self, collection_display: str, search: str, tags: List[str]) -> Tuple:
        """Handle search or filter change"""
        gallery_html = self._render_prompt_gallery(collection_display, search, tags)
        return gallery_html, gr.update(visible=False), None

    def _use_prompt_only(self, state: Dict, prompt_id: Optional[str], variables: str) -> Tuple:
        """Copy just the prompt text to the main input

        Args:
            state: Application state
            prompt_id: ID of prompt to use
            variables: Variable substitutions

        Returns:
            Tuple of (tab_update, status_message)
        """
        if not prompt_id:
            return gr.update(), "⚠️ Please select a prompt first"

        prompt_data = self.library.get_prompt(prompt_id)
        if not prompt_data:
            return gr.update(), "❌ Prompt not found"

        # Substitute variables
        final_prompt = self._substitute_variables(prompt_data["prompt"], variables)

        # Update state with new prompt
        if self.get_current_model_settings:
            settings = self.get_current_model_settings(state)
            settings["prompt"] = final_prompt
            if prompt_data.get("negative_prompt"):
                settings["negative_prompt"] = prompt_data["negative_prompt"]

        # Record usage
        self.library.record_usage(prompt_id)

        # Switch to video generation tab
        return gr.Tabs(selected="video_gen"), f"✅ Loaded prompt: {prompt_data['name']}"

    def _use_with_settings(self, state: Dict, prompt_id: Optional[str], variables: str) -> Tuple:
        """Copy prompt AND apply all saved settings

        Args:
            state: Application state
            prompt_id: ID of prompt to use
            variables: Variable substitutions

        Returns:
            Tuple of (refresh_trigger, tab_update, status_message)
        """
        if not prompt_id:
            return gr.update(), gr.update(), "⚠️ Please select a prompt first"

        prompt_data = self.library.get_prompt(prompt_id)
        if not prompt_data:
            return gr.update(), gr.update(), "❌ Prompt not found"

        # Substitute variables
        final_prompt = self._substitute_variables(prompt_data["prompt"], variables)

        # Update state with prompt and settings
        if self.get_current_model_settings:
            settings = self.get_current_model_settings(state)
            settings["prompt"] = final_prompt

            if prompt_data.get("negative_prompt"):
                settings["negative_prompt"] = prompt_data["negative_prompt"]

            # Apply saved settings
            saved_settings = prompt_data.get("settings", {})
            for key, value in saved_settings.items():
                if key in settings:
                    settings[key] = value

        # Record usage
        self.library.record_usage(prompt_id)

        # Trigger form refresh and switch to video tab
        return time.time(), gr.Tabs(selected="video_gen"), f"✅ Loaded prompt with settings: {prompt_data['name']}"

    def _substitute_variables(self, prompt: str, variable_string: str) -> str:
        """Replace {variable} placeholders with values

        Args:
            prompt: Prompt text with {placeholders}
            variable_string: String like "key=value, key2=value2"

        Returns:
            Prompt with substituted values
        """
        if not variable_string or not variable_string.strip():
            return prompt

        # Parse variable string
        variables = {}
        for pair in variable_string.split(","):
            pair = pair.strip()
            if "=" in pair:
                key, value = pair.split("=", 1)
                variables[key.strip()] = value.strip()

        # Replace {key} with value
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", value)

        return prompt

    def _show_edit_dialog(self, prompt_id: Optional[str]) -> Tuple:
        """Show prompt details for editing

        Args:
            prompt_id: ID of prompt to edit

        Returns:
            Tuple of display values
        """
        if not prompt_id:
            return "", "", "", ""

        prompt_data = self.library.get_prompt(prompt_id)
        if not prompt_data:
            return "", "", "", ""

        tags_str = ", ".join(prompt_data.get("tags", []))

        return (
            prompt_data["name"],
            prompt_data["prompt"],
            prompt_data.get("negative_prompt", ""),
            tags_str
        )

    def _delete_prompt(self, prompt_id: Optional[str], collection_display: str) -> Tuple:
        """Delete a prompt

        Args:
            prompt_id: ID of prompt to delete
            collection_display: Current collection display name

        Returns:
            Tuple of (gallery_update, details_update, status_message)
        """
        if not prompt_id:
            return gr.update(), gr.update(), "⚠️ Please select a prompt first"

        prompt_data = self.library.get_prompt(prompt_id)
        if not prompt_data:
            return gr.update(), gr.update(), "❌ Prompt not found"

        prompt_name = prompt_data["name"]

        if self.library.delete_prompt(prompt_id):
            gallery_html = self._render_prompt_gallery(collection_display)
            return gallery_html, gr.update(visible=False), f"✅ Deleted prompt: {prompt_name}"
        else:
            return gr.update(), gr.update(), f"❌ Failed to delete prompt: {prompt_name}"

    def _toggle_favorite(self, prompt_id: Optional[str]) -> Tuple:
        """Toggle favorite status of a prompt

        Args:
            prompt_id: ID of prompt to toggle

        Returns:
            Tuple of (status_message, button_update)
        """
        if not prompt_id:
            return "⚠️ Please select a prompt first", gr.update()

        is_favorite = self.library.is_in_favorites(prompt_id)

        if is_favorite:
            if self.library.remove_from_favorites(prompt_id):
                return "✅ Removed from favorites", gr.Button(value="☆ Favorite")
            else:
                return "❌ Failed to remove from favorites", gr.update()
        else:
            if self.library.add_to_favorites(prompt_id):
                return "✅ Added to favorites", gr.Button(value="⭐ Unfavorite")
            else:
                return "❌ Failed to add to favorites", gr.update()

    def _fetch_current_settings(self, state: Dict) -> Tuple:
        """Fetch current settings from main tab"""
        settings = self.get_current_model_settings(state) if self.get_current_model_settings else {}
        prompt = settings.get("prompt", "")
        negative_prompt = settings.get("negative_prompt", "")

        return prompt, negative_prompt, "✅ Fetched settings from main tab"

    def _save_current_prompt(
        self,
        state: Dict,
        name: str,
        collection: str,
        prompt_input: str,
        negative_prompt_input: str,
        tags_str: str,
        include_settings: bool
    ) -> Tuple:
        """Save current prompt to library

        Args:
            state: Application state
            name: Name for the prompt
            collection: Collection to save to
            prompt_input: Manual prompt input
            negative_prompt_input: Manual negative prompt input
            tags_str: Comma-separated tags
            include_settings: Whether to save generation settings

        Returns:
            Tuple of (status_message, gallery_update, tag_filter_update)
        """
        if not name or not name.strip():
            return "⚠️ Please provide a name for the prompt", gr.update(), gr.update()

        # Get prompt from input or settings
        prompt = prompt_input
        negative_prompt = negative_prompt_input

        # Get current settings for background data
        settings = self.get_current_model_settings(state) if self.get_current_model_settings else {}

        if not prompt:
            # Fallback to settings
            prompt = settings.get("prompt", "")
            if not negative_prompt:
                negative_prompt = settings.get("negative_prompt", "")

        if not prompt:
            return "⚠️ No prompt to save. Enter a prompt or fetch from main tab.", gr.update(), gr.update()

        # Parse tags
        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

        # Capture settings if requested
        saved_settings = {}
        if include_settings:
            # Save relevant generation settings
            settings_to_save = [
                "model_type", "resolution", "steps", "guidance_scale",
                "num_frames", "fps", "seed", "sampler"
            ]
            for key in settings_to_save:
                if key in settings:
                    saved_settings[key] = settings[key]

            # Save LoRA settings if present
            if "loras" in settings:
                saved_settings["loras"] = settings["loras"]

        # Add prompt to library
        prompt_id = self.library.add_prompt(
            collection_id=collection,
            name=name.strip(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            tags=tags,
            settings=saved_settings
        )

        if prompt_id:
            # Update gallery if we're viewing the same collection
            gallery_update = gr.update()
            if self.selected_collection == collection:
                collection_display = None
                for display, coll_id in self.library.get_collection_names():
                    if coll_id == collection:
                        collection_display = display
                        break
                if collection_display:
                    gallery_update = self._render_prompt_gallery(collection_display)

            # Update tag filter
            tag_update = gr.Dropdown(choices=self.library.get_all_tags())

            return f"✅ Saved prompt: {name}", gallery_update, tag_update
        else:
            return "❌ Failed to save prompt", gr.update(), gr.update()

    def _create_new_collection(self, name: str) -> Tuple:
        """Create a new collection"""
        if not name or not name.strip():
            return gr.update(), gr.update(), "⚠️ Please enter a collection name", gr.update(visible=True), gr.update()

        collection_id = name.lower().replace(" ", "_")
        # Ensure unique ID
        if collection_id in self.library.data["collections"]:
             collection_id = f"{collection_id}_{int(time.time())}"

        if self.library.create_collection(collection_id, name):
            # Update choices
            new_choices = [name for name, _ in self.library.get_collection_names()]

            # Select the new collection
            radio_update = gr.Radio(choices=new_choices, value=name)
            dropdown_update = gr.Dropdown(choices=[coll_id for _, coll_id in self.library.get_collection_names()], value=collection_id)

            return (
                radio_update,
                dropdown_update,
                f"✅ Created collection: {name}",
                gr.Group(visible=False), # Hide creation group
                "" # Clear input
            )
        else:
            return gr.update(), gr.update(), "❌ Failed to create collection", gr.update(visible=True), gr.update()

    def _delete_collection(self, collection_display: str) -> Tuple:
        """Delete a collection

        Args:
            collection_display: Display name of collection to delete

        Returns:
            Tuple of (radio_update, gallery_update, status_message)
        """
        collection_id = self._extract_collection_id(collection_display)

        if not collection_id:
            return gr.update(), gr.update(), "❌ Collection not found"

        if collection_id == "favorites":
            return gr.update(), gr.update(), "⚠️ Cannot delete Favorites collection"

        if self.library.delete_collection(collection_id):
            # Update radio choices
            new_choices = [name for name, _ in self.library.get_collection_names()]
            radio_update = gr.Radio(choices=new_choices, value=new_choices[0] if new_choices else None)

            # Update gallery
            gallery_update = self._render_prompt_gallery(new_choices[0] if new_choices else "favorites")

            return radio_update, gallery_update, f"✅ Deleted collection: {collection_display}"
        else:
            return gr.update(), gr.update(), f"❌ Failed to delete collection: {collection_display}"

    def _import_collection(self, file_obj, merge: bool) -> Tuple:
        """Import a collection from JSON file

        Args:
            file_obj: Uploaded file object
            merge: Whether to merge or replace

        Returns:
            Tuple of (status_message, radio_update, gallery_update)
        """
        if not file_obj:
            return "⚠️ Please select a file to import", gr.update(), gr.update()

        try:
            # Read file
            if hasattr(file_obj, 'name'):
                file_path = file_obj.name
            else:
                file_path = file_obj

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Import
            if self.library.import_collection(data, merge=merge):
                # Update UI
                new_choices = [name for name, _ in self.library.get_collection_names()]
                radio_update = gr.Radio(choices=new_choices, value=new_choices[0] if new_choices else None)
                gallery_update = self._render_prompt_gallery(new_choices[0] if new_choices else "favorites")

                return "✅ Collection imported successfully", radio_update, gallery_update
            else:
                return "❌ Failed to import collection", gr.update(), gr.update()

        except Exception as e:
            return f"❌ Error importing collection: {str(e)}", gr.update(), gr.update()

    def _export_collection(self, collection_display: str) -> str:
        """Export a collection to JSON file

        Args:
            collection_display: Display name of collection to export

        Returns:
            Status message with download link
        """
        collection_id = self._extract_collection_id(collection_display)

        if not collection_id:
            return "❌ Collection not found"

        export_data = self.library.export_collection(collection_id)
        if not export_data:
            return "❌ Failed to export collection"

        # Save to file
        output_dir = Path.home() / ".wan2gp" / "exports"
        output_dir.mkdir(exist_ok=True)

        filename = f"{collection_id}_export.json"
        output_path = output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return f"✅ Collection exported to: {output_path}"
        except Exception as e:
            return f"❌ Error exporting collection: {str(e)}"

    def track_prompt_usage(self, configs: Dict, **kwargs) -> Dict:
        """Data hook - track when saved prompts are used

        Args:
            configs: Generation configs
            **kwargs: Additional hook data

        Returns:
            Modified configs (unchanged in this case)
        """
        prompt = configs.get("prompt", "")
        if prompt:
            # Check if this matches a saved prompt
            matching = self.library.find_by_prompt(prompt)
            if matching:
                self.library.record_usage(matching["id"])

        return configs

    def _add_custom_css(self):
        """Add custom CSS for the prompt library UI"""
        css = """
        <style>
        #prompt_library_gallery {
            min-height: 400px;
        }
        </style>
        """
        # Note: Custom CSS would be injected via the plugin system's add_custom_js method
        # For now, it's included in the HTML output

    def on_tab_select(self, state: Dict[str, Any]) -> None:
        """Called when the Prompt Library tab is selected

        Args:
            state: Application state
        """
        # Refresh library from disk in case it was modified externally
        self.library.data = self.library._load_library()

    def on_tab_deselect(self, state: Dict[str, Any]) -> None:
        """Called when leaving the Prompt Library tab

        Args:
            state: Application state
        """
        # Save any pending changes
        self.library.save_library()
