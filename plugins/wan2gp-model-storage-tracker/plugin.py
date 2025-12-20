"""
Model Storage Tracker Plugin for Wan2GP
Track downloaded models, checkpoints, and LoRAs with storage usage information
Extended with comprehensive LoRA tracking and metadata management
"""

import gradio as gr
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


from shared.utils.plugins import WAN2GPPlugin


class ModelStorageTrackerPlugin(WAN2GPPlugin):
    """Plugin for tracking model storage usage with extended LoRA tracking"""

    def __init__(self):
        super().__init__()
        self.name = "Model Storage Tracker"
        self.version = "2.0.0"
        self.description = "Track downloaded models, checkpoints, and LoRAs with rich metadata"

        # Cache for scan results
        self.last_scan_time = None
        self.cached_models = []
        self.cached_stats = {}

        # LoRA metadata database
        self.lora_metadata_path = Path.home() / ".wan2gp" / "lora_metadata.json"
        self.lora_metadata = self._load_lora_metadata()

        # Currently selected LoRA for editing
        self.selected_lora_path = None

    def _load_lora_metadata(self) -> Dict[str, Any]:
        """Load LoRA metadata from disk"""
        if self.lora_metadata_path.exists():
            try:
                with open(self.lora_metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"loras": {}}
        return {"loras": {}}

    def _save_lora_metadata(self):
        """Save LoRA metadata to disk"""
        self.lora_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.lora_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.lora_metadata, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving LoRA metadata: {e}")

    def _get_lora_metadata(self, lora_path: str) -> Dict[str, Any]:
        """Get metadata for a specific LoRA, creating default if not exists"""
        if lora_path not in self.lora_metadata.get("loras", {}):
            self.lora_metadata.setdefault("loras", {})[lora_path] = {
                "trigger_words": [],
                "tags": [],
                "notes": "",
                "rating": 0,
                "model_type": self._detect_model_type(lora_path),
                "usage_count": 0,
                "last_used": None,
                "source_url": "",
                "date_added": datetime.now().isoformat()
            }
            self._save_lora_metadata()
        return self.lora_metadata["loras"][lora_path]

    def _update_lora_metadata(self, lora_path: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a specific LoRA. Returns True on success."""
        if not lora_path:
            return False
        # Ensure the LoRA has metadata entry (creates default if missing)
        self._get_lora_metadata(lora_path)
        # Now update with the new values
        if lora_path in self.lora_metadata.get("loras", {}):
            self.lora_metadata["loras"][lora_path].update(updates)
            self._save_lora_metadata()
            return True
        return False

    def _detect_model_type(self, lora_path: str) -> str:
        """Detect model type from LoRA path"""
        path_lower = lora_path.lower()
        if "hunyuan_i2v" in path_lower or "loras_hunyuan_i2v" in path_lower:
            return "hunyuan_i2v"
        elif "hunyuan" in path_lower or "loras_hunyuan" in path_lower:
            return "hunyuan"
        elif "flux" in path_lower or "loras_flux" in path_lower:
            return "flux"
        elif "ltxv" in path_lower or "loras_ltxv" in path_lower:
            return "ltxv"
        elif "qwen" in path_lower or "loras_qwen" in path_lower:
            return "qwen"
        elif "tts" in path_lower or "loras_tts" in path_lower:
            return "tts"
        elif "i2v" in path_lower or "loras_i2v" in path_lower:
            return "wan_i2v"
        else:
            return "wan_t2v"

    def setup_ui(self):
        """Setup plugin UI"""
        self.request_component("state")
        self.request_global("server_config")

        self.add_tab(
            tab_id="model_storage_tracker",
            label="💾 Storage",
            component_constructor=self._build_ui,
        )

    def _build_ui(self):
        """Build the main UI for the storage tracker"""

        with gr.Column():
            gr.Markdown("## 💾 Model Storage Tracker")
            gr.Markdown("Track your downloaded models, checkpoints, and LoRAs")

            # Summary stats at the top
            with gr.Row():
                self.total_size_display = gr.Markdown("### Total Size: Calculating...")
                self.total_count_display = gr.Markdown("### Total Files: Calculating...")
                self.last_scan_display = gr.Markdown("### Last Scan: Never")

            # Action buttons
            with gr.Row():
                self.scan_btn = gr.Button("🔄 Scan Storage", variant="primary", scale=1)
                self.filter_dropdown = gr.Dropdown(
                    choices=["All", "Models", "LoRAs", "Checkpoints", "> 1GB", "> 5GB", "> 10GB"],
                    value="All",
                    label="Filter",
                    scale=1
                )
                self.sort_dropdown = gr.Dropdown(
                    choices=["Size (Largest)", "Size (Smallest)", "Name (A-Z)", "Name (Z-A)", "Date (Newest)", "Date (Oldest)"],
                    value="Size (Largest)",
                    label="Sort By",
                    scale=1
                )

            # Storage breakdown by category
            with gr.Accordion("📊 Storage Breakdown", open=True):
                self.breakdown_html = gr.HTML()

            # LoRA Library Section
            with gr.Accordion("🎨 LoRA Library", open=True):
                gr.Markdown("### Manage your LoRA collection with metadata, tags, and usage tracking")

                with gr.Row():
                    self.lora_filter_dropdown = gr.Dropdown(
                        choices=[
                            "All LoRAs",
                            "Wan T2V", "Wan I2V",
                            "Hunyuan", "Hunyuan I2V",
                            "Flux", "LTXV", "Qwen", "TTS",
                            "★★★★★ (5 stars)", "★★★★ (4+ stars)", "★★★ (3+ stars)",
                            "Unrated", "Recently Used", "Most Used", "Never Used"
                        ],
                        value="All LoRAs",
                        label="Filter LoRAs",
                        scale=1
                    )
                    self.lora_search_box = gr.Textbox(
                        placeholder="🔍 Search by name, trigger words, or tags...",
                        show_label=False,
                        scale=2
                    )
                    self.lora_sort_dropdown = gr.Dropdown(
                        choices=["Name (A-Z)", "Name (Z-A)", "Rating (High)", "Rating (Low)", "Most Used", "Recently Used", "Size (Largest)", "Size (Smallest)"],
                        value="Name (A-Z)",
                        label="Sort",
                        scale=1
                    )

                # LoRA cards display
                self.lora_library_html = gr.HTML()

                # LoRA Editor Section
                with gr.Accordion("✏️ Edit LoRA Metadata", open=False) as self.lora_editor_accordion:
                    self.selected_lora_display = gr.Markdown("*Select a LoRA from the library above to edit*")
                    self.selected_lora_path_state = gr.State(value=None)

                    with gr.Row():
                        self.lora_select_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select LoRA to Edit",
                            scale=3
                        )
                        self.load_lora_btn = gr.Button("📂 Load", scale=1)

                    with gr.Row():
                        with gr.Column(scale=1):
                            self.trigger_words_input = gr.Textbox(
                                label="🏷️ Trigger Words",
                                placeholder="word1, word2, word3",
                                info="Comma-separated keywords that activate this LoRA"
                            )
                            self.tags_input = gr.Textbox(
                                label="📁 Tags",
                                placeholder="style, character, motion, anime",
                                info="Comma-separated tags for organization"
                            )
                            self.model_type_dropdown = gr.Dropdown(
                                choices=["wan_t2v", "wan_i2v", "hunyuan", "hunyuan_i2v", "flux", "ltxv", "qwen", "tts"],
                                label="🎬 Model Type",
                                info="Which base model this LoRA is for"
                            )

                        with gr.Column(scale=1):
                            self.rating_slider = gr.Slider(
                                minimum=0,
                                maximum=5,
                                step=1,
                                value=0,
                                label="⭐ Rating",
                                info="Your personal rating (0 = unrated)"
                            )
                            self.source_url_input = gr.Textbox(
                                label="🔗 Source URL",
                                placeholder="https://civitai.com/models/... or https://huggingface.co/...",
                                info="Where you downloaded this LoRA from"
                            )
                            self.notes_input = gr.Textbox(
                                label="📝 Notes",
                                placeholder="Works best with strength 0.8, good for anime styles...",
                                lines=3,
                                info="Your notes about this LoRA"
                            )

                    with gr.Row():
                        self.save_lora_btn = gr.Button("💾 Save Metadata", variant="primary", scale=2)
                        self.clear_lora_btn = gr.Button("🗑️ Clear Form", scale=1)

                    self.lora_save_status = gr.Markdown("")

                # Usage Statistics
                with gr.Accordion("📈 LoRA Usage Statistics", open=False):
                    self.usage_stats_html = gr.HTML()
                    self.refresh_stats_btn = gr.Button("🔄 Refresh Statistics")

            # Detailed file list
            with gr.Accordion("📁 Detailed File List", open=False):
                with gr.Row():
                    self.search_box = gr.Textbox(
                        placeholder="🔍 Search files...",
                        show_label=False,
                        scale=4
                    )
                    self.export_btn = gr.Button("📤 Export List", scale=1)

                self.file_list_html = gr.HTML()

            # Export status
            self.export_status = gr.Markdown("")

        # Wire up events
        self._setup_events()

        # Trigger initial scan
        self.on_tab_outputs = [
            self.total_size_display,
            self.total_count_display,
            self.last_scan_display,
            self.breakdown_html,
            self.file_list_html,
            self.lora_library_html,
            self.lora_select_dropdown,
            self.usage_stats_html
        ]

    def _setup_events(self):
        """Wire up UI event handlers"""

        # Scan button
        self.scan_btn.click(
            fn=self._scan_storage,
            inputs=[],
            outputs=[
                self.total_size_display,
                self.total_count_display,
                self.last_scan_display,
                self.breakdown_html,
                self.file_list_html,
                self.lora_library_html,
                self.lora_select_dropdown,
                self.usage_stats_html
            ]
        )

        # Filter change
        self.filter_dropdown.change(
            fn=self._apply_filter_and_sort,
            inputs=[self.filter_dropdown, self.sort_dropdown, self.search_box],
            outputs=[self.file_list_html]
        )

        # Sort change
        self.sort_dropdown.change(
            fn=self._apply_filter_and_sort,
            inputs=[self.filter_dropdown, self.sort_dropdown, self.search_box],
            outputs=[self.file_list_html]
        )

        # Search
        self.search_box.change(
            fn=self._apply_filter_and_sort,
            inputs=[self.filter_dropdown, self.sort_dropdown, self.search_box],
            outputs=[self.file_list_html]
        )

        # Export button
        self.export_btn.click(
            fn=self._export_list,
            inputs=[self.filter_dropdown, self.sort_dropdown, self.search_box],
            outputs=[self.export_status]
        )

        # LoRA Library filters
        self.lora_filter_dropdown.change(
            fn=self._apply_lora_filters,
            inputs=[self.lora_filter_dropdown, self.lora_search_box, self.lora_sort_dropdown],
            outputs=[self.lora_library_html]
        )

        self.lora_search_box.change(
            fn=self._apply_lora_filters,
            inputs=[self.lora_filter_dropdown, self.lora_search_box, self.lora_sort_dropdown],
            outputs=[self.lora_library_html]
        )

        self.lora_sort_dropdown.change(
            fn=self._apply_lora_filters,
            inputs=[self.lora_filter_dropdown, self.lora_search_box, self.lora_sort_dropdown],
            outputs=[self.lora_library_html]
        )

        # LoRA Editor
        self.load_lora_btn.click(
            fn=self._load_lora_for_editing,
            inputs=[self.lora_select_dropdown],
            outputs=[
                self.selected_lora_display,
                self.selected_lora_path_state,
                self.trigger_words_input,
                self.tags_input,
                self.model_type_dropdown,
                self.rating_slider,
                self.source_url_input,
                self.notes_input
            ]
        )

        self.save_lora_btn.click(
            fn=self._save_lora_metadata_ui,
            inputs=[
                self.selected_lora_path_state,
                self.trigger_words_input,
                self.tags_input,
                self.model_type_dropdown,
                self.rating_slider,
                self.source_url_input,
                self.notes_input
            ],
            outputs=[self.lora_save_status, self.lora_library_html]
        )

        self.clear_lora_btn.click(
            fn=self._clear_lora_form,
            inputs=[],
            outputs=[
                self.selected_lora_display,
                self.selected_lora_path_state,
                self.trigger_words_input,
                self.tags_input,
                self.model_type_dropdown,
                self.rating_slider,
                self.source_url_input,
                self.notes_input,
                self.lora_save_status
            ]
        )

        # Usage statistics refresh
        self.refresh_stats_btn.click(
            fn=self._render_usage_stats,
            inputs=[],
            outputs=[self.usage_stats_html]
        )

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def _get_model_paths(self) -> Dict[str, List[str]]:
        """Get all model, checkpoint, and LoRA paths"""
        base_dir = Path(__file__).parent.parent.parent

        paths = {
            "models": [],
            "loras": [],
            "checkpoints": []
        }

        # Main models directory
        models_dir = base_dir / "models"
        if models_dir.exists():
            paths["models"].append(str(models_dir))

        # All LoRA directories
        lora_dirs = [
            "loras",           # General t2v
            "loras_i2v",       # Image-to-video
            "loras_hunyuan",   # Hunyuan Video t2v
            "loras_hunyuan_i2v", # Hunyuan Video i2v
            "loras_flux",      # Flux
            "loras_ltxv",      # LTX Video
            "loras_qwen",      # Qwen
            "loras_tts"        # TTS
        ]

        for lora_dir in lora_dirs:
            lora_path = base_dir / lora_dir
            if lora_path.exists():
                paths["loras"].append(str(lora_path))

        # Get checkpoint paths from server config
        if hasattr(self, 'server_config') and self.server_config:
            checkpoint_paths = self.server_config.get("checkpoints_paths", [])
            for path in checkpoint_paths:
                if path and os.path.exists(path):
                    paths["checkpoints"].append(path)

        return paths

    def _scan_directory(self, directory: str, category: str) -> List[Dict[str, Any]]:
        """Scan a directory for model files"""
        files = []
        extensions = ['.safetensors', '.ckpt', '.pth', '.pt', '.bin', '.sft']

        try:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    if any(filename.lower().endswith(ext) for ext in extensions):
                        filepath = os.path.join(root, filename)
                        try:
                            stat_info = os.stat(filepath)
                            relative_path = os.path.relpath(filepath, directory)

                            file_data = {
                                'name': filename,
                                'path': filepath,
                                'relative_path': relative_path,
                                'size': stat_info.st_size,
                                'modified': datetime.fromtimestamp(stat_info.st_mtime),
                                'category': category,
                                'directory': directory
                            }

                            # Add LoRA metadata if it's a LoRA
                            if category == 'loras':
                                file_data['lora_metadata'] = self._get_lora_metadata(filepath)

                            files.append(file_data)
                        except (OSError, FileNotFoundError):
                            continue
        except (OSError, PermissionError):
            pass

        return files

    def _scan_storage(self) -> Tuple:
        """Scan all storage locations and return summary"""
        all_files = []

        paths = self._get_model_paths()

        # Scan each category
        for category, directories in paths.items():
            for directory in directories:
                files = self._scan_directory(directory, category)
                all_files.extend(files)

        # Update cache
        self.cached_models = all_files
        self.last_scan_time = datetime.now()

        # Calculate stats
        total_size = sum(f['size'] for f in all_files)
        total_count = len(all_files)

        # Calculate category breakdown
        category_stats = {}
        for category in ['models', 'loras', 'checkpoints']:
            cat_files = [f for f in all_files if f['category'] == category]
            category_stats[category] = {
                'count': len(cat_files),
                'size': sum(f['size'] for f in cat_files)
            }

        self.cached_stats = category_stats

        # Format displays
        total_size_md = f"### Total Size: **{self._format_size(total_size)}**"
        total_count_md = f"### Total Files: **{total_count}**"
        last_scan_md = f"### Last Scan: **{self.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}**"

        breakdown_html = self._render_breakdown()
        file_list_html = self._render_file_list(all_files)
        lora_library_html = self._render_lora_library()
        lora_dropdown_choices = self._get_lora_dropdown_choices()
        usage_stats_html = self._render_usage_stats()

        return (
            total_size_md,
            total_count_md,
            last_scan_md,
            breakdown_html,
            file_list_html,
            lora_library_html,
            gr.update(choices=lora_dropdown_choices),
            usage_stats_html
        )

    def _render_breakdown(self) -> str:
        """Render storage breakdown by category"""
        if not self.cached_stats:
            return "<p>No data available. Click 'Scan Storage' to begin.</p>"

        html = """
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0;">
        """

        icons = {
            'models': '🎬',
            'loras': '🎨',
            'checkpoints': '💾'
        }

        colors = {
            'models': '#4CAF50',
            'loras': '#2196F3',
            'checkpoints': '#FF9800'
        }

        for category, stats in self.cached_stats.items():
            icon = icons.get(category, '📦')
            color = colors.get(category, '#888')

            html += f"""
            <div style="border: 2px solid {color}; border-radius: 12px; padding: 20px; text-align: center; background: linear-gradient(135deg, {color}15, {color}05);">
                <div style="font-size: 48px; margin-bottom: 10px;">{icon}</div>
                <div style="font-size: 20px; font-weight: bold; color: {color}; text-transform: uppercase; margin-bottom: 10px;">
                    {category}
                </div>
                <div style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">
                    {self._format_size(stats['size'])}
                </div>
                <div style="font-size: 14px; color: #666;">
                    {stats['count']} file{'s' if stats['count'] != 1 else ''}
                </div>
            </div>
            """

        html += "</div>"
        return html

    def _render_star_rating(self, rating: int) -> str:
        """Render star rating as HTML"""
        filled = "★" * rating
        empty = "☆" * (5 - rating)
        color = "#FFD700" if rating > 0 else "#ccc"
        return f'<span style="color: {color}; font-size: 16px;">{filled}{empty}</span>'

    def _get_model_type_badge(self, model_type: str) -> str:
        """Get colored badge for model type"""
        colors = {
            "wan_t2v": "#4CAF50",
            "wan_i2v": "#8BC34A",
            "hunyuan": "#2196F3",
            "hunyuan_i2v": "#03A9F4",
            "flux": "#9C27B0",
            "ltxv": "#FF5722",
            "qwen": "#795548",
            "tts": "#607D8B"
        }
        labels = {
            "wan_t2v": "Wan T2V",
            "wan_i2v": "Wan I2V",
            "hunyuan": "Hunyuan",
            "hunyuan_i2v": "Hunyuan I2V",
            "flux": "Flux",
            "ltxv": "LTXV",
            "qwen": "Qwen",
            "tts": "TTS"
        }
        color = colors.get(model_type, "#888")
        label = labels.get(model_type, model_type)
        return f'<span style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold;">{label}</span>'

    def _render_lora_library(self, loras: Optional[List[Dict[str, Any]]] = None) -> str:
        """Render the LoRA library with metadata cards.

        Args:
            loras: Optional pre-filtered list of LoRAs. If None, uses all LoRAs from cached_models.
        """
        if loras is None:
            loras = [f for f in self.cached_models if f['category'] == 'loras']

        if not loras:
            return """
            <div style="text-align: center; padding: 60px; color: #888;">
                <div style="font-size: 48px; margin-bottom: 20px;">🎨</div>
                <p style="font-size: 18px;">No LoRAs found</p>
                <p>Add LoRA files to your loras directories and click 'Scan Storage'</p>
            </div>
            """

        html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 16px; padding: 10px;">'

        for lora in loras:
            meta = lora.get('lora_metadata', {})
            trigger_words = meta.get('trigger_words', [])
            tags = meta.get('tags', [])
            rating = meta.get('rating', 0)
            model_type = meta.get('model_type', 'unknown')
            usage_count = meta.get('usage_count', 0)
            last_used = meta.get('last_used')
            notes = meta.get('notes', '')

            # Format trigger words
            trigger_html = ""
            if trigger_words:
                trigger_html = '<div style="margin-top: 8px;"><strong style="color: #666; font-size: 11px;">TRIGGERS:</strong> '
                trigger_html += ' '.join([f'<code style="background: #e3f2fd; padding: 2px 6px; border-radius: 4px; font-size: 12px;">{w}</code>' for w in trigger_words[:5]])
                if len(trigger_words) > 5:
                    trigger_html += f' <span style="color: #888;">+{len(trigger_words) - 5} more</span>'
                trigger_html += '</div>'

            # Format tags
            tags_html = ""
            if tags:
                tags_html = '<div style="margin-top: 6px;">'
                tags_html += ' '.join([f'<span style="background: #f5f5f5; color: #666; padding: 2px 8px; border-radius: 10px; font-size: 11px;">#{t}</span>' for t in tags[:4]])
                if len(tags) > 4:
                    tags_html += f' <span style="color: #888;">+{len(tags) - 4}</span>'
                tags_html += '</div>'

            # Format last used
            last_used_str = "Never"
            if last_used:
                try:
                    last_used_dt = datetime.fromisoformat(last_used)
                    last_used_str = last_used_dt.strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    pass  # Invalid date format, keep "Never"

            # Notes preview
            notes_html = ""
            if notes:
                notes_preview = notes[:80] + "..." if len(notes) > 80 else notes
                notes_html = f'<div style="margin-top: 8px; font-style: italic; color: #666; font-size: 12px;">📝 {notes_preview}</div>'

            html += f"""
            <div style="border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 16px; background: var(--background-fill-secondary); box-shadow: var(--shadow-drop);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="font-weight: bold; font-size: 14px; word-break: break-word; flex: 1; color: var(--body-text-color);">{lora['name']}</div>
                    {self._render_star_rating(rating)}
                </div>

                <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 8px;">
                    {self._get_model_type_badge(model_type)}
                    <span style="color: var(--body-text-color-subdued); font-size: 12px;">📦 {self._format_size(lora['size'])}</span>
                    <span style="color: var(--body-text-color-subdued); font-size: 12px;">📊 Used {usage_count}x</span>
                </div>

                {trigger_html}
                {tags_html}
                {notes_html}

                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-color-primary); font-size: 11px; color: var(--body-text-color-subdued);">
                    <span>Last used: {last_used_str}</span>
                    <span style="float: right;">{lora['relative_path']}</span>
                </div>
            </div>
            """

        html += '</div>'
        return html

    def _apply_lora_filters(self, filter_by: str, search: str, sort_by: str) -> str:
        """Apply filters to LoRA library"""
        loras = [f for f in self.cached_models if f['category'] == 'loras']

        if not loras:
            return self._render_lora_library()

        # Apply filter
        if filter_by == "Wan T2V":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'wan_t2v']
        elif filter_by == "Wan I2V":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'wan_i2v']
        elif filter_by == "Hunyuan":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'hunyuan']
        elif filter_by == "Hunyuan I2V":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'hunyuan_i2v']
        elif filter_by == "Flux":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'flux']
        elif filter_by == "LTXV":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'ltxv']
        elif filter_by == "Qwen":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'qwen']
        elif filter_by == "TTS":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('model_type') == 'tts']
        elif filter_by == "★★★★★ (5 stars)":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('rating', 0) == 5]
        elif filter_by == "★★★★ (4+ stars)":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('rating', 0) >= 4]
        elif filter_by == "★★★ (3+ stars)":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('rating', 0) >= 3]
        elif filter_by == "Unrated":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('rating', 0) == 0]
        elif filter_by == "Recently Used":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('last_used')]
            loras.sort(key=lambda x: x.get('lora_metadata', {}).get('last_used', ''), reverse=True)
        elif filter_by == "Most Used":
            loras.sort(key=lambda x: x.get('lora_metadata', {}).get('usage_count', 0), reverse=True)
        elif filter_by == "Never Used":
            loras = [l for l in loras if l.get('lora_metadata', {}).get('usage_count', 0) == 0]

        # Apply search
        if search and search.strip():
            search_lower = search.lower().strip()
            filtered = []
            for lora in loras:
                meta = lora.get('lora_metadata', {})
                # Search in name
                if search_lower in lora['name'].lower():
                    filtered.append(lora)
                    continue
                # Search in trigger words
                trigger_words = meta.get('trigger_words', [])
                if any(search_lower in tw.lower() for tw in trigger_words):
                    filtered.append(lora)
                    continue
                # Search in tags
                tags = meta.get('tags', [])
                if any(search_lower in tag.lower() for tag in tags):
                    filtered.append(lora)
                    continue
                # Search in notes
                notes = meta.get('notes', '')
                if search_lower in notes.lower():
                    filtered.append(lora)
                    continue
            loras = filtered

        # Apply sort
        if sort_by == "Name (A-Z)":
            loras.sort(key=lambda x: x['name'].lower())
        elif sort_by == "Name (Z-A)":
            loras.sort(key=lambda x: x['name'].lower(), reverse=True)
        elif sort_by == "Rating (High)":
            loras.sort(key=lambda x: x.get('lora_metadata', {}).get('rating', 0), reverse=True)
        elif sort_by == "Rating (Low)":
            loras.sort(key=lambda x: x.get('lora_metadata', {}).get('rating', 0))
        elif sort_by == "Most Used":
            loras.sort(key=lambda x: x.get('lora_metadata', {}).get('usage_count', 0), reverse=True)
        elif sort_by == "Recently Used":
            loras.sort(key=lambda x: x.get('lora_metadata', {}).get('last_used', ''), reverse=True)
        elif sort_by == "Size (Largest)":
            loras.sort(key=lambda x: x['size'], reverse=True)
        elif sort_by == "Size (Smallest)":
            loras.sort(key=lambda x: x['size'])

        # Pass filtered loras directly to render method
        return self._render_lora_library(loras)

    def _get_lora_dropdown_choices(self) -> List[Tuple[str, str]]:
        """Get dropdown choices for LoRA selector as (label, value) tuples"""
        loras = [f for f in self.cached_models if f['category'] == 'loras']
        # Return tuples of (display_label, path_value) for robust selection
        return [(f"{l['name']} ({self._format_size(l['size'])})", l['path'])
                for l in sorted(loras, key=lambda x: x['name'].lower())]

    def _load_lora_for_editing(self, selected: str) -> Tuple:
        """Load a LoRA's metadata for editing"""
        if not selected:
            return (
                "*Select a LoRA from the dropdown above*",
                None,
                "",
                "",
                "wan_t2v",
                0,
                "",
                ""
            )

        # The dropdown now returns the path directly as the value
        path = selected

        # Find the LoRA
        lora = None
        for f in self.cached_models:
            if f['path'] == path:
                lora = f
                break

        if not lora:
            return (
                f"*LoRA not found: {path}*",
                None,
                "",
                "",
                "wan_t2v",
                0,
                "",
                ""
            )

        meta = lora.get('lora_metadata', self._get_lora_metadata(path))

        return (
            f"**Editing:** `{lora['name']}`\n\n**Path:** `{lora['path']}`\n\n**Size:** {self._format_size(lora['size'])}",
            path,
            ", ".join(meta.get('trigger_words', [])),
            ", ".join(meta.get('tags', [])),
            meta.get('model_type', 'wan_t2v'),
            meta.get('rating', 0),
            meta.get('source_url', ''),
            meta.get('notes', '')
        )

    def _save_lora_metadata_ui(
        self,
        lora_path: str,
        trigger_words: str,
        tags: str,
        model_type: str,
        rating: int,
        source_url: str,
        notes: str
    ) -> Tuple[str, str]:
        """Save LoRA metadata from UI"""
        if not lora_path:
            return "⚠️ No LoRA selected", self._render_lora_library()

        # Parse comma-separated values
        trigger_list = [w.strip() for w in trigger_words.split(",") if w.strip()]
        tags_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Update metadata
        self._update_lora_metadata(lora_path, {
            'trigger_words': trigger_list,
            'tags': tags_list,
            'model_type': model_type,
            'rating': int(rating),
            'source_url': source_url.strip(),
            'notes': notes.strip()
        })

        # Update cached data
        for f in self.cached_models:
            if f['path'] == lora_path:
                f['lora_metadata'] = self._get_lora_metadata(lora_path)
                break

        return f"✅ Saved metadata for `{Path(lora_path).name}`", self._render_lora_library()

    def _clear_lora_form(self) -> Tuple:
        """Clear the LoRA editing form"""
        return (
            "*Select a LoRA from the dropdown above to edit*",
            None,
            "",
            "",
            "wan_t2v",
            0,
            "",
            "",
            ""
        )

    def _render_usage_stats(self) -> str:
        """Render LoRA usage statistics"""
        loras = [f for f in self.cached_models if f['category'] == 'loras']

        if not loras:
            return "<p>No LoRAs found. Scan storage first.</p>"

        # Calculate statistics
        total_loras = len(loras)
        rated_loras = len([l for l in loras if l.get('lora_metadata', {}).get('rating', 0) > 0])
        used_loras = len([l for l in loras if l.get('lora_metadata', {}).get('usage_count', 0) > 0])
        total_usage = sum(l.get('lora_metadata', {}).get('usage_count', 0) for l in loras)

        # Most used
        most_used = sorted(loras, key=lambda x: x.get('lora_metadata', {}).get('usage_count', 0), reverse=True)[:5]

        # Highest rated
        highest_rated = sorted(
            [l for l in loras if l.get('lora_metadata', {}).get('rating', 0) > 0],
            key=lambda x: x.get('lora_metadata', {}).get('rating', 0),
            reverse=True
        )[:5]

        # By model type
        model_type_counts = {}
        for lora in loras:
            mt = lora.get('lora_metadata', {}).get('model_type', 'unknown')
            model_type_counts[mt] = model_type_counts.get(mt, 0) + 1

        html = f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px;">
            <div style="background: #e3f2fd; padding: 16px; border-radius: 8px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #1976D2;">{total_loras}</div>
                <div style="color: #666;">Total LoRAs</div>
            </div>
            <div style="background: #fff3e0; padding: 16px; border-radius: 8px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #F57C00;">{rated_loras}</div>
                <div style="color: #666;">Rated</div>
            </div>
            <div style="background: #e8f5e9; padding: 16px; border-radius: 8px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #388E3C;">{used_loras}</div>
                <div style="color: #666;">Used</div>
            </div>
            <div style="background: #fce4ec; padding: 16px; border-radius: 8px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #C2185B;">{total_usage}</div>
                <div style="color: #666;">Total Uses</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div>
                <h4 style="margin-bottom: 10px;">🏆 Most Used</h4>
                <table style="width: 100%; font-size: 13px;">
        """

        for lora in most_used:
            usage = lora.get('lora_metadata', {}).get('usage_count', 0)
            html += f"""
                <tr>
                    <td style="padding: 4px 0;">{lora['name'][:30]}{'...' if len(lora['name']) > 30 else ''}</td>
                    <td style="text-align: right; color: #1976D2; font-weight: bold;">{usage}x</td>
                </tr>
            """

        html += """
                </table>
            </div>
            <div>
                <h4 style="margin-bottom: 10px;">⭐ Highest Rated</h4>
                <table style="width: 100%; font-size: 13px;">
        """

        for lora in highest_rated:
            rating = lora.get('lora_metadata', {}).get('rating', 0)
            html += f"""
                <tr>
                    <td style="padding: 4px 0;">{lora['name'][:30]}{'...' if len(lora['name']) > 30 else ''}</td>
                    <td style="text-align: right;">{self._render_star_rating(rating)}</td>
                </tr>
            """

        if not highest_rated:
            html += '<tr><td colspan="2" style="color: #888;">No rated LoRAs yet</td></tr>'

        html += """
                </table>
            </div>
        </div>

        <div style="margin-top: 20px;">
            <h4 style="margin-bottom: 10px;">📊 By Model Type</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        """

        for model_type, count in sorted(model_type_counts.items(), key=lambda x: x[1], reverse=True):
            html += f"""
                <div style="display: flex; align-items: center; gap: 8px;">
                    {self._get_model_type_badge(model_type)}
                    <span style="font-weight: bold;">{count}</span>
                </div>
            """

        html += """
            </div>
        </div>
        """

        return html

    def _render_file_list(self, files: List[Dict[str, Any]], search: str = "") -> str:
        """Render the detailed file list as HTML table"""
        if not files:
            return "<p style='text-align: center; padding: 40px; color: #888;'>No files found. Click 'Scan Storage' to begin.</p>"

        html = """
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="background: var(--background-fill-secondary); border-bottom: 2px solid var(--border-color-primary);">
                    <th style="padding: 12px; text-align: left; color: var(--body-text-color);">📁 File Name</th>
                    <th style="padding: 12px; text-align: left; color: var(--body-text-color);">📂 Category</th>
                    <th style="padding: 12px; text-align: right; color: var(--body-text-color);">💾 Size</th>
                    <th style="padding: 12px; text-align: left; color: var(--body-text-color);">📅 Modified</th>
                    <th style="padding: 12px; text-align: left; color: var(--body-text-color);">📍 Path</th>
                </tr>
            </thead>
            <tbody>
        """

        colors = {
            'models': '#4CAF50',
            'loras': '#2196F3',
            'checkpoints': '#FF9800'
        }

        for i, file in enumerate(files):
            bg_color = "var(--background-fill-primary)" if i % 2 == 0 else "var(--background-fill-secondary)"
            category_color = colors.get(file['category'], '#888')

            # Highlight large files
            size_style = "color: var(--body-text-color);"
            if file['size'] > 10 * 1024 * 1024 * 1024:  # > 10GB
                size_style = "color: #f44336; font-weight: bold;"
            elif file['size'] > 5 * 1024 * 1024 * 1024:  # > 5GB
                size_style = "color: #ff9800; font-weight: bold;"

            html += f"""
            <tr style="background: {bg_color}; border-bottom: 1px solid var(--border-color-primary);">
                <td style="padding: 12px; font-family: monospace; color: var(--body-text-color);">
                    <strong>{file['name']}</strong>
                </td>
                <td style="padding: 12px;">
                    <span style="background: {category_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                        {file['category'].upper()}
                    </span>
                </td>
                <td style="padding: 12px; text-align: right; {size_style}">
                    {self._format_size(file['size'])}
                </td>
                <td style="padding: 12px; color: var(--body-text-color-subdued);">
                    {file['modified'].strftime('%Y-%m-%d %H:%M')}
                </td>
                <td style="padding: 12px; font-size: 12px; color: var(--body-text-color-subdued); word-break: break-all;">
                    {file['relative_path']}
                </td>
            </tr>
            """

        html += """
            </tbody>
        </table>
        </div>
        """

        return html

    def _filter_files(
        self,
        files: List[Dict[str, Any]],
        filter_by: str,
        search: str
    ) -> List[Dict[str, Any]]:
        """Apply category/size filter and search to a list of files.

        Args:
            files: List of file dictionaries to filter
            filter_by: Filter category ("All", "Models", "LoRAs", "Checkpoints", "> 1GB", etc.)
            search: Search string to match against name and path

        Returns:
            Filtered list of files
        """
        result = files.copy()

        # Apply search filter
        if search and search.strip():
            search_lower = search.lower().strip()
            result = [f for f in result if search_lower in f['name'].lower() or search_lower in f['relative_path'].lower()]

        # Apply category/size filter
        filter_map = {
            "Models": lambda f: f['category'] == 'models',
            "LoRAs": lambda f: f['category'] == 'loras',
            "Checkpoints": lambda f: f['category'] == 'checkpoints',
            "> 1GB": lambda f: f['size'] > 1024 * 1024 * 1024,
            "> 5GB": lambda f: f['size'] > 5 * 1024 * 1024 * 1024,
            "> 10GB": lambda f: f['size'] > 10 * 1024 * 1024 * 1024,
        }

        if filter_by in filter_map:
            result = [f for f in result if filter_map[filter_by](f)]

        return result

    def _sort_files(
        self,
        files: List[Dict[str, Any]],
        sort_by: str
    ) -> List[Dict[str, Any]]:
        """Sort a list of files by the specified criteria.

        Args:
            files: List of file dictionaries to sort
            sort_by: Sort criteria ("Size (Largest)", "Name (A-Z)", etc.)

        Returns:
            Sorted list of files (sorted in place, also returns reference)
        """
        sort_map = {
            "Size (Largest)": (lambda x: x['size'], True),
            "Size (Smallest)": (lambda x: x['size'], False),
            "Name (A-Z)": (lambda x: x['name'].lower(), False),
            "Name (Z-A)": (lambda x: x['name'].lower(), True),
            "Date (Newest)": (lambda x: x['modified'], True),
            "Date (Oldest)": (lambda x: x['modified'], False),
        }

        if sort_by in sort_map:
            key_func, reverse = sort_map[sort_by]
            files.sort(key=key_func, reverse=reverse)

        return files

    def _apply_filter_and_sort(self, filter_by: str, sort_by: str, search: str) -> str:
        """Apply filters and sorting to the file list"""
        if not self.cached_models:
            return "<p style='text-align: center; padding: 40px; color: #888;'>No data available. Click 'Scan Storage' to begin.</p>"

        files = self._filter_files(self.cached_models, filter_by, search)
        files = self._sort_files(files, sort_by)

        return self._render_file_list(files)

    def _export_list(self, filter_by: str, sort_by: str, search: str) -> str:
        """Export the current file list to JSON"""
        if not self.cached_models:
            return "⚠️ No data to export. Please scan storage first."

        # Use shared filter helper
        files = self._filter_files(self.cached_models, filter_by, search)

        # Create export data
        export_data = {
            'scan_date': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'total_files': len(files),
            'total_size': sum(f['size'] for f in files),
            'total_size_formatted': self._format_size(sum(f['size'] for f in files)),
            'files': [
                {
                    'name': f['name'],
                    'category': f['category'],
                    'size': f['size'],
                    'size_formatted': self._format_size(f['size']),
                    'modified': f['modified'].isoformat(),
                    'path': f['path'],
                    'relative_path': f['relative_path'],
                    # Include LoRA metadata if present
                    'lora_metadata': f.get('lora_metadata') if f['category'] == 'loras' else None
                }
                for f in files
            ]
        }

        # Save to file
        output_dir = Path.home() / ".wan2gp" / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_storage_{timestamp}.json"
        output_path = output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return f"✅ Exported {len(files)} files to: `{output_path}`"
        except Exception as e:
            return f"❌ Export failed: {str(e)}"

    def on_tab_select(self, state: Dict[str, Any]) -> Tuple:
        """Called when the Storage tab is selected - trigger initial scan"""
        if not self.cached_models:
            return self._scan_storage()
        else:
            # Return cached data
            total_size = sum(f['size'] for f in self.cached_models)
            total_count = len(self.cached_models)

            total_size_md = f"### Total Size: **{self._format_size(total_size)}**"
            total_count_md = f"### Total Files: **{total_count}**"
            last_scan_md = f"### Last Scan: **{self.last_scan_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_scan_time else 'Never'}**"

            breakdown_html = self._render_breakdown()
            file_list_html = self._render_file_list(self.cached_models)
            lora_library_html = self._render_lora_library()
            lora_dropdown_choices = self._get_lora_dropdown_choices()
            usage_stats_html = self._render_usage_stats()

            return (
                total_size_md,
                total_count_md,
                last_scan_md,
                breakdown_html,
                file_list_html,
                lora_library_html,
                gr.update(choices=lora_dropdown_choices),
                usage_stats_html
            )

    def on_tab_deselect(self, state: Dict[str, Any]) -> None:
        """Called when leaving the Storage tab"""
        pass

    # Public API for other plugins to track LoRA usage
    def record_lora_usage(self, lora_path: str):
        """Record that a LoRA was used (can be called by other plugins)"""
        if lora_path in self.lora_metadata.get("loras", {}):
            current = self.lora_metadata["loras"][lora_path]
            current['usage_count'] = current.get('usage_count', 0) + 1
            current['last_used'] = datetime.now().isoformat()
            self._save_lora_metadata()
