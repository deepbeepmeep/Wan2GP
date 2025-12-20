"""
Wan2GP Model Tracker Plugin

A plugin that tracks which models are downloaded vs. missing, displays performance metrics
(VRAM/speed/quality), and helps users choose the right model for their needs.
"""

import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import pandas as pd
from typing import Optional

# Lazy imports - will be imported when needed to avoid startup issues
_ModelAnalyzer = None
_PerformanceDatabase = None
_ui_components = None


def _ensure_imports():
    """Lazy load all imports to avoid blocking during plugin discovery"""
    global _ModelAnalyzer, _PerformanceDatabase, _ui_components

    if _ModelAnalyzer is not None:
        return

    try:
        from .model_analyzer import ModelAnalyzer
        from .performance_db import PerformanceDatabase
        from . import ui_components
    except ImportError:
        # Handle relative imports when loaded as plugin
        from model_analyzer import ModelAnalyzer
        from performance_db import PerformanceDatabase
        import ui_components

    _ModelAnalyzer = ModelAnalyzer
    _PerformanceDatabase = PerformanceDatabase
    _ui_components = ui_components


class ModelTrackerPlugin(WAN2GPPlugin):
    """
    Wan2GP Plugin for tracking model status and performance metrics

    This plugin provides:
    - Model download status tracking (downloaded/missing/partial)
    - Performance metrics display (VRAM/speed/quality)
    - Smart filtering and sorting
    - Detailed model information
    - Export functionality
    """

    def __init__(self):
        """Initialize the plugin"""
        super().__init__()
        _ensure_imports()  # Load dependencies

        self.name = "Model Tracker"
        self.version = "1.0.0"
        self.description = "Track which models are downloaded and view performance metrics"

        self.analyzer = None

    def setup_ui(self):
        """Setup the plugin UI by adding a tab"""
        # Request access to global variables needed by the analyzer
        self.request_global("models_def")
        self.request_global("files_locator")
        self.request_global("get_local_model_filename")

        # Request access to model selector component
        self.request_component("model_choice")
        self.request_component("model_family")

        # Add the Models tab
        self.add_tab(
            tab_id="model_tracker_tab",
            label="Models",
            component_constructor=self.create_ui,
            position=2
        )

    def _initialize_analyzer(self):
        """Initialize the ModelAnalyzer with Wan2GP's model system"""
        import os
        try:
            # Access Wan2GP's model definitions and utilities
            models_def = getattr(self, 'models_def', {})
            files_locator = getattr(self, 'files_locator', None)
            get_local_model_filename = getattr(self, 'get_local_model_filename', None)

            # Write debug info to file
            debug_info = f"""[Model Tracker] Initializing analyzer...
[Model Tracker] - models_def: {len(models_def)} models found
[Model Tracker] - files_locator: {'Available' if files_locator else 'Not available'}
[Model Tracker] - get_local_model_filename: {'Available' if get_local_model_filename else 'Not available'}
"""
            with open('model_tracker_debug.txt', 'w') as f:
                f.write(debug_info)

            self.analyzer = _ModelAnalyzer(
                models_def=models_def,
                files_locator=files_locator,
                get_local_model_filename_func=get_local_model_filename or self._fallback_get_filename
            )

        except Exception as e:
            debug_info = f"[Model Tracker] Error initializing analyzer: {e}\n"
            import traceback
            debug_info += traceback.format_exc()
            with open('model_tracker_debug.txt', 'w') as f:
                f.write(debug_info)
            # Create analyzer with empty models_def as fallback
            self.analyzer = _ModelAnalyzer({}, None, self._fallback_get_filename)

    def _fallback_get_filename(self, url: str) -> str:
        """Fallback function for getting local filename from URL"""
        return url.split('/')[-1]

    def post_ui_setup(self, components):
        """Add model status indicator to video generator page"""
        # Guard against duplicate setup
        if hasattr(self, '_status_indicator_setup'):
            return {}
        self._status_indicator_setup = True

        if not hasattr(self, 'model_choice') or self.model_choice is None:
            return {}

        # Initialize analyzer if not already done
        if self.analyzer is None:
            self._initialize_analyzer()

        # Insert status display after model selector
        def create_status_display():
            self.model_status_display = gr.Markdown(
                "",
                elem_id="model_status_indicator",
                visible=True
            )

            # Hook up event to update status when model changes
            def update_model_status(model_choice):
                """Update status display when model selection changes"""
                if not model_choice or not self.analyzer:
                    return ""

                status = self.analyzer.get_model_status(model_choice)

                if status == "downloaded":
                    return "✓ **Model Downloaded**"
                elif status == "partial":
                    return "◐ **Partially Downloaded** - Some files missing"
                elif status == "missing":
                    return "✗ **Model Not Downloaded**"
                else:
                    return ""

            self.model_choice.change(
                fn=update_model_status,
                inputs=[self.model_choice],
                outputs=[self.model_status_display],
                show_progress=False
            )

        self.insert_after("model_choice", create_status_display)

        return {}

    def create_ui(self):
        """
        Create the Gradio UI for the plugin

        Returns:
            Gradio interface
        """
        # Initialize the analyzer when creating the UI
        self._initialize_analyzer()

        filter_choices = _ui_components.get_filter_choices()

        with gr.Blocks() as ui:
            gr.Markdown("# Model Tracker")
            gr.Markdown("Track which models are downloaded and view performance metrics")

            # Summary statistics
            with gr.Row():
                summary_display = gr.Markdown(
                    self._get_summary_markdown(),
                    elem_id="summary_stats"
                )

            # Filters and controls
            with gr.Row():
                status_filter = gr.Dropdown(
                    choices=filter_choices["status"],
                    value="all",
                    label="Filter by Status",
                    scale=1
                )
                speed_filter = gr.Dropdown(
                    choices=filter_choices["speed"],
                    value="all",
                    label="Filter by Speed",
                    scale=1
                )
                vram_filter = gr.Dropdown(
                    choices=filter_choices["vram"],
                    value="all",
                    label="Filter by VRAM",
                    scale=1
                )
                sort_by = gr.Dropdown(
                    choices=filter_choices["sort"],
                    value="name",
                    label="Sort By",
                    scale=1
                )

            # Action buttons
            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="primary", scale=1)
                export_btn = gr.Button("Export Report", variant="secondary", scale=1)

            # Model table
            with gr.Row():
                model_table = gr.Dataframe(
                    value=self._get_table_data(),
                    interactive=False,
                    wrap=True
                )

            # Model details section
            gr.Markdown("---")
            gr.Markdown("## Model Details")

            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["Select a model..."] + self._get_model_choices(),
                    value="Select a model...",
                    label="Select Model for Details",
                    scale=3
                )

            with gr.Row():
                model_details = gr.Markdown(
                    "Select a model from the dropdown to view details.",
                    elem_id="model_details"
                )

            # Export status message
            export_status = gr.Markdown("", visible=False)

            # Event handlers
            def update_table(status_f, speed_f, vram_f, sort):
                """Update table based on filters"""
                return self._get_table_data(status_f, speed_f, vram_f, sort)

            def update_details(model_type):
                """Update details display"""
                if not model_type or model_type == "Select a model...":
                    return "Select a model from the dropdown to view details."
                return _ui_components.format_model_details(self.analyzer, model_type)

            def refresh_all():
                """Refresh all data"""
                self._initialize_analyzer()  # Re-scan models
                summary = self._get_summary_markdown()
                table = self._get_table_data()
                choices = ["Select a model..."] + self._get_model_choices()
                return summary, table, gr.Dropdown(choices=choices, value="Select a model...")

            def export_report():
                """Export model inventory report"""
                result = _ui_components.export_model_report(self.analyzer)
                return gr.Markdown(value=f"✓ {result}", visible=True)

            # Connect event handlers
            for component in [status_filter, speed_filter, vram_filter, sort_by]:
                component.change(
                    fn=update_table,
                    inputs=[status_filter, speed_filter, vram_filter, sort_by],
                    outputs=model_table
                )

            model_selector.change(
                fn=update_details,
                inputs=model_selector,
                outputs=model_details
            )

            refresh_btn.click(
                fn=refresh_all,
                inputs=[],
                outputs=[summary_display, model_table, model_selector]
            )

            export_btn.click(
                fn=export_report,
                inputs=[],
                outputs=export_status
            )

        return ui

    # Helper methods for UI data

    def _get_summary_markdown(self) -> str:
        """Get formatted summary statistics"""
        if not self.analyzer:
            return "**Error:** Analyzer not initialized"

        try:
            stats = self.analyzer.get_summary_stats()
            return _ui_components.format_summary_stats(stats)
        except Exception as e:
            return f"**Error getting summary:** {e}"

    def _get_table_data(self, status_filter: str = "all",
                       speed_filter: str = "all",
                       vram_filter: str = "all",
                       sort_by: str = "name"):
        """Get model table data with filters applied"""
        if not self.analyzer:
            return pd.DataFrame()

        try:
            all_models = self.analyzer.get_all_models()
            return _ui_components.create_model_table_data(
                self.analyzer,
                all_models,
                status_filter,
                speed_filter,
                vram_filter,
                sort_by
            )
        except Exception as e:
            print(f"[Model Tracker] Error getting table data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _get_model_choices(self) -> list:
        """Get list of models for dropdown"""
        if not self.analyzer:
            return []

        try:
            all_models = self.analyzer.get_all_models()
            # Return sorted by display name
            model_tuples = [(m, self.analyzer.get_model_name(m)) for m in all_models]
            model_tuples.sort(key=lambda x: x[1])
            return [m[0] for m in model_tuples]
        except Exception as e:
            print(f"[Model Tracker] Error getting model choices: {e}")
            return []


# Alias for compatibility
WAN2GPPlugin = ModelTrackerPlugin
