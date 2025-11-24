"""
Storyboard Plugin for Wan2GP

Plan and generate multi-scene video projects with visual timeline,
transitions, and automatic concatenation.
"""

import time
import json
import gradio as gr
from pathlib import Path
from typing import Dict, Any, Optional, List

from shared.utils.plugins import WAN2GPPlugin

from .storyboard import (
    StoryboardManager,
    StoryboardProject,
    Scene,
    Transition,
    GlobalSettings
)
from .transitions import (
    TRANSITIONS,
    get_transition_ids,
    get_transition_names,
    SceneConcatenator
)


class StoryboardPlugin(WAN2GPPlugin):
    """Multi-scene video planning and generation plugin"""

    def __init__(self):
        super().__init__()
        self.name = "Storyboard"
        self.version = "1.0.0"
        self.description = "Plan and generate multi-scene videos with transitions"

        self.manager = StoryboardManager()
        self.concatenator = SceneConcatenator()
        self.selected_scene_id: Optional[str] = None

    def setup_ui(self):
        # Request main components
        self.request_component("state")
        self.request_component("main_tabs")

        # Request global functions for queue integration
        self.request_global("get_model_def")
        self.request_global("server_config")

        # Add custom JS for timeline interactivity
        self.add_custom_js(self._get_timeline_js())

        # Add the storyboard tab
        self.add_tab(
            tab_id="storyboard",
            label="Storyboard",
            component_constructor=self._build_ui,
        )

    def _build_ui(self):
        """Build the storyboard UI"""
        # Hidden state for selected scene
        self.scene_id_state = gr.State(value=None)

        with gr.Column(elem_id="storyboard_plugin"):
            # Project management row
            with gr.Row():
                self.project_name = gr.Textbox(
                    label="Project Name",
                    value="Untitled Project",
                    scale=3
                )
                self.new_btn = gr.Button("New", scale=1)
                self.load_btn = gr.UploadButton(
                    "Load",
                    file_types=[".json"],
                    scale=1
                )
                self.save_btn = gr.Button("Save", scale=1)

            # Global settings accordion
            with gr.Accordion("Global Settings", open=False):
                with gr.Row():
                    self.global_model = gr.Dropdown(
                        label="Model",
                        choices=self._get_model_choices(),
                        value="wan2.1-t2v-14B",
                        scale=2
                    )
                    self.global_resolution = gr.Dropdown(
                        label="Resolution",
                        choices=["1280x720", "1024x576", "832x480", "768x512", "640x480"],
                        value="1280x720",
                        scale=1
                    )
                    self.global_fps = gr.Dropdown(
                        label="FPS",
                        choices=["8", "12", "16", "24"],
                        value="16",
                        scale=1
                    )

                with gr.Row():
                    self.seed_mode = gr.Radio(
                        choices=["fixed", "increment", "random"],
                        value="fixed",
                        label="Seed Mode",
                        scale=2
                    )
                    self.global_seed = gr.Number(
                        label="Base Seed",
                        value=42,
                        precision=0,
                        scale=1
                    )

                with gr.Row():
                    self.global_guidance = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                    self.global_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=30,
                        step=1,
                        label="Inference Steps"
                    )

            # Timeline section
            gr.Markdown("### Timeline")
            with gr.Row():
                self.timeline_html = gr.HTML(
                    value=self._render_timeline(),
                    elem_id="storyboard_timeline"
                )

            # Add scene button and summary
            with gr.Row():
                self.add_scene_btn = gr.Button("+ Add Scene", variant="primary")
                self.duration_display = gr.Markdown("Total: 0 scenes, 0.0 seconds")

            # Scene editor panel
            with gr.Accordion("Scene Editor", open=True, visible=False) as self.scene_editor:
                with gr.Row():
                    self.scene_name = gr.Textbox(
                        label="Scene Name",
                        scale=3
                    )
                    self.scene_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=1
                    )

                self.scene_prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Describe what happens in this scene..."
                )

                self.scene_negative = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    placeholder="What to avoid..."
                )

                with gr.Row():
                    self.scene_duration = gr.Slider(
                        minimum=1.0,
                        maximum=30.0,
                        value=5.0,
                        step=0.5,
                        label="Duration (seconds)"
                    )
                    self.scene_frames = gr.Number(
                        label="Frames",
                        interactive=False,
                        value=80
                    )

                self.scene_reference = gr.Image(
                    label="Reference Image (for I2V)",
                    type="filepath",
                    height=150
                )

                with gr.Row():
                    self.transition_type = gr.Dropdown(
                        choices=get_transition_ids(),
                        value="cut",
                        label="Transition to Next Scene"
                    )
                    self.transition_duration = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=0.0,
                        step=0.25,
                        label="Transition Duration (s)"
                    )

                with gr.Row():
                    self.move_up_btn = gr.Button("Move Up")
                    self.move_down_btn = gr.Button("Move Down")
                    self.duplicate_btn = gr.Button("Duplicate")
                    self.delete_scene_btn = gr.Button("Delete", variant="stop")

            # Action buttons
            gr.Markdown("### Actions")
            with gr.Row():
                self.queue_all_btn = gr.Button(
                    "Queue All Scenes",
                    variant="primary"
                )
                self.queue_remaining_btn = gr.Button("Queue Remaining")
                self.concatenate_btn = gr.Button("Concatenate Completed")

            with gr.Row():
                self.import_csv_btn = gr.UploadButton(
                    "Import CSV",
                    file_types=[".csv"]
                )
                self.export_csv_btn = gr.Button("Export CSV")

            # Status/output
            self.status_md = gr.Markdown("Ready. Create or load a project to begin.")

            # Hidden trigger for scene selection from JS
            self.scene_select_trigger = gr.Textbox(
                visible=False,
                elem_id="storyboard_scene_select"
            )

        # Wire up all events
        self._setup_events()

    def _setup_events(self):
        """Setup all event handlers"""

        # Project management
        self.new_btn.click(
            fn=self._new_project,
            inputs=[self.project_name],
            outputs=[
                self.timeline_html,
                self.duration_display,
                self.status_md,
                self.scene_editor
            ]
        )

        self.save_btn.click(
            fn=self._save_project,
            inputs=[self.project_name],
            outputs=[self.status_md]
        )

        self.load_btn.upload(
            fn=self._load_project,
            inputs=[self.load_btn],
            outputs=[
                self.project_name,
                self.timeline_html,
                self.duration_display,
                self.status_md,
                self.scene_editor,
                self.global_model,
                self.global_resolution,
                self.global_fps,
                self.seed_mode,
                self.global_seed,
                self.global_guidance,
                self.global_steps
            ]
        )

        # Add scene
        self.add_scene_btn.click(
            fn=self._add_scene,
            inputs=[],
            outputs=[
                self.timeline_html,
                self.duration_display,
                self.scene_editor,
                self.scene_id_state
            ]
        ).then(
            fn=self._load_scene_to_editor,
            inputs=[self.scene_id_state],
            outputs=[
                self.scene_name,
                self.scene_prompt,
                self.scene_negative,
                self.scene_duration,
                self.scene_frames,
                self.scene_reference,
                self.transition_type,
                self.transition_duration,
                self.scene_status
            ]
        )

        # Scene selection from timeline (via hidden textbox)
        self.scene_select_trigger.change(
            fn=self._on_scene_select,
            inputs=[self.scene_select_trigger],
            outputs=[
                self.scene_editor,
                self.scene_id_state
            ]
        ).then(
            fn=self._load_scene_to_editor,
            inputs=[self.scene_id_state],
            outputs=[
                self.scene_name,
                self.scene_prompt,
                self.scene_negative,
                self.scene_duration,
                self.scene_frames,
                self.scene_reference,
                self.transition_type,
                self.transition_duration,
                self.scene_status
            ]
        )

        # Scene editing - auto-save on change
        scene_edit_outputs = [self.timeline_html, self.duration_display]

        self.scene_name.change(
            fn=self._update_scene_field,
            inputs=[self.scene_id_state, gr.State("name"), self.scene_name],
            outputs=scene_edit_outputs
        )

        self.scene_prompt.change(
            fn=self._update_scene_field,
            inputs=[self.scene_id_state, gr.State("prompt"), self.scene_prompt],
            outputs=scene_edit_outputs
        )

        self.scene_negative.change(
            fn=self._update_scene_field,
            inputs=[self.scene_id_state, gr.State("negative_prompt"), self.scene_negative],
            outputs=scene_edit_outputs
        )

        self.scene_duration.change(
            fn=self._update_scene_duration,
            inputs=[self.scene_id_state, self.scene_duration, self.global_fps],
            outputs=[self.timeline_html, self.duration_display, self.scene_frames]
        )

        self.transition_type.change(
            fn=self._update_transition,
            inputs=[self.scene_id_state, self.transition_type, self.transition_duration, self.global_fps],
            outputs=scene_edit_outputs
        )

        self.transition_duration.change(
            fn=self._update_transition,
            inputs=[self.scene_id_state, self.transition_type, self.transition_duration, self.global_fps],
            outputs=scene_edit_outputs
        )

        # Scene actions
        self.move_up_btn.click(
            fn=self._move_scene,
            inputs=[self.scene_id_state, gr.State(-1)],
            outputs=[self.timeline_html, self.duration_display]
        )

        self.move_down_btn.click(
            fn=self._move_scene,
            inputs=[self.scene_id_state, gr.State(1)],
            outputs=[self.timeline_html, self.duration_display]
        )

        self.duplicate_btn.click(
            fn=self._duplicate_scene,
            inputs=[self.scene_id_state],
            outputs=[
                self.timeline_html,
                self.duration_display,
                self.scene_id_state
            ]
        ).then(
            fn=self._load_scene_to_editor,
            inputs=[self.scene_id_state],
            outputs=[
                self.scene_name,
                self.scene_prompt,
                self.scene_negative,
                self.scene_duration,
                self.scene_frames,
                self.scene_reference,
                self.transition_type,
                self.transition_duration,
                self.scene_status
            ]
        )

        self.delete_scene_btn.click(
            fn=self._delete_scene,
            inputs=[self.scene_id_state],
            outputs=[
                self.timeline_html,
                self.duration_display,
                self.scene_editor,
                self.scene_id_state
            ]
        )

        # Queue actions
        self.queue_all_btn.click(
            fn=self._queue_all_scenes,
            inputs=[self.state, self.global_model, self.global_resolution,
                   self.global_fps, self.seed_mode, self.global_seed,
                   self.global_guidance, self.global_steps],
            outputs=[self.timeline_html, self.status_md]
        )

        self.queue_remaining_btn.click(
            fn=self._queue_remaining_scenes,
            inputs=[self.state, self.global_model, self.global_resolution,
                   self.global_fps, self.seed_mode, self.global_seed,
                   self.global_guidance, self.global_steps],
            outputs=[self.timeline_html, self.status_md]
        )

        self.concatenate_btn.click(
            fn=self._concatenate_completed,
            inputs=[self.global_fps],
            outputs=[self.status_md]
        )

        # Import/Export
        self.import_csv_btn.upload(
            fn=self._import_csv,
            inputs=[self.import_csv_btn],
            outputs=[self.timeline_html, self.duration_display, self.status_md]
        )

        self.export_csv_btn.click(
            fn=self._export_csv,
            inputs=[],
            outputs=[self.status_md]
        )

        # Global settings changes update project
        for component in [self.global_model, self.global_resolution, self.global_fps,
                         self.seed_mode, self.global_seed, self.global_guidance, self.global_steps]:
            component.change(
                fn=self._update_global_settings,
                inputs=[self.global_model, self.global_resolution, self.global_fps,
                       self.seed_mode, self.global_seed, self.global_guidance, self.global_steps],
                outputs=[]
            )

    # =========================================================================
    # Project Management
    # =========================================================================

    def _new_project(self, name: str):
        """Create a new project"""
        project = self.manager.new_project(name or "Untitled Project")
        self.selected_scene_id = None

        return (
            self._render_timeline(),
            self._get_duration_display(),
            f"Created new project: {project.name}",
            gr.Accordion(visible=False)  # Hide scene editor
        )

    def _save_project(self, name: str):
        """Save current project"""
        if self.manager.current_project is None:
            return "No project to save. Create a new project first."

        self.manager.current_project.name = name
        try:
            path = self.manager.save_project()
            return f"Project saved to: {path}"
        except Exception as e:
            return f"Error saving project: {str(e)}"

    def _load_project(self, file):
        """Load a project from file"""
        if file is None:
            return [gr.update()] * 12

        try:
            project = self.manager.load_project(file.name)
            gs = project.global_settings
            self.selected_scene_id = None

            return (
                project.name,
                self._render_timeline(),
                self._get_duration_display(),
                f"Loaded project: {project.name} ({len(project.scenes)} scenes)",
                gr.Accordion(visible=False),
                gs.model_type,
                gs.resolution,
                str(gs.fps),
                gs.seed_mode,
                gs.seed,
                gs.guidance_scale,
                gs.num_inference_steps
            )
        except Exception as e:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                f"Error loading project: {str(e)}",
                *([gr.update()] * 8)
            )

    # =========================================================================
    # Scene Management
    # =========================================================================

    def _add_scene(self):
        """Add a new scene"""
        if self.manager.current_project is None:
            self.manager.new_project("Untitled Project")

        scene = self.manager.current_project.add_scene()
        self.selected_scene_id = scene.id

        return (
            self._render_timeline(),
            self._get_duration_display(),
            gr.Accordion(visible=True),
            scene.id
        )

    def _on_scene_select(self, scene_id: str):
        """Handle scene selection from timeline"""
        if not scene_id or self.manager.current_project is None:
            return gr.Accordion(visible=False), None

        scene = self.manager.current_project.get_scene(scene_id)
        if scene is None:
            return gr.Accordion(visible=False), None

        self.selected_scene_id = scene_id
        return gr.Accordion(visible=True), scene_id

    def _load_scene_to_editor(self, scene_id: str):
        """Load scene data into editor fields"""
        if not scene_id or self.manager.current_project is None:
            return [gr.update()] * 9

        scene = self.manager.current_project.get_scene(scene_id)
        if scene is None:
            return [gr.update()] * 9

        fps = self.manager.current_project.global_settings.fps

        return (
            scene.name,
            scene.prompt,
            scene.negative_prompt,
            scene.duration_frames / fps,
            scene.duration_frames,
            scene.reference_image,
            scene.transition_to_next.type,
            scene.transition_to_next.duration_frames / fps,
            scene.status
        )

    def _update_scene_field(self, scene_id: str, field: str, value):
        """Update a single scene field"""
        if not scene_id or self.manager.current_project is None:
            return self._render_timeline(), self._get_duration_display()

        self.manager.current_project.update_scene(scene_id, **{field: value})
        return self._render_timeline(), self._get_duration_display()

    def _update_scene_duration(self, scene_id: str, duration_seconds: float, fps_str: str):
        """Update scene duration"""
        if not scene_id or self.manager.current_project is None:
            return self._render_timeline(), self._get_duration_display(), 0

        fps = int(fps_str)
        frames = int(duration_seconds * fps)

        self.manager.current_project.update_scene(scene_id, duration_frames=frames)
        return self._render_timeline(), self._get_duration_display(), frames

    def _update_transition(self, scene_id: str, trans_type: str, trans_duration: float, fps_str: str):
        """Update scene transition"""
        if not scene_id or self.manager.current_project is None:
            return self._render_timeline(), self._get_duration_display()

        fps = int(fps_str)
        scene = self.manager.current_project.get_scene(scene_id)
        if scene:
            scene.transition_to_next = Transition(
                type=trans_type,
                duration_frames=int(trans_duration * fps)
            )

        return self._render_timeline(), self._get_duration_display()

    def _move_scene(self, scene_id: str, direction: int):
        """Move scene up or down"""
        if not scene_id or self.manager.current_project is None:
            return self._render_timeline(), self._get_duration_display()

        scene = self.manager.current_project.get_scene(scene_id)
        if scene:
            new_order = scene.order + direction
            self.manager.current_project.move_scene(scene_id, new_order)

        return self._render_timeline(), self._get_duration_display()

    def _duplicate_scene(self, scene_id: str):
        """Duplicate a scene"""
        if not scene_id or self.manager.current_project is None:
            return self._render_timeline(), self._get_duration_display(), None

        new_scene = self.manager.current_project.duplicate_scene(scene_id)
        if new_scene:
            self.selected_scene_id = new_scene.id
            return self._render_timeline(), self._get_duration_display(), new_scene.id

        return self._render_timeline(), self._get_duration_display(), scene_id

    def _delete_scene(self, scene_id: str):
        """Delete a scene"""
        if not scene_id or self.manager.current_project is None:
            return self._render_timeline(), self._get_duration_display(), gr.Accordion(visible=False), None

        self.manager.current_project.remove_scene(scene_id)
        self.selected_scene_id = None

        return (
            self._render_timeline(),
            self._get_duration_display(),
            gr.Accordion(visible=False),
            None
        )

    # =========================================================================
    # Global Settings
    # =========================================================================

    def _update_global_settings(self, model, resolution, fps, seed_mode, seed, guidance, steps):
        """Update project global settings"""
        if self.manager.current_project is None:
            return

        gs = self.manager.current_project.global_settings
        gs.model_type = model
        gs.resolution = resolution
        gs.fps = int(fps)
        gs.seed_mode = seed_mode
        gs.seed = int(seed)
        gs.guidance_scale = float(guidance)
        gs.num_inference_steps = int(steps)

    # =========================================================================
    # Queue Integration
    # =========================================================================

    def _queue_all_scenes(self, state, model, resolution, fps, seed_mode, seed, guidance, steps):
        """Queue all scenes for generation"""
        return self._queue_scenes(state, model, resolution, fps, seed_mode, seed, guidance, steps, all_scenes=True)

    def _queue_remaining_scenes(self, state, model, resolution, fps, seed_mode, seed, guidance, steps):
        """Queue only pending/failed scenes"""
        return self._queue_scenes(state, model, resolution, fps, seed_mode, seed, guidance, steps, all_scenes=False)

    def _queue_scenes(self, state, model, resolution, fps_str, seed_mode, seed, guidance, steps, all_scenes=True):
        """Queue scenes for generation"""
        if self.manager.current_project is None:
            return self._render_timeline(), "No project loaded. Create or load a project first."

        project = self.manager.current_project
        fps = int(fps_str)

        if all_scenes:
            scenes_to_queue = sorted(project.scenes, key=lambda s: s.order)
        else:
            scenes_to_queue = project.get_pending_scenes()

        if not scenes_to_queue:
            return self._render_timeline(), "No scenes to queue."

        # Update global settings
        self._update_global_settings(model, resolution, fps_str, seed_mode, seed, guidance, steps)

        queued_count = 0
        messages = []

        for scene in scenes_to_queue:
            # Calculate seed based on mode
            if seed_mode == "fixed":
                scene_seed = int(seed)
            elif seed_mode == "increment":
                scene_seed = int(seed) + scene.order
            else:  # random
                scene_seed = -1

            # Mark as queued
            scene.status = "queued"
            queued_count += 1

            # Build info message (actual queueing would require deeper integration)
            messages.append(f"Scene {scene.order + 1}: {scene.name}")

        status = f"Queued {queued_count} scene(s) for generation.\n\n"
        status += "**Note:** To generate, copy each scene's prompt to the main Video Generator tab.\n\n"
        status += "Scenes queued:\n" + "\n".join(f"- {m}" for m in messages)

        return self._render_timeline(), status

    # =========================================================================
    # Concatenation
    # =========================================================================

    def _concatenate_completed(self, fps_str: str):
        """Concatenate all completed scenes"""
        if self.manager.current_project is None:
            return "No project loaded."

        completed = self.manager.current_project.get_completed_scenes()

        if len(completed) < 2:
            return "Need at least 2 completed scenes to concatenate."

        fps = int(fps_str)

        # Build scene data for concatenator
        scene_data = []
        for scene in completed:
            scene_data.append({
                "output_path": scene.output_path,
                "transition_type": scene.transition_to_next.type,
                "transition_duration_frames": scene.transition_to_next.duration_frames
            })

        try:
            output_name = f"storyboard_{self.manager.current_project.name}_{int(time.time())}"
            output_path = self.concatenator.concatenate_with_transitions(
                scene_data,
                output_name,
                fps=fps
            )
            return f"Video created: {output_path}"
        except Exception as e:
            return f"Error concatenating: {str(e)}"

    # =========================================================================
    # Import/Export
    # =========================================================================

    def _import_csv(self, file):
        """Import scenes from CSV"""
        if file is None:
            return self._render_timeline(), self._get_duration_display(), "No file selected."

        try:
            count = self.manager.import_from_csv(file.name)
            return (
                self._render_timeline(),
                self._get_duration_display(),
                f"Imported {count} scenes from CSV."
            )
        except Exception as e:
            return (
                self._render_timeline(),
                self._get_duration_display(),
                f"Error importing CSV: {str(e)}"
            )

    def _export_csv(self):
        """Export scenes to CSV"""
        if self.manager.current_project is None:
            return "No project to export."

        try:
            name = self.manager.current_project.name.replace(" ", "_")
            path = f"storyboards/{name}_export.csv"
            self.manager.export_to_csv(path)
            return f"Exported to: {path}"
        except Exception as e:
            return f"Error exporting: {str(e)}"

    # =========================================================================
    # UI Rendering
    # =========================================================================

    def _render_timeline(self) -> str:
        """Render the timeline as HTML"""
        if self.manager.current_project is None:
            return self._get_empty_timeline_html()

        project = self.manager.current_project
        scenes = sorted(project.scenes, key=lambda s: s.order)
        fps = project.global_settings.fps

        if not scenes:
            return self._get_empty_timeline_html()

        html = '<div class="storyboard-timeline" style="display: flex; flex-wrap: wrap; gap: 8px; padding: 10px;">'

        for i, scene in enumerate(scenes):
            status_icon = {
                "pending": "&#9998;",      # Pencil
                "queued": "&#9203;",       # Hourglass
                "generating": "&#9881;",   # Gear
                "complete": "&#10004;",    # Checkmark
                "failed": "&#10060;"       # X
            }.get(scene.status, "?")

            duration_s = scene.duration_frames / fps
            selected = "border: 2px solid #3b82f6;" if scene.id == self.selected_scene_id else "border: 2px solid #374151;"

            trans = scene.transition_to_next
            trans_indicator = ""
            if i < len(scenes) - 1 and trans.type != "cut":
                trans_indicator = f'<div style="position: absolute; right: -14px; top: 50%; transform: translateY(-50%); font-size: 10px;">&#8594;</div>'

            html += f'''
            <div class="scene-card" data-scene-id="{scene.id}"
                 onclick="storyboardSelectScene('{scene.id}')"
                 style="position: relative; width: 140px; padding: 8px; background: #1f2937;
                        border-radius: 8px; cursor: pointer; {selected}">
                <div style="font-weight: bold; font-size: 12px; margin-bottom: 4px;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {scene.name}
                </div>
                <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;
                            height: 32px; overflow: hidden;">
                    {scene.prompt[:50] + "..." if len(scene.prompt) > 50 else scene.prompt or "(no prompt)"}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px;">
                    <span>{duration_s:.1f}s</span>
                    <span>{status_icon}</span>
                </div>
                {trans_indicator}
            </div>
            '''

        html += '</div>'
        return html

    def _get_empty_timeline_html(self) -> str:
        """Get HTML for empty timeline"""
        return '''
        <div style="padding: 40px; text-align: center; color: #6b7280; background: #1f2937;
                    border-radius: 8px; border: 2px dashed #374151;">
            <p style="margin: 0;">No scenes yet. Click "Add Scene" to begin.</p>
        </div>
        '''

    def _get_duration_display(self) -> str:
        """Get duration summary text"""
        if self.manager.current_project is None:
            return "Total: 0 scenes, 0.0 seconds"

        project = self.manager.current_project
        total_frames = project.get_total_duration_frames()
        total_seconds = project.get_total_duration_seconds()

        return f"Total: {len(project.scenes)} scene(s), {total_seconds:.1f} seconds ({total_frames} frames)"

    def _get_model_choices(self) -> List[str]:
        """Get available model choices"""
        # Return common model types - in real integration would query available models
        return [
            "wan2.1-t2v-14B",
            "wan2.1-t2v-1.3B",
            "wan2.1-i2v-14B",
            "wan2.2-t2v",
            "wan2.2-i2v",
            "hunyuan-t2v",
            "ltxv-13B"
        ]

    def _get_timeline_js(self) -> str:
        """Get JavaScript for timeline interactivity"""
        return '''
        window.storyboardSelectScene = function(sceneId) {
            // Find the hidden input and update it
            const input = document.querySelector('#storyboard_scene_select textarea');
            if (input) {
                input.value = sceneId;
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }

            // Update visual selection
            document.querySelectorAll('.scene-card').forEach(card => {
                if (card.dataset.sceneId === sceneId) {
                    card.style.border = '2px solid #3b82f6';
                } else {
                    card.style.border = '2px solid #374151';
                }
            });
        };
        '''
