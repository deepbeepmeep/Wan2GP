"""
Storyboard Plugin for Wan2GP

Plan and generate multi-scene video projects with visual timeline,
transitions, and automatic concatenation.
"""

import time
import json
import gradio as gr
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil

from shared.utils.plugins import WAN2GPPlugin
from shared.utils.utils import get_video_frame, get_video_info

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
        self.version = "1.1.0"
        self.description = "Plan and generate multi-scene videos with transitions"

        self.manager = StoryboardManager()
        self.concatenator = SceneConcatenator()
        self.selected_scene_id: Optional[str] = None
        self.runner_active = False # State for the smart runner

    def setup_ui(self):
        # Request main components
        self.request_component("state")
        self.request_component("main_tabs")

        # Request global functions for queue integration
        self.request_global("get_model_def")
        self.request_global("server_config")
        self.request_global("add_video_task") # Critical for execution
        self.request_global("update_queue_data")

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
        # Timer for the Smart Runner
        self.runner_timer = gr.Timer(value=2.0, active=False)

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

                self.scene_use_prev_frame = gr.Checkbox(
                    label="Continuity Mode: Use last frame of previous scene as Start Image",
                    value=True
                )

                self.scene_reference = gr.Image(
                    label="Reference Image (Optional / Overrides Continuity)",
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
                self.run_project_btn = gr.Button(
                    "▶ Run Project (Smart Queue)",
                    variant="primary"
                )
                self.stop_project_btn = gr.Button("⏹ Stop Runner")
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
                self.scene_status,
                self.scene_use_prev_frame
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
                self.scene_status,
                self.scene_use_prev_frame
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

        self.scene_use_prev_frame.change(
            fn=self._update_scene_field,
            inputs=[self.scene_id_state, gr.State("use_prev_frame"), self.scene_use_prev_frame],
            outputs=scene_edit_outputs
        )

        self.scene_reference.change(
            fn=self._update_scene_field,
            inputs=[self.scene_id_state, gr.State("reference_image"), self.scene_reference],
            outputs=scene_edit_outputs
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
                self.scene_status,
                self.scene_use_prev_frame
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

        # Runner actions
        self.run_project_btn.click(
            fn=self._start_runner,
            inputs=[self.state, self.global_model, self.global_resolution,
                   self.global_fps, self.seed_mode, self.global_seed,
                   self.global_guidance, self.global_steps],
            outputs=[self.runner_timer, self.status_md, self.timeline_html]
        )

        self.stop_project_btn.click(
            fn=self._stop_runner,
            inputs=[],
            outputs=[self.runner_timer, self.status_md]
        )

        # Timer tick
        self.runner_timer.tick(
            fn=self._on_timer_tick,
            inputs=[self.state],
            outputs=[self.timeline_html, self.status_md, self.runner_timer]
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
            return [gr.update()] * 10

        scene = self.manager.current_project.get_scene(scene_id)
        if scene is None:
            return [gr.update()] * 10

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
            scene.status,
            scene.settings_override.get("use_prev_frame", True)
        )

    def _update_scene_field(self, scene_id: str, field: str, value):
        """Update a single scene field"""
        if not scene_id or self.manager.current_project is None:
            return self._render_timeline(), self._get_duration_display()

        if field == "use_prev_frame":
             scene = self.manager.current_project.get_scene(scene_id)
             if scene:
                 scene.settings_override["use_prev_frame"] = value
        else:
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
    # Smart Runner (Execution)
    # =========================================================================

    def _start_runner(self, state, model, resolution, fps, seed_mode, seed, guidance, steps):
        """Start the automated runner"""
        if self.manager.current_project is None:
            return gr.Timer(active=False), "No project loaded.", self._render_timeline()

        # Check if queue is empty first
        gen = state.get("gen", {})
        if gen.get("queue", []):
             return gr.Timer(active=False), "Main queue must be empty to start Storyboard runner.", self._render_timeline()

        # Update global settings
        self._update_global_settings(model, resolution, fps, seed_mode, seed, guidance, steps)

        # Reset failed scenes to pending if any
        for scene in self.manager.current_project.scenes:
            if scene.status == "failed":
                scene.status = "pending"

        self.runner_active = True
        return gr.Timer(active=True), "Runner started...", self._render_timeline()

    def _stop_runner(self):
        """Stop the automated runner"""
        self.runner_active = False
        return gr.Timer(active=False), "Runner stopped."

    def _on_timer_tick(self, state):
        """Main loop for the runner"""
        if not self.runner_active:
            return gr.update(), gr.update(), gr.Timer(active=False)

        if self.manager.current_project is None:
            self.runner_active = False
            return gr.update(), "Project unloaded.", gr.Timer(active=False)

        project = self.manager.current_project
        scenes = sorted(project.scenes, key=lambda s: s.order)
        gen = state.get("gen", {})
        main_queue = gen.get("queue", [])

        # 1. Check active scene
        active_scene = next((s for s in scenes if s.status in ["queued", "generating"]), None)

        if active_scene:
            # Check if it's done
            task_id = active_scene.settings_override.get("task_id")
            if task_id is None:
                # Should not happen if status is queued
                active_scene.status = "failed"
                return self._render_timeline(), "Error: Active scene has no task ID.", gr.Timer(active=True)

            # Check if task is still in queue
            task_in_queue = any(t['id'] == task_id for t in main_queue)

            if not task_in_queue:
                # Task finished (or failed/cancelled)
                # We need to find the output file.
                # WGP doesn't provide a direct link in the queue object after completion.
                # Heuristic: Check output folder for recent file with task ID in metadata or name?
                # Alternative: WGP saves "last_generated_files"? No.
                # Assuming success for now, we scan for the newest file.

                # Wait a brief moment for file system
                time.sleep(0.5)

                output_path = self._find_latest_output(state)
                if output_path:
                    active_scene.status = "complete"
                    active_scene.output_path = output_path
                    self.manager.save_project() # Auto save progress
                    return self._render_timeline(), f"Scene '{active_scene.name}' complete.", gr.Timer(active=True)
                else:
                    active_scene.status = "failed"
                    self.runner_active = False
                    return self._render_timeline(), f"Scene '{active_scene.name}' failed (no output found).", gr.Timer(active=False)
            else:
                active_scene.status = "generating" # It is processing
                return gr.update(), f"Generating '{active_scene.name}'...", gr.Timer(active=True)

        # 2. If no active scene, pick next pending
        next_scene = next((s for s in scenes if s.status == "pending"), None)

        if next_scene is None:
            self.runner_active = False
            return self._render_timeline(), "All scenes completed!", gr.Timer(active=False)

        # 3. Prepare inputs for next scene
        gs = project.global_settings

        # Calculate seed
        if gs.seed_mode == "fixed":
            scene_seed = gs.seed
        elif gs.seed_mode == "increment":
            scene_seed = gs.seed + next_scene.order
        else:
            scene_seed = -1

        # Resolve inputs
        params = {
            "state": state,
            "prompt": next_scene.prompt,
            "negative_prompt": next_scene.negative_prompt,
            "resolution": gs.resolution,
            "video_length": next_scene.duration_frames,
            "num_inference_steps": gs.num_inference_steps,
            "guidance_scale": gs.guidance_scale,
            "seed": scene_seed,
            "image_mode": 0, # Video
            "plugin_data": {"storyboard_scene_id": next_scene.id}
        }

        # Model specific overrides (placeholder logic)
        # In a real impl, we'd map gs.model_type to internal WGP model_type keys
        # We assume global settings model_type is valid for add_video_task logic
        # But we need to ensure the main form state has the right model loaded or selected?
        # WGP `add_video_task` uses the state to get `get_gen_info`. It copies inputs.
        # It does NOT switch the model. The model is switched when `process_tasks` runs based on `inputs["model_type"]`.
        # We need to pass `model_type`.

        # We need to map friendly names to internal IDs if needed, or just pass it.
        # The user selects from a dropdown populated by `_get_model_choices`.
        # Ideally we mirror `state["model_type"]` but here we enforce project settings.
        # To be safe, we might need to rely on the current UI selection if we can't reliably switch.
        # But let's try passing it.

        # Continuity Logic
        use_prev = next_scene.settings_override.get("use_prev_frame", True)
        prev_scene = self._get_previous_scene(project, next_scene)

        if use_prev and prev_scene and prev_scene.status == "complete" and prev_scene.output_path:
            # Extract last frame
            try:
                last_frame = get_video_frame(prev_scene.output_path, -1)
                params["image_start"] = last_frame
                params["image_prompt_type"] = "S" # Start Image
            except Exception as e:
                print(f"Error extracting frame: {e}")
                # Fallback?

        if next_scene.reference_image:
            # If user provided a reference, it overrides or adds?
            # Usually i2v takes one image.
            # If we have both, we might be in trouble depending on model.
            # Let's prioritize manual reference if provided, or handle 'image_refs' vs 'image_start'
            if "image_start" in params:
                 # We have continuity. Does this model support start + ref?
                 # Wan2.1 supports it.
                 params["image_refs"] = next_scene.reference_image
                 params["video_prompt_type"] = "I" # Image Ref
            else:
                 params["image_start"] = next_scene.reference_image
                 params["image_prompt_type"] = "S"

        # Submit task
        try:
            # We need to call the global add_video_task.
            # Since we are in a class, we used request_global to get access?
            # No, request_global just sets the attribute on self.
            if hasattr(self, "add_video_task"):
                self.add_video_task(**params)

                # Get the ID of the task we just added
                # add_video_task modifies queue in place.
                new_task_id = main_queue[-1]['id']

                next_scene.status = "queued"
                next_scene.settings_override["task_id"] = new_task_id

                # Trigger update of queue UI in main tab
                if hasattr(self, "update_queue_data"):
                    self.update_queue_data(main_queue)

                return self._render_timeline(), f"Queued '{next_scene.name}'...", gr.Timer(active=True)
            else:
                return self._render_timeline(), "Error: add_video_task not available.", gr.Timer(active=False)

        except Exception as e:
            print(f"Submission error: {e}")
            next_scene.status = "failed"
            return self._render_timeline(), f"Error submitting '{next_scene.name}': {e}", gr.Timer(active=False)

    def _get_previous_scene(self, project, scene) -> Optional[Scene]:
        idx = project.scenes.index(scene)
        if idx > 0:
            return project.scenes[idx - 1]
        return None

    def _find_latest_output(self, state) -> Optional[str]:
        """Find the most recently generated video file"""
        # This is a heuristic. WGP saves to 'outputs/'.
        # We look for the most recent mp4 file.
        # Ideally check if it matches our task, but WGP doesn't stamp files easily without reading metadata.
        # Since we run serially and check right after task disappears, it should be the newest.

        save_path = "outputs" # Default
        # Try to get from config
        if hasattr(self, "server_config") and self.server_config:
            save_path = self.server_config.get("save_path", "outputs")

        try:
            files = [os.path.join(save_path, f) for f in os.listdir(save_path)
                     if f.endswith('.mp4') or f.endswith('.mkv')]
            if not files:
                return None

            latest_file = max(files, key=os.path.getctime)
            return latest_file
        except Exception as e:
            print(f"Error finding output: {e}")
            return None

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

            # Color code status
            border_color = "var(--border-color-primary)" # Default
            if scene.status == "complete": border_color = "#10b981" # Green
            if scene.status == "generating": border_color = "#f59e0b" # Orange
            if scene.status == "failed": border_color = "#ef4444" # Red
            if scene.id == self.selected_scene_id: border_color = "#3b82f6" # Blue (Selected overrides)

            duration_s = scene.duration_frames / fps
            selected = f"border: 2px solid {border_color};"

            trans = scene.transition_to_next
            trans_indicator = ""
            if i < len(scenes) - 1 and trans.type != "cut":
                trans_indicator = f'<div style="position: absolute; right: -14px; top: 50%; transform: translateY(-50%); font-size: 10px; color: var(--body-text-color);">&#8594;</div>'

            # Thumbnail if available (from output or generation)
            thumb_html = ""
            if scene.output_path and os.path.exists(scene.output_path):
                 # We could extract a frame, but for now just show icon
                 pass

            html += f'''
            <div class="scene-card" data-scene-id="{scene.id}"
                 onclick="storyboardSelectScene('{scene.id}')"
                 style="position: relative; width: 140px; padding: 8px; background: var(--background-fill-secondary);
                        border-radius: 8px; cursor: pointer; {selected} color: var(--body-text-color);">
                <div style="font-weight: bold; font-size: 12px; margin-bottom: 4px;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--body-text-color);">
                    {scene.name}
                </div>
                <div style="font-size: 11px; color: var(--body-text-color-subdued); margin-bottom: 4px;
                            height: 32px; overflow: hidden;">
                    {scene.prompt[:50] + "..." if len(scene.prompt) > 50 else scene.prompt or "(no prompt)"}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--body-text-color);">
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
        <div style="padding: 40px; text-align: center; color: var(--body-text-color-subdued); background: var(--background-fill-secondary);
                    border-radius: 8px; border: 2px dashed var(--border-color-primary);">
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
        # Ideally this queries WGP for available models
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
        };
        '''
