from __future__ import annotations

import json

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import character_pack as packs


PlugIn_Name = "Character & Environment Consistency Studio"
PlugIn_Id = "CharacterConsistencyStudio"


class CharacterConsistencyPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.0.0"
        self.description = "Create reusable character and environment packs for reference-consistent WanGP video shots."
        self.uninstallable = False

    def setup_ui(self):
        self.add_tab(tab_id=PlugIn_Id, label="Characters", component_constructor=self.create_config_ui, position=2)

    def create_config_ui(self, api_session):
        active_job = {"job": None}

        def _paths(path_text, uploaded_files):
            return packs.coerce_reference_paths(path_text, uploaded_files)

        def _build_payload(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video):
            refs = _paths(path_text, uploaded_files)
            shot_prompts = packs.split_shots(shot_prompts_text)
            issues = packs.validate_pack(mode, refs, identity_prompt, shot_prompts)
            manifest = packs.build_manifest(
                character_name=character_name,
                mode=mode,
                identity_prompt=identity_prompt,
                environment_prompt=environment_prompt,
                shot_prompts=shot_prompts,
                refs=refs,
                negative_prompt=negative_prompt,
                style_prompt=style_prompt,
                resolution=resolution,
                video_length=int(video_length),
                seed=int(seed),
                control_video=control_video,
            ) if not any(issue.startswith("Add ") for issue in issues) else []
            pack = {
                "character_name": character_name,
                "mode": mode,
                "mode_label": packs.MODE_LABELS.get(mode, mode),
                "identity_prompt": identity_prompt,
                "environment_prompt": environment_prompt,
                "negative_prompt": negative_prompt,
                "style_prompt": style_prompt,
                "reference_images": refs,
                "control_video": str(control_video or "").strip(),
                "shot_prompts": shot_prompts,
                "settings": manifest,
            }
            return pack, issues

        def preview_settings(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video):
            pack, issues = _build_payload(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video)
            issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "Ready."
            return issue_text, json.dumps(pack.get("settings", []), indent=2, ensure_ascii=True)

        def export_pack(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video):
            pack, issues = _build_payload(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video)
            blocking = [issue for issue in issues if issue.startswith("Add ")]
            if blocking:
                raise gr.Error("\n".join(blocking))
            payload = {key: value for key, value in pack.items() if key != "settings"}
            return str(packs.save_pack_json(character_name, payload))

        def export_settings(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video):
            pack, issues = _build_payload(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video)
            blocking = [issue for issue in issues if issue.startswith("Add ")]
            if blocking:
                raise gr.Error("\n".join(blocking))
            return str(packs.save_manifest_json(character_name, pack["settings"]))

        def generate_first_shot(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video, progress=gr.Progress(track_tqdm=False)):
            pack, issues = _build_payload(character_name, mode, identity_prompt, environment_prompt, negative_prompt, style_prompt, shot_prompts_text, path_text, uploaded_files, resolution, video_length, seed, control_video)
            blocking = [issue for issue in issues if issue.startswith("Add ")]
            if blocking:
                raise gr.Error("\n".join(blocking))
            settings = pack["settings"][0]

            class CharacterCallbacks:
                ratio = 0.0

                def on_status(self, status):
                    status = str(status or "").strip()
                    if status:
                        progress(self.ratio, desc=status)

                def on_progress(self, update):
                    self.ratio = max(0.0, min(1.0, float(getattr(update, "progress", 0)) / 100.0))
                    progress(self.ratio, desc=str(getattr(update, "status", "") or "Generating..."))

            job = api_session.submit_task(settings, callbacks=CharacterCallbacks())
            active_job["job"] = job
            try:
                result = job.result()
            finally:
                if active_job.get("job") is job:
                    active_job["job"] = None
            if result.success and result.generated_files:
                return result.generated_files[0]
            if result.cancelled:
                return gr.update()
            errors = list(result.errors or [])
            raise gr.Error(str(errors[0] if errors else "WanGP completed without returning an output file."))

        def cancel_generation():
            job = active_job.get("job")
            if job is not None and not job.done:
                job.cancel()

        mode_choices = [(label, key) for key, label in packs.MODE_LABELS.items()]
        default_identity = (
            "A named character with a consistent face, hairstyle, body proportions, age, wardrobe details, "
            "signature colors, and recognizable silhouette. Describe the exact face, hair, clothes, and any persistent props here."
        )
        default_environment = (
            "A recurring world with stable location details, architecture, props, era, weather, lighting direction, "
            "and color palette. Describe what must stay consistent between shots."
        )
        default_shots = (
            "Medium close-up introduction shot, the character looks toward camera, subtle natural expression, cinematic soft light.\n\n"
            "Wide shot in a new location, the same character walks through the scene, same outfit and recognizable silhouette."
        )

        with gr.Column():
            gr.Markdown("## Character & Environment Consistency Studio")
            gr.Markdown(packs.research_summary_markdown())
            with gr.Row():
                with gr.Column(scale=5):
                    character_name = gr.Textbox(label="Character name", value="My Character")
                    mode = gr.Dropdown(label="WanGP consistency mode", choices=mode_choices, value="bernini_ingredients")
                    identity_prompt = gr.Textbox(label="Locked identity prompt", value=default_identity, lines=5)
                    environment_prompt = gr.Textbox(label="Locked environment/world prompt", value=default_environment, lines=3)
                    negative_prompt = gr.Textbox(
                        label="Negative drift prompt",
                        value="different person, face drift, changed hairstyle, changed age, inconsistent outfit, changed location, inconsistent environment, distorted hands, extra fingers, broken anatomy",
                        lines=2,
                    )
                    style_prompt = gr.Textbox(label="Optional locked visual style prompt", value="", lines=2)
                    shot_prompts = gr.Textbox(label="Shot prompts (blank line separates shots)", value=default_shots, lines=7)
                with gr.Column(scale=4):
                    refs_upload = gr.File(label="Reference image uploads", file_count="multiple", file_types=["image"])
                    refs_paths = gr.Textbox(label="Reference image paths (one per line)", lines=5)
                    control_video = gr.Textbox(label="Optional control video path for SCAIL/VACE-style workflows", lines=1)
                    with gr.Row():
                        resolution = gr.Dropdown(label="Resolution", choices=["832x480", "1280x720", "720x1280", "1024x576", "576x1024"], value="1280x720")
                        video_length = gr.Number(label="Frames", value=121, precision=0)
                        seed = gr.Number(label="Seed", value=-1, precision=0)
                    issue_box = gr.Textbox(label="Validation", value="Ready.", lines=4)

            with gr.Row():
                preview_btn = gr.Button("Preview Settings")
                export_pack_btn = gr.Button("Save Character Pack")
                export_settings_btn = gr.Button("Export WanGP Settings")
                generate_btn = gr.Button("Generate First Shot", variant="primary")
                cancel_btn = gr.Button("Cancel")

            settings_json = gr.Textbox(label="WanGP settings preview", lines=18, max_lines=30)
            with gr.Row():
                pack_file = gr.File(label="Saved character pack")
                settings_file = gr.File(label="Exported WanGP settings")
            output_video = gr.Video(label="Generated first shot")

        inputs = [
            character_name,
            mode,
            identity_prompt,
            environment_prompt,
            negative_prompt,
            style_prompt,
            shot_prompts,
            refs_paths,
            refs_upload,
            resolution,
            video_length,
            seed,
            control_video,
        ]
        preview_btn.click(preview_settings, inputs=inputs, outputs=[issue_box, settings_json])
        export_pack_btn.click(export_pack, inputs=inputs, outputs=[pack_file])
        export_settings_btn.click(export_settings, inputs=inputs, outputs=[settings_file])
        generate_btn.click(generate_first_shot, inputs=inputs, outputs=[output_video])
        cancel_btn.click(cancel_generation, queue=False)
