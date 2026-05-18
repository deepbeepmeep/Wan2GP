from datetime import datetime
import time
import json
import gradio as gr

from shared.utils.plugins import WAN2GPPlugin
from shared.utils.process_locks import acquire_GPU_ressources, any_GPU_process_running, release_GPU_ressources

PlugIn_Name = "Sample Plugin"
PlugIn_Id = "SamplePlugin"


def acquire_GPU(state):
    GPU_process_running = any_GPU_process_running(state, PlugIn_Id)
    if GPU_process_running:
        gr.Error("Another PlugIn is using the GPU")
    acquire_GPU_ressources(state, PlugIn_Id, PlugIn_Name, gr=gr)


def release_GPU(state):
    release_GPU_ressources(state, PlugIn_Id)


class ConfigTabPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("get_current_model_settings")
        self.request_global("refresh_model_defs")        
        self.request_component("refresh_form_trigger")
        self.request_component("state")
        self.request_component("resolution")
        self.request_component("main_tabs")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    @staticmethod
    def _demo_settings(_state) -> dict[str, object]:
        return {
            "model_type": "ltx2_22B_distilled",
            "prompt": "A reframed cinematic 10 second video shot: a woman walks slowly through a rainy neon street at night, camera gently orbiting, reflections shimmering on the wet pavement, natural motion, realistic lighting.",
            "resolution": "1280x720",
            "num_inference_steps": 8,
            "video_length": 241,
            "sliding_window_size": 481,
            "sliding_window_overlap": 17,
        }

    def on_tab_select(self, state: dict) -> None:
        settings = self.get_current_model_settings(state)
        prompt = settings["prompt"]
        return prompt

    def on_tab_deselect(self, state: dict) -> None:
        pass

    def on_model_change(self, state: dict, model_type) -> None:
        # print(f"new model selected is {model_type}")
        pass

    def create_config_ui(self, api_session):
        active_job = {"job": None}

        def update_prompt(state, text):
            settings = self.get_current_model_settings(state)
            settings["prompt"] = text
            return time.time()

        def big_process(state):
            acquire_GPU(state)
            gr.Info("Doing something important that Requires Full VRAM & GPU available")
            time.sleep(30)
            release_GPU(state)
            return "42"

        def generate_video(progress=gr.Progress(track_tqdm=False)):
            class DemoCallbacks:
                ratio = 0.0

                def on_status(self, status):
                    status = str(status or "").strip()
                    if status:
                        progress(self.ratio, desc=status)

                def on_progress(self, update):
                    self.ratio = max(0.0, min(1.0, float(getattr(update, "progress", 0)) / 100.0))
                    progress(self.ratio, desc=str(getattr(update, "status", "") or "Generating..."))

            job = api_session.submit_task(self._demo_settings(self.state.value), callbacks=DemoCallbacks())
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

        def cancel_demo():
            job = active_job.get("job")
            if job is not None and not job.done:
                job.cancel()

        def write_finetune():
            the_time= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            finetune_def =  {
                "model": {
                        "name": f"Finetune of {the_time}",
                        "architecture": "ltx2_22B",
                        "description": "enjoy this finetune of the moment",
                "URLs": "ltx2_22B",
                "preload_URLs":"ltx2_22B",        
                "ltx2_pipeline": "distilled"
                },
                "prompt": f"A video that is all about {the_time}",
            }
            with open(f"finetunes/to_be_deleted{the_time}.json", "w", encoding="utf-8") as writer:
                writer.write(json.dumps(finetune_def, indent=4))
            self.refresh_model_defs()
            gr.Info(f"finetune {the_time} had been created")


        with gr.Column():
            state = self.state
            settings = self.get_current_model_settings(state.value)
            prompt = settings["prompt"]
            gr.HTML("<B><B>Sample Plugin that illustrates</B>:<BR>-How to get Settings from Main Form and then Modify them<BR>-How to suspend the Video Gen (and release VRAM) to execute your own GPU intensive process.<BR>-How to switch back automatically to the Main Tab<BR>-How to trigger a Video Gen from a plugin an track its progress<BR>-Add a new finetune on the fly")
            sample_text = gr.Text(label="Prompt Copy", value=prompt, lines=5)
            update_btn = gr.Button("Update Prompt On Main Page")
            gr.Markdown()
            process_btn = gr.Button("Use GPU To Do Something Important")
            process_output = gr.Text(label="Process Output", value="")
            goto_btn = gr.Button("Goto Video Tab")
            gr.Markdown("---")
            start_btn = gr.Button("Generate a LTX 2.3 Video")
            output_video = gr.Video(label="Output")
            abort_btn = gr.Button("Abort Generation")

            write_finetune_btn = gr.Button("Generate a Random LTX 2 Finetune")

        self.on_tab_outputs = [sample_text]

        update_btn.click(fn=update_prompt, inputs=[state, sample_text], outputs=[self.refresh_form_trigger])
        process_btn.click(fn=big_process, inputs=[state], outputs=[process_output])
        goto_btn.click(fn=self.goto_video_tab, inputs=[state], outputs=[self.main_tabs])
        start_btn.click(fn=generate_video, outputs=[output_video], queue=False)
        abort_btn.click(fn=cancel_demo, queue=False)
        write_finetune_btn.click(fn=write_finetune)

