import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources, any_GPU_process_running
import time

PlugIn_Name = "Sample Plugin"
PlugIn_Id ="SamplePlugin"

def acquire_GPU(state):
    GPU_process_running = any_GPU_process_running(state, PlugIn_Id)
    if GPU_process_running:
        gr.Error("Another PlugIn is using the GPU")
    acquire_GPU_ressources(state, PlugIn_Id, PlugIn_Name, gr= gr)      

def release_GPU(state):
    release_GPU_ressources(state, PlugIn_Id)

class ConfigTabPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.0.0"
        self.description = PlugIn_Name

    def setup_ui(self):
        self.request_global("get_current_model_settings")
        self.request_component("refresh_form_trigger")      
        self.request_component("state")
        self.request_component("resolution")
        self.request_component("main_tabs")

        self.add_tab(
            tab_id=PlugIn_Id,
            label=PlugIn_Name,
            component_constructor=self.create_config_ui,
        )


    def on_tab_select(self, state: dict) -> None:
        settings = self.get_current_model_settings(state)
        prompt = settings["prompt"]
        return prompt


    def on_tab_deselect(self, state: dict) -> None:
        pass

    def create_config_ui(self):
        def update_prompt(state, text):
            settings = self.get_current_model_settings(state)
            settings["prompt"] = text
            return time.time()

        def big_process(state):
            acquire_GPU(state)
            gr.Info("Doing something important")
            time.sleep(30)
            release_GPU(state)
            return "42"
        
        with gr.Column():
            state = self.state
            settings = self.get_current_model_settings(state.value)            
            prompt = settings["prompt"]
            gr.HTML("<B><B>Sample Plugin that illustrates</B>:<BR>-How to get Settings from Main Form and then Modify them<BR>-How to suspend the Video Gen (and release VRAM) to execute your own GPU intensive process.<BR>-How to switch back automatically to the Main Tab")
            sample_text = gr.Text(label="Prompt Copy", value=prompt, lines=5)
            update_btn = gr.Button("Update Prompt On Main Page")
            gr.Markdown()            
            process_btn = gr.Button("Use GPU To Do Something Important")
            process_output = gr.Text(label="Process Output", value='')
            goto_btn = gr.Button("Goto Video Tab")

        self.on_tab_outputs = [sample_text]

        update_btn.click(
            fn=update_prompt,
            inputs=[state, sample_text],
            outputs=[ self.refresh_form_trigger ]
        )


        process_btn.click(
            fn=big_process,
            inputs=[state],
            outputs=[ process_output ]
        )

        goto_btn.click(
            fn=self.goto_video_tab,
            inputs=[state],
            outputs=[ self.main_tabs ]
        )

   