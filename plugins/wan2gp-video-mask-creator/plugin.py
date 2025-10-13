import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import json
import traceback

class VideoMaskCreatorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Video Mask Creator"
        self.version = "1.0.0"
        self.description = "Create masks for your videos with Matanyone"

    def setup_ui(self):
        self.request_global("vmc_event_handler")
        self.request_global("gen_in_progress")
        self.request_global("download_shared_done")
        self.request_global("download_models")
        self.request_global("server_config")
        self.request_global("get_current_model_settings")
        self.request_global("matanyone_app")
        self.request_component("main_tabs")
        self.request_component("tab_state")
        self.request_component("state")
        self.request_component("refresh_form_trigger")
        self.request_component("save_form_trigger")

    def create_mask_creator_ui(self):
        pass

    def post_ui_setup(self, components: dict):
        with self.main_tabs:
            with gr.Tab("Video Mask Creator", id="video_mask_creator"):
                 self.matanyone_app.display(
                    self.main_tabs, 
                    self.tab_state, 
                    self.state, 
                    self.refresh_form_trigger, 
                    self.server_config, 
                    self.get_current_model_settings
                )

        return {}