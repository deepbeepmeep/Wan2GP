import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from preprocessing.matanyone import app as matanyone_app

class VideoMaskCreatorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Video Mask Creator"
        self.version = "1.2.0"
        self.description = "Create masks for your videos with Matanyone. Now fully integrated with the plugin system."
        self._is_active = False
        
        self.matanyone_app = matanyone_app
        self.vmc_event_handler = self.matanyone_app.get_vmc_event_handler()

    def setup_ui(self):
        self.request_global("download_shared_done")
        self.request_global("download_models")
        self.request_global("server_config")
        self.request_global("get_current_model_settings")
        
        self.request_component("main_tabs")
        self.request_component("state")
        self.request_component("refresh_form_trigger")
        
        self.add_tab(
            tab_id="video_mask_creator",
            label="Video Mask Creator",
            component_constructor=self.create_mask_creator_ui,
        )

    def create_mask_creator_ui(self):
        matanyone_tab_state = gr.State({ "tab_no": 0 })
        self.matanyone_app.display(
            tabs=self.main_tabs,
            tab_state=matanyone_tab_state,
            state=self.state,
            refresh_form_trigger=self.refresh_form_trigger,
            server_config=self.server_config,
            get_current_model_settings_fn=self.get_current_model_settings
        )
        self.matanyone_app.PlugIn = self

    def on_tab_select(self, state: dict) -> None:
        # print("[VideoMaskCreatorPlugin] Tab selected. Loading models...")
        if not self.download_shared_done:
            self.download_models()
        self.vmc_event_handler(state, True)
        self._is_active = True

    def on_tab_deselect(self, state: dict) -> None:
        if not self._is_active:
            return
        # print("[VideoMaskCreatorPlugin] Tab deselected. Unloading models...")
        self.vmc_event_handler(state, False)
        self._is_active = False
