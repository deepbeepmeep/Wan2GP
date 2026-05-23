import gradio as gr

# ROCm Windows PyTorch lacks CPU LAPACK, breaking nn.init.orthogonal_ during
# matanyone model construction (the QR decomposition needs torch.geqrf).
# Our torch's GPU linalg works fine; detour CPU calls through GPU only when
# the CPU path raises a LAPACK error. Zero overhead on systems where CPU LAPACK works.
# Reversible: pip install --force-reinstall torch to restore stock behavior.
import torch
import torch.nn.init as _wan2gp_init
_wan2gp_original_orthogonal = _wan2gp_init.orthogonal_

def _wan2gp_gpu_safe_orthogonal_(tensor, gain=1, generator=None):
    try:
        return _wan2gp_original_orthogonal(tensor, gain=gain, generator=generator)
    except RuntimeError as e:
        msg = str(e)
        if (tensor.device.type == 'cpu'
                and torch.cuda.is_available()
                and ('LAPACK' in msg or 'geqrf' in msg)):
            gpu_t = tensor.detach().to('cuda')
            _wan2gp_original_orthogonal(gpu_t, gain=gain, generator=generator)
            tensor.copy_(gpu_t.to('cpu'))
            del gpu_t
            return tensor
        raise

_wan2gp_init.orthogonal_ = _wan2gp_gpu_safe_orthogonal_

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
        self.matanyone_app.ensure_selected_assets(self.server_config)
        self.vmc_event_handler(state, True)
        self._is_active = True

    def on_tab_deselect(self, state: dict) -> None:
        if not self._is_active:
            return
        # print("[VideoMaskCreatorPlugin] Tab deselected. Unloading models...")
        self.vmc_event_handler(state, False)
        self._is_active = False
