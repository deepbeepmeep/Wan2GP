from __future__ import annotations

import time

from shared.utils.plugins import WAN2GPPlugin

from .plugin_ui import create_config_ui as _create_config_ui

PlugIn_Name = "Process Full Video"
PlugIn_Id = "ProcessFullVideo"


class ConfigTabPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("get_model_def")
        self.request_global("get_lora_dir")
        self.request_global("get_base_model_type")
        self.request_global("server_config")
        self.request_component("state")
        self.request_component("lset_name")
        self.request_component("refresh_form_trigger")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    def on_tab_select(self, state: dict) -> str:
        return str(time.time_ns())

    def create_config_ui(self, api_session):
        return _create_config_ui(self, api_session)
