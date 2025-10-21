import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class GuidesPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Guides Tab"
        self.version = "1.0.0"
        self.description = "Guides for using WAN2GP"

    def setup_ui(self):
        self.add_tab(
            tab_id="info",
            label="Guides",
            component_constructor=self.create_guides_ui,
            position=2
        )

    def create_guides_ui(self):
        with open("docs/VACE.md", "r", encoding="utf-8") as reader:
            vace= reader.read()

        with open("docs/MODELS.md", "r", encoding="utf-8") as reader:
            models = reader.read()

        with open("docs/LORAS.md", "r", encoding="utf-8") as reader:
            loras = reader.read()

        with open("docs/FINETUNES.md", "r", encoding="utf-8") as reader:
            finetunes = reader.read()

        with open("docs/PLUGINS.md", "r", encoding="utf-8") as reader:
            plugins = reader.read()

        with gr.Tabs() :
            with gr.Tab("Models", id="models"):
                gr.Markdown(models)
            with gr.Tab("Loras", id="loras"):
                gr.Markdown(loras)
            with gr.Tab("Vace", id="vace"):
                gr.Markdown(vace)
            with gr.Tab("Finetunes", id="finetunes"):
                gr.Markdown(finetunes)
            with gr.Tab("Plugins", id="plugins"):
                gr.Markdown(plugins)