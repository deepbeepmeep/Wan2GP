import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import shutil
from pathlib import Path

class DownloadsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Downloads Tab"
        self.version = "1.0.0"
        self.description = "Download our in-house loras!"

    def setup_ui(self):
        self.request_global("get_lora_dir")
        self.request_global("refresh_lora_list")

        self.request_component("state")
        self.request_component("lset_name")
        self.request_component("loras_choices")

        self.add_tab(
            tab_id="downloads",
            label="Downloads",
            component_constructor=self.create_downloads_ui,
        )

    def create_downloads_ui(self):
        with gr.Row():
            with gr.Row(scale=2):
                gr.Markdown("<I>WanGP's Lora Festival ! Press the following button to download i2v <B>Remade_AI</B> Loras collection (and bonuses Loras).")
            with gr.Row(scale=1):
                self.download_loras_btn = gr.Button("---> Let the Lora's Festival Start !", scale=1)
            with gr.Row(scale=1):
                gr.Markdown("")
        with gr.Row() as self.download_status_row: 
            self.download_status = gr.Markdown()
        self.download_loras_btn.click(
            fn=self.download_loras_action, 
            inputs=[], 
            outputs=[self.download_status_row, self.download_status]
        ).then(
            fn=self.refresh_lora_list, 
            inputs=[self.state, self.lset_name, self.loras_choices], 
            outputs=[self.lset_name, self.loras_choices]
        )

    def download_loras_action(self):
        from huggingface_hub import snapshot_download    
        yield gr.Row(visible=True), "<B><FONT SIZE=3>Please wait while the Loras are being downloaded</B></FONT>"
        lora_dir = self.get_lora_dir("i2v")
        log_path = os.path.join(lora_dir, "log.txt")
        if not os.path.isfile(log_path):
            tmp_path = os.path.join(lora_dir, "tmp_lora_dowload")
            import glob
            snapshot_download(repo_id="DeepBeepMeep/Wan2.1", allow_patterns="loras_i2v/*", local_dir=tmp_path)
            for f in glob.glob(os.path.join(tmp_path, "loras_i2v", "*.*")):
                target_file = os.path.join(lora_dir, Path(f).parts[-1])
                if os.path.isfile(target_file):
                    os.remove(f)
                else:
                    shutil.move(f, lora_dir)
        try:
            os.remove(tmp_path)
        except:
            pass
        yield gr.Row(visible=True), "<B><FONT SIZE=3>Loras have been completely downloaded</B></FONT>"

        from datetime import datetime
        import time
        dt = datetime.today().strftime('%Y-%m-%d')
        with open(log_path, "w", encoding="utf-8") as writer:
            writer.write(f"Loras downloaded on the {dt} at {time.time()} on the {time.time()}")
        return
