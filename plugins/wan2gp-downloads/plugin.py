import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import shutil
import glob
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download

class DownloadsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()

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
        self.download_status = gr.Markdown()
        self.download_loras_btn.click(
            fn=self.download_loras_action, 
            inputs=[], 
            outputs=[self.download_status]
        ).then(
            fn=self.refresh_lora_list, 
            inputs=[self.state, self.lset_name, self.loras_choices], 
            outputs=[self.lset_name, self.loras_choices]
        )

    def download_loras_action(self):
        yield "<B><FONT SIZE=3>Please wait while the Loras are being downloaded</B></FONT>"
        lora_dir = self.get_lora_dir("i2v")
        log_path = os.path.join(lora_dir, "log.txt")
        if not os.path.isfile(log_path):
            tmp_path = os.path.join(lora_dir, "tmp_lora_download")
            snapshot_download(repo_id="DeepBeepMeep/Wan2.1", allow_patterns="loras_i2v/*", local_dir=tmp_path)
            for f in glob.glob(os.path.join(tmp_path, "loras_i2v", "*")):
                if os.path.isfile(f):
                    target_file = os.path.join(lora_dir, os.path.basename(f))
                    if os.path.exists(target_file):
                        os.remove(target_file)
                    shutil.move(f, lora_dir)
            try:
                shutil.rmtree(tmp_path)
            except Exception as e:
                print(f"Failed to remove tmp_path: {e}")

            dt = datetime.today().strftime('%Y-%m-%d')
            tm = datetime.now().strftime('%H:%M:%S')
            with open(log_path, "w", encoding="utf-8") as writer:
                writer.write(f"Loras downloaded on {dt} at {tm}")
            yield "<B><FONT SIZE=3 COLOR=green>Loras's Festival is successfully STARTED !</B></FONT>"
        else:
            yield "<B><FONT SIZE=3>Loras's Festival is already ON ! (Loras already downloaded)</B></FONT>"
        return
