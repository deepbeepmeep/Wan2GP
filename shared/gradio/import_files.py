# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Gallery uploads retain their original names alongside Gradio's cache paths."""
import gradio as gr


class ImportFiles(gr.File):
    def get_block_name(self):
        return 'file'

    def _process_single_file(self, file):
        path = super()._process_single_file(file)
        path.orig_name = file.orig_name
        return path
