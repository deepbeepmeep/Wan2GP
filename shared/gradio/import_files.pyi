"""Gallery uploads retain their original names alongside Gradio's cache paths."""
import gradio as gr

from gradio.events import Dependency

class ImportFiles(gr.File):
    def get_block_name(self):
        return 'file'

    def _process_single_file(self, file):
        path = super()._process_single_file(file)
        path.orig_name = file.orig_name
        return path
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component