"""Compatibility entry point; all progress rendering lives in WangpProgress."""

from shared.gradio.progress import WangpProgress


def render_download_progress(data, aborting=False, description="Downloading"):
    if data is None:
        return ""
    progress = WangpProgress()
    progress.status(description)
    progress.set_download(data)
    return progress.render(aborting=aborting)
