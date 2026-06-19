from typing import Any, Callable

from postprocessing.audio_processors import (
    AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,
    REMOVE_BACKGROUND_METHOD,
    method_metadata,
)


class BackgroundRemovalProcessor:
    def __init__(self, server_config=None, files_locator=None):
        pass

    @classmethod
    def query_audio_processor_def(cls) -> dict[str, Any]:
        return {
            "name": "Remove Music / Background noise",
            "processor_types": (AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,),
            "methods": [("Remove Music / Background noise", REMOVE_BACKGROUND_METHOD)],
            "method_types": {REMOVE_BACKGROUND_METHOD: (AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,)},
            "status": {REMOVE_BACKGROUND_METHOD: "Removing Music / Background noise"},
            "pos": 10,
        }

    def download(self, method, process_files: Callable[..., Any], send_cmd=None, status_text=None, **_kwargs) -> bool:
        from shared.utils.download import download_audio_background_replacement

        return download_audio_background_replacement(send_cmd, status_text or "Downloading audio background replacement model files...")

    def query_download_defs(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        from shared.utils.download import query_audio_background_replacement_download_def

        return [query_audio_background_replacement_download_def()]

    def process_audio_file(self, method, audio_source, output_path, send_cmd=None, status_callback=None, **_kwargs) -> str:
        from preprocessing.extract_vocals import get_vocals

        if status_callback is not None:
            status_callback(method_metadata(method)["status"])
        return get_vocals(audio_source, output_path)

