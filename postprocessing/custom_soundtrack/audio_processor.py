import os
from typing import Any

from postprocessing.audio_processors import (
    AUDIO_PROCESSOR_TYPE_SOUNDTRACK,
    CUSTOM_SOUNDTRACK_METHOD,
)


class CustomSoundtrackProcessor:
    def __init__(self, server_config=None, files_locator=None):
        pass

    @classmethod
    def query_audio_processor_def(cls) -> dict[str, Any]:
        return {
            "name": "Custom Soundtrack",
            "processor_types": (AUDIO_PROCESSOR_TYPE_SOUNDTRACK,),
            "methods": [("Custom Soundtrack", CUSTOM_SOUNDTRACK_METHOD)],
            "method_types": {CUSTOM_SOUNDTRACK_METHOD: (AUDIO_PROCESSOR_TYPE_SOUNDTRACK,)},
            "needs_audio_source": {CUSTOM_SOUNDTRACK_METHOD: True},
            "status": {CUSTOM_SOUNDTRACK_METHOD: "Custom Audio Remuxing"},
            "pos": 10,
        }

    def validate_method(self, method, audio_source=None, has_audio_file_extension=None, media_source_exists=None, **_kwargs) -> str:
        if not media_source_exists(audio_source) or not has_audio_file_extension(audio_source):
            return "You must provide a custom Audio"
        return ""

    def generate_soundtrack(self, method, audio_source=None, **_kwargs) -> str:
        return os.fspath(audio_source)

