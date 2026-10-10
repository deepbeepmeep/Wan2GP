"""Preserve Gradio error display options across Deepy's event and HTTP APIs."""
from gradio import Error
from shared.utils.form_sync import FormConflict


class DeepyBusy(ValueError):
    """An action cannot run yet; present this as information in the UI."""


def error_payload(error):
    if isinstance(error, (DeepyBusy, FormConflict)):
        return {'message': str(error), 'title': 'Info', 'duration': 5, 'visible': True, 'level': 'info'}
    if isinstance(error, Error):
        message = error.message
        while isinstance(message, Error):
            message = message.message
        return {'message': str(message), **{key: getattr(error, key) for key in ('title', 'duration', 'visible')}}
    return {'message': str(error), 'title': 'Error', 'duration': 10, 'visible': True}
