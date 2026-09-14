"""Stop abandoned Gradio media transfers and batch small range reads."""
from gradio import ranged_response, routes

from shared.utils.http_disconnect import DisconnectAwareFileResponse, DisconnectAwareResponse, install_http_disconnect_patch


class _RangedFileResponse(DisconnectAwareResponse, ranged_response.RangedFileResponse):
    chunk_size = 256 * 1024


def install():
    if ranged_response.RangedFileResponse is _RangedFileResponse:
        return
    install_http_disconnect_patch()
    routes.FileResponse = DisconnectAwareFileResponse
    ranged_response.RangedFileResponse = _RangedFileResponse
