"""Notify ASGI when Uvicorn closes HTTP on EOF, before buffered writes drain."""
import anyio
from starlette.responses import FileResponse


class DisconnectAwareResponse:
    async def __call__(self, scope, receive, send):
        async def watch_disconnect():
            while (await receive())["type"] != "http.disconnect":
                pass
            tasks.cancel_scope.cancel()

        async def send_response(message):
            await send(message)
            if message['type'] == 'http.response.start':
                # BaseHTTPMiddleware must receive headers before we can stop
                # the response, even if the browser has already disconnected.
                tasks.start_soon(watch_disconnect)

        async with anyio.create_task_group() as tasks:
            await super().__call__(scope, receive, send_response)
            tasks.cancel_scope.cancel()


class DisconnectAwareFileResponse(DisconnectAwareResponse, FileResponse):
    pass


def install_http_disconnect_patch():
    from uvicorn.protocols.http.h11_impl import H11Protocol

    original = H11Protocol.eof_received
    if getattr(original, '_wangp_disconnect_patch', False):
        return

    def eof_received(self):
        keep_open = original(self)
        # H11 closes on EOF. Windows Proactor delays connection_lost until its
        # write buffer drains; without this, ASGI keeps writing to a closing socket.
        if not keep_open and self.cycle is not None:
            self.cycle.disconnected = True
            self.cycle.message_event.set()
            self.flow.resume_writing()
        return keep_open

    eof_received._wangp_disconnect_patch = True
    H11Protocol.eof_received = eof_received
