"""Identify writes to lost asyncio transports without filtering their warnings."""
import asyncio
import logging
import sys
import traceback
import weakref


class LostSocketDiagnostics(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.reported = weakref.WeakSet()

    def emit(self, record):
        if record.getMessage() != "socket.send() raised exception.":
            return
        frame = sys._getframe()
        try:
            while frame is not None:
                transport = frame.f_locals.get("self")
                if isinstance(transport, asyncio.BaseTransport):
                    if transport in self.reported:
                        return
                    self.reported.add(transport)
                    protocol = transport.get_protocol()
                    cycle = getattr(protocol, "cycle", None)
                    scope = getattr(protocol, "scope", None) or getattr(cycle, "scope", None) or {}
                    logging.getLogger("wangp.network").error(
                        "Write after connection loss: transport=%s protocol=%s method=%s path=%s peer=%s closing=%s lost_writes=%s\n%s",
                        type(transport).__name__, type(protocol).__name__, scope.get("method"), scope.get("path"),
                        transport.get_extra_info("peername"), transport.is_closing(), getattr(transport, "_conn_lost", None),
                        "".join(traceback.format_stack(frame, limit=20)),
                    )
                    return
                frame = frame.f_back
        finally:
            del frame


def install_network_diagnostics():
    logger = logging.getLogger("asyncio")
    if not any(isinstance(handler, LostSocketDiagnostics) for handler in logger.handlers):
        logger.addHandler(LostSocketDiagnostics())
