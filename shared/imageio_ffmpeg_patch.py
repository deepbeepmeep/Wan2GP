"""Keep imageio's Linux FFmpeg launches from copying CUDA-pinned host RAM."""
import os
import sys


def _install():
    if sys.platform != "linux":
        return

    from imageio_ffmpeg import _io, _utils

    original = _utils._popen_kwargs

    def _popen_kwargs(prevent_sigint=False):
        kwargs = original(prevent_sigint)
        if kwargs.get("preexec_fn") is os.setpgrp:
            # A Python pre-exec callback forces fork(), which copies pinned pages.
            # Native subprocess options preserve SIGINT isolation and allow vfork().
            kwargs["preexec_fn"] = None
            if sys.version_info >= (3, 11):
                kwargs["process_group"] = 0
            else:
                kwargs["start_new_session"] = True
        return kwargs

    # _io imports the helper by value; update both reader/writer and utility calls.
    _io._popen_kwargs = _utils._popen_kwargs = _popen_kwargs


_install()
