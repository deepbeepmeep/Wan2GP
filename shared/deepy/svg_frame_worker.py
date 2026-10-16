# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Renders frames of an animated drawing in a separate process, so that several frames render at once.

stdin: one JSON line {"options": render options, "mode": "RGB" or "RGBA"}, then length-prefixed UTF-8 SVG documents (little-endian uint32 length).
stdout: per document, uint32 width and height then the pixel bytes; width 0 means an error, followed by its UTF-8 message.
It imports only the renderer and PIL, not WanGP."""
import io
import json
import struct
import sys

import resvg_py
from PIL import Image


def main():
    read, write = sys.stdin.buffer, sys.stdout.buffer
    config = json.loads(read.readline())
    options, mode = config["options"], config["mode"]
    while header := read.read(4):
        svg = read.read(struct.unpack("<I", header)[0]).decode("utf-8")
        try:
            frame = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, **options)))).convert(mode)
        except Exception as error:
            message = str(error).encode("utf-8")
            write.write(struct.pack("<II", 0, len(message)) + message)
        else:
            write.write(struct.pack("<II", *frame.size) + frame.tobytes())
        write.flush()


if __name__ == "__main__":
    main()
