# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""SVG drawings for Deepy's Draw SVG tool, rasterized by resvg.

Only the drawing itself is trusted to resvg, which would otherwise read local files named in href attributes: every
reference outside the drawing (an href that is not #id or an embedded raster image) is resolved by the caller through
its media and file access rules, then embedded as a data URI. url() may only point inside the drawing, and DOCTYPE /
ENTITY declarations are rejected.
"""
from __future__ import annotations

import base64
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

SVG_NS, XLINK_NS = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
HREFS = ("href", f"{{{XLINK_NS}}}href")
MAX_SVG_CHARS, MAX_SIDE, MAX_REFERENCES = 2_000_000, 4096, 32
EMBEDDED_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
_EMBEDDED = re.compile(r"data:image/(png|jpeg|gif);base64,", re.IGNORECASE)
_EXTERNAL_URL = re.compile(r"url\(\s*['\"]?(?!#)", re.IGNORECASE)
_LENGTH = re.compile(r"^\s*([0-9.]+)\s*(px)?\s*$")
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _parse(svg: str) -> ET.Element:
    if not isinstance(svg, str) or not svg.strip():
        raise ValueError("svg must be a non-empty SVG document.")
    if len(svg) > MAX_SVG_CHARS:
        raise ValueError(f"The SVG exceeds {MAX_SVG_CHARS:,} characters.")
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", svg, re.IGNORECASE):
        raise ValueError("The SVG must not contain DOCTYPE or ENTITY declarations.")
    if _EXTERNAL_URL.search(svg):
        raise ValueError("url() may only reference elements of the drawing, as url(#id) (gradients, patterns, filters, clip paths, masks).")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as error:
        raise ValueError(f"Invalid SVG markup: {error}.") from error
    if root.tag not in ("svg", f"{{{SVG_NS}}}svg"):
        raise ValueError("The root element must be <svg>.")
    return root


def references(svg: str) -> list[str]:
    """Distinct external references of a drawing, in document order: every href that is neither #id nor an embedded
    PNG / JPEG / GIF data URI."""
    found = []
    for element in _parse(svg).iter():
        for name in HREFS:
            value = (element.get(name) or "").strip()
            if value and not value.startswith("#") and not _EMBEDDED.match(value) and value not in found:
                found.append(value)
    if len(found) > MAX_REFERENCES:
        raise ValueError(f"A drawing can reference at most {MAX_REFERENCES} different images.")
    return found


def rewrite(svg: str, replacements: dict[str, str]) -> str:
    """The drawing with each external reference replaced (exact href values)."""
    root = _parse(svg)
    for element in root.iter():
        for name in HREFS:
            value = (element.get(name) or "").strip()
            if value in replacements:
                element.set(name, replacements[value])
    return ET.tostring(root, encoding="unicode")


def data_uri(path: str) -> str:
    """An image file as a data URI: PNG, JPEG and GIF as they are, other formats converted to PNG (first frame)."""
    mime = EMBEDDED_TYPES.get(Path(path).suffix.lower())
    if mime is not None:
        data = Path(path).read_bytes()
    else:
        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGBA").save(buffer, "PNG")
        data, mime = buffer.getvalue(), "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _intrinsic_size(root: ET.Element) -> tuple[float, float] | None:
    lengths = [_LENGTH.match(root.get(name) or "") for name in ("width", "height")]
    if all(lengths):
        return float(lengths[0].group(1)), float(lengths[1].group(1))
    box = (root.get("viewBox") or "").replace(",", " ").split()
    return (float(box[2]), float(box[3])) if len(box) == 4 else None


def rasterize(svg: str, width: int | None = None, height: int | None = None, background: str | None = None) -> Image.Image:
    """RGBA image of a self-contained drawing (no external reference left), at most MAX_SIDE pixels per side: width /
    height scale the drawing (one of them keeps its aspect ratio), background is a CSS color (transparent by default)."""
    import resvg_py

    if references(svg):
        raise ValueError("The drawing still has external references.")
    if any(value is not None and not 1 <= value <= MAX_SIDE for value in (width, height)):
        raise ValueError(f"width and height must be between 1 and {MAX_SIDE} pixels.")
    zoom = None
    if width is None and height is None and (size := _intrinsic_size(_parse(svg))) and max(size) > MAX_SIDE:
        zoom = MAX_SIDE / max(size)
    image = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height, zoom=zoom, background=background))))
    image.load()
    if max(image.size) > MAX_SIDE:
        raise ValueError(f"The drawing renders at {image.width}x{image.height}; set width or height (at most {MAX_SIDE}).")
    return image.convert("RGBA")
