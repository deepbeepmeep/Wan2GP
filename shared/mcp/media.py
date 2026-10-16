# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Visual inspection for external MCP clients: images for the client's own vision model (view_media), or answers from WanGP's Visual Inspector."""

import base64
import io
import math

from PIL import Image, ImageDraw, ImageFont
from mcp.types import ImageContent

from shared.deepy import video_tools, vision
from shared.utils.utils import get_video_info

INSPECTION_MODE_KEY, INSPECTION_MODES = "mcp_visual_inspection", ("agent", "wangp", "both")
VIEW_MAX_IMAGES, VIEW_MAX_EDGE, VIEW_MAX_PIXELS = 32, 1024, 8 << 20
SHEET_LABEL_HEIGHT = 22


def inspection_mode(server_config):
    mode = server_config.get(INSPECTION_MODE_KEY, "agent")
    if mode not in INSPECTION_MODES:
        raise ValueError("Invalid visual inspection mode for agents.")
    return mode


def bind_visual_inspector(toolbox, runtime, state):
    """inspect_media / inspect_video answered on the server, only text going back, by the Visual Inspector configured now:
    Deepy's remote engine, or the prompt enhancer's local Qwen VL model; where it runs sets the sample counts and image sizes."""
    from shared.remote_llm.config import is_remote_engine, resolve_role_engine

    config = runtime.module.server_config
    remote = is_remote_engine(resolve_role_engine(config, "visual_inspector"))

    def answer(media_record, question, frame_no=None, max_image_edge=None):
        if remote:
            from shared.remote_llm.deepy_runner import _visual_query

            return _visual_query(config, media_record, question, frame_no, max_image_edge)
        return runtime.module._deepy.answer_visual_query(state, media_record, question, frame_no, max_image_edge)

    toolbox.bind_runtime_tools(vision_query_callback=answer, vision_is_remote=remote)


def video_frames(path, *, times_s=None, frame_nos=None, range_s=None, count=8):
    """(frame_no, time_s) pairs: the chosen times or frame numbers, else count frames spread evenly over range_s (default: the whole video)."""
    if times_s is not None and frame_nos is not None or range_s is not None and (times_s is not None or frame_nos is not None):
        raise ValueError("Choose frames with one of times_s, frame_nos, or range_s with count.")
    fps, _width, _height, frame_count = get_video_info(path)
    fps, last = video_tools.get_precise_video_fps(path) or float(fps), max(0, int(frame_count) - 1)
    at = lambda seconds: min(last, math.floor(seconds * fps + 1e-9))
    if frame_nos is not None:
        frames = [min(last, frame) for frame in frame_nos]
    elif times_s is not None:
        frames = [at(seconds) for seconds in times_s]
    else:
        start, end = range_s or (0, (last + 1) / fps)
        if end <= start:
            raise ValueError("range_s needs an end after its start.")
        first, final = at(start), min(last, max(at(start), math.ceil(end * fps - 1e-9) - 1))
        frames = [first] if count == 1 else [round(first + (final - first) * index / (count - 1)) for index in range(count)]
    return [(frame, round(frame / fps, 3)) for frame in frames]


def tile_sheet(images, labels):
    """Tile the images in a near-square grid, each above its label."""
    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    cell_width, cell_height = max(image.width for image in images), max(image.height for image in images)
    sheet = Image.new("RGB", (columns * cell_width, rows * (cell_height + SHEET_LABEL_HEIGHT)), "black")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=SHEET_LABEL_HEIGHT - 6)
    for index, (image, label) in enumerate(zip(images, labels)):
        left, top = index % columns * cell_width, index // columns * (cell_height + SHEET_LABEL_HEIGHT)
        sheet.paste(image, (left + (cell_width - image.width) // 2, top + (cell_height - image.height) // 2))
        draw.text((left + cell_width // 2, top + cell_height + SHEET_LABEL_HEIGHT // 2), label, fill="white", font=font, anchor="mm")
    return sheet, columns, rows


def view_images(records, *, times_s=None, frame_nos=None, range_s=None, count=8, max_edge=512, bbox=None, contact_sheet=False):
    """Images and video frames for the calling agent's own vision, within the size caps."""
    bbox = None if bbox is None else vision.normalize_inspection_bbox(bbox)
    selections = []
    for record in records:
        if record["media_type"] == "video":
            selections += [(record, frame, time) for frame, time in video_frames(record["path"], times_s=times_s, frame_nos=frame_nos, range_s=range_s, count=count)]
        elif record["media_type"] == "image":
            selections.append((record, None, None))
        else:
            raise ValueError(f"view_media shows images and videos; {record['media_id']} is {record['media_type']}. Render 3D assets with render_3d_views first.")
    if len(selections) > VIEW_MAX_IMAGES:
        raise ValueError(f"view_media returns at most {VIEW_MAX_IMAGES} images; this request selects {len(selections)}. Choose fewer media, times or frames.")
    edge = min(max_edge, math.isqrt(VIEW_MAX_PIXELS // len(selections)))  # the total stays within the pixel budget
    images, decoded = [None] * len(selections), {}
    for index, (record, frame, _time) in enumerate(selections):
        if frame is None:
            with Image.open(record["path"]) as source:
                images[index] = vision.prepare_inspection_image(source, max_edge=edge, bbox=bbox)
        else:
            decoded.setdefault(record["path"], []).append((index, frame))
    for path, frames in decoded.items():
        for (index, _frame), image in zip(frames, vision.decode_inspection_video_frames(path, [frame for _index, frame in frames], max_edge=edge, bboxes=[bbox] * len(frames))):
            images[index] = image
    shown = [{"media_id": record["media_id"], **({"frame_no": frame, "time_s": time} if frame is not None else {})} for record, frame, time in selections]
    result = {"status": "done", "max_edge": edge}
    videos = {record["media_id"]: record["path"] for record in records if record["media_type"] == "video"}
    checks = {media_id: video_tools.picture_checks(path, *(range_s or (0.0, None))) for media_id, path in videos.items()}  # measured, whatever the frames show
    warnings = []
    if black := {media_id: check["black_ranges_seconds"] for media_id, check in checks.items() if "black_ranges_seconds" in check}:
        result["black_ranges_seconds"] = black
        warnings.append("Measured black stretches (seconds, per media_id): unless an intended fade, regenerate or cut them.")
    if cuts := {media_id: check["cut_times_seconds"] for media_id, check in checks.items() if "cut_times_seconds" in check}:
        result["cut_times_seconds"] = cuts
        warnings.append("Measured hard cuts (seconds, per media_id): shot changes in an edited sequence; inside one generated shot, the model cut away to another shot.")
    if warnings:
        result["warning"] = " ".join(warnings)
    if contact_sheet:
        sheet, columns, rows = tile_sheet(images, [f"{time:.2f}s" if time is not None else record["media_id"] for record, _frame, time in selections])
        images = [sheet]
        result["contact_sheet"] = {"columns": columns, "rows": rows, "width": sheet.width, "height": sheet.height, "tiles": shown, "order": "left to right, then top to bottom"}
    else:
        result["images"] = [{"image_index": index + 1, **item, "width": image.width, "height": image.height} for index, (item, image) in enumerate(zip(shown, images))]
    content = []
    for image in images:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=90)
            content.append(ImageContent(type="image", mimeType="image/jpeg", data=base64.b64encode(buffer.getvalue()).decode("ascii")))
        image.close()
    return {**result, "_mcp_images": content}

