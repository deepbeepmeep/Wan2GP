# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU-only visual delivery to an external MCP client's own vision model."""

import base64
import io

from PIL import Image
from mcp.types import ImageContent

from shared.deepy import video_tools, vision


def inspection_images(media, question, frame_no=None, max_image_edge=None):
    records = media if isinstance(media, list) else [media]
    edge = vision.VISION_REMOTE_MAX_IMAGE_EDGE if max_image_edge is None else max_image_edge
    images, visuals, videos = [None] * len(records), [], {}
    for index, record in enumerate(records):
        path, kind = record["path"], record["media_type"]
        bbox, frame = record.get("bbox"), None
        if kind == "video":
            frame = video_tools.resolve_video_frame_no(path, frame_no=record.get("frame_no", frame_no), time_seconds=record.get("time_seconds"))
            videos.setdefault(path, []).append((index, frame, bbox))
        else:
            with Image.open(path) as source:
                images[index] = vision.prepare_inspection_image(source, max_edge=edge, bbox=bbox)
        visuals.append({"image_index": index + 1, "media_id": record["media_id"], "media_type": record.get("source_media_type", kind), "frame_no": frame, "time_seconds": record.get("time_seconds"), "bbox": bbox})
    for path, inputs in videos.items():
        sizing = {"max_edge": edge} if max_image_edge is None else {"max_pixels": edge * edge}
        decoded = vision.decode_inspection_video_frames(path, [frame for _, frame, _ in inputs], bboxes=[bbox for _, _, bbox in inputs], **sizing)
        for (index, _, _), image in zip(inputs, decoded):
            images[index] = image
    content = []
    for image, visual in zip(images, visuals):
        visual.update(width=image.width, height=image.height)
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            content.append(ImageContent(type="image", mimeType="image/png", data=base64.b64encode(buffer.getvalue()).decode("ascii")))
        image.close()
    return {"status": "done", "inspection": "client", "question": question, "visuals": visuals, "note": "Answer the question using the attached images in image_index order. Video inspection covers only the listed samples.", "_mcp_images": content}
