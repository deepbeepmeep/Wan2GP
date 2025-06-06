import av
import torch
import io
import numpy as np


def _encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def _decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def compress(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    image_array = (
        (image[: (image.shape[0] // 2) * 2, : (image.shape[1] // 2) * 2] * 255.0)
        .byte()
        .cpu()
        .numpy()
    )
    with io.BytesIO() as output_file:
        _encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = _decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor
