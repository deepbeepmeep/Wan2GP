from pathlib import Path
from typing import Union

import torch
from torio.io import StreamingMediaDecoder, StreamingMediaEncoder


class VideoJoiner:

    def __init__(self, src_root: Union[str, Path], output_root: Union[str, Path], sample_rate: int,
                 duration_seconds: float):
        self.src_root = Path(src_root)
        self.output_root = Path(output_root)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds

        self.output_root.mkdir(parents=True, exist_ok=True)

    def join(self, video_id: str, output_name: str, audio: torch.Tensor):
        video_path = self.src_root / f'{video_id}.mp4'
        output_path = self.output_root / f'{output_name}.mp4'
        merge_audio_into_video(video_path, output_path, audio, self.sample_rate,
                               self.duration_seconds)


def merge_audio_into_video(video_path: Union[str, Path], output_path: Union[str, Path],
                           audio: torch.Tensor, sample_rate: int, duration_seconds: float):
    # audio: (num_samples, num_channels=1/2)

    frame_rate = 24
    # read the video
    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=int(frame_rate * duration_seconds),
        # buffer_chunk_size=1, # does not work with this -- extracted audio would be too short
        format="rgb24",
        frame_rate=frame_rate,
    )

    reader.fill_buffer()
    video_chunk = reader.pop_chunks()[0]
    t, _, h, w = video_chunk.shape

    writer = StreamingMediaEncoder(output_path)
    writer.add_audio_stream(
        sample_rate=sample_rate,
        num_channels=audio.shape[-1],
        encoder="libmp3lame",
    )
    writer.add_video_stream(frame_rate=frame_rate,
                            width=w,
                            height=h,
                            format="rgb24",
                            encoder="libx264",
                            encoder_format="yuv420p")

    with writer.open():
        writer.write_audio_chunk(0, audio.float())
        writer.write_video_chunk(1, video_chunk)


if __name__ == '__main__':
    # Usage example
    import sys
    audio = torch.randn(16000 * 4, 1)
    merge_audio_into_video(sys.argv[1], sys.argv[2], audio, 16000, 4)
