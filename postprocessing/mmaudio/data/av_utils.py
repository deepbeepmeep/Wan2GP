from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
import torch
from av import AudioFrame


@dataclass
class VideoInfo:
    duration_sec: float
    fps: Fraction
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    all_frames: Optional[list[np.ndarray]]

    @property
    def height(self):
        return self.all_frames[0].shape[0]

    @property
    def width(self):
        return self.all_frames[0].shape[1]

    @classmethod
    def from_image_info(cls, image_info: 'ImageInfo', duration_sec: float,
                        fps: Fraction) -> 'VideoInfo':
        num_frames = int(duration_sec * fps)
        all_frames = [image_info.original_frame] * num_frames
        return cls(duration_sec=duration_sec,
                   fps=fps,
                   clip_frames=image_info.clip_frames,
                   sync_frames=image_info.sync_frames,
                   all_frames=all_frames)


@dataclass
class ImageInfo:
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    original_frame: Optional[np.ndarray]

    @property
    def height(self):
        return self.original_frame.shape[0]

    @property
    def width(self):
        return self.original_frame.shape[1]


def read_frames(video_path: Path, list_of_fps: list[float], start_sec: float, end_sec: float,
                need_all_frames: bool) -> tuple[list[np.ndarray], list[np.ndarray], Fraction]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    if not fps_val or fps_val <= 0:
        cap.release()
        raise RuntimeError(f"Could not read fps from {video_path}")
    fps = Fraction(fps_val).limit_denominator()

    start_frame = int(start_sec * fps_val)
    end_frame = int(end_sec * fps_val)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_frames = [[] for _ in list_of_fps]
    next_frame_time_for_each_fps = [start_sec for _ in list_of_fps]
    time_delta_for_each_fps = [1 / f for f in list_of_fps]
    all_frames = []

    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_time = frame_idx / fps_val  # seconds
        frame_rgb = None

        if need_all_frames:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)

        for i, _ in enumerate(list_of_fps):
            while frame_time >= next_frame_time_for_each_fps[i]:
                if frame_rgb is None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                output_frames[i].append(frame_rgb)
                next_frame_time_for_each_fps[i] += time_delta_for_each_fps[i]

    cap.release()
    output_frames = [np.stack(frames) for frames in output_frames]
    return output_frames, all_frames, fps


def reencode_with_audio(video_info: VideoInfo, output_path: Path, audio: torch.Tensor,
                        sampling_rate: int):
    container = av.open(output_path, 'w')
    output_video_stream = container.add_stream('h264', video_info.fps)
    output_video_stream.codec_context.bit_rate = 10 * 1e6  # 10 Mbps
    output_video_stream.width = video_info.width
    output_video_stream.height = video_info.height
    output_video_stream.pix_fmt = 'yuv420p'

    output_audio_stream = container.add_stream('aac', sampling_rate)

    # encode video
    for image in video_info.all_frames:
        image = av.VideoFrame.from_ndarray(image)
        packet = output_video_stream.encode(image)
        container.mux(packet)

    for packet in output_video_stream.encode():
        container.mux(packet)

    # convert float tensor audio to numpy array
    audio_np = audio.numpy().astype(np.float32)
    audio_frame = AudioFrame.from_ndarray(audio_np, format='flt', layout='mono')
    audio_frame.sample_rate = sampling_rate

    for packet in output_audio_stream.encode(audio_frame):
        container.mux(packet)

    for packet in output_audio_stream.encode():
        container.mux(packet)

    container.close()



import subprocess
import tempfile
from pathlib import Path
import torch

def remux_with_audio(video_path: Path, output_path: Path, audio: torch.Tensor, sampling_rate: int):
    from shared.utils.audio_video import extract_audio_tracks, combine_video_with_audio_tracks, cleanup_temp_audio_files

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = Path(f.name)
    temp_path_str= str(temp_path)
    import torchaudio
    torchaudio.save(temp_path_str, audio.unsqueeze(0) if audio.dim() == 1 else audio, sampling_rate)

    combine_video_with_audio_tracks(video_path, [temp_path_str], output_path )
    temp_path.unlink(missing_ok=True)

def remux_with_audio_old(video_path: Path, audio: torch.Tensor, output_path: Path, sampling_rate: int):
    """
    NOTE: I don't think we can get the exact video duration right without re-encoding
    so we are not using this but keeping it here for reference
    """
    video = av.open(video_path)
    output = av.open(output_path, 'w')
    input_video_stream = video.streams.video[0]
    output_video_stream = output.add_stream(template=input_video_stream)
    output_audio_stream = output.add_stream('aac', sampling_rate)

    duration_sec = audio.shape[-1] / sampling_rate

    for packet in video.demux(input_video_stream):
        # We need to skip the "flushing" packets that `demux` generates.
        if packet.dts is None:
            continue
        # We need to assign the packet to the new stream.
        packet.stream = output_video_stream
        output.mux(packet)

    # convert float tensor audio to numpy array
    audio_np = audio.numpy().astype(np.float32)
    audio_frame = av.AudioFrame.from_ndarray(audio_np, format='flt', layout='mono')
    audio_frame.sample_rate = sampling_rate

    for packet in output_audio_stream.encode(audio_frame):
        output.mux(packet)

    for packet in output_audio_stream.encode():
        output.mux(packet)

    video.close()
    output.close()

    output.close()
