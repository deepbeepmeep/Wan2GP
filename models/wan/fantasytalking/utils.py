# Copyright Alibaba Inc. All Rights Reserved.

import imageio
import librosa
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def resize_image_by_longest_edge(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(
        save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
    )
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def get_audio_features(wav2vec, audio_processor, audio_path, fps, start_frame, num_frames):
    sr = 16000
    audio_input, sample_rate = librosa.load(audio_path, sr=sr)  # 采样率为 16kHz    start_time = 0
    if start_frame  < 0:
        pad = int(abs(start_frame)/ fps * sr)
        audio_input = np.concatenate([np.zeros(pad), audio_input])
        end_frame = num_frames
    else:
        end_frame = start_frame + num_frames

    start_time = start_frame / fps
    end_time = end_frame / fps

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    try:
        audio_segment = audio_input[start_sample:end_sample]
    except:
        audio_segment = audio_input

    input_values = audio_processor(
        audio_segment, sampling_rate=sample_rate, return_tensors="pt"
    ).input_values.to("cuda")

    with torch.no_grad():
        fea = wav2vec(input_values).last_hidden_state

    return fea
