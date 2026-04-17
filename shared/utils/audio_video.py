import subprocess
import tempfile, os
import ffmpeg
import struct
from typing import Any
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import tempfile
import imageio
import binascii
import torchvision
import torch
from PIL import Image
import os.path as osp
import json
import numpy as np
import soundfile as sf
import zlib

from .video_decode import probe_video_stream_metadata
from .virtual_media import get_virtual_media_entry, parse_virtual_media_path, strip_virtual_media_suffix

def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def _prepare_audio_array(audio_data):
    if torch.is_tensor(audio_data):
        audio_data = audio_data.detach().cpu().float().numpy()
    else:
        audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.ndim == 2 and audio_data.shape[0] <= 8 and audio_data.shape[1] > audio_data.shape[0]:
        audio_data = audio_data.T
    return audio_data


def write_wav_file(path, audio_data, sample_rate):
    audio_array = _prepare_audio_array(audio_data)
    sf.write(path, audio_array, int(sample_rate))
    return path


def resample_audio_array(audio_data, source_sample_rate, target_sample_rate):
    audio_array = np.asarray(audio_data, dtype=np.float32)
    source_sample_rate = int(source_sample_rate or 0)
    target_sample_rate = int(target_sample_rate or 0)
    if audio_array.size == 0 or source_sample_rate <= 0 or target_sample_rate <= 0 or source_sample_rate == target_sample_rate:
        return audio_array.astype(np.float32, copy=False)
    import torchaudio.functional as taF
    wave = torch.from_numpy(audio_array.T.copy() if audio_array.ndim == 2 else audio_array[None].copy()).to(dtype=torch.float32)
    resampled = taF.resample(wave, source_sample_rate, target_sample_rate).cpu().numpy()
    return (resampled.T if audio_array.ndim == 2 else resampled[0]).astype(np.float32, copy=False)


def append_sliding_window_audio(existing_audio_data, existing_audio_path, generated_audio, audio_sampling_rate, committed_audio_samples, existing_audio_sample_rate=None):
    generated_audio = np.asarray(generated_audio, dtype=np.float32)
    if generated_audio.size == 0:
        return generated_audio
    prefix_sample_rate = int(existing_audio_sample_rate or audio_sampling_rate)
    if existing_audio_data is not None:
        prefix_audio = np.asarray(existing_audio_data, dtype=np.float32)
    elif existing_audio_path:
        prefix_audio, prefix_sample_rate = sf.read(os.fspath(existing_audio_path), dtype="float32", always_2d=generated_audio.ndim == 2)
    else:
        return generated_audio
    if prefix_sample_rate != int(audio_sampling_rate):
        prefix_audio = resample_audio_array(prefix_audio, prefix_sample_rate, audio_sampling_rate)
    prefix_audio = prefix_audio[:max(0, int(committed_audio_samples))]
    if prefix_audio.size == 0:
        return generated_audio
    if prefix_audio.ndim != generated_audio.ndim:
        prefix_audio = prefix_audio[:, None] if prefix_audio.ndim == 1 else prefix_audio
        generated_audio = generated_audio[:, None] if generated_audio.ndim == 1 else generated_audio
    if prefix_audio.ndim == 2 and prefix_audio.shape[1] != generated_audio.shape[1]:
        prefix_audio = np.repeat(prefix_audio[:, :1], generated_audio.shape[1], axis=1) if prefix_audio.shape[1] == 1 else prefix_audio[:, :generated_audio.shape[1]]
    return np.concatenate([prefix_audio, generated_audio], axis=0)


def create_silent_wav_file(output_dir=None, duration_seconds=0.0, sample_rate=16000, prefix="null_audio_"):
    sample_rate = int(sample_rate)
    num_samples = max(1, int(np.ceil(float(duration_seconds) * sample_rate)))
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=output_dir)
    os.close(fd)
    return write_wav_file(path, np.zeros(num_samples, dtype=np.float32), sample_rate)


def _compute_active_abs_amplitude(audio_data):
    abs_audio = np.abs(np.asarray(audio_data, dtype=np.float32)).reshape(-1)
    if abs_audio.size == 0:
        return 0.0, 0.0
    avg_abs = float(abs_audio.mean())
    if avg_abs <= 0.0:
        return 0.0, 0.0
    threshold = 0.1 * avg_abs
    active_mask = abs_audio > threshold
    active_avg_abs = float(abs_audio[active_mask].mean()) if np.any(active_mask) else avg_abs
    return avg_abs, active_avg_abs


def normalize_audio_pair_volumes_to_temp_files(audio_path1, audio_path2, output_dir=None, prefix="audio_norm_"):
    audio1, sr1 = sf.read(os.fspath(audio_path1), dtype="float32", always_2d=False)
    audio2, sr2 = sf.read(os.fspath(audio_path2), dtype="float32", always_2d=False)

    avg1, active1 = _compute_active_abs_amplitude(audio1)
    avg2, active2 = _compute_active_abs_amplitude(audio2)
    midpoint = 0.5 * (active1 + active2)
    eps = 1e-8
    gain1 = midpoint / active1 if active1 > eps else 1.0
    gain2 = midpoint / active2 if active2 > eps else 1.0

    norm1 = np.clip(np.asarray(audio1, dtype=np.float32) * float(gain1), -1.0, 1.0)
    norm2 = np.clip(np.asarray(audio2, dtype=np.float32) * float(gain2), -1.0, 1.0)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    fd1, out1 = tempfile.mkstemp(prefix=prefix + "1_", suffix=".wav", dir=output_dir)
    os.close(fd1)
    fd2, out2 = tempfile.mkstemp(prefix=prefix + "2_", suffix=".wav", dir=output_dir)
    os.close(fd2)
    sf.write(out1, norm1, int(sr1))
    sf.write(out2, norm2, int(sr2))

    stats = {
        "audio1_avg_abs": float(avg1),
        "audio2_avg_abs": float(avg2),
        "audio1_active_avg_abs": float(active1),
        "audio2_active_avg_abs": float(active2),
        "target_active_avg_abs": float(midpoint),
        "audio1_gain": float(gain1),
        "audio2_gain": float(gain2),
    }
    return out1, out2, stats


def _get_audio_codec_settings(codec_key):
    if not codec_key:
        codec_key = "wav"
    codec_key = str(codec_key).lower()
    if codec_key == "mp3":
        codec_key = "mp3_192"
    settings = {
        "wav": {"ext": "wav", "format": "wav"},
        "mp3_128": {"ext": "mp3", "format": "mp3", "bitrate": "128k"},
        "mp3_192": {"ext": "mp3", "format": "mp3", "bitrate": "192k"},
        "mp3_320": {"ext": "mp3", "format": "mp3", "bitrate": "320k"},
    }
    return settings.get(codec_key, settings["wav"])


def get_mp4_audio_codec_settings(codec_key):
    codec_key = "aac_128" if not codec_key else str(codec_key).lower()
    settings = {
        "aac_128": {"codec": "aac", "bitrate": "128k", "ext": ".aac"},
        "aac_192": {"codec": "aac", "bitrate": "192k", "ext": ".aac"},
        "aac_256": {"codec": "aac", "bitrate": "256k", "ext": ".aac"},
        "aac_320": {"codec": "aac", "bitrate": "320k", "ext": ".aac"},
        "alac": {"codec": "alac", "bitrate": None, "ext": ".m4a"},
    }
    return settings.get(codec_key, settings["aac_128"])


def get_video_encode_args(codec_key: str | None, container: str | None) -> list[str]:
    codec_key = str(codec_key or "libx264_8").strip().lower() or "libx264_8"
    container = str(container or "mp4").strip().lower() or "mp4"
    if codec_key == "libx264_8":
        return ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"]
    if codec_key == "libx264_10":
        return ["-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p"]
    if codec_key == "libx265_28":
        return ["-c:v", "libx265", "-crf", "28", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none"]
    if codec_key == "libx265_8":
        return ["-c:v", "libx265", "-crf", "8", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none"]
    if codec_key == "libx264_lossless":
        if container == "mkv":
            return ["-c:v", "ffv1", "-pix_fmt", "rgb24"]
        return ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"]
    return ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"]


def get_audio_codec_extension(codec_key):
    return _get_audio_codec_settings(codec_key)["ext"]


def _run_ffmpeg_encode(input_path, output_path, codec, bitrate=None, sample_rate=None, drop_video=False):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", input_path]
    if drop_video:
        cmd.append("-vn")
    cmd += ["-c:a", codec]
    if bitrate:
        cmd += ["-b:a", bitrate]
    if sample_rate:
        cmd += ["-ar", str(int(sample_rate))]
    cmd.append(output_path)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def save_audio_file(path, audio_data, sample_rate, codec_key="wav"):
    settings = _get_audio_codec_settings(codec_key)
    ext = settings["ext"]
    if not path.lower().endswith(f".{ext}"):
        path = osp.splitext(path)[0] + f".{ext}"
    if settings["format"] == "wav":
        return write_wav_file(path, audio_data, sample_rate)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="audio_")
    os.close(fd)
    try:
        write_wav_file(tmp_path, audio_data, sample_rate)
        _run_ffmpeg_encode(tmp_path, path, "libmp3lame", bitrate=settings.get("bitrate"), sample_rate=sample_rate)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return path


def _resolve_virtual_audio_segment(video_path: str) -> tuple[str, dict[str, Any], int]:
    if isinstance(video_path, Image.Image):
        return "", {}, 0
    if get_virtual_media_entry(video_path) is not None:
        return "", {}, 0
    spec = parse_virtual_media_path(video_path)
    source_path = os.fspath(strip_virtual_media_suffix(video_path))
    time_args: dict[str, Any] = {}
    if spec is None:
        return source_path, time_args, 0
    metadata = probe_video_stream_metadata(video_path)
    if metadata is not None and metadata.get("virtual_end_frame") is not None:
        start_frame = int(metadata.get("virtual_start_frame") or 0)
        end_frame = int(metadata.get("virtual_end_frame") or start_frame)
        fps_float = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
        if fps_float > 0:
            time_args["ss"] = max(0.0, start_frame / fps_float)
            time_args["to"] = max(time_args["ss"], (end_frame + 1) / fps_float)
    audio_track_no = 1 if spec.audio_track_no is None else max(1, int(spec.audio_track_no))
    return source_path, time_args, audio_track_no - 1


def extract_audio_track_to_wav(video_path, output_path):
    if not video_path:
        return None
    if isinstance(video_path, Image.Image):
        return None
    video_path = os.fspath(video_path)
    source_path, time_args, audio_track_index = _resolve_virtual_audio_segment(video_path)
    if len(source_path) == 0:
        return None
    import ffmpeg
    try:
        output_kwargs = {"map": f"0:a:{audio_track_index}", "acodec": "pcm_s16le"}
        ffmpeg.input(source_path, **time_args).output(output_path, **output_kwargs).overwrite_output().run(quiet=True)
    except ffmpeg.Error as err:
        stderr = getattr(err, "stderr", b"")
        if isinstance(stderr, (bytes, bytearray)):
            stderr = stderr.decode("utf-8", errors="ignore")
        stderr = (stderr or str(err)).strip()
        raise RuntimeError(f"ffmpeg audio extract failed for {source_path} -> {output_path}: {stderr}") from err
    return output_path



def extract_audio_tracks(source_video, verbose=False, query_only=False, codec_key="aac_128", temp_format=None):
    """
    Extract all audio tracks from a source video into temporary audio files.

    Returns:
        Tuple:
          - List of temp file paths for extracted audio tracks
          - List of corresponding metadata dicts:
              {'codec', 'sample_rate', 'channels', 'duration', 'language'}
              where 'duration' is set to container duration (for consistency).
    """
    if isinstance(source_video, Image.Image):
        return 0 if query_only else ([], [])
    source_path, time_args, selected_track_index = _resolve_virtual_audio_segment(source_video)
    if len(source_path) == 0:
        return 0 if query_only else ([], [])
    if not os.path.exists(source_path):
        msg = f"ffprobe skipped; file not found: {source_video}"
        if verbose:
            print(msg)
        raise FileNotFoundError(msg)

    try:
        probe = ffmpeg.probe(source_path)
    except ffmpeg.Error as err:
        stderr = getattr(err, 'stderr', b'')
        if isinstance(stderr, (bytes, bytearray)):
            stderr = stderr.decode('utf-8', errors='ignore')
        stderr = (stderr or str(err)).strip()
        message = f"ffprobe failed for {source_path}: {stderr}"
        if verbose:
            print(message)
        raise RuntimeError(message) from err
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    container_duration = float(probe['format'].get('duration', 0.0))
    if selected_track_index is not None:
        audio_streams = [audio_streams[selected_track_index]] if 0 <= selected_track_index < len(audio_streams) else []

    if not audio_streams:
        if query_only: return 0
        if verbose: print(f"No audio track found in {source_video}")
        return [], []

    if query_only:
        return len(audio_streams)

    if verbose:
        print(f"Found {len(audio_streams)} audio track(s), container duration = {container_duration:.3f}s")

    file_paths = []
    metadata = []
    if temp_format == "wav":
        audio_settings = {"codec": "pcm_s16le", "bitrate": None, "ext": ".wav"}
    else:
        audio_settings = get_mp4_audio_codec_settings(codec_key)

    for i, stream in enumerate(audio_streams):
        fd, temp_path = tempfile.mkstemp(suffix=f'_track{i}{audio_settings["ext"]}', prefix='audio_')
        os.close(fd)

        file_paths.append(temp_path)
        metadata.append({
            'codec': stream.get('codec_name'),
            'sample_rate': int(stream.get('sample_rate', 0)),
            'channels': int(stream.get('channels', 0)),
            'duration': container_duration,
            'language': stream.get('tags', {}).get('language', None)
        })

        stream_index = i if selected_track_index is None else selected_track_index
        output_kwargs = {f'map': f'0:a:{stream_index}', 'acodec': audio_settings["codec"]}
        if audio_settings["bitrate"]:
            output_kwargs['b:a'] = audio_settings["bitrate"]
        ffmpeg.input(source_path, **time_args).output(temp_path, **output_kwargs).overwrite_output().run(quiet=not verbose)

    return file_paths, metadata



def combine_and_concatenate_video_with_audio_tracks(
    save_path_tmp, video_path,
    source_audio_tracks, new_audio_tracks,
    source_audio_duration, audio_sampling_rate,
    new_audio_from_start=False,
    source_audio_metadata=None,
    audio_codec_key="aac_128",
    verbose = False
):
    audio_settings = get_mp4_audio_codec_settings(audio_codec_key)
    audio_codec = audio_settings["codec"]
    audio_bitrate = audio_settings["bitrate"]
    inputs, filters, maps, idx = ['-i', video_path], [], ['-map', '0:v'], 1
    metadata_args = []
    sources = source_audio_tracks or []
    news = new_audio_tracks or []

    duplicate_source = len(sources) == 1 and len(news) > 1
    N = len(news) if source_audio_duration == 0 else max(len(sources), len(news)) or 1

    for i in range(N):
        s = (sources[i] if i < len(sources)
             else sources[0] if duplicate_source else None)
        n = news[i] if len(news) == N else (news[0] if news else None)

        if source_audio_duration == 0:
            if n:
                inputs += ['-i', n]
                filters.append(f'[{idx}:a]apad=pad_dur=100[aout{i}]')
                idx += 1
            else:
                filters.append(f'anullsrc=r={audio_sampling_rate}:cl=mono,apad=pad_dur=100[aout{i}]')
        else:
            if s:
                inputs += ['-i', s]
                meta = source_audio_metadata[i] if source_audio_metadata and i < len(source_audio_metadata) else {}
                needs_filter = (
                    meta.get('codec') != audio_codec or
                    meta.get('sample_rate') != audio_sampling_rate or
                    meta.get('channels') != 1 or
                    meta.get('duration', 0) < source_audio_duration
                )
                if needs_filter:
                    filters.append(
                        f'[{idx}:a]aresample={audio_sampling_rate},aformat=channel_layouts=mono,'
                        f'apad=pad_dur={source_audio_duration},atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')
                else:
                    filters.append(
                        f'[{idx}:a]apad=pad_dur={source_audio_duration},atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')
                if lang := meta.get('language'):
                    metadata_args += ['-metadata:s:a:' + str(i), f'language={lang}']
                idx += 1
            else:
                filters.append(
                    f'anullsrc=r={audio_sampling_rate}:cl=mono,atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')

            if n:
                inputs += ['-i', n]
                start = '0' if new_audio_from_start else source_audio_duration
                filters.append(
                    f'[{idx}:a]aresample={audio_sampling_rate},aformat=channel_layouts=mono,'
                    f'atrim=start={start},asetpts=PTS-STARTPTS[n{i}]')
                filters.append(f'[s{i}][n{i}]concat=n=2:v=0:a=1[aout{i}]')
                idx += 1
            else:
                filters.append(f'[s{i}]apad=pad_dur=100[aout{i}]')

        maps += ['-map', f'[aout{i}]']

    cmd = ['ffmpeg', '-y', *inputs,
           '-filter_complex', ';'.join(filters),  # ✅ Only change made
           *maps, *metadata_args,
           '-c:v', 'copy',
           '-c:a', audio_codec,
           '-ar', str(audio_sampling_rate),
           '-ac', '1',
           '-shortest', save_path_tmp]
    if audio_bitrate:
        cmd[-6:-6] = ['-b:a', audio_bitrate]

    if verbose:
        print(f"ffmpeg command: {cmd}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr}")


def combine_video_with_audio_tracks(target_video, audio_tracks, output_video,
                                     audio_metadata=None, verbose=False):
    if not audio_tracks:
        if verbose: print("No audio tracks to combine."); return False

    dur = float(next(s for s in ffmpeg.probe(target_video)['streams']
                     if s['codec_type'] == 'video')['duration'])
    if verbose: print(f"Video duration: {dur:.3f}s")

    cmd = ['ffmpeg', '-y', '-i', target_video]
    for path in audio_tracks:
        cmd += ['-i', path]

    cmd += ['-map', '0:v']
    for i in range(len(audio_tracks)):
        cmd += ['-map', f'{i+1}:a']

    for i, meta in enumerate(audio_metadata or []):
        if (lang := meta.get('language')):
            cmd += ['-metadata:s:a:' + str(i), f'language={lang}']

    cmd += ['-c:v', 'copy', '-c:a', 'copy', '-t', str(dur), output_video]

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error:\n{result.stderr}")
    if verbose:
        print(f"Created {output_video} with {len(audio_tracks)} audio track(s)")
    return True


def cleanup_temp_audio_files(audio_tracks, verbose=False):
    """
    Clean up temporary audio files.
    
    Args:
        audio_tracks: List of audio file paths to delete
        verbose: Enable verbose output (default: False)
        
    Returns:
        Number of files successfully deleted
    """
    deleted_count = 0
    
    for audio_path in audio_tracks:
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                deleted_count += 1
                if verbose:
                    print(f"Cleaned up {audio_path}")
        except PermissionError:
            print(f"Warning: Could not delete {audio_path} (file may be in use)")
        except Exception as e:
            print(f"Warning: Error deleting {audio_path}: {e}")
    
    if verbose and deleted_count > 0:
        print(f"Successfully deleted {deleted_count} temporary audio file(s)")
    
    return deleted_count


def save_video(tensor,
                save_file=None,
                fps=30,
                codec_type='libx264_8',
                container='mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    """Save tensor as video with configurable codec and container options."""
        
    if torch.is_tensor(tensor) and len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
        
    suffix = f'.{container}'
    cache_file = osp.join('/tmp', rand_name(suffix=suffix)) if save_file is None else save_file
    if not cache_file.endswith(suffix):
        cache_file = osp.splitext(cache_file)[0] + suffix
    
    # Configure codec parameters
    codec_params = _get_codec_params(codec_type, container)
    
    # Process and save
    error = None
    for _ in range(retry):
        try:
            # Write video (silence ffmpeg logs)
            writer = imageio.get_writer(cache_file, fps=fps, ffmpeg_log_level='error', **codec_params)
            try:
                if torch.is_tensor(tensor):
                    # Stream frames to avoid materializing the full video on CPU.
                    if tensor.dtype == torch.uint8 and tensor.ndim == 5 and tensor.shape[0] == 1 and nrow == 1:
                        frames = tensor[0].permute(1, 2, 3, 0)
                        for frame in frames:
                            writer.append_data(frame.cpu().numpy())
                    else:
                        if tensor.dtype == torch.uint8:
                            tensor = tensor.float().div_(127.5).sub_(1.0)
                        for u in tensor.unbind(2):
                            u = u.clamp(min(value_range), max(value_range))
                            grid = torchvision.utils.make_grid(
                                u, nrow=nrow, normalize=normalize, value_range=value_range
                            )
                            frame = grid.mul(255).type(torch.uint8).permute(1, 2, 0).cpu().numpy()
                            writer.append_data(frame)
                elif isinstance(tensor, (list, tuple)) and tensor and torch.is_tensor(tensor[0]):
                    for chunk in tensor:
                        if chunk is None:
                            continue
                        if chunk.ndim == 4:
                            if chunk.shape[-1] in (1, 3, 4):
                                frames = chunk
                            else:
                                frames = chunk.permute(1, 2, 3, 0)
                            for frame in frames:
                                writer.append_data(frame.cpu().numpy())
                        else:
                            writer.append_data(chunk)
                else:
                    for frame in tensor:
                        writer.append_data(frame)
            finally:
                writer.close()

            return cache_file

        except Exception as e:
            error = e
            print(f"error saving {save_file}: {e}")


def _get_codec_params(codec_type, container):
    """Get codec parameters based on codec type and container."""
    if codec_type == 'libx264_8':
        return {'codec': 'libx264', 'quality': 8, 'pixelformat': 'yuv420p'}
    elif codec_type == 'libx264_10':
        return {'codec': 'libx264', 'quality': 10, 'pixelformat': 'yuv420p'}
    elif codec_type == 'libx265_28':
        return {'codec': 'libx265', 'pixelformat': 'yuv420p', 'output_params': ['-crf', '28', '-x265-params', 'log-level=none','-hide_banner', '-nostats']}
    elif codec_type == 'libx265_8':
        return {'codec': 'libx265', 'pixelformat': 'yuv420p', 'output_params': ['-crf', '8', '-x265-params', 'log-level=none','-hide_banner', '-nostats']}
    elif codec_type == 'libx264_lossless':
        if container == 'mkv':
            return {'codec': 'ffv1', 'pixelformat': 'rgb24'}
        else:  # mp4
            return {'codec': 'libx264', 'output_params': ['-crf', '0'], 'pixelformat': 'yuv444p'}
    else:  # libx264
        return {'codec': 'libx264', 'pixelformat': 'yuv420p'}




def save_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                quality='jpeg_95',  # 'jpeg_95', 'jpeg_85', 'jpeg_70', 'jpeg_50', 'webp_95', 'webp_85', 'webp_70', 'webp_50', 'png', 'webp_lossless'
                retry=5):
    """Save tensor as image with configurable format and quality."""

    RGBA = tensor.shape[0] == 4
    if RGBA:
        quality = "png"

    # Get format and quality settings
    format_info = _get_format_info(quality)
    
    # Rename file extension to match requested format
    save_file = osp.splitext(save_file)[0] + format_info['ext']
    
    # Save image
    error = None
                         
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            
            if format_info['use_pil'] or RGBA:
                # Use PIL for WebP and advanced options
                grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
                # Convert to PIL Image
                grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                mode = 'RGBA' if RGBA else 'RGB'
                img = Image.fromarray(grid, mode=mode)
                img.save(save_file, **format_info['params'])
            else:
                # Use torchvision for JPEG and PNG
                torchvision.utils.save_image(
                    tensor, save_file, nrow=nrow, normalize=normalize, 
                    value_range=value_range, **format_info['params']
                )
            break
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_image failed, error: {error}', flush=True)
    
    return save_file


def _get_format_info(quality):
    """Get format extension and parameters."""
    formats = {
        # JPEG with PIL (so 'quality' works)
        'jpeg_95': {'ext': '.jpg', 'params': {'quality': 95}, 'use_pil': True},
        'jpeg_85': {'ext': '.jpg', 'params': {'quality': 85}, 'use_pil': True},
        'jpeg_70': {'ext': '.jpg', 'params': {'quality': 70}, 'use_pil': True},
        'jpeg_50': {'ext': '.jpg', 'params': {'quality': 50}, 'use_pil': True},

        # PNG with torchvision
        'png': {'ext': '.png', 'params': {}, 'use_pil': False},

        # WebP with PIL (for quality control)
        'webp_95': {'ext': '.webp', 'params': {'quality': 95}, 'use_pil': True},
        'webp_85': {'ext': '.webp', 'params': {'quality': 85}, 'use_pil': True},
        'webp_70': {'ext': '.webp', 'params': {'quality': 70}, 'use_pil': True},
        'webp_50': {'ext': '.webp', 'params': {'quality': 50}, 'use_pil': True},
        'webp_lossless': {'ext': '.webp', 'params': {'lossless': True}, 'use_pil': True},
    }
    return formats.get(quality, formats['jpeg_95'])


from PIL import Image, PngImagePlugin

def _enc_uc(s):
    try: return b"ASCII\0\0\0" + s.encode("ascii")
    except UnicodeEncodeError: return b"UNICODE\0" + s.encode("utf-16le")

def _dec_uc(b):
    if not isinstance(b, (bytes, bytearray)):
        try: b = bytes(b)
        except Exception: return None
    if b.startswith(b"ASCII\0\0\0"): return b[8:].decode("ascii", "ignore")
    if b.startswith(b"UNICODE\0"):   return b[8:].decode("utf-16le", "ignore")
    return b.decode("utf-8", "ignore")


def _blank_exif_dict():
    return {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}


def _load_exif_dict(image_path, ext):
    import piexif
    try:
        if ext in (".jpg", ".jpeg"):
            return piexif.load(image_path)
        if ext == ".webp":
            with Image.open(image_path) as im:
                exif_bytes = im.info.get("exif")
            return piexif.load(exif_bytes) if exif_bytes else _blank_exif_dict()
    except Exception:
        pass
    return _blank_exif_dict()


def _insert_exif_user_comment(image_path, comment_text, ext):
    import piexif
    exif_dict = _load_exif_dict(image_path, ext)
    exif_dict.setdefault("Exif", {})
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = _enc_uc(comment_text)
    piexif.insert(piexif.dump(exif_dict), image_path)


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _build_png_chunk(chunk_type, data):
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xffffffff)


def _is_png_comment_chunk(chunk_type, data):
    if chunk_type not in {b"tEXt", b"zTXt", b"iTXt"}:
        return False
    return data.split(b"\x00", 1)[0] == b"comment"


def _write_png_comment_metadata(image_path, comment_text):
    raw = open(image_path, "rb").read()
    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    comment_chunk = _build_png_chunk(b"iTXt", b"comment\x00\x00\x00\x00\x00" + comment_text.encode("utf-8"))
    out = bytearray(_PNG_SIGNATURE)
    pos = len(_PNG_SIGNATURE)
    inserted = False
    while pos < len(raw):
        if pos + 8 > len(raw):
            raise ValueError("Corrupted PNG chunk header")
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        chunk_type = raw[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(raw):
            raise ValueError("Corrupted PNG chunk payload")
        chunk_data = raw[pos + 8:pos + 8 + length]
        chunk = raw[pos:end]
        pos = end
        if _is_png_comment_chunk(chunk_type, chunk_data):
            continue
        if not inserted and chunk_type == b"IDAT":
            out.extend(comment_chunk)
            inserted = True
        out.extend(chunk)
    if not inserted:
        raise ValueError("PNG image data chunk not found")
    with open(image_path, "wb") as writer:
        writer.write(out)

def save_image_metadata(image_path, metadata_dict, **save_kwargs):
    try:
        j = json.dumps(metadata_dict, ensure_ascii=False)
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            _write_png_comment_metadata(image_path, j); return True
        if ext in (".jpg", ".jpeg", ".webp"):
            _insert_exif_user_comment(image_path, j, ext); return True
        raise ValueError("Unsupported format")
    except Exception as e:
        print(f"Error saving metadata: {e}"); return False

def read_image_metadata(image_path):
    try:
        ext = os.path.splitext(image_path)[1].lower()
        with Image.open(image_path) as im:
            if ext == ".png":
                val = (getattr(im, "text", {}) or {}).get("comment") or im.info.get("comment")
                return json.loads(val) if val else None
            if ext in (".jpg", ".jpeg"):
                import piexif
                try:
                    uc = piexif.load(image_path).get("Exif", {}).get(piexif.ExifIFD.UserComment)
                    s = _dec_uc(uc) if uc else None
                    if s:
                        return json.loads(s)
                except Exception:
                    pass
                val = im.info.get("comment")
                if isinstance(val, (bytes, bytearray)): val = val.decode("utf-8", "ignore")
                if val:
                    try: return json.loads(val)
                    except Exception: pass
                exif = getattr(im, "getexif", lambda: None)()
                if exif:
                    uc = exif.get(37510)  # UserComment
                    s = _dec_uc(uc) if uc else None
                    if s:
                        try: return json.loads(s)
                        except Exception: pass
                return None
            if ext == ".webp":
                import piexif
                exif_bytes = im.info.get("exif")
                if not exif_bytes: return None
                uc = piexif.load(exif_bytes).get("Exif", {}).get(piexif.ExifIFD.UserComment)
                s = _dec_uc(uc) if uc else None
                return json.loads(s) if s else None
            return None
    except Exception as e:
        print(f"Error reading metadata: {e}"); return None

