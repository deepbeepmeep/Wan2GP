import subprocess
import tempfile, os
import ffmpeg
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

def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name



def extract_audio_tracks(source_video, verbose=False, query_only=False):
    """
    Extract all audio tracks from a source video into temporary AAC files.

    Returns:
        Tuple:
          - List of temp file paths for extracted audio tracks
          - List of corresponding metadata dicts:
              {'codec', 'sample_rate', 'channels', 'duration', 'language'}
              where 'duration' is set to container duration (for consistency).
    """
    if not os.path.exists(source_video):
        msg = f"ffprobe skipped; file not found: {source_video}"
        if verbose:
            print(msg)
        raise FileNotFoundError(msg)

    try:
        probe = ffmpeg.probe(source_video)
    except ffmpeg.Error as err:
        stderr = getattr(err, 'stderr', b'')
        if isinstance(stderr, (bytes, bytearray)):
            stderr = stderr.decode('utf-8', errors='ignore')
        stderr = (stderr or str(err)).strip()
        message = f"ffprobe failed for {source_video}: {stderr}"
        if verbose:
            print(message)
        raise RuntimeError(message) from err
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    container_duration = float(probe['format'].get('duration', 0.0))

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

    for i, stream in enumerate(audio_streams):
        fd, temp_path = tempfile.mkstemp(suffix=f'_track{i}.aac', prefix='audio_')
        os.close(fd)

        file_paths.append(temp_path)
        metadata.append({
            'codec': stream.get('codec_name'),
            'sample_rate': int(stream.get('sample_rate', 0)),
            'channels': int(stream.get('channels', 0)),
            'duration': container_duration,
            'language': stream.get('tags', {}).get('language', None)
        })

        ffmpeg.input(source_video).output(
            temp_path,
            **{f'map': f'0:a:{i}', 'acodec': 'aac', 'b:a': '128k'}
        ).overwrite_output().run(quiet=not verbose)

    return file_paths, metadata



def combine_and_concatenate_video_with_audio_tracks(
    save_path_tmp, video_path,
    source_audio_tracks, new_audio_tracks,
    source_audio_duration, audio_sampling_rate,
    new_audio_from_start=False,
    source_audio_metadata=None,
    audio_bitrate='128k',
    audio_codec='aac',
    verbose = False
):
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
           '-b:a', audio_bitrate,
           '-ar', str(audio_sampling_rate),
           '-ac', '1',
           '-shortest', save_path_tmp]

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
            if torch.is_tensor(tensor):
                # Preprocess tensor
                tensor = tensor.clamp(min(value_range), max(value_range))
                tensor = torch.stack([
                    torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range)
                    for u in tensor.unbind(2)
                ], dim=1).permute(1, 2, 3, 0)
                tensor = (tensor * 255).type(torch.uint8).cpu()
                arrays = tensor.numpy()
            else:
                arrays = tensor

            # Write video (silence ffmpeg logs)
            writer = imageio.get_writer(cache_file, fps=fps, ffmpeg_log_level='error', **codec_params)
            for frame in arrays:
                writer.append_data(frame)
        
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

def save_image_metadata(image_path, metadata_dict, **save_kwargs):
    try:
        j = json.dumps(metadata_dict, ensure_ascii=False)
        ext = os.path.splitext(image_path)[1].lower()
        with Image.open(image_path) as im:
            if ext == ".png":
                pi = PngImagePlugin.PngInfo(); pi.add_text("comment", j)
                im.save(image_path, pnginfo=pi, **save_kwargs); return True
            if ext in (".jpg", ".jpeg"):
                im.save(image_path, comment=j.encode("utf-8"), **save_kwargs); return True
            if ext == ".webp":
                import piexif
                exif = {"0th":{}, "Exif":{piexif.ExifIFD.UserComment:_enc_uc(j)}, "GPS":{}, "1st":{}, "thumbnail":None}
                im.save(image_path, format="WEBP", exif=piexif.dump(exif), **save_kwargs); return True
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
                exif_bytes = Image.open(image_path).info.get("exif")
                if not exif_bytes: return None
                import piexif
                uc = piexif.load(exif_bytes).get("Exif", {}).get(piexif.ExifIFD.UserComment)
                s = _dec_uc(uc) if uc else None
                return json.loads(s) if s else None
            return None
    except Exception as e:
        print(f"Error reading metadata: {e}"); return None
