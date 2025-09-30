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
    probe = ffmpeg.probe(source_video)
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
                retry=5,
                source_images=None):
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
            
            # Embed source images if provided and container supports it
            if source_images and container in ['mkv', 'mp4']:
                try:
                    if container == 'mp4':
                        cache_file = embed_source_images_metadata_mp4(cache_file, source_images)
                    else:  # mkv
                        cache_file = embed_source_images_metadata(cache_file, source_images)
                except Exception as e:
                    print(f"Warning: Failed to embed source images: {e}")
            
            return cache_file
            
        except Exception as e:
            error = e
            print(f"error saving {save_file}: {e}")


def embed_source_images_metadata_mp4(video_path, source_images):
    """
    Embed source images as cover art in MP4 files using mutagen.
    
    Args:
        video_path (str): Path to the MP4 video file
        source_images (dict): Dictionary containing source images
            Expected keys: 'image_start', 'image_end', 'image_refs'
            Values should be PIL Images or file paths
    
    Returns:
        str: Path to the video file with embedded cover art
    """
    from mutagen.mp4 import MP4, MP4Cover, AtomDataType
    import io
    
    if not source_images:
        return video_path
    
    try:
        file = MP4(video_path)
        if file.tags is None:
            file.add_tags()
        
        # Convert source images to cover art
        cover_data = []
        
        # Process each source image type
        for img_type, img_data in source_images.items():
            if img_data is None:
                continue
                
            # Handle different image input types
            if isinstance(img_data, list):
                # Multiple images (e.g., image_refs)
                for i, img in enumerate(img_data):
                    if img is not None:
                        cover_bytes, image_format = _convert_image_to_bytes(img)
                        if cover_bytes:
                            cover_data.append(MP4Cover(cover_bytes, image_format))
            else:
                # Single image
                cover_bytes, image_format = _convert_image_to_bytes(img_data)
                if cover_bytes:
                    cover_data.append(MP4Cover(cover_bytes, image_format))
        
        if cover_data:
            file.tags['covr'] = cover_data
            file.save()
            print(f"Successfully embedded {len(cover_data)} cover images in {video_path}")
        
    except Exception as e:
        print(f"Failed to embed cover art with mutagen: {e}")
        print(f"This might be due to image format or MP4 file structure issues")
    
    return video_path


def _convert_image_to_bytes(img_data):
    """
    Convert PIL Image or file path to image bytes with proper format detection.
    
    Args:
        img_data: PIL Image object or file path string
        
    Returns:
        tuple: (bytes, AtomDataType) - image data and format, or (None, None) if conversion failed
    """
    from mutagen.mp4 import AtomDataType
    import io
    
    try:
        if hasattr(img_data, 'save'):  # PIL Image
            # For PIL Images, save as JPEG for better compatibility
            img_bytes = io.BytesIO()
            img_data.save(img_bytes, format='JPEG')
            return img_bytes.getvalue(), AtomDataType.JPEG
            
        elif isinstance(img_data, str) and os.path.exists(img_data):  # File path
            with open(img_data, 'rb') as f:
                # Detect format based on file extension
                if img_data.lower().endswith(('.jpg', '.jpeg')):
                    return f.read(), AtomDataType.JPEG
                elif img_data.lower().endswith('.png'):
                    return f.read(), AtomDataType.PNG
                else:
                    # Convert unknown formats to JPEG using PIL
                    img = Image.open(f)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='JPEG')
                    return img_bytes.getvalue(), AtomDataType.JPEG
                    
    except Exception as e:
        print(f"Failed to convert image to bytes: {e}")
        return None, None


def embed_source_images_metadata(video_path, source_images):
    """
    Embed source images as attachments in MKV video files using FFmpeg.
    For MP4 files, use embed_source_images_metadata_mp4 instead.
    
    Args:
        video_path (str): Path to the video file
        source_images (dict): Dictionary containing source images
            Expected keys: 'image_start', 'image_end', 'image_refs'
            Values should be PIL Images or file paths
    
    Returns:
        str: Path to the video file with embedded attachments
    """
    import tempfile
    import subprocess
    import os
    
    if not source_images:
        return video_path
    
    # Create temporary directory for image files
    temp_dir = tempfile.mkdtemp()
    try:
        attachment_files = []
        
        # Process each source image type
        for img_type, img_data in source_images.items():
            if img_data is None:
                continue
                
            # Handle different image input types
            if isinstance(img_data, list):
                # Multiple images (e.g., image_refs)
                for i, img in enumerate(img_data):
                    if img is not None:
                        img_path = _save_temp_image(img, temp_dir, f"{img_type}_{i}")
                        if img_path:
                            attachment_files.append((img_path, f"{img_type}_{i}.jpg"))
            else:
                # Single image
                img_path = _save_temp_image(img_data, temp_dir, img_type)
                if img_path:
                    attachment_files.append((img_path, f"{img_type}.jpg"))
        
        if not attachment_files:
            return video_path
        
        # Build FFmpeg command
        ffmpeg_cmd = ['ffmpeg', '-y', '-i', video_path]
        
        # Add attachment parameters
        for i, (file_path, filename) in enumerate(attachment_files):
            ffmpeg_cmd.extend(['-attach', file_path])
            ffmpeg_cmd.extend(['-metadata:s:t:' + str(i), f'mimetype=image/jpeg'])
            ffmpeg_cmd.extend(['-metadata:s:t:' + str(i), f'filename={filename}'])
        
        # Output parameters
        ffmpeg_cmd.extend(['-c', 'copy'])  # Copy streams without re-encoding
        
        # Create output file
        output_path = video_path.replace('.mkv', '_with_sources.mkv')
        ffmpeg_cmd.append(output_path)
        
        # Verify all attachment files exist before running FFmpeg
        for file_path, filename in attachment_files:
            if not os.path.exists(file_path):
                print(f"ERROR: Attachment file missing: {file_path}")
                return video_path
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            
            # Verify output file was created
            if not os.path.exists(output_path):
                print(f"ERROR: FFmpeg completed but output file {output_path} was not created")
                return video_path
            
            # Check output file size and streams before replacing
            output_size = os.path.getsize(output_path)
            
            # Verify the output file actually has attachments
            try:
                import subprocess as sp
                probe_result = sp.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_streams', output_path
                ], capture_output=True, text=True)
                
                if probe_result.returncode == 0:
                    import json
                    probe_data = json.loads(probe_result.stdout)
                    streams = probe_data.get('streams', [])
                    attachment_streams = [s for s in streams if s.get('disposition', {}).get('attached_pic') == 1]
                    
                    if len(attachment_streams) == 0:
                        print(f"WARNING: Output file has no attachment streams despite FFmpeg success!")
                    
            except Exception as probe_error:
                pass
            
            # Replace original file with the one containing attachments
            import shutil
            
            try:
                # Backup original file first
                backup_path = video_path + ".backup"
                shutil.copy2(video_path, backup_path)
                
                # Replace original with new file - use explicit error handling
                try:
                    shutil.move(output_path, video_path)
                except Exception as move_error:
                    print(f"ERROR: shutil.move() failed: {move_error}")
                    # Restore backup and return
                    if os.path.exists(backup_path):
                        shutil.move(backup_path, video_path)
                    return video_path
                
                # Verify replacement actually worked by checking file exists and size
                if not os.path.exists(video_path):
                    print(f"ERROR: File replacement failed - target file doesn't exist!")
                    # Restore backup
                    if os.path.exists(backup_path):
                        shutil.move(backup_path, video_path)
                    return video_path
                
                final_size = os.path.getsize(video_path)
                
                if final_size == output_size:
                    # Remove backup
                    os.remove(backup_path)
                else:
                    print(f"ERROR: File replacement failed - size mismatch! Expected {output_size}, got {final_size}")
                    # Restore backup
                    if os.path.exists(backup_path):
                        shutil.move(backup_path, video_path)
                    return video_path
                    
            except Exception as move_error:
                print(f"ERROR: File replacement failed: {move_error}")
                # Try to restore backup if it exists
                backup_path = video_path + ".backup"
                if os.path.exists(backup_path):
                    try:
                        shutil.move(backup_path, video_path)
                    except:
                        pass
                return video_path
            
            return video_path
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error embedding source images: {e.stderr}")
            # Clean up temp file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            return video_path
            
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def _save_temp_image(img_data, temp_dir, name):
    """
    Save image data to a temporary file.
    
    Args:
        img_data: PIL Image, file path, or tensor
        temp_dir: Temporary directory path
        name: Base name for the file
    
    Returns:
        str: Path to saved temporary file, or None if failed
    """
    import os
    from PIL import Image
    import torch
    
    try:
        temp_path = os.path.join(temp_dir, f"{name}.jpg")
        
        if isinstance(img_data, str):
            # File path - copy the file
            if os.path.exists(img_data):
                import shutil
                shutil.copy2(img_data, temp_path)
                return temp_path
        elif hasattr(img_data, 'save'):
            # PIL Image
            img_data.save(temp_path, 'JPEG', quality=95)
            return temp_path
        elif torch.is_tensor(img_data):
            # Tensor - convert to PIL and save
            if img_data.dim() == 4:
                img_data = img_data.squeeze(0)
            if img_data.dim() == 3:
                # Convert from tensor to PIL
                if img_data.shape[0] == 3:  # CHW format
                    img_data = img_data.permute(1, 2, 0)
                # Normalize to 0-255 range
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).clamp(0, 255)
                img_array = img_data.cpu().numpy().astype('uint8')
                img_pil = Image.fromarray(img_array)
                img_pil.save(temp_path, 'JPEG', quality=95)
                return temp_path
        
        return None
        
    except Exception as e:
        print(f"ERROR: Exception in _save_temp_image for {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_mp4_cover_art(video_path, output_dir):
    """
    Extract cover art from MP4 files using mutagen.
    
    Args:
        video_path (str): Path to the MP4 file
        output_dir (str): Directory to save extracted images
    
    Returns:
        list: List of extracted image file paths
    """
    try:
        from mutagen.mp4 import MP4, AtomDataType
        import os
        
        print(f"DEBUG: Extracting cover art from MP4: {video_path}")
        
        file = MP4(video_path)
        
        if file.tags is None:
            print("DEBUG: No tags found in MP4 file")
            return []
            
        if 'covr' not in file.tags:
            print("DEBUG: No cover art found in MP4 tags")
            print(f"DEBUG: Available tags: {list(file.tags.keys())}")
            return []
        
        cover_art = file.tags['covr']
        print(f"DEBUG: Found {len(cover_art)} cover art images")
        extracted_files = []
        
        # Map cover art to semantic names based on order
        # Since MP4 doesn't preserve semantic meaning, we'll use a heuristic:
        # - First image: image_start
        # - Second image: image_end  
        # - Additional images: image_refs
        semantic_names = ['image_start', 'image_end'] + [f'image_refs_{i}' for i in range(10)]
        
        for i, cover in enumerate(cover_art):
            # Determine file extension based on format
            if cover.imageformat == AtomDataType.JPEG:
                ext = '.jpg'
            elif cover.imageformat == AtomDataType.PNG:
                ext = '.png'
            else:
                ext = '.jpg'  # Default to JPEG
            
            # Create semantic filename for GUI recognition
            if i < len(semantic_names):
                filename = f"{semantic_names[i]}{ext}"
            else:
                filename = f"cover_art_{i+1}{ext}"
            
            output_file = os.path.join(output_dir, filename)
            
            print(f"DEBUG: Writing cover art {i+1} to: {filename} ({len(cover):,} bytes)")
            
            # Write cover art to file
            with open(output_file, 'wb') as f:
                f.write(cover)
            
            if os.path.exists(output_file):
                extracted_files.append(output_file)
                print(f"DEBUG: Successfully extracted: {filename}")
            else:
                print(f"DEBUG: Failed to create file: {filename}")
        
        print(f"DEBUG: Total extracted files: {len(extracted_files)}")
        return extracted_files
        
    except Exception as e:
        print(f"DEBUG: Error extracting cover art from MP4: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_source_images(video_path, output_dir=None):
    """
    Extract embedded source images from video files.
    Supports MKV (attachments via ffmpeg) and MP4 (cover art via mutagen).
    
    Args:
        video_path (str): Path to the video file (MKV or MP4)
        output_dir (str): Directory to save extracted images (optional)
    
    Returns:
        list: List of extracted image file paths
    """
    import os
    
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle MP4 files with mutagen
    if video_path.lower().endswith('.mp4'):
        print(f"DEBUG: Detected MP4 file, using mutagen extraction")
        return _extract_mp4_cover_art(video_path, output_dir)
    
    # Handle MKV files with ffmpeg (existing logic)
    try:
        # First, probe the video to find attachment streams (attached pics)
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        import json as json_module
        probe_data = json_module.loads(result.stdout)
        
        # Find attachment streams (attached pics)
        attachment_streams = []
        for i, stream in enumerate(probe_data.get('streams', [])):
            # Check for attachment streams in multiple ways:
            # 1. Traditional attached_pic flag
            # 2. Video streams with image-like metadata (filename, mimetype)
            # 3. MJPEG codec which is commonly used for embedded images
            is_attached_pic = stream.get('disposition', {}).get('attached_pic', 0) == 1
            
            # Check for image metadata in video streams (our case after metadata embedding)
            tags = stream.get('tags', {})
            has_image_metadata = (
                'FILENAME' in tags and tags['FILENAME'].lower().endswith(('.jpg', '.jpeg', '.png')) or
                'filename' in tags and tags['filename'].lower().endswith(('.jpg', '.jpeg', '.png')) or
                'MIMETYPE' in tags and tags['MIMETYPE'].startswith('image/') or
                'mimetype' in tags and tags['mimetype'].startswith('image/')
            )
            
            # Check for MJPEG codec (common for embedded images)
            is_mjpeg = stream.get('codec_name') == 'mjpeg'
            
            if (stream.get('codec_type') == 'video' and 
                (is_attached_pic or (has_image_metadata and is_mjpeg))):
                attachment_streams.append(i)
        
        if not attachment_streams:
            return []
        
        # Extract each attachment stream
        extracted_files = []
        used_filenames = set()  # Track filenames to avoid collisions
        
        for stream_idx in attachment_streams:
            # Get original filename from metadata if available
            stream_info = probe_data['streams'][stream_idx]
            tags = stream_info.get('tags', {})
            original_filename = (
                tags.get('filename') or 
                tags.get('FILENAME') or 
                f'attachment_{stream_idx}.png'
            )
            
            # Clean filename for filesystem
            safe_filename = os.path.basename(original_filename)
            if not safe_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                safe_filename += '.png'
            
            # Handle filename collisions
            base_name, ext = os.path.splitext(safe_filename)
            counter = 0
            final_filename = safe_filename
            while final_filename in used_filenames:
                counter += 1
                final_filename = f"{base_name}_{counter}{ext}"
            used_filenames.add(final_filename)
            
            output_file = os.path.join(output_dir, final_filename)
            
            # Extract the attachment stream
            extract_cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-map', f'0:{stream_idx}', '-frames:v', '1',
                output_file
            ]
            
            try:
                subprocess.run(extract_cmd, capture_output=True, text=True, check=True)
                if os.path.exists(output_file):
                    extracted_files.append(output_file)
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract attachment {stream_idx} from {os.path.basename(video_path)}: {e.stderr}")
        
        return extracted_files
            
    except subprocess.CalledProcessError as e:
        print(f"Error extracting source images from {os.path.basename(video_path)}: {e.stderr}")
        return []


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
    
    # Get format and quality settings
    format_info = _get_format_info(quality)
    
    # Rename file extension to match requested format
    save_file = osp.splitext(save_file)[0] + format_info['ext']
    
    # Save image
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            
            if format_info['use_pil']:
                # Use PIL for WebP and advanced options
                grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
                # Convert to PIL Image
                grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                img = Image.fromarray(grid)
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