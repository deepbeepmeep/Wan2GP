"""
Source Image Embedding and Extraction Utilities

This module provides functionality to embed and extract source images from video files.
- MP4: Uses mutagen to embed images as cover art
- MKV: Uses FFmpeg to embed images as attachments
"""

import os
import subprocess
import tempfile
import shutil
from PIL import Image
import torch


def prepare_source_images_dict(config, image_start=None, image_end=None, image_refs=None):
    """
    Prepare source images dictionary for embedding based on configuration.
    
    Args:
        config (dict): Server configuration dictionary
        image_start: Start image (PIL Image or path)
        image_end: End image (PIL Image or path)
        image_refs: Reference images (list or single image)
    
    Returns:
        dict or None: Dictionary of source images to embed, or None if feature disabled
    """
    # Check if embedding is enabled
    if not config.get("embed_source_images", False):
        return None
    
    # Check if container format supports embedding
    container = config.get("video_container", "mp4")
    if container not in ["mkv", "mp4"]:
        return None
    
    # Collect available source images
    source_images = {}
    
    if image_start is not None:
        source_images["image_start"] = image_start
    
    if image_end is not None:
        source_images["image_end"] = image_end
    
    if image_refs is not None:
        source_images["image_refs"] = image_refs
    
    # Return None if no images to embed
    return source_images if source_images else None


def process_extracted_images(extracted_files):
    """
    Process extracted image files and map them to their semantic types.
    
    Args:
        extracted_files (list): List of extracted image file paths
    
    Returns:
        dict: Dictionary mapping image types to PIL Images
              Keys: 'image_start', 'image_end', 'image_refs'
              Values: PIL Images or list of PIL Images
    """
    result = {
        'image_start': None,
        'image_end': None,
        'image_refs': []
    }
    
    for img_path in extracted_files:
        img_name = os.path.basename(img_path).lower()
        
        try:
            # Load the image
            pil_image = Image.open(img_path)
            
            # Map based on filename
            if 'image_start' in img_name:
                result['image_start'] = pil_image
            elif 'image_end' in img_name:
                result['image_end'] = pil_image
            elif 'image_refs' in img_name:
                result['image_refs'].append(pil_image)
                
        except Exception as e:
            print(f"Error loading extracted image {img_path}: {e}")
            continue
    
    return result


def apply_images_to_settings(current_settings, processed_images):
    """
    Apply processed images to settings dictionary.
    Pure business logic function with no UI dependencies.
    
    Args:
        current_settings (dict): Current settings dictionary to update
        processed_images (dict): Processed images from process_extracted_images()
    
    Returns:
        tuple: (updated_settings, applied_count)
    """
    applied_count = 0
    
    # Apply image_start
    if processed_images['image_start'] is not None:
        current_settings['image_start'] = [processed_images['image_start']]
        applied_count += 1
    
    # Apply image_end
    if processed_images['image_end'] is not None:
        current_settings['image_end'] = [processed_images['image_end']]
        applied_count += 1
    
    # Apply image_refs
    if processed_images['image_refs']:
        existing_refs = current_settings.get('image_refs', [])
        if not isinstance(existing_refs, list):
            existing_refs = []
        existing_refs.extend(processed_images['image_refs'])
        current_settings['image_refs'] = existing_refs
        applied_count += len(processed_images['image_refs'])
    
    return current_settings, applied_count


def extract_and_apply_to_settings(file_path, current_settings):
    """
    Complete orchestration: Extract images from video and apply to settings.
    ALL application logic is here - UI layer just needs to get/set state.
    
    Args:
        file_path (str): Path to the video file (MKV or MP4)
        current_settings (dict): Current settings dictionary
    
    Returns:
        tuple: (updated_settings, applied_count)
               Returns (current_settings, 0) if nothing extracted
    """
    # Validate file type
    if not (file_path.endswith('.mkv') or file_path.endswith('.mp4')):
        return current_settings, 0
    
    try:
        # Extract images from video
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_files = extract_source_images(file_path, temp_dir)
            
            if not extracted_files:
                return current_settings, 0
            
            # Process extracted files to image types
            processed_images = process_extracted_images(extracted_files)
            
            # Apply to settings
            updated_settings, applied_count = apply_images_to_settings(current_settings, processed_images)
            
            return updated_settings, applied_count
            
    except Exception as e:
        print(f"Error extracting source images from {file_path}: {e}")
        return current_settings, 0


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
    try:
        temp_path = os.path.join(temp_dir, f"{name}.jpg")
        
        if isinstance(img_data, str):
            # File path - copy the file
            if os.path.exists(img_data):
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
        
        file = MP4(video_path)
        
        if file.tags is None:
            return []
            
        if 'covr' not in file.tags:
            return []
        
        cover_art = file.tags['covr']
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
            
            # Write cover art to file
            with open(output_file, 'wb') as f:
                f.write(cover)
            
            if os.path.exists(output_file):
                extracted_files.append(output_file)
        
        return extracted_files
        
    except Exception as e:
        print(f"Error extracting cover art from MP4: {e}")
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
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle MP4 files with mutagen
    if video_path.lower().endswith('.mp4'):
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


