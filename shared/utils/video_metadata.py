"""
Video Metadata Utilities

This module provides functionality to read and write JSON metadata to video files.
- MP4: Uses mutagen to store metadata in ©cmt tag
- MKV: Uses FFmpeg to store metadata in comment/description tags
"""

import json
import subprocess
import os
import shutil
import tempfile

def _convert_image_to_bytes(img):
    """
    Convert various image formats to bytes suitable for MP4 cover art.
    
    Args:
        img: Can be:
            - PIL Image object
            - File path (str)
            - bytes
    
    Returns:
        tuple: (image_bytes, image_format)
            - image_bytes: Binary image data
            - image_format: AtomDataType constant (JPEG or PNG)
    """
    from mutagen.mp4 import AtomDataType
    from PIL import Image
    import io
    import os
    
    try:
        # If it's already bytes, detect format and return
        if isinstance(img, bytes):
            # Detect format from magic numbers
            if img.startswith(b'\x89PNG'):
                return img, AtomDataType.PNG
            else:
                return img, AtomDataType.JPEG
        
        # If it's a file path, read and convert
        if isinstance(img, str):
            if not os.path.exists(img):
                print(f"Warning: Image file not found: {img}")
                return None, None
            
            # Determine format from extension
            ext = os.path.splitext(img)[1].lower()
            
            # Open with PIL for conversion
            pil_img = Image.open(img)
            
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if pil_img.mode not in ('RGB', 'L'):
                if pil_img.mode == 'RGBA':
                    # Create white background for transparency
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[3])
                    pil_img = background
                else:
                    pil_img = pil_img.convert('RGB')
            
            # Save to bytes
            img_bytes = io.BytesIO()
            
            # Use PNG for lossless formats, JPEG for others
            if ext in ['.png', '.bmp', '.tiff', '.tif']:
                pil_img.save(img_bytes, format='PNG')
                img_format = AtomDataType.PNG
            else:
                pil_img.save(img_bytes, format='JPEG', quality=95)
                img_format = AtomDataType.JPEG
            
            return img_bytes.getvalue(), img_format
        
        # If it's a PIL Image
        if isinstance(img, Image.Image):
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Save to bytes (prefer PNG for quality)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue(), AtomDataType.PNG
        
        print(f"Warning: Unsupported image type: {type(img)}")
        return None, None
        
    except Exception as e:
        print(f"Error converting image to bytes: {e}")
        return None, None

def embed_source_images_metadata_mp4(file, source_images):
    from mutagen.mp4 import MP4, MP4Cover, AtomDataType
    import json
    import os
    
    if not source_images:
        return file
    
    try:
        
        # Convert source images to cover art and build metadata
        cover_data = []
        image_metadata = {}  # Maps tag to list of {index, filename, extension}
        
        # Process each source image type
        for img_tag, img_data in source_images.items():
            if img_data is None:
                continue
            
            tag_images = []
            
            # Normalize to list for uniform processing
            img_list = img_data if isinstance(img_data, list) else [img_data]
            
            for img in img_list:
                if img is not None:
                    cover_bytes, image_format = _convert_image_to_bytes(img)
                    if cover_bytes:
                        # Extract filename and extension
                        if isinstance(img, str) and os.path.exists(img):
                            filename = os.path.basename(img)
                            extension = os.path.splitext(filename)[1]
                        else:
                            # PIL Image or unknown - infer from format
                            extension = '.png' if image_format == AtomDataType.PNG else '.jpg'
                            filename = f"{img_tag}{extension}"
                        
                        tag_images.append({
                            'index': len(cover_data),
                            'filename': filename,
                            'extension': extension
                        })
                        cover_data.append(MP4Cover(cover_bytes, image_format))
            
            if tag_images:
                image_metadata[img_tag] = tag_images
        
        if cover_data:
            file.tags['----:com.apple.iTunes:EMBEDDED_IMAGES'] = cover_data
            # Store the complete metadata as JSON
            file.tags['----:com.apple.iTunes:IMAGE_METADATA'] = json.dumps(image_metadata).encode('utf-8')
            # print(f"Successfully embedded {len(cover_data)} cover images")
            # print(f"Image tags: {list(image_metadata.keys())}")
        
    except Exception as e:
        print(f"Failed to embed cover art with mutagen: {e}")
        print(f"This might be due to image format or MP4 file structure issues")
    
    return file


def save_metadata_to_mp4(file_path, metadata_dict, source_images = None):
    """
    Save JSON metadata to MP4 file using mutagen.
    
    Args:
        file_path (str): Path to MP4 file
        metadata_dict (dict): Metadata dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from mutagen.mp4 import MP4
        file = MP4(file_path)
        file.tags['©cmt'] = [json.dumps(metadata_dict)]
        if source_images is not None:
            embed_source_images_metadata_mp4(file, source_images)
        file.save()
        return True
    except Exception as e:
        print(f"Error saving metadata to MP4 {file_path}: {e}")
        return False


def save_metadata_to_mkv(file_path, metadata_dict):
    """
    Save JSON metadata to MKV file using FFmpeg.
    
    Args:
        file_path (str): Path to MKV file
        metadata_dict (dict): Metadata dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create temporary file with metadata
        temp_path = file_path.replace('.mkv', '_temp_with_metadata.mkv')
        
        # Use FFmpeg to add metadata while preserving ALL streams (including attachments)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-metadata', f'comment={json.dumps(metadata_dict)}',
            '-map', '0',  # Map all streams from input (including attachments)
            '-c', 'copy',  # Copy streams without re-encoding
            temp_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Replace original with metadata version
            shutil.move(temp_path, file_path)
            return True
        else:
            print(f"Warning: Failed to add metadata to MKV file: {result.stderr}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
                
    except Exception as e:
        print(f"Error saving metadata to MKV {file_path}: {e}")
        return False



def save_video_metadata(file_path, metadata_dict, source_images=  None):
    """
    Save JSON metadata to video file (auto-detects MP4 vs MKV).
    
    Args:
        file_path (str): Path to video file
        metadata_dict (dict): Metadata dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """

    if file_path.endswith('.mp4'):
        return save_metadata_to_mp4(file_path, metadata_dict, source_images)
    elif file_path.endswith('.mkv'):
        return save_metadata_to_mkv(file_path, metadata_dict)
    else:
        return False


def read_metadata_from_mp4(file_path):
    """
    Read JSON metadata from MP4 file using mutagen.
    
    Args:
        file_path (str): Path to MP4 file
    
    Returns:
        dict or None: Metadata dictionary if found, None otherwise
    """
    try:
        from mutagen.mp4 import MP4
        file = MP4(file_path)
        tags = file.tags['©cmt'][0]
        return json.loads(tags)
    except Exception:
        return None


def read_metadata_from_mkv(file_path):
    """
    Read JSON metadata from MKV file using ffprobe.
    
    Args:
        file_path (str): Path to MKV file
    
    Returns:
        dict or None: Metadata dictionary if found, None otherwise
    """
    try:
        # Try to get metadata using ffprobe
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', file_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            format_tags = probe_data.get('format', {}).get('tags', {})
            
            # Look for our metadata in various possible tag locations
            for tag_key in ['comment', 'COMMENT', 'description', 'DESCRIPTION']:
                if tag_key in format_tags:
                    try:
                        return json.loads(format_tags[tag_key])
                    except:
                        continue
        return None
    except Exception:
        return None


def read_metadata_from_video(file_path):
    """
    Read JSON metadata from video file (auto-detects MP4 vs MKV).
    
    Args:
        file_path (str): Path to video file
    
    Returns:
        dict or None: Metadata dictionary if found, None otherwise
    """
    if file_path.endswith('.mp4'):
        return read_metadata_from_mp4(file_path)
    elif file_path.endswith('.mkv'):
        return read_metadata_from_mkv(file_path)
    else:
        return None

def _extract_mp4_cover_art(video_path, output_dir = None):
    """
    Extract cover art from MP4 files using mutagen with proper tag association.
    
    Args:
        video_path (str): Path to the MP4 file
        output_dir (str): Directory to save extracted images
    
    Returns:
        dict: Dictionary mapping tags to lists of extracted image file paths
              Format: {tag_name: [path1, path2, ...], ...}
    """
    try:
        from mutagen.mp4 import MP4
        import json
        
        file = MP4(video_path)
        
        if file.tags is None or '----:com.apple.iTunes:EMBEDDED_IMAGES' not in file.tags:
            return {}
        
        cover_art =  file.tags['----:com.apple.iTunes:EMBEDDED_IMAGES']
        
        # Retrieve the image metadata
        metadata_data = file.tags.get('----:com.apple.iTunes:IMAGE_METADATA')
        
        if metadata_data:
            # Deserialize metadata and extract with original filenames
            image_metadata = json.loads(metadata_data[0].decode('utf-8'))
            extracted_files = {}
            
            for tag, tag_images in image_metadata.items():
                extracted_files[tag] = []
                
                for img_info in tag_images:
                    cover_idx = img_info['index']
                    
                    if cover_idx >= len(cover_art):
                        continue
                    if output_dir is None: output_dir = _create_temp_dir()
                    os.makedirs(output_dir, exist_ok=True)

                    cover = cover_art[cover_idx]
                    
                    # Use original filename
                    filename = img_info['filename']
                    output_file = os.path.join(output_dir, filename)
                    
                    # Handle duplicate filenames by adding suffix
                    if os.path.exists(output_file):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(output_file):
                            filename = f"{base}_{counter}{ext}"
                            output_file = os.path.join(output_dir, filename)
                            counter += 1


                    # Write cover art to file
                    with open(output_file, 'wb') as f:
                        f.write(cover)
                    
                    if os.path.exists(output_file):
                        extracted_files[tag].append(output_file)
            
            return extracted_files
        
        else:
            # Fallback: Extract all images with generic naming
            print(f"Warning: No IMAGE_METADATA found in {video_path}, using generic extraction")
            extracted_files = {'unknown': []}
            
            for i, cover in enumerate(cover_art):
                if output_dir is None: output_dir = _create_temp_dir()
                os.makedirs(output_dir, exist_ok=True)

                filename = f"cover_art_{i}.jpg"
                output_file = os.path.join(output_dir, filename)
                
                with open(output_file, 'wb') as f:
                    f.write(cover)
                
                if os.path.exists(output_file):
                    extracted_files['unknown'].append(output_file)
            
            return extracted_files
        
    except Exception as e:
        print(f"Error extracting cover art from MP4: {e}")
        return {}

def _create_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def extract_source_images(video_path, output_dir = None):
    
    # Handle MP4 files with mutagen
    if video_path.lower().endswith('.mp4'):
        return _extract_mp4_cover_art(video_path, output_dir)
    if output_dir is None:
        output_dir = _create_temp_dir()

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

