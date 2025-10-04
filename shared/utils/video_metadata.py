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


def save_metadata_to_mp4(file_path, metadata_dict):
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


def save_metadata_to_video(file_path, metadata_dict):
    """
    Save JSON metadata to video file (auto-detects MP4 vs MKV).
    
    Args:
        file_path (str): Path to video file
        metadata_dict (dict): Metadata dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """
    if file_path.endswith('.mp4'):
        return save_metadata_to_mp4(file_path, metadata_dict)
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

