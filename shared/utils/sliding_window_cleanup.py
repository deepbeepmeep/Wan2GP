"""
Sliding Window Video Cleanup Utility

This module provides functionality to automatically delete shorter intermediate videos
during sliding window generation to save disk space.
"""

import os


def cleanup_previous_video(previous_video_path, file_list, file_settings_list, lock):
    """
    Delete a previous video file and remove it from UI file lists.
    
    Args:
        previous_video_path: Path to the video file to delete
        file_list: List of file paths in UI preview
        file_settings_list: List of file settings in UI preview
        lock: Threading lock for safe list manipulation
        
    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    if previous_video_path is None:
        return False
        
    try:
        if os.path.isfile(previous_video_path):
            os.remove(previous_video_path)
            
            # Also remove from UI file list to prevent dead links in preview
            with lock:
                if previous_video_path in file_list:
                    index = file_list.index(previous_video_path)
                    file_list.pop(index)
                    if index < len(file_settings_list):
                        file_settings_list.pop(index)
            
            return True
    except Exception as e:
        # Silently handle cleanup errors to not interrupt generation
        pass
    
    return False


def should_cleanup_video(sliding_window, sliding_window_keep_only_longest, is_image=False):
    """
    Determine if video cleanup should be performed.
    
    Args:
        sliding_window: Whether sliding window is enabled
        sliding_window_keep_only_longest: Whether cleanup is enabled
        is_image: Whether the output is an image (cleanup not needed)
        
    Returns:
        bool: True if cleanup should be performed
    """
    return sliding_window and sliding_window_keep_only_longest and not is_image


def get_cleanup_status_text(server_config):
    """
    Get the cleanup status text for info messages.
    
    Args:
        server_config: Server configuration dictionary
        
    Returns:
        str: "Enabled" or "Disabled"
    """
    return "Enabled" if server_config.get("sliding_window_keep_only_longest", False) else "Disabled"

