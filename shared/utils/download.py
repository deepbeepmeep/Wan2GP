import sys, time

# Global variables to track download progress
_start_time = None
_last_time = None
_last_downloaded = 0
_speed_history = []
_update_interval = 0.5  # Update speed every 0.5 seconds

def progress_hook(block_num, block_size, total_size, filename=None):
    """
    Simple progress bar hook for urlretrieve
    
    Args:
        block_num: Number of blocks downloaded so far
        block_size: Size of each block in bytes
        total_size: Total size of the file in bytes
        filename: Name of the file being downloaded (optional)
    """
    global _start_time, _last_time, _last_downloaded, _speed_history, _update_interval
    
    current_time = time.time()
    downloaded = block_num * block_size
    
    # Initialize timing on first call
    if _start_time is None or block_num == 0:
        _start_time = current_time
        _last_time = current_time
        _last_downloaded = 0
        _speed_history = []
    
    # Calculate download speed only at specified intervals
    speed = 0
    if current_time - _last_time >= _update_interval:
        if _last_time > 0:
            current_speed = (downloaded - _last_downloaded) / (current_time - _last_time)
            _speed_history.append(current_speed)
            # Keep only last 5 speed measurements for smoothing
            if len(_speed_history) > 5:
                _speed_history.pop(0)
            # Average the recent speeds for smoother display
            speed = sum(_speed_history) / len(_speed_history)
        
        _last_time = current_time
        _last_downloaded = downloaded
    elif _speed_history:
        # Use the last calculated average speed
        speed = sum(_speed_history) / len(_speed_history)
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    file_display = filename if filename else "Unknown file"
    
    if total_size <= 0:
        # If total size is unknown, show downloaded bytes
        speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
        line = f"\r{file_display}: {format_bytes(downloaded)}{speed_str}"
        # Clear any trailing characters by padding with spaces
        sys.stdout.write(line.ljust(80))
        sys.stdout.flush()
        return
    
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    
    # Create progress bar (40 characters wide to leave room for other info)
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
    
    # Display progress with filename first
    line = f"\r{file_display}: [{bar}] {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)}){speed_str}"
    # Clear any trailing characters by padding with spaces
    sys.stdout.write(line.ljust(100))
    sys.stdout.flush()
    
    # Print newline when complete
    if percent >= 100:
        print()

# Wrapper function to include filename in progress hook
def create_progress_hook(filename):
    """Creates a progress hook with the filename included"""
    global _start_time, _last_time, _last_downloaded, _speed_history
    # Reset timing variables for new download
    _start_time = None
    _last_time = None
    _last_downloaded = 0
    _speed_history = []
    
    def hook(block_num, block_size, total_size):
        return progress_hook(block_num, block_size, total_size, filename)
    return hook


