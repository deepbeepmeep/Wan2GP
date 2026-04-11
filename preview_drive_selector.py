"""Quick preview of the drive selector UX. Run: python preview_drive_selector.py"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Wan2GP"))

from shared.utils.drive_detector import detect_drives, is_single_drive, suggest_checkpoint_path, DriveInfo
from shared.utils.storage_selector import select_storage_drive

print("=== Detected drives ===")
drives = detect_drives()
for d in drives:
    free_gb = d.free_bytes / (1024**3)
    total_gb = d.total_bytes / (1024**3)
    print(f"  {d.mount_point}  ({d.label})  {free_gb:.1f} GB free / {total_gb:.1f} GB total")

print(f"\nSingle drive: {is_single_drive(drives)}")

if is_single_drive(drives):
    print("\nOnly one drive — selector would be skipped. Adding a fake drive for preview...\n")
    drives.append(DriveInfo("/fake/E:", "Fake External", 500 * 1024**3, 450 * 1024**3))

print("=== Launching selector ===\n")
chosen = select_storage_drive(drives)

if chosen is None:
    print("User cancelled — would use defaults: ['ckpts', '.']")
else:
    path = suggest_checkpoint_path(chosen, os.getcwd())
    print(f"Selected: {chosen}")
    print(f"Checkpoint path would be: {path}")
    print(f"Config would save: {[path, 'ckpts', '.']}")
