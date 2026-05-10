"""Minimal MPS inference test for Wan2GP - loads the 1.3B model and runs one forward pass."""
import os, sys, gc, time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("=" * 60)
print("Wan2GP MPS Inference Test")
print("=" * 60)

# Step 1: Apply MPS patch early
import torch
from shared.mps.device_patch import apply_mps_patch
apply_mps_patch()

print(f"\n[1] PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}")
print(f"    Default device: {torch.get_default_device()}")

# Step 2: Try to load the wan handler
print("\n[2] Loading wan handler...")
try:
    from models.wan.wan_handler import family_handler as WanHandler
    from models.wan.configs import WAN_CONFIGS
    print("    family_handler imported OK")
except ImportError:
    print("    family_handler not available, trying wan_handler...")
    try:
        from models.wan.wan_handler import family_handler
        print("    family_handler imported OK")
    except ImportError as e:
        print(f"    SKIPPED: {e} (may need model weights installed)")
        sys.exit(0)

# Step 3: Check available model files
print("\n[3] Checking model files...")
model_dir = os.path.join(REPO_ROOT, "ckpts")
if not os.path.exists(model_dir):
    print(f"    Model dir {model_dir} does not exist!")
    print("    Need to download Wan2.1 1.3B model weights first.")
    # List what's available
    if os.path.exists(REPO_ROOT):
        for root, dirs, files in os.walk(REPO_ROOT, topdown=True):
            # Only go 2 levels deep
            level = root.replace(REPO_ROOT, '').count(os.sep)
            if level > 2:
                dirs.clear()
                continue
            for f in files:
                if f.endswith(('.safetensors', '.pt', '.bin', '.pth')):
                    fp = os.path.join(root, f)
                    size = os.path.getsize(fp) / (1024**3)
                    print(f"    {fp}: {size:.2f}GB")
else:
    for root, dirs, files in os.walk(model_dir, topdown=True):
        level = root.replace(model_dir, '').count(os.sep)
        if level > 2:
            dirs.clear()
            continue
        for f in files:
            if f.endswith(('.safetensors', '.pt', '.bin', '.pth')):
                fp = os.path.join(root, f)
                size = os.path.getsize(fp) / (1024**3)
                print(f"    {fp}: {size:.2f}GB")

print("\n[4] Done. Check if any model weights were found above.")
print("    If no weights found, you need to download Wan2.1 1.3B first.")
