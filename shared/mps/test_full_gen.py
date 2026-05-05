#!/usr/bin/env python3
"""Test the full Wan2GP generation path on MPS."""
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

print("=== Step 1: Import torch ===")
import torch
print(f"torch version: {torch.__version__}")

print("\n=== Step 2: Apply MPS patch ===")
from shared.mps.device_patch import apply_mps_patch
apply_mps_patch()

print("\n=== Step 3: Verify patches ===")
print(f"torch.compile: {torch.compile}")
print(f"torch.autocast: {torch.autocast}")
print(f"torch.cuda.amp.autocast: {torch.cuda.amp.autocast}")

print("\n=== Step 4: Import mmgp ===")
from mmgp import offload, safetensors2, profile_type, quant_router
print("mmgp imported OK")

print("\n=== Step 5: Test torch.autocast('cuda') ===")
try:
    with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        print("  torch.autocast('cuda') OK")
except Exception as e:
    print(f"  torch.autocast('cuda') FAILED: {type(e).__name__}: {e}")

print("\n=== Step 6: Test torch.amp.autocast('cuda') ===")
try:
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        print("  torch.amp.autocast('cuda') OK")
except Exception as e:
    print(f"  torch.amp.autocast('cuda') FAILED: {type(e).__name__}: {e}")

print("\n=== Step 7: Test torch.amp.autocast(device_type='cuda') ===")
try:
    with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
        print("  torch.amp.autocast(device_type='cuda') OK")
except Exception as e:
    print(f"  torch.amp.autocast(device_type='cuda') FAILED: {type(e).__name__}: {e}")

print("\n=== Step 8: Import wan model ===")
try:
    from models.wan.ovi_fusion_engine import OviFusionEngine
    print("OviFusionEngine imported OK")
except Exception as e:
    print(f"OviFusionEngine import FAILED: {type(e).__name__}: {e}")

print("\n=== Step 9: Import wgp module ===")
try:
    # This triggers the full wgp import chain
    import wgp
    print(f"wgp imported OK")
    print(f"wgp.compile = '{getattr(wgp, 'compile', 'NOT SET')}'")
except Exception as e:
    import traceback
    print(f"wgp import FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== All tests done ===")
