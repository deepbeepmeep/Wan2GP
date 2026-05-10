#!/usr/bin/env python3
"""Test the full generation path for Wan2GP on MPS."""
import os
import sys
import traceback
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# Track what gets called
class CallTracker:
    def __init__(self):
        self.calls = []

    def add(self, msg):
        self.calls.append(msg)
        print(f"[TRACKER] {msg}")

tracker = CallTracker()

# Import torch FIRST
tracker.add("Importing torch")
import torch

tracker.add(f"torch version: {torch.__version__}")
tracker.add(f"cuda compiled: {torch.cuda.is_compiled() if hasattr(torch.cuda, 'is_compiled') else 'N/A'}")
tracker.add(f"mps available: {torch.backends.mps.is_available()}")

# Now import and apply device_patch
tracker.add("Applying MPS patch")
from shared.mps.device_patch import apply_mps_patch
apply_mps_patch()

tracker.add(f"torch.compile patched: {torch.compile.__name__ == '_patched_compile'}")

# Now import the model module that would be used
tracker.add("Importing wan model module")
from models.wan.ovi_fusion_engine import OviFusionEngine
tracker.add("OviFusionEngine imported successfully")

# Test torch.autocast('mps')
tracker.add("Testing torch.autocast('mps')")
try:
    with torch.autocast('mps', enabled=True, dtype=torch.bfloat16):
        x = torch.randn(2, device='mps')
    tracker.add("torch.autocast('mps') OK")
except Exception as e:
    tracker.add(f"torch.autocast('mps') FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test the generate function's autocast line
tracker.add("Testing autocast('cuda') in model code")
try:
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        tracker.add("autocast('cuda') succeeded (with warning)")
except Exception as e:
    tracker.add(f"autocast('cuda') FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test torch.compile with inductor backend (this was the original issue)
tracker.add("Testing torch.compile(inductor)")
def test_fn(x):
    return x + 1
try:
    compiled = torch.compile(test_fn, backend='inductor')
    result = compiled(torch.tensor([1.0]))
    tracker.add(f"torch.compile(inductor) OK: {result}")
except Exception as e:
    tracker.add(f"torch.compile(inductor) FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test torch.cuda.amp.autocast (used in models)
tracker.add("Testing torch.cuda.amp.autocast")
try:
    import torch.cuda.amp as amp
    with amp.autocast(enabled=False):
        pass
    tracker.add("torch.cuda.amp.autocast OK")
except Exception as e:
    tracker.add(f"torch.cuda.amp.autocast FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

tracker.add("All tests completed")
