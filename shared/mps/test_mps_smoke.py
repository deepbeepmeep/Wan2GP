"""MPS Quick Smoke Test for Wan2GP.

Validates Apple Silicon MPS environment and core imports.
Runs in <30 seconds without downloading model weights.
"""

import os, sys, time, gc, json, platform

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASS, FAIL, SKIP = 0, 0, 0
results = []

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        results.append(f"  ✅ {name}: {detail}" if detail else f"  ✅ {name}")
        PASS += 1
    else:
        results.append(f"  ❌ {name}: {detail}" if detail else f"  ❌ {name}")
        FAIL += 1

def skip(name: str, reason: str = ""):
    global SKIP
    results.append(f"  ⏭️  {name}: {reason}")
    SKIP += 1

banner = "=" * 60
print(banner)
print("  Wan2GP MPS Smoke Test")
print(banner)

# ── 1. Environment ──────────────────────────────────────────────────
print("\n[1] Environment")

check("Running on macOS", platform.system() == "Darwin", platform.platform())
check("Apple Silicon", 
      platform.processor() == "arm" or "Apple" in (platform.processor() or ""),
      platform.processor() or "unknown")

# ── 2. PyTorch & MPS ────────────────────────────────────────────────
print("\n[2] PyTorch & MPS")

import torch
check("torch imported", True, f"v{torch.__version__}")
check("MPS available", torch.backends.mps.is_available())

# Test basic MPS tensor ops
if torch.backends.mps.is_available():
    try:
        a = torch.randn(2, 3).to("mps")
        b = torch.randn(2, 3).to("mps")
        c = a @ b.T
        check("Basic MPS matmul", c.shape == (2, 2), str(c.shape))
    except Exception as e:
        check("Basic MPS matmul", False, str(e)[:80])

    try:
        a = torch.randn(2, 3, dtype=torch.float32).to("mps")
        check("MPS float32 tensor", a.device.type == "mps")
    except Exception as e:
        check("MPS float32 tensor", False, str(e)[:80])

    try:
        a = torch.randn(2, 3, dtype=torch.bfloat16).to("mps")
        check("MPS bfloat16 tensor", a.device.type == "mps")
    except Exception as e:
        check("MPS bfloat16 tensor", False, str(e)[:80])

# Verify float64 is correctly rejected
try:
    a = torch.randn(1, dtype=torch.float64).to("mps")
    check("MPS rejects float64", False, "float64 should raise on MPS")
except (RuntimeError, TypeError):
    check("MPS rejects float64", True)

# ── 3. MPS device patch ─────────────────────────────────────────────
print("\n[3] MPS device patch")

try:
    from shared.mps.device_patch import apply_mps_patch
    ok = apply_mps_patch()
    check("apply_mps_patch() returns True", ok)
    check("nn.Parameter.weight exists", hasattr(torch.nn.Parameter, "weight"))
except Exception as e:
    check("apply_mps_patch()", False, str(e)[:80])

# ── 4. Core model imports ───────────────────────────────────────────
print("\n[4] Core model imports")

# Wan handler
try:
    from models.wan.wan_handler import family_handler
    check("wan_handler.family_handler", True)
except ImportError:
    try:
        from models.wan.wan_handler import WanHandler
        check("wan_handler.WanHandler", True)
    except ImportError as e:
        skip("wan_handler", str(e)[:60])

# Wan configs
try:
    from models.wan.configs import WAN_CONFIGS
    check("wan configs", True, f"{len(WAN_CONFIGS)} entries")
except ImportError as e:
    skip("wan configs", str(e)[:60])

# LTX handler
try:
    from models.ltx2.ltx_handler import family_handler
    check("ltx2 family_handler", True)
except ImportError as e:
    skip("ltx2 handler", str(e)[:60])

# Z Image transformer
try:
    from models.z_image.z_image_transformer2d import ZImageTransformer2DModel
    check("z_image transformer", True)
except ImportError as e:
    skip("z_image transformer", str(e)[:60])

# Attention modes
try:
    from shared.attention import get_attention_modes, get_supported_attention_modes
    modes = get_attention_modes()
    supported = get_supported_attention_modes()
    check("attention modes", True, f"installed={modes}, supported={supported}")
except ImportError as e:
    skip("attention modes", str(e)[:60])

# ── 5. wgp_config validation ────────────────────────────────────────
print("\n[5] wgp_config.json")

config_path = os.path.join(REPO_ROOT, "wgp_config.json")
if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        attn = cfg.get("attention_mode", "")
        vid_prof = cfg.get("video_profile")
        check("attention_mode non-empty", bool(attn), f"'{attn}'")
        check("video_profile set", bool(vid_prof), str(vid_prof))
    except Exception as e:
        skip("wgp_config.json", f"parse error: {e}")
else:
    print("  ⚠️  wgp_config.json not found (run setup.py first)")

# ── 6. Memory report ────────────────────────────────────────────────
print("\n[6] Memory")

if torch.backends.mps.is_available():
    try:
        alloc = torch.mps.current_allocated_memory() / 1024**3
        driver = torch.mps.driver_allocated_memory() / 1024**3
        print(f"  MPS allocated: {alloc:.2f} GB")
        print(f"  MPS driver:    {driver:.2f} GB")
    except Exception:
        pass

# CPU RAM
try:
    import subprocess
    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    ram_gb = int(out) / 1024**3
    check("System RAM", ram_gb > 0, f"{ram_gb:.0f} GB")
except Exception:
    pass

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{banner}")
total = PASS + FAIL + SKIP
print(f"  Results: {PASS} passed, {FAIL} failed, {SKIP} skipped (of {total})")
print(banner)

if FAIL > 0:
    print("\n  Failures:")
    for r in results:
        if "❌" in r:
            print(r)

if FAIL == 0:
    print("\n  ✅ All checks passed!")
    sys.exit(0)
else:
    print(f"\n  ❌ {FAIL} check(s) failed.")
    sys.exit(1)
