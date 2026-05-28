"""Minimal MPS forward pass test - creates WanModel with random weights and tests inference on MPS."""
import os, sys, gc, time
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("=" * 60)
print("Wan2GP MPS Forward Pass Test (Random Weights)")
print("=" * 60)

# Step 1: Apply MPS patch early
import torch
from shared.mps.device_patch import apply_mps_patch
apply_mps_patch()

print(f"\n[1] PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}")
print(f"    Default device: {torch.get_default_device()}")

# Step 2: Import WanModel
print("\n[2] Importing WanModel...")
try:
    from models.wan.modules.model import WanModel
    print("    WanModel imported OK")
except Exception as e:
    print(f"    FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 3: Create a small WanModel (1.3B config)
print("\n[3] Creating WanModel with 1.3B config...")
config = {
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "num_heads": 12,
    "num_layers": 30,
    "patch_size": (1, 2, 2),
    "window_size": (-1, -1),
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "text_len": 512,
    "vae_stride": (4, 8, 8),
}

try:
    model = WanModel(
        dim=config["dim"],
        ffn_dim=config["ffn_dim"],
        freq_dim=config["freq_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        patch_size=config["patch_size"],
        window_size=config["window_size"],
        qk_norm=config["qk_norm"],
        cross_attn_norm=config["cross_attn_norm"],
        eps=config["eps"],
    )
    print(f"    Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
except Exception as e:
    print(f"    FAILED creating model: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 4: Move to MPS
print("\n[4] Moving model to MPS...")
try:
    model = model.to("mps", dtype=torch.bfloat16)
    model.eval()
    print("    Model on MPS OK")
except Exception as e:
    print(f"    FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 5: Create dummy inputs
print("\n[5] Creating dummy inputs...")
batch_size = 1
# Wan patch embedding is Conv3d with patch_size (1,2,2), input needs (B, C, T, H, W)
T, H, W = 9, 30, 52  # frames, height, width (divisible by vae_stride)
patch_size = config["patch_size"]  # (1, 2, 2)
dim = config["dim"]

# Random latent input: (B, in_channels, T, H, W) where in_channels matches patch_embedding
# patch_embedding: Conv3d(in_channels=4, out_channels=dim, kernel_size=patch_size, stride=patch_size)
# After patch embedding: (B, dim, T//1, H//2, W//2) -> flattened to (B, seq_len, dim)
in_channels = 16  # Wan2.1 VAE latent channels
latent_T = T // patch_size[0]  # 9
latent_H = H // patch_size[1]  # 15
latent_W = W // patch_size[2]  # 26
seq_len = latent_T * latent_H * latent_W  # 3510

x_5d = torch.randn(batch_size, in_channels, T, H, W, device="mps", dtype=torch.bfloat16)
print(f"    5D Latent shape: {x_5d.shape}")

# WanModel forward expects x as a LIST of 5D tensors
x_list_input = [x_5d]

# Disable skips_steps_cache (TeaCache step skipping - not needed for basic inference test)
if hasattr(model, "cache"):
    model.cache = None

# text_embedding: Linear(4096, dim) - UMT5-XXL text encoder output is 4096-dim
text_encoder_dim = 4096
text_len = config["text_len"]
# context must be a LIST of 2D tensors [seq_len, text_dim] per item in x_list
context = torch.randn(batch_size, text_len, text_encoder_dim, device="mps", dtype=torch.bfloat16)
# model expects list of 2D tensors: context_list = [context[i] for i in range(batch)]
context_list = [context[i] for i in range(batch_size)]
print(f"    Context list: {len(context_list)} x {context_list[0].shape}")

t = torch.tensor([500] * batch_size, device="mps", dtype=torch.bfloat16)
print(f"    Timestep: {t}")

# Step 6: Forward pass
print("\n[6] Running forward pass...")
torch.mps.synchronize()
start = time.time()
try:
    with torch.no_grad():
        with torch.autocast("mps", dtype=torch.bfloat16):
            output = model(x_list_input, t, context_list)
    torch.mps.synchronize()
    elapsed = time.time() - start
    print(f"    Forward pass OK: output type = {type(output)}")
    if isinstance(output, list):
        for j, o in enumerate(output):
            print(f"    Output[{j}] shape = {o.shape}, dtype = {o.dtype}")
            print(f"    Output[{j}] range: [{o.min():.3f}, {o.max():.3f}]")
    else:
        print(f"    Output shape = {output.shape}, dtype = {output.dtype}")
except Exception as e:
    elapsed = time.time() - start
    print(f"    FAILED after {elapsed:.2f}s: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 7: Memory check
print("\n[7] Memory stats...")
print(f"    MPS memory allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f}GB")
print(f"    MPS memory driver allocated: {torch.mps.driver_allocated_memory() / 1024**3:.2f}GB")

print("\n" + "=" * 60)
print("MPS Forward Pass Test PASSED!")
print("=" * 60)
