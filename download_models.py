#!/usr/bin/env python3
"""
Download script for Wan2GP models:
- flux2_klein_9b
- flux2_klein_4b
- pi_flux2
- hunyuan_1_5_480_i2v_step_distilled

Usage: python download_models.py
"""

import os
import sys
import urllib.request
import time

# Configuration
CHECKPOINTS_DIR = "ckpts"
HF_BASE = "https://huggingface.co/DeepBeepMeep"

def format_bytes(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"

def download_file(url, dest_path, show_progress=True):
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        print(f"  Already exists: {os.path.basename(dest_path)} ({format_bytes(size)})")
        return True
    
    # Create destination directory if needed
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Get file size
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
    except Exception as e:
        print(f"  Error getting file size: {e}")
        total_size = 0
    
    # Download with progress
    print(f"  Downloading: {os.path.basename(dest_path)}")
    
    start_time = time.time()
    downloaded = 0
    
    try:
        with urllib.request.urlopen(url) as response:
            with open(dest_path, 'wb') as out_file:
                block_size = 8192
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        bar_len = 30
                        filled = int(bar_len * percent / 100)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f"\r    [{bar}] {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)}) @ {format_bytes(speed)}/s", end='', flush=True)
        
        if show_progress and total_size > 0:
            print()
        return True
        
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def download_model_list(files, subfolder=""):
    """Download a list of files."""
    success = True
    for filename, url in files:
        dest = os.path.join(CHECKPOINTS_DIR, subfolder, filename) if subfolder else os.path.join(CHECKPOINTS_DIR, filename)
        if not download_file(url, dest):
            success = False
    return success

def main():
    print("=" * 60)
    print("Wan2GP Model Downloader")
    print("=" * 60)
    print(f"\nModels will be downloaded to: {os.path.abspath(CHECKPOINTS_DIR)}")
    print()
    
    # =========================================================================
    # flux2_klein_9b
    # =========================================================================
    print("\n" + "=" * 60)
    print("Downloading flux2_klein_9b...")
    print("=" * 60)
    
    flux2_klein_9b_files = [
        # Main model
        ("flux-2-klein-9b.safetensors", f"{HF_BASE}/Flux2/resolve/main/flux-2-klein-9b.safetensors"),
        ("flux-2-klein-9b_quanto_bf16_int8.safetensors", f"{HF_BASE}/Flux2/resolve/main/flux-2-klein-9b_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(flux2_klein_9b_files)
    
    # Text encoder (Qwen3 8B)
    print("\n  Text Encoder (Qwen3 8B)...")
    qwen3_8b_files = [
        ("qwen3_8b_bf16.safetensors", f"{HF_BASE}/Flux2/resolve/main/qwen3_8b/qwen3_8b_bf16.safetensors"),
        ("qwen3_8b_quanto_bf16_int8.safetensors", f"{HF_BASE}/Flux2/resolve/main/qwen3_8b/qwen3_8b_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(qwen3_8b_files, "qwen3_8b")
    
    # Mistral3 small (shared text encoder for flux2 variants)
    print("\n  Mistral3 Small (shared)...")
    mistral3_small_files = [
        ("mistral3_small_bf16.safetensors", f"{HF_BASE}/Flux2/resolve/main/mistral3small/mistral3_small_bf16.safetensors"),
        ("mistral3_small_quanto_bf16_int8.safetensors", f"{HF_BASE}/Flux2/resolve/main/mistral3small/mistral3_small_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(mistral3_small_files, "mistral3small")
    
    # VAE
    print("\n  VAE...")
    vae_files = [
        ("flux2_vae.safetensors", f"{HF_BASE}/Flux2/resolve/main/flux2_vae.safetensors"),
    ]
    download_model_list(vae_files, "flux2_vae")
    
    # LoRA (improved_klein)
    print("\n  LoRA (improved_klein)...")
    lora_dir = "loras/flux2_klein_9b"
    os.makedirs(lora_dir, exist_ok=True)
    # Note: This uses a Dropbox URL with a temporary key - users may need to download manually
    # or find an alternative mirror
    print(f"  Note: improved_klein.safetensors needs to be downloaded from:")
    print(f"  https://www.dropbox.com/scl/fi/v48q3apj77w4o6g61yugc/improved_klein.safetensors")
    print(f"  Save to: {os.path.abspath(lora_dir)}/improved_klein.safetensors")
    
    # pi-FLUX2 ID LoRA (shared, top-level)
    print("\n  pi-FLUX2 ID LoRA...")
    piid_lora_files = [
        ("gmflux2_k8_piid_4step_lora.safetensors", f"{HF_BASE}/Flux2/resolve/main/gmflux2_k8_piid_4step_lora.safetensors"),
    ]
    download_model_list(piid_lora_files)
    
    # =========================================================================
    # flux2_klein_4b
    # =========================================================================
    print("\n" + "=" * 60)
    print("Downloading flux2_klein_4b...")
    print("=" * 60)
    
    flux2_klein_4b_files = [
        # Main model
        ("flux-2-klein-4b.safetensors", f"{HF_BASE}/Flux2/resolve/main/flux-2-klein-4b.safetensors"),
        ("flux-2-klein-4b_quanto_bf16_int8.safetensors", f"{HF_BASE}/Flux2/resolve/main/flux-2-klein-4b_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(flux2_klein_4b_files)
    
    # Text encoder (Qwen3)
    print("\n  Text Encoder (Qwen3)...")
    qwen3_files = [
        ("qwen3_bf16.safetensors", f"{HF_BASE}/Z-Image/resolve/main/qwen3_bf16.safetensors"),
        ("qwen3_quanto_bf16_int8.safetensors", f"{HF_BASE}/Z-Image/resolve/main/qwen3_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(qwen3_files, "Qwen3")
    
    # VAE (already downloaded above, but needed for both)
    print("\n  VAE (already downloaded above if needed)...")
    
    # =========================================================================
    # pi_flux2
    # =========================================================================
    print("\n" + "=" * 60)
    print("Downloading pi_flux2...")
    print("=" * 60)
    
    # Base flux2_dev model
    print("\n  Base Model (flux2_dev)...")
    flux2_dev_files = [
        ("flux2-dev_quanto_bf16_int8.safetensors", f"{HF_BASE}/Flux2/resolve/main/flux2-dev_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(flux2_dev_files)
    
    # pi-FLUX2 heads
    print("\n  pi-FLUX2 Heads...")
    pi_flux2_heads_files = [
        ("pi_flux2_heads_bf16.safetensors", f"{HF_BASE}/Flux2/resolve/main/pi_flux2_heads_bf16.safetensors"),
    ]
    download_model_list(pi_flux2_heads_files)
    
    # pi-FLUX2 LoRA (already downloaded above)
    print("\n  pi-FLUX2 LoRA (already downloaded above)...")
    
    # Text encoder (Mistral3 - already downloaded with flux2_klein_9b)
    print("\n  Text Encoder (Mistral3 - already downloaded above)...")
    
    # VAE (already downloaded above)
    print("\n  VAE (already downloaded above if needed)...")
    
    # =========================================================================
    # hunyuan_1_5_480_i2v_step_distilled
    # =========================================================================
    print("\n" + "=" * 60)
    print("Downloading hunyuan_1_5_480_i2v_step_distilled...")
    print("=" * 60)
    
    hunyuan_i2v_files = [
        # Main model
        ("hunyuan_video_1.5_i2v_480_step_distilled_bf16.safetensors", f"{HF_BASE}/HunyuanVideo1.5/resolve/main/hunyuan_video_1.5_i2v_480_step_distilled_bf16.safetensors"),
        ("hunyuan_video_1.5_i2v_480_step_distilled_quanto_bf16_int8.safetensors", f"{HF_BASE}/HunyuanVideo1.5/resolve/main/hunyuan_video_1.5_i2v_480_step_distilled_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(hunyuan_i2v_files)
    
    # Text encoder (Qwen2.5-VL-7B)
    print("\n  Text Encoder (Qwen2.5-VL-7B)...")
    qwen25_vl_files = [
        ("Qwen2.5-VL-7B-Instruct_bf16.safetensors", f"{HF_BASE}/Qwen_image/resolve/main/Qwen2.5-VL-7B-Instruct_bf16.safetensors"),
        ("Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors", f"{HF_BASE}/Qwen_image/resolve/main/Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors"),
    ]
    download_model_list(qwen25_vl_files, "Qwen2.5-VL-7B-Instruct")
    
    # VAE files at top level (ckpts/)
    print("\n  VAE files...")
    hunyuan_vae_files = [
        ("hunyuan_video_1_5_VAE_fp32.safetensors", f"{HF_BASE}/HunyuanVideo1.5/resolve/main/hunyuan_video_1_5_VAE_fp32.safetensors"),
        ("hunyuan_video_1_5_VAE.json", f"{HF_BASE}/HunyuanVideo1.5/resolve/main/hunyuan_video_1_5_VAE.json"),
    ]
    download_model_list(hunyuan_vae_files)
    
    # Qwen VAE (for text encoding)
    print("\n  Qwen VAE (for text encoding)...")
    qwen_vae_files = [
        ("qwen_vae.safetensors", f"{HF_BASE}/Qwen_image/resolve/main/qwen_vae.safetensors"),
        ("qwen_vae_config.json", f"{HF_BASE}/Qwen_image/resolve/main/qwen_vae_config.json"),
    ]
    download_model_list(qwen_vae_files)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nModels saved to: {os.path.abspath(CHECKPOINTS_DIR)}")
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(CHECKPOINTS_DIR):
        level = root.replace(CHECKPOINTS_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

if __name__ == "__main__":
    main()