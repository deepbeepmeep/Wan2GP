--vace-1-3B--vace-1-3B# Command Line Reference

This document covers all available command line options for WanGP.

## Basic Usage

```bash
# Default launch 
python wgp.py

# Specific model modes
python wgp.py --i2v           # Image-to-video
python wgp.py --t2v           # Text-to-video (default)
python wgp.py --t2v-14B       # 14B text-to-video model
python wgp.py --t2v-1-3B      # 1.3B text-to-video model
python wgp.py --i2v-14B       # 14B image-to-video model
python wgp.py --i2v-1-3B      # Fun InP 1.3B image-to-video model
python wgp.py --vace-1-3B     # VACE ControlNet 1.3B model
```

## Model and Performance Options

### Model Configuration
```bash
--quantize-transformer BOOL   # Enable/disable transformer quantization (default: True)
--compile                     # Enable PyTorch compilation (requires Triton)
--attention MODE              # Force attention mode: sdpa, flash, sage, sage2
--profile NUMBER              # Performance profile 1-5 (default: 4)
--preload NUMBER              # Preload N MB of diffusion model in VRAM
--fp16                        # Force fp16 instead of bf16 models
--gpu DEVICE                  # Run on specific GPU device (e.g., "cuda:1")
```

### Performance Profiles
- **Profile 1**: Load entire current model in VRAM and keep all unused models in reserved RAM for fast VRAM tranfers 
- **Profile 2**: Load model parts as needed, keep all unused models in reserved RAM for fast VRAM tranfers
- **Profile 3**: Load entire current model in VRAM (requires 24GB for 14B model)
- **Profile 4**: Default and recommended, load model parts as needed, most flexible option
- **Profile 5**: Minimum RAM usage

### Memory Management
```bash
--perc-reserved-mem-max FLOAT # Max percentage of RAM for reserved memory (< 0.5)
```

## Lora Configuration

```bash
--lora-dir PATH              # Path to Wan t2v loras directory
--lora-dir-i2v PATH          # Path to Wan i2v loras directory
--lora-dir-hunyuan PATH      # Path to Hunyuan t2v loras directory
--lora-dir-hunyuan-i2v PATH  # Path to Hunyuan i2v loras directory
--lora-dir-ltxv PATH         # Path to LTX Video loras directory
--lora-preset PRESET         # Load lora preset file (.lset) on startup
--check-loras                # Filter incompatible loras (slower startup)
```

## Generation Settings

### Basic Generation
```bash
--seed NUMBER                # Set default seed value
--frames NUMBER              # Set default number of frames to generate
--steps NUMBER               # Set default number of denoising steps
--advanced                   # Launch with advanced mode enabled
```

### Advanced Generation
```bash
--teacache MULTIPLIER        # TeaCache speed multiplier: 0, 1.5, 1.75, 2.0, 2.25, 2.5
```

## Interface and Server Options

### Server Configuration
```bash
--server-port PORT           # Gradio server port (default: 7860)
--server-name NAME           # Gradio server name (default: localhost)
--listen                     # Make server accessible on network
--share                      # Create shareable HuggingFace URL for remote access
--open-browser               # Open browser automatically when launching
```

### Interface Options
```bash
--lock-config                # Prevent modifying video engine configuration from interface
--theme THEME_NAME           # UI theme: "default" or "gradio"
```

## File and Directory Options

```bash
--settings PATH              # Path to folder containing default settings for all models
--verbose LEVEL              # Information level 0-2 (default: 1)
```

## Examples

### Basic Usage Examples
```bash
# Launch with specific model and loras
python wgp.py --t2v-14B --lora-preset mystyle.lset

# High-performance setup with compilation
python wgp.py --compile --attention sage2 --profile 3

# Low VRAM setup
python wgp.py --t2v-1-3B --profile 4 --attention sdpa

# Multiple images with custom lora directory
python wgp.py --i2v --multiple-images --lora-dir /path/to/shared/loras
```

### Server Configuration Examples
```bash
# Network accessible server
python wgp.py --listen --server-port 8080

# Shareable server with custom theme
python wgp.py --share --theme gradio --open-browser

# Locked configuration for public use
python wgp.py --lock-config --share
```

### Advanced Performance Examples
```bash
# Maximum performance (requires high-end GPU)
python wgp.py --compile --attention sage2 --profile 3 --preload 2000

# Optimized for RTX 2080Ti
python wgp.py --profile 4 --attention sdpa --teacache 2.0

# Memory-efficient setup
python wgp.py --fp16 --profile 4 --perc-reserved-mem-max 0.3
```

### TeaCache Configuration
```bash
# Different speed multipliers
python wgp.py --teacache 1.5   # 1.5x speed, minimal quality loss
python wgp.py --teacache 2.0   # 2x speed, some quality loss
python wgp.py --teacache 2.5   # 2.5x speed, noticeable quality loss
python wgp.py --teacache 0     # Disable TeaCache
```

## Attention Modes

### SDPA (Default)
```bash
python wgp.py --attention sdpa
```
- Available by default with PyTorch
- Good compatibility with all GPUs
- Moderate performance

### Sage Attention
```bash
python wgp.py --attention sage
```
- Requires Triton installation
- 30% faster than SDPA
- Small quality cost

### Sage2 Attention
```bash
python wgp.py --attention sage2
```
- Requires Triton and SageAttention 2.x
- 40% faster than SDPA
- Best performance option

### Flash Attention
```bash
python wgp.py --attention flash
```
- May require CUDA kernel compilation
- Good performance
- Can be complex to install on Windows

## Troubleshooting Command Lines

### Fallback to Basic Setup
```bash
# If advanced features don't work
python wgp.py --attention sdpa --profile 4 --fp16
```

### Debug Mode
```bash
# Maximum verbosity for troubleshooting
python wgp.py --verbose 2 --check-loras
```

### Memory Issue Debugging
```bash
# Minimal memory usage
python wgp.py --profile 4 --attention sdpa --perc-reserved-mem-max 0.2
```



## Configuration Files

### Settings Files
Load custom settings:
```bash
python wgp.py --settings /path/to/settings/folder
```

### Lora Presets
Create and share lora configurations:
```bash
# Load specific preset
python wgp.py --lora-preset anime_style.lset

# With custom lora directory
python wgp.py --lora-preset mystyle.lset --lora-dir /shared/loras
```

## Environment Variables

While not command line options, these environment variables can affect behavior:
- `CUDA_VISIBLE_DEVICES` - Limit visible GPUs
- `PYTORCH_CUDA_ALLOC_CONF` - CUDA memory allocation settings
- `TRITON_CACHE_DIR` - Triton cache directory (for Sage attention) 