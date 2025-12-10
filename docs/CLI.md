# Command Line Reference

This document covers all available command line options for WanGP.

## Basic Usage

```bash
# Default launch
python wgp.py

```

## CLI Queue Processing (Headless Mode)

Process saved queues without launching the web UI. Useful for batch processing or automated workflows.

### Quick Start
```bash
# Process a saved queue (ZIP with attachments)
python wgp.py --process my_queue.zip

# Process a settings file (JSON)
python wgp.py --process my_settings.json

# Validate without generating (dry-run)
python wgp.py --process my_queue.zip --dry-run

# Process with custom output directory
python wgp.py --process my_queue.zip --output-dir ./batch_outputs
```

### Supported File Formats
| Format | Description |
|--------|-------------|
| `.zip` | Full queue with embedded attachments (images, videos, audio). Created via "Save Queue" button. |
| `.json` | Settings file only. Media paths are used as-is (absolute or relative to WanGP folder). Created via "Export Settings" button. |

### Workflow
1. **Create your queue** in the web UI using the normal interface
2. **Save the queue** using the "Save Queue" button (creates a .zip file)
3. **Close the web UI** if desired
4. **Process the queue** via command line:
   ```bash
   python wgp.py --process saved_queue.zip --output-dir ./my_outputs
   ```

### CLI Queue Options
```bash
--process PATH          # Path to queue (.zip) or settings (.json) file (enables headless mode)
--dry-run               # Validate file without generating (use with --process)
--output-dir PATH       # Override output directory (use with --process)
--verbose LEVEL         # Verbosity level 0-2 for detailed logging
```

### Console Output
The CLI mode provides real-time feedback:
```
WanGP CLI Mode - Processing queue: my_queue.zip
Output directory: ./batch_outputs
Loaded 3 task(s)

[Task 1/3] A beautiful sunset over the ocean...
  [12/30] Prompt 1/3 - Denoising | Phase 2/2 Low Noise
  Video saved
  Task 1 completed

[Task 2/3] A cat playing with yarn...
  [30/30] Prompt 2/3 - VAE Decoding
  Video saved
  Task 2 completed

==================================================
Queue completed: 3/3 tasks in 5m 23s
```

### Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Success (all tasks completed) |
| 1 | Error (file not found, invalid queue, or task failures) |
| 130 | Interrupted by user (Ctrl+C) |

### Examples
```bash
# Overnight batch processing
python wgp.py --process overnight_jobs.zip --output-dir ./renders

# Quick validation before long run
python wgp.py --process big_queue.zip --dry-run

# Verbose mode for debugging
python wgp.py --process my_queue.zip --verbose 2

# Combined with other options
python wgp.py --process queue.zip --output-dir ./out --attention sage2
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
- **Profile 4+** (4.5): Profile 4 variation, can save up to 1 GB of VRAM, but will be slighlty slower on some configs
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
--lora-dir-hunyuan-1-5 PATH  # Path to Hunyuan 1.5 loras directory
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
python wgp.py ----lora-preset mystyle.lset

# High-performance setup with compilation
python wgp.py --compile --attention sage2 --profile 3

# Low VRAM setup
python wgp.py --profile 4 --attention sdpa
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