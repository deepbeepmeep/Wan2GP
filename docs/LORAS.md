# Loras Guide

Loras (Low-Rank Adaptations) allow you to customize video generation models by adding specific styles, characters, or effects to your videos.

## Directory Structure

Loras are organized in different folders based on the model they're designed for:

### Text-to-Video Models
- `loras/` - General t2v loras
- `loras/1.3B/` - Loras specifically for 1.3B models
- `loras/14B/` - Loras specifically for 14B models

### Image-to-Video Models
- `loras_i2v/` - Image-to-video loras

### Other Models
- `loras_hunyuan/` - Hunyuan Video t2v loras
- `loras_hunyuan_i2v/` - Hunyuan Video i2v loras
- `loras_ltxv/` - LTX Video loras

## Custom Lora Directory

You can specify custom lora directories when launching the app:

```bash
# Use shared lora directory for both t2v and i2v
python wgp.py --lora-dir /path/to/shared/loras --lora-dir-i2v /path/to/shared/loras

# Specify different directories for different models
python wgp.py --lora-dir-hunyuan /path/to/hunyuan/loras --lora-dir-ltxv /path/to/ltx/loras
```

## Using Loras

### Basic Usage

1. Place your lora files in the appropriate directory
2. Launch WanGP
3. In the Advanced Tab, select the "Loras" section
4. Check the loras you want to activate
5. Set multipliers for each lora (default is 1.0)

### Lora Multipliers

Multipliers control the strength of each lora's effect:

#### Simple Multipliers
```
1.2 0.8
```
- First lora: 1.2 strength
- Second lora: 0.8 strength

#### Time-based Multipliers
For dynamic effects over generation steps, use comma-separated values:
```
0.9,0.8,0.7
1.2,1.1,1.0
```
- For 30 steps: steps 0-9 use first value, 10-19 use second, 20-29 use third
- First lora: 0.9 → 0.8 → 0.7
- Second lora: 1.2 → 1.1 → 1.0

## Lora Presets

Presets are combinations of loras with predefined multipliers and prompts.

### Creating Presets
1. Configure your loras and multipliers
2. Write a prompt with comments (lines starting with #)
3. Save as a preset with `.lset` extension

### Example Preset
```
# Use the keyword "ohnvx" to trigger the lora
A ohnvx character is driving a car through the city
```

### Using Presets
```bash
# Load preset on startup
python wgp.py --lora-preset mypreset.lset
```

### Managing Presets
- Edit, save, or delete presets directly from the web interface
- Presets include comments with usage instructions
- Share `.lset` files with other users

## CausVid Lora (Video Generation Accelerator)

CausVid is a distilled Wan model that generates videos in 4-12 steps with 2x speed improvement.

### Setup Instructions
1. Download the CausVid Lora:
   ```
   https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors
   ```
2. Place in your `loras/` directory

### Usage
1. Select a Wan t2v model (e.g., Wan 2.1 text2video 13B or Vace 13B)
2. Enable Advanced Mode
3. In Advanced Generation Tab:
   - Set Guidance Scale = 1
   - Set Shift Scale = 7
4. In Advanced Lora Tab:
   - Select CausVid Lora
   - Set multiplier to 0.3
5. Set generation steps to 12
6. Generate!

### CausVid Step/Multiplier Relationship
- **12 steps**: 0.3 multiplier (recommended)
- **8 steps**: 0.5-0.7 multiplier
- **4 steps**: 0.8-1.0 multiplier

*Note: Lower steps = lower quality (especially motion)*

## Supported Formats

WanGP supports multiple lora formats:
- **Safetensors** (.safetensors)
- **Replicate** format
- **Standard PyTorch** (.pt, .pth)

## AccVid Lora (Video Generation Accelerator)

AccVid is a distilled Wan model that generates videos with a 2x speed improvement since classifier free guidance is no longer needed (that is cfg = 1).

### Setup Instructions
1. Download the CausVid Lora:

- for t2v models:
   ```
   https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors
   ```

- for i2v models:
   ```
   https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors
   ```

2. Place in your `loras/` directory or `loras_i2v/` directory

### Usage
1. Select a Wan t2v model (e.g., Wan 2.1 text2video 13B or Vace 13B) or Wan i2v model
2. Enable Advanced Mode
3. In Advanced Generation Tab:
   - Set Guidance Scale = 1
   - Set Shift Scale = 5
4. The number steps remain unchanged compared to what you would use with the original model but it will be two times faster since classifier free guidance is not needed

## Performance Tips

### Fast Loading/Unloading
- Loras can be added/removed without restarting the app
- Use the "Refresh" button to detect new loras
- Enable `--check-loras` to filter incompatible loras (slower startup)

### Memory Management
- Loras are loaded on-demand to save VRAM
- Multiple loras can be used simultaneously
- Time-based multipliers don't use extra memory

## Finding Loras

### Sources
- **[Civitai](https://civitai.com/)** - Large community collection
- **HuggingFace** - Official and community loras
- **Discord Server** - Community recommendations

### Creating Loras
- **Kohya** - Popular training tool
- **OneTrainer** - Alternative training solution
- **Custom datasets** - Train on your own content

## Macro System (Advanced)

Create multiple prompts from templates using macros:

```
! {Subject}="cat","woman","man", {Location}="forest","lake","city", {Possessive}="its","her","his"
In the video, a {Subject} is presented. The {Subject} is in a {Location} and looks at {Possessive} watch.
```

This generates:
1. "In the video, a cat is presented. The cat is in a forest and looks at its watch."
2. "In the video, a woman is presented. The woman is in a lake and looks at her watch."
3. "In the video, a man is presented. The man is in a city and looks at his watch."

## Troubleshooting

### Lora Not Working
1. Check if lora is compatible with your model size (1.3B vs 14B)
2. Verify lora format is supported
3. Try different multiplier values
4. Check the lora was trained for your model type (t2v vs i2v)

### Performance Issues
1. Reduce number of active loras
2. Lower multiplier values
3. Use `--check-loras` to filter incompatible files
4. Clear lora cache if issues persist

### Memory Errors
1. Use fewer loras simultaneously
2. Reduce model size (use 1.3B instead of 14B)
3. Lower video resolution or frame count
4. Enable quantization if not already active

## Command Line Options

```bash
# Lora-related command line options
--lora-dir path                    # Path to t2v loras directory
--lora-dir-i2v path               # Path to i2v loras directory  
--lora-dir-hunyuan path           # Path to Hunyuan t2v loras
--lora-dir-hunyuan-i2v path       # Path to Hunyuan i2v loras
--lora-dir-ltxv path              # Path to LTX Video loras
--lora-preset preset              # Load preset on startup
--check-loras                     # Filter incompatible loras
``` 