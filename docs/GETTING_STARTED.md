# Getting Started with WanGP

This guide will help you get started with WanGP video generation quickly and easily.

## Prerequisites

Before starting, ensure you have:
- A compatible GPU (RTX 10XX or newer recommended)
- Python 3.10.9 installed
- At least 6GB of VRAM for basic models
- Internet connection for model downloads

## Quick Setup

### Option 1: One-Click Installation (Recommended)
Use [Pinokio App](https://pinokio.computer/) for the easiest installation experience.

### Option 2: Manual Installation
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

## First Launch

### Basic Launch
```bash
python wgp.py
```
This launches the WanGP generator with default settings. You will be able to pick from a Drop Down menu which model you want to use.

### Alternative Modes
```bash
python wgp.py --i2v        # Wan Image-to-video mode
python wgp.py --t2v-1-3B   # Wan Smaller, faster model
```

## Understanding the Interface

When you launch WanGP, you'll see a web interface with several sections:

### Main Generation Panel
- **Model Selection**: Dropdown to choose between different models
- **Prompt**: Text description of what you want to generate
- **Generate Button**: Start the video generation process

### Advanced Settings (click checkbox to enable)
- **Generation Settings**: Steps, guidance, seeds
- **Loras**: Additional style customizations
- **Sliding Window**: For longer videos

## Your First Video

Let's generate a simple text-to-video:

1. **Launch WanGP**: `python wgp.py`
2. **Open Browser**: Navigate to `http://localhost:7860`
3. **Enter Prompt**: "A cat walking in a garden"
4. **Click Generate**: Wait for the video to be created
5. **View Result**: The video will appear in the output section

### Recommended First Settings
- **Model**: Wan 2.1 text2video 1.3B (faster, lower VRAM)
- **Frames**: 49 (about 2 seconds)
- **Steps**: 20 (good balance of speed/quality)

## Model Selection

### Text-to-Video Models
- **Wan 2.1 T2V 1.3B**: Fastest, lowest VRAM (6GB), good quality
- **Wan 2.1 T2V 14B**: Best quality, requires more VRAM (12GB+)
- **Hunyuan Video**: Excellent quality, slower generation
- **LTX Video**: Good for longer videos

### Image-to-Video Models
- **Wan Fun InP 1.3B**: Fast image animation
- **Wan Fun InP 14B**: Higher quality image animation
- **VACE**: Advanced control over video generation

### Choosing the Right Model
- **Low VRAM (6-8GB)**: Use 1.3B models
- **Medium VRAM (10-12GB)**: Use 14B models or Hunyuan
- **High VRAM (16GB+)**: Any model, longer videos

## Basic Settings Explained

### Generation Settings
- **Frames**: Number of frames (more = longer video)
  - 25 frames ≈ 1 second
  - 49 frames ≈ 2 seconds
  - 73 frames ≈ 3 seconds

- **Steps**: Quality vs Speed tradeoff
  - 15 steps: Fast, lower quality
  - 20 steps: Good balance
  - 30+ steps: High quality, slower

- **Guidance Scale**: How closely to follow the prompt
  - 3-5: More creative interpretation
  - 7-10: Closer to prompt description
  - 12+: Very literal interpretation

### Seeds
- **Random Seed**: Different result each time
- **Fixed Seed**: Reproducible results
- **Use same seed + prompt**: Generate variations

## Common Beginner Issues

### "Out of Memory" Errors
1. Use smaller models (1.3B instead of 14B)
2. Reduce frame count
3. Lower resolution in advanced settings
4. Enable quantization (usually on by default)

### Slow Generation
1. Use 1.3B models for speed
2. Reduce number of steps
3. Install Sage attention (see [INSTALLATION.md](INSTALLATION.md))
4. Enable TeaCache: `python wgp.py --teacache 2.0`

### Poor Quality Results
1. Increase number of steps (25-30)
2. Improve prompt description
3. Use 14B models if you have enough VRAM
4. Enable Skip Layer Guidance in advanced settings

## Writing Good Prompts

### Basic Structure
```
[Subject] [Action] [Setting] [Style/Quality modifiers]
```

### Examples
```
A red sports car driving through a mountain road at sunset, cinematic, high quality

A woman with long hair walking on a beach, waves in the background, realistic, detailed

A cat sitting on a windowsill watching rain, cozy atmosphere, soft lighting
```

### Tips
- Be specific about what you want
- Include style descriptions (cinematic, realistic, etc.)
- Mention lighting and atmosphere
- Describe the setting in detail
- Use quality modifiers (high quality, detailed, etc.)

## Next Steps

Once you're comfortable with basic generation:

1. **Explore Advanced Features**:
   - [Loras Guide](LORAS.md) - Customize styles and characters
   - [VACE ControlNet](VACE.md) - Advanced video control
   - [Command Line Options](CLI.md) - Optimize performance

2. **Improve Performance**:
   - Install better attention mechanisms
   - Optimize memory settings
   - Use compilation for speed

3. **Join the Community**:
   - [Discord Server](https://discord.gg/g7efUW9jGV) - Get help and share videos
   - Share your best results
   - Learn from other users

## Troubleshooting First Steps

### Installation Issues
- Ensure Python 3.10.9 is used
- Check CUDA version compatibility
- See [INSTALLATION.md](INSTALLATION.md) for detailed steps

### Generation Issues
- Check GPU compatibility
- Verify sufficient VRAM
- Try basic settings first
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for specific issues

### Performance Issues
- Use appropriate model for your hardware
- Enable performance optimizations
- Check [CLI.md](CLI.md) for optimization flags

Remember: Start simple and gradually explore more advanced features as you become comfortable with the basics! 