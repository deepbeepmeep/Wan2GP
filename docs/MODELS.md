# Models Overview

WanGP supports multiple video generation models, each optimized for different use cases and hardware configurations. 

Most models can combined with Loras Accelerators (check the Lora guide) to accelerate the generation of a video x2 or x3 with little quality loss


## Wan 2.1 Text2Video Models
Please note that that the term *Text2Video* refers to the underlying Wan architecture but as it has been greatly improved overtime many derived Text2Video models can now  generate videos using images.

#### Wan 2.1 Text2Video 1.3B
- **Size**: 1.3 billion parameters
- **VRAM**: 6GB minimum
- **Speed**: Fast generation
- **Quality**: Good quality for the size
- **Best for**: Quick iterations, lower-end hardware
- **Command**: `python wgp.py --t2v-1-3B`

#### Wan 2.1 Text2Video 14B
- **Size**: 14 billion parameters  
- **VRAM**: 12GB+ recommended
- **Speed**: Slower but higher quality
- **Quality**: Excellent detail and coherence
- **Best for**: Final production videos
- **Command**: `python wgp.py --t2v-14B`

#### Wan Vace 1.3B
- **Type**: ControlNet for advanced video control
- **VRAM**: 6GB minimum
- **Features**: Motion transfer, object injection, inpainting
- **Best for**: Advanced video manipulation
- **Command**: `python wgp.py --vace-1.3B`

#### Wan Vace 14B
- **Type**: Large ControlNet model
- **VRAM**: 12GB+ recommended
- **Features**: All Vace features with higher quality
- **Best for**: Professional video editing workflows

#### MoviiGen (Experimental)
- **Resolution**: Claims 1080p capability
- **VRAM**: 20GB+ required
- **Speed**: Very slow generation
- **Features**: Should generate cinema like video, specialized for 2.1 / 1 ratios
- **Status**: Experimental, feedback welcome

<BR>

## Wan 2.1 Image-to-Video Models

#### Wan 2.1 Image2Video 14B
- **Size**: 14 billion parameters  
- **VRAM**: 12GB+ recommended
- **Speed**: Slower but higher quality
- **Quality**: Excellent detail and coherence
- **Best for**: Most Loras available work with this model
- **Command**: `python wgp.py --i2v-14B`

#### FLF2V
- **Type**: Start/end frame specialist
- **Resolution**: Optimized for 720p
- **Official**: Wan team supported
- **Use case**: Image-to-video with specific endpoints


<BR>

## Wan 2.1 Specialized Models

#### Multitalk
- **Type**: Multi Talking head animation
- **Input**: Voice track + image
- **Works on**: People
- **Use case**: Lip-sync and voice-driven animation for up to two people

#### FantasySpeaking
- **Type**: Talking head animation
- **Input**: Voice track + image
- **Works on**: People and objects
- **Use case**: Lip-sync and voice-driven animation

#### Phantom
- **Type**: Person/object transfer
- **Resolution**: Works well at 720p
- **Requirements**: 30+ steps for good results
- **Best for**: Transferring subjects between videos

#### Recam Master
- **Type**: Viewpoint change
- **Requirements**: 81+ frame input videos, 15+ denoising steps
- **Use case**: View same scene from different angles

#### Sky Reels v2 Diffusion
- **Type**: Diffusion Forcing model
- **Specialty**: "Infinite length" videos
- **Features**: High quality continuous generation


<BR>

## Wan Fun InP Models

#### Wan Fun InP 1.3B
- **Size**: 1.3 billion parameters
- **VRAM**: 6GB minimum
- **Quality**: Good for the size, accessible to lower hardware
- **Best for**: Entry-level image animation
- **Command**: `python wgp.py --i2v-1-3B`

#### Wan Fun InP 14B
- **Size**: 14 billion parameters
- **VRAM**: 12GB+ recommended
- **Quality**: Better end image support
- **Limitation**: Existing loras don't work as well

<BR>


## Hunyuan Video Models

#### Hunyuan Video Text2Video
- **Quality**: Among the best open source t2v models
- **VRAM**: 12GB+ recommended
- **Speed**: Slower generation but excellent results
- **Features**: Superior text adherence and video quality, up to 10s of video
- **Best for**: High-quality text-to-video generation

#### Hunyuan Video Custom
- **Specialty**: Identity preservation
- **Use case**: Injecting specific people into videos
- **Quality**: Excellent for character consistency
- **Best for**: Character-focused video generation

#### Hunyuan Video Avater
- **Specialty**: Generate up to 15s of high quality speech / song driven Video .
- **Use case**: Injecting specific people into videos
- **Quality**: Excellent for character consistency
- **Best for**: Character-focused video generation, Video synchronized with voice


<BR>

## LTX Video Models

#### LTX Video 13B
- **Specialty**: Long video generation
- **Resolution**: Fast 720p generation
- **VRAM**: Optimized by WanGP (4x reduction in requirements)
- **Best for**: Longer duration videos

#### LTX Video 13B Distilled
- **Speed**: Generate in less than one minute
- **Quality**: Very high quality despite speed
- **Best for**: Rapid prototyping and quick results

<BR>

## Model Selection Guide

### By Hardware (VRAM)

#### 6-8GB VRAM
- Wan 2.1 T2V 1.3B
- Wan Fun InP 1.3B
- Wan Vace 1.3B

#### 10-12GB VRAM
- Wan 2.1 T2V 14B
- Wan Fun InP 14B
- Hunyuan Video (with optimizations)
- LTX Video 13B

#### 16GB+ VRAM
- All models supported
- Longer videos possible
- Higher resolutions
- Multiple simultaneous Loras

#### 20GB+ VRAM
- MoviiGen (experimental 1080p)
- Very long videos
- Maximum quality settings

### By Use Case

#### Quick Prototyping
1. **LTX Video 13B Distilled** - Fastest, high quality
2. **Wan 2.1 T2V 1.3B** - Fast, good quality
3. **CausVid Lora** - 4-12 steps, very fast

#### Best Quality
1. **Hunyuan Video** - Overall best t2v quality
2. **Wan 2.1 T2V 14B** - Excellent Wan quality
3. **Wan Vace 14B** - Best for controlled generation

#### Advanced Control
1. **Wan Vace 14B/1.3B** - Motion transfer, object injection
2. **Phantom** - Person/object transfer
3. **FantasySpeaking** - Voice-driven animation

#### Long Videos
1. **LTX Video 13B** - Specialized for length
2. **Sky Reels v2** - Infinite length videos
3. **Wan Vace + Sliding Windows** - Up to 1 minute

#### Lower Hardware
1. **Wan Fun InP 1.3B** - Image-to-video
2. **Wan 2.1 T2V 1.3B** - Text-to-video
3. **Wan Vace 1.3B** - Advanced control

<BR>

## Performance Comparison

### Speed (Relative)
1. **CausVid Lora** (4-12 steps) - Fastest
2. **LTX Video Distilled** - Very fast
3. **Wan 1.3B models** - Fast
4. **Wan 14B models** - Medium
5. **Hunyuan Video** - Slower
6. **MoviiGen** - Slowest

### Quality (Subjective)
1. **Hunyuan Video** - Highest overall
2. **Wan 14B models** - Excellent
3. **LTX Video models** - Very good
4. **Wan 1.3B models** - Good
5. **CausVid** - Good (varies with steps)

### VRAM Efficiency
1. **Wan 1.3B models** - Most efficient
2. **LTX Video** (with WanGP optimizations)
3. **Wan 14B models**
4. **Hunyuan Video**
5. **MoviiGen** - Least efficient

<BR>

## Model Switching

WanGP allows switching between models without restarting:

1. Use the dropdown menu in the web interface
2. Models are loaded on-demand
3. Previous model is unloaded to save VRAM
4. Settings are preserved when possible

<BR>

## Tips for Model Selection

### First Time Users
Start with **Wan 2.1 T2V 1.3B** to learn the interface and test your hardware.

### Production Work
Use **Hunyuan Video** or **Wan 14B** models for final output quality.

### Experimentation
**CausVid Lora** or **LTX Distilled** for rapid iteration and testing.

### Specialized Tasks
- **VACE** for advanced control
- **FantasySpeaking** for talking heads
- **LTX Video** for long sequences

### Hardware Optimization
Always start with the largest model your VRAM can handle, then optimize settings for speed vs quality based on your needs. 