# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor</b>
</p>

WanGP supports the Wan (and derived models) but also Hunyuan Video, Flux, Qwen, Z-Image, LongCat, Kandinsky, LTXV, LTX-2, Qwen3 TTS, Chatterbox, HearMula, ... with:
- Low VRAM requirements (as low as 6 GB of VRAM is sufficient for certain models)
- Support for old Nvidia GPUs (RTX 10XX, 20xx, ...)
- Support for AMD GPUs (RDNA 4, 3, 3.5, and 2), instructions in the Installation Section Below.
- Very Fast on the latest GPUs
- Easy to use Full Web based interface
- Support for many checkpoint Quantized formats: int8, fp8, gguf, NV FP4, Nunchaku
- Auto download of the required model adapted to your specific architecture
- Tools integrated to facilitate Video Generation : Mask Editor, Prompt Enhancer, Temporal and Spatial Generation, MMAudio, Video Browser, Pose / Depth / Flow extractor, Motion Designer
- Plenty of ready to use Plug Ins: Gallery Browser, Upscaler, Models/Checkpoints Manager, CivitAI browser and downloader, ...
- Loras Support to customize each model
- Queuing system : make your shopping list of videos to generate and come back later
- Headless mode: launch the generation of multiple image / videos / audio files using a command line

**Discord Server to get Help from the WanGP Community and show your Best Gens:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)


## 🔥 Latest Updates : 
### March 30th 2026: WanGP v11.13, The Machine Within The Machine

Meet **Deepy** your friendly *WanGP Agent*.

It works *offline* with as little of *8 GB of VRAM* and won't *divulge your secrets*. It is *100% free* (no need for a ChatGPT/Claude subscription).

You can ask Deepy to perform for you tedious tasks such as: 
```text
generate a black frame, crop a  video, extract a specific frame from a video, trim an audio, ...
```

Deepy can also perform full workflows:
```text
1) Generate an image of a robot disco dancing on top of a horse in a nightclub.
2) Now edit the image so the setting stays the same, but the robot has gotten off the horse and the horse is standing next to the robot.
3) Verify that the edited image matches the description; if it does not, generate another one.
4) Generate a transition between the two images.
```
or

```text
Create a high quality image portrait that you think represents you best in your favorite setting. Then create an audio sample in which you will introduce the users to your capabilities. When done generate a video based on these two files.
```

Deepy can also transcribe the audio content of a video (*new to WanGP 11.11*)
```text
extract the video from the moment it says "Deepy changed my life"
```

*Deepy* reuses the *Qwen3VL Abliterated* checkpoints and it is highly recommended to install the *GGUF kernels* (check docs/INSTALLATION.md) for low VRAM / fast inference. **now available with Linux!**

Please install also *flash attention 2* and *triton* to enable *vllm* and get x2/x3 speed gain and lower VRAM usage.

You can customize Deepy to use the settings of your choice when generating a video, image, ... (please check docs/DEEPY.Md). 

*Go the Config > Prompt Enhancer / Deep tab to enable Deepy (you must first choose a Qwen3.5VL Prompt Enhancer)*

**Important**: in order to save Deepy from learning all the specificities of each model to generate image, videos or audio, Deepy uses *Predefined Settings Templates* for its six main tools (*Generate Video*, *Generate Image*, ...). You can change the templates used in a session or even add your own settings. Just have a look at the doc.

With WanGP 11.11 you can *ask Deepy to generate a Video or an Image in specific dimensions and also a number of frames for a video*. You can also specify an optional *number of inference of steps* or *loras* to use with *multipliers*. If you don't mention any of these to Deepy, Deepy Default settings or the current Templated Settings will be used instead.

WanGP 11 addresses a long standing Gradio issue: *Queues keep being processed even if your Web Browser is in the background*. Beware this feature may drain more battery, so you can disable it in the *Config / General tab*.

You have maybe also noticed the new option *Keep Intermediate Sliding Windows* in the *Config / Outputs* tab that allows you to discard intermediate *Sliding Windows*


### March 17th 2026: WanGP v10.9875, Prompt Enhancer has just Been Abliterated

- **Qwen3.5 VL Abliterated Prompt Enhancer**: new choice of Prompt Enhancer
   * Based on widely acclaimed *Qwen3.5 model* that has just been released
   * *Uncensored* thanks to the *Abliterating* process that nullifies any *LLM will* to decline any of your request
   * 4 choices of models: depending on how much VRAM you have *4B & 9B models*, and *GGUF Q4* or *Int8*
   * *vllm accelerated* x5 faster, if Flash Attention 2 & Triton are installed (please check docs/INSTALLATION.md) 
   * *Think Mode*: for complex prompt queries
   
   Also you can now expand or override a *System Prompt prompt Enhancer* with add @ or @@ (check new doc *PROMPTS.md*)

- **GGUF CUDA Kernels**: 15% speed gain when using GGUF on Diffusion Video Models & x3 speed with GGUF LLM (*Qwen 3.5 VL GGUF* for instance). GGUF Kernels are for the moment only available for Windows (please check docs/INSTALLATION.md).

- **LTX2.3 Improvements**
   * *End Frame without Start Frame*: you know how your story ends but want to see how it started, just give an End Frame (no start Frame) 
   * New GGUF Checkpoints
   * VAE Decoding hopefully should expose less banding
   * *Multiple Frames Injections*: inject at different positions the reference frames of your choice (works for LTX-2.0 too)
   * *Image Strength* can be applied now too *End Frames* & *Injected Frames*
   * New Spatial Upsampler 1.1, hotfix supposed to improved quality with long video
   * *More VRAM optimisations*: Oops I dit it again ! not that is was needed since WanGP is by far the LTX2 implementation that needs the least VRAM. But now we can in theory (output wont look nice due to LTX2 limitations) generate 15s at full 4K with 24GB of VRAM. So it means that with lower config you should be able to generate longer videos at 720p/1080p. As a bonus you get a 8% speedup.
   * *NVFP4 Dev checkpoint*: if you have a RTX 50xx, help yourself 

- **WanGP API**: rejoice developers (or agents) among you ! WanGP offers now an internal API that allows you to use WanGP as a backend for your apps. It is subject to compliance to the terms & conditions of WanGP license and more specifically to inform the users of your app that WanGP is working behind the scene.

- **LTX Desktop WanGP**: as a sample app (made just for fun) that uses WanGP API, you may try LTX Desktop. This app offers Video / Audio nice editing capabilities but will require 32+ VRAM to run. As now it uses WanGP as its core engine, VRAM requirements are much smaller. It will use LTX 2.3 for Video Gen & Z Image turbo fo Image gen. You can reuse (in theory) your current WanGP install with *LTX Destop WanGP*. https://github.com/deepbeepmeep/LTX-Desktop-WanGP

- **New Audio Ouput formats in mp4**: audio stored in video file can now be of higher quality (*AAC192 - AAC320*) or *ALAC* (lossless). Please note that you wont be able listen to ALAC audio track directly in the webapp.

Also note as people preferred mataynone v1 over v2 I have added an option to select matanyone version in the Config / Extension tab

*update 10.9871*: Improved Qwen3.5 GGUF Prompt Enhancer Output Quality & added Think mode\
*update 10.9872*: Added LTX 2.0/2.3 frames injection\
*update 10.9873*: Fixed low fidelity LTX2 injected frames + added Image Strength slider for end & injected frames\
*update 10.9874*: Replaced LTX-2.3 spatial upsampler by hotfix v1.1\
*update 10.9875*: LTX-2 more VRAM optimisations + NVFP4 checkpoint

### March 7th 2026: WanGP v10.981, Expecting an Update ? 

- **LTX-2 2.3**: 0 day delivery of LTX 2 latest version with better *audio*, *image 2 video* and *greater details*. This model is bigger (22B versus 19B), but with WanGP VRAM usage will be still ridiculously low. Try it at 720p or 1080p, this is where it will shine the most !

*Control Video Support* (*Ic lora Union Control*) will let you transfer *Human Motion*, *Edges*, ... in your new video.

For expert users, *Dev* finetune offers extra new configurable settings (*modality guidance*, *audio guidance*, *STG pertubation/skip self attention *, *guidance rescaling*). LTX team suggests: Cfg=3, Audio cfg=7, Modality Cfg=3, Rescale=0.7, STG Perturbation Skip Attention on all steps.

I recommend to stick to the *Distilled* finetune for higher resolutions (see sample video below) as it seems to have been distilled from a higher quality model (pro model?).

- **Kiwi Edit**: a great model that lets you edit video and / or inject objects in a video. It exists in 3 flavours depending on what you want to do

- **SVI PRO2 End Frames**: this should allow in theory to generate very long shots by splitting one shot into sub shots (sliding windows) by inserting key frames (the *End Frames*). This is an alternative to the *Infinitalk* references frames method (see my old release notes). I am waiting for your feedback to know which method is the best one.

- **Upgraded Models Selector** with *already Downloaded indicator*: Next to each model or finetune, you will find a colored square: *Blue* = fully downloaded & available, *Yellow* = partially downloaded & *Black* = not downloaded at all. Please note that the square color will depend on your current choices of requested model quantization.

- **Upgraded Models Manager**: colors squares have also been added so that you can see in glance what has already been downloaded. New filter for a quick model lookout. List of missing files per finetune.

- **Matanyone 2**: everyone favorite Mask extractor has been been updated and is now more precise

*update 10.981*: LTX2.3 Ic Lora Support & expert settings, Matanyone 2, SVI Pro end frames



See full changelog: **[Changelog](docs/CHANGELOG.md)**


## 🚀 Quick Start

**One-click installation:** 
Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.


**Manual installation: (old python 3.10, to be deprecated)**
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

**Manual installation: (new python 3.11 setup)**
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

**Run the application:**
```bash
python wgp.py
```

First time using WanGP ? Just check the *Guides* tab, and you will find a selection of recommended models to use.

**Update the application (stay in the old pyton / pytorch version):**
If using Pinokio use Pinokio to update otherwise:
Get in the directory where WanGP is installed and:
```bash
git pull
conda activate wan2gp
pip install -r requirements.txt
```

**Upgrade to 3.11, Pytorch 2.10, Cuda 13/13.1** (for non GTX10xx users)
I recommend creating a new conda env for the Python 3.11 to avoid bad surprises. Let's call the new conda env *wangp* (instead of *wan2gp* the old name of this project)
Get in the directory where WanGP is installed and:
```bash
git pull
conda create -n wangp python=3.11.9
conda activate wangp
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

**Git Errors**
Once you are done you will have to reinstall *Sage Attention*, *Triton*, *Flash Attention*. Check the **[Installation Guide](docs/INSTALLATION.md)** -

if you get some error messages related to git, you may try the following (beware this will overwrite local changes made to the source code of WanGP):
```bash
git fetch origin && git reset --hard origin/main
conda activate wangp
pip install -r requirements.txt
```
When you have the confirmation it works well you can then delete the old conda env:
```bash
conda uninstall -n wan2gp --all  
```
**Run headless (batch processing):**

Process saved queues without launching the web UI:
```bash
# Process a saved queue
python wgp.py --process my_queue.zip
```
Create your queue in the web UI, save it with "Save Queue", then process it headless. See [CLI Documentation](docs/CLI.md) for details.

## 🐳 Docker:

**For Debian-based systems (Ubuntu, Debian, etc.):**

```bash
./run-docker-cuda-deb.sh
```

This automated script will:

- Detect your GPU model and VRAM automatically
- Select optimal CUDA architecture for your GPU
- Install NVIDIA Docker runtime if needed
- Build a Docker image with all dependencies
- Run WanGP with optimal settings for your hardware

**Docker environment includes:**

- NVIDIA CUDA 12.4.1 with cuDNN support
- PyTorch 2.6.0 with CUDA 12.4 support
- SageAttention compiled for your specific GPU architecture
- Optimized environment variables for performance (TF32, threading, etc.)
- Automatic cache directory mounting for faster subsequent runs
- Current directory mounted in container - all downloaded models, loras, generated videos and files are saved locally

**Supported GPUs:** RTX 40XX, RTX 30XX, RTX 20XX, GTX 16XX, GTX 10XX, Tesla V100, A100, H100, and more.

## 📦 Installation

### Nvidia
For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions for RTX 10XX to RTX 50XX

### AMD
For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/AMD-INSTALLATION.md)** - Complete setup instructions for RDNA 4, 3, 3.5, and 2

## 🎯 Usage

### Basic Usage
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - First steps and basic usage
- **[Models Overview](docs/MODELS.md)** - Available models and their capabilities
- **[Prompts Guide](docs/PROMPTS.md)** - How WanGP interprets prompts, images as prompts, enhancers, and macros

### Advanced Features
- **[Deepy Assistant](docs/DEEPY.md)** - Enable Deepy, configure its tool presets, use selected media and frames, and run Deepy from the CLI
- **[Loras Guide](docs/LORAS.md)** - Using and managing Loras for customization
- **[Finetunes](docs/FINETUNES.md)** - Add manually new models to WanGP
- **[VACE ControlNet](docs/VACE.md)** - Advanced video control and manipulation
- **[Command Line Reference](docs/CLI.md)** - All available command line options

## 📚 Documentation

- **[Changelog](docs/CHANGELOG.md)** - Latest updates and version history
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 📚 Video Guides
- Nice Video that explain how to use Vace:\
https://www.youtube.com/watch?v=FMo9oN2EAvE
- Another Vace guide:\
https://www.youtube.com/watch?v=T5jNiEhf9xk

## 🔗 Related Projects

### Other Models for the GPU Poor
- **[HuanyuanVideoGP](https://github.com/deepbeepmeep/HunyuanVideoGP)** - One of the best open source Text to Video generators
- **[Hunyuan3D-2GP](https://github.com/deepbeepmeep/Hunyuan3D-2GP)** - Image to 3D and text to 3D tool
- **[FluxFillGP](https://github.com/deepbeepmeep/FluxFillGP)** - Inpainting/outpainting tools based on Flux
- **[Cosmos1GP](https://github.com/deepbeepmeep/Cosmos1GP)** - Text to world generator and image/video to world
- **[OminiControlGP](https://github.com/deepbeepmeep/OminiControlGP)** - Flux-derived application for object transfer
- **[YuE GP](https://github.com/deepbeepmeep/YuEGP)** - Song generator with instruments and singer's voice

---

<p align="center">
Made with ❤️ by DeepBeepMeep
</p>
