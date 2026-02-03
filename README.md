# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor</b>
</p>

WanGP supports the Wan (and derived models) but also Hunyuan Video, Flux, Qwen, Z-Image, LongCat, Kandinsky, LTXV, LTX-2, Qwen3 TTS, Chatterbox, HearMula, ... with:
- Low VRAM requirements (as low as 6 GB of VRAM is sufficient for certain models)
- Support for old Nvidia GPUs (RTX 10XX, 20xx, ...)
- Support for AMD GPUs Radeon RX 76XX, 77XX, 78XX & 79XX, instructions in the Installation Section Below.
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
### February 1st 2026: WanGP v10.61, Upgrade Time !

- **LTX-2 Base Tweaks**: new *Quality* features if you found the base model was too fast :
   - New *Modality Guidance* should improve audio / video (lipsync...) according to *LTX-2 team* (beware first *denoising phase* will be 50% slower when used that is if modality guidance> 1)
   - *CFG star*, *Adaptive Project Guidance* should improve quality and better prompt adherence
   - *Skip Layer Guidance*: skipping layer 29 during phase may or may not improve quality
Note that these features are only triggered during first phase of denoising because second phase is distilled denoising no matter what (even on the non distilled model)


- **Flux Klein 4B & 9B Base Models**: *Z Image* has its *base model* in WanGP, so it was fair that *Flux Klein* would have its base model too. Base Models require more steps (up 50) and guidance > 1 but are good starting points for finetunes

The real novelty about this new release is that is has been tested and tuned to work with more recent versions of *Python, Pytorch & Cuda*.
My end goal is to have everbody upgrade to **Python 3.11, Pytorch 2.10, Cuda 13/13.1**.
Once we are all there it will be much easier to provide precompiled kernels for *Nunchaku* *NVPF4*, *Sage Attention*, *Flash Attention*, ...
So please follow the *manual upgrade instructions below* (no Pinokio auto upgrade for the moment) and let me know on Discord if it works with all generations of GPUs (starting from GTX10xx to RTX50xx).
You will find the kernels for this new setup in the **guides/INSTALLATION.md**.

- **Wan Motion Self Refiner**: You will have to thank **Steve Jabz** (*Tophness*) for this one as he has been a big sponsor of the *Self Refiner* and did some extensive study to show me its beauty. The *Self Refiner* should improve the quality of the motion (find it in the *Quality Tab*). It relies on a *Refiner Plan* which indicate which steps should be refined for instance: "2-5:3" (default plan suits well for *light2xv* 4 steps) means steps 2-3 will be refined 3 times (that is 3 denoising attempts will be made to improve each of them, so if the self refiner is used the gen will be up to 3x slower). For the moment the *Self Refiner* is enabled only on Wan t2v & i2v. If you are happy with it, we will support more models.


**Note that PyTorch 2.10 represents at last a decent upgrade, no memory leak when switching models (pytorch 2.8) and bad perfs / VRAM peaks with VAE decoding (pytorch 2.9).**

*Update*: It seems GTX10xx doesnt support Cuda 13.0. Dont't worry I will keep WanGP compatibility with Pytorch 2.7.1 / Cuda 12.8.\
*Update 10.61*: added Self Refiner

### January 29th 2026: WanGP v10.56, Music for your Hearts

WanGP Special *TTS* (Text To Speech) Release:

- **Heart Mula**: *Suno* quality song with lyrics on your local PC. You can generate up to 4 min of music.

- **Ace Step v1**: while waiting for *Ace Step v1.5* (which should be released very soon), enjoy this oldie (2025!) but goodie song generatpr as an appetizer. Ace Step v1 is a very fast Song generator. It is a *Diffusion* based, so dont hesitate to turn on Profile 4 to go as low as 4B VRAM while remaining fast.

- **Qwen 3 TTS**: you can either do *Voice Cloning*, *Generate a Custom Voice based on a Prompt* or use a *Predefined Voice*

- **TTS Features**:
   - **Early stop** : you can abort a gen, while still keeping what has been generated (will work only for TTS models which are *Autoregressive Models*, no need to ask that for Image/Video gens which are *Diffusion Models*)
   - **Specialized Prompt Enhancers**: if you enter the prompt in Heart Mula *"a song about AI generation"*, *WanGP Prompt Enhancer* will generate the corresponding masterpiece for you. Likewise you can enhance "A speech about AI generation" when using Qwen3 TTS or ChatterBox.
   - **Custom Output folder for Audio Gens**: you can now choose a different folder for the *Audio Outputs*
   - **Default Memory Profile for Audio Models**: TTS models can get very slow if you use profile 4 (being autoregressive models, they will need to load all the layers one per one to generate one single audio token then rinse & repeat). On the other hand, they dont't need as much VRAM, so you can now define a more agressive profile (3+ for instance)

- **Z Image Base**: try it if you are into the *Z Image* hype but it will be probably useless for you unless you are a researcher and / or want to build a finetune out of it. This model requires from 35 to 50 steps (4x to 6x slower than *Z Image turbo*) and cfg > 1 (an additional 2x slower) and there is no *Reinforcement Learning* so Output Images wont be as good. The plus side is a higher diversity and *Native Negative Prompt* (versus Z Image virtual Negative Prompt using *NAG*).

Note that Z Image Base is very sensitive to the *Attention Mode*: it is not compatible with *Sage 1* as it produces black frames. So I have disabled Sage for RTX 30xx. Also there are reports it produces some vertical banding artifacts with *Sage 2*

- **Flux 1/2 NAG** : *Flux 2 Klein* is your new best friend but you miss *Negative Prompts*, *NAG* support for Distilled models will make you best buddies forever as NAG simulates Negative prompts.

- **Various Improvements**:
   - Video /Audio Galleries now support deletions of gens done outside WanGP
   - added *MP3 support* for audio outputs
   - *Check for Updates* button for *Plugins* to see in a glance if any of your plugin can be updated
   - *Prompt Enhancer* generates a different enhanced prompt each timee you click on it. You can define in the config tab its gen parameters (top k, temperature)
   - New *Root Loras* folder can be defined in the config Tab. Useful if you have multiple WanGP instances or want to store easily all your loras in a different hard drive 
   - added new setting *Attention Mode Override* in the *Misc* tab
   - Experimental: allowed changing *Configuration* during a *Generation*

*update 10.51*: new Heart Mula Finetune better at following instructions, Extra settings (cfg, top k) for TTS models, Rife v4\
*update 10.52*: updated plugin list and added version tracking\
*update 10.53*: video/audio galleries now support deletions\
*update 10.54*: added Z Image Base, prompt enhancers improvements, configurable loras root folder\
*update 10.55*: blocked Sage with Z Image on RTX30xx and added override attention mode settings, allowed changing config during generation\
*update 10.56*: added NAG for Flux 1/2 & Ace Step v1

### January 20th 2026: WanGP v10.43, The Cost Saver
*GPUs are expensive, RAM is expensive, SSD are expensive, sadly we live now in a GPU & RAM poor.*

WanGP comes again to the rescue:

- **GGUF support**: as some of you know, I am not a big fan of this format because when used with image / video generative models we don't get any speed boost (matrices multiplications are still done at 16 bits), VRAM savings are small and quality is worse than with int8/fp8. Still gguf has one advantage: it consumes less RAM and harddrive space. So enjoy gguf support. I have added ready to use *Kijai gguf finetunes* for *LTX-2*.

- **Models Manager PlugIn**: use this *Plugin* to identify how much space is taken by each *model* / *finetune* and delete the ones you no longer use. Try to avoid deleting shared files otherwise they will be downloaded again.  

- **LTX-2 Dual Video & Audio Control**: you no longer need to extract the audio track of a *Control Video* if you want to use it as well to drive the video generation. New mode will allow you to use both motion and audio from Video Control.

- **LTX-2 - Custom VAE URL**: some users have asked if they could use the old *Distiller VAE* instead of the new one. To do that, create a *finetune* def based on an existing model definition and save it in the *finetunes/* folder with this entry (check the *docs/FINETUNES.md* doc):
```
		"VAE_URLs": ["https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/ltx-2-19b_vae_old.safetensors"]
```

- **Flux 2 Klein 4B & 9B**: try these distilled models as fast as Z_Image if not faster but with out of the box image edition capabiltities

- **Flux 2 & Qwen Outpainting + Lanpaint**: the inpaint mode of these models support now *outpainting* + more combination possible with *Lanpaint* 

- **RAM Optimizations for multi minutes Videos**: processing, saving, spatial & Temporal upsampling very long videos should require much less RAM. 

- **Text Encoder Cache**: if you are asking a Text prompt already used recently with the current model, it will be taken straight from a cache. The cache is optimized to consume little RAM. It wont work with certain models such as Qwen where the Text Prompt is combined internally with an Image.

*update 10.41*: added Flux 2 klein\
*update 10.42*: added RAM optimizations & Text Encoder Cache\
*update 10.43*: added outpainting for Qwen & Flux 2, Lanpaint for Flux 2

### January 15th 2026: WanGP v10.30, The Need for Speed ...

- **LTX Distilled VAE Upgrade**: *Kijai* has observed that the Distilled VAE produces images that were less sharp that the VAE of the Non Distilled model. I have used this as an opportunity to repackage all the LTX-2 checkpoints and reduce their overal HD footprint since they all share around 5GB. 

**So dont be surprised if the old checkpoints are deleted and new are downloaded !!!**.

- **LTX-2 Multi Passes Loras multipliers**: *LTX-2* supports now loras multiplier that depend on the Pass No. For instance "1;0.5" means 1 will the strength for the first LTX-2 pass and 0.5 will be the strength for the second pass.

- **New Profile 3.5**: here is the lost kid of *Profile 3* & *Profile 5*, you got tons of VRAM, but little RAM ? Profile 3.5 will be your new friend as it will no longer use Reserved RAM to accelerate transfers. Use Profile 3.5 only if you can fit entirely a *Diffusion / Transformer* model in VRAM, otherwise the gen may be much slower.

- **NVFP4 Quantization for LTX-2 & Flux 2**: you will now be able to load *NV FP4* model checkpoints in WanGP. On top of *Wan NV4* which was added recently, we now have *LTX-2 (non distilled)* & *Flux 2* support. NV FP4 uses slightly less VRAM and up to 30% less RAM. 

To enjoy fully the NV FP4 checkpoints (**at least 30% faster gens**), you will need a RTX 50xx and to upgrade to *Pytorch 2.9.1 / Cuda 13* with the latest version of *lightx2v kernels* (check *docs/INSTALLATION.md*). To observe the speed gain, you have to make sure the workload is quite high (high res, long video).


### January 13th 2026: WanGP v10.24, When there is no VRAM left there is still some VRAM left ...

- **LTX-2 - SUPER VRAM OPTIMIZATIONS**  

*With WanGP 10.21 HD 720p Video Gens of 10s just need now 8GB of VRAM!*

LTX Team said this video gen was for 4k. So I had no choice but to squeeze more VRAM with further optimizations.

After much suffering I have managed to reduce by at least 1/3 the VRAM requirements of LTX-2, which means:
  - 10s at 720p can be done with only 8GB of VRAM
  - 10s at 1080p with only 12 GB of VRAM
  - 20s at 1080p with only 16 GB of VRAM
  - 10s at Full 4k (3840 x 2176 !!!) with 24 GB of VRAM.  However the bad news is LTX-2 video is not for 4K, as 4K outputs may give you nightmares ...

3K/4K resolutions will be available only if you enable them in the *Config* / *General* tab.

- **Ic Loras support**: Use a *Control Video* to transfer *Pose*, *Depth*, *Canny Edges*. I have added some extra tweaks: with WanGP you can restrict the transfer to a *masked area*, define a *denoising strength* (how much the control video is going to be followed) and a *masking strength* (how much unmasked area is impacted) 

- **Start Image Strength**: This new slider will appear below a *Start Image* or Source *Video*. If you set it to values lower than 1 you may to reduce the static image effect, you get sometime with LTX-2 i2v
 
- **Custom Gemma Text Encoder for LTX-2**: As a practical case, the *Heretic* text encoder is now supported by WanGP. Check the *finetune* doc, but in short create a *finetune* that has a *text_encoder_URLS* key that contains a list of one or more file paths or URLs.  

- **Experimental Auto Recovery Failed Lora Pin**: Some users (with usually PC with less than 64 GB of RAM) have reported Out Of Memory although a model seemed to load just fine when starting a gen with Loras. This is sometime related to WanGP attempting (and failing due to unsufficient reserved RAM) to pin the Loras to Reserved Memory for faster gen. I have experimented a recovery mode that should release sufficient ressources to continue the Video Gen. This may solve the oom crashes with *LTX-2 Default (non distilled)* 

- **Max Loras Pinned Slider**:  If the Auto Recovery Mode is still not sufficient, I have added a Slider at the bottom of the  *Configuration*  / *Performance* tab that you can use to prevent WanGP from Pinning Loras (to do so set it to 0). As if there is no loading attempt there wont be any crash...

*update 10.21*: added slider Loras Max Pinning slider\
*update 10.22*: added support for custom LTX-2 Text Encoder + Auto Recovery mode if Lora Pinning failed\
*update 10.23*: Fixed text prompt ignore in profile 1 & 2 (this created random output videos)

### January 9st 2026: WanGP v10.11, Spoiled again

- **LTX-2**: here is the long awaited *Ovi Challenger*, LTX-2 generates video and an audio soundtrack. As usual this WanGP version is *low VRAM*. You should be able to run it with as low as 10 GB of VRAM. If you have at least 24 GB of VRAM you will be able to generate 20s at 720p in a single window in only 2 minutes with the distilled model.  WanGP LTX-2 version supports on day one, *Start/End keyframes*, *Sliding-Window* / *Video Continuation* and *Generation Preview*. A *LTX-2 distilled* is part of the package for a very fast generation.

With WanGP v10.11 you can now force your soundtrack, it works like *Multitalk* / *Avatar* except in theory it should work with any kind of sound (not just vocals). Thanks to *Kijai* for showing it was possible.

- **Z Image Twin Folder Turbo**: Z Image even faster as this variant can generate images with as little as 1 step (3 steps recommend) 

- **Qwen LanPaint**: very precise *In Painting*, offers a better integration of the inpainted area in the rest of the image. Beware it is up to 5x slower as it "searches" for the best replacement. 

- **Optimized Pytorch Compiler** : *Patience is the Mother of Virtue*. Finally I may (or may not) have fixed the PyTorch compiler with the Wan models. It should work in much diverse situations and takes much less time. 

- **LongCat Video**: experimental support which includes *LongCat Avatar* a talking head model. For the moment it is mostly for models collectors as it is very slow. It needs 40+ steps and each step contains up 3 passes.

- **MMaudio NSFW**: for alternative audio background

*update v10.11*: LTX-2, use your own soundtrack




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

**Update the application (upgrade from python 3.10 to 3.11):**
I recommend creating a new conda env for the Python 3.11 to avoid bad surprises. Let's call the new conda env *wangp* (instead of *wan2gp* the old name of this project)
Get in the directory where WanGP is installed and:
```bash
git pull
conda create -n wangp python=3.11.9
conda activate wangp
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Once you are done you will have to reinstall *Sage Attention*, *Triton*, *Flash Attention* (if installed). Check the **[Installation Guide](docs/INSTALLATION.md)** -

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
- **[Installation Guide](docs/AMD-INSTALLATION.md)** - Complete setup instructions for Radeon RX 76XX, 77XX, 78XX & 79XX

## 🎯 Usage

### Basic Usage
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - First steps and basic usage
- **[Models Overview](docs/MODELS.md)** - Available models and their capabilities

### Advanced Features
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
