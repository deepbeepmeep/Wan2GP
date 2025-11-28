# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor</b>
</p>

WanGP supports the Wan (and derived models), Hunyuan Video and LTV Video models with:
- Low VRAM requirements (as low as 6 GB of VRAM is sufficient for certain models)
- Support for old Nvidia GPUs (RTX 10XX, 20xx, ...)
- Support for AMD GPUs Radeon RX 76XX, 77XX, 78XX & 79XX, instructions in the Installation Section Below.
- Very Fast on the latest GPUs
- Easy to use Full Web based interface
- Auto download of the required model adapted to your specific architecture
- Tools integrated to facilitate Video Generation : Mask Editor, Prompt Enhancer, Temporal and Spatial Generation, MMAudio, Video Browser, Pose / Depth / Flow extractor
- Loras Support to customize each model
- Queuing system : make your shopping list of videos to generate and come back later

**Discord Server to get Help from Other Users and show your Best Videos:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

-----

### You have your choice of Dark or Light Theme


<img width="1895" height="1023" alt="Screenshot 2025-10-23 210313" src="https://github.com/user-attachments/assets/3778ae4e-6a95-4752-ba47-bb160c653310" />

-----
<img width="1899" height="1020" alt="Screenshot 2025-10-23 210500" src="https://github.com/user-attachments/assets/5e524260-ad24-4203-acf2-6622676a83bb" />

-----
![Screen Recording 2025-10-23 210625 - frame at 0m9s](https://github.com/user-attachments/assets/c65a815e-09fa-41a7-bc49-5f879b0b8ece)

-----

## 🔥 Latest Updates : 

### November 28 2025: WanGP v9.72, The Alpha & the Omega

- **Flux 2**: the best ever open source *Image Generator* has just landed. It does everything very well: generate an *Image* based a *Text Prompt* or combine up to 10 *Images References* 

The only snag is that it is a 60B parameters for the *Transformer* part and 40B parameters for the *Text Encoder* part.

Behold the WanGP Miracle ! Flux 2 wil work with only 8 GB of VRAM if you are happy with 8 bits quantization (no need for lower quality 4bits). With 9GB of VRAM you can run the model at full power. You will need at least 64 GB of RAM. If not maybe Memory Profile 5 will be your friend.

- **Z-Image**: a small model, very fast (8 steps), very low VRAM (optimized even more in WanGP for fun, just in case you want to generate 16 images at a time) that produces outstanding Image quality. Not yet the Flux 2 level, and no Image editing yet but a very good trade-off.

I have added a new Memory Profile *Profile 4+* that is sligthly slower than *Profile 4* but can save you up to 1GB of VRAM with Flux 2.

Also as we have now quite few models and Loras folders. *I have moved all the loras folder in the 'loras' folder*. There are also now unique subfolders for *Wan 5B* and *Wan 1.3B* models. A conversion script should have moved the loras in the right locations, but I advise that you check just in case.

*update 9.71* : added missing source file, have fun !
*update 9.72* : added Z-Image & Loras reorg

### November 24 2025: WanGP v9.62, The Return of the King

So here is *Tencet* who is back in the race: let's welcome **Hunyuan Video 1.5**

Despite only 8B parameters it offers quite a high level of quality. It is not just one model but a family of models:
- Text 2 Video
- Image 2 Video
- Upsamplers (720p & 1080p)

Each model comes on day one with several finetunes specialized for a specific resolution.
The downside right now is that to get the best quality you need to use guidance > 1 and a high number of Steps (20+). 

But dont go away yet ! **LightX2V** (https://huggingface.co/lightx2v/Hy1.5-Distill-Models/) is on deck and has already delivered an *Accelerated 4 steps Finetune* for the *t2v 480p* model. It is part of today's delivery.

I have extracted *LighX2V Magic* into an *8 steps Accelerator Lora* that seems to work for i2v and the other resolutions. This should be good enough while waiting for other the official LighX2V releases (just select this lora in the *Settings* Dropdown Box).

WanGP implementation of Hunyuan 1.5 is quite complete as you will get straight away *Video Gen Preview* (WanGP exclusivity!) and *Sliding Window* support. It is also ready for *Tea Cache* or *Mag Cache* (just waiting for the official parameters) 

*WanGP Hunyuan 1.5 is super VRAM optimized, you will need less than 20 GB of VRAM to generate 12s (289 frames) at 720p.*

Please note Hunyuan v1 Loras are not compatible since the latent space is different. You can add loras for Hunyuan Video 1.5 in the *loras_hunyuan/1.5* folder. 

*Update 9.62* : Added Lora Accelerator\
*Update 9.61* : Added VAE Temporal Tiling

### November 21 2025: WanGP v9.52, And there was motion

In this release WanGP turns you into a Motion Master:
- **Motion Designer**: this new preinstalled home made Graphical Plugin will let you design trajectories for *Vace* and for *Wan 2.2 i2v Time to Move*. 

- **Vace Motion**: this is a less known feature of the almighty *Vace* (this was last Vace feature not yet implemented in WanGP), just put some moving rectangles in your *Control Video* (in Vace raw format) and you will be able to move around people / objects or even the camera. The *Motion Designer* will let you create these trajectories in only a few clicks.

- **Wan 2.2 i2v Time to Move**: a few brillant people (https://github.com/time-to-move/TTM) discovered that you could steer the motion of a model such as *Wan 2.2 i2v* without changing its weights. You just need to apply specific *Control* and *Mask* videos. The *Motion Designer* has an *i2v TTM* mode that will let you generate the videos in the right format. The way it works is that using a *Start Image* you are going to define objects and their corresponding trajectories. For best results, it is recommended to provide as well a *Background Image* which is the Start Image without the objects you are moving (use Qwen for that). TTM works with Loras Accelerators.

*TTM Suggested Settings:  Lightning i2v v1.0 2 Phases (8 Steps), Video to Video, Denoising Strenght 0.9, Masking Strength 0.1*. I will upload Sample Settings later in the *Settings Channel* 

- **PainterI2V**: (https://github.com/princepainter/). You found that the i2v loras accelerators kill the motion ? This is an alternative to 3 phases guidance to restore motion, it is free as it doesnt require any extra processing or changing the weights. It works best in a scene where the background remains the same. In order to control the acceleration in i2v models, you will find a new *Motion Amplitude* slider in the *Quality* tab.

- **Nexus 1.3B**: this is an incredible *Wan 2.1 1.3B* finetune made by @Nexus. It is specialized in *Human Motion* (dance, fights, gym, ...). It is fast as it is already *Causvid* accelerated. Try it with the *Prompt Enhancer* at 720p.

- **Black Start Frames** for Wan 2.1/2.2 i2v: some i2v models can be turned into powerful t2v models by providing a **black frame** as a *Start Frame*. From now on if you dont provide any start frame, WanGP will generate automatically a black start frame of the current output resolution or of the correspondig *End frame resolution* (if any). 

*update 9.51*: Fixed Chrono Edit Output, added Temporal Reasoning Video\
*update 9.52*: Black start frames support for Wan i2v models

### November 12 2025: WanGP v9.44, Free Lunch

**VAE Upsampler** for Wan 2.1/2.2 Text 2 Image and Qwen Image: *spacepxl* has tweaked the VAE Decoder used by *Wan* & *Qwen* so that it can decode and upsample x2 at the same time. The end Result is a Fast High Quality Image Upsampler (much better than Lanczos). Check the *Postprocessing Tab* / *Spatial Upsampling* Dropdown box. Unfortunately this will work only with Image Generation, no support yet for Video Generation. I have also added a VAE Refiner that keeps the existing resolution but slightly improves the details.

**Mocha**: a very requested alternative to *Wan Animate* . Use this model to replace a person in a control video. For best results you will need to provide two reference images for the new the person, the second image should be a face close up. This model seems to be optimized to generate 81 frames. First output frame is often messed up. *Lightx2v t2v 4 steps Lora Accelarator* works well. Please note this model is VRAM hungry, for 81 frames to generate it will process internaly 161 frames.

**Lucy Edit v1.1**: a new version (finetune) has been released. Not sure yet if I like it better than the original one. In theory it should work better with changing the background setting for instance.

**Ovi 1.1**: This new version exists in two flavors 5s & 10s ! Thanks to WanGP VRAM optimisations only 8 GB will be only needed for a 10s generation. Beware, the Prompt syntax has slightly changed since an audio background is now introduced using *"Audio:"* instead of using tags.

**Top Models Selection**: if you are new to WanGP or are simply lost among the numerous models offered by WanGP, just check the updated *Guides* tab. You will find a list of highlighted models and advice about how & when to use them.


*update 9.41*: Added Mocha & Lucy Edit 1.1\
*update 9.42*: Added Ovi 1.1
*update 9.43*: Improved Linux support: no more visual artifacts with fp8 finetunes, auto install ffmpeg, detect audio device, ...
*update 9.44*: Added links to highlighted models in Guide tab

### November 6 2025: WanGP v9.35, How many bananas are too many bananas ?

- **Chrono Edit**: a new original way to edit an Image. This one will generate a Video will that performs the full edition work and return the last Image. It can be hit or a miss but when it works it is quite impressive. Please note you must absolutely use the *Prompt Enhancer* on your *Prompt Instruction* because this model expects a very specific format. The Prompt Enhancer for this model has a specific System Prompt to generate the right Chrono Edit Prompt.

- **LyCoris** support: preliminary basic Lycoris support for this Lora format. At least Qwen Multi Camera should work (https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles). If you have a Lycoris that does not work and it may be interesting please mention it in the Request Channel

- **i2v Enhanced Lightning v2** (update 9.37): added this impressive *Finetune* in the default selection of models, not only it is accelerated (4 steps), but it is very good at following camera and timing instructions.

This finetune loves long prompts. Therefore to increase the prompt readability WanGP supports now multilines prompts (in option).

*update 9.35*: Added a Sample PlugIn App that shows how to collect and modify settings from a PlugIn\
*update 9.37*: Added i2v Enhanced Lightning

### October 29 2025: WanGP v9.21, Why isn't all my VRAM used ?


*WanGP exclusive*:  VRAM requirements have never been that low !

**Wan 2.2 Ovi 10 GB** for all the GPU Poors of the World: *only 6 GB of VRAM to generate 121 frames at 720p*. With 16 GB of VRAM, you may even be able to load all the model in VRAM with *Memory Profile 3*

To get the x10 speed effect just apply the FastWan Lora Accelerator that comes prepackaged with Ovi (acccessible in the  dropdown box Settings at the top)

After thorough testing it appears that *Pytorch 2.8* is causing RAM memory leaks when switching models as it won't release all the RAM. I could not find any workaround. So the default Pytorch version to use with WanGP is back to *Pytorch 2.7*
Unless you want absolutely to use Pytorch compilation which is not stable with Pytorch 2.7 with RTX 50xx , it is recommended to switch back to Pytorch 2.7.1 (tradeoff between 2.8 and 2.7):
```bash
cd Wan2GP
conda activate wan2gp
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```
You will need to reinstall SageAttention FlashAttnetion, ...

*update v9.21*: Got FastWan to work with Ovi: it is now 10 times faster ! (not including the VAE)\
*update v9.25*: added Chroma Radiance october edition + reverted to pytorch 2.7

### October 24 2025: WanGP v9.10, What else will you ever need after this one ?

With WanGP v9 you will have enough features to go to a desert island with no internet connection and comes back with a full Hollywood movie.

First here are the new models supported:
- **Wan 2.1 Alpha** : a very requested model that can generate videos with *semi transparent background* (as it is very lora picky it supports only the *Self Forcing / lightning* loras accelerators)
- **Chatterbox Multilingual**: the first *Voice Generator* in WanGP. Let's say you have a flu and lost your voice (somehow I can't think of another usecase), the world will still be able to hear you as *Chatterbox* can generate up to 15s clips of your voice using a recorded voice sample. Chatterbox works with numerous languages out the box.
- **Flux DreamOmni2** : another wannabe *Nano Banana* image Editor / image composer. The *Edit Mode* ("Conditional Image is first Main Subject ...") seems to work better than the *Gen Mode* (Conditional Images are People / Objects ..."). If you have at least 16 GB of VRAM it is recommended to force profile 3 for this model (it uses an autoregressive model for the prompt encoding and the start may be slow).
- **Ditto** (new with *WanGP 9.1* !): a powerful Video 2 Video model, can change for instance the style or the material visible in the video. Be aware it is an instruct based model, so the prompt should contain intructions. 

Upgraded Features:
- A new **Audio Gallery** to store your Chatterbox generations and import your audio assets. *Metadata support* (stored gen settings) for *Wav files* generated with WanGP available from day one. 
- **Matanyone** improvements: you can now use it during a video gen, it will *suspend gracefully the Gen in progress*. *Input Video / Images* can be resized for faster processing & lower VRAM. Image version can now generate *Green screens* (not used by WanGP but I did it because someone asked for it and I am nice) and *Alpha masks*.
- **Images Stored in Metadata**: Video Gen *Settings Metadata* that are stored in the Generated Videos can now contain the Start Image, Image Refs used to generate the Video. Many thanks to **Gunther-Schulz** for this contribution
- **Three Levels of Hierarchy** to browse the models / finetunes: you can collect as many finetunes as you want now and they will no longer encumber the UI.
- Added **Loras Accelerators** for *Wan 2.1 1.3B*, *Wan 2.2 i2v*, *Flux* and the latest *Wan 2.2 Lightning*
- Finetunes now support **Custom Text Encoders** : you will need to use the "text_encoder_URLs" key. Please check the finetunes doc. 
- Sometime Less is More: removed the palingenesis finetunes that were controversial

Huge Kudos & Thanks to **Tophness** that has outdone himself with these Great Features:
- **Multicolors Queue** items with **Drag & Drop** to reorder them
- **Edit a Gen Request** that is already in the queue
- Added **Plugin support** to WanGP : found that features are missing in WanGP, you can now add tabs at the top in WanGP. Each tab may contain a full embedded App that can share data with the Video Generator of WanGP. Please check the Plugin guide written by Tophness and don't hesitate to contact him or me on the Discord if you have a plugin you want to share. I have added a new Plugins channels to discuss idea of plugins and help each other developing plugins. *Idea for a PlugIn that may end up popular*: a screen where you view the hard drive space used per model and that will let you remove unused models weights
- Two Plugins ready to use designed & developped by **Tophness**: an **Extended Gallery** and a **Lora multipliers Wizard**

WanGP v9 is now targetting Pytorch 2.8 although it should still work with 2.7, don't forget to upgrade by doing:
```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```
You will need to upgrade Sage Attention or Flash (check the installation guide)

*Update info: you might have some git error message while upgrading to v9 if WanGP is already installed.*
Sorry about that if that's the case, you will need to reinstall WanGP.
There are two different ways to fix this issue while still preserving your data:
1) **Command Line**
If you have access to a terminal window :
```
cd installation_path_of_wangp
git fetch origin && git reset --hard origin/main
pip install -r requirements.txt
```

2) **Generic Method**
a) move outside the installation WanGP folder the folders **ckpts**, **settings**, **outputs** and all the **loras** folders and the file **wgp_config.json**
b) delete the WanGP folder and reinstall
c) move back what you moved in a)




See full changelog: **[Changelog](docs/CHANGELOG.md)**

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)

## 🚀 Quick Start

**One-click installation:** 
- Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.

- Use Redtash1 [One Click Install with Sage](https://github.com/Redtash1/Wan2GP-Windows-One-Click-Install-With-Sage)

**Manual installation:**
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

**Run the application:**
```bash
python wgp.py
```

First time using WanGP ? Just check the *Guides* tab, and you will find a selection of recommended models to use.

**Update the application:**
If using Pinokio use Pinokio to update otherwise:
Get in the directory where WanGP is installed and:
```bash
git pull
conda activate wan2gp
pip install -r requirements.txt
```

if you get some error messages related to git, you may try the following (beware this will overwrite local changes made to the source code of WanGP):
```bash
git fetch origin && git reset --hard origin/main
conda activate wan2gp
pip install -r requirements.txt
```

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
