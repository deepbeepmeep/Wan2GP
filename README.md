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
### October 26 2025: WanGP v9.21, Why isn't all my VRAM used ?

*WanGP exclusive*

**Wan 2.2 Ovi 10 GB** for all the GPU Poors of the World: *only 6 GB of VRAM to generate 121 frames at 720p*. With 16 GB of VRAM, you may even be able to load all the model in VRAM with *Memory Profile 3*

*update v9.21*: Got FastWan to work with Ovi: it is now 10 times faster ! (not including the VAE)

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

## 🔥 Latest Updates : 
### October 6 2025: WanGP v8.999 - A few last things before the Big Unknown ...

This new version hasn't any new model...

...but temptation to upgrade will be high as it contains a few Loras related features that may change your Life:
- **Ready to use Loras Accelerators Profiles** per type of model that you can apply on your current *Generation Settings*. Next time I will recommend a *Lora Accelerator*, it will be only one click away. And best of all of the required Loras will be downloaded automatically. When you apply an *Accelerator Profile*, input fields like the *Number of Denoising Steps* *Activated Loras*, *Loras Multipliers* (such as "1;0 0;1" ...) will be automatically filled. However your video specific fields will be preserved, so it will be easy to switch between Profiles to experiment. With *WanGP 8.993*, the *Accelerator Loras* are now merged with *Non Accelerator Loras". Things are getting too easy...

- **Embedded Loras URL** : WanGP will now try to remember every Lora URLs it sees. For instance if someone sends you some settings that contain Loras URLs or you extract the Settings of Video generated by a friend with Loras URLs, these URLs will be automatically added to *WanGP URL Cache*. Conversely everything you will share (Videos, Settings, Lset files) will contain the download URLs if they are known. You can also download directly a Lora in WanGP by using the *Download Lora* button a the bottom. The Lora will be immediatly available and added to WanGP lora URL cache. This will work with *Hugging Face* as a repository. Support for CivitAi will come as soon as someone will nice enough to post a GitHub PR ...

- **.lset file** supports embedded Loras URLs. It has never been easier to share a Lora with a friend. As a reminder a .lset file can be created directly from *WanGP Web Interface* and it contains a list of Loras and their multipliers, a Prompt and Instructions how to use these loras (like the Lora's *Trigger*). So with embedded Loras URL, you can send an .lset file by email or share it on discord: it is just a 1 KB tiny text, but with it other people will be able to use Gigabytes Loras as these will be automatically downloaded. 

I have created the new Discord Channel **share-your-settings** where you can post your *Settings* or *Lset files*. I will be pleased to add new Loras Accelerators in the list of WanGP *Accelerators Profiles if you post some good ones there. 

*With the 8.993 update*, I have added support for **Scaled FP8 format**. As a sample case, I have created finetunes for the **Wan 2.2 PalinGenesis** Finetune which is quite popular recently. You will find it in 3 flavors : *t2v*, *i2v* and *Lightning Accelerated for t2v*.

The *Scaled FP8 format* is widely used as it the format used by ... *ComfyUI*. So I except a flood of Finetunes in the *share-your-finetune* channel. If not it means this feature was useless and I will remove it &#x1F608;&#x1F608;&#x1F608;

Not enough Space left on your SSD to download more models ? Would like to reuse Scaled FP8 files in your ComfyUI Folder without duplicating them ? Here comes *WanGP 8.994* **Multiple Checkpoints Folders** : you just need to move the files into different folders / hard drives or reuse existing folders and let know WanGP about it in the *Config Tab* and WanGP will be able to put all the parts together.   

Last but not least the Lora's documentation has been updated.

*update 8.991*: full power of *Vace Lynx* unleashed with new combinations such as Landscape + Face / Clothes + Face  / Injectd Frame (Start/End frames/...) + Face
*update 8.992*: optimized gen with Lora, should be 10% faster if many loras
*update 8.993*: Support for *Scaled FP8* format and samples *Paligenesis* finetunes, merged Loras Accelerators and Non Accelerators
*update 8.994*: Added custom checkpoints folders
*update 8.999*: fixed a lora + fp8 bug and version sync for the jump to the unknown 

### September 30 2025: WanGP v8.9 - Combinatorics

This new version of WanGP introduces **Wan 2.1 Lynx** the best Control Net so far to transfer *Facial Identity*. You will be amazed to recognize your friends even with a completely different hair style. Congrats to the *Byte Dance team* for this achievement. Lynx works quite with well *Fusionix t2v* 10 steps.

*WanGP 8.9* also illustrate how existing WanGP features can be easily combined with new models. For instance with *Lynx* you will get out of the box *Video to Video* and *Image/Text to Image*.

Another fun combination is *Vace* + *Lynx*, which works much better than *Vace StandIn*. I have added sliders to change the weight of Vace & Lynx to allow you to tune the effects.



See full changelog: **[Changelog](docs/CHANGELOG.md)**

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)

## 🚀 Quick Start

**One-click installation:** 
- Get started instantly with [Pinokio App](https://pinokio.computer/)
- Use Redtash1 [One Click Install with Sage](https://github.com/Redtash1/Wan2GP-Windows-One-Click-Install-With-Sage)

**Manual installation:**
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

**Run the application:**
```bash
python wgp.py
```

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
