# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Generative Models Accessible to the GPU Poor</b>
</p>

WanGP is a one-stop super app for the best open source generative models across video, image, audio, and text-to-speech.

## Highlights

| Modality | Supported models |
| --- | --- |
| **Video** | **Wan 2.1/2.2** and derived models, **LTX-2**, **Hunyuan Video 1/1.5**, **LongCat**, **Kandinsky**, **LTXV**, **MagiHuman** |
| **Image** | **Qwen Image**, **Z-Image**, **Flux 1/2** (Klein, Chroma), **HiDream** |
| **Audio / TTS** | **Qwen3 TTS**, **Ace Step 1/2/XL**, **Omnivoice**, **Index TTS2**, **KugelAudio**, **HearMula**, **Chatterbox** |

### Run More Models on More Hardware

- **Low VRAM requirements**: run select models with as little as **6 GB of VRAM**.
- **Older Nvidia GPU support**: use RTX 10XX, 20XX, and newer cards.
- **AMD GPU support**: run on RDNA 4, 3, 3.5, and 2 hardware; see the Installation section below.
- **Fast latest-GPU performance**: take advantage of modern GPU acceleration.
- **Full web interface**: generate, manage, and reuse outputs from an easy browser UI.
- **LoRA customization**: adapt each model with LoRAs, reuse LoRAs stored in another App.
- **Many quantized checkpoint formats**: use int8, fp8, gguf, NV FP4, and Nunchaku.
- **Architecture-aware downloads**: automatically fetch the model files suited to your hardware.
- **Finetunes**: add your own finetunes / checkpoints or the ones you found on Hugging Face or CivitAI
- **Generation queue**: line up videos, images, and audio jobs, then come back later.
- **Headless mode**: launch batches from the command line for images, videos, and audio.
- **WanGP API**: add generative capabilities to your own apps.

### Built-In Creation Tools

- **Video, image, and audio galleries**: browse generations and reuse them as new inputs.
- **Reusable settings**: extract settings from any generation, create templates, and share them.
- **Per-model prompt enhancer**: improve prompts with model-specific syntax and expectations.
- **Input preparation tools**: use the mask editor, background remover, pose/depth/flow extractors, speaker diarization, and background noise/song remover.
- **Deepy low-VRAM offline agent**: orchestrate generation jobs and tedious tasks such as transcription, video splitting, and color-frame generation while you are away.
- **Temporal and spatial upsampling**: improve outputs with RIFE, FlashVSR, and Lanczos.
- **Audio postprocessing**: generate soundtracks with MMAudio, replace voices with SeedVC, or remux a video with any soundtrack.
- **Ready-to-use plug-ins**: Gallery Browser, Motion Designer, Models/Checkpoints Manager, CivitAI browser and downloader, and more.

**Discord Server to get Help from the WanGP Community and show your Best Gens:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)


## 🔥 Latest Updates : 
### 1st of July 2026: WanGP v12.3, The VRAM Digger

- **Krea2 Lanpaint**: Krea2 can now do *inpainting* thanks to *Lanpaint*. To get the best results you will need to adjust the prompt and increase the number of Lanpaint steps.

- **Krea2 NAG**: WanGP exclusivity, *NAG* will allow you to define *Negative Prompts* with distilled models such as *Krea2 Turbo*

- **Gradio Optimizations**: thanks to numerous exclusive optimizations, Gradio UI should be faster (especially using the *Image Editor*) 

- **Chrome CPU Only Scripts**: you probably noticed that you Web Browser takes away VRAM just to display the UI. If you disable GPU Usage in Chrome for instance **you could save between 1GB of VRAM and 5GB of VRAM !!!**. The more VRAM capacity your GPU has the greater the gain (as Chrome tends to be greedier). I have added in the *Scripts* folder two scripts to disable GPU when using Chrome. WanGP has been optimized to still offer decent UI speed even if the web browser uses only the CPU. 

### 26th of June 2026: WanGP v12.278, Let's Experiment!

- **KREA-2** : new Image Generator model that claims to be the most aesthetic open-source image model available.

- **LTX-2.3 Multiple Subject Reference**: Here comes another way to add *Reference Images* when using LTX 2.3. This finetune combines Distilled 1.1 and a new LoRA from *LiconStudio*. Just provide 2 to 5 reference images; background first, then subjects and objects. Please note that the embedded lora is quite fond of character sheets with white background.

I added an experimental support for text to image, not sure it works as MSR doesnt seem to be made for that. 

- **LTX 2.3 Inpainting**: you will find this new *Inpainting* capability for LTX2 in the *Process List*. It is based on the set of *LoRAs* just released by the LTX Team. If you see glitches dont hesitate to expand the mask.

- **LTX 2.3 Ingredients**: part of the same new LoRAs collection the *Ingredients* process allows you to inject a character defined in a character sheet, preferably on a white background with black separator lines between individual pieces. Dont expect miracles with slidings windows or start frames.

- **Easy Frames Cap based on Control Video/Audio**: for supported models (*LTX2*, *Vace*) if you provide a control video or source audio you can ask WanGP either to stop when the control video / audio is done or continue until all the requested frames have been produced.

- **Ideograms v4 unlocked**: most hidden settings are now exposed (*mu*, *std*), you can change guidance half way, use a different scheduler. I added  resolutions used by Ideograms. Also please note that Ideograms runs two transformers in parallel *cond* and *uncond*. If you want to apply different *loras multipliers* to each transformer, use the new ":" separator, for instance with *1:1.2*, 1 will be applied to cond and 1.2 to uncond.

- **Ideograms v4 Turbo Time**: distilled version of Ideograms v4, from 4 steps to 8 steps and no guidance. 

- **Experimental Scail 2 Parallel Subwindows**: in order to reduce image degradations with long videos, I am experimenting a new concept: *Parallel Subwindows*, the idea is to work on a much larger Sliding Windows than usual (>200 frames) and to generate multiple sub windows (of 80 frames of so) in parallel. It is experimental, may end up a big fail and removed in next version, let me know...

- **Scail 2 Start Frame Fix**: you should no longer see a few bad frames at the beginning of the video in *Animate* mode. Many thanks to @pauldps that gave me part of the solution.

- **Scail 2 Experimental Multi References**: you can now provide different point of view of your character. This is an official feature but experimental.

- **PrismAudio**: this a *video to audio* processor, an alternative to *MMaudio*, quite good to add sound to an existing video. It requires a prompt. It is not made to generate spoken words.

- **More Plugins Types: Temporal Upsamplers / Audio Processors**: you can now add your own Temporal Upsampler (*Rife* alternative) or *Audio Processor* (*MMAudio* alternative). As a reminder the previous version allowed already to add a custom *Spatial Upsampler*.

- **API+, MCP+.**: I have improved the API capabilties (please check *docs/API.md*), and widened *MCP* support. Feel free to share feedback on Discord

- **Finetune Resolutions**: define custom resolutions directly in finetunes

*Update 12.25*: Ideograms v4 Turbo Time, MSR t2i, Scail2 parallel subwindows\
*Update 12.26*: LTX2 inpainting & ingredients, Easy cap, Scail2 fix\
*Update 12.27*: KREA2, Scail2 multiref


### 14th of June 2026: WanGP v12.22, Go with the Flow
- **Media Flow Plugin**: the *Full Video Process* is now named *Media Flow* because it can process *Images* as well as *Videos*. Even better, the new *Batch* mode can process any number of files: for instance, give *Media Flow* the path to the folder containing your collection of butterfly pictures and *all the corresponding images will be upsampled in one click*!

- **Scail 2**: the sequel to one of the best video *Character Animators*, and a very good alternative to *Wan 2.2 Animate*. You can either *Animate* up to 5 people by providing a *Start Image* and a *Control Video* that contains the movement, or *Replace* one person in an existing Control Video. Animate mode preserves identity well thanks to the new *Reference Image* input and, best of all, it supports *Sliding Windows* for non-stop dancing!

Please note that Scail 2 *Replace* and *Animate* modes require colored masks if more than two people are being replaced or animated. You can build them easily with *WanGP Magic Mask* (remember the magic wand icon). Also, for best results, I recommend using a *Reference Image* or a *Start Image* that is closely aligned to the first frame of the control video; you can use an *Image Model* generator for this.

Version *update 12.21* introduces RAM optimisations when using many *sliding windows* and added support for *Lora accelerator lightx2v 4 steps*

- **Int8 ConvRot Support**: model checkpoints saved in this quantized int8 format used by Comfy can now be loaded in WanGP.

- **LTX2 Image Generator (t2i)**: this one was always within grasp but required a little bit of packaging. Here we go we, just pick the *text to image* tab and use *LTX2* to use your favorite *Ic LoRAs* (outpainting, refiner, ...) on *images*. Best of all, the *LTX2 Image Processes* are available in the *Media Flow* Plugin.

- **Bernini 1.3B**: a much more gentle version (*lower VRAM requirements and faster*) for your GPU. Not as good as the 14GB version, but still produces some nice outputs.

- **Chain of Zoom Upsampler**: new upsampler that can magnify up to x16, quite good with hair and skin. However it expects low quality image so it may reinvent existing details. WanGP optimized: low VRAM and up to x4 times faster

- **Upsampler & Model Plugins**: PlugIn developers can now create plugins that add new *Spatial Upsampler* or new *Models*

As sample plugins, enjoy:
   - **Stable Diffusion 1.4**: the father of all image generators !
   - **Pixel Upsampler**: upsample by duplicating the same pixel for a grandiose Pixel Art effect !

*update 12.21*: Scail 2 RAM optim + lightx2v support, added LTX2 t2v & Bernini 1.3B\
*update 12.22*: Chain of Zoom Upsampler,  Upsampler & Model Plugins

### 7th of June 2026: WanGP v12.13, Prompt Control
- **Ideogram 4 Prompt Helper**: the great thing about Ideogram 4 is that you can position every object or text exactly where you expect it in your output image. Ideogram 4 now has a *Visual Helper* to create and edit its JSON prompt format. Click the *Magic Wand* next to the prompt to draw or resize text/object boxes, tune the main prompt fields, and apply the final JSON back to the prompt. *Magic Prompt* can still create the first draft for you.

- **JoyAI-Echo**: this new LTX-2.3 model is the closest thing to *SeeDance 2* that you may find in the open source world. It is an audio-video model for connected multi-window stories. JoyAI-Echo keeps compact memories between windows so later shots can reuse characters, voices, objects, and places. WanGP implementation of *JoyAI-Echo* goes well beyond the original implementation:
   - With the new *Sliding Window commands* (see below), you can extend existing *Sliding Windows*, *Create New Shots*, or *Continue a Video*.
   - The new memory command system (`[/store_mem]`, `[/load_mem]`, and `[/drop_mem]`) lets you pick which sliding windows can be reused for future memory and which ones should no longer be used. Please check the JoyAI-Echo *Prompt help* for the full syntax.
   - Use a *Control Video* to target audio/video segments in the *Joy Memory Positions* field and seed the first memories with characters and background. 
For instance *Joy Memory Positions* could be `man=4s,woman=12s`, if a man is speaking at around 4s and a woman at around 12s. The two memories can be used in later windows with the command [/load_mem=man] or  [/load_mem=woman] 

- **Sliding Window Commands**: thanks to new inline prompt commands (for instance `[/duration=...]`, `[/overlap=...]`, and `[/new_shot]`), you can now define a different duration, number of frames, or transition style on a per-window basis. You can also change the LoRAs multipliers of the current window with `[/loras_mult=1;0]`. See `docs/PROMPTS.md` for the full syntax and examples.

### 4th of June 2026: WanGP v12.00, The Journey Continues
- **PiD**: a new high quality x4 spatial upsampler for images by Nvidia. It is supposed to work with only Flux/Flux2 compatible models since it needs to plug directly to the VAE Decoder. However thanks to a simple trick it is available everywhere. Some automated Tiling may be triggered if you ask for very high out res. WanGP version is as usual ultra optimized and should require little VRAM even when tiling is not used.

- **Ideograms v4**: this image generator claims to be the best open source image generator. It consumes a special *Json Prompt Format* that WanGP *Prompt Enhancer* can produce for you. There is a snag though: occasionnaly, even a harmless prompt may trigger a *Safety Filter*. No way to get around this as it is hardcoded in the model weights.

- **Stable Audio 3**:  WanGP *Text To Speech* (TTS) collection of models is now completed with a model that can generate sounds, background music or special effects 

- **Bernini 14B**: the video model derived from Wan 2.2 is really incredible. You can ask it to modify the content of an existing video or to generate a new video with any number of *References Images*. and *it just works*. There is a price to pay though: to generate 81 frames, you will need 12 GB of VRAM for *v2v* / 16GB for *v2v + ref frames*. v2v  works quite well with Lora Accelerators such as *lightning 4 steps* . But as soon as you include reference frames, you will have to go for at least 15 steps with guidance and no lora accelerator. You are not allowed to complain, this model is advertised to work on a H100 and thanks to WanGP magic you can run it at home.

- **MCP Server & Agent Skills**: WanGP includes now a *MCP server* to make life much easier to your AI Agents. WanGP exposes also new discovery functions that can be queried by to agent to get the list of all generative models and features that are available.

### 1st of June 2026: WanGP v11.90, Everything will be fine...
**Finetune Creator / Editor**
*Create* a new *Finetune* (use an existing model with your own checkpoints), *Edit* or *Import* an existing Finetune in only one click directly from the *WanGP UI*. You can then share easily a finetune with other users by clicking the *Export* button.

Look for the new **+** in the *WanGP Tool Bar*.

The finetune creator allows you not only to customize an existing models with *Custom URLs* or *Local Paths* for both the main *Transformer files* & *Text encoders* but also to define *User help* and set *Custom System Templates* to be used with the finetune *Prompt Enhancer*.

Please check *docs/FINETUNE.md* doc for info about finetunes.

### 29th of May 2026: WanGP v11.88, Humans Accelerators
- **Create Hierarchies of Loras / Change Order of Loras**

- **WanGP Toolbar** with keyboard shortcuts:
    - **Search**: switch quickly to another model by just entering a few letters of its name
    - **Refresh Model List**: no longer needed to restart the app to add or modify a finetune
    - **Unload All**: free most of the RAM/VRAM used by WanGP

- **MOV/MKV Container Support**: beside *mp4* files you can now store you video gens in *mov* and *mkv* containers

- **ProRes422 & DNxHR HQ Video Codecs**: these professional video codecs have some fans out there

- **LTX-2 Guide**: click the "i" to the right of the model description to get tips / explanations on how to use LTX2 models

- **LTX2 Smearing Fix**: the smearing / ghosting is now mostly gone

- **Omnivoice Fix**: you will enjoy this fix unless you liked the gibberish generator of the previous version




See full changelog: **[Changelog](docs/CHANGELOG.md)**


## 🚀 Quick Start

### One-click Bat/SH Script Auto-installer:

The 1-click automated scripts for both **Windows (`.bat`)** and **Linux/macOS (`.sh`)** make installation, environment management, and updates as seamless as possible. These scripts will not only install WanGP but also best acceleration kernels (Triton, Sage, Flash, GGuf, Lightx2v, Nunchaku) available for your config.

*👉 **Windows Users:** Double-click the `.bat` files. **Linux Users:** Run the `.sh` files in your terminal.*

#### **1️⃣ Installation (`scripts\install.bat` | `scripts/install.sh`)**

**Choose Installation Type**
- **Auto Install**
- **Manual Install**

**Manual Install**

If you selected Manual Install, you will be guided through:

1. **Choose your package manager**
2. **Name your environment**
3. **Select your Install Mode**

#### 2️⃣ Starting the App (`scripts\run.bat` | `scripts/run.sh`)
Once installed, use this script to launch the application. It runs WAN2GP using your active environment.

*   **⚙️ Customizing Launch Arguments (`args.txt`)**
    *   If you want to pass extra command-line flags to the launcher (like enabling advanced UI features or automatically opening your browser), create an `args.txt` file in your `scripts` folder.
    *   **Example `args.txt`:**
        ```text
        --advanced --open-browser
        ```

#### 3️⃣ Updating & Upgrading (`scripts\update.bat` | `scripts/update.sh`)
Use this script to get the latest updates for WAN2GP and upgrade dependencies.
* **1. Update:** Fetches the latest code from GitHub and updates requirements.
* **2. Upgrade:** Allows you to manually individually upgrade heavy backend components (like PyTorch, Triton, Sage Attention).

#### 4️⃣ Managing Environments (`scripts\manage.bat` | `scripts/manage.sh`)
Use this script to manage and switch between your sandboxed environments safely.

* **Example Scenario 1: Migrating an Existing Setup**
    * If you have a folder named `venv` that works perfectly and want to use it with the new one-click scripts, run `manage.bat` and select **Add Existing Environment**.
    * Copy-paste the folder path (e.g., `C:\WAN2GP\venv`), select type `venv`, then use **Set Active Environment** to make it the default. Now `run.bat` and `update.bat` will target your existing setup.

* **Example Scenario 2: Testing New Configurations**
    * Let's say you have an environment named `env_stable` that works perfectly, but you want to try the new "Use Latest" combo. Instead of risking your working setup, run `install.bat`, create a *new* environment called `env_testing`, and select **Use Latest**.
    * If the testing environment breaks, simply open `manage.bat`, select **Set Active Environment**, and switch back to `env_stable`. You are back up and running instantly.

---

### One-click (Pinokio) installer:

Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.

---


### Manual installation: (for RTX20xx - RTX50xx)

```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### Manual installation: (for GTX 10xx)

```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

#### Run the application:

```bash
python wgp.py
```
If you are low on VRAM, there is a trick to increase the amount of VRAM available (between 1GB and 5GB of VRAM to be gained depending on the GPU): *disable GPU Usage in your Web Browser*.

Run *scripts/start-chrome-no-gpu.bat* or *scripts/start-chrome-no-gpu.sh* to launch Chrome without using your GPU. 

First time using WanGP ? Just check the *Guides* tab, and you will find a selection of recommended models to use.

#### Update the application (stay in the current python / pytorch version):
If using Pinokio use Pinokio to update otherwise:
Get in the directory where WanGP is installed and:
```bash
git pull
conda activate wan2gp
pip install -r requirements.txt
```

#### Upgrade from Python 3.10, Pytorch 2.7.1, Cuda 12.8 to Python 3.11, Pytorch 2.10, Cuda 13/13.1 (for non GTX10xx users)
I recommend renaming first the old conda environment to avoid bad surprises when installing a different config in this old environment.

```bash
conda rename -n wan2gp  old_wan2gp
```

Get in the directory where WanGP is installed and:
```bash
git pull
conda create -n wan2gp python=3.11.9
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Once you are done you will have to reinstall *Sage Attention*, *Triton*, *Flash Attention*. Check the **[Installation Guide](docs/INSTALLATION.md)** -

if you get some error messages related to git, you may try the following (beware this will overwrite local changes made to the source code of WanGP):
```bash
git fetch origin && git reset --hard origin/main
conda activate wan2gp
pip install -r requirements.txt
```
When you have the confirmation it works well you can then delete the old conda env:
```bash
conda uninstall -n old_wan2gp --all  
```

#### Run headless (batch processing):

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
- **[Processing Guide](docs/PROCESSING.md)** - Preprocessing, masks, sliding windows, and postprocessing
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
