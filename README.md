# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor</b>
</p>

WanGP supports the Wan (and derived models), Hunyuan Video and LTV Video models with:
- Low VRAM requirements (as low as 6 GB of VRAM is sufficient for certain models)
- Support for old GPUs (RTX 10XX, 20xx, ...)
- Very Fast on the latest GPUs
- Easy to use Full Web based interface
- Auto download of the required model adapted to your specific architecture
- Tools integrated to facilitate Video Generation : Mask Editor, Prompt Enhancer, Temporal and Spatial Generation
- Loras Support to customize each model
- Queuing system : make your shopping list of videos to generate and come back later 

**Discord Server to get Help from Other Users and show your Best Videos:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

## 🔥 Latest Updates
### May 28 2025: WanGP v5.41
👋 Bonus release: Support for **AccVideo** Lora to speed up x2 Video generations in Wan models. Check the Loras documentation to get the usage instructions of AccVideo.\
You will need to do a *pip install -r requirements.txt*

### May 28 2025: WanGP v5.4
👋 World Exclusive : **Hunyuan Video Avatar** Support ! You won't need 80 GB of VRAM nor 32 GB oF VRAM, just 10 GB of VRAM will be sufficient to generate up to 15s of high quality speech / song driven Video at a high speed with no quality degradation. Support for TeaCache included.\
Here is a link to the original repo where you will find some very interesting documentation and examples. https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar. Kudos to the Hunyuan Video Avatar team for the best model of its kind.\
Also many thanks to Reevoy24 for his repackaging / completing the documentation

### May 28 2025: WanGP v5.31
👋 Added **Phantom 14B**, a model that you can use to transfer objects / people in the video. My preference goes to Vace that remains the king of controlnets.
VACE improvements: Better sliding window transitions, image mask support in Matanyone, new Extend Video feature, and enhanced background removal options.

### May 26, 2025: WanGP v5.3
👋 Settings management revolution! Now you can:
- Select any generated video and click *Use Selected Video Settings* to instantly reuse its configuration
- Drag & drop videos to automatically extract their settings metadata
- Export/import settings as JSON files for easy sharing and backup

### May 20, 2025: WanGP v5.2
👋 **CausVid support** - Generate videos in just 4-12 steps with the new distilled Wan model! Also added experimental MoviiGen for 1080p generation (20GB+ VRAM required). Check the Loras documentation to get the usage instructions of CausVid.

### May 18, 2025: WanGP v5.1
👋 **LTX Video 13B Distilled** - Generate high-quality videos in less than one minute!

### May 17, 2025: WanGP v5.0
👋 **One App to Rule Them All!** Added Hunyuan Video and LTX Video support, plus Vace 14B and integrated prompt enhancer.

See full changelog: **[Changelog](docs/CHANGELOG.md)**

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)

## 🚀 Quick Start

**One-click installation:** Get started instantly with [Pinokio App](https://pinokio.computer/)

**Manual installation:**
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
```

**Run the application:**
```bash
python wgp.py  # Text-to-video (default)
python wgp.py --i2v  # Image-to-video
```

## 📦 Installation

For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions for RTX 10XX to RTX 50XX

## 🎯 Usage

### Basic Usage
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - First steps and basic usage
- **[Models Overview](docs/MODELS.md)** - Available models and their capabilities

### Advanced Features
- **[Loras Guide](docs/LORAS.md)** - Using and managing Loras for customization
- **[VACE ControlNet](docs/VACE.md)** - Advanced video control and manipulation
- **[Command Line Reference](docs/CLI.md)** - All available command line options

## 📚 Documentation

- **[Changelog](docs/CHANGELOG.md)** - Latest updates and version history
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

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
