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
- Tools integrated to facilitate Video Generation : Mask Editor, Prompt Enhancer, Temporal and Spatial Generation, MMAudio, Video Browser, Pose / Depth / Flow extractor
- Loras Support to customize each model
- Queuing system : make your shopping list of videos to generate and come back later 

**Discord Server to get Help from Other Users and show your Best Videos:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

## 🔥 Latest Updates
### July 21 2025: WanGP v7.12
- Flux Family Reunion : *Flux Dev* and *Flux Schnell* have been invited aboard WanGP. To celebrate that, Loras support for the Flux *diffusers* format has also been added.

- LTX Video upgraded to version 0.9.8: you can now generate 1800 frames (1 min of video !) in one go without a sliding window. With the distilled model it will take only 5 minutes with a RTX 4090 (you will need 22 GB of VRAM though). I have added options to select higher humber frames if you want to experiment

- LTX Video ControlNet : it is a Control Net that allows you for instance to transfer a Human motion or Depth from a control video. It is not as powerful as Vace but can produce interesting things especially as now you can generate quickly a 1 min video. Under the scene IC-Loras (see below) for Pose, Depth and Canny are automatically loaded for you, no need to add them. 

- LTX IC-Lora support: these are special Loras that consumes a conditional image or video
Beside the pose, depth and canny IC-Loras transparently loaded there is the *detailer* (https://huggingface.co/Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8) which is basically an upsampler. Add the *detailer* as a Lora and use LTX Raw Format as control net choice to use it.

- Matanyone is now also for the GPU Poor as its VRAM requirements have been divided by 2! (7.12 shadow update)

- Easier way to select video resolution 


### July 15 2025: WanGP v7.0 is an AI Powered Photoshop
This release turns the Wan models into Image Generators. This goes way more than allowing to generate a video made of single frame :
- Multiple Images generated at the same time so that you can choose the one you like best.It is Highly VRAM optimized so that you can generate for instance 4 720p Images at the same time with less than 10 GB
- With the *image2image* the original text2video WanGP becomes an image upsampler / restorer
- *Vace image2image* comes out of the box with image outpainting, person / object replacement, ...
- You can use in one click a newly Image generated as Start Image or Reference Image for a Video generation

And to complete the full suite of AI Image Generators, Ladies and Gentlemen please welcome for the first time in WanGP : **Flux Kontext**.\
As a reminder Flux Kontext is an image editor : give it an image and a prompt and it will do the change for you.\
This highly optimized version of Flux Kontext will make you feel that you have been cheated all this time as WanGP Flux Kontext requires only 8 GB of VRAM to generate 4 images at the same time with no need for quantization.

WanGP v7 comes with *Image2image* vanilla and *Vace FusinoniX*. However you can build your own finetune where you will combine a text2video or Vace model with any combination of Loras.

Also in the news:
- You can now enter the *Bbox* for each speaker in *Multitalk* to precisely locate who is speaking. And to save some headaches the *Image Mask generator* will give you the *Bbox* coordinates of an area you have selected.
- *Film Grain* post processing to add a vintage look at your video
- *First Last Frame to Video* model should work much better now as I have discovered rencently its implementation was not complete
- More power for the finetuners, you can now embed Loras directly in the finetune definition. You can also override the default models (titles, visibility, ...) with your own finetunes. Check the doc that has been updated.


### July 10 2025: WanGP v6.7, is NAG a game changer ? you tell me
Maybe you knew that already but most *Loras accelerators* we use today (Causvid, FusioniX) don't use *Guidance* at all (that it is *CFG* is set to 1). This helps to get much faster generations but the downside is that *Negative Prompts* are completely ignored (including the default ones set by the models). **NAG** (https://github.com/ChenDarYen/Normalized-Attention-Guidance) aims to solve that by injecting the *Negative Prompt* during the *attention* processing phase.

So WanGP 6.7 gives you NAG, but not any NAG, a *Low VRAM* implementation, the default one ends being VRAM greedy. You will find NAG in the *General* advanced tab for most Wan models. 

Use NAG especially when Guidance is set to 1. To turn it on set the **NAG scale** to something around 10. There are other NAG parameters **NAG tau** and **NAG alpha** which I recommend to change only if you don't get good results by just playing with the NAG scale. Don't hesitate to share on this discord server the best combinations for these 3 parameters.

The authors of NAG claim that NAG can also be used when using a Guidance (CFG > 1) and to improve the prompt adherence.

### July 8 2025: WanGP v6.6, WanGP offers you **Vace Multitalk Dual Voices Fusionix Infinite** :
**Vace** our beloved super Control Net has been combined with **Multitalk** the new king in town that can animate up to two people speaking (**Dual Voices**). It is accelerated by the **Fusionix** model and thanks to *Sliding Windows* support and *Adaptive Projected Guidance* (much slower but should reduce the reddish effect with long videos) your two people will be able to talk for very a long time (which is an **Infinite** amount of time in the field of video generation).

Of course you will get as well *Multitalk* vanilla and also *Multitalk 720p* as a bonus.

And since I am mister nice guy I have enclosed as an exclusivity an *Audio Separator* that will save you time to isolate each voice when using Multitalk with two people.

As I feel like resting a bit I haven't produced yet a nice sample Video to illustrate all these new capabilities. But here is the thing, I ams sure you will publish in the *Share Your Best Video* channel your *Master Pieces*. The best ones will be added to the *Announcements Channel* and will bring eternal fame to its authors.

But wait, there is more:
- Sliding Windows support has been added anywhere with Wan models, so imagine with text2video recently upgraded in 6.5 into a video2video, you can now upsample very long videos regardless of your VRAM. The good old image2video model can now reuse the last image to produce new videos (as requested by many of you)
- I have added also the capability to transfer the audio of the original control video (Misc. advanced tab) and an option to preserve the fps into the generated video, so from now on you will be to upsample / restore your old families video and keep the audio at their original pace. Be aware that the duration will be limited to 1000 frames as I still need to add streaming support for unlimited video sizes.

Also, of interest too:
- Extract video info from Videos that have not been generated by WanGP, even better you can also apply post processing (Upsampling / MMAudio) on non WanGP videos
- Force the generated video fps to your liking, works wery well with Vace when using a Control Video
- Ability to chain URLs of Finetune models (for instance put the URLs of a model in your main finetune and reference this finetune in other finetune models to save time)

### July 2 2025: WanGP v6.5.1, WanGP takes care of you: lots of quality of life features:
- View directly inside WanGP the properties (seed, resolutions, length, most settings...) of the past generations
- In one click use the newly generated video as a Control Video or Source Video to be continued 
- Manage multiple settings for the same model and switch between them using a dropdown box 
- WanGP will keep the last generated videos in the Gallery and will remember the last model you used if you restart the app but kept the Web page open
- Custom resolutions : add a file in the WanGP folder with the list of resolutions you want to see in WanGP (look at the instruction readme in this folder)

Taking care of your life is not enough, you want new stuff to play with ?
- MMAudio directly inside WanGP : add an audio soundtrack that matches the content of your video. By the way it is a low VRAM MMAudio and 6 GB of VRAM should be sufficient. You will need to go in the *Extensions* tab of the WanGP *Configuration* to enable MMAudio
- Forgot to upsample your video during the generation ? want to try another MMAudio variation ? Fear not you can also apply upsampling or add an MMAudio track once the video generation is done. Even better you can ask WangGP for multiple variations of MMAudio to pick the one you like best
- MagCache support: a new step skipping approach, supposed to be better than TeaCache. Makes a difference if you usually generate with a high number of steps
- SageAttention2++ support : not just the compatibility but also a slightly reduced VRAM usage
- Video2Video in Wan Text2Video : this is the paradox, a text2video can become a video2video if you start the denoising process later on an existing video
- FusioniX upsampler: this is an illustration of Video2Video in Text2Video. Use the FusioniX text2video model with an output resolution of 1080p and a denoising strength of 0.25 and you will get one of the best upsamplers (in only 2/3 steps, you will need lots of VRAM though). Increase the denoising strength and you will get one of the best Video Restorer
- Choice of Wan Samplers / Schedulers
- More Lora formats support

**If you had upgraded to v6.5 please upgrade again to 6.5.1 as this will fix a bug that ignored Loras beyond the first one**

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
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
```

**Run the application:**
```bash
# If you are in mainland China and cannot download model directly, try prepend HF_ENDPOINT=https://hf-mirror.com
python wgp.py  # Text-to-video (default)
python wgp.py --i2v  # Image-to-video
```

**Update the application:**
If using Pinokio use Pinokio to update otherwise:
Get in the directory where WanGP is installed and:
```bash
git pull
pip install -r requirements.txt
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
