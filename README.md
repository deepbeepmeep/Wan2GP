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

## 🔥 Latest Updates : 
### August 24 2025: WanGP v8.1 -  the RAM Liberator

- Reserved RAM entirely freed when switching models, you should get much less out of memory related to RAM. I have also added a button in *Configuration / Performance* that will release most of the RAM used by WanGP if you want to use another application without quitting WanGP 
- InfiniteTalk support: improved version of Multitalk that supposedly supports very long video generations based on an audio track. Exists in two flavors (*Single Speaker* and *Multi Speakers*) but doesnt seem to be compatible with Vace. One key new feature compared to Multitalk is that you can have different visual shots associated to the same audio: each Reference frame you provide you will be associated to a new Sliding Window. If only Reference frame is provided, it will be used for all windows. When Continuing a video, you can either continue the current shot (no Reference Frame) or add new shots (one or more Reference Frames).\
If you are not into audio, you can use still this model to generate infinite long image2video, just select "no speaker". Last but not least, Infinitetalk works works with all the Loras accelerators.
- Flux Chroma 1 HD support: uncensored flux based model and lighter than Flux (8.9B versus 12B) and can fit entirely in VRAM with only 16 GB of VRAM. Unfortunalely it is not distilled and you will need CFG at minimum 20 steps

### August 21 2025: WanGP v8.01 - the killer of seven

- Qwen Image Edit : Flux Kontext challenger (prompt driven image edition). Best results (including Identity preservation) will be obtained at 720p. Beyond you may get image outpainting and / or lose identity preservation. Below 720p prompt adherence will be worse. Qwen Image Edit works with Qwen Lora Lightning 4 steps. I have also unlocked all the resolutions for Qwen models. Bonus Zone: support for multiple image compositions but identity preservation won't be as good.
- On demand Prompt Enhancer (needs to be enabled in Configuration Tab) that you can use to Enhance a Text Prompt before starting a Generation. You can refine the Enhanced Prompt or change the original Prompt.
- Choice of a non censored Prompt Enhancer. Beware this is one is VRAM hungry and will require 12 GB of VRAM to work
- Memory Profile customizable per model : useful to set for instance Profile 3 (preload the model entirely in VRAM) with only Image Generation models, if you have 24 GB of VRAM. In that case Generation will be much faster because with Image generators (contrary to Video generators) as a lot of time is wasted in offloading 
- Expert Guidance Mode: change the Guidance during the generation up to 2 times. Very useful with Wan 2.2 Ligthning to reduce the slow motion effect. The idea is to insert a CFG phase before the 2 accelerated phases that follow and have no Guidance. I have added the finetune *Wan2.2 Vace Lightning 3 Phases 14B* with a prebuilt configuration. Please note that it is a 8 steps process although the lora lightning is 4 steps. This expert guidance mode is also available with Wan 2.1.

*WanGP 8.01 update, improved Qwen Image Edit Identity Preservation*
### August 12 2025: WanGP v7.7777 - Lucky Day(s)

This is your lucky day ! thanks to new configuration options that will let you store generated Videos and Images in lossless compressed formats, you will find they in fact they look two times better without doing anything !

Just kidding, they will be only marginally better, but at least this opens the way to professionnal editing.

Support:
- Video: x264, x264 lossless, x265
- Images: jpeg, png, webp, wbp lossless
Generation Settings are stored in each of the above regardless of the format (that was the hard part).

Also you can now choose different output directories for images and videos.

unexpected luck: fixed lightning 8 steps for Qwen, and lightning 4 steps for Wan 2.2, now you just need 1x multiplier no weird numbers. 
*update 7.777 : oops got a crash a with FastWan ? Luck comes and goes, try a new update, maybe you will have a better chance this time*
*update 7.7777 : Sometime good luck seems to last forever. For instance what if Qwen Lightning 4 steps could also work with WanGP ?*
- https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors (Qwen Lightning 4 steps)
- https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors (new improved version of Qwen Lightning 8 steps)


### August 10 2025: WanGP v7.76 - Faster than the VAE ...
We have a funny one here today: FastWan 2.2 5B, the Fastest Video Generator, only 20s to generate 121 frames at 720p. The snag is that VAE is twice as slow... 
Thanks to Kijai for extracting the Lora that is used to build the corresponding finetune.

*WanGP 7.76: fixed the messed up I did to i2v models (loras path was wrong for Wan2.2 and Clip broken)*

### August 9 2025: WanGP v7.74 - Qwen Rebirth part 2
Added support for Qwen Lightning lora for a 8 steps generation (https://huggingface.co/lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Lightning-8steps-V1.0.safetensors). Lora is not normalized and you can use a multiplier around 0.1.

Mag Cache support for all the Wan2.2 models Don't forget to set guidance to 1 and 8 denoising steps , your gen will be 7x faster !

### August 8 2025: WanGP v7.73 - Qwen Rebirth
Ever wondered what impact not using Guidance has on a model that expects it ? Just look at Qween Image in WanGP 7.71 whose outputs were erratic. Somehow I had convinced myself that Qwen was a distilled model. In fact Qwen was dying for a negative prompt. And in WanGP 7.72 there is at last one for him.

As Qwen is not so picky after all I have added also quantized text encoder which reduces the RAM requirements of Qwen by 10 GB (the text encoder quantized version produced garbage before) 

Unfortunately still the Sage bug for older GPU architectures. Added Sdpa fallback for these architectures.

*7.73 update: still Sage / Sage2 bug for GPUs before RTX40xx. I have added a detection mechanism that forces Sdpa attention if that's the case*


### August 6 2025: WanGP v7.71 - Picky, picky

This release comes with two new models :
- Qwen Image: a Commercial grade Image generator capable to inject full sentences in the generated Image while still offering incredible visuals
- Wan 2.2 TextImage to Video 5B: the last Wan 2.2 needed if you want to complete your Wan 2.2 collection (loras for this folder can be stored in "\loras\5B"     )

There is catch though, they are very picky if you want to get good generations: first they both need lots of steps (50 ?) to show what they have to offer. Then for Qwen Image I had to hardcode the supported resolutions, because if you try anything else, you will get garbage. Likewise Wan 2.2 5B will remind you of Wan 1.0 if you don't ask for at least  720p. 

*7.71 update: Added VAE Tiling for both Qwen Image and Wan 2.2 TextImage to Video 5B, for low VRAM during a whole gen.*


### August 4 2025: WanGP v7.6 - Remuxed

With this new version you won't have any excuse if there is no sound in your video.

*Continue Video* now works with any video that has already some sound (hint: Multitalk ).

Also, on top of MMaudio and the various sound driven models I have added the ability to use your own soundtrack.

As a result you can apply a different sound source on each new video segment when doing a *Continue Video*. 

For instance:
- first video part: use Multitalk with two people speaking
- second video part: you apply your own soundtrack which will gently follow the multitalk conversation
- third video part: you use Vace effect and its corresponding control audio will be concatenated to the rest of the audio

To multiply the combinations I have also implemented *Continue Video* with the various image2video models.

Also:
- End Frame support added for LTX Video models
- Loras can now be targetted specifically at the High noise or Low noise models with Wan 2.2, check the Loras and Finetune guides
- Flux Krea Dev support

### July 30 2025: WanGP v7.5:  Just another release ... Wan 2.2 part 2
Here is now Wan 2.2 image2video a very good model if you want to set Start and End frames. Two Wan 2.2 models delivered, only one to go ...

Please note that although it is an image2video model it is structurally very close to Wan 2.2 text2video (same layers with only a different initial projection). Given that Wan 2.1 image2video loras don't work too well (half of their tensors are not supported), I have decided that this model will look for its loras in the text2video loras folder instead of the image2video folder.

I have also optimized RAM management with Wan 2.2 so that loras and modules will be loaded only once in RAM and Reserved RAM, this saves up to 5 GB of RAM which can make a difference...

And this time I really removed Vace Cocktail Light which gave a blurry vision.

### July 29 2025: WanGP v7.4:  Just another release ... Wan 2.2 Preview
Wan 2.2 is here.  The good news is that WanGP wont require a single byte of extra VRAM to run it and it will be as fast as Wan 2.1. The bad news is that you will need much more RAM if you want to leverage entirely this new model since it has twice has many parameters.

So here is a preview version of Wan 2.2 that is without the 5B model and Wan 2.2 image to video for the moment.

However as I felt bad to deliver only half of the wares, I gave you instead .....** Wan 2.2 Vace Experimental Cocktail** !

Very good surprise indeed, the loras and Vace partially work with Wan 2.2. We will need to wait for the official Vace 2.2 release since some Vace features are broken like identity preservation

Bonus zone: Flux multi images conditions has been added, or maybe not if I broke everything as I have been distracted by Wan...

7.4 update: I forgot to update the version number. I also removed Vace Cocktail light which didnt work well.

### July 27 2025: WanGP v7.3 : Interlude
While waiting for Wan 2.2, you will appreciate the model selection hierarchy which is very useful to collect even more models. You will also appreciate that WanGP remembers which model you used last in each model family.

### July 26 2025: WanGP v7.2 : Ode to Vace
I am really convinced that Vace can do everything the other models can do and in a better way especially as Vace can be combined with Multitalk.

Here are some new Vace improvements:
- I have provided a default finetune named *Vace Cocktail*  which is a model created on the fly using the Wan text 2 video model and the Loras used to build FusioniX. The weight of the *Detail Enhancer* Lora has been reduced to improve identity preservation. Copy the model definition in *defaults/vace_14B_cocktail.json* in the *finetunes/* folder to change the Cocktail composition. Cocktail contains already some Loras acccelerators so no need to add on top a Lora Accvid, Causvid or Fusionix, ... . The whole point of Cocktail is to be able  to build you own FusioniX (which originally is a combination of 4 loras) but without the inconvenient of FusioniX.
- Talking about identity preservation, it tends to go away when one generates a single Frame instead of a Video which is shame for our Vace photoshop. But there is a solution : I have added an Advanced Quality option, that tells WanGP to generate a little more than a frame (it will still keep only the first frame). It will be a little slower but you will be amazed how Vace Cocktail combined with this option will preserve identities (bye bye *Phantom*). 
- As in practise I have observed one switches frequently between *Vace text2video* and *Vace text2image* I have put them in the same place they are now just one tab away, no need to reload the model. Likewise *Wan text2video* and *Wan tex2image* have been merged.
- Color fixing when using Sliding Windows. A new postprocessing *Color Correction* applied automatically by default (you can disable it in the *Advanced tab Sliding Window*) will try to match the colors of the new window with that of the previous window. It doesnt fix all the unwanted artifacts of the new window but at least this makes the transition smoother. Thanks to the multitalk team for the original code.

Also you will enjoy our new real time statistics (CPU / GPU usage, RAM / VRAM used, ... ). Many thanks to **Redtash1** for providing the framework for this new feature ! You need to go in the Config tab to enable real time stats.


### July 21 2025: WanGP v7.12
- Flux Family Reunion : *Flux Dev* and *Flux Schnell* have been invited aboard WanGP. To celebrate that, Loras support for the Flux *diffusers* format has also been added.

- LTX Video upgraded to version 0.9.8: you can now generate 1800 frames (1 min of video !) in one go without a sliding window. With the distilled model it will take only 5 minutes with a RTX 4090 (you will need 22 GB of VRAM though). I have added options to select higher humber frames if you want to experiment (go to Configuration Tab / General / Increase the Max Number of Frames, change the value and restart the App)

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
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

**Run the application:**
```bash
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
