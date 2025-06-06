# Changelog

## 🔥 Latest News
### May 28 2025: WanGP v5.41
👋 Bonus release: Support for **AccVideo** Lora to speed up x2 Video generations in Wan models. Check the Loras documentation to get the usage instructions of AccVideo.

### May 28 2025: WanGP v5.4
👋 World Exclusive : Hunyuan Video Avatar Support ! You won't need 80 GB of VRAM nor 32 GB oF VRAM, just 10 GB of VRAM will be sufficient to generate up to 15s of high quality speech / song driven Video at a high speed with no quality degradation. Support for TeaCache included.

### May 26, 2025: WanGP v5.3
👋 Happy with a Video generation and want to do more generations using the same settings but you can't remember what you did or you find it too hard to copy/paste one per one each setting from the file metadata? Rejoice! There are now multiple ways to turn this tedious process into a one click task:
- Select one Video recently generated in the Video Gallery and click *Use Selected Video Settings*
- Click *Drop File Here* and select a Video you saved somewhere, if the settings metadata have been saved with the Video you will be able to extract them automatically
- Click *Export Settings to File* to save on your harddrive the current settings. You will be able to use them later again by clicking *Drop File Here* and select this time a Settings json file

### May 23, 2025: WanGP v5.21
👋 Improvements for Vace: better transitions between Sliding Windows, Support for Image masks in Matanyone, new Extend Video for Vace, different types of automated background removal

### May 20, 2025: WanGP v5.2
👋 Added support for Wan CausVid which is a distilled Wan model that can generate nice looking videos in only 4 to 12 steps. The great thing is that Kijai (Kudos to him!) has created a CausVid Lora that can be combined with any existing Wan t2v model 14B like Wan Vace 14B. See [LORAS.md](LORAS.md) for instructions on how to use CausVid.

Also as an experiment I have added support for the MoviiGen, the first model that claims to be capable of generating 1080p videos (if you have enough VRAM (20GB...) and be ready to wait for a long time...). Don't hesitate to share your impressions on the Discord server.

### May 18, 2025: WanGP v5.1
👋 Bonus Day, added LTX Video 13B Distilled: generate in less than one minute, very high quality Videos!

### May 17, 2025: WanGP v5.0
👋 One App to Rule Them All! Added support for the other great open source architectures:
- **Hunyuan Video**: text 2 video (one of the best, if not the best t2v), image 2 video and the recently released Hunyuan Custom (very good identity preservation when injecting a person into a video)
- **LTX Video 13B** (released last week): very long video support and fast 720p generation. Wan GP version has been greatly optimized and reduced LTX Video VRAM requirements by 4!

Also:
- Added support for the best Control Video Model, released 2 days ago: Vace 14B
- New Integrated prompt enhancer to increase the quality of the generated videos

*You will need one more `pip install -r requirements.txt`*

### May 5, 2025: WanGP v4.5
👋 FantasySpeaking model, you can animate a talking head using a voice track. This works not only on people but also on objects. Also better seamless transitions between Vace sliding windows for very long videos. New high quality processing features (mixed 16/32 bits calculation and 32 bits VAE)

### April 27, 2025: WanGP v4.4
👋 Phantom model support, very good model to transfer people or objects into video, works quite well at 720p and with the number of steps > 30

### April 25, 2025: WanGP v4.3
👋 Added preview mode and support for Sky Reels v2 Diffusion Forcing for high quality "infinite length videos". Note that Skyreel uses causal attention that is only supported by Sdpa attention so even if you choose another type of attention, some of the processes will use Sdpa attention.

### April 18, 2025: WanGP v4.2
👋 FLF2V model support, official support from Wan for image2video start and end frames specialized for 720p.

### April 17, 2025: WanGP v4.1
👋 Recam Master model support, view a video from a different angle. The video to process must be at least 81 frames long and you should set at least 15 steps denoising to get good results.

### April 13, 2025: WanGP v4.0
👋 Lots of goodies for you!
- A new UI, tabs were replaced by a Dropdown box to easily switch models
- A new queuing system that lets you stack in a queue as many text2video, image2video tasks, ... as you want. Each task can rely on complete different generation parameters (different number of frames, steps, loras, ...). Many thanks to **Tophness** for being a big contributor on this new feature
- Temporal upsampling (Rife) and spatial upsampling (Lanczos) for a smoother video (32 fps or 64 fps) and to enlarge your video by x2 or x4. Check these new advanced options.
- Wan Vace Control Net support: with Vace you can inject in the scene people or objects, animate a person, perform inpainting or outpainting, continue a video, ... See [VACE.md](VACE.md) for introduction guide.
- Integrated *Matanyone* tool directly inside WanGP so that you can create easily inpainting masks used in Vace
- Sliding Window generation for Vace, create windows that can last dozens of seconds
- New optimizations for old generation GPUs: Generate 5s (81 frames, 15 steps) of Vace 1.3B with only 5GB and in only 6 minutes on a RTX 2080Ti and 5s of t2v 14B in less than 10 minutes.

### March 27, 2025
👋 Added support for the new Wan Fun InP models (image2video). The 14B Fun InP has probably better end image support but unfortunately existing loras do not work so well with it. The great novelty is the Fun InP image2 1.3B model: Image 2 Video is now accessible to even lower hardware configuration. It is not as good as the 14B models but very impressive for its size. Many thanks to the VideoX-Fun team (https://github.com/aigc-apps/VideoX-Fun)

### March 26, 2025
👋 Good news! Official support for RTX 50xx please check the [installation instructions](INSTALLATION.md).

### March 24, 2025: Wan2.1GP v3.2
👋 
- Added Classifier-Free Guidance Zero Star. The video should match better the text prompt (especially with text2video) at no performance cost: many thanks to the **CFG Zero * Team**. Don't hesitate to give them a star if you appreciate the results: https://github.com/WeichenFan/CFG-Zero-star
- Added back support for PyTorch compilation with Loras. It seems it had been broken for some time
- Added possibility to keep a number of pregenerated videos in the Video Gallery (useful to compare outputs of different settings)

*You will need one more `pip install -r requirements.txt`*

### March 19, 2025: Wan2.1GP v3.1
👋 Faster launch and RAM optimizations (should require less RAM to run)

*You will need one more `pip install -r requirements.txt`*

### March 18, 2025: Wan2.1GP v3.0
👋 
- New Tab based interface, you can switch from i2v to t2v conversely without restarting the app
- Experimental Dual Frames mode for i2v, you can also specify an End frame. It doesn't always work, so you will need a few attempts.
- You can save default settings in the files *i2v_settings.json* and *t2v_settings.json* that will be used when launching the app (you can also specify the path to different settings files)
- Slight acceleration with loras

*You will need one more `pip install -r requirements.txt`*

Many thanks to *Tophness* who created the framework (and did a big part of the work) of the multitabs and saved settings features

### March 18, 2025: Wan2.1GP v2.11
👋 Added more command line parameters to prefill the generation settings + customizable output directory and choice of type of metadata for generated videos. Many thanks to *Tophness* for his contributions.

*You will need one more `pip install -r requirements.txt` to reflect new dependencies*

### March 18, 2025: Wan2.1GP v2.1
👋 More Loras!: added support for 'Safetensors' and 'Replicate' Lora formats.

*You will need to refresh the requirements with a `pip install -r requirements.txt`*

### March 17, 2025: Wan2.1GP v2.0
👋 The Lora festival continues:
- Clearer user interface
- Download 30 Loras in one click to try them all (expand the info section)
- Very easy to use Loras as now Lora presets can input the subject (or other needed terms) of the Lora so that you don't have to modify manually a prompt
- Added basic macro prompt language to prefill prompts with different values. With one prompt template, you can generate multiple prompts.
- New Multiple images prompts: you can now combine any number of images with any number of text prompts (need to launch the app with --multiple-images)
- New command lines options to launch directly the 1.3B t2v model or the 14B t2v model

### March 14, 2025: Wan2.1GP v1.7
👋 
- Lora Fest special edition: very fast loading/unload of loras for those Loras collectors around. You can also now add/remove loras in the Lora folder without restarting the app.
- Added experimental Skip Layer Guidance (advanced settings), that should improve the image quality at no extra cost. Many thanks to the *AmericanPresidentJimmyCarter* for the original implementation

*You will need to refresh the requirements `pip install -r requirements.txt`*

### March 13, 2025: Wan2.1GP v1.6
👋 Better Loras support, accelerated loading Loras.

*You will need to refresh the requirements `pip install -r requirements.txt`*

### March 10, 2025: Wan2.1GP v1.5
👋 Official Teacache support + Smart Teacache (find automatically best parameters for a requested speed multiplier), 10% speed boost with no quality loss, improved lora presets (they can now include prompts and comments to guide the user)

### March 7, 2025: Wan2.1GP v1.4
👋 Fix PyTorch compilation, now it is really 20% faster when activated

### March 4, 2025: Wan2.1GP v1.3
👋 Support for Image to Video with multiples images for different images/prompts combinations (requires *--multiple-images* switch), and added command line *--preload x* to preload in VRAM x MB of the main diffusion model if you find there is too much unused VRAM and you want to (slightly) accelerate the generation process.

*If you upgrade you will need to do a `pip install -r requirements.txt` again.*

### March 4, 2025: Wan2.1GP v1.2
👋 Implemented tiling on VAE encoding and decoding. No more VRAM peaks at the beginning and at the end

### March 3, 2025: Wan2.1GP v1.1
👋 Added Tea Cache support for faster generations: optimization of kijai's implementation (https://github.com/kijai/ComfyUI-WanVideoWrapper/) of teacache (https://github.com/ali-vilab/TeaCache)

### March 2, 2025: Wan2.1GP by DeepBeepMeep v1
👋 Brings:
- Support for all Wan including the Image to Video model
- Reduced memory consumption by 2, with possibility to generate more than 10s of video at 720p with a RTX 4090 and 10s of video at 480p with less than 12GB of VRAM. Many thanks to REFLEx (https://github.com/thu-ml/RIFLEx) for their algorithm that allows generating nice looking video longer than 5s.
- The usual perks: web interface, multiple generations, loras support, sage attention, auto download of models, ...

## Original Wan Releases

### February 25, 2025
👋 We've released the inference code and weights of Wan2.1.

### February 27, 2025
👋 Wan2.1 has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/). Enjoy! 