# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Generative Models Accessible to the GPU Poor</b>
</p>

WanGP is a one-stop super app for the best open source generative models across video, image, audio, and text-to-speech.

## Highlights

| Modality | Supported models |
| --- | --- |
| **Video** | **Wan 2.1/2.2** and derived models, **MiniMax H3**, **LTX-2**, **Hunyuan Video 1/1.5**, **LongCat**, **Kandinsky**, **LTXV**, **MagiHuman** |
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

**Official WanGP Web Site**: https://wangp.ai/

> [!IMPORTANT]
> **WanGP is free to use locally.** The official project will never ask you to pay a license fee, subscription, or donation to run WanGP on your own computer (see the license for terms).
>
> **Use only the official GitHub repository or wangp.ai / wan2gp.ai websites. WanGP is not affiliated to any other third-party service using the WanGP/Wan2GP names**, unless explicitly stated here.


## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)


## 🔥 Latest Updates : 

## 9th of August 2026: WanGP v12.45, Meet The One

**MiniMax H3 had all that potential waiting to be unleashed. We found the keys.**

- **Sliding Windows / Continue Video:** both *FL2VA* and *Ref2VA* can now build longer videos. WanGP carries the previous window's closing motion and matching audio into the next one. Most importantly, overlap is no longer limited to a single frame: using multiple overlap frames gives H3 real motion and sound context across the join, delivering much smoother transitions.

- **Start Image / End Image for Ref2VA:** launch a new shot from a chosen image, aim for a specific ending, or give a continued video the destination it deserves. Ref2VA preserves its selected reference memories across every Sliding Window, so later windows can keep following the same people, places, motion, and sound.

- **Frames Injection in FL2VA:** place several selected images at exact moments in an FL2VA video. Enter frame positions for precise timing or `L` for the end of a sliding-window segment—digital storyboarding without the sticky notes.

- **Audio Source:** FL2VA can create everything from text, follow an uploaded soundtrack, use a Control Video with its original audio, or keep the video unchanged while composing a new soundtrack. Full-length source audio is preserved in the final file; if it runs out early, H3 takes over instead of serving silence.

- **Spectrum v0.2.1 with offline replay:** H3 Spectrum now captures a clean accelerated trajectory and performs a transformer-free smoothing replay. Video and audio are reconstructed independently for better audio quality.

- **Control Video / Denoising Strength:** FL2VA can stay close to a Control Video or wander further from it as the strength increases. At `1.0` with *Whole Frame*, the visual control is unnecessary, so WanGP skips the extra work—your GPU may now take a very short coffee break.

- **Video Mask / Masking Strength:** choose *Whole Frame*, *Masked Area*, or *Non Masked Area* to decide where FL2VA may make changes and how firmly the remaining picture should follow the original.


> **Best practices for longer H3 videos**
> **For a multi-sequence video →** Direct it window by window: give each part its own prompt and duration, connect it smoothly with overlap, or use `[/new_shot]` for a hard cut. WanGP hands you the clapperboard instead of deciding where the story changes. Please check the Prompt Inline Help for the syntax.
> **For one very long continuous shot →** Use one Start Image followed by several End Images. Each End Image becomes the destination of a later Sliding Window, guiding the action from one visual milestone to the next. This works with both FL2VA and Ref2VA.

**Bonus:**  
- **Wan2.2 Animate 2**. *Animate* is back—and it wants to reclaim the crown *Scail 2* snatched away. Give it a character image and a driving video, and it will make that character follow the video's movements, expressions, and camera action: dance routines, performances, gestures, fashion clips, creature animation, and more.

*update 12.45*: spectrum upgraded, animate 2

## 6th of August 2026: WanGP v12.434, Cache Me If You Can

**MiniMax H3 shifts up a gear!**

H3 now has new accelerators and RAM shrinkers. Pick one or stack them—the exact gain depends on your video, hardware, and settings.

- **First Block Cache:** under *Advanced Mode / Steps Skipping*, H3 runs the first block and reuses the remaining blocks' previous result when little has changed—think TeaCache's cool cousin, driven by the first block's output. *Balanced (0.08)* is the upstream default; higher thresholds can skip more work and go faster, with a possible trade in motion or fine detail. The cache is tuned to add very little VRAM overhead. And yes, *Skip Steps starting moment in % of generation* means exactly what it says: it chooses when skipping may begin, not an acceleration factor.

- **Sol-Attn:** under *Advanced Mode / Misc. / Override Attention Mode*, sparse attention speeds up large visual sequences. It requires BF16, Triton 3.6+, and a compatible NVIDIA GPU (RTX 40/50-series, H100/H200, or B100/B200). Expected gains range from 10–20% on RTX 40-series to around 30% on RTX 50-series, with a possible small quality trade-off.

- **Mix and match:** *Spectrum* and *First Block Cache* are alternative step-skipping modes. Sol-Attn can technically run with either, but stacking approximations may reduce quality and should be checked with the same seed before relying on the combination.

- **Lower-RAM Video VAE:** select *FP8 Mixed Precision* under *Advanced Mode / Misc. / Video VAE* to reduce the RAM occupied by H3's Video VAE weights. Thanks to *Kijai* for creating this quantized VAE.

- **New W4A8 INT8 support:** H3 can now load asymmetric W4A8 checkpoints. Their 4-bit weights reduce checkpoint size and system RAM use, while 8-bit activations use optimized INT8 kernels on compatible NVIDIA GPUs (RTX 30-series or newer). Seriously short on RAM? Look for compatible community H3 W4A8/Q4 or NVFP4 checkpoints already available online. See *docs/FINETUNES.md* to add them to WanGP—and don't forget to share the finetune files you create on the Discord server!

- **Ref2VA tune-up:** this one is on me—I followed the original implementation and could end up feeding H3 a 4K reference image for a 480p video. Great for detail, less great for your stopwatch! You can now choose the reference-image pixel budget from 50% to 400%: lower is faster, 100% matches the output, and higher favors fidelity. The immediate payoff: **WanGP H3 Ref2VA is now twice as fast as before.**

- **New control-video choices:** use a *Reference Video* to reuse subjects, appearance, or motion without changing the output size; *Depth Control* to guide the scene's depth and layout; or *Generic Control* to feed the clip directly to H3. Control videos define the output canvas, while reference videos do not.

- **No LoRA Lost in Translation:** Pruned and non-pruned models can now read either LoRA format—the translation happens automatically as they load. Pruned (4 rank) LoRAs can also be used on original WanGP pruned checkpoints (rank 64) if you still use them.

- **LoRAs Accelerators**: kudos to *Lightx2v* and *larryvrh* for delivering the first *LoRAs accelerators* for Minimax H3. You will find them in WanGP as predefined profiles in the *Settings* dropdown box at the top. You may need to increase the number of steps to 8 if not happy with the quality and / or to play with the *LoRA multiplier* (default is 0.5 as 1.0 seems too strong)

*Update v12.431 + Update v12.432*: more LoRAs format supported, fixed NVFP4 Format, on the fly LoRA conversion of Non Pruned Loras\
*Update v12.433 + Update v12.434*: even more LoRAs format and quantization supported, LoRAs accelerators

## 5th of August 2026: WanGP v12.42, No Time for Taglines

**MiniMax H3**

MiniMax H3 is a top-notch open-weight contender to Seedance 2, combining cinematic video generation, convincing motion, strong prompt adherence, and a synchronized native stereo soundtrack in one model.

Given no *Steps Distilled Checkpoints* is available for the moment, 15-20 inference steps is a minimum.

But rejoice WanGP version is as usual Ultra Optimized: **5-6GB of VRAM only for 5s (124 frames) and 8-9GB of VRAM for 15s at 832x480**. 

- **MiniMax H3 FL2VA: create or continue a shot**: choose this version to generate synchronized video and stereo audio from text alone, start from an image or the last frame of a previous video, target an end image, or constrain both ends of the shot. It also supports longer generations with sliding windows.

- **MiniMax H3 Ref2VA: reuse people, scenes, motion, or voices**: choose this version when the new video should follow *Reference Images*, *Reference Videos*, or *Reference Audio*. References guide the newly generated result rather than becoming fixed frames, and remain available across sliding windows.

Both flavours offer the same controls in full 33B and lighter pruned 20B versions.

- **Spectrum step skipping**: Spectrum can make MiniMax H3 generation substantially faster, with a modest potential quality tradeoff. Its default offline replay retains every actual-step anchor in system RAM, reconstructs skipped steps from bracketing and spectral estimates, and keeps audio on local interpolation. Enable it under *Advanced Mode / Steps Skipping* by setting *Skip Steps Cache Type* to *Spectrum Feature Forecasting*.

- **Spatial upsampler improvements**: high-resolution MiniMax H3 generation can be slow, so a practical alternative is to generate at a lower resolution, such as 480p, and upscale the result afterward.
 - **FlashVSR optimizations**: FlashVSR has been further optimized to reduce system RAM usage.
 - **SeedVR2**: this high-quality image and video upsampler previously required too much VRAM for longer videos on many consumer GPUs. The WanGP integration reduces its VRAM requirement to roughly one-third of the original implementation. SeedVR2 is available under *Advanced Mode / Post Processing*, in *Late Post Processing*, and through the *Media Flow* plugin.

- **Memory priority**: MiniMax H3 defaults to *Lower VRAM*. If system RAM is the limiting factor and you have spare VRAM, select *Lower RAM* under *Advanced Mode / Misc. / Priority*.

- **Updated pruned checkpoints**: WanGP now uses ComfyUI-compatible pruned H3 checkpoints so upcoming H3 LoRAs can work with both applications. The previous WanGP-specific checkpoints were slightly less compressed and offered a small, usually imperceptible quality advantage. Existing installations will download the replacement files after upgrading. The old checkpoints are not removed automatically; keep them only if you plan to create finetunes from them, otherwise they can be deleted. The replacements require about 1 GB less disk space and system RAM.

- **Lower-RAM text encoders**: if system RAM is limited, open *Advanced Mode / Misc. / Text Encoder* and select one of the quantized Qwen3-VL variants.

*WanGP v12.41*: Added quantized text-encoder selection.\
*WanGP v12.42*: Added Spectrum, SeedVR2, the memory-priority selector, and updated pruned checkpoints.
## 25th of July 2026: Featured Plugins / Apps

WanGP's growing community has developed more than 20 plugins that expand what you can do. Here is a selection of seven newly available community plugins, all of which can be installed or updated directly from the WanGP Plugin Manager:

- **Finetune Manager** by *GKartist* — Browse community finetunes, load them into WanGP, and create, improve, or share your own.
- **Image Suite** by *saintorphan* — Create and edit images with text-to-image, image-to-image, layered canvases, masks, inpainting, cropping, resizing, and color adjustments.
- **Prompt Library** by *saintorphan* — Save your favorite prompts and generation settings, then reuse them with any supported model.
- **Prompt Manager** by *David Brum* — Search and organize generated images and videos, copy their settings, and manage reusable prompts in one place.
- **Queue Notifier** by *Javier-bat* — Get progress, completion, and failure alerts through services such as Discord, Telegram, WhatsApp, and Google Chat.
- **VRAM / RAM Adjuster** by *g3n3rativ3* — Tune how much graphics and system memory WanGP uses without manually editing configuration files.
- **Wildcards** by *GKartist* — Add reusable variables and random choices to prompts so you can quickly produce controlled variations.

**New Wan2GP Desktop Installer** — [Wan2GP Desktop](https://github.com/GKartist75/wan2gp-desktop) by GKArtist lets you install, update, and launch WanGP from a single window. It handles Git, Python, CUDA, and PyTorch setup for you, making it the easiest way to get started on Windows.

### 29th of July 2026: WanGP v12.3456, Increasingly Greater

- **Krea 2 Identity Edit**: this Krea2 finetune adds Editing capabilities to Krea 2. You can edit an existing image or combine up to 2 *Reference Images* to produce a new one. WanGP implementation comes out of the box with *Inpainting* and *Outpainting*   

- **PiD 1.5**: The *PiD Spatial Upsampler* has been updated and should deliver better quality (v1 still there if you prefer it) and also now exists in *Qwen VAE* flavor (that is it can be plugged directly to Wan2.1 t2i, Qwen or Krea2 latent output for best quality)

- **LTX2 MSR 2.0**: this new version of this LTX2 finetune with Image Reference support preserves better Identity. WanGP v12.345 adds the setting *MSR Reference Video Length* that will let you control how the *Reference Images* are packed (please check model help for more info)

- **Joy Echo Surgical**: as a reminder the *Joy Echo* LTX2 variant lets you reuse characters identities between shots. This *Surgical* finetune claims to preserve better identity between shots and offers better audio quality.

- **ConvRot LoRA support**: Int8 ConvRot checkpoints can now use LoRAs without producing garbage output 

- **Text Encoder GGUF Support**: you should be able now to use in your Finetunes Text Encoder *GGUF* Checkpoints with LTX2, Krea2, Flux 1/2 and Wan 2.1/2.2

- **Onmnivoice Speed ajustment**: a new option gives you more control on the pace on spoken words (for instance to you fit more words in a shorter timespan)

- **More Krea2 LoRA Support**: more LoRAs formats are supported 

- **Shotplan**: some form *PromptRelay* for Wan 2.1 and Wan 2.2, you can divide a gen into different shots that reuses the same characters or objects

- **PrunaVAED**: this an alternative *VAE Video Decoder** for LTX2 that is 2.7 faster and requires half the VRAM during the *VAE decoding*. You will be able to select PrunaVAED in the *Misc* Tab at the bottom, in the new *Config* dropdown box

- **WanGP Configs**: if you just want to change the VAE or the text encoder you no longer need to create multiple finetune, you can now just create a single finetune with multiple configs in it. Please check *docs/FINETUNES.md*

*update 12.345*: Text Encode GGUF Support, Joy Echo Surgical, MSR2 new setting, Omnivoice Speed adjustment, new Krea2 LoRA formats\
*update 12.346*: Shotplan, PrunaVAED, WanGP Configs

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

### One-click Installers
- Pinokio installer
Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.

- Wan2GP Desktop by GKArtist
[Wan2GP Desktop](https://github.com/GKartist75/wan2gp-desktop) is a desktop launcher for Wan2GP that installs, updates, and runs it from one window — handling Git, Python, CUDA, and PyTorch setup so you don't have to configure them manually.

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
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions for GTX 10XX, RTX 20XX to RTX 50XX

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
