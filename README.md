# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor</b>
</p>

WanGP supports the Wan (and derived models), Hunyuan Video, Flux, Qwen, Z-Image, LongCat, Kandinsky and LTX Video models with:
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
- Headless mode: launch the generation of multiple image / videos using a command line

**Discord Server to get Help from Other Users and show your Best Videos:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep


## 🔥 Latest Updates : 

### January 23th 2026: WanGP v10.50, Music for your Hearts

WanGP Special *TTS* (Text To Speech) Release:

- **Heart Mula**: *Suno* quality song with lyrics on your local PC. You can generate up to 4 min of music.

- **Qwen 3 TTS**: you can either do *Voice Cloning*, *Generate a Custom Voice based on a Prompt* or use a *Predefined Voice*

- **TTS Features**:
   - **Early stop** : you can abort a gen, while still keeping what has been generated (will work only for TTS models which are *Autoregressive Models*, no need to ask that for Image/Video gens which are *Diffusion Models*)
   - **Specialized Prompt Enhancers**: if you enter the prompt in Heart Mula *"a song about AI generation"*, *WanGP Prompt Enhancer* will generate the corresponding masterpiece for you. Likewise you can enhance "A speech about AI generation" when using Qwen3 TTS or ChatterBox.
   - **Custom Output folder for Audio Gens**: you can now choose a different folder for the *Audio Outputs*
   - **Default Memory Profile for Audio Models**: TTS models can get very slow if you use profile 4 (being autoregressive models, they will need to load all the layers one per one to generate one single audio token then rinse & repeat). On the other hand, they dont't need as much VRAM, so you can now define a more agressive profile (3+ for instance)


### January 20th 2026: WanGP v10.43, The Cost Saver
*GPUs are expensive, RAM is expensive, SSD are expensive, sadly we live now in a GPU & RAM poor.*

WanGP comes again to the rescue:

- **GGUF support**: as some of you know, I am not a big fan of this format because when used with image / video generative models we don't get any speed boost (matrices multiplications are still done at 16 bits), VRAM savings are small and quality is worse than with int8/fp8. Still gguf has one advantage: it consumes less RAM and harddrive space. So enjoy gguf support. I have added ready to use *Kijai gguf finetunes* for *LTX 2*.

- **Models Manager PlugIn**: use this *Plugin* to identify how much space is taken by each *model* / *finetune* and delete the ones you no longer use. Try to avoid deleting shared files otherwise they will be downloaded again.  

- **LTX 2 Dual Video & Audio Control**: you no longer need to extract the audio track of a *Control Video* if you want to use it as well to drive the video generation. New mode will allow you to use both motion and audio from Video Control.

- **LTX 2 - Custom VAE URL**: some users have asked if they could use the old *Distiller VAE* instead of the new one. To do that, create a *finetune* def based on an existing model definition and save it in the *finetunes/* folder with this entry (check the *docs/FINETUNES.md* doc):
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

- **LTX Distilled VAE Upgrade**: *Kijai* has observed that the Distilled VAE produces images that were less sharp that the VAE of the Non Distilled model. I have used this as an opportunity to repackage all the LTX 2 checkpoints and reduce their overal HD footprint since they all share around 5GB. 

**So dont be surprised if the old checkpoints are deleted and new are downloaded !!!**.

- **LTX2 Multi Passes Loras multipliers**: *LTX2* supports now loras multiplier that depend on the Pass No. For instance "1;0.5" means 1 will the strength for the first LTX2 pass and 0.5 will be the strength for the second pass.

- **New Profile 3.5**: here is the lost kid of *Profile 3* & *Profile 5*, you got tons of VRAM, but little RAM ? Profile 3.5 will be your new friend as it will no longer use Reserved RAM to accelerate transfers. Use Profile 3.5 only if you can fit entirely a *Diffusion / Transformer* model in VRAM, otherwise the gen may be much slower.

- **NVFP4 Quantization for LTX 2 & Flux 2**: you will now be able to load *NV FP4* model checkpoints in WanGP. On top of *Wan NV4* which was added recently, we now have *LTX 2 (non distilled)* & *Flux 2* support. NV FP4 uses slightly less VRAM and up to 30% less RAM. 

To enjoy fully the NV FP4 checkpoints (**at least 30% faster gens**), you will need a RTX 50xx and to upgrade to *Pytorch 2.9.1 / Cuda 13* with the latest version of *lightx2v kernels* (check *docs/INSTALLATION.md*). To observe the speed gain, you have to make sure the workload is quite high (high res, long video).


### January 13th 2026: WanGP v10.24, When there is no VRAM left there is still some VRAM left ...

- **LTX 2 - SUPER VRAM OPTIMIZATIONS**  

*With WanGP 10.21 HD 720p Video Gens of 10s just need now 8GB of VRAM!*

LTX Team said this video gen was for 4k. So I had no choice but to squeeze more VRAM with further optimizations.

After much suffering I have managed to reduce by at least 1/3 the VRAM requirements of LTX 2, which means:
  - 10s at 720p can be done with only 8GB of VRAM
  - 10s at 1080p with only 12 GB of VRAM
  - 20s at 1080p with only 16 GB of VRAM
  - 10s at Full 4k (3840 x 2176 !!!) with 24 GB of VRAM.  However the bad news is LTX 2 video is not for 4K, as 4K outputs may give you nightmares ...

3K/4K resolutions will be available only if you enable them in the *Config* / *General* tab.

- **Ic Loras support**: Use a *Control Video* to transfer *Pose*, *Depth*, *Canny Edges*. I have added some extra tweaks: with WanGP you can restrict the transfer to a *masked area*, define a *denoising strength* (how much the control video is going to be followed) and a *masking strength* (how much unmasked area is impacted) 

- **Start Image Strength**: This new slider will appear below a *Start Image* or Source *Video*. If you set it to values lower than 1 you may to reduce the static image effect, you get sometime with LTX2 i2v
 
- **Custom Gemma Text Encoder for LTX 2**: As a practical case, the *Heretic* text encoder is now supported by WanGP. Check the *finetune* doc, but in short create a *finetune* that has a *text_encoder_URLS* key that contains a list of one or more file paths or URLs.  

- **Experimental Auto Recovery Failed Lora Pin**: Some users (with usually PC with less than 64 GB of RAM) have reported Out Of Memory although a model seemed to load just fine when starting a gen with Loras. This is sometime related to WanGP attempting (and failing due to unsufficient reserved RAM) to pin the Loras to Reserved Memory for faster gen. I have experimented a recovery mode that should release sufficient ressources to continue the Video Gen. This may solve the oom crashes with *LTX2 Default (non distilled)* 

- **Max Loras Pinned Slider**:  If the Auto Recovery Mode is still not sufficient, I have added a Slider at the bottom of the  *Configuration*  / *Performance* tab that you can use to prevent WanGP from Pinning Loras (to do so set it to 0). As if there is no loading attempt there wont be any crash...

*update 10.21*: added slider Loras Max Pinning slider\
*update 10.22*: added support for custom Ltx2 Text Encoder + Auto Recovery mode if Lora Pinning failed\
*update 10.23*: Fixed text prompt ignore in profile 1 & 2 (this created random output videos)

### January 9st 2026: WanGP v10.11, Spoiled again

- **LTX 2**: here is the long awaited *Ovi Challenger*, LTX-2 generates video and an audio soundtrack. As usual this WanGP version is *low VRAM*. You should be able to run it with as low as 10 GB of VRAM. If you have at least 24 GB of VRAM you will be able to generate 20s at 720p in a single window in only 2 minutes with the distilled model.  WanGP LTX 2 version supports on day one, *Start/End keyframes*, *Sliding-Window* / *Video Continuation* and *Generation Preview*. A *LTX 2 distilled* is part of the package for a very fast generation.

With WanGP v10.11 you can now force your soundtrack, it works like *Multitalk* / *Avatar* except in theory it should work with any kind of sound (not just vocals). Thanks to *Kijai* for showing it was possible.

- **Z Image Twin Folder Turbo**: Z Image even faster as this variant can generate images with as little as 1 step (3 steps recommend) 

- **Qwen LanPaint**: very precise *In Painting*, offers a better integration of the inpainted area in the rest of the image. Beware it is up to 5x slower as it "searches" for the best replacement. 

- **Optimized Pytorch Compiler** : *Patience is the Mother of Virtue*. Finally I may (or may not) have fixed the PyTorch compiler with the Wan models. It should work in much diverse situations and takes much less time. 

- **LongCat Video**: experimental support which includes *LongCat Avatar* a talking head model. For the moment it is mostly for models collectors as it is very slow. It needs 40+ steps and each step contains up 3 passes.

- **MMaudio NSFW**: for alternative audio background

*update v10.11*: LTX 2, use your own soundtrack

### January 1st 2026: WanGP v10.01, Happy New Year !

- **Wan 2.2 i2v Stable Vision Infinity Pro 2**: SVI Pro 2 offers potentially unlimited Videos to Continue for i2v models. It will use either the Start frame as a Reference Image or you may provide an Anchor image to be used across all the windows or multiple Anchor Images one per Window.

- **Wan 2.1 Alpha 2**: This new version of Alpha generates transparent videos with fine-grained alpha detail (hair, glow, smoke). 

- **Qwen Image 2512**: This December release offers Enhanced Human Realism, Finer Natural Details & Improved Text Rendering.

- **Wan NVP4**: *Light2xv nvfp4 support for Wan 2.1 i2v & t2v 1.3B*, you can now load nvfp4 (4 bits quantized file) in WanGP. These will make really a difference with RTX 50xx as they support natively scaled FP4 calculation. Other GPUs will get the pytorch fallback which is slower. This model can be useful for machines with low RAM but don't expect significant VRAM reduction of much faster speed for non RTX 50xx owners. You will need to install the Light2xv kernels.

- **Nunckaku int4 & fp4 support for Qwen 2509 & Z Image**: int4 versions will work with most GPUs, fp4 will accelerate only RTX50xx. You will need to install the nunchaku kernels. See light2xv nvfp4 above, as the other comments apply here too.

- **Z Image Control Net 2.1**: Control Net upgraded should work better. I have enabled as well inpainting for the control net.

- **New Qwen Loras Accelerators Added**

*Quantization Kernels Wheels for Windows / Python 3.10 / Pytorch 2.70:*
- *Light2xv (WAN-FP4)*
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/WAN_NVP4/lightx2v_kernel-0.0.1-cp39-abi3-win_amd64.whl
   ```

- *Nunchaku*
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Nunchaku/nunchaku-1.1.0+torch2.7-cp310-cp310-win_amd64.whl
   ```
   
### December 23 2025: WanGP v9.92, Early Christmas

- **SCAIL Preview**: enjoy this *Wan Animate*, *Steady Dancer* contender that can support multiple people. Thanks to its 3D positioning, it can take into account which parts of the body are hidden and which are not. 

WanGP version has the following perks: 3D pose Preprocessing entirely rewritten to be fast,  and compatible with any pytorch version, very Low VRAM requirements for multicharacters, experimental long gen mode / sliding windows (SCAIL Preview doesnt  support officialy long gen yet)

- **pi-Flux 2**: you don't use Flux 2 because you find it too slow ? You won't be able to use this excuse anymore: pi-Flux 2 is *4 steps distills* of the best image generator. It supports both image edition and text to image generation.

- **Zandinksy v5** : for the video models collectors among you, you can try the Zandinsky model families, the 2B model quality is especially impressive given its small size

- **Qwen Image Layered**: a new Qwen Image variant that lets you extract RGBA layers of your images so that  each layer can be edited separately

- **Qwen Image Edit Plus 2511**: Qwen Image Edit Plus 2511 improves identity preservation (especially at 1080p) and integrates out of the box popular effects such as religthing and camera changes

- **loras accelerator**: *loras accelerator* for *Wan 2.2 t2v* and *Wan 2.1 i2v* have been added (activable using the *Profile settings* as usual) 

*update 9.91*: added Kandinsky 5 & Qwen Image Layered\
*update 9.92*: added Qwen Image Edit Plus 2511

### December 14 2025: WanGP v9.86, Simple Pleasures...

These two features are going to change the life of many people:
- **Pause Button**: ever had a urge to use your GPU for a very important task that can't wait (a game for instance ?), here comes your new friend the *Pause* button. Not only it will suspend the current gen in progress but it will free most of the VRAM used by WanGP (please note that the RAM by WanGP used wont be released). When you are done just click the *Resume* button to restart exactly from where you stopped.

- **WanGP Headless**:  trouble running remotely WanGP or having some stability issues with Gradio or your Web Browser. This is all past thanks to *WanGP Headless* mode. Here is how it works : first make you shopping list of Video Gen using the classic WanGP gradio interface. When you are done, click the *Save Queue* button and quit WanGP.

Then in your terminal window just write this:
```bash
python wgp.py --process my_queue.zip
```
With WanGP 9.82, you can also process settings file (.json file exported using th *Export Settings* button):
```bash
python wgp.py --process my_settings.json
```
Processing Settings can be useful to do some quick gen / testing if you don't need to provide source image files (otherwise you will need to fill the paths to Start Images, Ref Image, ...)

- **Output Filename Customization**: in the *Misc* tab you can now customize how the file names of new Generation are created, for example:
```
{date(YYYY-MM-DD_HH-mm-ss)}_{seed}_{prompt(50)}, {num_inference_steps}
```

- **Hunyuan Video 1.5 i2v distilled** : for those in need of their daily dose of new models, added *Hunyuan Video 1.5 i2v Distilled* (official release) + Lora Accelerator extracted from it (to be used in future finetunes). Also added *Magcache* support (optimized for 20 steps) for Hunyuan Video 1.5.

- **Wan-Move** : Another model specialized to control motion using a *Start Image* and *Trajectories*. According to the author's paper it is the best one. *Motion Designer* has been upgraded to generate also trajectories for *Wan-Move*.

- **Z-Image Control Net v2** : This is an upgrade of Z-Image Control Net. It offers much better results but requires much more processing an VRAM. But don't panic yet, as it was VRAM optimized. It was not an easy trick as this one is complex. It has also Inpainting support,but I need more info to release this feature.

*update 9.81*: added Hunyuan Video 1.5 i2v distilled + magcache\
*update 9.82*: added Settings headless processing, output file customization, refactored Task edition and queue processing\
*update 9.83*: Qwen Edit+ upgraded: no more any zoom out at 1080p, enabled mask, enabled image refs with inpainting\
*update 9.84*: added Wan-Move support\
*update 9.85*: added Z-Image Control net v2\
*update 9.86*: added NAG support for Z-Image

### December 4 2025: WanGP v9.74, The Alpha & the Omega ... and the Dancer

- **Flux 2**: the best ever open source *Image Generator* has just landed. It does everything very well: generate an *Image* based a *Text Prompt* or combine up to 10 *Images References* 

The only snag is that it is a 60B parameters for the *Transformer* part and 40B parameters for the *Text Encoder* part.

Behold the WanGP Miracle ! Flux 2 wil work with only 8 GB of VRAM if you are happy with 8 bits quantization (no need for lower quality 4bits). With 9GB of VRAM you can run the model at full power. You will need at least 64 GB of RAM. If not maybe Memory Profile 5 will be your friend.

*With WanGP v9.74*, **Flux 2 Control Net** hidden power has also been unleashed from the vanilla model. You can now enjoy Flux 2 *Inpainting* and *Pose transfer*. This can be combined with *Image Refs* to get the best *Identity Preservation* / *Face Swapping* an Image Model can offer: just target the effect to a specific area using a *Mask* and set *Denoising Strength* to 0.9-1.0 and *Masking Strength* to 0.3-0.4 for a perfect blending 

- **Z-Image**: a small model, very fast (8 steps), very low VRAM (optimized even more in WanGP for fun, just in case you want to generate 16 images at a time) that produces outstanding Image quality. Not yet the Flux 2 level, and no Image editing yet but a very good trade-off.

While waiting for Z-Image edit, *WanGP 9.74* offers now support for **Z-Image Fun Control Net**. You can use it for *Pose transfer*, *Canny Edge* transfer. Don't be surprised if it is a bit slower. Please note it will work best at 1080p and will require a minimum of 9 steps.

- **Steady Dancer**: here is *Wan Steady Dancer* a very nice alternative to *Wan Animate*. You can transfer the motion of a Control video in a very smooth way. It will work best with Videos where the action happens center stage (hint: *dancing*). Use the *Lora accelerator* *Fusionix i2v 10 steps* for a fast generation. For higher quality you can set *Condition Guidance* to 2 or if you are very patient keep *Guidance* to a value greater than 1.   

I have added a new Memory Profile *Profile 4+* that is sligthly slower than *Profile 4* but can save you up to 1GB of VRAM with Flux 2.

Also as we have now quite few models and Loras folders. *I have moved all the loras folder in the 'loras' folder*. There are also now unique subfolders for *Wan 5B* and *Wan 1.3B* models. A conversion script should have moved the loras in the right locations, but I advise that you check just in case.

*update 9.71* : added missing source file, have fun !\
*update 9.72* : added Z-Image & Loras reorg\
*update 9.73* : added Steady Dancer\
*update 9.74* : added Z-Image Fun Control Net & Flux 2 Control Net + Masking

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
Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.


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
