# Changelog
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

*update 8.991*: full power of *Vace Lynx* unleashed with new combinations such as Landscape + Face / Clothes + Face  / Injectd Frame (Start/End frames/...) + Face\
*update 8.992*: optimized gen with Lora, should be 10% faster if many loras\
*update 8.993*: Support for *Scaled FP8* format and samples *Paligenesis* finetunes, merged Loras Accelerators and Non Accelerators\
*update 8.994*: Added custom checkpoints folders\
*update 8.999*: fixed a lora + fp8 bug and version sync for the jump to the unknown 

### September 30 2025: WanGP v8.9 - Combinatorics

This new version of WanGP introduces **Wan 2.1 Lynx** the best Control Net so far to transfer *Facial Identity*. You will be amazed to recognize your friends even with a completely different hair style. Congrats to the *Byte Dance team* for this achievement. Lynx works quite with well *Fusionix t2v* 10 steps.

*WanGP 8.9* also illustrate how existing WanGP features can be easily combined with new models. For instance with *Lynx* you will get out of the box *Video to Video* and *Image/Text to Image*.

Another fun combination is *Vace* + *Lynx*, which works much better than *Vace StandIn*. I have added sliders to change the weight of Vace & Lynx to allow you to tune the effects.

### September 28 2025: WanGP v8.76 - ~~Here Are Two Three New Contenders in the Vace Arena !~~ The Never Ending Release 

So in ~~today's~~ this release you will find two Wannabe Vace that covers each only a subset of Vace features but offers some interesting advantages:
- **Wan 2.2 Animate**: this model is specialized in *Body Motion* and *Facial Motion transfers*. It does that very well. You can use this model to either *Replace* a person in an in Video or *Animate* the person of your choice using an existing *Pose Video* (remember *Animate Anyone* ?). By default it will keep the original soundtrack. *Wan 2.2 Animate* seems to be under the hood a derived i2v model and should support the corresponding Loras Accelerators (for instance *FusioniX t2v*). Also as a WanGP exclusivity, you will find support for *Outpainting*.

In order to use Wan 2.2 Animate you will need first to stop by the *Mat Anyone* embedded tool, to extract the *Video Mask* of the person from which you want to extract the motion.

With version WanGP 8.74, there is an extra option that allows you to apply *Relighting* when Replacing a person. Also, you can now Animate a person without providing a Video Mask to target the source of the motion (with the risk it will be less precise) 

For those of you who have a mask halo effect when Animating a character I recommend trying *SDPA attention* and to use the *FusioniX i2v* lora. If this issue persists (this will depend on the control video) you have now a choice of the two *Animate Mask Options* in *WanGP 8.76*. The old masking option which was a WanGP exclusive has been renamed *See Through Mask* because the background behind the animated character was preserved but this creates sometime visual artifacts. The new option which has the shorter name is what you may find elsewhere online. As it uses internally a much larger mask, there is no halo. However the immediate background behind the character is not preserved and may end completely different.

- **Lucy Edit**: this one claims to be a *Nano Banana* for Videos. Give it a video and asks it to change it (it is specialized in clothes changing) and voila ! The nice thing about it is that is it based on the *Wan 2.2 5B* model and therefore is very fast especially if you the *FastWan* finetune that is also part of the package.

Also because I wanted to spoil you:
- **Qwen Edit Plus**: also known as the *Qwen Edit 25th September Update* which is specialized in combining multiple Objects / People. There is also a new support for *Pose transfer* & *Recolorisation*. All of this made easy to use in WanGP. You will find right now only the quantized version since HF crashes when uploading the unquantized version.

- **T2V Video 2 Video Masking**: ever wanted to apply a Lora, a process (for instance Upsampling) or a Text Prompt on only a (moving) part of a Source Video. Look no further, I have added *Masked Video 2 Video* (which works also in image2image) in the *Text 2 Video* models. As usual you just need to use *Matanyone* to creatre the mask.


*Update 8.71*: fixed Fast Lucy Edit that didnt contain the lora
*Update 8.72*: shadow drop of Qwen Edit Plus
*Update 8.73*: Qwen Preview & InfiniteTalk Start image 
*Update 8.74*: Animate Relighting / Nomask mode , t2v Masked Video to Video  
*Update 8.75*: REDACTED  
*Update 8.76*: Alternate Animate masking that fixes the mask halo effect that some users have    

### September 15 2025: WanGP v8.6 - Attack of the Clones

- The long awaited **Vace for Wan 2.2** is at last here or maybe not: it has been released by the *Fun Team* of *Alibaba* and it is not official. You can play with the vanilla version (**Vace Fun**) or with the one accelerated with Loras (**Vace Fan Cocktail**)

- **First Frame / Last Frame for Vace** : Vace models are so powerful that they could do *First frame / Last frame* since day one using the *Injected Frames* feature. However this required to compute by hand the locations of each end frame since this feature expects frames positions. I made it easier to compute these locations by using the "L" alias :

For a video Gen from scratch *"1 L L L"* means the 4 Injected Frames will be injected like this: frame no 1 at the first position, the next frame at the end of the first window, then the following frame at the end of the next window, and so on ....
If you *Continue a Video* , you just need *"L L L"* since the first frame is the last frame of the *Source Video*. In any case remember that numeral frames positions (like "1") are aligned by default to the beginning of the source window, so low values such as 1 will be considered in the past unless you change this behaviour in *Sliding Window Tab/ Control Video, Injected Frames aligment*.

- **Qwen Edit Inpainting** exists now in two versions: the original version of the previous release and a Lora based version. Each version has its pros and cons. For instance the Lora version supports also **Outpainting** ! However it tends to change slightly the original image even outside the outpainted area.

- **Better Lipsync with all the Audio to Video models**: you probably noticed that *Multitalk*, *InfiniteTalk* or *Hunyuan Avatar* had so so lipsync when the audio provided contained some background music. The problem should be solved now thanks to an automated background music removal all done by IA. Don't worry you will still hear the music as it is added back in the generated Video.

### September 11 2025: WanGP v8.5/8.55 - Wanna be a Cropper or a Painter ?

I have done some intensive internal refactoring of the generation pipeline to ease support of existing models or add new models. Nothing really visible but this makes WanGP is little more future proof.

Otherwise in the news:
- **Cropped Input Image Prompts**: as quite often most *Image Prompts* provided (*Start Image, Input Video, Reference Image,  Control Video, ...*) rarely matched your requested *Output Resolution*. In that case I used the resolution you gave either as a *Pixels Budget* or as an *Outer Canvas* for the Generated Video. However in some occasion you really want the requested Output Resolution and nothing else. Besides some models deliver much better Generations if you stick to one of their supported resolutions. In order to address this need I have added a new Output Resolution choice in the *Configuration Tab*:  **Dimensions Correspond to the Ouput Weight & Height as the Prompt Images will be Cropped to fit Exactly these dimensins**. In short if needed the *Input Prompt Images* will be cropped (centered cropped for the moment). You will see this can make quite a difference for some models

- *Qwen Edit* has now a new sub Tab called **Inpainting**, that lets you target with a brush which part of the *Image Prompt* you want to modify. This is quite convenient if you find that Qwen Edit modifies usually too many things. Of course, as there are more constraints for Qwen Edit don't be surprised if sometime it will return the original image unchanged. A piece of advise: describe in your *Text Prompt* where (for instance *left to the man*, *top*, ...) the parts that you want to modify are located.

The mask inpainting is fully compatible with *Matanyone Mask generator*: generate first an *Image Mask* with Matanyone, transfer it to the current Image Generator and modify the mask with the *Paint Brush*. Talking about matanyone I have fixed a bug that caused a mask degradation with long videos (now WanGP Matanyone is as good as the original app and still requires 3 times less VRAM)

- This **Inpainting Mask Editor** has been added also to *Vace Image Mode*. Vace is probably still one of best Image Editor today. Here is a very simple & efficient workflow that do marvels with Vace:
Select *Vace Cocktail > Control Image Process = Perform Inpainting & Area Processed = Masked Area > Upload a Control Image, then draw your mask directly on top of the image & enter a text Prompt that describes the expected change > Generate > Below the Video Gallery click 'To Control Image' > Keep on doing more changes*.

Doing more sophisticated thing Vace Image Editor works very well too: try Image Outpainting, Pose transfer, ...

For the best quality I recommend to set in *Quality Tab* the option: "*Generate a 9 Frames Long video...*" 

**update 8.55**: Flux Festival
- **Inpainting Mode** also added for *Flux Kontext*
- **Flux SRPO** : new finetune with x3 better quality vs Flux Dev according to its authors. I have also created a *Flux SRPO USO* finetune which is certainly the best open source *Style Transfer* tool available
- **Flux UMO**: model specialized in combining multiple reference objects / people together. Works quite well at 768x768

Good luck with finding your way through all the Flux models names !

### September 5 2025: WanGP v8.4 - Take me to Outer Space
You have probably seen these short AI generated movies created using *Nano Banana* and the *First Frame - Last Frame* feature of *Kling 2.0*. The idea is to generate an image, modify a part of it with Nano Banana and give the these two images to Kling that will generate the Video between these two images, use now the previous Last Frame as the new First Frame, rinse and repeat and you get a full movie.

I have made it easier to do just that with *Qwen Edit* and *Wan*:
- **End Frames can now be combined with Continue a Video** (and not just a Start Frame)
- **Multiple End Frames can be inputed**, each End Frame will be used for a different Sliding Window

You can plan in advance all your shots (one shot = one Sliding Window) : I recommend using Wan 2.2 Image to Image with multiple End Frames (one for each shot / Sliding Window), and a different Text Prompt for each shot / Sliding Winow (remember to enable *Sliding Windows/Text Prompts Will be used for a new Sliding Window of the same Video Generation*)

The results can quite be impressive. However, Wan 2.1 & 2.2 Image 2 Image are restricted to a single overlap frame when using Slide Windows, which means only one frame is reeused for the motion. This may be unsufficient if you are trying to connect two shots with fast movement.

This is where *InfinitTalk* comes into play. Beside being one best models to generate animated audio driven avatars, InfiniteTalk uses internally more one than motion frames. It is quite good to maintain the motions between two shots. I have tweaked InfinitTalk so that **its motion engine can be used even if no audio is provided**.
So here is how to use InfiniteTalk: enable *Sliding Windows/Text Prompts Will be used for a new Sliding Window of the same Video Generation*), and if you continue an existing Video  *Misc/Override Frames per Second" should be set to "Source Video*. Each Reference Frame inputed will play the same role as the End Frame except it wont be exactly an End Frame (it will correspond more to a middle frame, the actual End Frame will differ but will be close)


You will find below a 33s movie I have created using these two methods. Quality could be much better as I havent tuned at all the settings (I couldn't bother, I used 10 steps generation without Loras Accelerators for most of the gens).

### September 2 2025: WanGP v8.31 - At last the pain stops

- This single new feature should give you the strength to face all the potential bugs of this new release:
**Images Management (multiple additions or deletions, reordering) for Start Images / End Images / Images References.**  

- Unofficial **Video to Video (Non Sparse this time) for InfinitTalk**. Use the Strength Noise slider to decide how much motion of the original window you want to keep. I have also *greatly reduced the VRAM requirements for Multitalk / Infinitalk* (especially the multispeakers version & when generating at 1080p). 

- **Experimental Sage 3 Attention support**: you will need to deserve this one, first you need a Blackwell GPU (RTX50xx) and request an access to Sage 3 Github repo, then you will have to compile Sage 3, install it and cross your fingers ...


*update 8.31: one shouldnt talk about bugs if one doesn't want to attract bugs*


## 🔥 Latest News
### August 29 2025: WanGP v8.21 -  Here Goes Your Weekend

- **InfiniteTalk Video to Video**: this feature can be used for Video Dubbing. Keep in mind that it is a *Sparse Video to Video*, that is internally only image is used by Sliding Window. However thanks to the new *Smooth Transition* mode, each new clip is connected to the previous and all the camera work is done by InfiniteTalk. If you dont get any transition, increase the number of frames of a Sliding Window (81 frames recommended)

- **StandIn**: very light model specialized in Identity Transfer. I have provided two versions of Standin: a basic one derived from the text 2 video model and another based on Vace. If used with Vace, the last reference frame given to Vace will be also used for StandIn

- **Flux ESO**: a new Flux dervied *Image Editing tool*, but this one is specialized both in *Identity Transfer* and *Style Transfer*. Style has to be understood in its wide meaning: give a reference picture of a person and another one of Sushis and you will turn this person into Sushis

### August 24 2025: WanGP v8.1 -  the RAM Liberator

- **Reserved RAM entirely freed when switching models**, you should get much less out of memory related to RAM. I have also added a button in *Configuration / Performance* that will release most of the RAM used by WanGP if you want to use another application without quitting WanGP 
- **InfiniteTalk** support: improved version of Multitalk that supposedly supports very long video generations based on an audio track. Exists in two flavors (*Single Speaker* and *Multi Speakers*) but doesnt seem to be compatible with Vace. One key new feature compared to Multitalk is that you can have different visual shots associated to the same audio: each Reference frame you provide you will be associated to a new Sliding Window. If only Reference frame is provided, it will be used for all windows. When Continuing a video, you can either continue the current shot (no Reference Frame) or add new shots (one or more Reference Frames).\
If you are not into audio, you can use still this model to generate infinite long image2video, just select "no speaker". Last but not least, Infinitetalk works works with all the Loras accelerators.
- **Flux Chroma 1 HD** support: uncensored flux based model and lighter than Flux (8.9B versus 12B) and can fit entirely in VRAM with only 16 GB of VRAM. Unfortunalely it is not distilled and you will need CFG at minimum 20 steps

### August 21 2025: WanGP v8.01 - the killer of seven

- **Qwen Image Edit** : Flux Kontext challenger (prompt driven image edition). Best results (including Identity preservation) will be obtained at 720p. Beyond you may get image outpainting and / or lose identity preservation. Below 720p prompt adherence will be worse. Qwen Image Edit works with Qwen Lora Lightning 4 steps. I have also unlocked all the resolutions for Qwen models. Bonus Zone: support for multiple image compositions but identity preservation won't be as good.
- **On demand Prompt Enhancer** (needs to be enabled in Configuration Tab) that you can use to Enhance a Text Prompt before starting a Generation. You can refine the Enhanced Prompt or change the original Prompt.
- Choice of a **Non censored Prompt Enhancer**. Beware this is one is VRAM hungry and will require 12 GB of VRAM to work
- **Memory Profile customizable per model** : useful to set for instance Profile 3 (preload the model entirely in VRAM) with only Image Generation models, if you have 24 GB of VRAM. In that case Generation will be much faster because with Image generators (contrary to Video generators) as a lot of time is wasted in offloading 
- **Expert Guidance Mode**: change the Guidance during the generation up to 2 times. Very useful with Wan 2.2 Ligthning to reduce the slow motion effect. The idea is to insert a CFG phase before the 2 accelerated phases that follow and have no Guidance. I have added the finetune *Wan2.2 Vace Lightning 3 Phases 14B* with a prebuilt configuration. Please note that it is a 8 steps process although the lora lightning is 4 steps. This expert guidance mode is also available with Wan 2.1.

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

### June 23 2025: WanGP v6.3, Vace Unleashed. Thought we couldnt squeeze Vace even more ?
- Multithreaded preprocessing when possible for faster generations
- Multithreaded frames Lanczos Upsampling as a bonus
- A new Vace preprocessor : *Flow* to extract fluid motion
- Multi Vace Controlnets: you can now transfer several properties at the same time. This opens new possibilities to explore, for instance if you transfer *Human Movement* and *Shapes* at the same time for some reasons the lighting of your character will take into account much more the environment of your character.
- Injected Frames Outpainting, in case you missed it in WanGP 6.21

Don't know how to use all of the Vace features ? Check the Vace Guide embedded in WanGP as it has also been updated.


### June 19 2025: WanGP v6.2, Vace even more Powercharged
👋 Have I told you that I am a big fan of Vace ? Here are more goodies to unleash its power: 
- If you ever wanted to watch Star Wars in 4:3, just use the new *Outpainting* feature and it will add the missing bits of image at the top and the bottom of the screen. The best thing is *Outpainting* can be combined with all the other Vace modifications, for instance you can change the main character of your favorite movie at the same time  
- More processing can combined at the same time  (for instance the depth process can be applied outside the mask)
- Upgraded the depth extractor to Depth Anything 2 which is much more detailed

As a bonus, I have added two finetunes based on the Safe-Forcing technology (which requires only 4 steps to generate a video): Wan 2.1 text2video Self-Forcing and Vace Self-Forcing. I know there is Lora around but the quality of the Lora is worse (at least with Vace) compared to the full model. Don't hesitate to share your opinion about this on the discord server. 
### June 17 2025: WanGP v6.1, Vace Powercharged
👋 Lots of improvements for Vace the Mother of all Models:
- masks can now be combined with on the fly processing of a control video, for instance you can extract the motion of a specific person defined by a mask
- on the fly modification of masks : reversed masks (with the same mask you can modify the background instead of the people covered by the masks), enlarged masks (you can cover more area if for instance the person you are trying to inject is larger than the one in the mask), ...
- view these modified masks directly inside WanGP during the video generation to check they are really as expected
- multiple frames injections: multiples frames can be injected at any location of the video
- expand past videos in on click: just select one generated video to expand it

Of course all these new stuff work on all Vace finetunes (including Vace Fusionix).

Thanks also to Reevoy24 for adding a Notfication sound at the end of a generation and for fixing the background color of the current generation summary.

### June 12 2025: WanGP v6.0
👋 *Finetune models*: You find the 20 models supported by WanGP not sufficient ? Too impatient to wait for the next release to get the support for a newly released model ? Your prayers have been answered: if a new model is compatible with a model architecture supported by WanGP, you can add by yourself the support for this model in WanGP by just creating a finetune model definition. You can then store this model in the cloud (for instance in Huggingface) and the very light finetune definition file can be easily shared with other users. WanGP will download automatically the finetuned model for them.

To celebrate the new finetunes support, here are a few finetune gifts (directly accessible from the model selection menu):
- *Fast Hunyuan Video* : generate model t2v in only 6 steps
- *Hunyuan Vido AccVideo* : generate model t2v in only 5 steps
- *Wan FusioniX*: it is a combo of AccVideo / CausVid ans other models and can generate high quality Wan videos in only 8 steps

One more thing...

The new finetune system can be used to combine complementaty models : what happens when you combine  Fusionix Text2Video and Vace Control Net ?

You get **Vace FusioniX**: the Ultimate Vace Model, Fast (10 steps, no need for guidance) and with a much better quality Video than the original slower model (despite being the best Control Net out there). Here goes one more finetune...

Check the *Finetune Guide* to create finetune models definitions and share them on the WanGP discord server.

### June 11 2025: WanGP v5.5
👋 *Hunyuan Video Custom Audio*: it is similar to Hunyuan Video Avatar excpet there isn't any lower limit on the number of frames and you can use your reference images in a different context than the image itself\
*Hunyuan Video Custom Edit*: Hunyuan Video Controlnet, use it to do inpainting and replace a person in a video while still keeping his poses. Similar to Vace but less restricted than the Wan models in terms of content...

### June 6 2025: WanGP v5.41
👋 Bonus release: Support for **AccVideo** Lora to speed up x2 Video generations in Wan models. Check the Loras documentation to get the usage instructions of AccVideo.

### June 6 2025: WanGP v5.4
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