# WanGP

-----
<p align="center">
<b>Wan2.1 GP by DeepBeepMeep based on Wan2.1's Alibaba: Open and Advanced Large-Scale Video Generative Models for the GPU Poor</b>
</p>

**NEW Discord Server to get Help from Other Users and show your Best Videos:** https://discord.gg/g7efUW9jGV



## 🔥 Latest News!!
* April 25 2025: 👋 Wan 2.1GP v4.3: Added preview mode and support for Sky Reels v2 Diffusion Forcing for high quality "infinite length videos" (see Window Sliding section below) 
* April 18 2025: 👋 Wan 2.1GP v4.2: FLF2V model support, official support from Wan for image2video start and end frames specialized for 720p.  
* April 17 2025: 👋 Wan 2.1GP v4.1: Recam Master model support, view a video from a different angle. The video to process must be at least 81 frames long and you should set at least 15 steps denoising to get good results.
* April 13 2025: 👋 Wan 2.1GP v4.0: lots of goodies for you !
    - A new UI, tabs were replaced by a Dropdown box to easily switch models
    - A new queuing system that lets you stack in a queue as many text2video, imag2video tasks, ... as you want. Each task can rely on complete different generation parameters (different number of frames, steps, loras, ...). Many thanks to **Tophness** for being a big contributor on this new feature
    - Temporal upsampling (Rife) and spatial upsampling (Lanczos) for a smoother video (32 fps or 64 fps) and to enlarge your video by x2 or x4. Check these new advanced options.
    - Wan Vace Control Net support : with Vace you can inject in the scene people or objects, animate a person, perform inpainting or outpainting, continue a video, ... I have provided an introduction guide below.
    - Integrated *Matanyone* tool directly inside WanGP so that you can create easily inpainting masks used in Vace
    - Sliding Window generation for Vace, create windows that can last dozen of seconds
    - New optimisations for old generation GPUs: Generate 5s (81 frames, 15 steps) of Vace 1.3B with only 5GB and in only 6 minutes on a RTX 2080Ti and 5s of t2v 14B in less than 10 minutes.

* Mar 27 2025: 👋 Added support for the new Wan Fun InP models (image2video). The 14B Fun InP has probably better end image support but unfortunately existing loras do not work so well with it. The great novelty is the Fun InP image2 1.3B model : Image 2 Video is now accessible to even lower hardware configuration. It is not as good as the 14B models but very impressive for its size. You can choose any of those models in the Configuration tab. Many thanks to the VideoX-Fun team  (https://github.com/aigc-apps/VideoX-Fun)
* Mar 26 2025: 👋 Good news ! Official support for RTX 50xx please check the installation instructions below. 
* Mar 24 2025: 👋 Wan2.1GP v3.2: 
    - Added Classifier-Free Guidance Zero Star. The video should match better the text prompt (especially with text2video) at no performance cost: many thanks to the **CFG Zero * Team:**\
    Dont hesitate to give them a star if you appreciate the results:  https://github.com/WeichenFan/CFG-Zero-star 
    - Added back support for Pytorch compilation with Loras. It seems it had been broken for some time
    - Added possibility to keep a number of pregenerated videos in the Video Gallery (useful to compare outputs of different settings)
    You will need one more *pip install -r requirements.txt*
* Mar 19 2025: 👋 Wan2.1GP v3.1: Faster launch and RAM optimizations (should require less RAM to run)\ 
    You will need one more *pip install -r requirements.txt*
* Mar 18 2025: 👋 Wan2.1GP v3.0: 
    - New Tab based interface, yon can switch from i2v to t2v conversely without restarting the app
    - Experimental Dual Frames mode for i2v, you can also specify an End frame. It doesn't always work, so you will need a few attempts.
    - You can save default settings in the files *i2v_settings.json* and *t2v_settings.json* that will be used when launching the app (you can also specify the path to different settings files)
    - Slight acceleration with loras\
    You will need one more *pip install -r requirements.txt*
    Many thanks to *Tophness* who created the framework (and did a big part of the work) of the multitabs and saved settings features 
* Mar 18 2025: 👋 Wan2.1GP v2.11: Added more command line parameters to prefill the generation settings + customizable output directory and choice of type of metadata for generated videos. Many thanks to *Tophness* for his contributions. You will need one more *pip install -r requirements.txt* to reflect new dependencies\
* Mar 18 2025: 👋 Wan2.1GP v2.1: More Loras !: added support for 'Safetensors' and 'Replicate' Lora formats.\
You will need to refresh the requirements with a *pip install -r requirements.txt*
* Mar 17 2025: 👋 Wan2.1GP v2.0: The Lora festival continues:
    - Clearer user interface
    - Download 30 Loras in one click to try them all (expand the info section)
    - Very to use Loras as now Lora presets can input the subject (or other need terms) of the Lora so that you dont have to modify manually a prompt 
    - Added basic macro prompt language to prefill prompts with differnent values. With one prompt template, you can generate multiple prompts.
    - New Multiple images prompts: you can now combine any number of images with any number of text promtps (need to launch the app with --multiple-images)
    - New command lines options to launch directly the 1.3B t2v model or the 14B t2v model
* Mar 14, 2025: 👋 Wan2.1GP v1.7: 
    - Lora Fest special edition: very fast loading / unload of loras for those Loras collectors around. You can also now add / remove loras in the Lora folder without restarting the app. You will need to refresh the requirements *pip install -r requirements.txt*
    - Added experimental Skip Layer Guidance (advanced settings), that should improve the image quality at no extra cost. Many thanks to the *AmericanPresidentJimmyCarter* for the original implementation
* Mar 13, 2025: 👋 Wan2.1GP v1.6: Better Loras support, accelerated loading Loras. You will need to refresh the requirements *pip install -r requirements.txt*
* Mar 10, 2025: 👋 Wan2.1GP v1.5: Official Teacache support + Smart Teacache (find automatically best parameters for a requested speed multiplier), 10% speed boost with no quality loss, improved lora presets (they can now  include prompts and comments to guide the user)
* Mar 07, 2025: 👋 Wan2.1GP v1.4: Fix Pytorch compilation, now it is really 20% faster when activated
* Mar 04, 2025: 👋 Wan2.1GP v1.3: Support for Image to Video with multiples images for different images / prompts combinations (requires *--multiple-images* switch), and added command line *--preload x*  to preload in VRAM x MB of the main diffusion model if you find there is too much unused VRAM and you want to (slightly) accelerate the generation process.
If you upgrade you will need to do a 'pip install -r requirements.txt' again.
* Mar 04, 2025: 👋 Wan2.1GP v1.2: Implemented tiling on VAE encoding and decoding. No more VRAM peaks at the beginning and at the end 
* Mar 03, 2025: 👋 Wan2.1GP v1.1: added Tea Cache support for faster generations:  optimization of kijai's implementation (https://github.com/kijai/ComfyUI-WanVideoWrapper/) of teacache (https://github.com/ali-vilab/TeaCache)  
* Mar 02, 2025: 👋 Wan2.1GP by DeepBeepMeep v1 brings: 
    - Support for all Wan including the Image to Video model
    - Reduced memory consumption by 2, with possiblity to generate more than 10s of video at 720p with a RTX 4090 and 10s of video at 480p with less than 12GB of VRAM. Many thanks to REFLEx (https://github.com/thu-ml/RIFLEx) for their algorithm that allows generating nice looking video longer than 5s.
    - The usual perks: web interface, multiple generations, loras support, sage attebtion, auto download of models, ...

* Feb 25, 2025: 👋 We've released the inference code and weights of Wan2.1.
* Feb 27, 2025: 👋 Wan2.1 has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/). Enjoy!


## Features
*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly on any GPU.*

This version has the following improvements over the original Alibaba model:
- Reduce greatly the RAM requirements and VRAM requirements 
- Much faster thanks to compilation and fast loading / unloading
- Multiple profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM)
- Autodownloading of the needed model files
- Improved gradio interface with progression bar and more options
- Multiples prompts / multiple generations per prompt
- Support multiple pretrained Loras with 32 GB of RAM or less
- Much simpler installation


This fork by DeepBeepMeep is an integration of the mmpg module on the original model

It is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with changing only a few lines of code in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp

You will find the original Wan2.1 Video repository here: https://github.com/Wan-Video/Wan2.1

 


## Installation Guide for Linux and Windows for GPUs up to RTX40xx

**If you are looking for a one click installation, just go to the Pinokio App store : https://pinokio.computer/**\
Otherwise you will find the instructions below:

This app has been tested on Python 3.10 / 2.6.0  / Cuda 12.4.

```shell
# 0 Download the source and create a Python 3.10.9 environment using conda or create a venv using python
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp

# 1 Install pytorch 2.6.0
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124  

# 2. Install pip dependencies
pip install -r requirements.txt

# 3.1 optional Sage attention support (30% faster)
# Windows only: extra step only needed for windows as triton is included in pytorch with the Linux version of pytorch
pip install triton-windows 
# For both Windows and Linux
pip install sageattention==1.0.6 


# 3.2 optional Sage 2 attention support (40% faster)
# Windows only
pip install triton-windows 
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu126torch2.6.0-cp310-cp310-win_amd64.whl
# Linux only (sorry only manual compilation for the moment, but is straight forward with Linux)
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .

# 3.3 optional Flash attention support (easy to install on Linux but may be complex on Windows as it will try to compile the cuda kernels)
pip install flash-attn==2.7.2.post1

```

Note pytorch *sdpa attention* is available by default. It is worth installing *Sage attention* (albout not as simple as it sounds) because it offers a 30% speed boost over *sdpa attention* at a small quality cost.
In order to install Sage, you will need to install also Triton. If Triton is installed you can turn on *Pytorch Compilation* which will give you an additional 20% speed boost and reduced VRAM consumption.

## Installation Guide for Linux and Windows for GPUs up to RTX50xx
RTX50XX are only supported by pytorch starting from pytorch 2.7.0 which is still in beta. Therefore this version may be less stable.\
It is important to use Python 3.10 otherwise the pip wheels may not be compatible.
```
# 0 Download the source and create a Python 3.10.9 environment using conda or create a venv using python
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp

# 1 Install pytorch 2.7.0:
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128

# 2. Install pip dependencies
pip install -r requirements.txt

# 3.1 optional Sage attention support (30% faster)
# Windows only: extra step only needed for windows as triton is included in pytorch with the Linux version of pytorch
pip install triton-windows 
# For both Windows and Linux
pip install sageattention==1.0.6 


# 3.2 optional Sage 2 attention support (40% faster)
# Windows only
pip install triton-windows 
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl 

# Linux only (sorry only manual compilation for the moment, but is straight forward with Linux)
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .
```

## Run the application

### Run a Gradio Server on port 7860 (recommended)

To run the text to video generator (in Low VRAM mode): 
```bash
python wgp.py
#or
python wgp.py --t2v #launch the default text 2 video model
#or
python wgp.py --t2v-14B #for the 14B model 
#or
python wgp.py --t2v-1-3B #for the 1.3B model

```

To run the image to video generator (in Low VRAM mode): 
```bash
python wgp.py --i2v
```
To run the 1.3B Fun InP image to video generator (in Low VRAM mode): 
```bash
python wgp.py --i2v-1-3B
```

To be able to input multiple images with the image to video generator:
```bash
python wgp.py --i2v --multiple-images
```

Within the application you can configure which video generator will be launched without specifying a command line switch.

To run the application while loading entirely the diffusion model in VRAM (slightly faster but requires 24 GB of VRAM for a 8 bits quantized 14B model )
```bash
python wgp.py --profile 3
```

**Trouble shooting**:\
If you have installed Sage attention, it may seem that it works because *pip install sageattention* didn't produce and error or because sage is offered as on option but in fact it doesnt work :  in order to be fully operatioal Sage needs to compile its triton kernels the first time it is run (that is the first time you try to generate a video).

Sometime fixing Sage compilation is easy (clear the triton cache, check triton is properly installed) sometime it is simply not possible because Sage is not supported on some older GPUs

Therefore you may have no choice but to fallback to sdpa attention, to do so:
- In the configuration menu inside the application, switch "Attention mode" to "sdpa"
or
- Launch the application this way:
```bash
python wgp.py --attention sdpa
```

### Loras support


Every lora stored in the subfoler 'loras' for t2v and 'loras_i2v' will be automatically loaded. You will be then able to activate / desactive any of them when running the application by selecting them in the area below "Activated Loras" .

If you want to manage in different areas Loras for the 1.3B model and the 14B as they are not compatible, just create the following subfolders:
- loras/1.3B
- loras/14B


For each activated Lora, you may specify a *multiplier* that is one float number that corresponds to its weight (default is 1.0) .The multipliers for each Lora should be separated by a space character or a carriage return. For instance:\
*1.2 0.8* means that the first lora will have a 1.2 multiplier and the second one will have 0.8. 

Alternatively for each Lora's multiplier you may specify a list of float numbers multipliers  separated by a "," (no space) that gives the evolution of this Lora's multiplier over the steps. For instance let's assume there are 30 denoising steps and the multiplier is *0.9,0.8,0.7* then for the steps ranges 0-9, 10-19 and 20-29 the Lora multiplier will be respectively 0.9, 0.8 and 0.7. 

If multiple Loras are defined, remember that each multiplier associated to different Loras should be separated by a space or a carriage return, so we can specify the evolution of multipliers for multiple Loras. For instance for two Loras (press Shift Return to force a carriage return):

```
0.9,0.8,0.7 
1.2,1.1,1.0
```
You can edit, save or delete Loras presets (combinations of loras with their corresponding multipliers) directly from the gradio Web interface. These presets will save the *comment* part of the prompt that should contain some instructions how to use the corresponding the loras (for instance by specifying a trigger word or providing an example).A comment in the prompt is a line that starts that a #. It will be ignored by the video generator. For instance:

```
# use they keyword ohnvx to trigger the Lora*
A ohnvx is driving a car
```
Each preset, is a file with ".lset" extension stored in the loras directory and can be shared with other users

Last but not least you can pre activate Loras corresponding and prefill a prompt (comments only or full prompt) by specifying a preset when launching the gradio server:
```bash
python wgp.py --lora-preset  mylorapreset.lset # where 'mylorapreset.lset' is a preset stored in the 'loras' folder
```

You will find prebuilt Loras on https://civitai.com/ or you will be able to build them with tools such as kohya or onetrainer.

### Macros (basic)
In *Advanced Mode*, you can starts prompt lines with a "!" , for instance:\ 
```
! {Subject}="cat","woman","man", {Location}="forest","lake","city", {Possessive}="its", "her", "his"
In the video, a {Subject} is presented. The {Subject} is in a {Location} and looks at {Possessive} watch.
```

This will create automatically 3 prompts that will cause the generation of 3 videos:
```
In the video, a cat is presented. The cat is in a forest and looks at its watch.
In the video, a man is presented. The man is in a lake  and looks at his watch.
In the video, a woman is presented. The woman is in a city and looks at her watch.
```

You can define multiple lines of macros. If there is only one macro line, the app will generate a simple user interface to enter the macro variables when getting back to *Normal Mode* (advanced mode turned off)

### VACE ControlNet introduction

Vace is a ControlNet 1.3B text2video model that allows you to do Video to Video and Reference to Video (inject your own images into the output video). So with Vace you can inject in the scene people or objects of your choice, animate a person, perform inpainting or outpainting, continue a video, ... 

First you need to select the Vace 1.3B model in the Drop Down box at the top. Please note that Vace works well for the moment only with videos up to 5s (81 frames).

Beside the usual Text Prompt, three new types of visual hints can be provided (and combined !):
- a Control Video: Based on your choice, you can decide to transfer the motion, the depth in a new Video. You can tell WanGP to use only the first n frames of Control Video and to extrapolate the rest. You can also do inpainting ). If the video contains area of grey color 127, they will be considered as masks and will be filled based on the Text prompt of the reference Images. 

- reference Images: Use this to inject people or objects of your choice in the video. You can select multiple reference Images. The integration of the image is more efficient if the background is replaced by the full white color. You can do that with your preferred background remover or use the built in background remover by checking the box *Remove background*

- a Video Mask
This offers a stronger mechanism to tell Vace which parts should be kept (black) or replaced (white). You can do as well inpainting / outpainting, fill the missing part of a video more efficientlty with just the video hint. If a video mask is white, it will be generated so with black frames at the beginning and at the end and the rest white, you could generate the missing frames in between.


Examples:
- Inject people and / objects into a scene describe by a text prompt: Ref. Images + text Prompt
- Animate a character described in a text prompt: a Video of person moving + text Prompt
- Animate a character of your choice (motion transfer) : Ref Images + a Video of person moving + text Prompt
- Change the style of a scene (depth transfer): a Video that contains objects / person at differen depths + text Prompt


There are lots of possible combinations. Some of them require to prepare some materials (masks on top of video, full masks, etc...).

Vace provides on its github (https://github.com/ali-vilab/VACE/tree/main/vace/gradios) annotators / preprocessors Gradio tool that can help you build some of these materials depending on the task you want to achieve.

There is also a guide that describes the various combination of hints (https://github.com/ali-vilab/VACE/blob/main/UserGuide.md).Good luck ! 

It seems you will get better results if you turn on "Skip Layer Guidance" with its default configuration

### VACE and Sky Reels v2 Diffusion Forcing Slidig Window
With this mode (that works for the moment only with Vace and Sky Reels v2) you can merge mutiple Videos to form a very long video (up to 1 min). 

When combined with Vace this feature can use the same control video to generate the full Video that results from concatenining the different windows. For instance the first 0-4s of the control video will be used to generate the first window then the next 4-8s of the control video will be used to generate the second window, and so on. So if your control video contains a person walking, your generate video could contain up to one minute of this person walking.

When combined with Sky Reels V2, you can extend an existing video indefinetely.

Sliding Windows are turned on by default and are triggered as soon as you try to generate a Video longer than the Window Size. You can go in the Advanced Settings Tab *Sliding Window* to set this Window Size. You can make the Video even longer during the generation process by adding one more Window to generate each time you click "Extend the Video Sample, Please !" button.

Although the window duration is set by the *Sliding Window Size* form field, the actual number of frames generated by each iteration will be less, because of the *overlap frames* and *discard last frames*: 
- *overlap frames* : the first frames of a new window are filled with last frames of the previous window in order to ensure continuity between the two windows
- *discard last frames* : quite often (Vace model Only) the last frames of a window have a worse quality. You can decide here how many ending frames of a new window should be dropped.
s
Number of Generated Frames = [Number of Windows - 1] * ([Window Size] - [Overlap Frames] - [Discard Last Frames]) +  [Window Size]

Experimental: if your prompt is broken into multiple lines (each line separated by a carriage return), then each line of the prompt will be used for a new window. If there are more windows to generate than prompt lines, the last prompt line will be repeated. 

### Command line parameters for Gradio Server
--i2v : launch the image to video generator\
--t2v : launch the text to video generator (default defined in the configuration)\
--t2v-14B : launch the 14B model text to video generator\
--t2v-1-3B : launch the 1.3B model text to video generator\
--i2v-14B : launch the 14B model image to video generator\
--i2v-1-3B : launch the Fun InP 1.3B model image to video generator\
--vace : launch the Vace ControlNet 1.3B model image to video generator\
--quantize-transformer bool: (default True) : enable / disable on the fly transformer quantization\
--lora-dir path : Path of directory that contains Loras in diffusers / safetensor format\
--lora-preset preset : name of preset gile (without the extension) to preload
--verbose level : default (1) : level of information between 0 and 2\
--server-port portno : default (7860) : Gradio port no\
--server-name name : default (localhost) : Gradio server name\
--open-browser : open automatically Browser when launching Gradio Server\
--lock-config : prevent modifying the video engine configuration from the interface\
--share : create a shareable URL on huggingface so that your server can be accessed remotely\
--multiple-images : allow the users to choose multiple images as different starting points for new videos\
--compile : turn on pytorch compilation\
--attention mode: force attention mode among, sdpa, flash, sage, sage2\
--profile no : default (4) : no of profile between 1 and 5\
--preload no : number in Megabytes to preload partially the diffusion model in VRAM , may offer speed gains on older hardware, on recent hardware (RTX 30XX, RTX40XX and RTX50XX) speed gain is only 10% and not worth it. Works only with profile 2 and 4.\
--seed no : set default seed value\
--frames no : set the default number of frames to generate\
--steps no : set the default number of denoising steps\
--teacache speed multiplier: Tea cache speed multiplier,  choices=["0", "1.5", "1.75", "2.0", "2.25", "2.5"]\
--slg : turn on skip layer guidance for improved quality\
--check-loras : filter loras that are incompatible (will take a few seconds while refreshing the lora list or while starting the app)\
--advanced : turn on the advanced mode while launching the app\
--listen : make server accessible on network\
--gpu device : run Wan on device for instance "cuda:1"\
--settings: path a folder that contains the default settings for all the models\
--fp16: force to use fp16 versions of models instead of bf16 versions\
--perc-reserved-mem-max float_less_than_1 : max percentage of RAM to allocate to reserved RAM, allow faster transfers RAM<->VRAM. Value should remain below 0.5 to keep the OS stable\
--theme theme_name: load the UI with the specified Theme Name, so far only two are supported, "default" and "gradio". You may submit your own nice looking Gradio theme and I will add them

### Profiles (for power users only)
You can choose between 5 profiles, but two are really relevant here :
- LowRAM_HighVRAM  (3): loads entirely the model in VRAM if possible, slightly faster, but less VRAM available for the video data after that
- LowRAM_LowVRAM  (4): loads only the part of the model that is needed, low VRAM and low RAM requirement but slightly slower

You can adjust the number of megabytes to preload a model, with --preload nnn (nnn is the number of megabytes to preload)
### Other Models for the GPU Poor

- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP :\
One of the best open source Text to Video generator

- Hunyuan3D-2GP: https://github.com/deepbeepmeep/Hunyuan3D-2GP :\
A great image to 3D and text to 3D tool by the Tencent team. Thanks to mmgp it can run with less than 6 GB of VRAM

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP :\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP :\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).

- OminiControlGP: https://github.com/deepbeepmeep/OminiControlGP :\
A Flux derived application very powerful that can be used to transfer an object of your choice in a prompted scene. With mmgp you can run it with only 6 GB of VRAM.

- YuE GP: https://github.com/deepbeepmeep/YuEGP :\
A great song generator (instruments + singer's voice) based on prompted Lyrics and a genre description. Thanks to mmgp you can run it with less than 10 GB of VRAM without waiting forever.


