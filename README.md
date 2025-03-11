# Wan2.1 GP


<p align="center">
    💜 <a href=""><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.1">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper (Coming soon)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wanxai.com">Blog</a> &nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat Group</a>&nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://discord.gg/p5XbdQV7">Discord</a>&nbsp&nbsp
<br>

-----
<p align="center">
<b>Wan2.1 GP by DeepBeepMeep based on Wan2.1's Alibaba: Open and Advanced Large-Scale Video Generative Models for the GPU Poor</b>
</p>

In this repository, we present **Wan2.1**, a comprehensive and open suite of video foundation models that pushes the boundaries of video generation. **Wan2.1** offers these key features:
- 👍 **SOTA Performance**: **Wan2.1** consistently outperforms existing open-source models and state-of-the-art commercial solutions across multiple benchmarks.
- 👍 **Supports Consumer-grade GPUs**: The T2V-1.3B model requires only 8.19 GB VRAM, making it compatible with almost all consumer-grade GPUs. It can generate a 5-second 480P video on an RTX 4090 in about 4 minutes (without optimization techniques like quantization). Its performance is even comparable to some closed-source models.
- 👍 **Multiple Tasks**: **Wan2.1** excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio, advancing the field of video generation.
- 👍 **Visual Text Generation**: **Wan2.1** is the first video model capable of generating both Chinese and English text, featuring robust text generation that enhances its practical applications.
- 👍 **Powerful Video VAE**: **Wan-VAE** delivers exceptional efficiency and performance, encoding and decoding 1080P videos of any length while preserving temporal information, making it an ideal foundation for video and image generation.


## 🔥 Latest News!!
* Mar 10, 2025: 👋 Wan2.1GP v1.5: Official Teachache support + Smart Teacache (find automatically best parameters for a requested speed multiplier), 10% speed boost with no quality loss, improved lora presets (they can now  include prompts and comments to guide the user)
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

 


## Installation Guide for Linux and Windows

**If you are looking for a one click installation, just go to the Pinokio App store : https://pinokio.computer/**

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

# 3.1 optional Sage attention support (30% faster, easy to install on Linux but much harder on Windows)
pip install sageattention==1.0.6 

# or for Sage Attention 2 (40% faster, sorry only manual compilation for the moment)
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .

# 3.2 optional Flash attention support (easy to install on Linux but much harder on Windows)
pip install flash-attn==2.7.2.post1

```

Note pytorch *sdpa attention* is available by default. It is worth installing *Sage attention* (albout not as simple as it sounds) because it offers a 30% speed boost over *sdpa attention* at a small quality cost.
In order to install Sage, you will need to install also Triton. If Triton is installed you can turn on *Pytorch Compilation* which will give you an additional 20% speed boost and reduced VRAM consumption.

### Ready to use python wheels for Windows users
I provide here links to simplify the installation for Windows users with Python 3.10 / Pytorch 2.51 / Cuda 12.4. I won't be able to provide support neither guarantee they do what they should do.
- Triton attention (needed for *pytorch compilation* and *Sage attention*)
```
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl # triton for pytorch 2.6.0
```

- Sage attention
```
pip install https://github.com/deepbeepmeep/SageAttention/raw/refs/heads/main/releases/sageattention-2.1.0-cp310-cp310-win_amd64.whl # for pytorch 2.6.0 (experimental, if it works, otherwise you you will need to install and compile manually, see above) 
 
```

## Run the application

### Run a Gradio Server on port 7860 (recommended)

To run the text to video generator (in Low VRAM mode): 
```bash
python gradio_server.py
#or
python gradio_server.py --t2v

```

To run the image to video generator (in Low VRAM mode): 
```bash
python gradio_server.py --i2v
```

To be able to input multiple images with the image to video generator:
```bash
python gradio_server.py --i2v --multiple-images
```

Within the application you can configure which video generator will be launched without specifying a command line switch.

To run the application while loading entirely the diffusion model in VRAM (slightly faster but requires 24 GB of VRAM for a 8 bits quantized 14B model )
```bash
python gradio_server.py --profile 3
```

**Trouble shooting**:\
If you have installed Sage attention, it may seem that it works because *pip install sageattention* didn't produce and error or because sage is offered as on option but in fact it doesnt work :  in order to be fully operatioal Sage needs to compile its triton kernels the first time it is run (that is the first time you try to generate a video).

Sometime fixing Sage compilation is easy (clear the triton cache, check triton is properly installed) sometime it is simply not possible because Sage is not supported on some older GPUs

Therefore you may have no choice but to fallback to sdpa attention, to do so:
- In the configuration menu inside the application, switch "Attention mode" to "sdpa"
or
- Launch the application this way:
```bash
python gradio_server.py --attention sdpa
```

### Loras support


Every lora stored in the subfoler 'loras' for t2v and 'loras_i2v' will be automatically loaded. You will be then able to activate / desactive any of them when running the application by selecting them in the area below "Activated Loras" .

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
python gradio_server.py --lora-preset  mylorapreset.lset # where 'mylorapreset.lset' is a preset stored in the 'loras' folder
```

You will find prebuilt Loras on https://civitai.com/ or you will be able to build them with tools such as kohya or onetrainer.


### Command line parameters for Gradio Server
--i2v : launch the image to video generator\
--t2v : launch the text to video generator\
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
--preload no : number in Megabytes to preload partially the diffusion model in VRAM , may offer slight speed gains especially on older hardware. Works only with profile 2 and 4.

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


