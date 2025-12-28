# Loras Guide

Loras (Low-Rank Adaptations) allow you to customize video generation models by adding specific styles, characters, or effects to your videos.

## Directory Structure

Loras are organized in different folders based on the model they're designed for:

All loras now live under the single `loras/` root:

### Wan Models
- `loras/wan/`    - Wan t2v (14B / general) loras
- `loras/wan_5B/` - Wan 5B loras
- `loras/wan_1.3B/` - Wan 1.3B loras
- `loras/wan_i2v/` - Wan i2v loras

### Other Models
- `loras/hunyuan/` - Hunyuan Video t2v loras
- `loras/hunyuan/1.5/` - Loras specifically for Hunyuan 1.5 models
- `loras/hunyuan_i2v/` - Hunyuan Video i2v loras
- `loras/ltxv/` - LTX Video loras
- `loras/flux/` and `loras/flux2/` - Flux loras
- `loras/qwen/` - Qwen loras
- `loras/z_image/` - Z-Image loras
- `loras/tts/` - Chatterbox / TTS presets

## Custom Lora Directory

You can specify custom lora directories when launching the app:

```bash
# Use shared lora directory for Wan t2v and i2v
python wgp.py --lora-dir /path/to/loras/wan --lora-dir-i2v /path/to/loras/wan_i2v

# Specify different directories for different models
python wgp.py --lora-dir-hunyuan /path/to/loras/hunyuan --lora-dir-ltxv /path/to/loras/ltxv
```

## Using Loras

### Basic Usage

1. Place your lora files in the appropriate directory
2. Launch WanGP
3. In the Advanced Tab, select the "Loras" section
4. Check the loras you want to activate
5. Set multipliers for each lora (default is 1.0 if multiplier is not mentioned)

If you store loras in the loras folder once WanGP has been launched, click the *Refresh* button at the top so that it can become selectable.

### Autodownload of Loras
WanGP will try to remember where a Lora was obtained and will store the corresponding Download URL in the Generation settings that are embedded in the Generated Video. This is useful to share this information or to easily recover lost loras after a reinstall.

This works very well if the Loras are stored in repositories such *Hugging Face* but won't work for the moment for Loras that requires a Login (like *CivitAi*) to be downloaded.

WanGP will update its internal URL Lora Cache whenener one of this events will occur:
- when applying or importing a *Accelerator Profile*, *Settings* or *Lset* file that contains Loras with full URLs (not just local paths) 
- when extracting the settings of a Video that was generated with Loras and that contained the full Loras URLs
- when downloading manually a Lora using the *Download Lora* button at the bottom

So the more you use WanGP the more the URL cache File will get updated. The file is *loras_url_cache.json* and is located in the root folder of WanGP.

You can delete this file with any risk if needed or share it with friends to save them time locating the Loras. You will need to restart WanGP if you modify manually this file or delete it.

### Lora Multipliers

Multipliers control the strength of each lora's effect:

#### Simple Multipliers
```
1.2 0.8
```
- First lora: 1.2 strength
- Second lora: 0.8 strength

#### Time-based and Phase-based Multipliers
For dynamic effects over generation steps, use comma-separated values:
```
0.9,0.8,0.7
1.2,1.1,1.0
```
- For 30 steps: steps 0-9 use first value, 10-19 use second, 20-29 use third
- First lora: 0.9 → 0.8 → 0.7
- Second lora: 1.2 → 1.1 → 1.0

With models like Wan 2.2 that uses internally two diffusion models (*High noise* / *Low Noise*) you can specify which Loras you want to be applied for a specific phase by separating each phase with a ";".

For instance, if you want to disable a lora for phase *High Noise* and enables it only for phase *Low Noise*:
```
0;1
```

Also with Wan 2.2, if you have two loras and you want the first one to be applied only during the High noise and the second one during the Low noise phase:
```
1;0 0;1 
```

As usual, you can use any float for a multiplier and have a multiplier varries throughout one phase for one Lora:
```
0.9,0.8;1.2,1.1,1
```
In this example multiplier 0.9 and 0.8 will be used during the *High Noise* phase and 1.2, 1.1 and 1 during the *Low Noise* phase.

Here is another example for two loras:
```
0.9,0.8;1.2,1.1,1
0.5;0,0.7
```

If one of several of your Lora multipliers are phased based (that is with a ";") and there are also Loras Multipliers that are only time based  (don't have a ";" but have a ",") the time only multiplier will ignore the phases. For instance, let's assume we have a 6 steps denoising process in the following example:

```
1;0
0;1 
0.8,0.7,0.5
```
Here the first lora will be as expected only used with the High Noise model and the second lora only used with the Low noise model. However for the third Lora: for steps 1-2 the multiplier will be (regadless of the phase) 0.8 then for steps 3-4 the multiplier will be 0.7 and finally for steps 5-6 the multiplier will be 0.5

You can use phased Lora multipliers even if have a single model (that is without any High / Low models) as Lora multiplier phases are aligned with Guidance phases. Let's assume you have defined 3 guidance phases (for instance guidance=3, then guidance=1.5 and at last guidance=1 ):
```
0;1;0 
0;0;1 
```
In that case no lora will be applied during the first phase when guidance is 3. Then the fist lora will be only used when guidance is 1.5 and the second lora only when guidance is 1.

Best of all you can combine 3 guidance phases with High / Low models. Let's take this practical example with *Lightning 4/8 steps loras accelerators for Wan 2.2* where we want to increase the motion by adding some guidance at the very beginning (in that case a first phase that lasts only 1 step should be sufficient): 
```
Guidances: 3.5, 1 and 1
Model transition: Phase 2-3
Loras Multipliers: 0;1;0 0;0;1 
```
Here during the first phase with guidance 3.5, the High model will be used but there won't be any lora at all. Then during phase 2 only the High lora will be used (which requires to set the guidance to 1). At last in phase 3 WanGP will switch to the Low model and then only the Low lora will be used.

*Note that the syntax for multipliers can also be used in a Finetune model definition file (except that each multiplier definition is a string in a json list)*
## Lora Presets (.Lset file)

Lora Presets contains all the information needed to use a Lora or a combination of Loras:
- The full download URLs of the Loras
- Default Loras Multipliers
- Sample Prompt to use the Loras with their corresponding *Trigger Words* (usually as comments)
Optionaly they may contain advanced prompts with macros to generate automatically Prompt using keywords.

A Lora Preset is a text file of only of few kilobytes and can be easily shared between users. Don't hesitate to use this format if you have created a Lora.

### Creating Presets
1. Configure your loras and multipliers
2. Write a prompt with comments lines starting with # that contains instructions
3. Save as a preset with `.lset` extension by clicking the *Save* button at the top, select *Save Only Loras & Full Prompt* and finally click *Go Ahead Save it!*

### Example Lora Preset Prompt
```
# Use the keyword "ohnvx" to trigger the lora
A ohnvx character is driving a car through the city
```

Using a macro (check the doc below), the user will just have to enter two words and the Prompt will be generated for him:
```
! {Person}="man" : {Object}="car"
This {Person} is cleaning his {Object}.
```


### Managing Loras Presets (.lset Files)
- Edit, save, or delete presets directly from the web interface
- Presets include comments with usage instructions
- Share `.lset` files with other users (make sure the full Loras URLs are in it)

A *.lset* file may contain only local paths to the Loras if WanGP doesn't know where you got it. You can edit the .lset file with a text editor and replace the local path with its URL. If you store your Lora in Hugging Face, you can easily obtain its URL by selecting the file and clicking *Copy Download Link*.

To share a *.Lset* file you will need (for the moment) to get it directly in the Lora folder where it is stored.

## Supported Formats

WanGP supports multiple most lora formats:
- **Safetensors** (.safetensors)
- **Replicate** format
- ...


## Loras Accelerators
Most Loras are used to apply a specific style or to alter the content of the output of the generated video.
However some Loras have been designed to tranform a model into a distilled model which requires fewer steps to generate a video.
Loras accelerators usually require to the set the Guidance to 1. Don't forget to do it as not only the quality of the generate video will be bad but it will two times slower.

You will find most *Loras Accelerators* below:
- Wan 2.1
https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/loras_accelerators
- Wan 2.2
https://huggingface.co/DeepBeepMeep/Wan2.2/tree/main/loras_accelerators
- Qwen:
https://huggingface.co/DeepBeepMeep/Qwen_image/tree/main/loras_accelerators


### Setup Instructions
There are three ways to setup Loras accelerators:
1) **Finetune with Embedded Loras Accelerators**
Some models Finetunes such as *Vace FusioniX* or *Vace Coctail* have the Loras Accelerators already set up in their own definition and you won't have to do anything as they will be downloaded with the Finetune.

2) **Accelerators Profiles**
Predefined *Accelerator Profiles* can be selected using the *Settings* Dropdown box at the top. The choices of Accelerators will depend on the models. No accelerator will be offered if the finetune / model is already accelerated. Just click *Apply* and the Accelerators Loras will be setup in the Loras tab at the bottom. Any missing Lora will be downloaded automatically the first time you try to generate a Video. Be aware that when applying an *Accelerator Profile*, inputs such as *Activated Loras*, *Number of Inference Steps*, ... will be updated. However if you have already existing Loras set up (that are non Loras Accelerators) they will be preserved so that you can easily switch between Accelerators Profiles. 

You will see the "|" character at the end of the Multipliers text input associated to Loras Accelerators. It plays the same role than the Space character to separate Multipliers except it tells WanGP where the Loras Accelerators multipliers end so that it can merge Loras Accelerators with Non Loras Accelerators.

3) **Manual Install**
- Download the Lora
- Place it in the Lora Directory of the correspondig model
- Configure the Loras Multipliers, CFG as described in the later sections

## FusioniX (or FusionX) Lora for Wan 2.1 / Wan 2.2 
If you need just one Lora accelerator use this one. It is a combination of multiple Loras acelerators (including Causvid below) and style loras. It will not only accelerate the video generation but it will also improve the quality. There are two versions of this lora whether you use it for t2v or i2v

### Usage
1. Select a Wan t2v model (e.g., Wan 2.1 text2video 13B or Vace 13B)
2. Enable Advanced Mode
3. In Advanced Generation Tab:
   - Set Guidance Scale = 1
   - Set Shift Scale = 2
4. In Advanced Lora Tab:
   - Select CausVid Lora
   - Set multiplier to 1
5. Set generation steps from 8-10
6. Generate!

## Self-Forcing lightx2v Lora (Video Generation Accelerator) for Wan 2.1 / Wan 2.2
Selg forcing Lora has been created by Kijai from the Self-Forcing lightx2v distilled Wan model and can generate videos with only 2 steps and offers also a 2x speed improvement since it doesnt require classifier free guidance. It works on both t2v and i2v models
You will find it under the name of *Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors*
 
### Usage
1. Select a Wan t2v or i2v model (e.g., Wan 2.1 text2video 13B or Vace 13B)
2. Enable Advanced Mode
3. In Advanced Generation Tab:
   - Set Guidance Scale = 1
   - Set Shift Scale = 5
4. In Advanced Lora Tab:
   - Select the Lora above
   - Set multiplier to 1
5. Set generation steps to 2-8
6. Generate!


## CausVid Lora (Video Generation Accelerator) for Wan 2.1 / Wan 2.2
CausVid is a distilled Wan model that generates videos in 4-12 steps with 2x speed improvement.

### Usage
1. Select a Wan t2v model (e.g., Wan 2.1 text2video 13B or Vace 13B)
2. Enable Advanced Mode
3. In Advanced Generation Tab:
   - Set Guidance Scale = 1
   - Set Shift Scale = 7
4. In Advanced Lora Tab:
   - Select CausVid Lora
   - Set multiplier to 0.3
5. Set generation steps to 12
6. Generate!

### CausVid Step/Multiplier Relationship
- **12 steps**: 0.3 multiplier (recommended)
- **8 steps**: 0.5-0.7 multiplier
- **4 steps**: 0.8-1.0 multiplier

*Note: Lower steps = lower quality (especially motion)*


## AccVid Lora (Video Generation Accelerator) for Wan 2.1 / Wan 2.2

AccVid is a distilled Wan model that generates videos with a 2x speed improvement since classifier free guidance is no longer needed (that is cfg = 1).

### Usage
1. Select a Wan t2v model (e.g., Wan 2.1 text2video 13B or Vace 13B) or Wan i2v model
2. Enable Advanced Mode
3. In Advanced Generation Tab:
   - Set Guidance Scale = 1
   - Set Shift Scale = 5
4. The number steps remain unchanged compared to what you would use with the original model but it will be two times faster since classifier free guidance is not needed

## Lightx2v 4 steps Lora (Video Generation Accelerator) for Wan 2.2
This lora is in fact composed of two loras, one for the High model and one for the Low Wan 2.2 model.

You need to select these two loras and set the following Loras multipliers:

```
1;0 0;1  (the High lora should be only enabled when only the High model is loaded, same for the Low lora)
```

Don't forget to set guidance to 1 !
## Qwen Image Lightning 4 steps / Lightning 8 steps
Very powerful lora that you can use to reduce the number of steps from 30 to only 4 !
Just install the lora in *lora_qwen* folder, select the lora and set Guidance to 1 and the number of steps to 4 or 8



https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors

## Performance Tips

### Fast Loading/Unloading
- Loras can be added/removed without restarting the app
- Use the "Refresh" button to detect new loras
- Enable `--check-loras` to filter incompatible loras (slower startup)

### Memory Management
- Loras are loaded on-demand to save VRAM
- Multiple loras can be used simultaneously
- Time-based multipliers don't use extra memory
- The order of Loras doesn't matter (as long as the loras multipliers are in the right order of course !)

## Finding Loras

### Sources
- **[Civitai](https://civitai.com/)** - Large community collection
- **HuggingFace** - Official and community loras
- **Discord Server** - Community recommendations

### Creating Loras
- **Kohya** - Popular training tool
- **OneTrainer** - Alternative training solution
- **Custom datasets** - Train on your own content

## Macro System (Advanced)

Create multiple prompts from templates using macros. This allows you to generate variations of a sentence by defining lists of values for different variables.

**Syntax Rule:**

Define your variables on a single line starting with `!`. Each complete variable definition, including its name and values, **must be separated by a colon (`:`)**.

**Format:**

```
! {Variable1}="valueA","valueB" : {Variable2}="valueC","valueD"
This is a template using {Variable1} and {Variable2}.
```

**Example:**

The following macro will generate three distinct prompts by cycling through the values for each variable.

**Macro Definition:**

```
! {Subject}="cat","woman","man" : {Location}="forest","lake","city" : {Possessive}="its","her","his"
In the video, a {Subject} is presented. The {Subject} is in a {Location} and looks at {Possessive} watch.
```

**Generated Output:**

```
In the video, a cat is presented. The cat is in a forest and looks at its watch.
In the video, a woman is presented. The woman is in a lake and looks at her watch.
In the video, a man is presented. The man is in a city and looks at his watch.
```


## Troubleshooting

### Lora Not Working
0. If it is a lora accelerator, Guidance should be set to 1
1. Check if lora is compatible with your model size (1.3B vs 14B)
2. Verify lora format is supported
3. Try different multiplier values
4. Check the lora was trained for your model type (t2v vs i2v)

### Performance Issues
1. Reduce number of active loras
2. Lower multiplier values
3. Use `--check-loras` to filter incompatible files
4. Clear lora cache if issues persist

### Memory Errors
1. Use fewer loras simultaneously
2. Reduce model size (use 1.3B instead of 14B)
3. Lower video resolution or frame count
4. Enable quantization if not already active

## Command Line Options

```bash
# Lora-related command line options
--lora-dir path                   # Path to Wan t2v loras (default loras/wan)
--lora-dir-wan-5b path            # Path to Wan 5B loras (default loras/wan_5B)
--lora-dir-wan-1-3b path          # Path to Wan 1.3B loras (default loras/wan_1.3B)
--lora-dir-i2v path               # Path to Wan i2v loras (default loras/wan_i2v)
--lora-dir-wan-i2v path           # Alias for Wan i2v loras
--lora-dir-hunyuan path           # Path to Hunyuan t2v loras (default loras/hunyuan)
--lora-dir-hunyuan-i2v path       # Path to Hunyuan i2v loras (default loras/hunyuan_i2v)
--lora-dir-ltxv path              # Path to LTX Video loras (default loras/ltxv)
--lora-dir-flux path              # Path to Flux loras (default loras/flux)
--lora-dir-flux2 path             # Path to Flux2 loras (default loras/flux2)
--lora-dir-qwen path              # Path to Qwen loras (default loras/qwen)
--lora-dir-z-image path           # Path to Z-Image loras (default loras/z_image)
--lora-dir-tts path               # Path to TTS presets (default loras/tts)
--lora-preset preset              # Load preset on startup
--check-loras                     # Filter incompatible loras
``` 
