
<p align="center">
  <H2>Memory Management 4.0.0 for the GPU Poor by DeepBeepMeep</H2>	
</p>


mmgp lets large PyTorch models (video, image and audio generators, LLMs) run fast on consumer GPUs, including models larger than the VRAM. It keeps the models in RAM and moves them to the GPU whole or block by block while the GPU computes, quantizes them on the fly or loads prequantized checkpoints, applies LoRAs without merging them, and can replace PyTorch's VRAM allocator to recycle more efficiently the VRAM no longer used. Most of it is plug and play: one call once the models of a pipeline have been created.

It replaces the offloading of the accelerate library, which does not work well with models that are loaded and unloaded several times in a pipeline (for instance a VAE).

Requirements: Python 3.10+, PyTorch 2.1+ (2.3+ for the VRAM allocator), an NVIDIA GPU. The RAM and VRAM needed depend on the models and on the profile: WanGP runs some video models with 6 GB of VRAM.

## Features
- **5 memory profiles** for common amounts of RAM and VRAM, every setting of which can be overridden.
- **Block swapping**: a model larger than its VRAM budget is processed block by block. Blocks are copied to the GPU ahead of their use while the previous ones are computed, in the order learned during the first step (including from one tower of blocks to the next and from one step to the next), into fixed VRAM slots taken from the budget of the model.
- **VRAM preloading**: the part of a budget not needed by the blocks in transit keeps blocks permanently in VRAM, so fewer blocks are transferred at each step.
- **Reserved RAM**: models kept in reserved (pinned) RAM are transferred up to 2 times faster. The reserved RAM is shared between the models by priority; a model that does not fit gets the blocks used at each step first, spread evenly, and models left out go through a small staging ring of reserved RAM, almost as fast as pinned models when the GPU is busy. Pinning copies the weights in parallel into RAM that is then locked in place.
- **Low RAM loading**: a rewritten safetensors library (memory mapped, with read-only zero-copy loading), fast loading of transformers / diffusers models from single files, on the fly 8 bits quantization or prequantized checkpoints.
- **Pluggable quantization formats**: quantization handlers convert other checkpoint formats to quanto when they are loaded, including files that are not safetensors. Scaled FP8 is built in; applications can add their own formats (WanGP adds GGUF, NV FP4, Nunchaku, INT8 ConvRot...).
- **LoRAs applied on the fly**: LoRA, DoRA, LoKr and diff adapters are applied during the forward pass without merging them, which also works with quantized models, with multipliers that may change at each denoising step.
- **MMGP Optimized VRAM Allocator** (Windows and Linux): several GB less peak VRAM with long videos and large images.
- **Co-tenancy**: models used alternately can stay in VRAM together.
- **Progress reporting and cancellation** of loading, quantization and pinning.
- **PyTorch compilation** of the main model.

### What's new in 4.0
- MMGP Optimized VRAM Allocator (*mmgp.allocator*).
- Faster block swapping with the same VRAM: prefetch window (*prefetchWindow*, set automatically), learned block order across towers and steps, fixed VRAM slots for swapped blocks.
- Staging ring of reserved RAM for the models that are not pinned (*smartPinning*), and a shared reserved RAM plan: blocks used at each step first, spread evenly when they do not all fit, then the blocks preloaded in VRAM, then the rest.
- VRAM preloading and reserved RAM chosen together, and faster pinning (parallel copies into RAM locked in place).
- LoRA and diff adapters whose multiplier is 0 at the current step are skipped and not transferred.
- Fixes: budgets in percent acted as unlimited budgets (80% was read as 80 times the VRAM), and a budget given as a single number instead of a dictionary was ignored.

## Sample applications that use mmgp
It is recommended to have a look at these applications to see how mmgp was implemented in each of them:
- WanGP: https://github.com/deepbeepmeep/Wan2GP (https://wangp.ai):\
A one-stop app for the best open source generative models: video (Wan 2.1/2.2, MiniMax H3, LTX-2, Hunyuan Video 1/1.5, LongCat, Kandinsky...), image (Flux 1/2, Qwen Image, Z-Image, Krea, HiDream...), audio and speech (Ace Step, Qwen3 TTS, Index TTS, Chatterbox, Stable Audio...). It offers a web interface, a generation queue, LoRAs, many quantized formats (int8, fp8, GGUF, NV FP4, Nunchaku) and Deepy, a low VRAM offline agent that orchestrates generation tasks. WanGP uses every feature of mmgp: some of its video models run with 6 GB of VRAM, also on older NVIDIA GPUs.

Earlier standalone applications:
- Hunyuan3D-2GP: https://github.com/deepbeepmeep/Hunyuan3D-2GP :\
A great image to 3D and text to 3D tool by the Tencent team. Thanks to mmgp it can run with less than 6 GB of VRAM

- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP :\
One of the best open source Text to Video generator

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP :\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP :\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).

- OminiControlGP: https://github.com/deepbeepmeep/OminiControlGP :\
A Flux derived application very powerful that can be used to transfer an object of your choice in a prompted scene. With mmgp you can run it with only 6 GB of VRAM.

- YuE GP: https://github.com/deepbeepmeep/YuEGP :\
A great song generator (instruments + singer's voice) based on prompted Lyrics and a genre description. Thanks to mmgp you can run it with less than 10 GB of VRAM without waiting forever.

## Installation
First you need to install the module in your current project with:
```shell
pip install mmgp
```


## Usage 

It is almost plug and play and just needs to be invoked from the main app just after the model pipeline has been created.
1) First make sure that the pipeline explicitly loads the models in the CPU device, for instance:
```
  pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")
```

2) Then add the following lines for a quick setup:
```
  from mmgp import offload, profile_type
  offloadobj = offload.profile(pipe, profile_type.LowRAM_LowVRAM)
```
Instead of a pipeline you may pass a dictionary of models, for instance *{ "transformer": model, "text_encoder": t5, "vae": vae }* (see *Special cases* below). Models that will receive LoRAs must be declared with the *loras* parameter (see *LoRAs* below).

You can choose between 5 profiles depending on your hardware:
- HighRAM_HighVRAM  (1): each model loaded whole in VRAM, all models kept in reserved RAM. The fastest generations and model switches, needs the most RAM and VRAM
- HighRAM_LowVRAM  (2): all models kept in reserved RAM, sent to the GPU part by part. Runs models larger than the VRAM, leaves VRAM for long videos or large images and switches models fast, needs a lot of RAM
- LowRAM_HighVRAM  (3): each model loaded whole in VRAM, only the main model ('transformer') kept in reserved RAM. Fast generations with less RAM, needs enough VRAM for the whole model
- LowRAM_LowVRAM  (4): only the main model ('transformer') kept in reserved RAM, sent to the GPU part by part. The most versatile: runs models larger than the VRAM and leaves VRAM for long videos or large images
- VerylowRAM_LowVRAM  (5): no reserved RAM (only a small staging ring with smartPinning), all models sent to the GPU part by part. For PCs short of RAM and VRAM, slower with short steps such as images

Profile 4 is recommended for most configurations, and profile 2 if you have plenty of RAM. Profiles 1 and 3 are faster when the whole model fits in VRAM, which is often the case with image models. Profile 5, the default of *offload.profile*, is the safest: if your application runs out of memory, try it first and move to profile 4 and then to profile 2 as long as it remains responsive.

By default the model named 'transformer' is quantized to 8 bits with every profile; set *quantizeTransformer = False* if you do not want it. Profiles 3, 4 and 5 also quantize the text encoders named 'text_encoder' or 'text_encoder_2' when they are T5, Llama or other LLM models.

Every parameter set by a profile can be overridden with one or several parameters accepted by *offload.all* (see below):
```
  from mmgp import offload, profile_type
  offload.profile(pipe, profile_type.LowRAM_LowVRAM, budgets = { "transformer": 6000, "*": 3000 })
```
To see the parameters set by a profile, add *verboseLevel = 2*.

**It is highly recommended to put *from mmgp import offload, profile_type* at the top of your main python file (that is as the first import) so that the calls to safetensors (including those of transformers and accelerate) are redirected to mmgp, which needs less RAM.**

*offload.profile* and *offload.all* return an offload object:
- *release()*: unloads the models from the VRAM, releases their reserved RAM and unloads their LoRAs. Call it before setting up other models.
- *unload_all(keep = ())*: unloads from the VRAM every model except those whose ids are in *keep*.


## Alternatively you may want to create your own profile with specific parameters:

For example:
```
  from mmgp import offload
  offload.all(pipe, pinnedMemory = True, extraModelsToQuantize = ["text_encoder_2"] )
```  
- pinnedMemory: True (every model), False, or a model id or list of model ids to keep in reserved RAM. Every model pinned to reserved RAM is transferred up to 2 times faster to the GPU, but this requires more RAM.
- partialPinning: pin only the main blocks of the pinned models (their towers of repeated blocks), not their other parameters.
- pinnedPEFTLora: also pin the LoRA weights stored in PEFT modules of the models (False by default).
- perc_reserved_mem_max: maximum share of the RAM that may be reserved, as a fraction (for instance 0.4). 0 by default: the *perc_reserved_mem_max* environment variable, else 40% on Windows and 80% on Linux.
- smartPinning: None by default. Minimum pinning of the models processed block by block that *pinnedMemory* does not pin. With *asyncTransfers*, 0 copies their blocks ahead, with a background thread, into a small staging ring of reserved RAM (three of the largest such blocks), from which they are transferred to the GPU like pinned blocks while the previous blocks are computed, with the same VRAM; a number n > 0 also pins 1 block in n of these models, spread evenly. When the GPU spends long enough on each block (high resolutions, long videos, large models), blocks copied through the ring are as fast as pinned blocks; when blocks are computed quickly, copying them through RAM becomes the limit.
- quantizeTransformer: True by default. The 'transformer' model of the pipe, usually the video or image generator, is quantized on the fly to 8 bits. If you want to save time on disk and reduce the loading time, you may want to load directly a prequantized model. If you don't want to quantize the generator, set *quantizeTransformer* to *False*.
- extraModelsToQuantize: list of additional model ids of models to quantize on the fly. If the corresponding model is already quantized, this option is ignored.
- quantizationType: quanto type used for on the fly quantization (qint8 by default).
- convertWeightsFloatTo: dtype of the FP32 weights of the models (torch.bfloat16 by default), None keeps them in FP32. A *_convertWeightsFloatTo* attribute on a model overrides it for that model, and the modules or parameters that have a *_lock_dtype* attribute keep their dtype.
- budgets: either a number in mega bytes (for all models, 0 for an unlimited budget), a string that is a percentage of the total VRAM (for instance "80%"), or a dictionary that maps model ids to mega bytes or percentages: the approximate VRAM allocated to a model. To define the default value in the dictionary, add an entry named "*".\
The smaller this number, the more VRAM is left for image data / longer videos, but the slower, since the parts of the model are transferred between the RAM and the VRAM at each step. A model too big for its budget is broken down into blocks loaded one after the other; the part of the budget that these transfers do not need keeps blocks permanently in VRAM. The speed of a low budget increases (up to 2 times) with reserved RAM and asyncTransfers.
- workingVRAM: either a number in mega bytes, a string that is a percentage of the total VRAM, or a dictionary that maps model ids to mega bytes: a minimum amount of VRAM that should be left for the data processed by the model. This number prevails if it is in conflict with a too high budget defined for the same model.
- vram_safety_coefficient: float between 0 and 1, 0.8 by default: the maximum share of the VRAM that the models may occupy.
- asyncTransfers: True by default. The next blocks of a model are transferred to the GPU while the current block is computed. This may increase speed by 20% (mostly visible on fast modern GPUs).
- prefetchWindow: None by default: 2 for the models whose budget also preloads blocks in VRAM, 1 for the others. With asyncTransfers, number of blocks transferred ahead of the block being computed, for the models that are pinned or copied through the staging ring. The GPU itself orders these transfers after the computation of the blocks whose memory they reuse, so the next transfers are prepared during the computation; the blocks in transit (window + 1) are taken from the budget of the model, so a window of 2 costs no VRAM beyond it. Windows beyond 1 are only used if their extra blocks fit in 5% of the VRAM. 0 waits for the GPU after every block.
- autoPreload: False by default. With the MMGP Optimized VRAM Allocator (`mmgp.allocator.install()`), the models processed block by block keep in VRAM as much of their blocks as each workload (a denoising stage, a resolution, a model of the pipe) leaves free, instead of the preload of their budget: the first step of a new workload measures what it needs, the next ones fill the VRAM left, and part of the model leaves the VRAM if it runs short, instead of an out of memory error. *True* keeps what was measured for the process; a dictionary keeps it in it, so that an application can tie it to a model definition. Not applied to compiled models or without asynchronous transfers.
- loras: a model id or list of model ids that will receive LoRAs (see *LoRAs*).
- coTenantsMap: a dictionary that maps a model id to a list of other models with which it accepts to share the VRAM at the same time. This is useful to avoid inefficient loading / unloading when two models are used alternately. For instance with *coTenantsMap = { "text_encoder_2": ["text_encoder"] }*, when *text_encoder_2* is loaded it won't unload *text_encoder*. Please note that the reverse is not true as these maps are by design not symmetrical, to allow tailored workflows. If you also need *text_encoder* not to unload *text_encoder_2*: *coTenantsMap = { "text_encoder_2": ["text_encoder"], "text_encoder": ["text_encoder_2"] }*. A small auxiliary model mapped to "*" stays in VRAM with any other model and is never unloaded for one.
- compile: list of model ids to compile, which may accelerate up to x2 depending on the GPU. It makes sense to compile only the model that is used the most, such as the "transformer" model of a video or image generator. Compilation requires Triton, available out of the box on Linux and WSL; on Windows install https://github.com/woct0rdho/triton-windows. *compile_mode* sets the mode of torch.compile ("default"). A submodule with a *_compile_me* attribute set to True or False is compiled or left out whatever its position.
- loading_callback: an *offload.LoadingCallback* to report progress and accept cancellation (see *Loading progress and cancellation*).
- verboseLevel: number between 0 and 2 (1 by default), provides various levels of feedback on the different processes.

If you are short on RAM and plan to work with quantized models, it is recommended to load pre-quantized models directly rather than using on the fly quantization: it is faster and consumes slightly less RAM.

### How the reserved RAM is shared
Within the limit of *perc_reserved_mem_max*, the reserved RAM goes in this order to: the staging ring when blocks of pinned models are left unpinned; the models of *pinnedMemory*, in the order of the pipe: first their blocks computed at each step (spread evenly if they do not all fit, the ring taking the others), then their blocks preloaded in VRAM by their budget, then the rest of these models; then the minimum of *smartPinning*.

### Hints on the models
Attributes that an application may set on its models to tune mmgp:
- *_mmgp_ignore_models* on the pipe: ids of models that mmgp must leave untouched, for instance a component managed by another pipeline.
- *_offload_hooks* on a model or one of its direct submodules: list of methods besides *forward* that require the model in VRAM, for instance *vae._offload_hooks = ["encode", "decode"]*.
- *_convertWeightsFloatTo* on a model and *_lock_dtype* on a module or parameter: see *convertWeightsFloatTo*.
- *_compile_me* on a submodule: see *compile*.
- *_offload_separate_blocks* on a model: paths of submodules that become blocks of their own when the model has a VRAM budget, loaded only while they run, for instance the token embedding and output head of an LLM that a generator uses for its prompt but not while it denoises: *("language_model.model.embed_tokens", "language_model.lm_head")*.


## LoRAs
LoRAs are applied during the forward pass, without merging them in the weights: they work with quantized models, can be changed between generations without reloading the model, and their multipliers can change at each denoising step. Declare the models that will receive them when you set up mmgp, then load the LoRAs at any time:
```
  offloadobj = offload.profile(pipe, profile_type.LowRAM_LowVRAM, loras = "transformer")
  offload.load_loras_into_model(pipe.transformer, ["style.safetensors", "detail.safetensors"], [1.0, [0.8] * 10 + [0.4] * 20])
  offload.set_step_no_for_lora(pipe.transformer, step_no) # at each denoising step, with multipliers that change at each step
```
Supported formats: LoRA (lora_A / lora_B or lora_down / lora_up, with alpha), DoRA, LoKr and diff adapters (full weight or bias differences). The key names must match the module names of the model, if needed after conversion by *preprocess_sd*.
- *load_loras_into_model(model, lora_path, lora_multi = None, activate_all_loras = True, check_only = False, ignore_model_variations = False, pinnedLora = False, maxReservedLoras = -1, split_linear_modules_map = None, preprocess_sd = None)*\
Loads a list of LoRA files with a list of multipliers: a number, or a list of numbers with one value per denoising step. *pinnedLora* keeps the LoRAs in reserved RAM, up to *maxReservedLoras* mega bytes (-1 for no limit). *ignore_model_variations* skips the modules of a LoRA whose shapes do not match the model (a LoRA made for a variant of the model) instead of rejecting it. *split_linear_modules_map* applies LoRAs made for separate linear layers to fused layers and conversely. *check_only* only checks which LoRAs are compatible. The loaded LoRAs are named "0", "1"... in their order in *lora_path*; by default all of them are activated.
- *activate_loras(model, lora_nos, lora_multi = None)*\
Activates the LoRAs whose numbers are in *lora_nos*, with their multipliers. Every LoRA that is not in this list and that was activated previously is deactivated.
- *set_step_no_for_lora(model, step_no)*: selects the multipliers of a denoising step. LoRA and diff adapters whose multiplier is 0 at this step are skipped and, with block swapping, not transferred to the GPU (DoRA and LoKr adapters still apply at 0).
- *unload_loras_from_model(model)*: unloads all the LoRAs of a model.
- *sync_models_loras(model, model2)*: shares the LoRAs of a model with another model of the same architecture (for instance two transformers used one after the other).


## Loading and saving models

The module includes several tools to package a light version of your favorite video / image generator:
- *extract_models(obj, prefix)*\
This tool will try to detect for you models that are embedded in a pipeline or in some custom class. It will save you time by building the dictionary of models required by *offload.all* or *offload.profile*. The prefix corresponds to the text that will appear before the name of each model in the dictionary.

- *save_model(model, file_path, do_quantize = False, quantizationType = qint8, config_file_path = None, filter_sd = None, quantize_exclude = None)*\
Saves the tensors of a model already loaded in memory in the safetensors format (much faster to reload). You can save it in a quantized format (default qint8 quantization recommended), except the modules listed in *quantize_exclude*.
The resulting safetensors file contains extra fields in its metadata such as the quantization map and the configuration of the model, so you can move the file around without files such as *config.json* or *file_map.json*.
You will need *load_model_data* or *fast_load_transformers_model* to read the file again. You may also load it using the default *safetensors* library, however you will need to provide in the same directory any complementary file that is usually requested (for instance *config.json*).

- *fast_load_transformers_model(model_path, do_quantize = False, quantizationType = qint8, pinToMemory = False, partialPinning = False, forcedConfigPath = None, defaultConfigPath = None, modelClass = None, writable_tensors = True, default_dtype = torch.bfloat16, ...)*\
Initializes (builds the model hierarchy in memory) and fast loads the corresponding tensors of a 'transformers' or 'diffusers' library model.
The advantages over the original *from_pretrained* method are that a full model can fit into a single file with a filename of your choosing (therefore you can have multiple 'transformers' versions of the same model in the same directory) and that prequantized models are processed in a transparent way. *model_path* may also be a list of files that together form the model. The configuration is read from the metadata of the file, else from *forcedConfigPath* or *defaultConfigPath*; *modelClass* forces the class of the model.
You can also pin to reserved RAM on the fly the whole model or its main blocks (*partialPinning = True*) in a more efficient way (faster and requires less RAM) than if you did through *offload.all* or *offload.profile*.

- *load_model_data(model, file_path, do_quantize = False, quantizationType = qint8, pinToMemory = False, partialPinning = False, writable_tensors = True, default_dtype = torch.bfloat16, ...)*\
Loads the tensors of a model already initialized with no data (for instance with *init_empty_weights*). It detects and handles quantized models, saved previously with *save_model* or in a format of a quantization handler. A model can also be quantized on the fly while being loaded.

Options common to both loading functions:
- *writable_tensors = False*: memory maps the file read-only, without copying its tensors, which needs much less RAM. Use it unless the weights are modified in place.
- *default_dtype*: dtype of the weights that are not quantized; None keeps the dtypes of the checkpoint, and *fp32_dtype* then converts only its FP32 weights.
- *do_quantize* with *quantize_exclude*: list of modules that on the fly quantization leaves out.
- *preprocess_sd*: function or dictionary of renaming rules applied to the state dictionary before loading (for instance to convert the key names of another implementation).
- *fused_split_map*: splits fused weights of the checkpoint (for instance qkv) between the separate modules of the model.
- *modules* and *return_shared_modules*: loads in a model extra module files and returns their tensors, so that another model can be loaded with the same tensors instead of a second copy in RAM.
- *ignore_unused_weights*, *ignore_missing_keys* (*load_model_data*): tolerate weights of the file that the model does not use, or parameters of the model that the file does not provide.

- *load_sd(file_path, filters = None, keep_prefixes = False, writable_tensors = True)*: loads a state dictionary, its quantization map and its tied weights, optionally keeping only the keys with the prefixes of *filters*.


The typical workflow will be:
1) temporarily insert the *save_model* function just after a model has been fully loaded to save a copy of the model / quantized model.
2) replace the full initializing / loading logic with *fast_load_transformers_model* (if there is a *from_pretrained* call to a transformers object) or only the tensor loading functions (*torch.load_model_file* and *torch.load_state_dict*) with *load_model_data* after the initializing logic.

### Quantization formats
*mmgp.quant_router* converts checkpoints in other quantized formats to quanto when they are loaded. Scaled FP8 checkpoints (e4m3fn / e5m2 weights with scales) are supported out of the box. An application can add formats:
- *quant_router.register_handler(handler)*: a module (or its import path) that defines *detect()* and *convert_to_quanto()*, and optionally the quanto qtypes and the Linear module that computes with them.
- *quant_router.register_file_extension(extension, handler)*: loads files of another type through a handler (WanGP loads GGUF files this way).

### Loading progress and cancellation
```
  callback = offload.LoadingCallback(abort_requested = lambda: stop_requested, progress = lambda phase, completed, total, model_id: print(phase, completed, total, model_id))
  with offload.loading_context(callback, model_ids = { "ckpts/transformer.safetensors": "transformer" }):
      transformer = offload.fast_load_transformers_model("ckpts/transformer.safetensors")
  offloadobj = offload.profile(pipe, profile_type.LowRAM_LowVRAM, loading_callback = callback)
```
The progress callback receives the current phase (reading, quantizing, pinning...), the progress within it and the model id. When *abort_requested* returns True, loading stops with an *offload.LoadingCancelled* exception, and *offload.profile* / *offload.all* first release what they had set up.


## MMGP Optimized VRAM Allocator
Recycles more efficiently the VRAM no longer used: several GB less peak VRAM with long videos and large images, the same outputs and speed.

```
from mmgp import allocator
allocator.install()   # before anything initializes CUDA, e.g. at the top of the application
```
- *install(mode = "vmm", spill = False)*: replaces PyTorch's CUDA allocator. It must be called before CUDA is initialized, as PyTorch cannot replace an allocator it has started. With *spill = True*, what VRAM has no room for goes to system RAM instead of raising an out of memory error (slower).
- *stats(device = None)*: allocated, reserved and peak bytes of the allocator. *torch.cuda.memory_allocated / memory_reserved / max_memory_allocated / max_memory_reserved / reset_peak_memory_stats / memory_stats / empty_cache* are redirected to it.

Requirements: Windows (x64) or Linux (x86_64, glibc 2.17+), PyTorch 2.3 or newer, an NVIDIA driver supporting CUDA 11.2+ (no CUDA toolkit needed). The prebuilt libraries are in *mmgp/allocator*; after changing *vmm_alloc.cpp*, rebuild them with `python -m mmgp.allocator.build` (MSVC on Windows, which also rebuilds the Linux library through WSL when available; g++ on Linux).

Differences with PyTorch's allocator: running out of VRAM raises an out of memory error (PyTorch's own *torch.OutOfMemoryError* with PyTorch 2.6 to 2.15) instead of slowly spilling into shared GPU memory, and *torch.cuda.memory_snapshot* is not available.

### VRAM debug mode
Which tensors fill the VRAM at its peak, and where in the code they come from: with the allocator installed, *mmgp.allocator.debug* records every allocation above a size threshold with the innermost module running (its path in the models that mmgp loads), a tag and the Python stack, and copies these records at the peak of each phase (one phase per model that mmgp loads to the GPU) and when an allocation fails.
```
from mmgp.allocator import debug
debug.start(min_mb = 16)          # after allocator.install()
...                               # the workload
debug.report("vram.json")         # JSON for an agent, and vram.md, a summary
```
- *x._allocator_tag = "kv cache"* names the allocation of a tensor (views share it), *with debug.tag("decode tile"):* the allocations of a block of code. Without debug mode, both cost nothing and may stay in the code.
- *mark(name)* starts a phase of your own, *name_models({"name": model})* names the modules of models that mmgp does not load, *reset()* starts new peaks (between two generations), *stop()* ends the recording.
- The report lists, for each phase, the tensors alive at its peak grouped by origin (module path, allocation site, tags), with their sizes and ages at the peak and the full stacks; the allocations still alive when it is written (leaks); the totals of each origin (count, bytes, mean lifetime: short-lived large allocations are scratch buffers).
- *python -m mmgp.allocator.debug summary report.json* prints the summary, *diff before.json after.json* compares the peaks of two reports.

A recorded allocation costs about 8 us (its stack is formatted only in the report); generation speed is otherwise unchanged. The origin is asked from inside the allocation, which takes the GIL: a thread holding the GIL while waiting for a lock that the allocating thread holds would deadlock (PyTorch's own memory history has the same constraint).

## Special cases
Sometimes there isn't an explicit pipe object as each submodel is loaded separately in the main app. If this is the case, you may try to use *extract_models* or create a dictionary that manually maps all the models.\
For instance :


- for flux derived models: 
```
pipe = { "text_encoder": clip, "text_encoder_2": t5, "transformer": model, "vae":ae }
```
- for mochi: 
```
pipe = { "text_encoder": self.text_encoder, "transformer": self.dit, "vae":self.decoder }
```


Please note it is recommended to have always one model whose Id is 'transformer' so that you can leverage predefined profiles. The 'transformer' corresponds to the main image / video model which usually needs to be quantized (this is done on the fly by default when loading the model).

Be careful, lots of models use the T5 XXL as a text encoder. However, quite often their corresponding pipeline configurations point at the official Google T5 XXL repository 
where there is a huge 40GB model to download and load. It is cumbersome as it is a 32 bits model and contains the decoder part of T5 that is not used. 
I suggest you use instead one of the 16 bits encoder only versions available around, for instance:
```
text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=torch.float16)
```

Sometimes just providing the pipe won't be sufficient as you will need to change the content of the core model: 
- For instance you may need to disable an existing CPU offload logic that already exists (such as manual calls to move tensors between cuda and the cpu)
- mmgp tries to fake the device as being "cuda" but sometimes some code won't be fooled and it will create tensors in the cpu device and this may cause some issues.

mmgp is part of WanGP and is licensed under the WanGP Community License 2.0 (see LICENSE.txt at the root of the WanGP repository). You may contact me on twitter @deepbeepmeep

Thanks to
---------
- Huggingface / accelerate for the hooking examples
- Huggingface / quanto for their very useful quantizer
- gau-nernst for his Pinnig RAM samples
