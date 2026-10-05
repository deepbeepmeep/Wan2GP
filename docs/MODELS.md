# Models

WanGP brings video, image, speech, music and sound generation into one workspace. This guide covers the main families and how to choose between their workflows. The model selector and each model's **Help** describe the variants, controls and settings available in your installation.

Model links below select a model when this guide is opened inside WanGP. When reading on GitHub, select the same name in WanGP's model selector or toolbar search.

- [Choose a starting point](#recommended-starting-points)
- [Video models](#video-models)
- [Image models](#image-models)
- [Speech, music and sound](#audio-models)
- [Upscaling and finishing](#upscaling-and-finishing)
- [Choose a variant for your hardware](#choosing-variants)

## Recommended starting points

| Goal | Models to try | How to choose |
| --- | --- | --- |
| Video with a synchronized soundtrack | [MiniMax H3 FL2VA Pruned](modeltype:minimax_h3_fl2va_pruned), [LTX 2.5 Distilled](modeltype:ltx2_25_22B_distilled) | H3 offers text and first/last-frame direction with stereo audio. LTX 2.5 Distilled is an eight-step starting point with keyframes, controls and continuation. |
| Video following reference subjects, scenes or voices | [MiniMax H3 Ref2VA Pruned](modeltype:minimax_h3_ref2va_pruned), [LTX 2.5 MSR](modeltype:ltx2_25_22B_msr) | Ref2VA accepts image, video and audio references. MSR combines subject/object references, with an optional background reference. |
| Character animation and replacement | [SCAIL-2](modeltype:scail2_14B), [Wan 2.2 Animate 2](modeltype:animate2), [Viggle Animate](modeltype:viggle_animate) | SCAIL-2 and Animate 2 are Wan-derived workflows. Viggle is H3-derived and needs an edited frame from the original driving video. |
| General text-to-video or image-to-video | [Wan 2.2 T2V](modeltype:t2v_2_2), [Wan 2.2 I2V](modeltype:i2v_2_2) | General-purpose generation with a broad choice of LoRAs and specialist variants. |
| Connected stories with recurring characters | [JoyAI-Echo Surgical](modeltype:joyai_echo_surgical) | This LTX-2 variant reuses character, voice, object and location memories between shots. |
| Video editing and outpainting | [H3 ControlNet-Union Pruned](modeltype:minimax_h3_control_pruned), [Wan VACE](modeltype:vace_14B), [Bernini-R](modeltype:bernini) | Pick H3 for controlled audio-video generation, VACE for mask/control editing, or Bernini for video edits and multiple references. |
| Talking characters and longer dialogue | [LongCat Avatar 1.5](modeltype:longcat_avatar_v1_5), [InfiniteTalk](modeltype:infinitetalk), [Magi Human Distill](modeltype:magi_human_distill) | LongCat Avatar and InfiniteTalk follow speech audio. Magi Human can generate speech with motion or follow supplied audio. |
| General image generation | [Krea 2 Turbo](modeltype:krea2_turbo), [Z-Image Turbo](modeltype:z_image), [Flux 2 Klein 4B](modeltype:flux2_klein_4b) | Distilled choices for iteration. Compare their appearance, reference support and memory use on your own prompts. |
| Image editing and references | [Qwen Image 2.1](modeltype:qwen_image_21_7B), [Qwen Image Edit Plus 2511](modeltype:qwen_image_edit_plus2_20B), [Krea 2 Identity Edit](modeltype:krea2_raw_edit) | Qwen 2.1 combines generation and editing with up to ten references. Edit Plus emphasizes image edits and identity. Krea Identity Edit accepts up to two references. |
| Posters, typography and infographics | [Ming Image Design](modeltype:ming_image_0_1_design), [Ideogram 4](modeltype:ideogram4), [SenseNova U1.5](modeltype:sensenova_u1_5_8b_mot) | Ming and Ideogram offer structured layout helpers; SenseNova supports detailed native-4K layouts. Proofread generated text. |
| Editable transparent layers | [Ming Image Design-Layer](modeltype:ming_image_0_1_design_layer), [Qwen Image Layered](modeltype:qwen_image_layered_20B) | Decompose a flattened image into separate RGBA layers for further editing. |
| Speech generation, editing and cleanup | [AuK Flash](modeltype:auk_flash), [AuK Base](modeltype:auk) | Describe the speech or the change to a source recording. Flash uses four steps; Base offers more sampling control. |
| Cloned voices and dialogue | [Qwen3 TTS Base](modeltype:qwen3_tts_base), [IndexTTS 2.5](modeltype:index_tts25), [OmniVoice](modeltype:omnivoice) | Use short, clean voice references. Compatible dialogue modes accept up to three speakers; IndexTTS adds emotion controls. |
| Songs with lyrics | [YuE2](modeltype:yue2), [MiniMax Music 3](modeltype:minimax_music3), [ACE-Step 1.5 XL](modeltype:ace_step_v1_5_xl) | YuE2 adds composition, cover and score workflows. MiniMax Music 3 generates songs from lyrics and a description. ACE-Step offers fast music generation with optional planning models. |
| A song from a hummed melody | [YuE2 Hum-to-Song](modeltype:yue2_hum) | Supply a clear hum, your lyrics and a musical style. It follows the melody, not the humming voice's identity. |
| Instrumentals, ambience and sound effects | [Stable Audio 3](modeltype:stable_audio3_small), [YuE2](modeltype:yue2) in instrumental mode | Stable Audio covers descriptive sound and music prompts; YuE2 supports structured instrumental composition. |

<a id="video-models"></a>

## Video models

### MiniMax H3

H3 creates video with native stereo audio. The main variants are available as full **33B** and smaller **Pruned 20B** models:

- **[FL2VA](modeltype:minimax_h3_fl2va_pruned)**: generate from text, a start image, an end image or both. Use injected frames for visual milestones, or sliding windows to continue a longer shot. It can follow an uploaded soundtrack or a control video's audio.
- **[Ref2VA](modeltype:minimax_h3_ref2va_pruned)**: guide appearance, subjects, motion or voices with reference images, videos and audio. It supports up to three reference videos and three audio references. Video references share a 15-second budget; audio references have a separate 15-second budget. Reference excerpts let you select useful moments from a longer clip. Audio choices marked **Soundtrack Kept** preserve the supplied soundtrack.
- **[FL2VA ControlNet-Union](modeltype:minimax_h3_control_pruned)**: follow pose, depth, edges, shapes or recoloring controls, or work with inpainting and spatial outpainting. Select this variant when those explicit controls matter to the shot.

Both main workflows support continuation, reference or frame guidance as appropriate, and still-image generation. For straightforward text, start/end or injected-frame direction, start with FL2VA; choose Ref2VA when reusing reference content is central.

H3's **Two Phases** option generates at a lower resolution before refining at the target size. This can accelerate high-resolution work, but does not by itself lower peak VRAM. **Two Phases with Tiling** lowers the high-resolution phase's memory requirement, with a possible trade-off in tile seams.

Acceleration choices have different trade-offs. **PDD** variants use a fixed eight-step Euler workflow. **VDN** variants accelerate longer video and work best with the required Triton support. Spectrum, First Block Cache and supported sparse attention settings offer other speed/quality choices; start with the selected variant's defaults before combining them.

**[Viggle Animate](modeltype:viggle_animate) is an H3-derived character-replacement model.** Supply the original control video and one frame from it in which an image editor has replaced the character. Preserve that frame's pose, background, framing and dimensions. Viggle follows the original motion in a fixed three-step workflow, using sliding windows for longer footage. Its text generation prompt is ignored: express the change in the edited frame. Large pose changes, interacting characters and cuts are more challenging.

The family also provides an audio-only **[H3 Voice Clone](modeltype:minimax_h3_tts_ref2va_pruned)** preset and **H3 Face Refiner** for finishing video; these are related H3 workflows, not separate model families.

<a id="ltx-2-and-ltx-23"></a>

### LTX-2, LTX 2.3 and LTX 2.5

LTX-2 generates video and sound together. **[LTX 2.5 Distilled](modeltype:ltx2_25_22B_distilled)** is the current fast eight-step starting point; **[LTX 2.5 Dev](modeltype:ltx2_25_22B)** gives more control over generation. LTX 2.0 and 2.3 variants remain available for existing workflows and compatible LoRAs.

Depending on the selected variant and process, the family supports:

- start/end images, injected frames, control video and soundtrack guidance;
- continuation and sliding windows for longer sequences;
- pose, depth, edge, inpainting, outpainting and HDR workflows;
- reference-voice conditioning and talking-character workflows;
- multi-phase generation and video upscaling.

LTX 2.5 uses **Ingredients** when reference images are supplied in its compatible workflow. Choose **[LTX 2.5 MSR](modeltype:ltx2_25_22B_msr)** for one to four subject/object references, or two to five references with the background first. **[LTX 2.3 MSR V2](modeltype:ltx2_22B_msr_v2)** remains another reference-based option. Describe what each reference contributes.

**[JoyAI-Echo](modeltype:joyai_echo)** and **[JoyAI-Echo Surgical](modeltype:joyai_echo_surgical)** are LTX-2 variants for connected shots with reusable memories. They are useful when the same people, voices or locations must recur.

Decoder choices affect both memory and appearance. The optional **NAD** decoder for LTX 2.3/2.5 trades additional time and VRAM for a different quality profile. Use the original decoder as the initial baseline, and choose only decoder/LoRA combinations offered by the selected model. LTX-based upscaling and detail refinement are also available in postprocessing and Media Flow.

### Wan 2.1 and Wan 2.2

The Wan family covers general generation and many specialist animation, editing and control workflows:

- **[Wan 2.2 T2V](modeltype:t2v_2_2)** and **[Wan 2.2 I2V](modeltype:i2v_2_2)**: general text-to-video and image-to-video.
- **[Wan 2.2 TI2V 5B](modeltype:ti2v_2_2)**: a smaller unified text/image-to-video option.
- **[Wan 2.2 Animate 2](modeltype:animate2)** and its **[Distilled variant](modeltype:animate2_distilled)**: transfer body motion, expressions and camera movement from a driving video to a reference character. The original Animate remains available.
- **[SCAIL-2](modeltype:scail2_14B)**: a Wan 2.1-derived character animator with reference, pose and mask guidance. Animate mode can direct multiple characters; replacement mode changes a person in an existing video. Use aligned references, and the model's mask guidance for multi-character scenes. Sliding windows support longer performances.
- **[VACE](modeltype:vace_14B)**: masked edits, outpainting, object replacement and pose/depth/motion control.
- **[Bernini-R](modeltype:bernini)**: video-to-video generation and multiple reference images; a smaller **[1.3B variant](modeltype:bernini_1.3B)** is also available.
- **[Lynx](modeltype:lynx)** and **[VACE Lynx](modeltype:vace_lynx_14B)**: identity-guided face replacement.
- **[MultiTalk](modeltype:multitalk)** and **[InfiniteTalk](modeltype:infinitetalk)**: audio-driven talking characters and dialogue.
- **[Vista4D](modeltype:vista4d)**: reshoot a dynamic scene along a new camera trajectory.

Names such as FusioniX, FastWan, Lightning and Self-Forcing identify particular variants or acceleration workflows. NVFP4 names identify a weight format. Check the underlying model before choosing compatible LoRAs or assuming that two variants expose the same controls.

### Other video families

These are independent families, rather than variants of each other:

| Family | Main uses and choices |
| --- | --- |
| **HunyuanVideo** | [HunyuanVideo 1.5](modeltype:hunyuan_1_5_t2v) offers text-to-video and image-to-video, with distilled and upsampler variants. Original HunyuanVideo, Custom and Avatar workflows remain available. |
| **LongCat** | [LongCat Video](modeltype:longcat_video) handles general generation; [LongCat Avatar 1.5](modeltype:longcat_avatar_v1_5) is an audio-driven avatar variant. |
| **Kandinsky 5** | [Pro](modeltype:k5_pro_t2v) offers text/image-to-video with camera direction; [Lite](modeltype:k5_lite_t2v) and distilled variants provide smaller or faster choices. |
| **Magi Human** | [Base](modeltype:magi_human) and [Distill](modeltype:magi_human_distill) animate a start image with generated speech or supplied audio. Use the offered staged upsampler variants for higher-resolution results. |
| **Ovi** | [Ovi 1.1](modeltype:ovi_1_1) generates video with synchronized audio, including speaking characters. FastWan and longer-duration variants are available. |
| **LTX-Video** | [LTX-Video 0.9.8](modeltype:ltxv_13B) and its distilled variant retain the earlier LTX video workflow; they are distinct from the LTX-2 audio-video generation described above. |

Other specialist entries remain in the selector. Use their own Help for required inputs and limitations rather than transferring settings between unrelated models.

<a id="image-models"></a>

## Image models

<a id="qwen-image"></a>

### Qwen Image and Qwen Image 2.1

**[Qwen Image 2.1 7B](modeltype:qwen_image_21_7B)** combines image generation and editing. It accepts up to ten references and supports masked editing, LanPaint, outpainting, pose/depth/edge guidance and resolutions up to 4K. Enable **RGBA** and request a transparent background to save cutouts as PNG with an alpha channel.

Its optional **KV Cache** trades extra GPU memory for faster denoising. **Viggle Turbo** and **Pruna** acceleration profiles apply the matching adapter, scheduler and step settings; these image accelerators are unrelated to the H3-based Viggle Animate video workflow. Prefer short text in Qwen 2.1 images and proofread it; use a design-oriented model for dense infographics.

The older 20B family remains useful:

- **[Qwen Image 2512](modeltype:qwen_image_2512_20B)** for generation and text-heavy images;
- **[Edit Plus 2511](modeltype:qwen_image_edit_plus2_20B)** for reference-based editing and identity preservation, alongside original Edit and Edit Plus 2509;
- **[Qwen Image Layered](modeltype:qwen_image_layered_20B)** for decomposing an image into editable RGBA layers.

Selected 20B variants have Nunchaku checkpoints for supported hardware. Qwen 2.1 uses its own model and adapter formats: do not assume older Qwen Image LoRAs or VAEs are interchangeable.

### Ming Image

**[Ming Image 0.1 Design](modeltype:ming_image_0_1_design)** generates posters, infographics and arranged layouts, or edits one reference image. Use a normal prompt or the structured JSON format for more precise layout direction. The visual **Prompt Helper** lets you position layers and edit their descriptions, colors and text. Optional RGBA output supports transparent designs.

**[Ming Image Design-Layer](modeltype:ming_image_0_1_design_layer)** separates one flattened design into transparent layers. Supply the image and describe the layers from front to back. WanGP adds the RGBA layers to the gallery and saves the full set as a ZIP. Use this when you need individual elements for later compositing.

### SenseNova U1.5

**[SenseNova U1.5 8B MoT](modeltype:sensenova_u1_5_8b_mot)** generates and edits images, with native-4K output, detailed composition and text-heavy layouts. Its **Infographic Prompt** enhancer helps turn a short brief into a structured design. An eight-step acceleration profile and optional KV Cache offer speed/memory trade-offs.

### Krea 2

**[Krea 2 RAW](modeltype:krea2_raw)** is the undistilled option with guidance control; **[Krea 2 Turbo](modeltype:krea2_turbo)** is distilled for faster generation. **Identity Edit** variants add instruction-based edits and up to two reference images. WanGP also supports LanPaint inpainting and negative prompting through NAG on supported distilled workflows.

Krea 2 and compatible Qwen Image 20B/Edit models offer a **VAE** dropdown: the default keeps the established appearance, **Krea 2 Real** offers an alternative for photographic texture, and **Krea 2 HD** offers another for detail and contrast. Compare color and detail on your own images; alternatives can use more memory. Selecting the Spacepxl VAE upsampler overrides that choice. These alternatives do not apply to Qwen 2.1, Qwen Layered or Ming Image.

<a id="flux-z-image-and-hidream"></a>

### Flux 1 and Flux 2

Flux is a separate family from Krea 2, Z-Image and HiDream:

- **[Flux 1 Schnell](modeltype:flux_schnell)** and **[Dev](modeltype:flux)** cover general generation; Kontext, DreamOmni2, Krea, Chroma, SRPO and other variants specialize the workflow or appearance. **Flux 1 Krea is a Flux variant, not Krea 2.**
- **[Flux 2 Dev](modeltype:flux2_dev)** and **[Klein 4B](modeltype:flux2_klein_4b) / [9B](modeltype:flux2_klein_9b)** provide generation and reference-based workflows. Base and distilled Klein variants expose different speed/control choices; supported NVFP4 variants target compatible hardware.

Choose LoRAs and reference workflows for the exact Flux architecture and variant.

### Z-Image

**[Z-Image Turbo](modeltype:z_image)** is a distilled 6B generator. **Base**, **Fun ControlNet**, **TwinFlow** and supported **Nunchaku** variants offer different control, sampling and memory choices. Use a ControlNet variant when you need its particular pose/depth/edge or editing workflow, rather than expecting those controls on every Z-Image entry.

### Ideogram 4

**[Ideogram 4](modeltype:ideogram4)** focuses on typography, layout and structured graphic design. Plain prompts work; its Magic Prompt and visual helper assist with the model's JSON layout format. **TurboTime** and **NF4** variants offer different speed or memory trade-offs.

### HiDream

**[HiDream O1 Dev 2604](modeltype:hidream_o1_dev_2604)** supports text-to-image and reference-guided creation, with control-image and preview support. Full and other Dev checkpoints are also available. HiDream is independent of Ideogram and Flux; use its own model help and compatible settings.

<a id="audio-models"></a>

## Speech, music, and sound

### Speech and dialogue

| Family or derived workflow | When to use it |
| --- | --- |
| **[AuK](modeltype:auk) / [AuK Flash](modeltype:auk_flash)** | Generate speech from written instructions, clone a voice, edit spoken content, clean up a recording or separate speakers. Supply source audio for tasks that transform an existing recording, and describe what must stay unchanged. Flash is the fixed four-step option; Base allows more sampling control. |
| **[Qwen3 TTS](modeltype:qwen3_tts_base)** | Base clones reference voices; Custom Voice uses its offered voices; Voice Design creates a voice from a description. Base supports one to three voice references for dialogue. |
| **[IndexTTS 2](modeltype:index_tts2) / [2.5](modeltype:index_tts25)** | Voice cloning and expressive dialogue with text- or audio-guided emotion. Version 2.5 adds speech-speed control and pronunciation annotations, with support for Chinese, English, Japanese, Spanish and Arabic. |
| **[OmniVoice](modeltype:omnivoice)** | Multilingual speech, voice design, cloning and dialogue, with control over speaking speed. |
| **[KugelAudio](modeltype:kugelaudio_0_open)** | An alternative for generated speech and reference-voice dialogue. |
| **[Chatterbox](modeltype:chatterbox)** | Another reference-based speech synthesis workflow. |
| **[MiniMax H3 Voice Clone](modeltype:minimax_h3_tts_ref2va_pruned)** | H3's audio-only preset for cloned speech and general audio; it reuses H3 model files and saves stereo audio rather than video. |
| **[DramaBox](modeltype:dramabox_audio) / [Scenema Audio](modeltype:scenema_audio)** | LTX-2-derived expressive speech and dialogue workflows. Scenema applies voice references through SeedVC. |

For supported three-voice modes in **H3 Audio, Qwen3 TTS Base, OmniVoice, KugelAudio, IndexTTS 2/2.5, DramaBox and Scenema**, upload the references in speaker order and write `Speaker 1:`, `Speaker 2:` and `Speaker 3:` lines. Select the corresponding reference mode first. Do not assume that every speech model or every voice mode supports the same number of speakers.

### Music and sound

**[YuE2](modeltype:yue2)** creates songs with vocals and accompaniment from lyrics and a musical description. Its workflows include:

- composing a new song or choosing instrumental mode;
- transcribing melody and chords from source audio to guide a new cover, with lyrics supplied separately;
- importing or extending a compatible ABC score, including a score transcribed from audio;
- exporting the composition as ABC and MIDI when enabled;
- using compatible composition-stage and audio-generation LoRAs.

**[YuE2 Hum-to-Song](modeltype:yue2_hum)** turns a clear hummed melody, lyrics and a style into a new song. Choose whether to use the transcribed melody alone or continue it. Neither covers nor hum-to-song preserve the source recording or clone its singer. The duration setting is an upper limit; a result can end earlier or be cut off if the limit is too short.

**[MiniMax Music 3](modeltype:minimax_music3)** generates complete songs from lyrics and a music description, with stereo output and tracks up to five minutes. Its prompt enhancers can help write lyrics or the musical brief. Actual duration and quality depend on the prompt and settings.

**[ACE-Step 1.5 Turbo](modeltype:ace_step_v1_5)** and **[ACE-Step 1.5 XL Turbo](modeltype:ace_step_v1_5_xl)** provide fast music workflows; ACE-Step 1.0 remains available. The optional 0.6B, 1.7B or 4B planning-model variants add a stage that can improve structure and lyric matching at an extra time and memory cost.

**[HeartMuLa](modeltype:heartmula_oss_3b)** is another lyrics-driven song family, with base and updated checkpoints.

**[Stable Audio 3](modeltype:stable_audio3_small)** generates instrumentals, loops, ambience and sound effects. Small, Medium and a dedicated Small SFX variant serve different speed, size and sound-design needs.

FLAC import and 16-bit lossless export let you keep audio quality and embedded generation settings. WAV/MP3 and other offered output choices remain available.

## Upscaling and finishing

Generation models and finishing tools are separate choices. Select a processor in **Post Processing**, use **Late Post Processing** on a gallery result, or use **Media Flow** for supported batch workflows.

- **Spatial upscaling and detail**: SeedVR2, FlashVSR and LTX-based upscaling offer different appearance and memory trade-offs. LTX 2.5 Detail Refiner can restore texture in soft or compressed video. Lanczos provides conventional resizing; latent/VAE upsamplers apply only to compatible generation workflows.
- **Smoother motion**: RIFE supports x2/x3/x4 interpolation. Optional DLSS Frame Generation offers hardware-dependent multipliers on compatible RTX GPUs.
- **Neural refinement**: optional DLSS 5 can refine lighting, material appearance and detail at native or enlarged resolution, on supported hardware. See the [DLSS 5 guide](DLSS5.md) for setup and limits.
- **Faces**: H3 Face Refiner can refine up to five detected, tracked faces in a video.
- **Audio**: remove vocals, replace voices with SeedVC, add a soundtrack with MMAudio or PrismAudio, or remux existing tracks. Deepy Prime can also mix audio and package selectable audio/subtitle tracks.

See [Processing](PROCESSING.md) for input preparation, masks, control media and finishing workflows.

## Choosing variants

Model size alone does not determine whether a job will fit. Resolution, duration, references, batch size, quantization, attention backend, memory profile and tiling all affect RAM and VRAM. Select models can run with about **6 GB of VRAM**, but that is not a promise for every model or every setting.

For a new workflow:

1. Select the variant that matches your inputs and intended result. A reference editor, an animator and a text-to-video model are different workflows even within one family.
2. Start with the supplied defaults or an offered acceleration profile. A fixed-step distilled or PDD model should keep its required scheduler and step count.
3. Choose a memory profile suited to your GPU and available system RAM. WanGP's memory optimizations reduce the cost, but larger jobs still need more resources.
4. Test a short clip or moderate image resolution before increasing duration, batch size or references.
5. Use sliding windows for supported long-video workflows, and tiling where the model exposes it. Look for seams or continuity changes when tuning these options.
6. Add compatible LoRAs, controls and upsampling one feature at a time so their effects are easy to compare.

INT8, FP8, GGUF, NVFP4, NF4 and Nunchaku are checkpoint/quantization choices, not separate creative model families. Availability, memory savings, speed and output can vary with the model, GPU and installed kernels. Follow the [installation guide](INSTALLATION.md) or [AMD guide](AMD-INSTALLATION.md) for hardware-specific setup.

## Model switching and custom models

WanGP downloads models on demand. Optional preprocessing and postprocessing files also download only when a workflow needs them. Switching models does not require a restart; the previous model is unloaded as needed to recover memory.

Use the model selector or toolbar search, save reusable settings, and organize media in [Workspaces](WORKSPACES.md). The refreshed browser interface lets you check progress, inspect galleries and add generation jobs from another device connected to the same running WanGP server. The queue and galleries stay synchronized; each Gradio page keeps its own unsent prompts and draft settings. See [Authentication and network access](AUTHENTICATION.md) for remote setup.

[Deepy](DEEPY.md) can help select models and connect generation/editing steps. Its language-model engine is a separate choice from the image, video or audio model doing the generation; the phone-friendly Deepy Web App controls that same local workspace.

User-provided checkpoints belong in `finetunes/`, while model plugins can add other families. Refresh the model list after adding definitions. See [Finetunes](FINETUNES.md), [LoRAs](LORAS.md) and [Plugins](PLUGINS.md) for compatibility and setup.

---

> Applies to: Choosing WanGP video, image, speech, music and sound models; model-family and variant relationships; compatible references and workflows; upscaling and finishing choices; hardware-dependent memory and acceleration trade-offs. The model selector and model Help provide the current available variants and settings.
