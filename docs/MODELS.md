# Models

WanGP supports video, image, speech, music, and sound-generation models through a common interface. This page is a practical guide to the main model families, not an exhaustive list of every checkpoint, quantization, accelerator, or finetune.

The model selector in WanGP is the source of truth for the models available in your installation. It includes descriptions, model-specific help, and settings suited to each checkpoint. You can also use the toolbar search to switch models quickly.

Clicking a linked model name below loads it directly in WanGP.

## Recommended starting points

| Goal | Suggested model | Why pick it |
| --- | --- | --- |
| General image generation | [Krea 2](modeltype:krea2_raw), [Z-Image Turbo](modeltype:z_image) | Krea 2 targets polished, aesthetic images; Z-Image is an efficient 6B model suited to fast iteration. |
| Image editing and identity preservation | [Krea 2 Identity Edit](modeltype:krea2_raw_edit), [Qwen Image Edit Plus](modeltype:qwen_image_edit_plus_20B) | Krea 2 Identity Edit accepts up to two references and includes inpainting/outpainting. Qwen is strong at multi-subject composition and text in images. |
| Typography and graphic design | [Ideogram 4](modeltype:ideogram4) | Designed for layout, typography, and high-resolution graphics. Its visual prompt helper can build the model's structured JSON prompt. |
| Cinematic video with native audio | [LTX-2.3 Distilled 1.1](modeltype:ltx2_22B_distilled_1_1) | Generates video and audio together and supports start/end frames, control video, references, outpainting, injected frames, and sliding windows. |
| Connected stories with recurring characters | [JoyAI-Echo Surgical](modeltype:joyai_echo_surgical) | Uses reusable memories across shots to retain characters, voices, objects, and locations. |
| General text-to-video or image-to-video | [Wan 2.2 T2V](modeltype:t2v_2_2), [Wan 2.2 I2V](modeltype:i2v_2_2) | Mature general-purpose video models with broad LoRA and WanGP feature support. |
| Video editing and outpainting | [Wan VACE](modeltype:vace_14B), [Bernini-R](modeltype:bernini) | VACE provides mask/control workflows; Bernini can edit a video or generate from multiple reference images. |
| Character animation and motion transfer | [SCAIL-2](modeltype:scail2_14B), [Wan 2.2 Animate](modeltype:animate) | Animate or replace performers using reference images, pose/control video, and masks. |
| Talking heads and long dialogue | [LongCat Avatar 1.5](modeltype:longcat_avatar_v1_5), [InfiniteTalk](modeltype:infinitetalk) | Audio-driven avatar generation with sliding-window support for longer speech. |
| Voice cloning and dialogue | [Qwen3 TTS Base](modeltype:qwen3_tts_base), [IndexTTS2](modeltype:index_tts2) | Qwen3 is a flexible voice-cloning baseline; IndexTTS2 adds expressive emotion control. |
| Songs with lyrics | [ACE-Step 1.5 XL](modeltype:ace_step_v1_5_xl) | Strong lyric adherence and full-song generation with the XL audio model. |
| Music and sound effects | [Stable Audio 3](modeltype:stable_audio3_small) | Generates instrumentals, loops, ambience, and sound effects from descriptive prompts. |

<a id="video-models"></a>

## Video models

### LTX-2 and LTX-2.3

LTX-2 is WanGP's most flexible native audio-video family. Current 22B variants can generate images or videos, synthesize a soundtrack with the visuals, and work with:

- text, start/end frames, injected frames, control video, and control audio;
- video continuation and sliding windows;
- pose, depth, edge, HDR, inpainting, outpainting, and other IC-LoRA processes;
- talking-head generation and voice conditioning;
- multi-phase or direct high-resolution generation, depending on the selected model and process.

Use [LTX-2.3 Distilled 1.1](modeltype:ltx2_22B_distilled_1_1) as the fast general-purpose starting point. Dev variants trade speed for more control. [LTX-2.3 MSR V2](modeltype:ltx2_22B_msr_v2) accepts two to five subject/object references, while [JoyAI-Echo Surgical](modeltype:joyai_echo_surgical) is intended for connected multi-shot stories.

### Wan 2.1 and Wan 2.2

The Wan family includes general generation, editing, control, motion-transfer, identity, and talking-head architectures:

- [Wan 2.2 T2V](modeltype:t2v_2_2) and [Wan 2.2 I2V](modeltype:i2v_2_2) for general generation;
- [Wan 2.2 TI2V 5B](modeltype:ti2v_2_2) for a smaller unified text/image-to-video path;
- [VACE](modeltype:vace_14B) for inpainting, outpainting, object replacement, motion/depth/pose control, and other video-editing workflows;
- [Bernini-R](modeltype:bernini) for video-to-video generation and multi-reference conditioning;
- [SCAIL-2](modeltype:scail2_14B) and [Wan 2.2 Animate](modeltype:animate) for character animation or performer replacement;
- [Lynx](modeltype:lynx) and [VACE Lynx](modeltype:vace_lynx_14B) for identity-preserving face replacement;
- [MultiTalk](modeltype:multitalk) and [InfiniteTalk](modeltype:infinitetalk) for audio-driven dialogue;
- [Vista4D](modeltype:vista4d) for reshooting a dynamic scene from a new camera trajectory.

WanGP also ships accelerated defaults and specialist finetunes. Treat names such as FusioniX, FastWan, Lightning, Self-Forcing, Cocktail, Sparse, and NVFP4 as variants of a base workflow rather than separate model families.

### Other video families

- [Kandinsky 5 Pro](modeltype:k5_pro_t2v) provides 19B text-to-video and image-to-video models with controllable camera motion. Lite and distilled variants reduce generation cost.
- [HunyuanVideo 1.5](modeltype:hunyuan_1_5_t2v) provides 8.3B text-to-video and image-to-video models, plus distilled and upsampler variants.
- [LongCat Video](modeltype:longcat_video) is a general video model; [LongCat Avatar 1.5](modeltype:longcat_avatar_v1_5) is its distilled audio-driven avatar model.
- [Ovi](modeltype:ovi_1_1) generates video with a synchronized soundtrack and is especially suited to speaking characters.
- [Magi Human](modeltype:magi_human_distill) generates audio-driven talking-head video and offers staged high-resolution variants.
- Legacy families such as HunyuanVideo, LTX-Video 13B, Phantom, Recam, FantasySpeaking, SkyReels Diffusion Forcing, and Wan-Fun remain available where useful.

<a id="image-models"></a>

## Image models

### Krea 2

[Krea 2 RAW](modeltype:krea2_raw) is the undistilled CFG-guided checkpoint; [Krea 2 Turbo](modeltype:krea2_turbo) is the faster distilled choice. Identity Edit variants add instruction-based editing with up to two reference images. WanGP also exposes inpainting through LanPaint and negative prompting on distilled Krea 2 through NAG.

### Qwen Image

Qwen Image includes text-to-image, image editing, multi-reference editing, and layered-image variants. [Qwen Image Edit Plus](modeltype:qwen_image_edit_plus_20B) is a strong default for combining subjects and objects, preserving a scene while editing it, and rendering longer text. Quantized Nunchaku variants are available for supported hardware.

### Ideogram 4

[Ideogram 4](modeltype:ideogram4) focuses on typography, layout, graphic design, and structured composition. Plain prompts work, but the model-specific Magic Prompt and visual helper make it easier to author its JSON format. Turbo Time and NF4 variants provide alternative speed or memory trade-offs.

### Flux, Z-Image, and HiDream

- Flux includes Schnell, Dev, Kontext, Krea, Chroma, SRPO, Flux 2, and Klein architectures for generation, editing, and reference-guided workflows.
- [Z-Image Turbo](modeltype:z_image) is an efficient distilled 6B generator. Base, Control, Control 2.x, TwinFlow, and Nunchaku variants extend it to editing and lower-memory execution.
- [HiDream O1](modeltype:hidream_o1_dev_2604) supports text-to-image and image-reference generation, with control-image and latent-preview integration in WanGP.

<a id="audio-models"></a>

## Speech, music, and sound

WanGP groups audio-only models in the same selector:

- [Qwen3 TTS Base](modeltype:qwen3_tts_base) supports reference-audio voice cloning and two-speaker dialogue. Custom Voice and Voice Design are separate variants.
- [IndexTTS2](modeltype:index_tts2) supports zero-shot voice cloning, long dialogue, and text- or audio-guided emotion.
- [OmniVoice](modeltype:omnivoice) supports multilingual speech, voice design, cloning, and dialogue.
- Chatterbox and KugelAudio provide alternative speech and cloned-dialogue workflows.
- [ACE-Step 1.5 XL](modeltype:ace_step_v1_5_xl) and HeartMuLa generate songs, including lyrics-driven tracks.
- [Stable Audio 3](modeltype:stable_audio3_small) generates music, loops, ambience, and sound effects.
- Scenema Audio and DramaBox use LTX-2's audio knowledge for expressive scene-aware speech and dialogue.

Audio postprocessing is model-independent. For example, SeedVC can replace one or two voices in an existing audio or video result, while MMAudio and PrismAudio can add sound to silent video.

## Choosing variants

Model size alone does not determine whether a checkpoint will run. Resolution, frame count, attention backend, quantization, memory profile, tiling, sliding-window size, and enabled controls all affect RAM and VRAM use. The old fixed “6 GB / 12 GB / 20 GB” model tiers are therefore not reliable.

For a new workflow:

1. Start with the recommended or distilled variant.
2. Select a memory profile appropriate for your GPU.
3. Test a short clip or smaller image before increasing resolution or duration.
4. Use sliding windows for supported long-video workflows.
5. Add LoRAs, controls, references, and upsampling one feature at a time.

Quantized defaults such as quanto int8, FP8, GGUF, NVFP4, NF4, and Nunchaku reduce memory or target a specific backend. They should preserve the model's workflow, although speed, compatibility, and quality can vary with hardware and kernels.

## Model switching and custom models

WanGP loads models on demand and can switch without restarting. Use the model selector or toolbar search; the previous model is unloaded as needed to recover memory. Saved settings can retain model inputs and generation options.

The model list can be refreshed after adding or editing a finetune. User-provided checkpoints belong in `finetunes/`; model plugins can contribute additional families and defaults. See [Finetunes](FINETUNES.md), [LoRAs](LORAS.md), and [Plugins](PLUGINS.md) for details.
