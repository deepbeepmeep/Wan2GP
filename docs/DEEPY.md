# Deepy

Deepy is WanGP's assistant for multi-step media work. It can generate, inspect, edit, extract, transcribe, merge, and transform image, video, and audio while keeping conversation context.

This guide covers:

- general guidelines
- enabling Deepy
- configuring Deepy in the web UI
- linking WanGP settings files to Deepy generation tools
- using selected and previous media naturally
- understanding which generation settings Deepy can override directly
- asking Deepy about available LoRAs and current defaults
- using Deepy from the CLI

## General Guidelines
Once enabled (see below), Deepy becomes accessible by opening Deepy chat window when you click on the left dock `Ask Deepy`

Deepy is capable to generate image, video & audio, and then combine them to produce new media. All content produced by Deepy will be found in the `Image / Video Gallery` and the `Audio Gallery` at the top right of the WanGP `Video Generator` tab.

Deepy can also work with User Imported Media:
1) Expand the section  `Media Info / Late Post Processing / Import Media`
2) Switch to the `Import Media to Galleries` tab
3) Select files to Import
4) Click `Import Videos / Images / Audio Files`

Once the media are in the galleries, you can refer to them using wording like `the last audio file`, `the selected video` or describe their content (Deepy will query the prompts stored in the generation metadata if they exist).

Deepy simply can not infer the best generation settings based for your request, since the combinations are too many and depend on the generation model you want to use. So Deepy relies on predefined Template Settings for its main 6 generation tools (`Generate Image`, `Generate Video`, `Edit Image`, `Generate Video with Speaker`, `Generate Audio from description`, `Generate Audio from Sample`). 

WanGP comes with builtin templates ready to use but you may as well link presaved settings. You can access Deepy settings by clicking the `Settings` doc on the right of the Deepy Chat Window.

You can also define Default Width & Height to use for all the generation tools in Deepy Settings Window. These will be used only if the checkbox `Use Properties defined in Settings in Templates files` is not checked. This is convenient if you want to override the values defined in the templates without modifying them.

Last but not least you can ask directly Deepy to override the following template settings: `width`, `height`, `num of frames`, `fps`, `loras` or `num inference steps`. 

## Enabling Deepy

Deepy is available only when both of these conditions are met:

1. `Enable Deepy` is turned on.
2. Prompt Enhancer is set to a supported Qwen3.5VL mode.

Open the Configuration plugin and go to the `Prompt Enhancer / Deepy` tab.

Required Prompt Enhancer modes:

- `Qwen3.5VL Abliterated 4B`
- `Qwen3.5VL Abliterated 9B`

Deepy settings in that tab:

- `Enable Deepy`: turns Deepy on or off
- `Deepy VRAM Loading Mode`: controls whether Deepy stays in VRAM, unloads when idle, or unloads only when another WanGP component needs VRAM. The more Deepy stays in VRAM, the more responsive.
- `Context Window Tokens`: how much conversation and tool history Deepy tries to keep live
- `Custom System Prompt`: extra instructions appended to Deepy on the next user turn

When the requirement is met, the `Ask Deepy` launcher appears in the WanGP web UI.

## Deepy Web Settings

Open `Ask Deepy`, then open the `Settings` panel.

### Generation Properties

- `Auto-abort or remove Deepy-started generation on Stop/Reset.`  
  Controls whether Deepy-created queue work is cancelled or removed when you stop/reset Deepy. This setting is persisted immediately when changed.

- `Use Properties defined in Settings Templates files.`  
  When enabled, Deepy uses the selected tool template as-is. When disabled, Deepy still starts from the template, but replaces only width, height, video frame count, and seed with the panel defaults below.

- `Width` and `Height`  
  Default size overrides used only when template properties are disabled.

- `Number of Frames`  
  Default frame-count override for `Generate Video`, used only when template properties are disabled.

- `Seed (-1 for random)`  
  Default seed override, used only when template properties are disabled. `-1` means random.

Inference steps, FPS, LoRAs, and other model-specific values remain template-driven unless you ask for one of the supported per-request overrides described later in this guide.

The remaining Deepy settings in this panel are persisted when you use `Ask`.

### Tool Templates

Deepy has 6 generation-tool template selectors:

- `Video Generator`
- `Video With Speech`
- `Image Generator`
- `Image Editor`
- `Speech From Description`
- `Speech From Sample`

Each row has:

- a dropdown that selects the current template for that tool
- `+` to link that tool to the currently selected WanGP user settings file (in the dropdown in the upper left part of video gen tab )
- `trash` to remove the current live link and go back to the previous or default template

Deepy shows the selected template in the chat transcript for generation tools, for example:

```text
Generate Image [Z Image Turbo]
Generate Video [LTX-2 2.3 Distilled]
Edit Image [Flux Klein 9B]
```

## Linking WanGP Settings to Deepy Tools

Deepy templates are either:

- built-in Deepy templates shipped with WanGP
- live links to WanGP user settings files

### Link a tool from the UI

Practical workflow:

1. configure a normal WanGP generation the way you want
2. save it as a WanGP user settings file
3. select that user settings JSON in WanGP's `Lora / Settings` dropdown
4. open Deepy settings
5. click `+` next to the Deepy tool you want to link
6. confirm the link

When you use the tool later, Deepy reads the linked WanGP settings file directly, so changes to that file are picked up automatically.

### Important behavior

- Only WanGP user settings selected from the `Lora / Settings` dropdown can be linked this way.
- System profiles and LoRA presets are rejected.
- If the linked WanGP settings file changes later, Deepy sees the updated content automatically.
- If the linked file disappears, Deepy falls back to that tool's default template.
- If the linked file still exists but is no longer eligible for that tool, the tool returns an eligibility error.
- Built-in templates cannot be deleted from the UI.
- Linked templates are the right place for model-specific settings that Deepy does not expose directly. Deepy can still override width, height, frame count, FPS, inference steps, and LoRAs on the supported tools.

## How Deepy Interprets Media References

Deepy is designed to let you refer to existing media naturally.

In practice, Deepy will usually:

- prefer the currently selected image, video, or audio item when you say `selected`, `current`, `this image`, `this video`, `this audio`, or `this frame`
- use the selected video's current playback time when you refer to `the selected frame` or `the current frame`
- resolve short references such as `last image`, `previous video`, or `last audio`
- resolve older outputs when you describe a previous result
- ask for clarification instead of inventing a result when a reference is ambiguous

You can still use internal media ids such as `image_1` or `video_3`, but usually you do not need to.

## Using Selected Media

### In the web UI

For an image:

1. click the image you want
2. ask Deepy something like:
   - `edit this image so the sky is stormy`
   - `inspect the selected image and tell me whether the hands look correct`
   - `use the selected image as the start frame for a short video`
   - `use this image and the last audio clip to make a talking video`

For a video:

1. select the video
2. scrub the player to the moment you care about
3. ask Deepy something like:
   - `inspect this frame and tell me whether the face is sharp`
   - `extract the selected frame as an image`
   - `cut a 3 second clip starting at the selected time`
   - `transcribe this video`
   - `mute this video`
   - `replace the audio of the selected video with the last extracted audio`

For audio:

1. select or import an audio file
2. ask Deepy something like:
   - `transcribe this audio`
   - `transcribe this audio with word timestamps`
   - `create speech from this sample saying: Welcome to WanGP`
   - `use this audio with the selected image to make a talking video`

If your voice sample is inside a video, Deepy can extract the audio first.

### Previous outputs

Deepy can also resolve references such as:

- `last image`
- `previous video`
- `last audio`
- `the robot dancing image`
- `image_2`
- `video_3`

## What You Can Ask Deepy To Do

- generate images, edit images, generate videos, generate talking videos from a still image plus speech audio, and create speech audio from a voice description or a voice sample
- create solid-color frames for transitions, blank frames, or color cards
- inspect images and video frames, and read local image, video, or audio details such as dimensions, duration, FPS, frame count, or audio track count
- extract images, video clips, or audio clips; transcribe audio or video; mute videos; replace audio; resize/crop media; and merge videos
- tell you which LoRAs are available for the current generation tool and which defaults a generation tool will use right now
- answer WanGP-specific usage questions by searching the bundled docs


## Audio Transcription

Deepy can transcribe either audio or video.

- Segment timestamps are returned by default.
- Ask for word timestamps if you need more detailed timing.
- If a source has multiple audio tracks, mention which track you want.

Example requests:

```text
Transcribe the selected video.
```


```text
Transcribe audio track 2 from the selected video.
```

```text
Extract the video excerpt that starts with 'I will be back'.
```

## Example Requests

```text
Generate a cinematic image of a robot violinist on a rainy Paris rooftop at night.
```

```text
Edit the selected image so the background becomes a neon alley while keeping the character identity, and use 8 inference steps.
```

```text
Generate a short video of a paper boat floating through a glowing cave river at 24 fps with 97 frames and 8 inference steps.
```

```text
Generate a video of a dog playing under the rain using the Lego lora
```

```text
Use the selected portrait and the last audio clip to make a talking video.
```

```text
Create speech from this sample saying: Welcome to WanGP.
```

```text
How do I use VACE for outpainting?
```

Multi-step requests:

```text
1) Generate an image of a robot disco dancing on top of a horse in a nightclub.
2) Edit the image so the setting stays the same, but the robot has gotten off the horse and the horse is standing next to the robot.
3) Verify that the edited image matches the description; if it does not, generate another one.
4) Generate a transition between the two images.
```

```text
Create a high quality portrait that represents you well. Then create a speech sample in which you introduce your capabilities. When done generate a talking video from the portrait and the generated speech.
```

## Deepy CLI Mode

Launch Deepy in CLI mode with:

```bash
python wgp.py --ask-deepy
```

At startup, the CLI prints the Deepy logo and preloads the prompt-enhancer runtime so Deepy is ready before the first prompt.

### Prompt entry

Interactive multiline entry:

- `Enter`: send the current prompt
- `Ctrl+Enter`: insert a newline on terminals that expose it
- `Alt+Enter`: insert a newline
- `Ctrl+J`: newline fallback
- `Ctrl+S`: stop the current Deepy turn while it is running
- `Shift+Enter`: not available here because the console reports it as plain `Enter`


### CLI media selection

The CLI has its own virtual gallery. Add files to it, select one, and optionally set a playback time or frame for the selected video.

Examples:

```text
/video E:\media\my_clip.mp4
/frame 120
inspect the selected frame and tell me whether the subject is centered
```

```text
/audio E:\media\voice.wav
transcribe the selected audio with word timestamps
```

When a Deepy tool generates media in CLI mode, the CLI prints the generated output path.

### CLI commands

Media:

- `/add <path>`: add and select an image, video, or audio file
- `/image <path>`: add and select an image file
- `/video <path>`: add and select a video file
- `/audio <path>`: add and select an audio file
- `/list [scope]`: list known media; `scope` can be `all`, `media`, `image`, `video`, or `audio`
- `/media [scope]`: alias for `/list`
- `/clear-media`: remove all virtual gallery media

Selection:

- `/select <ref>`: select media by id, list index, or name fragment
- `/select-video <media_id>`: select a video by media id
- `/selected`: show the currently selected media
- `/selected-video`: show the selected video media id
- `/time <secs>`: set the selected video's playback time
- `/frame [index]`: show or set the selected video frame, 0-based

Deepy settings:

- `/settings`: show the current CLI Deepy settings
- `/size [WxH]`: show or set default generation size and disable template properties
- `/frames [count]`: show or set default `gen_video` frame count and disable template properties
- `/seed [value]`: show or set the default generation seed and disable template properties
- `/template <tool> <variant>`: set the template for any Deepy generation tool
- `/templates [tool]`: list available template variants
- `/template-props [on|off]`: show or toggle whether Deepy uses resolution, frame, and seed properties from templates

Session:

- `/help`: print the CLI command summary
- `/reset`: clear the Deepy conversation but keep the virtual gallery media
- `/quit`: exit the CLI session

Examples:

```text
/template gen_image "Z Image Turbo"
/template gen_video "LTX-2 2.3 Distilled"
/size 1280x720
/frames 97
/seed -1
```

## Practical Tips

- Deepy works best when your request clearly states the goal and how current media should be reused.
- For multi-step tasks, list the steps in order.
- If you need a model-specific setting that Deepy cannot override directly, store it in the linked template.
- Ask Deepy for available LoRAs or current defaults when you switch templates and want to confirm the setup.
- For image and video requests, be explicit about any must-keep details such as subject identity, composition, or mood.
- If you want Deepy to use the current video moment, scrub the selected video first, then refer to `this frame` or `the selected frame`.
- For transcription, mention if you want word timestamps or a specific audio track.
- If a tool fails, Deepy will tell you rather than inventing a result.
- For WanGP-specific questions, you can ask Deepy directly instead of searching the docs manually.
- Install GGUF kernels for fast inference and low VRAM.
