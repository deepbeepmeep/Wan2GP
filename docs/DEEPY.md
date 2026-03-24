# Deepy

Deepy is WanGP's assistant for multi-step media work. It can generate, inspect, edit, extract, merge, and transform media by calling WanGP tools while keeping conversation context.

This guide covers:

- enabling Deepy
- configuring Deepy in the web UI
- linking WanGP settings files to Deepy generation tools
- using selected media, selected videos, and selected frames
- using Deepy from the CLI

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
- `Custom System Prompt`: text appended after the built-in Deepy system prompt on the next user turn

When the requirement is met, the `Ask Deepy` launcher appears in the WanGP web UI.

## Deepy Web Settings

Open `Ask Deepy`, then open the `Settings` panel.

### Generation Properties

- `Auto-abort or remove Deepy-started generation on Stop/Reset.`  
  Controls whether Deepy-created queue work is cancelled or removed when you stop/reset Deepy. This setting is persisted immediately when changed.

- `Use Properties defined in Settings Templates files.`  
  When enabled, Deepy uses the resolution, frame-count, and seed values already stored in the selected tool template.

- `Width` and `Height`  
  Default size overrides used only when template properties are disabled.

- `Number of Frames`  
  Default frame-count override for `Generate Video`, used only when template properties are disabled.

- `Seed (-1 for random)`  
  Default seed override, used only when template properties are disabled. `-1` means random.

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
- `+` to link that tool to the currently selected WanGP user settings file
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

When the tool is used later, Deepy rebuilds the real path from that logical reference and loads the original WanGP settings file directly. So any change made to the settings file will be directly usable with Deepy.

### Important behavior

- Only WanGP user settings selected from the `Lora / Settings` dropdown can be linked this way.
- System profiles and LoRA presets are rejected.
- If the linked WanGP settings file changes later, Deepy sees the updated content automatically.
- If the linked file disappears, Deepy falls back to that tool's default template.
- If the linked file still exists but is no longer eligible for that tool, the tool returns an eligibility error.
- Built-in templates cannot be deleted from the UI.

## How Deepy Interprets Media References

Deepy is designed to let you refer to existing media naturally.

In practice, Deepy will usually:

- prefer the currently selected gallery item when you say `selected`, `current`, `this image`, `this video`, or `this frame`
- resolve older outputs when you say things like `last image`, `previous video`, or describe a previous result
- use the selected video's current playback time when you refer to `the selected frame` or `the current frame`
- ask for clarification instead of inventing missing results when a reference is ambiguous

You can still use internal media ids such as `image_1` or `video_3`, but usually you do not need to.

## Using Selected Media

### In the web UI

For an image:

1. click the image you want
2. ask Deepy something like:
   - `edit this image so the sky is stormy`
   - `inspect the selected image and tell me whether the hands look correct`
   - `use the selected image as the start frame for a short video`

For a video:

1. select the video
2. scrub the player to the moment you care about
3. ask Deepy something like:
   - `inspect this frame and tell me whether the face is sharp`
   - `extract the selected frame as an image`
   - `cut a 3 second clip starting at the selected time`
   - `mute this video`
   - `replace the audio of the selected video with the last extracted audio`

### Previous outputs

Deepy can also resolve references such as:

- `last image`
- `previous video`
- `the robot dancing image`
- `image_2`
- `video_3`

## Deepy Tool Surface

Deepy currently exposes these main tool categories:

- creation and editing:
  - `Create Color Frame`
  - `Generate Image`
  - `Generate Video`
  - `Edit Image`
- media lookup and inspection:
  - `Get Selected Media`
  - `Get Media Details`
  - `Resolve Media`
  - `Inspect Media`
- extraction and conversion:
  - `Extract Image`
  - `Extract Video`
  - `Extract Audio`
  - `Mute Video`
  - `Replace Audio`
  - `Resize Crop`
  - `Merge Videos`
- WanGP documentation lookup:
  - `Load Doc`

## Example Requests

```text
Generate a cinematic image of a robot violinist on a rainy Paris rooftop at night.
```

```text
Create a black frame at 1280x720 so we can use it as a transition plate.
```

```text
Generate a short video of a paper boat floating through a glowing cave river.
```

```text
Inspect the selected image and tell me whether the composition is centered.
```

```text
Extract the selected frame from the current video and save it as an image.
```

```text
Take the last image and edit it so the background becomes a neon alley, while keeping the character identity.
```

```text
Use the robot-on-horse image as the start image and the robot-standing-next-to-horse image as the end image, then generate a smooth transition video.
```

```text
Cut a 5 second clip from the selected video starting at the selected time.
```

```text
Merge the last two generated videos into one clip.
```

```text
How do I use VACE for outpainting?
```

Multisteps requests:

```text
1) generate a robot that dances disco on top of a horse in a night club
2) now edit the image, the place hasn't changed but now the robot has gotten off the horse and the horse is standing next to the robot
3) verify the edited image is as expected otherwise generate another one
4) generate the transition between the two images
```
	
```text
Create a high quality image portrait that you think represents you best in your favorite setting. Then create an audio sample in which you will introduce the users to your capabilities. When done generate a video based on these two files.
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

For debug output and replay/prefill logs:

```bash
python wgp.py --ask-deepy --verbose 2
```

### CLI media selection

The CLI has its own virtual gallery. Add files to it, select one, and optionally set a playback time or frame for the selected video.

Example:

```text
/video E:\media\my_clip.mp4
/frame 120
inspect the selected frame and tell me whether the subject is centered
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
- It helps to drive Deepy by listing each step
- If you want Deepy to use the current video moment, scrub the selected video first, then refer to `this frame` or `the selected frame`.
- If a tool fails, Deepy is instructed to say so rather than inventing a result.
- For WanGP-specific questions, you can ask Deepy directly instead of searching the docs manually.
- Install GGUF kernels for fast inference and low VRAM

