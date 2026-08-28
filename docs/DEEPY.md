# Deepy

Deepy is WanGP's conversational media assistant. It can generate, inspect, edit, extract, transcribe, merge, and transform images, video, and audio while keeping conversation context.

Deepy comes in two versions:

- **Deepy Zero** is the lightweight, fast version for straightforward requests. It should spend less time thinking and is designed to work well with a smaller LLM (for instance Qwen3.5 4B or 9B).
- **Deepy Prime** is the advanced version for ambitious, multi-step work. It can discover available models and capabilities, plan connected actions, combine several image, video, and audio assets, and use external MCP services when configured. It requires Qwen3.8 VL 27B when run locally, or it can use a configured remote LLM.

Both versions share the same chat, Deepy settings and templates, Gallery integration, and interruption controls.

This guide covers:

- general guidelines
- enabling Deepy
- configuring Deepy in the web UI
- linking WanGP settings files to Deepy generation tools
- using selected and previous media naturally
- understanding which generation settings Deepy can override directly
- asking Deepy about available LoRAs and current defaults
- using Deepy Prime for long, multi-step projects
- using Deepy from the CLI

**Deepy can make mistakes, so verify important results.**

## General Guidelines
Once enabled, open the Deepy chat window by clicking `Ask Deepy` in the left dock.

Both versions can generate images, video, and audio. Deepy Zero is best for focused requests using your selected tools and templates. Deepy Prime can plan longer workflows and combine several generated or existing media assets. Content produced by either version appears in the `Image / Video Gallery` or `Audio Gallery` at the top right of WanGP's `Media Generator` tab.

Deepy can also work with User Imported Media:
1) Expand the section  `Media Info / Late Post Processing / Import Media`
2) Switch to the `Import Media to Galleries` tab
3) Select files to Import
4) Click `Import Videos / Images / Audio Files`

Once the media are in the galleries, you can refer to them using wording like `the last audio file`, `the selected video` or describe their content (Deepy will query the prompts stored in the generation metadata if they exist).

Deepy relies on predefined Template Settings for its seven generation tasks (`Generate Image`, `Generate Video`, `Edit Image`, `Generate Video with Speaker`, `Generate Song`, `Generate Audio from Description`, and `Generate Audio from Sample`). Deepy Zero directly uses the curated template assigned to each tool. Deepy Prime starts from the corresponding default template when no model is named, but can discover and compare other compatible models when the request requires it.

For Deepy Prime, every derived generation step follows the same rule. If a video workflow needs a master image and edited end frames, Prime uses the current default `Generate Image` and `Edit Image` templates directly instead of browsing models first. It performs model discovery only when the user asks to choose or compare models, names a model, or a required capability is demonstrably incompatible with the default template. Template settings already include model defaults, so fetching raw model defaults afterward is unnecessary.

WanGP comes with builtin templates ready to use but you may as well link presaved settings. You can access Deepy settings by clicking the `Settings` control on the right of the Deepy chat window.

In the web UI, Deepy settings changes take effect for the current Deepy session as soon as you make them. Click `Save Deepy Settings` at the bottom of the settings panel when you want to write those settings to disk for future WanGP sessions.

You can also define default width, height, frame count, audio duration, and seed in the Deepy Settings window. Select `Use by Default Always Dimensions / Durations / Seed Below` to apply them instead of the corresponding template properties without editing the templates.

You can also ask Deepy directly to override supported template settings such as width, height, frame count, audio duration, FPS, LoRAs, or inference steps.

## Enabling Deepy

Deepy is available only when these base conditions are met:

1. `Deepy` is set to `Deepy Zero` or `Deepy Prime` rather than `Disabled`.
2. The shared Prompt Enhancer / Deepy engine is a supported local Qwen model or a configured remote LLM. Remote LLMs require Deepy Prime.

With a local LLM, Deepy Prime additionally requires `Compaction Type When Cache is Full` to be set to `Summarize` and `Context Window Tokens` to be at least 32,000. Selecting Deepy Prime in the Configuration UI automatically raises a smaller context to 32,000 and selects Summarize. Configuration saving and runtime startup both reject an invalid local Prime configuration. Deepy Prime can instead use Codex, Claude Code, or OpenCode; see [Remote LLMs](REMOTE_LLMS.md) for installation, authentication, lifecycle, privacy, and troubleshooting.

Open the Configuration plugin and go to the `Prompt Enhancer / Deepy` tab.

Supported local Prompt Enhancer modes:

- `Qwen3.5VL Abliterated 4B`
- `Qwen3.5VL Abliterated 9B`
- `Qwen3.8VL Uncensored 27B` (required for local Deepy Prime)

Deepy settings in that tab:

- `Model used to power Prompt Enhancer / Deepy`: selects the shared language/vision model.
- `Speculative Decoding`: `Auto` enables it for Qwen3.5 9B with at least 12 GB VRAM or Qwen3.8 27B with at least 24 GB VRAM. Explicit `Yes` and `No` remain available.
- `Deepy`: selects `Disabled`, `Deepy Zero` for lightweight, straightforward work with curated WanGP tools and templates, or `Deepy Prime (requires Qwen3.8 VL 27B LLM)` for advanced planning, model discovery, multimedia workflows, and optional external MCP capabilities. Selecting Prime raises the context window to at least 32,000 tokens and selects Summarize compaction.
- `Deepy Filesystem Access`: disabled by default. Select read-only or read/write access to WanGP output folders and additional folders. In this scoped mode, tools expose `@outputs`, `@outputs2`, and custom aliases instead of physical paths; plain relative paths use `@outputs`.
- `Additional Filesystem Folders`: one folder per line. Append a unique alias after the path, quoting paths that contain spaces (for example `"D:\My Media" projects`), or let WanGP derive a unique alias.
- `Read Everywhere (Warning!)`: permits absolute-path reads and physical paths in results. It does not expand write access beyond output and additional folders.
- `Deepy VRAM Loading Mode`: controls whether Deepy stays in VRAM, unloads when idle, or unloads only when another WanGP component needs VRAM. The more Deepy stays in VRAM, the more responsive.
- `Context Window Tokens`: how much conversation and tool history Deepy tries to keep live. Thought, statement, and aggregate tool-call actions each receive up to 4,096 tokens below a 48,000-token context, 6,144 tokens from 48,000 through 63,999, and 8,192 tokens from 64,000 upward. Deepy prints the selected maximum when its local runtime starts.
- `KV Cache Quantization`: `Auto` enables fast INT8 KV cache when GGUF kernels 1.0.11 or newer are installed; explicit BF16 and INT8 choices remain available.
- `Compaction Type When Cache is Full`: `Summarize` lets Deepy preserve the important goals, decisions, completed work, and next steps from a long conversation when the full history no longer fits. `Discard` removes the oldest history. Local Deepy Prime requires `Summarize` and a context window of at least 32,000 tokens. Successful summaries appear as expandable `Context` entries in the chat.
- `Deepy Zero Prompt`: edits independent extra instructions for Deepy Zero.
- `Deepy Prime Guidance`: edits standing user guidance directly. It is prefilled to prefer the highest-quality base or full model unless the user prioritizes speed or names another model, and is appended to Deepy Prime's trusted system instructions rather than replacing them.
- `External MCP Servers (JSON)`: optional stdio, SSE, or Streamable HTTP MCP server definitions used only by Deepy Prime. External tool names are prefixed with their server name to avoid collisions.
- `Allow Searching for Changed MCP Executable Paths`: when enabled, Deepy can find an external MCP executable again after its versioned installation folder changes. The saved server definition is not rewritten.

When the requirement is met, the `Ask Deepy` launcher appears in the WanGP web UI.

Deepy Prime submits generation jobs through the normal WanGP queue. Jobs and completed media therefore remain visible in the usual queue and Galleries. Filesystem access and external MCP servers extend what Prime can work with according to the permissions you configure.

### Long projects

For large file collections, long sliding-window videos, or multi-chapter writing, Deepy Prime can keep exact items and a small durable project ledger in session working data instead of trying to hold everything in the visible conversation. Each managed item and its corresponding ledger update are saved together, so an interruption cannot advance one without the other. This working data survives automatic conversation summarization, but `Reset` clears it.

You can state the complete outcome normally, for example `plan the prompts for a ten-minute sliding-window video` or `write a twelve-chapter story and save it as one Markdown file`. Deepy divides the work into manageable batches, retrieves exact earlier data only when needed, checks that the collection is complete, and passes the completed plan directly to the generation or file-writing operation. For a large text deliverable, the final file is assembled from the stored chapters without making Deepy reproduce the whole document in a final answer or tool request.

External server configuration example:

```json
{
  "filesystem": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\media"]
  },
  "remote": {
    "transport": "streamable-http",
    "url": "http://127.0.0.1:9000/mcp"
  }
}
```

## Deepy Web Settings

Open `Ask Deepy`, then open the `Settings` panel.

The settings panel contains two expanded sections:

- `Generation Properties`
- `Template Settings used by Tools`

All changes in this panel are used immediately by the current Deepy web session. To keep them for future sessions, click `Save Deepy Settings` at the bottom of the panel.

### Generation Properties

- `Auto-abort or remove Deepy-started generation on Stop/Reset.`  
  Controls whether Deepy-created queue work is cancelled or removed when you stop/reset Deepy.

- `Default Dimensions / Durations / Seed`
  Choose whether Deepy uses these properties from each selected template or always replaces them with the panel defaults below.

- `Width` and `Height`  
  Default size overrides used only when template properties are disabled.

- `Default Number of Frames` and `Default Audio Duration`
  Default video length in frames and audio/song duration in seconds when template properties are disabled. Audio duration defaults to 10 seconds.

- `Number of Frames`  
  Default frame-count override for `Generate Video`, used only when template properties are disabled.

- `Seed (-1 for random)`  
  Default seed override, used only when template properties are disabled. `-1` means random.

Inference steps, FPS, LoRAs, and other model-specific values remain template-driven unless you ask for one of the supported per-request overrides described later in this guide.

### Tool Templates

Deepy has seven generation-tool template selectors:

- `Media Generator`
- `Video With Speech`
- `Image Generator`
- `Image Editor`
- `Song Generator`
- `Speech From Description`
- `Speech From Sample`

Each row has:

- a dropdown that selects the current template for that tool
- `+` to link that tool to the currently selected WanGP user settings file (in the dropdown in the upper left part of video gen tab )
- `trash` to remove the current live link and go back to the previous or default template

Changing a template selector updates the active Deepy web session immediately. Click `Save Deepy Settings` if you want to reuse the same selectors the next time you launch WanGP.

Deepy shows the selected template in the chat transcript for generation tools, for example:

```text
Generate Image [Z Image Turbo]
Generate Video [LTX-2 2.3 Distilled 1.0]
Edit Image [Flux Klein 9B]
```
### Save Deepy Settings

Click `Save Deepy Settings` at the bottom of the Deepy settings panel to persist the current web settings to disk.

That save includes:

- generation-property values such as auto-abort behavior, template-property usage, width, height, frame count, audio duration, and seed
- the currently selected Deepy template for each generation tool


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
- Linked templates are the right place for model-specific settings that Deepy does not expose directly. Deepy can still override width, height, frame count, audio duration, FPS, inference steps, and LoRAs on the supported tools.

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
   - `compare these five images and tell me which one has the sharpest face`
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

Deepy Zero is intended for direct requests such as generating one asset with a selected template, editing or animating selected media, extracting a clip, resizing a file, or producing a transcript. Deepy Prime can perform the same work while also selecting among compatible models and coordinating several dependent actions across multiple media.

- generate images, edit images, generate videos, generate talking videos from a still image plus speech audio, and create speech audio from a voice description or a voice sample
- create solid-color frames for transitions, blank frames, or color cards
- inspect images and video frames, and read local image, video, or audio details such as dimensions, duration, FPS, frame count, or audio track count
- extract images, video clips, or audio clips; transcribe audio or video; mute videos; replace audio; resize/crop or compose media side by side; and merge videos
- tell you which LoRAs are available for the current generation tool and which defaults a generation tool will use right now
- answer WanGP-specific usage questions by searching the bundled docs

Inspect Media accepts an optional normalized `[x_min, y_min, x_max, y_max]` `bbox` from 0 to 1000. Deepy crops that source-resolution area before applying the local or remote inspection size cap.


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

Multi-step requests are where Deepy Prime is most useful:

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
- `/duration [seconds]`: show or set default audio duration and disable template properties
- `/seed [value]`: show or set the default generation seed and disable template properties
- `/template <tool> <variant>`: set the template for any Deepy generation tool
- `/templates [tool]`: list available template variants
- `/template-props [on|off]`: show or toggle whether Deepy uses resolution, frame, audio-duration, and seed properties from templates

Session:

- `/help`: print the CLI command summary
- `/reset`: clear the Deepy conversation but keep the virtual gallery media
- `/quit`: exit the CLI session

Examples:

```text
/template gen_image "Z Image Turbo"
/template gen_video "LTX-2 2.3 Distilled 1.0"
/size 1280x720
/frames 97
/seed -1
```

## Practical Tips

- Deepy works best when your request clearly states the goal and how current media should be reused.
- Use Deepy Zero for focused requests and Deepy Prime when the result requires planning, model selection, or several connected media operations.
- Deepy Prime can infer a workflow from the requested outcome, but listing mandatory steps or constraints explicitly makes the result more predictable.
- If you need a model-specific setting that Deepy cannot override directly, store it in the linked template.
- Ask Deepy for available LoRAs or current defaults when you switch templates and want to confirm the setup.
- For image and video requests, be explicit about any must-keep details such as subject identity, composition, or mood.
- If you want Deepy to use the current video moment, scrub the selected video first, then refer to `this frame` or `the selected frame`.
- For transcription, mention if you want word timestamps or a specific audio track.
- If a tool fails, Deepy will tell you rather than inventing a result.
- Stopping a turn requests cancellation of active Deepy-started work according to the `Auto-abort` setting. Completed work remains in the conversation; an unfinished action is not resumed automatically.
- For WanGP-specific questions, you can ask Deepy directly instead of searching the docs manually.
- Install GGUF kernels for fast inference and low VRAM.
