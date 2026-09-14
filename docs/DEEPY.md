# Deepy

Deepy is WanGP's conversational media assistant. Tell it what you want to create or change, and it can generate, inspect, edit, extract, transcribe, merge, and transform images, video, and audio while keeping the context of your project.

Use Deepy when you want to work toward an outcome instead of manually operating every generation and post-processing control. For example, you can ask it to turn a portrait and a voice recording into a talking video, compare several results, or build a sequence of related shots.

**Deepy can make mistakes, so verify important results.**

## Choose Deepy Zero or Deepy Prime

- **Deepy Zero** is fast and lightweight. Use it for focused tasks such as generating one asset, editing selected media, extracting a clip, resizing a file, or producing a transcript. It works well with the smaller supported Qwen models.
- **Deepy Prime** is for projects that need planning or several connected actions. Use it when Deepy must compare compatible models, combine multiple media assets, inspect intermediate results, manage project files, or work with external MCP services. Prime requires Qwen3.8 VL 27B locally or a configured remote LLM.

Both versions share generation features, galleries, media references, and saved sessions. Both offer a dedicated video-with-references template. Choose the assistant in **Configuration > Prompt Enhancer / Deepy**.

## Enable Deepy

1. Start WanGP with `python wgp.py`.
2. Open **Configuration > Prompt Enhancer / Deepy**.
3. Choose **Deepy Zero** or **Deepy Prime**.
4. Choose a supported local model or, for Prime, a configured remote LLM.
5. Save the configuration.

Supported local choices are:

- `Qwen3.5VL Abliterated 4B`
- `Qwen3.5VL Abliterated 9B`
- `Qwen3.8VL Uncensored 27B` (required for local Deepy Prime)

For a remote Prime engine, see [Remote LLMs](REMOTE_LLMS.md). Once Deepy is enabled, **Ask Deepy** appears in the Gradio left dock. The CLI and standalone Web app use the same saved configuration.

## Choose an interface

| Interface | Launch | Best for |
|---|---|---|
| **Gradio + Web** | `python wgp.py --listen --server-port 7860` | Full WanGP controls on the PC plus a synchronized, phone-friendly Deepy app at `/deepy/`. |
| **CLI** | `python wgp.py --ask-deepy` | Terminal-based work without a browser or server. |
| **Standalone Web** | `python wgp.py --deepy-server --listen --server-port 7860` | A focused browser app for chat, galleries, and basic Deepy settings without Gradio. |

All interfaces use the same engine configuration, tool templates, and saved-session format. Use the same `--config FOLDER` and `--deepy-sessions-dir FOLDER` when you want separate launches to use the same configuration and sessions. `--config` names the folder containing `wgp_config.json`, not the JSON file itself.

### Use Gradio and Web together

While Gradio is running, open:

- `http://<PC-address>:7860/` for Gradio
- `http://<PC-address>:7860/deepy/` for the phone-friendly app

You can also follow **Web app ->** in Deepy's settings. Both views share the current conversation, galleries, selected media, settings, progress, and generation queue. Start a request on one device and follow, pause, stop, or inspect its result on the other.

The Deepy Web app and all open Gradio pages are synchronized while connected to the same WanGP process. Gallery workspace changes and completed generations appear everywhere, while each Gradio page keeps its own unsent generation form and draft values.

Closing every browser does not stop accepted work while WanGP remains running. Reopen the same address to recover the current view. Live sharing applies only to Gradio and Web clients connected to the same WanGP process.

### Use standalone Web

For the same computer:

```powershell
python wgp.py --deepy-server
```

For a phone or another computer on the local network:

```powershell
python wgp.py --deepy-server --listen --server-port 7860
```

Open `http://<PC-address>:7860`, using the PC's LAN address, such as `192.168.1.50`. Do not enter `0.0.0.0`; that is only the listening address. If the page cannot connect, allow the selected port through the PC firewall.

Standalone Web and CLI run as separate processes and do not attach to another running instance. To continue work in a different process, enable saved sessions, finish current work, stop the old process, and resume the session using the same configuration and session folder.

## What Deepy can do

You can ask Deepy to:

- generate or edit images
- generate video from a prompt, image, or compatible media inputs
- create a talking video from a still image and speech audio
- create speech from a description or voice sample, generate songs, and generate sound from a description
- inspect images and video frames or compare several visuals
- report useful media details such as dimensions, duration, FPS, frame count, and audio tracks
- extract images, clips, or audio; transcribe audio or video; mute or replace audio; resize or crop media; compose assets side by side; and merge videos
- find available LoRAs, explain the active tool defaults, and answer WanGP usage questions

Prime is especially useful for requests with dependencies. It can create an intermediate asset, inspect it, revise it if necessary, and use the accepted result in the next stage.

## Everyday workflow

1. Select a suitable template for each generation tool.
2. Import or select any source media.
3. Describe the result you want and any details that must be preserved.
4. Review the generated media and ask for refinements in the same conversation.

Deepy uses the configured template for each generation task. Zero uses the selected template directly. Prime also starts with the selected template and looks for another compatible model when you request model selection, name a model or family, request a declared speciality such as infographics, or the current template cannot perform the task.

You can override supported values in a request, including width, height, frame count, audio duration, FPS, inference steps, seed, and LoRAs. Put model-specific choices that Deepy cannot override into a linked template.

## Work with media

### Import and attach media

In Gradio, either use the chat **+** button or open **Media Info / Late Post Processing / Import Media > Import Media to Galleries**. In the Web app, use the chat **+** for media needed by the next request, or import files from a gallery.

The attachment counter shows how many files are waiting for the next request. Sending the request consumes those attachments. Removing an attachment from the composer does not delete its workspace file.

### Select media and refer to it naturally

Select a gallery item, then use phrases such as:

- `edit this image`
- `inspect the selected frame`
- `transcribe the last audio`
- `use the previous video`
- `compare these images`

Deepy normally prefers the selected item for words such as `selected`, `current`, `this image`, `this video`, or `this audio`. It can also resolve recent results and descriptions such as `the robot dancing image`. Internal IDs such as `image_2` and `video_3` remain available when you need precision.

In Gradio, scrub a selected video before referring to `this frame` or `this time`. In the Web app, viewing or playing a video does not select a reference moment, so include the time or frame number in your request. Deepy asks for clarification when a reference is ambiguous.

When generating video from an image, the image can have different roles. A **start image** anchors the opening scene and composition. A **reference image** guides a subject's identity or appearance without fixing the opening frame, according to the model's supported mode. For example, “animate this photo” and “put this character in a new scene” use the source differently. A model may support one role, both, or other roles such as inserting frames at specified positions.

### Browse galleries

The image/video and audio galleries let you select, inspect, play, and download media. Selecting a tile marks it for your next request; it does not submit anything. The information pane shows prompts, model settings, dimensions, and creation details when available.

When the latest item is selected, new output is selected automatically. If you are reviewing an older item, Deepy preserves that selection and your position.

## Control work in progress

The chat and generation progress bars show what Deepy and WanGP are doing.

- **Pause** suspends Deepy's current turn without losing it. A generation or tool operation already underway is allowed to finish, then Deepy pauses before the next action. This is useful when another WanGP task needs the GPU.
- **Resume** continues the same turn from where it paused.
- **Stop** ends Deepy's current turn. Depending on the **Auto-abort** setting, it may also cancel or remove generation work started by Deepy.
- **Abort** cancels the active WanGP generation.

You can send a new instruction while Deepy is working. Use this to steer the current task, or queue the instruction for afterward when the interface offers that choice.

**Compacted View of Thoughts and Actions** keeps long turns readable. A collapsed turn shows its latest useful statement, generated media, and final answer; expand it to review individual thoughts and actions. The preference is shared between Gradio and Web.

## Tool templates and defaults

Open **Ask Deepy > Settings** in Gradio for the complete settings panel. The standalone Web app provides the commonly used generation defaults and existing template choices.

### Prime model preferences

At the top of **Templates Settings used by Tools**, Prime offers **Speed** and **Model Size**, also available in the Web app. They apply when Prime chooses beyond your configured templates; an explicit model or template request takes priority.

- **Speed:** Fast (default) favors models with native acceleration or accelerator profiles. Standard favors ordinary generation without automatically adding an accelerator. No preference ignores speed.
- **Model Size:** Smaller (default) favors declared lighter variants; Larger favors full variants. No preference ignores size. These labels compare variants, not absolute GB, VRAM use or quality.

Prime first checks required inputs and the requested speciality, then prefers candidates matching both preferences over those matching only one. For example, an H3 request with a subject reference can select a compatible lighter Ref2VA variant and its recommended accelerator. An infographic request can find a model declaring that strength. If only some speciality terms match, Prime receives the missing terms and checks essential requirements before generating.

Fast uses a model's recommended accelerator settings when available. Natively accelerated models already use accelerated defaults. Your configured templates keep their settings; these preferences do not replace them on ordinary requests. Changes reach Prime as a hidden runtime update on the next turn, including with a remote LLM; no conversation restart is needed. Deepy Zero has no model-preference controls or injection.

### Generation defaults

Choose whether each tool uses dimensions, durations, and seed from its template or replaces them with Deepy's defaults:

- width and height
- video frame count
- audio duration
- seed (`-1` means random)

Choose **Use template defaults first** to prefer the selected template's values, or **Use the values below** to prefer your Deepy defaults. For Deepy Prime, if a generation request omits dimensions, duration, or seed, WanGP fills the missing values from your Deepy defaults before using the model's factory settings. Values already supplied in the request are preserved.

Most changes apply immediately. Click **Save Deepy Settings** to reuse them after restarting WanGP.

### Template choices

Deepy has templates for:

- Image Generator
- Image Editor
- Video Generator
- Video With Speech
- Video Generator with Ref.
- Song Generator
- Speech From Description
- Speech From Sample

Deepy uses **Video Generator with Ref.** when images or videos supply subject identity, appearance or motion. A start image instead fixes the opening scene and uses the regular video template. Available reference templates are MiniMax H3 Ref2VA Pruned with its eight-step accelerator (default), LTX-2 2.3 MSR V2 Distilled 1.1, and Vace Fusionix. H3 accepts image and video references; the other templates accept image references. Deepy checks model support when combining references with other inputs.

WanGP includes built-in templates. To reuse your own model setup:

1. Configure and save normal WanGP generation settings.
2. Select that user settings file in WanGP's **Lora / Settings** dropdown.
3. Open Deepy settings.
4. Click **+** beside the Deepy tool that should use it.
5. Confirm the link and save Deepy settings.

The link follows later changes to that user settings file. The trash action removes the live link and returns to the previous or built-in template. If a linked file is deleted, Deepy returns to the tool's default template.

## Sessions and workspaces

Use saved sessions when you want to continue a conversation after restarting WanGP. Workspaces organize the media shown in the galleries.

### Choose a session mode

In **Ask Deepy > Settings > Sessions**, choose:

- **Disabled**: one temporary conversation. Reset clears it. Use this for quick, disposable work.
- **Multisessions with selectable Workspace**: save conversations while choosing which general gallery workspace each one uses. Use this when several projects share media collections.
- **Multisessions with dedicated Workspace**: give every saved conversation its own gallery workspace. Use this for self-contained projects and the clearest separation between jobs.

Save the setting and restart WanGP if Deepy has already started. In a dedicated workspace, the first request names the session and workspace. You may import media before that request; it is kept under **New Deepy session** until the first request supplies a name.

Sessions save automatically. There is no separate Save Session action. A saved session includes its conversation, completed actions, displayed results, workspace association, and media references. When resuming a long session, the conversation may appear before Deepy finishes preparing it.

In Gradio you can resume, rename, duplicate, export, import, or delete sessions. In Web, use the session selector in the top bar. In CLI, use `/sessions` and `/resume <ref>`. Wait for active or paused work to finish before switching sessions. If an automatic or manually entered session name is already used, Deepy adds an available number, such as **My video (2)**. Names differing only in capitalization count as duplicates.

### Keep links or copy media

The **Gallery Media** session setting controls portability:

- **Keep links to Gallery files** saves disk space, but the original files must remain at their recorded locations.
- **Copy Gallery files into each session** uses more disk space but makes the session independent of the original gallery files.

### Organize workspaces

A gallery workspace remembers gallery contents, media order, selected items, and the active gallery tab. It stores references to the original files rather than duplicating them. Ordinary workspaces use a folder icon; session-owned workspaces use a robot icon. Creating, renaming, or deleting a workspace changes its organization only; deleting a workspace does not delete its media files.

In Gradio, the magnifier beside the workspace selector opens the full workspace viewer. Use it to:

- inspect every item, including older media hidden by the gallery display limit
- select multiple items with Ctrl/Cmd-click, Shift-click, or rectangle selection
- reorder media or move a selection to the start or end
- eject items from a workspace without deleting their files
- copy items to another workspace
- download selected items as a ZIP
- import more media
- permanently delete selected files after confirmation

The lock protects a workspace from automatic archiving. The broom sets an inactivity period for archiving old, unlocked workspaces at WanGP startup. Archiving hides the workspace but leaves its media files in place. There is currently no restore button, so protect any workspace you expect to revisit.

See [Gallery Workspaces](WORKSPACES.md) for the relationship between workspaces and galleries, the complete workspace-manager workflow, synchronization rules, dedicated Deepy workspaces, safe deletion, archiving, restoration, and backups.

## Use Deepy on a phone

The Web app has tabs for **Chat**, **Image/video**, **Audio**, and **Settings**. Replies, results, selections, and progress stay synchronized with Gradio when both connect to the same process.

For an app-like view on iPhone, open the Deepy address in Safari and choose **Share > Add to Home Screen**. On Android, use **Install app** or **Add to home screen**. The connection overlay tells you when the phone loses contact with WanGP and disappears after reconnection.

### Protect network access

Authentication is optional and off by default. `--auth` enables one password-only login for **all Gradio and Deepy web access**, including APIs, galleries, downloads, uploads and live connections. No username is needed.

```powershell
# Generate a new password and print it in the terminal
python wgp.py --listen --auth

# Choose a fixed passphrase
python wgp.py --listen --auth --auth-password "your long private passphrase"

# The same options work with the standalone Deepy Web app
python wgp.py --deepy-server --listen --auth
```

Open the usual Gradio or Deepy address and enter the password. A login covers both interfaces on the same hostname. Browser sessions expire after 24 hours; restarting WanGP invalidates every session. A generated password also changes at each launch. To avoid putting a fixed passphrase in command history, set `WANGP_AUTH_PASSWORD` in the launch environment and use `--auth`. An explicit `--auth-password` takes precedence. Passwords supplied by you are not printed by WanGP.

Login attempts are limited across all clients and web interfaces in this process. The first four failures have no delay. After failure 5, wait 30 seconds; each further failure adds 30 seconds, reaching 450 seconds after failure 19. From failure 20, only one attempt every ten minutes is allowed. Only one password check can run at a time. Requests during the waiting period do not extend it. A successful login resets the failure counter. Existing signed-in sessions keep working during a cooldown. Restarting WanGP resets the counter as well as all sessions.

Choose protection according to how the server is reached:

- **Only this PC:** the default localhost access usually needs no application password or certificate.
- **Trusted private LAN:** authentication is useful on shared networks. HTTPS protects the passphrase and generated media from network interception.
- **VPN-only access:** application authentication can be optional if firewall/VPN rules restrict access to trusted users and the entire connection is protected. Keep public port forwarding closed. A VPN ending at your router may leave the final LAN connection unencrypted.
- **Public access, including NAT port forwarding:** enable authentication and trusted HTTPS. NAT alone does not protect a forwarded port. Forward only the HTTPS port; never expose a password login over plain HTTP.

Network MCP has a **separate OAuth login**, enabled with `--mcp-auth`. The web password and browser cookie do not authorize MCP clients. See [MCP authentication](API.md#mcp-authentication-and-https).

### Set up HTTPS

The certificate options apply to Gradio, Deepy and network MCP. Obtain a certificate and private key for the exact hostname clients will use. Public access needs a certificate trusted by those clients, commonly issued for your domain by a public certificate authority or managed by an HTTPS reverse proxy. For a private LAN, [mkcert](https://github.com/FiloSottile/mkcert) can create a local certificate; each client device must trust that local certificate authority. Keep its CA private key and the server private key private.

Serve HTTPS directly on the main port:

```powershell
python wgp.py --listen --auth --server-port 7860 --ssl-certfile C:\certs\wangp.pem --ssl-keyfile C:\certs\wangp-key.pem
```

Open `https://<certificate-hostname>:7860/`, or `/deepy/` for the mobile Web app. Add `--deepy-server` for standalone Deepy at `/`.

To redirect HTTP on the main port to a separate HTTPS port:

```powershell
python wgp.py --listen --auth --server-port 7860 --https-port 7861 --ssl-certfile C:\certs\wangp.pem --ssl-keyfile C:\certs\wangp-key.pem
```

Use `https://<certificate-hostname>:7861/`. The HTTP port redirects; it does not serve a second unencrypted application. Alternatively, set `WANGP_SSL_CERT` and `WANGP_SSL_KEY` in the launch environment. Command-line certificate paths take precedence. Missing, mismatched or unreadable certificate/key files stop startup.

An HTTPS reverse proxy can manage certificates instead. Keep its WanGP backend private, preserve the original Host header and forward the correct scheme from a trusted local proxy. Configure proxy authentication separately if you want another access restriction.

Browser microphone recording can require trusted HTTPS even over a VPN. Native phone keyboard dictation does not use Deepy's microphone access.

## Voice input and transcription

In Gradio and desktop Web, press the microphone beside **Send**, speak, then press it again to transcribe. The text remains editable before you submit it. On smartphones, the standalone Web app uses the phone keyboard's native dictation instead.

In **Configuration > General**, choose the microphone transcription mode:

- **Auto** uses the GPU when it is available and the CPU while the GPU is busy.
- **CUDA** waits for the GPU and is usually faster once it starts.
- **CPU** avoids competing for GPU memory.
- **Disabled** hides the microphone controls.

Choose **Voice transcription language** to improve short recordings, or leave it on **Auto**. You can override the language for one launch with `--deepy-voice-language es`, `en`, or `auto`. Whisper files download on first use.

Dictating a request is different from asking Deepy to transcribe an existing audio or video file. For media transcription, select or attach the file and ask for segment timestamps, word timestamps, or a particular audio track.

## CLI mode

Launch the terminal interface with:

```powershell
python wgp.py --ask-deepy
```

Add `--config`, `--deepy-sessions-dir`, or `--output-dir` when you want to use custom locations. Do not combine `--ask-deepy` with `--deepy-server`.

Prompt entry:

- `Enter`: send
- `Alt+Enter` or `Ctrl+J`: insert a newline
- `Ctrl+S`: stop the current turn

Useful commands:

| Command | Purpose |
|---|---|
| `/add <path>` | Add and select an image, video, or audio file. |
| `/image <path>`, `/video <path>`, `/audio <path>` | Add and select a specific media type. |
| `/list [all|image|video|audio]` | List known media. |
| `/select <ref>` | Select media by ID, list number, or part of its name. |
| `/selected` | Show the current selection. |
| `/time <seconds>` | Choose a time in the selected video. |
| `/frame <index>` | Choose a zero-based frame in the selected video. |
| `/size <WxH>`, `/frames <count>`, `/duration <seconds>`, `/seed <value>` | Override generation defaults. |
| `/templates [tool]` | List available templates. |
| `/template <tool> <variant>` | Select a template. |
| `/sessions`, `/resume <ref>`, `/new` | List, resume, or start saved sessions. |
| `/reset` | Clear the temporary conversation or start a new saved session, according to the session mode. |
| `/help` | Show the complete command summary. |
| `/quit` | Exit. |

For example:

```text
/video E:\media\my_clip.mp4
/frame 120
inspect the selected frame and tell me whether the subject is centered
```

## Important configuration choices

The settings under **Configuration > Prompt Enhancer / Deepy** determine Deepy's capabilities, quality, speed, memory use, and access to your files. The recommendations below are starting points; change them when your hardware or workflow has a different priority.

### Prompt Enhancer / Deepy LLM Engine

This selects the language and vision model that understands requests, plans work, and writes responses.

- **Qwen3.5VL Abliterated 4B** starts quickly and uses the least memory, but is less reliable with long instructions and multi-step decisions. **Recommended for:** Deepy Zero on limited hardware and simple, direct requests.
- **Qwen3.5VL Abliterated 9B** understands more complex instructions and media better than 4B, with higher VRAM and RAM use. **Recommended for:** the best general Deepy Zero experience when it fits comfortably.
- **Qwen3.8VL Uncensored 27B** offers the strongest local planning and is required for local Deepy Prime, but needs considerably more memory and takes longer to load. **Recommended for:** local Prime and complex multimedia projects.
- **A remote LLM** avoids loading the language model on the WanGP GPU and may provide stronger reasoning, but adds network latency and sends conversation content to the configured provider. Remote engines require Prime. **Recommended for:** Prime when local memory is insufficient or a supported remote engine is preferred. Review [Remote LLMs](REMOTE_LLMS.md) before using one with private media or instructions.

Changing the engine can require new model downloads and a runtime reload.

### Qwen LLM Quantization

Quantization trades model quality for lower VRAM/RAM use. It is shown only for local Qwen engines.

For Qwen3.5:

- **Quanto Int8** uses more memory but generally preserves better quality. **Recommended for:** most systems that can run it comfortably.
- **GGUF Q4** uses less memory and can be faster when compatible kernels are installed, with some quality loss. **Recommended for:** systems where Int8 does not leave enough memory for generation models.

For Qwen3.8 27B:

- **GGUF Q4** has the highest quality and memory use. **Recommended for:** quality-first work when it fits without forcing excessive unloading. Usually this will require a 24 GB VRAM GPU.
- **GGUF IQ3_S** is the middle ground for quality and memory. **Recommended for:** most local Prime installations. You may run this version with a 16 GB VRAM GPU.
- **GGUF Q2** has the lowest memory use and the largest quality loss. **Recommended for:** this version can be also used with a 16 GB VRAM GPU but the advantage over Q3 is that you will be able to expand the context window.

If Deepy frequently unloads other models, runs out of memory, or leaves too little VRAM for media generation, select a smaller model or lower quantization before reducing the context window drastically.

### Speculative Decoding (MTP)

Speculative decoding can generate Deepy's text faster by predicting several tokens ahead. More draft tokens can improve speed for some requests but use more VRAM, and the fastest setting varies by model and workload.

- **Auto** enables the feature only on supported models when WanGP detects enough VRAM: normally at least 12 GB for Qwen3.5 9B and 24 GB for Qwen3.8 27B. **Recommended for:** nearly everyone.
- **Disabled** saves the extra VRAM and avoids spending memory on acceleration. **Recommended for:** tight-memory systems or when Auto prevents a generation model from fitting.
- **Enabled with 2, 3, or 4 draft tokens** lets you tune for speed manually. Higher is not always faster. **Recommended for:** users willing to benchmark repeated, representative prompts; start with 2.

This setting changes response speed, not the quality or speed of image, video, or audio generation.

### Prompt Enhancer usage and sampling

These controls affect the shared local language model's output style. They are normally best left at their defaults.

- **Prompt Enhancer Usage** chooses whether normal WanGP generations enhance prompts automatically or only when you click the enhancer button. Automatic enhancement can add detail but can also reinterpret carefully written prompts. **Recommendation:** use **On-Demand Button Only** when prompt fidelity matters; use **Automatic on Generation** when you routinely start from short ideas.
- **Sampling Temperature** controls creativity. Lower values are more consistent and literal; higher values are more varied but more likely to wander. **Recommendation:** keep the default `0.6`; try `0.3-0.5` for precise instructions or `0.7-0.9` for ideation.
- **Sampling Top-p** controls how broad the model's word choices can be. Lower values narrow responses; higher values add variety. **Recommendation:** keep the default `0.9` and adjust temperature first.
- **Randomize Prompt Enhancer Seed** allows different wording and ideas on repeated requests. Disabling it improves repeatability when the prompt and settings are unchanged. **Recommendation:** keep it enabled for creative work; disable it when comparing configuration changes.

### Deepy

This chooses the assistant level:

- **Disabled** removes the conversational assistant while leaving ordinary WanGP generation available.
- **Deepy Zero** prioritizes speed and direct execution with the selected templates.
- **Deepy Prime** adds planning, model discovery, durable workspace files, and optional external services for multi-step work.

**Recommendation:** use Zero for isolated operations and Prime when the request requires multiple dependent steps, model comparison, project files, or external tools. See [Choose Deepy Zero or Deepy Prime](#choose-deepy-zero-or-deepy-prime) for engine requirements.

### Deepy VRAM Loading Mode

This controls how long a local Deepy model stays in GPU memory. It is not shown for a remote engine.

- **Unload from VRAM as soon as possible** frees memory after Deepy becomes idle, but the next request must reload the model and starts more slowly. **Recommended for:** GPUs that need nearly all available VRAM for image or video generation; this is the safest default.
- **Unload if VRAM is requested by another WanGP component** keeps Deepy responsive between requests but gives its memory back when another operation needs it. **Recommended for:** most systems with enough VRAM to hold Deepy during normal browsing and light generation.
- **Always loaded in VRAM** gives the fastest Deepy response times but permanently reduces the memory available to generation models. **Recommended for:** high-VRAM systems primarily used for Deepy, or remote/heavily offloaded generation workflows.

If generations fail to fit after changing this setting, move one level toward earlier unloading.

### Context Window Tokens

The context window is how much recent conversation, tool output, and project state the local model can consider at once. A larger value helps Deepy follow long projects and retain more detail before summarization, but its KV cache consumes more VRAM and long context can take longer to prepare.

- **8K-16K** is suitable for short Zero conversations and uses the least memory.
- **32K** is the practical minimum for Prime with **Summarize**.
- **48K or more** supports **Summarize with Thinking** and longer projects, with increasing memory cost.
- **Very large windows** are useful only when the extra history materially improves the workflow and the displayed KV-cache estimate fits comfortably alongside your generation models.

**Recommendation:** start around 16K for Zero, 32K for Prime, or 48K for Prime when using compaction thinking. Increase it only when Deepy summarizes too often or loses relevant recent detail.

### KV Cache Quantization

The KV cache holds the active conversation context in GPU memory. Quantizing it reduces that memory cost without changing the model checkpoint.

- **Auto** uses fast INT8 caching when compatible GGUF kernels are installed and otherwise uses BF16. **Recommended for:** most users.
- **Disabled (BF16)** uses more VRAM and avoids cache quantization. **Recommended for:** quality-sensitive troubleshooting, or systems where INT8 cache performance is worse.
- **INT8** uses about half the KV-cache VRAM and makes larger context windows practical. **Recommended for:** long Prime sessions or tight VRAM, provided the installed kernels support it efficiently.

The context-window label displays an estimated cache size. Treat that estimate as VRAM reserved before image or video model requirements.

### Compaction Type When Context is Full

Compaction decides what happens when the live conversation no longer fits in the context window.

- **Discard Oldest Entries** simply removes the oldest context. It is fast but may lose goals, decisions, references, and unfinished plans. **Recommended for:** short, disposable Zero chats only.
- **Summarize** condenses older work while preserving important goals, decisions, completed actions, file references, and next steps. It requires at least 32K context. **Recommended for:** most long sessions and all local Prime use.
- **Summarize with Thinking** lets Deepy reason before creating the summary, improving preservation in complicated projects at the cost of more time and a minimum 48K context. **Recommended for:** long Prime workflows with many dependencies, not routine one-step requests.

Summaries help but cannot guarantee perfect recall. Keep critical specifications in your request or in a workspace document when mistakes would be costly.

### Repetition Penalty

When enabled, this reduces rambling and repeated phrases across Deepy's responses and planning. It costs roughly 10% of local text-generation speed.

- **Enabled** produces cleaner long responses and plans. **Recommended for:** normal use, especially Prime.
- **Disabled** is slightly faster but can become repetitive. **Recommended for:** maximum-speed experiments or troubleshooting only.

### Filesystem access

Filesystem access determines which local files Deepy can inspect or create. Use the smallest scope that supports the task.

For Zero:

- **Disabled** prevents filesystem tools. **Recommended for:** generation-only use.
- **Read Outputs + Selected Folders** lets Deepy inspect existing files without changing them. **Recommended for:** analysis, lookup, and media-selection workflows.
- **Read / Write Outputs + Selected Folders** allows file creation and changes in the configured locations. **Recommended for:** tasks that explicitly require file editing or organization.

For Prime, its session workspace always remains available for drafts and project files. The outside-workspace choices are:

- **Outputs only (read)** keeps external access narrow. **Recommended for:** most users and projects that can keep working files in the session workspace.
- **Read outputs + selected folders** adds read-only access to chosen source folders. **Recommended for:** projects that consume an existing media or document library.
- **Read + create in outputs; read/write selected folders** allows new output files and full work in selected folders. Existing files under WanGP output folders remain protected from overwrite, rename, move, and deletion. **Recommended for:** only projects that need Deepy to maintain files outside its workspace.

### Additional Filesystem Folders

Add one folder per line to make specific project locations available. An optional alias gives Deepy a short, unambiguous name for the folder:

```text
"D:\My Media" media
E:\Projects\Current project
```

**Impact:** broader folders expose more filenames and content and, with write access, allow more changes. **Recommendation:** add the narrowest project folder possible instead of a drive root, give it a clear alias, and remove it when the project is finished.

### Read Everywhere

This allows Deepy to read absolute paths anywhere the WanGP process can access. It does not expand write access.

**Impact:** convenient references to arbitrary local files become possible, but Deepy can also read unrelated or sensitive files if asked. **Recommendation:** leave it disabled and use selected folders. Enable it temporarily only when read-only work genuinely spans many locations.

### Deepy Zero Prompt and Deepy Prime Guidance

These fields provide standing instructions applied to future interactions:

- **Deepy Zero Custom System Prompt** is useful for a preferred response style or repeated direct-task rule.
- **Deepy Prime User Guidance** is useful for durable project preferences such as quality priorities, approval points, naming conventions, or how to choose models.

**Impact:** broad or conflicting instructions affect every request and can make otherwise simple tasks less predictable. **Recommendation:** keep guidance short, specific, and reusable. Put one-off requirements in the request itself. Do not paste secrets into these fields, especially with a remote engine.

### External MCP Servers

External MCP servers give Prime access to additional applications, data, or specialized tools.

**Impact:** they can make cross-application workflows possible, but may send prompts or files to another process or service and may perform actions under that service's permissions. A broken server can also delay or prevent its tools from being used. **Recommendation:** configure only trusted services needed for a real workflow, review their permissions and privacy terms, and remove unused entries.

### Allow Searching for Changed MCP Executable Paths

Some locally installed MCP programs place each version in a different folder. When this option is enabled and the saved executable disappears, Deepy may look in sibling version folders for the newest executable with the same filename.

**Impact:** local integrations survive routine upgrades more easily, but WanGP may run a newer version than the one originally configured. **Recommendation:** leave it disabled for tightly controlled environments; enable it for trusted local tools that update frequently and whose version-folder layout is stable.

Use Prime's session workspace for plans, drafts, and other files that belong to a long project. It offers the safest default place for durable working material and remains useful after older conversation content is summarized.

## Example requests

Focused requests for Zero or Prime:

```text
Generate a cinematic image of a robot violinist on a rainy Paris rooftop at night.
```

```text
Edit the selected image so the background becomes a neon alley while keeping the character identity. Use 8 inference steps.
```

```text
Transcribe audio track 2 from the selected video with word timestamps.
```

```text
Use the selected portrait and the last audio clip to make a talking video.
```

Multi-step requests suited to Prime:

```text
Generate a master image of a robot dancing beside a horse in a nightclub. Create a second image with the same subjects and setting but a different pose. Check identity and composition, revise if needed, then generate a transition between the two images.
```

```text
Create a portrait for an introduction video, generate a short speech explaining WanGP's capabilities, then combine the portrait and speech into a talking video.
```

## Practical tips

- State the final goal, the source media to reuse, and any details that must not change.
- Use Zero for a direct task and Prime for planning, model comparison, validation, or a chain of dependent actions.
- Ask Deepy which template, defaults, or LoRAs are active when you want to confirm the setup.
- Put recurring model-specific choices in a linked template instead of repeating them in every prompt.
- Specify a time, frame, or audio track when the source contains several possible references.
- Ask for word timestamps when segment timestamps are not precise enough.
- Use **Pause** to temporarily free local resources without abandoning the turn.
- Use saved sessions and copied gallery media when a project must remain portable after source files move.
- You can ask Deepy WanGP-specific questions instead of searching the manuals yourself.

---

> Applies to: Choosing, enabling, launching, configuring, and using Deepy Zero or Deepy Prime in Gradio, CLI, and the standalone Web app, including media, templates, sessions, workspaces, phone access, voice input, and network protection.
