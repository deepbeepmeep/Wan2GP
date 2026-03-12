# Prompts Guide

This page explains how WanGP interprets the main text prompt: how multiline prompts are split or preserved, how prompt lines can be paired with multiple images, how the Prompt Enhancer changes the text, and how macros generate prompt variations.

## Prompt Types In Practice

In WanGP, the main prompt is always the text you write in the `prompt` box.

Depending on the model, that text can be used alone or together with:

- `Start Image`, `End Image`, or `Reference Images`
- `Control Video` or `Source Video`
- `Audio Source`, `Audio Prompt (Soundtrack)`, or `Speaker reference voice`

Even when a model also uses image, audio, or video conditioning, the text prompt still tells the model what you want to happen.

The same text box is used across very different kinds of models, so the best prompt style depends on the model you chose.

### Text-only models

Examples:

- `Wan2.1 Text2video 14B`
- `Qwen Image 20B`
- `TTS HeartMuLa OSS 3B`

Practical use:

- For `Wan2.1 Text2video 14B`, the prompt is your whole scene description.
- For `Qwen Image 20B`, the prompt can also describe posters, signs, or images with a lot of visible text, because this model is especially good at rendering long text inside images.
- For `TTS HeartMuLa OSS 3B`, the prompt is usually lyrics, while the extra tags field steers the musical style.

### Text + image models

Examples:

- `Wan2.1 Image2video 480p 14B`
- `Wan2.2 Image2video Enhanced Lightning v2 14B`

Practical use:

- The image gives the model the subject, appearance, and composition.
- The text prompt should focus on motion, action, camera movement, and scene evolution.

This is why image-to-video prompts are often more useful when they say things like:

```text
the woman smiles, turns toward the window, and the camera slowly pushes in
```

instead of re-describing every detail that is already visible in the image.

### Text + audio models

Examples:

- `TTS Chatterbox Multilingual`
- `TTS Index TTS 2`
- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`

Practical use:

- The text is the script or speech content.
- The audio input usually gives the voice identity, emotion, or speaking style.

For dialogue-capable models such as `TTS Index TTS 2`, `TTS Qwen3 Base (12Hz) 1.7B`, and `TTS KugelAudio 0 Open 7B`, the prompt can contain speaker tags such as `Speaker 1:` and `Speaker 2:`.

### Text + image + audio models

Examples:

- `Hunyuan Video Custom Audio 720p 13B`
- `Hunyuan Video Avatar 720p 13B`

Practical use:

- `Hunyuan Video Custom Audio 720p 13B` is useful when you want a person from a reference image to speak or sing using a recorded voice or song. The text prompt is still useful to describe the scene, mood, framing, or action.
- `Hunyuan Video Avatar 720p 13B` is useful when the audio drives the animation, but you still want the prompt to guide the result, for example by asking for a close-up, a realistic setting, or a particular emotional tone.

### Text + video models

Examples:

- `Hunyuan Video 1.5 Upsampler 720p 8B`
- control-video workflows in Wan and VACE models

Practical use:

- The control or source video gives the timing and structure.
- The text prompt tells the model what style, subject, or scene direction you want on top of that structure.

### Instruction-style editing models

Some models do not want a purely descriptive prompt.

They expect the prompt to be an instruction applied to an existing image or video.

Practical examples in WanGP:

- `Qwen Image Edit 20B`
- `Qwen Image Edit Plus (2509) 20B`
- `Qwen Image Edit Plus (2511) 20B`
- `Flux 1 Dev Kontext 12B`
- `Wan2.1 Chrono Edit 14B`
- `Ditto 14B`

For those models, the input image or video already gives the starting point.

So the prompt should say what to change, not just describe a final scene.

Good action verbs are:

- `add`
- `remove`
- `replace`
- `change`
- `turn`
- `rotate`
- `recolor`
- `relight`

Weak prompt for an edit model:

```text
a woman with a red hat in a rainy street
```

Better:

```text
Add a red wool hat to the woman, keep her face, hairstyle, and the rainy street unchanged.
```

More practical examples:

```text
Remove the people in the background and keep the main subject untouched.
Change the car color to matte black and keep the camera angle the same.
Replace the cloudy sky with a sunset sky, but keep the buildings unchanged.
Rotate the pose of the woman so that she is facing the right.
Render the subjects as classical sculptures carved from single blocks of pristine white marble.
```

This is especially important for `Qwen Image Edit, `Flux Kontext`, `Chrono Edit`,  `Ditto`, ...


For video edit models such as `Ditto 14B`, write instructions that apply to the whole clip or the whole frames, for example:

```text
Turn the whole video into a black-and-white film look while keeping the original motion.
Replace the material of all visible statues with polished gold.
```

For `Wan2.1 Chrono Edit 14B`, it is usually best to enable the Prompt Enhancer, because that model is stricter than Qwen Edit or Flux Kontext about prompt format.

## Text Prompt Basics

WanGP reads the prompt line by line before generation.

- A line starting with `#` is a comment.
- A line starting with `!` is a macro line (see Macro section below).
- Other lines are prompt content.

Empty lines are usually ignored, but some speech models keep them because they are useful as manual split markers for long speeches or dialogue.

This is especially practical with:

- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`
- `TTS Index TTS 2`

Those models are often used for long speeches or dialogue, so keeping a multiline script together is usually more useful than treating each line as a separate generation.

## How To Process Each Line Of The Text Prompt

The dropdown `How to Process each Line of the Text Prompt` changes how WanGP interprets multiline prompts.

The UI shows these choices:

- `Each New Line Will Add a new Video/Image/Audio Request to the Generation Queue`
- `Each Line Will be used for a new Sliding Window of the same Video Generation`
- `All the Lines are Part of the Same Prompt`

Which wording you see depends on the current model and whether it outputs video, image, or audio.

### 1. Each New Line Adds A New Queue Item

This is the most practical choice when each line is meant to be a separate idea.

Example:

```text
a fox running through snow
a whale breaching at sunset
a drone shot of a neon city in the rain
```

This creates 3 separate jobs.

Use this when:

- you want to batch several unrelated ideas
- you want quick A/B tests of prompt variants
- you want one `Start Image` reused with several motion prompts in an image-to-video model

Good examples:

- `Wan2.1 Text2video 14B`: try several scene ideas at once
- `Qwen Image 20B`: try several poster or ad concepts at once
- `Wan2.1 Image2video 480p 14B`: try several motion ideas from the same image

### 2. Each Line Is Used For A New Sliding Window Of The Same Video

This is useful for long-video workflows.

Instead of treating each line as a separate job, WanGP uses:

- line 1 for the first window
- line 2 for the second window
- line 3 for the third window
- and if the video needs more windows than lines, the last line is reused

Example:

```text
wide establishing shot of the city at dawn
the camera moves into the crowd and follows the singer
close shot of the singer as the chorus starts
```

This is practical when you want one long video with changing story beats.

It is especially useful for models meant for long or multi-window video generation, such as `Infinitetalk`, which is described as supporting very long videos with smooth transitions between shots.

Use this when:

- you want one line per shot or story beat
- you want a long talking or performance video to evolve over time
- you want a rough storyboard without writing one huge paragraph

Important practical limitation:

- only one `Start Image` is supported in this mode

### 3. All The Lines Are Part Of The Same Prompt

This is the right choice when line breaks are part of the prompt format itself.

Use this when the prompt is one structured block, such as:

- a second-by-second timeline
- a dialogue script
- song lyrics with sections like `[Verse]` and `[Chorus]`

Example: timeline prompt

```text
(at 0 seconds: the man stands in front of the fridge, camera static)
(at 1 second: he opens the fridge and reaches for a can)
(at 2 seconds: the camera shifts to a side angle as he drinks)
```

This is especially useful for `Wan2.2 Image2video Enhanced Lightning v2 14B`, because its own default prompt uses that kind of timeline format.

Example: dialogue prompt

```text
Speaker 1: We should leave before the rain gets heavier.
Speaker 2: Give me one minute, I still need my jacket.
Speaker 1: One minute, then we run for the bus.
```

This is the practical choice for:

- `TTS Index TTS 2`
- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`

Those models are used for speech and dialogue, so splitting each line into a separate job would usually destroy the conversation structure.

Example: lyrics prompt

```text
[Verse]
Morning light through the window pane
I hum a tune to chase the rain
[Chorus]
Stay with me through every mile
Hold this fire, hold this smile
```

This is the practical choice for:

- `TTS HeartMuLa OSS 3B`
- `TTS ACE-Step v1.5 Turbo 2B`

Those music models work best when the lyrics stay together as one structured prompt.

## Multiple Images As Text Prompts

The dropdown `Multiple Images as Texts Prompts` matters when you use several images together with several prompt lines.

This is mainly useful for image-to-video models such as:

- `Wan2.1 Image2video 480p 14B`
- `Wan2.2 Image2video Enhanced Lightning v2 14B`

The two choices are:

- `Generate every combination of images and texts`
- `Match images and text prompts`

In both modes, WanGP uses the image prompt field currently driving the job:

- if you provided `Start Image`, those images are used
- otherwise WanGP uses `End Image`
- if both are present, `Start Image` is the one paired with the text prompts

So in normal image-to-video practice, this usually means the `Start Image` field. It is not about `Reference Images`.

### Generate Every Combination Of Images And Texts

This is the exploration mode.

WanGP takes every prompt line and combines it with every image from the active image prompt field in use, usually `Start Image`.

If you have:

- 3 images
- 2 prompt lines

WanGP will create 6 jobs.

Use this when:

- you want to see which motion prompt works best on which image
- you are testing several portraits against several actions
- you are exploring, not pairing

Practical example:

- 3 portraits
- prompt 1: `the subject smiles and leans toward the camera`
- prompt 2: `the subject turns away and walks toward the window`

This is useful if you want to compare all results.

### Match Images And Text Prompts

This is the pairing mode.

Use it when image 1 should go with prompt 1, image 2 with prompt 2, and so on.

This is usually the better choice when:

- each image already has its own intended motion
- you are preparing a batch of separate image-to-video jobs
- you do not want the combinatorial explosion of testing everything against everything

Practical example:

- portrait 1 + `she smiles and waves`
- portrait 2 + `he turns left and starts walking`
- portrait 3 + `the child looks up and starts laughing`

If you used `Generate every combination of images and texts`, you would get many unwanted cross-combinations. `Match images and text prompts` keeps the batch clean.

## Prompt Enhancer

Prompt enhancement is easiest to think of as a writing assistant built into WanGP.

WanGP has two levels for it:

- a global Prompt Enhancer setup in `Configuration`
- a per-generation Prompt Enhancer control next to the main prompt box

If no Prompt Enhancer is enabled in `Configuration`, the Prompt Enhancer row does not appear in the generation UI.

### Where It Is Enabled

First enable a Prompt Enhancer in `Configuration`.

That global configuration decides:

- which Prompt Enhancer family WanGP loads
- whether it works automatically during generation or on demand
- for Qwen3.5-based enhancers, which quantized backend is used
- sampling behavior such as `temperature`, `top_p`, and random seed randomization

Once that is enabled, the generation screen shows the Prompt Enhancer row near the main prompt field.

Depending on your configuration, it either:

- runs automatically during generation
- or appears as a button you click manually before generation

### Automatic Versus On-Demand

Automatic mode:

- Pros: fastest workflow, no extra click, good when you always want prompt rewriting
- Pros: useful for models where the enhancer is almost part of the workflow, such as `Wan2.1 Chrono Edit 14B`
- Cons: you do not review the rewritten prompt before generation
- Cons: if the enhancer over-interprets your idea, you only notice it after the run starts or in the saved metadata

On-demand mode:

- Pros: you can inspect and edit the rewritten prompt before generating
- Pros: best for strict prompt formats, edits, lyrics, or dialogue where you want manual control
- Cons: one more step in the workflow
- Cons: with multiple `Start Image`s, it only works cleanly when `Multiple Images as Texts Prompts` is set to `Match images and text prompts`, and the number of images matches the number of prompt lines

### Prompt Enhancer Families

WanGP currently supports several Prompt Enhancer backends that you can choose in the `Config / Extensions` tab:

- `Llama 3.2`
- `Llama Joy`
- `Qwen3.5-4B Abliterated`
- `Qwen3.5-9B Abliterated`

Qwen 3.5 (especially the 9B / quanto int8 variant) should be the smartest Prompt Enhancer. In addition it supports the `Think` mode which will force the Prompt Enhancer to spend more time on your request (Think mode will require more VRAM too).

### Qwen3.5 Backend Choices

For Qwen3.5-based enhancers, WanGP can use different backends.

| Backend | Best for | Pros | Cons |
| --- | --- | --- | --- |
| `Int8` | The default Qwen setup | Simplest choice, best quality | Heavier on the VRAM |
| `GGUF` | Lower RAM/disk usage, especially if you already use GGUF tooling | Can be very fast with GGUF CUDA kernels, especially on Windows | Lower Prompt Compliance |

Qwen3.5 can be greatly accelerated if `Config / Performance / Language Models Decoder Engine` option is set to *cg* or *vllm*.

### The Main Prompt Enhancer Choices

The user-facing dropdown usually offers:

- `Disabled`
- `Based on Text Prompt Content`
- `Based on Images Prompts Content (such as Start Image and Reference Images)`
- `Based on both Text Prompt and Images Prompts Content`

But not every model shows exactly that set.

Some models expose model-specific options instead.

Practical examples:

- `TTS Index TTS 2`, `TTS Qwen3 Base (12Hz) 1.7B`, and `TTS KugelAudio 0 Open 7B` can show:
  - `A Speech based on current Prompt`
  - `A Dialogue between two People based on current Prompt`
- `Wan2.1 Chrono Edit 14B` restricts the Prompt Enhancer to the combined text+image mode, because that model expects both the edit instruction and the source image

So the safest rule is:

- use the generic `T`, `I`, and `TI` logic when those are the options you see
- prefer the model-specific labels when WanGP exposes them, because they were defined for that model on purpose

### Based On Text Prompt Content

This is the best choice when your text idea is short, rough, or under-detailed.

Practical use:

- Turn a simple video idea into a more cinematic paragraph
- Turn a rough speech topic into a proper spoken script
- Turn a basic song idea into structured lyrics

Good examples:

- `Wan2.1 Text2video 14B`: useful if you type a short scene idea and want richer cinematic detail
- `Qwen Image 20B`: useful if you want a more polished image prompt
- `TTS Chatterbox Multilingual`: useful if you want the app to draft the speech before generating the voice

### Based On Images Prompts Content

This is most useful when the image matters more than the text.

Practical use:

- Start from a portrait or reference image
- let the enhancer read the image
- then have it write a motion prompt that fits what is actually visible

This is often useful for image-to-video workflows where the face, clothing, pose, or framing should stay faithful to the source image.


### Based On Both Text Prompt And Images Prompts Content

This is the most useful choice when both the instruction and the image are important.

The clearest case is `Qwen Edit 14B`.

- the image tells the model what exists now
- the text tells it what you want changed
- the prompt enhancer rewrites that into the format the model expects

Pros:

- the best choice for edit models and instruction-following workflows
- balances image faithfulness with explicit user intent

### Model-Specific Writing Buttons

Common button labels are `Enhance Prompt`, `Write`, `Write Speech`, and `Compose Lyrics`.

Some models expose more specific prompt-writing workflows.

#### `Write`

Used on speech models such as:

- `TTS Index TTS 2`
- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`

Practical use:

- write a monologue from a short topic
- write a two-person dialogue from a rough idea

This is especially useful because those models are often used for spoken content, and two of them explicitly support `Speaker 1:` / `Speaker 2:` dialogue workflows.

#### `Write Speech`

Used on `TTS Chatterbox Multilingual`.

Practical use:

- quickly turn a topic into a spoken script before generating voice audio

This is helpful because Chatterbox is meant for multilingual speech generation, so many users start with an idea, not a finished script.

#### `Compose Lyrics`

Used on music models such as:

- `TTS HeartMuLa OSS 3B`
- `TTS ACE-Step`

Practical use:

- turn a short idea such as `a dreamy synth-pop song about missing someone`
- into actual lyrics with sections like `[Verse]` and `[Chorus]`

This is especially useful because those models are conditioned by lyrics, and in HeartMuLa's case also by style tags.

## `@` And `@@` Syntax

These two syntaxes only matter when the Prompt Enhancer is being used.

If the Prompt Enhancer is disabled, WanGP treats them as normal characters.

### `@` adds extra instructions to the enhancer

Format:

```text
your prompt @ extra instructions
```

Example:

```text
a serious speech about AI safety @ keep it under 6 sentences, natural spoken English
```

Practical use:

- keep a script short
- ask for a certain tone
- ask for a certain output shape without replacing the model's tuned enhancer instructions

This is the safer and more practical everyday option.

### `@@` replaces the enhancer instructions completely

Format:

```text
your prompt @@ full replacement instructions
```

Example:

```text
a woman opens a door @@ Output exactly 6 lines. Each line must start with "(at X seconds:" and describe only visible motion.
```

Practical use:

- force a very specific output format
- completely override the built-in enhancer behavior

Use this only when you know exactly what format you want. It is powerful, but easier to misuse than `@`.

## Think Mode

On Qwen3.5-based prompt enhancers, WanGP can show a `Think` checkbox.

Practical use:

- enable it when the prompt is vague, messy, or structurally difficult
- disable it when you want the fastest rewrite

It is most useful for tasks like:

- turning a rough story idea into a clean second-by-second prompt
- turning a topic into a believable two-person dialogue
- rewriting a difficult edit instruction so it stays aligned with the input image

The thinking text itself is not meant to become part of the final enhanced prompt.

## Macro System

Create multiple prompts from templates using macros. This allows you to generate variations of a sentence by defining lists of values for different variables.

Macros are expanded before WanGP applies the line-processing option from `How to Process each Line of the Text Prompt`. In practice, this means:

- macros first generate the prompt lines
- then WanGP decides whether those lines become separate jobs, sliding-window steps, or one single multiline prompt

### Syntax Rule

Define your variables on a single line starting with `!`. Each complete variable definition, including its name and values, must be separated by a colon (`:`).

### Format

```text
! {Variable1}="valueA","valueB" : {Variable2}="valueC","valueD"
This is a template using {Variable1} and {Variable2}.
```

### How It Works

- Every `{Variable}` in the following prompt lines is replaced with one value from its list.
- WanGP cycles through the values line by line.
- If two variables do not have the same number of values, the shorter one is reused cyclically.

This makes macros practical for:

- batch-testing several scene ideas
- generating multiple poster captions for image models
- creating several prompt variations from one reusable template

### Example

The following macro will generate three distinct prompts by cycling through the values for each variable.

**Macro Definition:**

```text
! {Subject}="cat","woman","man" : {Location}="forest","lake","city" : {Possessive}="its","her","his"
In the video, a {Subject} is presented. The {Subject} is in a {Location} and looks at {Possessive} watch.
```

**Generated Output:**

```text
In the video, a cat is presented. The cat is in a forest and looks at its watch.
In the video, a woman is presented. The woman is in a lake and looks at her watch.
In the video, a man is presented. The man is in a city and looks at his watch.
```

### Practical Macro Example

If you want to test several ad concepts in an image model, you can do this:

```text
! {Product}="watch","perfume","sneakers" : {Mood}="luxury","minimal","energetic"
Studio advertising photo of a {Product}, {Mood} style, clean background, premium lighting, centered composition
```

If your line-processing mode is set to add each new line as a new request, WanGP will queue one image job per generated prompt.

## Troubleshooting

- If dialogue lines are turning into separate jobs or a timeline prompt is split across several jobs, choose `All the Lines are Part of the Same Prompt`.
- If you want a long video with changing story beats, choose `Each Line Will be used for a new Sliding Window of the same Video Generation`.
- If you want several separate prompt ideas queued at once, choose `Each New Line Will Add a new ... Request to the Generation Queue`.
- If `Match images and text prompts` fails, the number of images and prompts must match cleanly.
- If `@` or `@@` seems ignored, the Prompt Enhancer is probably disabled or not used for that run.
