# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import EDIT_PROMPT, EDIT_PROMPT_SHORT, VIDEO_PROMPT, VIDEO_PROMPT_SHORT, help_text, specialities


def wan_help(architecture, definition):
    """Compose Wan's common controls with the selected workflow only."""
    info = "Generate a video from a description. Use the selected model's resolution and sampling defaults as a starting point."
    short = "Generate video from text. Start with this model's defaults."
    prompt, short_prompt = VIDEO_PROMPT, VIDEO_PROMPT_SHORT
    strengths = []
    variant_notes = []
    sources = ["https://github.com/Wan-Video/Wan2.1", "https://github.com/Wan-Video/Wan2.2"]
    title = "Wan Video"
    if definition.get("i2v_class") or architecture == "ti2v_2_2":
        info = "Animate a **Start Image** using the text description. The image establishes appearance and framing; the prompt describes how the scene develops. Use similarly framed start and end images when an end image is selected."
        short = "Start image establishes appearance; prompt describes motion. Keep any end image compatible with the start."
        if architecture == "ti2v_2_2":
            info += " Wan 2.2 TI2V also generates from text alone; an image is optional."
            short += " TI2V also supports text-only generation."
    if architecture == "flf2v_720p" or architecture.startswith("fun_inp"):
        title = "First and Last Frame Animation"
        info += "\n\nChoose **Start Image** and **End Image** to create a transition between two states. Describe a plausible connecting action; large identity or composition differences can produce morphing."
        short += " Start/end images define the transition; describe the connecting action."
        if architecture == "flf2v_720p":
            prompt += "\n\nWan FLF2V was trained mainly with Chinese captions; upstream recommends Chinese prompts."
            short_prompt += " FLF2V: prefer Chinese captions."
        else:
            sources.append("https://github.com/aigc-apps/VideoX-Fun")
    if definition.get("vace_class"):
        title = "VACE Creation and Editing"
        info = "Use a **Control Video**, a mask and/or reference images to guide generation. Choose pose, depth or edge processing for motion/structure transfer; use a raw source and mask to edit a region. Reference images supply subjects or a scene. Select their role before generation."
        short = "VACE: optional control video + mask + references. Choose preprocessing for structure/motion, raw source and mask for a local edit. White mask edits; black preserves. KI: scene first; FI: positioned frames first; I: subjects."
        prompt = EDIT_PROMPT + "\n\nFor pose/depth/edge control, describe the finished scene rather than the control-map colors. For reference-to-video without a source, describe the subjects and their actions."
        short_prompt = EDIT_PROMPT_SHORT + " With pose/depth/edges, describe the finished scene; with references alone, describe subjects and action."
        strengths += ["control video", "video inpainting", "video outpainting", "motion transfer"]
        sources.append("https://github.com/ali-vilab/VACE/blob/main/UserGuide.md")
    if architecture == "vace_ditto_14B":
        title = "Ditto Video Editing"
        info = "Supply the **Control Video** to transform. Choose **Global**, **Global Style** or **Sim 2 Real** to match the task. A mask can restrict the edit. This selection does not expose VACE's reference-image and structure-preprocessing workflows."
        short = "Source control video required. Ditto modes: Global / Global Style / Sim 2 Real; optional mask. No VACE reference-image workflow in this selection."
        prompt = EDIT_PROMPT + "\n\nFor **Sim 2 Real**, describe conversion to real footage, for example: Turn it into the real domain, retaining the same scene and action."
        short_prompt = EDIT_PROMPT_SHORT + " Sim 2 Real: request conversion to the real domain."
        strengths = ["video editing", "style transfer"]
        sources = ["https://github.com/EzioBy/Ditto"]
    if architecture in ("standin", "lynx", "lynx_lite", "alpha_lynx", "vace_standin_14B", "vace_lynx_14B"):
        identity = "Use a clear **face reference** to preserve facial identity while the prompt changes the scene, action or clothing. Identity guidance does not guarantee an exact copy."
        if architecture == "vace_lynx_14B":
            identity += " For VACE + Lynx, place the face last, after the other references."
        info += "\n\n### Identity Preservation\n\n" + identity
        short += " " + identity
        variant_notes.append(identity)
        strengths += ["identity preservation"]
        sources.append("https://github.com/bytedance/lynx" if "lynx" in architecture else "https://github.com/WeChatCV/Stand-In")
    if architecture.startswith("phantom"):
        title = "Phantom Subject References"
        info = "Supply one or more **Reference Images** of the people or objects to appear together in a new video. The prompt supplies their actions and scene. Use clear, distinct subjects and explicitly describe their relationship."
        short = "Reference images supply one/multiple subjects; prompt supplies scene, actions and relationships."
        strengths += ["identity preservation"]
        sources = ["https://github.com/Phantom-video/Phantom"]
    if definition.get("multitalk_class") or architecture == "fantasy":
        title = "Audio-Driven Character Animation"
        audio = "Supply a character image and **driving audio**. The recording supplies speech or singing and lip timing; the prompt controls expression, gesture and setting. Write a visual description, not replacement dialogue. Clean speech and a visible face make synchronization easier."
        info += "\n\n" + audio if architecture.startswith("vace_") else ""
        if not architecture.startswith("vace_"):
            info = audio
        short = short + " Driving audio supplies the spoken words and timing; prompt supplies visual behavior." if architecture.startswith("vace_") else audio
        if definition.get("multitalk_class"):
            info += "\n\nFor two people, select the matching audio mode and assign each voice to its visible speaker. Do not rely on text alone to bind voices to faces."
            short += " Match each voice to its speaker using the audio mode and speaker locations."
        variant_notes.append(audio)
        if architecture == "infinitetalk":
            info += "\n\n**Reference images** can begin successive shots/windows. **Sparse Video to Video** takes an anchor from each window; **Video to Video** transfers source movement according to Denoising Strength. Smooth or Sharp Transitions controls shot changes."
            short += " InfiniteTalk: KI anchors successive windows; RUV takes sparse video anchors; GUV transfers source motion. Choose smooth/sharp transitions."
        strengths += ["lip sync"]
        sources.append("https://github.com/Fantasy-AMAP/fantasy-talking" if architecture == "fantasy" else "https://github.com/MeiGen-AI/InfiniteTalk" if architecture == "infinitetalk" else "https://github.com/MeiGen-AI/MultiTalk")
    if architecture in ("animate", "animate2", "steadydancer", "scail", "scail2_14B", "scail2_1.3B", "mocha"):
        title = "Motion Transfer and Character Animation"
        info = "Combine the character image with a **Control / Driving Video**. The video supplies motion; the image supplies the appearance. Choose a clearly visible person and compatible body framing. Occlusions, fast movement and mismatched proportions can reduce fidelity."
        short = "Character image + driving video: appearance from image, motion from video. Prefer clear people and compatible framing."
        strengths += ["motion transfer"]
        if architecture == "animate":
            info += "\n\n**Animate** builds the scene around the reference character. **Replace** changes the selected person within the source scene; supply the person mask. Optional relighting helps match the source."
            short += " Animate creates a scene around the reference; Replace needs the target-person mask and retains source surroundings. Optional relighting."
            strengths += ["character replacement"]
        elif architecture == "animate2":
            info += "\n\nUse a start/continuation frame or select **Character Reference Image**. Describe the desired result in Text Prompt and describe the source action/background separately in **Driving Video Prompt**."
            short += " Use start/continue frame or Character Reference Image. prompt describes target; alt_prompt describes driving video."
            prompt += "\n\n**Driving Video Prompt example:** A person raises both arms and turns left against a plain wall.\n\n**Text Prompt example:** The illustrated character raises both arms and turns left in a garden."
            short_prompt += " alt_prompt describes the source action/background; prompt describes the target character/scene."
        elif architecture == "steadydancer":
            info += "\n\nUse **Start Image** for the first frame. The control video supplies poses; a mask can select the person whose movement should be followed."
            short += " SteadyDancer preserves Start Image and follows extracted source poses; a mask selects the person."
            sources.append("https://github.com/MCG-NJU/SteadyDancer")
        elif architecture.startswith("scail"):
            info += "\n\nChoose the number of people and use the person-location masks to match them. Describe appearance, clothing, scene and the driving action in detail."
            short += " Match person count and person-location masks. Use a detailed image + action caption."
            if architecture.startswith("scail2"):
                info += " SCAIL-2 also offers **Replace** and optional character references; **Animate Preprocessing** chooses raw video or extracted 3D poses."
                short += " Animate/Replace, optional references, raw-video/3D-pose choice."
                strengths += ["character replacement"]
            sources.append("https://github.com/zai-org/SCAIL-2" if architecture.startswith("scail2") else "https://github.com/zai-org/SCAIL")
        elif architecture == "mocha":
            info = "Replace a person in a **Control Video** using reference images and a mask identifying the target person. In WanGP, supply the character reference followed by a **close-up as the second image**. Preserve consistent identity across those references."
            short = "Character replacement: control video + target-person mask + references; second reference must be a close-up."
            prompt, short_prompt = EDIT_PROMPT, EDIT_PROMPT_SHORT
            strengths += ["character replacement"]
            sources.append("https://github.com/Orange-3DV-Team/MoCha")
    if architecture in ("lucy_edit", "kiwi_edit", "bernini", "bernini_1.3B"):
        title = "Instruction Video Editing"
        info = "Supply a **Control Video** and describe the requested change. Keep the original motion, camera and background explicit when they should remain unchanged. Start with one clear edit before combining several changes."
        short = "Control video + editing instruction. State changes and preservation constraints."
        prompt, short_prompt = EDIT_PROMPT, EDIT_PROMPT_SHORT
        strengths += ["video editing"]
        if architecture == "lucy_edit":
            info += "\n\nLucy works particularly well for clothing and character changes. Added accessories or held objects are usually easier than independent objects; exact color changes can be less reliable."
            prompt += "\n\nUse about **20–30 descriptive words**. Start with **Change** for clothing or color, **Add** for an accessory/object, **Replace** for a subject/object, or **Transform to** for a scene/style change. Describe the desired appearance and context."
            short_prompt += " Lucy: about 20–30 descriptive words; Change = clothing/color, Add = accessory/object, Replace = subject/object, Transform to = scene/style."
            strengths += ["character replacement", "style transfer"]
            sources = ["https://github.com/DecartAI/lucy-edit-comfyui", "https://huggingface.co/decart-ai/Lucy-Edit-Dev"]
        elif architecture == "kiwi_edit":
            variant = definition["kiwi_variant"]
            note = {"instruct_only": "**Instruction Only** uses source video and text; no reference image is needed.", "reference_only": "**Reference Only** requires a reference image along with the source video; describe the target appearance and the role of the reference.", "instruct_reference": "**Instruction + Reference** requires a reference image along with the source video and edit instruction; identify which details to borrow."}[variant]
            info += "\n\n" + note
            short += " " + note
            sources = ["https://github.com/showlab/Kiwi-Edit"]
        else:
            info += "\n\nBernini also accepts reference images for subject/style guidance, with or without a source video. A video mask restricts the edit. **Video Guidance** controls source adherence; **Reference Guidance** controls the image influence."
            short += " Bernini: optional source, mask and ordered references; Video Guidance and Reference Guidance have separate roles."
            if architecture == "bernini_1.3B":
                short += " Prefer simple/local edits; complex human motion and dense compositions are weaker than 14B."
            sources = ["https://github.com/bytedance/Bernini"]
    if architecture == "chrono_edit":
        title = "ChronoEdit Image Transformation"
        info = "Provide a **Start Image** and an edit instruction. **Fast Image Transformation** and **Long Image Transformation** return an edited image; **Temporal Reasoning Video** shows the transformation developing over time. Larger viewpoint or physical changes may need the longer process."
        short = "Start image + edit. Fast/Long Image Transformation produce edited images; Temporal Reasoning Video shows the transformation."
        prompt, short_prompt = EDIT_PROMPT, EDIT_PROMPT_SHORT
        strengths += ["image editing", "image viewpoint change"]
        prompt = "### Describe the Transformation\n\nStart with `The user wants to ...`, then explain the visible change and what should remain. For a substantial transformation, describe the target pose, placement, interaction and surrounding scene in a coherent paragraph.\n\n**Example:** The user wants to turn the view slightly to the left while keeping the bicycle design unchanged. Show the frame and front wheel from the new angle, with consistent proportions, lighting and a natural shadow."
        short_prompt = "Begin 'The user wants to ...'. Describe target pose, placement, interactions and preservation constraints in a coherent paragraph (about 80–100 words for complex edits)."
        sources = ["https://github.com/nv-tlabs/ChronoEdit"]
    if architecture in ("recam_1.3B", "vista4d"):
        title = "Video Viewpoint Change"
        info = "Supply the **Control Video** to reshoot from a different viewpoint. Select the camera movement with the camera controls; the prompt describes the scene and action to retain. Newly revealed areas are generated, so large camera changes can invent unseen content."
        short = "Reshoot control video from a new viewpoint. Camera controls set the movement; prompt describes source scene/action. Unseen regions are generated."
        strengths += ["video viewpoint change", "camera control"]
        if architecture == "recam_1.3B":
            info += "\n\nReCamMaster offers pan, tilt, zoom, translation and arc presets. This selection uses an 81-frame output."
            short += " ReCamMaster: choose pan/tilt/zoom/translation/arc preset; 81 frames."
            sources = ["https://github.com/KwaiVGI/ReCamMaster"]
        else:
            info += "\n\nVista4D offers camera presets or a custom **.npz camera path**. **Camera Strength** scales the move; **Scene Scale** adjusts its relation to the reconstructed scene. **Dynamic Object Keywords** selects moving content (`_all_` selects all pixels). Start with a modest move."
            short += " Vista4D: preset/custom .npz path, Camera Strength, Scene Scale and Dynamic Object Keywords (_all_ for all)."
            sources = ["https://github.com/Eyeline-Labs/Vista4D"]
    if architecture == "wanmove":
        title = "Wan-Move Trajectory Control"
        info = "Supply the **Start Image** and the required **Trajectory File (.npy)** to direct point-level motion. The trajectory sets where tracked points move; the prompt describes the subject, scene and intended action. A prose camera instruction cannot replace the trajectory file."
        short = "Start image + required .npy trajectory. Points control motion; prompt describes appearance and action."
        strengths += ["trajectory control"]
        sources = ["https://github.com/ali-vilab/Wan-Move"]
    if definition.get("alpha_class"):
        info += "\n\n### Transparent Output\n\nGenerate a foreground subject with an **alpha channel**. Describe a clearly separated foreground. Use the RGBA export for compositing; a checkerboard preview displays transparency but is not the transparent deliverable."
        short += " Generates foreground + alpha; use RGBA export for transparency, not the checkerboard preview."
    if definition.get("shotplan"):
        title = "ShotPlan Multi-Shot Video"
        info = "Plan a short sequence with explicit hard cuts. Put a shared description before the shot ranges, then give each shot its own framing and action. Start with two to four shots in the model's default clip length; retain consistent subject descriptions between shots."
        short = "Multi-shot sequence with 1–3 hard cuts; start at 81 frames, 832x480, 16 fps. No sliding windows."
        short_prompt = "Global caption, then contiguous ranges covering the clip: `[0%:50%] Wide view ...` and `[50%:] Close-up ...`. Boundaries request hard cuts. Do not add Shot N labels. Keep the combined prompt concise."
        strengths += ["multi-shot video"]
        sources = ["https://github.com/Pensioner-11/ShotPlan"]
    if architecture == "i2v_2_2_svi2pro":
        info += "\n\n**SVI 2 Pro** extends animation over successive windows. Optional anchor images guide each window. Keep the subject description consistent and inspect transitions for drift before extending further."
        short += " SVI 2 Pro: window continuation with optional per-window anchor images; inspect drift and seams."
        sources.append("https://github.com/vita-epfl/Stable-Video-Infinity/tree/svi_wan22")
    # Read actual output-frame constraints, so 5B, VACE and derivative choices stay consistent.
    timing = f"At the default {definition['fps']} fps, frame counts start at {definition['frames_minimum']} and increase in steps of {definition['frames_steps']}. Duration and frame count refer to the output video."
    info += "\n\n### Duration and Continuity\n\n" + timing
    if definition.get("sliding_window"):
        info += " Longer videos can use **Sliding Window** and continuation. Overlap joins successive windows; inspect joins and identity drift. Changing the prompt between windows is not an exact shot-timing guarantee."
        short += " Sliding windows/continuation support longer output; inspect joins and drift."
    if not definition.get("returns_audio"):
        info += " This model does not generate a soundtrack."
        short += " No generated soundtrack."
    result = help_text(title, info, prompt, short, short_prompt, sources)
    if architecture in ("vace_standin_14B", "vace_lynx_14B", "vace_multitalk_14B"):
        result["infos"] = definition["infos"] + "\n\n### Additional Inputs for This Variant\n\n" + "\n\n".join(variant_notes)
    elif architecture == "bernini_1.3B":
        result["infos"] += "\n\n" + definition["infos"]
    result["specialities"] = specialities(*dict.fromkeys(strengths))
    return result
