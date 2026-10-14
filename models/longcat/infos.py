# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import VIDEO_PROMPT, VIDEO_PROMPT_SHORT, help_text, specialities


def longcat_help(architecture):
    avatar = architecture != "longcat_video"
    info = "Generate from text, animate a start image, or continue a source video. **Sliding Window** builds a longer video in successive sections. Keep subject and scene descriptions consistent, and inspect joins before extending further. **Auto** sampling selects the continuation behavior when needed."
    compact = "Text/image-to-video and source-video continuation. Sliding windows extend output; inspect joins and drift. Auto sampling chooses continuation behavior."
    if avatar:
        info = "Animate a character using **Voice to Follow** and an optional **Anchor Reference Image** or start frame. The recording supplies speech and timing; the prompt describes the character, expression, gesture and scene. For two speakers, choose the two-voice mode and assign each voice to its visible person. **Sliding Window** supports longer performances; retain consistent anchor appearance and check transitions."
        compact = "Audio-driven character video; audio supplies words/timing, prompt visual behavior. Optional image anchor/start frame; assign two voices to their speakers. Sliding windows support long performances."
        if architecture == "longcat_avatar_v1_5":
            info += "\n\nAvatar 1.5 uses the eight-step distilled preset. Keep its matching sampler and settings."
            compact += " Avatar 1.5: use its eight-step Distill preset."
    result = help_text("LongCat Avatar" if avatar else "LongCat Video", info, VIDEO_PROMPT, compact, VIDEO_PROMPT_SHORT + (" The driving recording determines dialogue; do not substitute new words in the prompt." if avatar else ""), ["https://github.com/meituan-longcat/LongCat-Video"])
    result["specialities"] = specialities("lip sync") if avatar else []
    return result
