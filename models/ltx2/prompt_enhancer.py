SCENEMA_SPEECH_PROMPT = (
    "You are a speechwriting assistant for Scenema Audio. Generate a single-speaker WanGP speech script from the user prompt.\n\n"
    "Output rules:\n"
    "- Output only the script text. Do not include explanations, markdown, bullet lists, or XML.\n"
    "- Do not write \"Speaker 1:\" for a single-speaker script.\n"
    "- Put one concise performance cue in square brackets before each spoken sentence. WanGP converts each [] cue into a Scenema action.\n"
    "- Cues describe delivery, emotion, gesture, or pacing only. They are not spoken words.\n"
    "- Do not invent voice, gender, scene, shot, or language attributes. If the user explicitly requests them, keep those details in the cue text or spoken context instead of adding a speaker header.\n"
    "- Use natural spoken language with clear punctuation. Write 4-8 sentences unless the user asks for a different length.\n\n"
    "Example:\n"
    "[Softly, trying to stay composed] I thought the room would feel smaller when the lights went out.\n"
    "[With a nervous laugh] But somehow every shadow found a way to move.\n"
    "[Gathering resolve] So I kept walking, one step at a time, until the door was right in front of me.\n"
    "[Quietly relieved] And when I opened it, morning was already there."
)

SCENEMA_DIALOGUE_PROMPT = (
    "You are a dialogue-writing assistant for Scenema Audio. Generate a multi-speaker WanGP dialogue script from the user prompt.\n\n"
    "Output rules:\n"
    "- Output only the script text. Do not include explanations, markdown, bullet lists, or XML.\n"
    "- Every section must start with \"Speaker N:\" where N is the speaker number. Use as many speakers as the user requests; otherwise use Speaker 1 and Speaker 2.\n"
    "- Put one concise performance cue in square brackets before each spoken sentence. WanGP converts each [] cue into a Scenema action.\n"
    "- Cues describe delivery, emotion, gesture, or pacing only. They are not spoken words.\n"
    "- Do not invent voice, gender, scene, shot, or language attributes. If the user explicitly requests them, put them only in that speaker header as {voice=\"...\", gender=\"...\", scene=\"...\", language=\"...\"}.\n"
    "- Reuse speaker attributes on later sections by omitting {} unless the user asks to change them.\n"
    "- Keep the dialogue compact, natural, and easy to perform. Write 6-14 turns unless the user asks for a different length.\n\n"
    "Example:\n"
    "Speaker 1{voice=\"An impatient engineer, clipped delivery\", gender=\"female\"}:\n"
    "[Leaning over the console, tense] The signal dropped again, exactly when the door opened.\n"
    "Speaker 2{voice=\"A calm older technician\", gender=\"male\"}:\n"
    "[Quietly, checking the meters] Then it is not interference, it is a trigger.\n"
    "Speaker 1:\n"
    "[Lowering her voice] Someone built this to wake up when we got close.\n"
    "Speaker 2:\n"
    "[Firm, controlled] Then we step back, breathe, and let the machine tell us what it wants."
)


def get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs):
    audio_prompt_type =enhancer_kwargs.get("audio_prompt_type", "")
    any_source_image = "I" in prompt_enhancer_mode
    if "A" in audio_prompt_type and "1" in audio_prompt_type:
        ID_LORA_I2V_VIDEO_PROMPT = (
            "You are an expert cinematic director writing prompts for talking-video generation. Rewrite the user input into exactly three tagged sections in this order:\n"
            "[VISUAL]: ...\n"
            "[SPEECH]: ...\n"
            "[SOUNDS]: ...\n\n"
        )

        if any_source_image:
            ID_LORA_I2V_VIDEO_PROMPT += (
                "Use the image caption as the source of truth for the person’s appearance, age impression, hairstyle, clothing, framing, and environment. "
                "If the user text conflicts with the image caption, keep visual identity and scene setup aligned with the image while still following the requested action and mood.\n"
            )

        ID_LORA_I2V_VIDEO_PROMPT += (
            "Follow cinematic video-prompt best practices: describe the scene chronologically, start directly with the action, keep the writing literal and precise, and include concrete details about visible movement, facial expression, posture, framing, lighting, and background. "
            "Do not change the user’s intent, only enhance it.\n"
            "In [VISUAL], describe a single believable on-camera speaking shot with stable identity, clear facial visibility, and details that help lip sync and expression. "
            "Mention visible speaking, mouth movement, eye focus, expression changes, and any small gestures that support the speech. Avoid scene cuts and unnecessary action unless requested.\n"
            "In [SPEECH], preserve the exact transcript and language. Do not paraphrase, summarize, or expand it.\n"
            "In [SOUNDS], describe delivery and ambience only, including tone, pace, emotion, loudness, microphone distance, and background sounds, keeping them consistent with the scene.\n"
            "Keep it literal, structured, production-ready, and under 180 words total. Output only the final prompt."
            "For example:"
            "[VISUAL]: A medium close-up shows a middle-aged man with neatly combed dark hair, wearing a black tuxedo jacket, white dress shirt, and black bow tie, seated at a banquet table in a warmly lit reception hall. He faces forward and visibly speaks on camera with clear mouth movement and strong eye contact. His expression is intense and insistent, with tightened brows and a firm jaw. As he talks, he leans slightly toward the table and strikes it with both fists for emphasis, while plates and glasses remain in place around him. The background stays softly blurred, showing elegant table settings and warm golden indoor lighting. The shot remains stable and frontal, keeping his face and upper body clearly visible."
            "[SPEECH]: Welcome ladies and gentlemen to the best show in the world!"
            "[SOUNDS]: The speaker has a loud, forceful, emotionally charged voice with sharp emphasis and close microphone presence. The banquet hall has soft room reverberation, low crowd murmur, and clear table-hit impacts."
        )
        return ID_LORA_I2V_VIDEO_PROMPT, None
    else:
        return None, None
