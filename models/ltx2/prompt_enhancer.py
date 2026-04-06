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