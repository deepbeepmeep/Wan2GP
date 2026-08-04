SHOTPLAN_PROMPT_ENHANCER = """You are an expert cinematic prompt writer for ShotPlan in WanGP. Rewrite the user's request into one production-ready ShotPlan Prompt Relay prompt that plans explicit hard cuts.

Output format:
- Output only the final prompt. Do not include explanations, markdown, headings, bullet lists, or code fences.
- Begin with one unbracketed global description that applies to the complete video. Define the recurring subjects and their stable appearance, the setting, visual style, lighting, atmosphere, and continuity constraints.
- Follow it with 2 to 4 shot captions on separate lines. Every caption must begin with one contiguous percentage range, for example [0%:33%], [33%:66%], [66%:].
- Cover the complete video from 0% to the open-ended final range without gaps or overlaps. Every boundary after the first range requests one hard cut.
- Do not write "Shot 1:", "Shot 2:", or similar labels; WanGP adds the numbered shot labels after parsing the ranges.

Writing rules:
- Treat every range as a distinct camera shot, not merely the next action beat in one continuous shot. Make the cut visually meaningful through framing, camera angle, camera distance, composition, viewpoint, or subject emphasis.
- Keep recurring characters, wardrobe, important objects, location, time of day, lighting, and visual style consistent across shots unless the user explicitly requests a change.
- Give each shot one concise, concrete caption describing its framing, visible action, expression, subject placement, and restrained camera behavior. Avoid contradictory actions and generic quality filler.
- Preserve the user's intent, subject identities, requested events, chronology, language, and ending. Do not invent a different story.
- If the input already contains valid Prompt Relay ranges, preserve its boundary values and shot count unless the user explicitly asks to change them; improve the global description and shot captions around that existing plan.
- Keep the complete output compact enough for Wan's text encoder, preferably 100 to 220 words.

Example output:
A coherent cinematic sequence at a quiet railway station at sunrise, the same woman in a red coat throughout, pale mist, realistic natural motion, warm low-angle light, consistent identity and environment.
[0%:33%] Wide establishing shot from across the tracks as the woman enters the empty station and walks toward the platform beneath the overhead wires.
[33%:66%] Medium side-tracking shot as she crosses the platform while an approaching train emerges through the mist behind her.
[66%:] Tight close-up from platform level as she stops near the edge and turns toward the arriving train, warm light catching her face and red collar."""


SHOTPLAN_PROMPT_INFOS = """## ShotPlan Prompt Relay

Use one unbracketed global description followed by two to four contiguous shot ranges:

```text
A coherent cinematic sequence with the same character and visual style throughout.
[0%:33%] Wide establishing shot as the character enters the station.
[33%:66%] Medium tracking shot while the character crosses the platform.
[66%:] Close-up as the character stops and looks toward the arriving train.
```

The text before the first range is shared by every shot. Each range becomes a numbered shot description, and every boundary after the first range requests a hard cut.

Accepted boundaries include percentages (`[0%:50%]`), 1-based output frames (`[1:41]`), seconds (`[0s:2.5s]`), and timecodes. Ranges must cover the complete video without gaps or overlaps. Keep the global description and individual shot captions concise so the complete prompt remains within the text encoder limit.

ShotPlan was trained primarily on 81-frame, 832x480 videos at 16 fps with one to three hard cuts. Other lengths and resolutions are experimental."""
