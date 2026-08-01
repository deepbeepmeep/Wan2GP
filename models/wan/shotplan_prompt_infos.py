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
