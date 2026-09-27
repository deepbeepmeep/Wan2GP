"""Write input lyrics/prompts as timed .srt sidecar files.

The lyrics come from the generation input (song lyrics for audio-only
models, prompt for videos). Timing blocks are distributed evenly over the
actual output media duration, so the .srt stays in sync with the file.
"""

import os
import re

_SECTION_RE = re.compile(r"^\s*(\[.+?\])\s*(.*)$")


def parse_lyrics_blocks(lyrics_text, max_lines_per_block=2, max_chars_per_block=160):
    """Split lyrics text into display blocks for SRT cues."""
    if lyrics_text is None:
        return []
    text = str(lyrics_text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    # Split on blank lines first to preserve verse/chorus structure
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks = []
    for raw in raw_blocks:
        lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        # Keep [Section] tags attached to the following lines
        merged = []
        pending_tag = None
        for line in lines:
            m = _SECTION_RE.match(line)
            if m and m.group(2).strip():
                if pending_tag:
                    merged.append(pending_tag)
                pending_tag = m.group(1).strip()
                lines_rest = m.group(2).strip()
                if pending_tag and lines_rest:
                    merged.append(f"{pending_tag} {lines_rest}")
                    pending_tag = None
                elif pending_tag and not lines_rest:
                    continue
            elif m:
                if pending_tag:
                    merged.append(pending_tag)
                pending_tag = m.group(1).strip()
            else:
                if pending_tag:
                    merged.append(f"{pending_tag} {line}")
                    pending_tag = None
                else:
                    merged.append(line)
        if pending_tag:
            merged.append(pending_tag)
        # Chunk long blocks into smaller cues
        current = []
        current_len = 0
        for line in merged:
            line_len = len(line)
            if current and (len(current) >= max_lines_per_block or current_len + line_len > max_chars_per_block):
                blocks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len + 1
        if current:
            blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def format_srt_timestamp(seconds):
    """Format seconds as HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_lyrics_srt(lyrics_text, duration_seconds, srt_path):
    """Write lyrics distributed evenly over duration as .srt.

    Returns srt_path on success, None when there is nothing to write.
    """
    blocks = parse_lyrics_blocks(lyrics_text)
    if not blocks or not duration_seconds or duration_seconds <= 0:
        return None
    n = len(blocks)
    per_block = float(duration_seconds) / n
    # Keep a small gap between cues for readability
    gap = min(0.2, per_block * 0.05) if n > 1 else 0.0
    lines = []
    for i, block in enumerate(blocks):
        start = i * per_block
        end = (i + 1) * per_block if i < n - 1 else float(duration_seconds)
        end = max(start + 0.5, end - gap)
        if end > duration_seconds:
            end = float(duration_seconds)
        lines.append(str(i + 1))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(block)
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(srt_path)), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    return srt_path
