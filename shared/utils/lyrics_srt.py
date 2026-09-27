"""Write .srt sidecar files for generated songs and videos.

Two modes:
- Aligned (exact lyrics, exact timing): input lyrics are force-aligned to
  Whisper word timestamps, so cues show the true lyrics at the real
  vocal positions.
- Transcribed fallback: raw Whisper segments when alignment is impossible.
- Lyrics fallback (approximate): input lyrics blocks distributed evenly
  over the output media duration.
"""

import difflib
import os
import re

_WORD_NORM_RE = re.compile(r"^[^a-z0-9']+|[^a-z0-9']+$")

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


def _norm_word(word):
    return _WORD_NORM_RE.sub("", str(word or "").lower().strip())


def _split_lyric_lines(lyrics_text):
    """Split lyrics into display lines, keeping [Section] tags attached."""
    if lyrics_text is None:
        return []
    text = str(lyrics_text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    lines = []
    pending_tag = None
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            pending_tag = None
            continue
        m = _SECTION_RE.match(line)
        if m:
            tag, rest = m.group(1).strip(), m.group(2).strip()
            if rest:
                if pending_tag:
                    lines.append(pending_tag)
                lines.append(f"{tag} {rest}" if pending_tag is None else f"{pending_tag} {tag} {rest}")
                pending_tag = None
            else:
                if pending_tag:
                    lines.append(pending_tag)
                pending_tag = tag
        else:
            if pending_tag:
                lines.append(f"{pending_tag} {line}")
                pending_tag = None
            else:
                lines.append(line)
    if pending_tag:
        lines.append(pending_tag)
    return lines


def align_lyrics_to_words(lyrics_text, words, max_lines_per_block=2, max_chars_per_block=160):
    """Force-align input lyric lines to Whisper word timestamps.

    Returns a list of (start, end, text) cues with the exact input lyrics
    at the real vocal positions, or None when alignment is impossible.
    """
    wstarts, wends, wnorm = [], [], []
    for w in words or []:
        try:
            start, end = float(w.get("start", None)), float(w.get("end", None))
        except (TypeError, ValueError):
            continue
        norm = _norm_word(w.get("word", ""))
        if not norm or not end > start or start < 0:
            continue
        wstarts.append(start)
        wends.append(end)
        wnorm.append(norm)
    if not wnorm:
        return None
    lines = _split_lyric_lines(lyrics_text)
    if not lines:
        return None
    # Flatten lyric words, remembering each word's display line
    lyric_norm, lyric_line = [], []
    for li, line in enumerate(lines):
        ows = [w for w in re.split(r"\s+", line) if w]
        if not ows:
            continue
        for w in ows:
            lyric_norm.append(_norm_word(w))
            lyric_line.append(li)
    # Drop words that normalize to nothing (pure punctuation)
    kept = [(n, li) for n, li in zip(lyric_norm, lyric_line) if n]
    if not kept:
        return None
    lyric_norm = [n for n, _ in kept]
    lyric_line = [li for _, li in kept]
    matcher = difflib.SequenceMatcher(None, lyric_norm, wnorm, autojunk=False)
    t_start = [None] * len(lyric_norm)
    t_end = [None] * len(lyric_norm)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                t_start[i1 + k] = wstarts[j1 + k]
                t_end[i1 + k] = wends[j1 + k]
    anchors = [i for i, t in enumerate(t_start) if t is not None]
    if len(anchors) < 3:
        return None
    # Interpolate unmatched words between anchors
    for a, b in zip(anchors, anchors[1:]):
        if b - a < 2:
            continue
        t0, t1 = t_end[a], t_start[b]
        gap = b - a
        for k in range(a + 1, b):
            f = (k - a) / gap
            t = t0 + (t1 - t0) * f
            t_start[k] = t if t_start[k] is None else t_start[k]
            t_end[k] = t if t_end[k] is None else t_end[k]
    first, last = anchors[0], anchors[-1]
    for k in range(first - 1, -1, -1):
        t_start[k] = max(0.0, t_start[k + 1] - 0.4)
        t_end[k] = t_start[k + 1]
    for k in range(last + 1, len(lyric_norm)):
        t_start[k] = t_end[k - 1]
        t_end[k] = t_end[k - 1] + 0.4
    # Group display lines into cue blocks
    line_start = [None] * len(lines)
    line_end = [None] * len(lines)
    for idx, li in enumerate(lyric_line):
        if line_start[li] is None:
            line_start[li] = t_start[idx]
        line_end[li] = t_end[idx]
    cues = []
    block, block_len, bstart, bend = [], 0, None, None
    nonempty = [(li, ln) for li, ln in enumerate(lines) if line_start[li] is not None]
    for li, ln in nonempty:
        if block and (len(block) >= max_lines_per_block or block_len + len(ln) > max_chars_per_block):
            cues.append((bstart, bend, "\n".join(block)))
            block, block_len, bstart, bend = [], 0, None, None
        block.append(ln)
        block_len += len(ln) + 1
        bstart = line_start[li] if bstart is None else bstart
        bend = line_end[li]
    if block:
        cues.append((bstart, bend, "\n".join(block)))
    # Enforce positive, monotonic, non-overlapping cues
    fixed = []
    for start, end, text in cues:
        if end is None or start is None:
            continue
        end = max(start + 0.5, end)
        if fixed:
            start = max(start, fixed[-1][1])
            end = max(start + 0.5, end)
        fixed.append((start, end, text))
    return fixed or None


def write_aligned_lyrics_srt(lyrics_text, words, srt_path):
    """Write input lyrics aligned to Whisper word timestamps as .srt.

    Returns srt_path on success, None when alignment is impossible.
    """
    cues = align_lyrics_to_words(lyrics_text, words)
    if not cues:
        return None
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(srt_path)), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    return srt_path


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


def write_segments_srt(segments, srt_path):
    """Write Whisper-style segments ([{start, end, text}]) as .srt.

    Uses the real transcription timestamps, so cues are in sync with the
    performed audio. Returns srt_path on success, None when empty.
    """
    cues = []
    for seg in segments or []:
        try:
            start = float(seg.get("start", None))
            end = float(seg.get("end", None))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text", "") or "").strip()
        if not text or not end > start or start < 0:
            continue
        cues.append((start, end, text))
    if not cues:
        return None
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(srt_path)), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    return srt_path
