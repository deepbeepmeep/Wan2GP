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
import itertools
import os
import re

_WORD_NORM_RE = re.compile(r"^[^a-z0-9']+|[^a-z0-9']+$")

_SECTION_RE = re.compile(r"^\s*(\[.+?\])\s*(.*)$")


def parse_lyrics_blocks(lyrics_text, max_lines_per_block=1, max_chars_per_block=160):
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


def _norm_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


_MUSIC_JUNK = frozenset({
    "music", "music intro", "music outro", "intro music", "outro music",
    "instrumental", "instrumental music", "background music",
    "thanks for watching", "thank you for watching", "please subscribe",
    "subscribe", "like and subscribe", "silence", "applause", "laughter",
    "end of song", "end of music", "song ends", "music ends", "the end",
})


def clean_music_segments(segments):
    """Drop Whisper music-description hallucinations (not sung words).

    Catches cues like 'Music Outro', 'Thanks for watching!' that Whisper
    emits over pure music. Only exact short matches are dropped, so real
    lyric lines containing such words are preserved.
    """
    cleaned = []
    for s in segments or []:
        text = _norm_text(re.sub(r"[^\w\s]", "", s.get("text", "") if isinstance(s, dict) else s))
        if text and len(text.split()) <= 8 and text in _MUSIC_JUNK:
            continue
        cleaned.append(s)
    return cleaned


def _word_set(text):
    return set(w for w in re.split(r"\s+", _norm_text(text)) if w)


def pick_lyrics_base_multi(candidates, words=None, segments=None, min_hits=3, margin=3):
    """Pick whichever input text holds the actual song lyrics.

    Song generation may sing the enhanced prompt while save_prompt only
    holds the original idea, and lyrics/style split across prompt fields
    differently per model. Scores each candidate by word overlap with the
    transcribed vocals; ties fall back to the longest text.
    """
    texts = [str(c or "") for c in candidates or []]
    heard = set()
    for w in words or []:
        n = _norm_word(w.get("word", "") if isinstance(w, dict) else w)
        if n:
            heard.add(n)
    if not heard:
        for s in segments or []:
            heard |= _word_set(s.get("text", "") if isinstance(s, dict) else s)
    scored = [(_word_set(t) & heard, t) for t in texts if t.strip()]
    if not scored:
        return "", False
    scored.sort(key=lambda item: len(item[0]), reverse=True)
    best_hits = len(scored[0][0])
    runner_hits = len(scored[1][0]) if len(scored) > 1 else 0
    if heard and best_hits >= min_hits and best_hits - runner_hits >= margin:
        return scored[0][1], True
    longest = max(scored, key=lambda item: len(item[1]))
    return longest[1], False


def pick_lyrics_base(prompt_text, alt_text, words=None, segments=None):
    """Pick whichever input field holds the actual song lyrics.

    Song models split lyrics/style across prompt/alt_prompt differently.
    The lyrics field is the one whose words best overlap the transcribed
    vocals. Returns (base_text, is_alt).
    """
    prompt_text = str(prompt_text or "")
    alt_text = str(alt_text or "")
    heard = set()
    for w in words or []:
        n = _norm_word((w.get("word", "") if isinstance(w, dict) else w))
        if n:
            heard.add(n)
    if not heard:
        for s in segments or []:
            heard |= _word_set(s.get("text", "") if isinstance(s, dict) else s)
    if not heard:
        return (prompt_text if prompt_text.strip() else alt_text, False)
    p_hit = len(_word_set(prompt_text) & heard)
    a_hit = len(_word_set(alt_text) & heard)
    # Require a clear margin before preferring alt over prompt
    if alt_text.strip() and (a_hit > p_hit + 2 or (p_hit <= 2 and a_hit > p_hit)):
        return alt_text, True
    return prompt_text if prompt_text.strip() else alt_text, False


def is_hallucinated_repetition(segments, threshold=0.6):
    """Detect Whisper hallucinations on music: the same sentence repeated.

    Returns True when most segments share identical text (e.g. 3/3
    "we are now at upper session road"). Real songs stay far below the
    threshold (repeated choruses excepted, which still vary).
    """
    texts = [_norm_text(s.get("text", "")) for s in segments or []]
    texts = [t for t in texts if t]
    if len(texts) < 3:
        return False
    top_count = max(len(list(g)) for _, g in itertools.groupby(sorted(texts)))
    return top_count / len(texts) > threshold


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


def align_lyrics_to_words(lyrics_text, words, max_lines_per_block=1, max_chars_per_block=160):
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


def align_lyrics_to_segments(lyrics_text, segments, min_ratio=0.6, max_lines_per_block=1, max_chars_per_block=160, activity_spans=None):
    """Align input lyric lines to Whisper segment timestamps.

    Used when word timestamps are unavailable (DTW often fails on sung
    vocals): each exact lyric line inherits its matching segment's timing.
    Unmatched lines spread across vocal-active spans inside their gap
    (hallucinated segments still mark sung positions), so unheard verses
    land where the singing is, not over silence.
    Returns cue list or None when too few lines match.
    """
    segs = []
    for s in segments or []:
        try:
            start, end = float(s.get("start", None)), float(s.get("end", None))
        except (TypeError, ValueError):
            continue
        text = _norm_text(s.get("text", ""))
        if not text or not end > start or start < 0:
            continue
        segs.append((start, end, text))
    if not segs:
        return None
    lines = _split_lyric_lines(lyrics_text)
    if not lines:
        return None
    norm_lines = [_norm_text(ln) for ln in lines]
    # Monotonic best-match scan: each line matches at/after the last hit
    line_time = [None] * len(lines)
    pos, hits = 0, 0
    for li, nl in enumerate(norm_lines):
        best, best_j = min_ratio, -1
        for j in range(pos, min(pos + 6, len(segs))):
            r = difflib.SequenceMatcher(None, nl, segs[j][2], autojunk=False).ratio()
            if r > best:
                best, best_j = r, j
        if best_j >= 0:
            line_time[li] = (segs[best_j][0], segs[best_j][1])
            pos = best_j + 1
            hits += 1
    if hits < 2:
        return None
    # Spread unmatched runs across vocal-active spans inside their gaps so
    # every lyric line lands where singing was detected (hallucinated
    # segments still mark sung positions) instead of over silence.
    activity = sorted(
        (max(0.0, float(a)), max(0.0, float(b)))
        for a, b in (activity_spans or [])
        if b is not None and a is not None and float(b) > float(a)
    )
    def spread_run(li, k, gap_start, gap_end):
        active = []
        for a, b in activity:
            s, e = max(a, gap_start), min(b, gap_end)
            if e > s:
                active.append((s, e))
        if not active:
            active = [(gap_start, gap_end)] if gap_end is not None else []
        if not active:
            for m in range(k):
                s = gap_start + m * 4.0
                line_time[li + m] = (s, s + 3.5)
            return
        total = sum(e - s for s, e in active)
        per = max(total, k * 0.5) / k
        cursor, span_idx, m = active[0][0], 0, 0
        while m < k:
            s = cursor
            e = s + per
            span_end = active[span_idx][1]
            if e > span_end and span_idx + 1 < len(active):
                s = e = span_end
                cursor = active[span_idx + 1][0]
                span_idx += 1
                continue
            if m == k - 1 and gap_end is not None:
                e = gap_end
            line_time[li + m] = (s, max(s + 0.5, e))
            cursor, m = e, m + 1
    li = 0
    while li < len(lines):
        if line_time[li] is not None:
            li += 1
            continue
        j = li
        while j < len(lines) and line_time[j] is None:
            j += 1
        prev_end = line_time[li - 1][1] if li > 0 else 0.0
        next_start = line_time[j][0] if j < len(lines) else None
        spread_run(li, j - li, prev_end, next_start)
        li = j
    # Group into cue blocks, enforce monotonic non-overlap
    cues, block, bstart, bend, block_len = [], [], None, None, 0
    for li, ln in enumerate(lines):
        if block and (len(block) >= max_lines_per_block or block_len + len(ln) > max_chars_per_block):
            cues.append((bstart, bend, "\n".join(block)))
            block, bstart, bend, block_len = [], None, None, 0
        block.append(ln)
        block_len += len(ln) + 1
        bstart = line_time[li][0] if bstart is None else bstart
        bend = line_time[li][1]
    if block:
        cues.append((bstart, bend, "\n".join(block)))
    fixed = []
    for start, end, text in cues:
        end = max(start + 0.5, end)
        if fixed:
            start = max(start, fixed[-1][1])
            end = max(start + 0.5, end)
        fixed.append((start, end, text))
    return fixed or None


def write_segment_aligned_lyrics_srt(lyrics_text, segments, srt_path, activity_spans=None):
    """Write input lyrics aligned to Whisper segment timestamps as .srt."""
    cues = align_lyrics_to_segments(lyrics_text, segments, activity_spans=activity_spans)
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
