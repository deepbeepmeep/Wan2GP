# Adapted from fishaudio/fish-speech, revision 1ff3c3ed2f80d3106a83909dd25b83129c3d816f.
# Copyright Fish Audio contributors. See LICENSE_FISH_AUDIO for the upstream license.
import re
from collections.abc import Sequence
import torch
from tokenizers import Tokenizer
from .config import FishOmniConfig

DRAMA3_TTS_INSTRUCTION = "convert the provided text to speech"
DRAMA3_REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DRAMA3_MESSAGE_END = "<|im_end|>\n"
DRAMA3_GENERATION_START = "<|im_start|>assistant\n<|voice|>"
# Only the prompt builder writes these; other special tokens in caller text are content.
DRAMA3_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
# The tokenizer has one token per speaker, <|speaker:0|> to <|speaker:15|>.
MAX_SPEAKER_ID = 15
# The Unicode White_Space property; str.strip() without arguments also strips
# U+001C..U+001F, which would change the empty check and the speaker-tag rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
DRAMA3_MAX_PROMPT_ROWS = 8192
class Drama3PromptTooLongError(ValueError):
    """The prompt has more than DRAMA3_MAX_PROMPT_ROWS rows."""


def validate_drama3_text(text: str, location: str) -> None:
    """Non-empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    for token in DRAMA3_STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(
                f"{location}: {token} is written by the prompt builder"
            )
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise ValueError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )


# WanGP's dialogue convention, shared with its other speech models: "Speaker N:" (1..16) at the start of a line.
SPEAKER_LINE_PATTERN = re.compile(r"^([ \t]*)Speaker[ \t]*(\d+)[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)


# Drama 3's paired tags apply to the words between them; an unclosed one spoils the speech that follows.
PAIRED_TAGS = ("whisper", "stress", "slow")


def check_paired_tags(text: str, location: str) -> None:
    """Only Drama 3's paired tags may use angle brackets (besides <|...|> tokens), each one closed."""
    for match in re.finditer(r"</?([A-Za-z][^<>|]*)>?", text):  # tag-like: "<" before a letter (not "x < y" or "3<4")
        if match.group(1) not in PAIRED_TAGS or not match.group(0).endswith(">"):
            raise ValueError(f"{location}: {match.group(0)!r} is not a Drama 3 tag; tags are <whisper>, <stress> and <slow> around words, other directions go in square brackets such as [smiling]")
    for tag in PAIRED_TAGS:
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            raise ValueError(f"{location}: close each <{tag}> with </{tag}> after the words it applies to")


def speaker_lines_to_tags(text: str) -> str:
    """'Speaker N:' line labels -> Drama 3's <|speaker:N-1|>; native tags are kept."""
    def tag(match):
        number = int(match.group(2))
        if not 1 <= number <= MAX_SPEAKER_ID + 1:
            raise ValueError(f"speaker labels go from Speaker 1: to Speaker {MAX_SPEAKER_ID + 1}:; got Speaker {number}:")
        return f"{match.group(1)}<|speaker:{number - 1}|> "
    return SPEAKER_LINE_PATTERN.sub(tag, text)


def dialogue_parts(text: str, lines_per_part: int) -> list[str]:
    """The script in parts of lines_per_part whole lines (one part when 0 or shorter). A line without a leading tag continues
    the speaker before it, so a part starting there gets that speaker's tag."""
    lines = [line for line in text.splitlines() if line.strip(UNICODE_WHITE_SPACE)]
    if lines_per_part <= 0 or len(lines) <= lines_per_part:
        return [text]
    parts, speaker = [], "<|speaker:0|>"
    for start in range(0, len(lines), lines_per_part):
        group = lines[start:start + lines_per_part]
        if SPEAKER_TAG_PATTERN.match(group[0].lstrip(UNICODE_WHITE_SPACE)) is None:
            group[0] = f"{speaker} {group[0].lstrip(UNICODE_WHITE_SPACE)}"
        for line in group:
            for match in SPEAKER_TAG_PATTERN.finditer(line):
                speaker = match.group(0)
        parts.append("\n".join(group))
    return parts


def reference_transcripts(text: str, count: int) -> list[str]:
    """One transcript per voice reference ('' when not given): 'Speaker N:' sections, else one line per reference; with a
    single reference, the whole text."""
    matches = list(SPEAKER_LINE_PATTERN.finditer(text))
    if matches:
        values = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            values[int(match.group(2))] = text[match.end():end].strip(UNICODE_WHITE_SPACE)
        return [values.get(number, "") for number in range(1, count + 1)]
    if count == 1:
        return [text.strip(UNICODE_WHITE_SPACE)]
    lines = [line.strip(UNICODE_WHITE_SPACE) for line in text.splitlines() if line.strip(UNICODE_WHITE_SPACE)]
    return (lines + [""] * count)[:count]


def with_speaker_tag(text: str) -> str:
    """Prefix <|speaker:0|> unless a tag already leads the text after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def build_drama3_prompt(
    text: str,
    tokenizer: Tokenizer,
    config: FishOmniConfig,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[torch.Tensor] = (),
    history: Sequence[tuple[str, torch.Tensor]] = (),
) -> torch.Tensor:
    """CPU int64 rows [rows, 1 + num_codebooks] ending in the open assistant voice turn.

    Text rows are [token, 0, ..., 0]; audio rows are [c0 + semantic_start, c0..cN-1].
    Every literal is tokenized on its own: tokenizing the joined string changes ids
    at the seams. history holds earlier (text, codes) chunks, written as completed
    user and assistant voice turns before the current one.
    """
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")
    if len(prompt_texts) != len(prompt_tokens):
        raise ValueError("every reference needs both a transcript and codes")
    validate_drama3_text(text, "text")

    def check_codes(codes: torch.Tensor, location: str) -> None:
        if (
            codes.dtype.is_floating_point
            or codes.dtype == torch.bool
            or codes.ndim != 2
            or codes.shape[0] != num_codebooks
            or codes.shape[1] == 0
        ):
            raise ValueError(
                f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; "
                f"got {codes.dtype} {tuple(codes.shape)}"
            )
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise ValueError(
                f"{location}: codes must be in [0, {codebook_size})"
            )

    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if prompt_tokens else DRAMA3_TTS_INSTRUCTION,
    ]

    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        location = f"reference {index}"
        validate_drama3_text(transcript, location)
        check_codes(codes, location)
        # The tag must follow "Text: " directly; leading white space moves behind it.
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(DRAMA3_MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_drama3_text(turn_text, location)
        check_codes(codes, location)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            DRAMA3_MESSAGE_END,
            DRAMA3_GENERATION_START,
            codes,
            DRAMA3_MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment).ids, dtype=torch.long, device="cpu")
            text_rows = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long, device="cpu")
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(torch.cat((main, codes)).T)
    rows = torch.cat(rows)
    if len(rows) > DRAMA3_MAX_PROMPT_ROWS:
        raise Drama3PromptTooLongError(
            f"prompt has {len(rows)} rows; at most {DRAMA3_MAX_PROMPT_ROWS}"
        )
    return rows


