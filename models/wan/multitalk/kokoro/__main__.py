"""Kokoro TTS CLI
Example usage:
python3 -m kokoro --text "The sky above the port was the color of television, tuned to a dead channel." -o file.wav --debug

echo "Bom dia mundo, como vão vocês" > text.txt
python3 -m kokoro -i text.txt -l p --voice pm_alex > audio.wav

Common issues:
pip not installed: `uv pip install pip`
(Temporary workaround while https://github.com/explosion/spaCy/issues/13747 is not fixed)

espeak not installed: `apt-get install espeak-ng`
"""

import argparse
import wave
from pathlib import Path
from typing import Generator, TYPE_CHECKING

import numpy as np
from loguru import logger

languages = [
    "a",  # American English
    "b",  # British English
    "h",  # Hindi
    "e",  # Spanish
    "f",  # French
    "i",  # Italian
    "p",  # Brazilian Portuguese
    "j",  # Japanese
    "z",  # Mandarin Chinese
]

if TYPE_CHECKING:
    from kokoro import KPipeline


def generate_audio(
    text: str, kokoro_language: str, voice: str, speed=1
) -> Generator["KPipeline.Result", None, None]:
    from kokoro import KPipeline

    if not voice.startswith(kokoro_language):
        logger.warning(f"Voice {voice} is not made for language {kokoro_language}")
    pipeline = KPipeline(lang_code=kokoro_language)
    yield from pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")


def generate_and_save_audio(
    output_file: Path, text: str, kokoro_language: str, voice: str, speed=1
) -> None:
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        wav_file.setframerate(24000)  # Sample rate

        for result in generate_audio(
            text, kokoro_language=kokoro_language, voice=voice, speed=speed
        ):
            logger.debug(result.phonemes)
            if result.audio is None:
                continue
            audio_bytes = (result.audio.numpy() * 32767).astype(np.int16).tobytes()
            wav_file.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--voice",
        default="af_heart",
        help="Voice to use",
    )
    parser.add_argument(
        "-l",
        "--language",
        help="Language to use (defaults to the one corresponding to the voice)",
        choices=languages,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        "--output_file",
        type=Path,
        help="Path to output WAV file",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input-file",
        "--input_file",
        type=Path,
        help="Path to input text file (default: stdin)",
    )
    parser.add_argument(
        "-t",
        "--text",
        help="Text to use instead of reading from stdin",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print DEBUG messages to console",
    )
    args = parser.parse_args()
    if args.debug:
        logger.level("DEBUG")
    logger.debug(args)

    lang = args.language or args.voice[0]

    if args.text is not None and args.input_file is not None:
        raise Exception("You cannot specify both 'text' and 'input_file'")
    elif args.text:
        text = args.text
    elif args.input_file:
        file: Path = args.input_file
        text = file.read_text()
    else:
        import sys
        print("Press Ctrl+D to stop reading input and start generating", flush=True)
        text = '\n'.join(sys.stdin)

    logger.debug(f"Input text: {text!r}")

    out_file: Path = args.output_file
    if not out_file.suffix == ".wav":
        logger.warning("The output file name should end with .wav")
    generate_and_save_audio(
        output_file=out_file,
        text=text,
        kokoro_language=lang,
        voice=args.voice,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
