"""
stable-audio â€” command-line interface for Stable Audio 3.

Basic usage::

    stable-audio --model small-music -p "lo-fi hip hop beat, 90 BPM" --duration 30 -o beat.wav

"""

import argparse
import os
import torch
import torchaudio

from models.TTS.stable_audio3 import StableAudioModel


def _save_output(audio: torch.Tensor, sample_rate: int, output: str, batch_size: int):
    """Save generated audio tensor(s) to disk."""
    base, ext = os.path.splitext(output)
    if not ext:
        ext = ".wav"
    for i in range(batch_size):
        path = f"{base}_{i}{ext}" if batch_size > 1 else f"{base}{ext}"
        torchaudio.save(path, audio[i].cpu(), sample_rate)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        prog="stable-audio",
        description="Stable Audio 3 â€” CLI for text-to-audio, audio-to-audio, and inpainting",
    )

    # Model
    parser.add_argument(
        "--model",
        default="medium",
        choices=[
            "medium",
            "small-music",
            "small-sfx",
            "medium-base",
            "small-music-base",
            "small-sfx-base",
        ],
        help="Model to load (default: medium)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda / mps / cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--no-half", action="store_true", help="Disable half-precision (fp16) on CUDA"
    )

    # Generation
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        nargs="+",
        help="Text prompt(s). Pass multiple for per-batch prompts",
    )
    parser.add_argument(
        "--negative-prompt", nargs="+", default=None, help="Negative prompt(s)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        nargs="+",
        default=[120.0],
        help="Duration in seconds (default: 120). Pass multiple for per-batch durations",
    )
    parser.add_argument(
        "--steps", type=int, default=8, help="Diffusion steps (default: 8)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="CFG scale (default: 1.0; try 7.0 for base models)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Random seed (-1 = random, default: -1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: inferred from number of prompts, or 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.wav",
        help="Output file path (default: output.wav)",
    )

    # Audio-to-Audio
    parser.add_argument(
        "--init-audio",
        default=None,
        metavar="PATH",
        help="Source audio file for audio-to-audio generation",
    )
    parser.add_argument(
        "--init-noise-level",
        type=float,
        default=0.9,
        help="Noise level for audio-to-audio (0.0â€“1.0, default: 0.9)",
    )

    # Inpainting / Continuation
    parser.add_argument(
        "--inpaint-audio",
        default=None,
        metavar="PATH",
        help="Source audio file for inpainting or continuation",
    )
    parser.add_argument(
        "--inpaint-start",
        type=float,
        action="append",
        dest="inpaint_starts",
        metavar="SECONDS",
        help="Start of inpaint region in seconds. Repeat for multiple regions.",
    )
    parser.add_argument(
        "--inpaint-end",
        type=float,
        action="append",
        dest="inpaint_ends",
        metavar="SECONDS",
        help="End of inpaint region in seconds. Repeat for multiple regions.",
    )

    # Chunked decode
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument(
        "--chunked-decode",
        action="store_true",
        default=None,
        help="Force chunked decoding on",
    )
    decode_group.add_argument(
        "--no-chunked-decode",
        action="store_true",
        default=None,
        help="Force chunked decoding off",
    )

    # LoRA
    parser.add_argument(
        "--lora-ckpt-path",
        action="append",
        dest="loras",
        metavar="PATH",
        help="LoRA checkpoint path. Repeat to stack multiple LoRAs.",
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=None,
        help="LoRA strength (applied to all LoRAs)",
    )
    parser.add_argument(
        "--lora-index",
        type=int,
        default=None,
        help="Target a specific LoRA index when setting strength",
    )

    args = parser.parse_args()

    # --- Validate inpaint args ---
    if (args.inpaint_starts is None) != (args.inpaint_ends is None):
        parser.error("--inpaint-start and --inpaint-end must both be provided together")
    if args.inpaint_starts and len(args.inpaint_starts) != len(args.inpaint_ends):
        parser.error(
            "--inpaint-start and --inpaint-end must be specified the same number of times"
        )
    if args.inpaint_starts and not args.inpaint_audio:
        parser.error("--inpaint-start/--inpaint-end require --inpaint-audio")
    if args.inpaint_audio and not args.inpaint_starts:
        parser.error("--inpaint-audio requires --inpaint-start and --inpaint-end")

    # --- Resolve batch size ---
    n_prompts = len(args.prompt)
    if args.batch_size is None:
        batch_size = n_prompts
    elif n_prompts > 1 and args.batch_size != n_prompts:
        parser.error(
            f"--batch-size {args.batch_size} does not match the number of prompts "
            f"({n_prompts}); omit --batch-size to have it inferred automatically"
        )
    else:
        batch_size = args.batch_size

    # --- Validate list-flag lengths against batch size ---
    if (
        args.negative_prompt
        and len(args.negative_prompt) > 1
        and len(args.negative_prompt) != batch_size
    ):
        parser.error(
            f"Got {len(args.negative_prompt)} --negative-prompt values but batch size is {batch_size}"
        )
    if len(args.duration) > 1 and len(args.duration) != batch_size:
        parser.error(
            f"Got {len(args.duration)} --duration values but batch size is {batch_size}"
        )

    # --- Build scalar / list args ---
    prompt = args.prompt[0] if len(args.prompt) == 1 else args.prompt
    negative_prompt = None
    if args.negative_prompt:
        negative_prompt = (
            args.negative_prompt[0]
            if len(args.negative_prompt) == 1
            else args.negative_prompt
        )
    duration = args.duration[0] if len(args.duration) == 1 else args.duration

    # --- chunked_decode flag ---
    chunked_decode = None
    if args.chunked_decode:
        chunked_decode = True
    elif args.no_chunked_decode:
        chunked_decode = False

    # --- Load model ---
    print(f"Loading model '{args.model}'â€¦")
    model = StableAudioModel.from_pretrained(
        args.model, device=args.device, model_half=not args.no_half
    )

    # --- LoRA ---
    if args.loras:
        print(f"Loading LoRA(s): {args.loras}")
        model.load_lora(args.loras)
    if args.lora_strength is not None:
        model.set_lora_strength(args.lora_strength, lora_index=args.lora_index)

    # --- Load audio inputs ---
    # torchaudio.load returns (waveform, sample_rate); model.generate expects (sample_rate, waveform)
    init_audio = None
    if args.init_audio:
        waveform, sr = torchaudio.load(args.init_audio)
        init_audio = (sr, waveform)

    inpaint_audio = None
    if args.inpaint_audio:
        waveform, sr = torchaudio.load(args.inpaint_audio)
        inpaint_audio = (sr, waveform)

    inpaint_start = None
    inpaint_end = None
    if args.inpaint_starts:
        inpaint_start = (
            args.inpaint_starts[0]
            if len(args.inpaint_starts) == 1
            else args.inpaint_starts
        )
        inpaint_end = (
            args.inpaint_ends[0] if len(args.inpaint_ends) == 1 else args.inpaint_ends
        )

    # --- Generate ---
    print("Generatingâ€¦")
    audio = model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        batch_size=batch_size,
        init_audio=init_audio,
        init_noise_level=args.init_noise_level,
        inpaint_audio=inpaint_audio,
        inpaint_mask_start_seconds=inpaint_start,
        inpaint_mask_end_seconds=inpaint_end,
        chunked_decode=chunked_decode,
    )

    _save_output(audio, model.model.sample_rate, args.output, batch_size)


if __name__ == "__main__":
    main()

