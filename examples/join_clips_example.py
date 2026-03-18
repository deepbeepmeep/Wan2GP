#!/usr/bin/env python3
"""
Self-contained example: Using join_clips to bridge two video clips

This script demonstrates how to use the join_clips task to smoothly transition
between two video clips using VACE generation - completely independent of worker.py.

Usage:
    python examples/join_clips_example.py \
        --clip1 /path/to/first_clip.mp4 \
        --clip2 /path/to/second_clip.mp4 \
        --output /path/to/output.mp4 \
        --prompt "smooth camera transition between scenes"

The join_clips task:
1. Extracts context frames from the end of clip 1
2. Extracts context frames from the beginning of clip 2
3. Generates transition frames between them
4. Uses masking to preserve context frames and only generate the gap
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# Add project to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.task_handlers.join.generation import handle_join_clips_task


def run_join_clips(
    starting_video_path: str,
    ending_video_path: str,
    output_path: str,
    prompt: str = "smooth camera transition between scenes",
    context_frame_count: int = 8,
    gap_frame_count: int = 53,
    model: str = "wan_2_2_vace_lightning_baseline_2_2_2",
    num_inference_steps: int = 6,
    guidance_scale: float = 3.0,
    seed: int = -1,
    resolution: tuple = None,
    fps: int = None,
    negative_prompt: str = "",
    **kwargs
):
    """
    Run join_clips task directly without worker.py or database.

    Args:
        starting_video_path: Path to first video clip
        ending_video_path: Path to second video clip
        output_path: Path to save output video
        prompt: Description of the transition
        context_frame_count: Frames to extract from each clip (default: 8)
        gap_frame_count: Frames to generate between clips (default: 53)
        model: Model to use (default: wan_2_2_vace_lightning_baseline_2_2_2)
        num_inference_steps: Number of inference steps (default: 6 for Lightning)
        guidance_scale: CFG scale (default: 3.0 for Lightning)
        seed: Random seed (-1 for random)
        resolution: Optional (width, height) tuple to override resolution
        fps: Optional FPS override
        negative_prompt: Optional negative prompt
        **kwargs: Additional VACE parameters
    """

    print("=" * 80)
    print("JOIN CLIPS - STANDALONE EXECUTION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Validate inputs
    clip1 = Path(starting_video_path)
    clip2 = Path(ending_video_path)

    if not clip1.exists():
        print(f"❌ Starting video not found: {starting_video_path}")
        return False

    if not clip2.exists():
        print(f"❌ Ending video not found: {ending_video_path}")
        return False

    print(f"📹 Starting video: {clip1}")
    print(f"📹 Ending video: {clip2}")
    print(f"💾 Output path: {output_path}")
    print(f"📝 Prompt: {prompt}")
    print(f"🎬 Context frames: {context_frame_count}")
    print(f"🎬 Gap frames: {gap_frame_count}")
    print(f"🤖 Model: {model}")
    print()

    # Initialize task queue (like base_tester.py does)
    wan_root = Path(__file__).parent.parent / "Wan2GP"

    from headless_model_management import HeadlessTaskQueue

    print("🔄 Initializing model queue...")
    task_queue = HeadlessTaskQueue(wan_dir=str(wan_root), max_workers=1)
    task_queue.start()
    print("✅ Task queue initialized")
    print()

    try:
        # Build task parameters
        task_params = {
            "starting_video_path": str(clip1.absolute()),
            "ending_video_path": str(clip2.absolute()),
            "context_frame_count": context_frame_count,
            "gap_frame_count": gap_frame_count,
            "prompt": prompt,
            "model": model,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        if negative_prompt:
            task_params["negative_prompt"] = negative_prompt

        if resolution:
            task_params["resolution"] = list(resolution)

        if fps:
            task_params["fps"] = fps

        # Add any additional parameters
        task_params.update(kwargs)

        # Create output directory
        output_dir = Path(output_path).parent / "join_clips_temp"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique task ID
        task_id = f"join_clips_{int(time.time())}"

        print("🚀 Starting join_clips generation...")
        print("-" * 80)

        start_time = time.time()

        # Call join_clips handler directly
        success, result = handle_join_clips_task(
            task_params_from_db=task_params,
            main_output_dir_base=output_dir.parent,
            task_id=task_id,
            task_queue=task_queue,
        )

        generation_time = time.time() - start_time

        print("-" * 80)

        if success:
            print(f"✅ Generation completed in {generation_time:.1f}s")
            print(f"📹 Output: {result}")

            # Move to desired output location
            result_path = Path(result)
            final_output = Path(output_path)
            final_output.parent.mkdir(parents=True, exist_ok=True)

            if result_path.exists():
                import shutil
                shutil.move(str(result_path), str(final_output))

                BYTES_PER_MB = 1024 * 1024
                file_size = final_output.stat().st_size / BYTES_PER_MB  # MB
                print(f"💾 Saved to: {final_output}")
                print(f"📊 File size: {file_size:.1f}MB")

                # Clean up temp directory
                try:
                    shutil.rmtree(output_dir)
                    print(f"🧹 Cleaned up temp directory")
                except Exception as e:
                    print(f"Warning: Failed to clean up temp directory: {e}")

                print()
                print("=" * 80)
                print("✨ JOIN CLIPS COMPLETED SUCCESSFULLY")
                print("=" * 80)
                return True
            else:
                print(f"❌ Output file not found: {result_path}")
                return False
        else:
            print(f"❌ Generation failed: {result}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    finally:
        # Cleanup task queue
        print()
        print("🔄 Shutting down task queue...")
        try:
            task_queue.stop(timeout=30.0)
            print("✅ Task queue stopped")
        except Exception as e:
            print(f"⚠️ Error stopping queue: {e}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Join two video clips with smooth VACE-generated transition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with two clips
  python examples/join_clips_example.py \\
      --clip1 scene1.mp4 \\
      --clip2 scene2.mp4 \\
      --output transition.mp4 \\
      --prompt "smooth camera glide between scenes"

  # With custom parameters
  python examples/join_clips_example.py \\
      --clip1 scene1.mp4 \\
      --clip2 scene2.mp4 \\
      --output transition.mp4 \\
      --prompt "smooth camera glide" \\
      --context-frames 12 \\
      --gap-frames 81 \\
      --seed 42
        """
    )

    # Required arguments
    parser.add_argument("--clip1", required=True,
                       help="Path to starting video clip")
    parser.add_argument("--clip2", required=True,
                       help="Path to ending video clip")
    parser.add_argument("--output", required=True,
                       help="Path to save output video")
    parser.add_argument("--prompt", required=True,
                       help="Description of the transition")

    # Optional arguments
    parser.add_argument("--context-frames", type=int, default=8,
                       help="Frames to extract from each clip (default: 8)")
    parser.add_argument("--gap-frames", type=int, default=53,
                       help="Frames to generate between clips (default: 53)")
    parser.add_argument("--model", default="wan_2_2_vace_lightning_baseline_2_2_2",
                       help="Model to use (default: wan_2_2_vace_lightning_baseline_2_2_2)")
    parser.add_argument("--steps", type=int, default=6,
                       help="Number of inference steps (default: 6)")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                       help="CFG scale (default: 3.0)")
    parser.add_argument("--seed", type=int, default=-1,
                       help="Random seed, -1 for random (default: -1)")
    parser.add_argument("--resolution", type=str,
                       help="Resolution as WIDTHxHEIGHT (e.g. 1280x720)")
    parser.add_argument("--fps", type=int,
                       help="FPS override")
    parser.add_argument("--negative-prompt", default="",
                       help="Negative prompt")

    args = parser.parse_args()

    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            w, h = map(int, args.resolution.split('x'))
            resolution = (w, h)
        except Exception:
            print(f"❌ Invalid resolution format: {args.resolution}")
            print("   Use format: WIDTHxHEIGHT (e.g. 1280x720)")
            return 1

    options = SimpleNamespace(
        starting_video_path=args.clip1,
        ending_video_path=args.clip2,
        output_path=args.output,
        prompt=args.prompt,
        context_frame_count=args.context_frames,
        gap_frame_count=args.gap_frames,
        model=args.model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        resolution=resolution,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )

    try:
        success = run_join_clips(
            starting_video_path=args.clip1,
            ending_video_path=args.clip2,
            output_path=args.output,
            prompt=args.prompt,
            context_frame_count=args.context_frames,
            gap_frame_count=args.gap_frames,
            model=args.model,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            resolution=resolution,
            fps=args.fps,
            negative_prompt=args.negative_prompt,
            options=options,
        )
    except Exception as exc:
        print(f"Join-clips example failed: {exc}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
