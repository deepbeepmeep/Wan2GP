#!/usr/bin/env python3
"""
Self-contained example: Using inpaint_frames to regenerate frames within a video

This script demonstrates how to use the inpaint_frames task to replace/regenerate
a specific range of frames within a single video clip - completely independent of worker.py.

Usage:
    python examples/inpaint_frames_example.py \
        --video /path/to/video.mp4 \
        --start-frame 20 \
        --end-frame 73 \
        --output /path/to/output.mp4 \
        --prompt "smooth continuous camera motion"

Use cases:
- Fix corrupted frames
- Remove unwanted content from a specific time range
- Regenerate a problematic section
- Smooth out transitions
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# Add project to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.task_handlers.inpaint_frames import handle_inpaint_frames_task


def run_inpaint_frames(
    video_path: str,
    inpaint_start_frame: int,
    inpaint_end_frame: int,
    output_path: str,
    prompt: str = "smooth continuous camera motion",
    context_frame_count: int = 8,
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
    Run inpaint_frames task directly without worker.py or database.

    Args:
        video_path: Path to source video
        inpaint_start_frame: Start frame index (inclusive)
        inpaint_end_frame: End frame index (exclusive)
        output_path: Path to save output video
        prompt: Description of desired content
        context_frame_count: Frames to preserve on each side (default: 8)
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
    print("INPAINT FRAMES - STANDALONE EXECUTION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Validate inputs
    video = Path(video_path)

    if not video.exists():
        print(f"❌ Video not found: {video_path}")
        return False

    frames_to_generate = inpaint_end_frame - inpaint_start_frame
    total_output_frames = context_frame_count * 2 + frames_to_generate

    print(f"📹 Source video: {video}")
    print(f"💾 Output path: {output_path}")
    print(f"📝 Prompt: {prompt}")
    print(f"🎬 Inpaint range: frames [{inpaint_start_frame}, {inpaint_end_frame})")
    print(f"🎬 Frames to regenerate: {frames_to_generate}")
    print(f"🎬 Context frames: {context_frame_count} (on each side)")
    print(f"🎬 Total output frames: {total_output_frames}")
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
            "video_path": str(video.absolute()),
            "inpaint_start_frame": inpaint_start_frame,
            "inpaint_end_frame": inpaint_end_frame,
            "context_frame_count": context_frame_count,
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
        output_dir = Path(output_path).parent / "inpaint_frames_temp"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique task ID
        task_id = f"inpaint_frames_{int(time.time())}"

        print("🚀 Starting inpaint_frames generation...")
        print("-" * 80)

        start_time = time.time()

        # Call inpaint_frames handler directly
        success, result = handle_inpaint_frames_task(
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
                print("✨ INPAINT FRAMES COMPLETED SUCCESSFULLY")
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
        description="Regenerate a range of frames within a video using VACE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix corrupted frames 45-60
  python examples/inpaint_frames_example.py \\
      --video my_video.mp4 \\
      --start-frame 45 \\
      --end-frame 61 \\
      --output fixed_video.mp4 \\
      --prompt "smooth continuous motion" \\
      --context-frames 10

  # Remove unwanted content from frames 100-150
  python examples/inpaint_frames_example.py \\
      --video nature_scene.mp4 \\
      --start-frame 100 \\
      --end-frame 150 \\
      --output cleaned_scene.mp4 \\
      --prompt "empty natural landscape, no people" \\
      --negative-prompt "person, people, human"

  # Smooth jarring transition at frame 200
  python examples/inpaint_frames_example.py \\
      --video edited_video.mp4 \\
      --start-frame 195 \\
      --end-frame 205 \\
      --output smooth_video.mp4 \\
      --prompt "smooth gradual transition" \\
      --context-frames 12
        """
    )

    # Required arguments
    parser.add_argument("--video", required=True,
                       help="Path to source video")
    parser.add_argument("--start-frame", type=int, required=True,
                       help="Start frame index (inclusive)")
    parser.add_argument("--end-frame", type=int, required=True,
                       help="End frame index (exclusive)")
    parser.add_argument("--output", required=True,
                       help="Path to save output video")
    parser.add_argument("--prompt", required=True,
                       help="Description of desired content")

    # Optional arguments
    parser.add_argument("--context-frames", type=int, default=8,
                       help="Frames to preserve on each side (default: 8)")
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

    # Validate frame range
    if args.start_frame >= args.end_frame:
        print(f"❌ Invalid frame range: start_frame ({args.start_frame}) must be < end_frame ({args.end_frame})")
        return 1

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
        video_path=args.video,
        inpaint_start_frame=args.start_frame,
        inpaint_end_frame=args.end_frame,
        output_path=args.output,
        prompt=args.prompt,
        context_frame_count=args.context_frames,
        model=args.model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        resolution=resolution,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )

    try:
        success = run_inpaint_frames(
            video_path=args.video,
            inpaint_start_frame=args.start_frame,
            inpaint_end_frame=args.end_frame,
            output_path=args.output,
            prompt=args.prompt,
            context_frame_count=args.context_frames,
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
        print(f"Inpaint example failed: {exc}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
