from PIL import Image

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output/batch_interactive"
I2V_SCRIPT = "i2v_inference.py"
TEMP_BLACK_IMAGE = os.path.join(OUTPUT_DIR, "_temp_black.png")

# --- Helper Functions ---

def create_black_image(width, height, filepath):
    """Creates and saves a black PNG image."""
    try:
        img = Image.new('RGB', (width, height), 'black')
        img.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error creating black image: {e}")
        return None

def get_user_input(prompt):
    """Gets input from the user."""
    return input(prompt).strip()

def get_confirmation(prompt="Confirm? (y/n/r) 'y' to accept, 'n' to skip, 'r' to regenerate: "):
    """Gets user confirmation (yes/no/regenerate)."""
    while True:
        choice = get_user_input(prompt).lower()
        if choice in ['y', 'n', 'r']:
            return choice
        print("Invalid input. Please enter 'y', 'n', or 'r'.")

def select_images_for_scene(folder):
    """Lets the user select multiple image files from a folder."""
    all_images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_images:
        print(f"No images found in '{folder}'.")
        return []

    selected_paths = []
    while True:
        print("\nAvailable images:")
        # Filter out already selected images
        available_images = [f for f in all_images if os.path.join(folder, f) not in selected_paths]
        if not available_images:
            print("All available images have been selected.")
            break

        for i, f in enumerate(available_images):
            print(f"  {i+1}: {f}")

        if selected_paths:
            print(f"\nSelected: {', '.join([Path(p).name for p in selected_paths])}")

        selection = get_user_input("Select an image by number to add it, or press Enter to finish: ")
        if not selection:
            break # Done selecting

        try:
            index = int(selection) - 1
            if 0 <= index < len(available_images):
                selected_path = os.path.join(folder, available_images[index])
                selected_paths.append(selected_path)
            else:
                print("Invalid number.")
        except ValueError:
            print("Please enter a number.")
    return selected_paths

def extract_frame(video_path, output_dir):
    """Extracts a frame from a video using ffmpeg."""
    try:
        # Get video duration to validate frame number
        ffprobe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        duration_str = subprocess.check_output(ffprobe_cmd).decode('utf-8').strip()
        duration = float(duration_str)
        
        timestamp = get_user_input(f"Enter timestamp (in seconds, 0 to {duration:.2f}) to extract frame: ")
        frame_time = float(timestamp)

        if not (0 <= frame_time <= duration):
            print("Invalid timestamp.")
            return

        output_filename = f"{Path(video_path).stem}_frame_at_{frame_time:.2f}s.png"
        output_path = os.path.join(output_dir, output_filename)
        
        ffmpeg_cmd = [
            "ffmpeg", "-i", video_path, "-ss", str(frame_time),
            "-vframes", "1", output_path
        ]
        
        print(f"Running ffmpeg: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Successfully extracted frame to: {output_path}")

    except FileNotFoundError:
        print("ERROR: ffmpeg or ffprobe not found. Please ensure they are installed and in your PATH.")
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"ERROR: Frame extraction failed. {e}")


def run_generation(args):
    """Runs a generation script and returns True on success."""
    script = I2V_SCRIPT
    command = ["python", script] + args
    print("---" * 10)
    print(f'Running command: {" ".join(command)}')
    try:
        subprocess.run(command, check=True)
        print("---" * 10)
        return True
    except FileNotFoundError:
        print(f"ERROR: '{script}' not found.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Generation script failed with exit code {e.returncode}.")
    print("---" * 10)
    return False


def parse_prompts(filepath):
    """
    Parses storyboard files, treating each line as a separate prompt.
    """
    scenes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Use regex to find the timestamp and the prompt text
            match = re.match(r'\[(.*?)\]\s*(.*)', line)
            if match:
                timestamp, prompt_text = match.groups()
                scenes.append({
                    "ts": timestamp.strip(),
                    "prompt": prompt_text.strip()
                })
    return scenes

# --- Main Workflow ---

def main():
    """Main function to drive the interactive generation."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Load base parameters from JSON
    try:
        with open(os.path.join(INPUT_DIR, "first-shot.json"), 'r') as f:
            base_params = json.load(f)
        print("Loaded base parameters from 'first-shot.json'.")
    except FileNotFoundError:
        print("ERROR: 'input/first-shot.json' not found. Exiting.")
        return

    # --- Step 1: SCENE GENERATION ---
    print("\n=== STEP 1: SCENE GENERATION ===")
    storyboard_file = os.path.join(INPUT_DIR, "Pull Me Under.prompts.txt")
    if os.path.exists(storyboard_file):
        scenes = parse_prompts(storyboard_file)

        for i, scene_data in enumerate(scenes):
            print(f"\n--- Preparing Scene {i+1}/{len(scenes)} (Timestamp: {scene_data['ts']}) ---")
            
            confirmed = False
            while not confirmed:
                print("Scene Prompt:")
                print(f"  {scene_data['prompt']}")
                
                # Interactive Image Selection
                selected_images = select_images_for_scene(INPUT_DIR)
                
                start_image = None
                if not selected_images:
                    print("\nNo images selected. This will be a text-to-video generation.")
                    print("A temporary black image will be used as a placeholder.")
                    # Create a black image based on resolution in params
                    width, height = map(int, base_params.get("resolution", "832x480").split('x'))
                    start_image = create_black_image(width, height, TEMP_BLACK_IMAGE)
                    if not start_image:
                        print("Could not create placeholder image. Skipping scene.")
                        break
                else:
                    start_image = selected_images[0]
                    if len(selected_images) > 1:
                        print("\nNOTE: Multiple images selected.")
                        print(f"Using '{Path(start_image).name}' as the primary input image.")
                        print("The other selected images are noted but not used by the current generation script.")

                
                print("\nFinal parameters for this scene:")
                if len(selected_images) > 0:
                    print(f"  Input Images: {', '.join([Path(p).name for p in selected_images])}")
                else:
                    print("  Input Images: None (Text-to-Video)")
                print(f"  Prompt: {scene_data['prompt']}")
                
                user_choice = get_confirmation("Generate this scene? (y/n/r): ")
                if user_choice == 'n':
                    print("Skipping scene.")
                    break

                output_file = os.path.join(OUTPUT_DIR, f"scene_A_{i+1:03d}.mp4")
                gen_args = [
                    "--input-image", start_image,
                    "--output-file", output_file,
                    "--prompt", scene_data['prompt'],
                    "--resolution", base_params.get("resolution", "832x480"),
                    "--steps", str(base_params.get("num_inference_steps", "30")),
                    "--frames", str(base_params.get("video_length", "65")),
                    # Add other params from first-shot.json as needed
                    "--guidance-scale", str(base_params.get("guidance_scale", "5.0")),
                ]

                if run_generation(gen_args):
                    print(f"Scene clip saved to: {output_file}")
                    regen_choice = get_confirmation("Accept this video? (y/n/r): ")
                    if regen_choice == 'y':
                        confirmed = True
                        extract_f_choice = get_user_input("Extract a frame for future reference? (y/n): ").lower()
                        if extract_f_choice == 'y':
                            extract_frame(output_file, INPUT_DIR)
                    elif regen_choice == 'n':
                        break # Skip scene
                else:
                    print("Generation failed.")
                    if get_user_input("Try again? (y/n): ").lower() != 'y':
                        break
    
    print("\n=== SCRIPT FINISHED ===")
    print(f"All generated clips are in: {OUTPUT_DIR}")
    # Clean up the temporary black image if it exists
    if os.path.exists(TEMP_BLACK_IMAGE):
        os.remove(TEMP_BLACK_IMAGE)

if __name__ == "__main__":
    main()

