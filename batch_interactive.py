import os
import json
import subprocess
import re

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output/batch_interactive"
I2V_SCRIPT = "i2v_inference.py"

# --- Helper Functions ---

def run_inference(args):
    """Runs the i2v_inference.py script with the given arguments."""
    command = ["python", I2V_SCRIPT] + args
    print("---" * 10)
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"ERROR: '{I2V_SCRIPT}' not found. Make sure you are in the correct directory.")
        return
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Inference script failed with exit code {e.returncode}.")
    print("---" * 10)


def parse_character_prompts(filepath):
    """Parses the character prompts file."""
    characters = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    char_sections = content.split('# Casting Prompts: ')
    for section in char_sections:
        if not section.strip():
            continue
        char_name = section.split('\n')[0].strip()
        
        # Find the "In Character" prompt
        in_character_match = re.search(r'## Prompt 1: In Character\n(.*?)(?=\n##|$)', section, re.S)
        if in_character_match:
            prompt = in_character_match.group(1).strip()
            characters[char_name] = {
                "prompt": prompt,
                "image": os.path.join(INPUT_DIR, f"{char_name.replace('The ','')}.png")
            }
    return characters

def parse_storyboard(filepath):
    """Parses the main storyboard file."""
    scenes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\[([0-9:. ]+)\]\s*(.*)', line)
            if match:
                scenes.append({
                    "timestamp": match.group(1),
                    "prompt": match.group(2)
                })
    return scenes


def main():
    """Main function to drive the interactive generation."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Character Generation ---
    print("\n=== STEP 1: CHARACTER GENERATION ===")
    char_prompts_file = os.path.join(INPUT_DIR, "Pull Me Under.prompts.characters.txt")
    if os.path.exists(char_prompts_file):
        characters = parse_character_prompts(char_prompts_file)
        print(f"Found {len(characters)} characters to generate.")
        
        for name, data in characters.items():
            print(f"\nGenerating character: {name}")
            output_file = os.path.join(OUTPUT_DIR, f"char_{name.replace(' ', '_')}.mp4")
            
            if not os.path.exists(data['image']):
                print(f"WARNING: Character image not found at {data['image']}. Skipping generation.")
                continue

            inference_args = [
                "--input-image", data['image'],
                "--output-file", output_file,
                "--prompt", data['prompt'],
                "--resolution", "512x512", # Or another standard size for portraits
                "--steps", "30",
                "--frames", "17", # A short clip
            ]
            run_inference(inference_args)
            print(f"Character clip saved to: {output_file}")

    else:
        print("Character prompts file not found. Skipping character generation.")
    
    input("\nPress Enter to continue to the next step...")

    # --- Step 2: A-Roll (Main Scenes) Generation ---
    print("\n=== STEP 2: A-ROLL (MAIN SCENES) GENERATION ===")
    storyboard_file = os.path.join(INPUT_DIR, "Pull Me Under.prompts.txt")
    if os.path.exists(storyboard_file):
        scenes = parse_storyboard(storyboard_file)
        print(f"Found {len(scenes)} main scenes to generate.")

        for i, scene in enumerate(scenes):
            print(f"\nGenerating scene {i+1}/{len(scenes)} (Timestamp: {scene['timestamp']})")
            output_file = os.path.join(OUTPUT_DIR, f"scene_A_{i+1:03d}.mp4")
            
            # This is a simplified logic. We'd need to decide which character image to use.
            # For now, we'll just use the prompt.
            inference_args = [
                # We need a placeholder image for i2v, let's use the woman for now
                "--input-image", os.path.join(INPUT_DIR, "The woman.png"),
                "--output-file", output_file,
                "--prompt", scene['prompt'],
                "--resolution", "832x480",
                "--steps", "40",
                "--frames", "65", # ~4 seconds at 16fps
            ]
            run_inference(inference_args)
            print(f"Scene clip saved to: {output_file}")

    else:
        print("Storyboard file not found. Skipping A-Roll generation.")

    input("\nPress Enter to continue to the next step...")

    # --- Step 3: B-Roll (Atmospheric Scenes) Generation ---
    print("\n=== STEP 3: B-ROLL (ATMOSPHERIC SCENES) GENERATION ===")
    broll_file = os.path.join(INPUT_DIR, "Pull Me Under.prompts.extended.txt")
    if os.path.exists(broll_file):
        broll_scenes = parse_storyboard(broll_file) # Re-using the parser
        print(f"Found {len(broll_scenes)} B-roll scenes to generate.")

        for i, scene in enumerate(broll_scenes):
            print(f"\nGenerating B-roll scene {i+1}/{len(broll_scenes)}")
            output_file = os.path.join(OUTPUT_DIR, f"scene_B_{i+1:03d}.mp4")
            
            inference_args = [
                 # We need a placeholder image for i2v, let's use the woman for now
                "--input-image", os.path.join(INPUT_DIR, "The woman.png"),
                "--output-file", output_file,
                "--prompt", scene['prompt'],
                "--resolution", "832x480",
                "--steps", "30",
                "--frames", "33", # ~2 seconds
            ]
            run_inference(inference_args)
            print(f"B-roll clip saved to: {output_file}")
    else:
        print("B-roll file not found. Skipping B-roll generation.")

    print("\n=== SCRIPT FINISHED ===")
    print(f"All generated clips are in: {OUTPUT_DIR}")
    print("Next step would be to assemble these clips in a video editor using the timestamps as a guide.")


if __name__ == "__main__":
    main()
