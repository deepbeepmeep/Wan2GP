import os
import json
import re
import subprocess
import sys
from pathlib import Path
from PIL import Image
import argparse
import tempfile
import time

# --- Configuration ---
VENV_PYTHON = os.path.join("venv", "Scripts", "python.exe")

# --- Helper Functions ---

def get_user_input(prompt, default=None):
    """Gets input from the user, with an optional default value."""
    response = input(f"{prompt} ").strip()
    return response if response else default

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
            break

        try:
            index = int(selection) - 1
            if 0 <= index < len(available_images):
                selected_paths.append(os.path.join(folder, available_images[index]))
            else:
                print("Invalid number.")
        except ValueError:
            print("Please enter a number.")
    return selected_paths

def select_injection_mode(non_interactive=False):
    """Lets the user choose an image injection mode."""
    modes = {
        "1": {"name": "Start Image (i2v)", "type": "S"},
        "2": {"name": "Reference Images (for style/subject)", "type": "I"},
        "3": {"name": "Start Image + Reference Images", "type": "SI"},
        "4": {"name": "Inject Landscape then People", "type": "KI"},
        "5": {"name": "Text-to-Video (No Images)", "type": ""}
    }
    if non_interactive:
        return modes["5"]
        
    print("\nSelect an image injection mode:")
    for key, value in modes.items():
        print(f"  {key}: {value['name']}")
    
    while True:
        choice = get_user_input("Enter mode number: ", default="5")
        if choice in modes:
            return modes[choice]
        print("Invalid selection.")

def parse_prompts(filepath):
    """Parses storyboard files, treating each line as a separate prompt."""
    scenes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\[(.*?)\]\s+(.*)', line)
            if match:
                timestamp, prompt_text = match.groups()
                scenes.append({"ts": timestamp.strip(), "prompt": prompt_text.strip()})
    return scenes

def run_generation(params, output_dir):
    """Runs wgp.py with the given parameters using subprocess."""
    
    env = os.environ.copy()
    env["WGP_PARAMS"] = json.dumps(params)

    command = [
        VENV_PYTHON,
        "wgp.py",
    ]
    
    start_time = time.time()

    try:
        print("\n--- Running Generation ---")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', env=env)
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        if process.returncode != 0:
            print(f"\n--- Generation Failed (Exit Code: {process.returncode}) ---")
            return None

    except FileNotFoundError:
        print(f"ERROR: Could not find '{VENV_PYTHON}'. Make sure the virtual environment is set up correctly.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Find the newest file created after the generation started
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
    files = [f for f in files if os.path.isfile(f) and os.path.getmtime(f) > start_time]
    
    if files:
        return max(files, key=os.path.getmtime)
    return None


# --- Main Workflow ---

def main():
    """Main function to drive the interactive generation."""
    parser = argparse.ArgumentParser(description="Interactive video generation from a storyboard.")
    parser.add_argument("--input-dir", type=str, default="input", help="Directory for input files.")
    parser.add_argument("--output-dir", type=str, default="output/batch_interactive", help="Directory for output files.")
    parser.add_argument("--storyboard", type=str, default="Pull Me Under.prompts.txt", help="Storyboard file name.")
    parser.add_argument("--params", type=str, default="first-shot.json", help="Base parameters file name.")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    try:
        with open(os.path.join(input_dir, args.params), 'r') as f:
            base_params = json.load(f)
        print(f"Loaded base parameters from '{args.params}'.")
    except FileNotFoundError:
        print(f"ERROR: '{os.path.join(input_dir, args.params)}' not found. Exiting.")
        return

    storyboard_file = os.path.join(input_dir, args.storyboard)
    if not os.path.exists(storyboard_file):
        print(f"ERROR: Storyboard file not found at {storyboard_file}")
        return
        
    scenes = parse_prompts(storyboard_file)

    for i, scene_data in enumerate(scenes):
        print(f"\n--- Preparing Scene {i+1}/{len(scenes)} (Timestamp: {scene_data['ts']}) ---")
        
        confirmed = False
        while not confirmed:
            print("\nScene Prompt:")
            print(f"  {scene_data['prompt']}")
            
            mode_selection = select_injection_mode(args.non_interactive)
            
            selected_image_paths = []
            if mode_selection['type'] and not args.non_interactive:
                selected_image_paths = select_images_for_scene(input_dir)

            gen_kwargs = base_params.copy()
            gen_kwargs['prompt'] = scene_data['prompt']
            gen_kwargs['image_prompt_type'] = mode_selection['type']
            
            if selected_image_paths:
                if 'S' in mode_selection['type']:
                    gen_kwargs['image_start'] = selected_image_paths[0]
                    if len(selected_image_paths) > 1:
                        gen_kwargs['image_refs'] = selected_image_paths[1:]
                elif 'I' in mode_selection['type']:
                    gen_kwargs['image_refs'] = selected_image_paths
            
            print("\nFinal parameters for this scene:")
            print(f"  Mode: {mode_selection['name']}")
            if selected_image_paths:
                print(f"  Images: {', '.join([Path(p).name for p in selected_image_paths])}")
            
            if not args.non_interactive:
                user_choice = get_confirmation("Generate this scene? (y/n/r): ")
                if user_choice == 'n': break
                if user_choice == 'r': continue

            output_video_path = run_generation(gen_kwargs, output_dir)
            
            if output_video_path:
                print(f"Scene clip saved to: {output_video_path}")
                
                if not args.non_interactive:
                    regen_choice = get_confirmation("Accept this video? (y/n/r): ")
                    if regen_choice == 'y':
                        confirmed = True
                    elif regen_choice == 'n':
                        break 
                else:
                    confirmed = True
            else:
                if not args.non_interactive:
                    if get_user_input("Try again? (y/n): ").lower() != 'y':
                        break
                else:
                    break
    
    print("\n=== SCRIPT FINISHED ===")
    print(f"All generated clips are in: {output_dir}")

if __name__ == "__main__":
    main()
