import os
import json
import re
from pathlib import Path
from PIL import Image
import torch

# --- Project-specific Imports ---
# These imports hook into the actual project code.
import wgp
from wan.utils.utils import get_video_info, convert_image
from mmgp import offload, profile_type

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output/batch_interactive"

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

def select_injection_mode():
    """Lets the user choose an image injection mode."""
    modes = {
        "1": {"name": "Start Image (i2v)", "type": "S"},
        "2": {"name": "Reference Images (for style/subject)", "type": "I"},
        "3": {"name": "Start Image + Reference Images", "type": "SI"},
        "4": {"name": "Inject Landscape then People", "type": "KI"},
        "5": {"name": "Text-to-Video (No Images)", "type": ""}
    }
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
            match = re.match(r'\['(.*?)'\]\s*(.*)', line)
            if match:
                timestamp, prompt_text = match.groups()
                scenes.append({"ts": timestamp.strip(), "prompt": prompt_text.strip()})
    return scenes

def initialize_model(params):
    """Loads and initializes the model based on wgp.py logic."""
    print("\n--- Initializing Model ---")
    
    wgp.load_models_config()

    model_type = params.get('model_type')
    if not model_type:
        raise ValueError("model_type not found in parameters JSON.")

    model_def = wgp.get_model_def(model_type)
    if not model_def:
        raise ValueError(f"Could not find model definition for '{model_type}'.")

    print(f"Loading model: {model_def.get('name')}")

    wgp.wan_model = wgp.get_wan_model(model_type, params)
    if wgp.wan_model is None:
        raise RuntimeError("wgp.get_wan_model failed to return a model object.")
        
    wgp.wan_model.to(wgp.processing_device)
    
    pipe = wgp.get_pipe(wgp.wan_model)
    wgp.offloadobj = offload.profile(
        pipe,
        profile_no=wgp.server_config.get("profile", profile_type.LowRAM_LowVRAM),
        compile="",
        quantizeTransformer=False 
    )
    
    print("Model initialized successfully.")
    return wgp.wan_model

# --- Main Workflow ---

def main():
    """Main function to drive the interactive generation."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    try:
        with open(os.path.join(INPUT_DIR, "first-shot.json"), 'r') as f:
            base_params = json.load(f)
        print("Loaded base parameters from 'first-shot.json'.")
    except FileNotFoundError:
        print("ERROR: 'input/first-shot.json' not found. Exiting.")
        return

    try:
        wan_model = initialize_model(base_params)
    except Exception as e:
        print(f"\nFATAL: Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    storyboard_file = os.path.join(INPUT_DIR, "Pull Me Under.prompts.txt")
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
            
            mode_selection = select_injection_mode()
            
            selected_image_paths = []
            if mode_selection['type']:
                selected_image_paths = select_images_for_scene(INPUT_DIR)

            gen_kwargs = base_params.copy()
            gen_kwargs['input_prompt'] = scene_data['prompt']
            gen_kwargs['image_prompt_type'] = mode_selection['type']
            # The generate function saves the file itself based on other params,
            # so we don't need to specify an output file here.
            
            pil_images = [convert_image(Image.open(p)) for p in selected_image_paths]

            if 'S' in mode_selection['type'] and pil_images:
                gen_kwargs['image_start'] = pil_images[0]
                if len(pil_images) > 1:
                    gen_kwargs['image_refs'] = pil_images[1:]
            elif 'I' in mode_selection['type'] and pil_images:
                gen_kwargs['image_refs'] = pil_images
            
            print("\nFinal parameters for this scene:")
            print(f"  Mode: {mode_selection['name']}")
            if selected_image_paths:
                print(f"  Images: {', '.join([Path(p).name for p in selected_image_paths])}")
            
            user_choice = get_confirmation("Generate this scene? (y/n/r): ")
            if user_choice == 'n': break
            if user_choice == 'r': continue

            try:
                # The generate function returns the path to the output video
                output_video_path = wan_model.generate(**gen_kwargs)
                print(f"Scene clip saved to: {output_video_path}")
                
                regen_choice = get_confirmation("Accept this video? (y/n/r): ")
                if regen_choice == 'y':
                    confirmed = True
                    # Frame extraction logic would go here
                elif regen_choice == 'n':
                    break 
            except Exception as e:
                print(f"ERROR: Generation failed: {e}")
                import traceback
                traceback.print_exc()
                if get_user_input("Try again? (y/n): ").lower() != 'y':
                    break
    
    print("\n=== SCRIPT FINISHED ===")
    print(f"All generated clips are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()