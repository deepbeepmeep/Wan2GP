#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_dataset

# Add the current directory to path to import features_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features_utils import FeaturesUtils

def load_config(config_path):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load configuration
    config_path = "/home/weiminwang/audio/veo3/multimodal-generation/models/wan/modules/mmaudio/test_vae_config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found!")
        return
    
    config = load_config(config_path)
    
    # Check if VAE checkpoints are specified in config
    vae_config = config.get('audio_vae', {})
    tod_vae_ckpt = vae_config.get('tod_vae_ckpt')
    bigvgan_vocoder_ckpt = vae_config.get('bigvgan_vocoder_ckpt')
    mode = vae_config.get('mode', '16k')
    need_vae_encoder = vae_config.get('need_vae_encoder', True)
    
    if not tod_vae_ckpt:
        print("tod_vae_ckpt not specified in config!")
        print("Please update vae_config.yaml with the path to your VAE checkpoint")
        return
    
    # Create output directory
    output_dir = Path("rec_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading HuggingFace dataset...")
    # Load the NonverbalTTS dataset
    dataset = load_dataset("deepvk/NonverbalTTS")
    dev_split = dataset['dev']
    
    print(f"Dev split has {len(dev_split)} samples")
    
    # Select 10 random samples
    num_samples = min(10, len(dev_split))
    random_indices = random.sample(range(len(dev_split)), num_samples)
    
    print(f"Selected {num_samples} random samples: {random_indices}")
    
    # Initialize FeaturesUtils
    print("Initializing VAE...")
    try:
        features_utils = FeaturesUtils(
            tod_vae_ckpt=tod_vae_ckpt,
            bigvgan_vocoder_ckpt=bigvgan_vocoder_ckpt,
            mode=mode,
            need_vae_encoder=need_vae_encoder
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_utils = features_utils.to(device)
        features_utils.eval()
        
        print(f"VAE initialized on device: {device}")
        
    except Exception as e:
        print(f"Error initializing VAE: {e}")
        print("Make sure the checkpoint paths in vae_config.yaml are correct")
        return
    
    # Process each selected sample
    for i, idx in enumerate(random_indices):
        try:
            print(f"Processing sample {i+1}/{num_samples} (index {idx})...")
            
            # Get audio data
            audio_data = dev_split[idx]['audio']
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
            
            print(f"  Original audio shape: {audio_array.shape}, SR: {sampling_rate}")
            
            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(device)
            
            # Save original audio
            original_path = output_dir / f"{i}.wav"
            sf.write(original_path, audio_array, sampling_rate)
            
            # Encode audio
            print(f"  Encoding... {audio_tensor.shape}")
            latent_features = features_utils.wrapped_encode(audio_tensor)
            print(f"  Latent features shape: {latent_features.shape}")
            print(f" latent feature stats: {latent_features.mean().item()}, {latent_features.std().item()}, {latent_features.min().item()}, {latent_features.max().item()}")
            
            # Decode audio
            print("  Decoding...")
            reconstructed_audio = features_utils.wrapped_decode(latent_features)
            
            # Convert back to numpy
            reconstructed_np = reconstructed_audio.squeeze().cpu().numpy()
            print(f"  Reconstructed audio shape: {reconstructed_np.shape}")
            
            # Save reconstructed audio
            reconstructed_path = output_dir / f"{i}_rec.wav"
            sf.write(reconstructed_path, reconstructed_np, sampling_rate)
            
            print(f"  Saved: {original_path} and {reconstructed_path}")

            ## shape testing: 
            dummy = torch.randn((12, 195584)).float().to(device) # Example shape (B, F, T)
            latent_features = features_utils.wrapped_encode(dummy)

            print(f"Shape testing: {dummy.shape} -> {latent_features.shape}")

        except Exception as e:
            print(f"Error processing sample {i} (index {idx}): {e}")
            continue
    
    print(f"\nProcessing complete! Results saved in {output_dir}/")
    print("Files saved:")
    for i in range(num_samples):
        print(f"  {i}.wav - original audio")
        print(f"  {i}_rec.wav - reconstructed audio")

if __name__ == "__main__":
    main()