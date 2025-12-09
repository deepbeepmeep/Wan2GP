import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
import warnings

try:
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    warnings.warn("safetensors not available. Install with: pip install safetensors")

class LoRAExtractor:
    """
    Extract LoRA tensors from the difference between original and fine-tuned models.
    
    LoRA (Low-Rank Adaptation) decomposes weight updates as ΔW = B @ A where:
    - A (lora_down): [rank, input_dim] matrix (saved as diffusion_model.param_name.lora_down.weight)
    - B (lora_up): [output_dim, rank] matrix (saved as diffusion_model.param_name.lora_up.weight)
    
    The decomposition uses SVD: ΔW = U @ S @ V^T ≈ (U @ S) @ V^T where:
    - lora_up = U @ S (contains all singular values)
    - lora_down = V^T (orthogonal matrix)
    
    Parameter handling based on name AND dimension:
    - 2D weight tensors: LoRA decomposition (.lora_down.weight, .lora_up.weight) 
    - Any bias tensors: direct difference (.diff_b)
    - Other weight tensors (1D, 3D, 4D): full difference (.diff)
    
    Progress tracking and test mode are available for format validation and debugging.
    """
    
    def __init__(self, rank: int = 128, threshold: float = 1e-6, test_mode: bool = False, show_reconstruction_errors: bool = False):
        """
        Initialize LoRA extractor.
        
        Args:
            rank: Target rank for LoRA decomposition (default: 128)
            threshold: Minimum singular value threshold for decomposition
            test_mode: If True, creates zero tensors without computation for format testing
            show_reconstruction_errors: If True, calculates and displays reconstruction error for each LoRA pair
        """
        self.rank = rank
        self.threshold = threshold
        self.test_mode = test_mode
        self.show_reconstruction_errors = show_reconstruction_errors
    
    def extract_lora_from_state_dicts(
        self, 
        original_state_dict: Dict[str, torch.Tensor], 
        finetuned_state_dict: Dict[str, torch.Tensor],
        device: str = 'cpu',
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract LoRA tensors for all matching parameters between two state dictionaries.
        
        Args:
            original_state_dict: State dict of the original model
            finetuned_state_dict: State dict of the fine-tuned model
            device: Device to perform computations on
            show_progress: Whether to display progress information
            
        Returns:
            Dictionary mapping parameter names to their LoRA components:
            - For 2D weight tensors: 'diffusion_model.layer.lora_down.weight', 'diffusion_model.layer.lora_up.weight'
            - For any bias tensors: 'diffusion_model.layer.diff_b'  
            - For other weight tensors (1D, 3D, 4D): 'diffusion_model.layer.diff'
        """
        lora_tensors = {}
        
        # Find common parameters and sort alphabetically for consistent processing order
        common_keys = sorted(set(original_state_dict.keys()) & set(finetuned_state_dict.keys()))
        total_params = len(common_keys)
        processed_params = 0
        extracted_components = 0
        
        if show_progress:
            print(f"Starting LoRA extraction for {total_params} parameters on {device}...")
        
        # Pre-move threshold to device for faster comparisons
        threshold_tensor = torch.tensor(self.threshold, device=device)
        
        for param_name in common_keys:
            if show_progress:
                processed_params += 1
                progress_pct = (processed_params / total_params) * 100
                print(f"[{processed_params:4d}/{total_params}] ({progress_pct:5.1f}%) Processing: {param_name}")
            
            # Move tensors to device once
            original_tensor = original_state_dict[param_name]
            finetuned_tensor = finetuned_state_dict[param_name]
            
            # Check if tensors have the same shape before moving to device
            if original_tensor.shape != finetuned_tensor.shape:
                if show_progress:
                    print(f"    → Shape mismatch: {original_tensor.shape} vs {finetuned_tensor.shape}. Skipping.")
                continue
            
            # Move to device and compute difference in one go for efficiency (skip in test mode)
            if not self.test_mode:
                if original_tensor.device != torch.device(device):
                    original_tensor = original_tensor.to(device, non_blocking=True)
                if finetuned_tensor.device != torch.device(device):
                    finetuned_tensor = finetuned_tensor.to(device, non_blocking=True)
                
                # Compute difference on device
                delta_tensor = finetuned_tensor - original_tensor
                
                # Fast GPU-based threshold check
                max_abs_diff = torch.max(torch.abs(delta_tensor))
                if max_abs_diff <= threshold_tensor:
                    if show_progress:
                        print(f"    → No significant changes detected (max diff: {max_abs_diff:.2e}), skipping")
                    continue
            else:
                # Test mode - create dummy delta tensor with original shape and dtype
                delta_tensor = torch.zeros_like(original_tensor)
                if device != 'cpu':
                    delta_tensor = delta_tensor.to(device)
            
            # Extract LoRA components based on tensor dimensionality
            extracted_tensors = self._extract_lora_components(delta_tensor, param_name)
            
            if extracted_tensors:
                lora_tensors.update(extracted_tensors)
                extracted_components += len(extracted_tensors)
                if show_progress:
                    # Show meaningful component names instead of just 'weight'
                    component_names = []
                    for key in extracted_tensors.keys():
                        if key.endswith('.lora_down.weight'):
                            component_names.append('lora_down')
                        elif key.endswith('.lora_up.weight'):
                            component_names.append('lora_up')
                        elif key.endswith('.diff_b'):
                            component_names.append('diff_b')
                        elif key.endswith('.diff'):
                            component_names.append('diff')
                        else:
                            component_names.append(key.split('.')[-1])
                    print(f"    → Extracted {len(extracted_tensors)} components: {component_names}")
        
        if show_progress:
            print(f"\nExtraction completed!")
            print(f"Processed: {processed_params}/{total_params} parameters")
            print(f"Extracted: {extracted_components} LoRA components")
            print(f"LoRA rank: {self.rank}")
            
            # Summary by type
            lora_down_count = sum(1 for k in lora_tensors.keys() if k.endswith('.lora_down.weight'))
            lora_up_count = sum(1 for k in lora_tensors.keys() if k.endswith('.lora_up.weight'))
            diff_b_count = sum(1 for k in lora_tensors.keys() if k.endswith('.diff_b'))
            diff_count = sum(1 for k in lora_tensors.keys() if k.endswith('.diff'))
            
            print(f"Summary: {lora_down_count} lora_down, {lora_up_count} lora_up, {diff_b_count} diff_b, {diff_count} diff")
        
        return lora_tensors
    
    def _extract_lora_components(
        self, 
        delta_tensor: torch.Tensor, 
        param_name: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract LoRA components from a delta tensor.
        
        Args:
            delta_tensor: Difference between fine-tuned and original tensor
            param_name: Name of the parameter (for generating output keys)
            
        Returns:
            Dictionary with modified parameter names as keys and tensors as values
        """
        # Determine if this is a weight or bias parameter from the original name
        is_weight = 'weight' in param_name.lower()
        is_bias = 'bias' in param_name.lower()
        
        # Remove .weight or .bias suffix from parameter name
        base_name = param_name
        if base_name.endswith('.weight'):
            base_name = base_name[:-7]  # Remove '.weight'
        elif base_name.endswith('.bias'):
            base_name = base_name[:-5]  # Remove '.bias'
        
        # Add diffusion_model prefix
        base_name = f"diffusion_model.{base_name}"
        
        if self.test_mode:
            # Fast test mode - create zero tensors without computation
            if delta_tensor.dim() == 2 and is_weight:
                # 2D weight tensor -> LoRA decomposition
                output_dim, input_dim = delta_tensor.shape
                rank = min(self.rank, min(input_dim, output_dim))
                return {
                    f"{base_name}.lora_down.weight": torch.zeros(rank, input_dim, dtype=delta_tensor.dtype, device=delta_tensor.device),
                    f"{base_name}.lora_up.weight": torch.zeros(output_dim, rank, dtype=delta_tensor.dtype, device=delta_tensor.device)
                }
            elif is_bias:
                # Any bias tensor (1D, 2D, etc.) -> .diff_b
                return {f"{base_name}.diff_b": torch.zeros_like(delta_tensor)}
            else:
                # Any weight tensor that's not 2D, or other tensors -> .diff
                return {f"{base_name}.diff": torch.zeros_like(delta_tensor)}
        
        # Normal mode - check dimensions AND parameter type
        if delta_tensor.dim() == 2 and is_weight:
            # 2D weight tensor (linear layer weight) - apply SVD decomposition
            return self._decompose_2d_tensor(delta_tensor, base_name)
        
        elif is_bias:
            # Any bias tensor (regardless of dimension) - save as .diff_b
            return {f"{base_name}.diff_b": delta_tensor.clone()}
        
        else:
            # Any other tensor (weight tensors that are 1D, 3D, 4D, or unknown tensors) - save as .diff
            return {f"{base_name}.diff": delta_tensor.clone()}
    
    def _decompose_2d_tensor(self, delta_tensor: torch.Tensor, base_name: str) -> Dict[str, torch.Tensor]:
        """
        Decompose a 2D tensor using SVD on GPU for maximum performance.
        
        Args:
            delta_tensor: 2D tensor to decompose (output_dim × input_dim)
            base_name: Base name for the parameter (already processed, with diffusion_model prefix)
            
        Returns:
            Dictionary with lora_down and lora_up tensors:
            - lora_down: [rank, input_dim] 
            - lora_up: [output_dim, rank]
        """
        # Store original dtype and device
        dtype = delta_tensor.dtype
        device = delta_tensor.device
        
        # Perform SVD in float32 for numerical stability, but keep on same device
        delta_float = delta_tensor.float() if delta_tensor.dtype != torch.float32 else delta_tensor
        U, S, Vt = torch.linalg.svd(delta_float, full_matrices=False)
        
        # Determine effective rank (number of significant singular values)
        # Use GPU-accelerated operations
        significant_mask = S > self.threshold
        effective_rank = min(self.rank, torch.sum(significant_mask).item())
        effective_rank = self.rank

        if effective_rank == 0:
            warnings.warn(f"No significant singular values found for {base_name}")
            effective_rank = 1
        
        # Create LoRA matrices with correct SVD decomposition
        # Standard approach: put all singular values in lora_up, leave lora_down as V^T
        # This ensures: lora_up @ lora_down = (U @ S) @ V^T = U @ S @ V^T = ΔW ✓
        
        lora_up = U[:, :effective_rank] * S[:effective_rank].unsqueeze(0)  # [output_dim, rank]
        lora_down = Vt[:effective_rank, :]                                 # [rank, input_dim] 
        
        # Convert back to original dtype (keeping on same device)
        lora_up = lora_up.to(dtype)
        lora_down = lora_down.to(dtype)
        
        # Calculate and display reconstruction error if requested
        if self.show_reconstruction_errors:
            with torch.no_grad():
                # Reconstruct the original delta tensor
                reconstructed = lora_up @ lora_down
                
                # Calculate various error metrics
                mse_error = torch.mean((delta_tensor - reconstructed) ** 2).item()
                max_error = torch.max(torch.abs(delta_tensor - reconstructed)).item()
                
                # Relative error
                original_norm = torch.norm(delta_tensor).item()
                relative_error = (torch.norm(delta_tensor - reconstructed).item() / original_norm * 100) if original_norm > 0 else 0
                
                # Cosine similarity
                delta_flat = delta_tensor.flatten()
                reconstructed_flat = reconstructed.flatten()
                if torch.norm(delta_flat) > 0 and torch.norm(reconstructed_flat) > 0:
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        delta_flat.unsqueeze(0), 
                        reconstructed_flat.unsqueeze(0)
                    ).item()
                else:
                    cosine_sim = 0.0
                
                # Extract parameter name for display (remove diffusion_model prefix)
                display_name = base_name[16:] if base_name.startswith('diffusion_model.') else base_name
                
                print(f"    LoRA Error [{display_name}]: MSE={mse_error:.2e}, Max={max_error:.2e}, Rel={relative_error:.2f}%, Cos={cosine_sim:.4f}, Rank={effective_rank}")
        
        return {
            f"{base_name}.lora_down.weight": lora_down,
            f"{base_name}.lora_up.weight": lora_up
        }
    
    def verify_reconstruction(
        self, 
        lora_tensors: Dict[str, torch.Tensor], 
        original_deltas: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Verify the quality of LoRA reconstruction for 2D tensors.
        
        Args:
            lora_tensors: Dictionary with LoRA tensors (flat structure with diffusion_model prefix)
            original_deltas: Dictionary with original delta tensors (without prefix)
            
        Returns:
            Dictionary mapping parameter names to reconstruction errors
        """
        reconstruction_errors = {}
        
        # Group LoRA components by base parameter name
        lora_pairs = {}
        for key, tensor in lora_tensors.items():
            if key.endswith('.lora_down.weight'):
                base_name = key[:-18]  # Remove '.lora_down.weight'
                # Remove diffusion_model prefix for matching with original_deltas
                if base_name.startswith('diffusion_model.'):
                    original_key = base_name[16:]  # Remove 'diffusion_model.'
                else:
                    original_key = base_name
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {'original_key': original_key}
                lora_pairs[base_name]['lora_down'] = tensor
            elif key.endswith('.lora_up.weight'):
                base_name = key[:-16]  # Remove '.lora_up.weight'
                # Remove diffusion_model prefix for matching with original_deltas
                if base_name.startswith('diffusion_model.'):
                    original_key = base_name[16:]  # Remove 'diffusion_model.'
                else:
                    original_key = base_name
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {'original_key': original_key}
                lora_pairs[base_name]['lora_up'] = tensor
        
        # Verify reconstruction for each complete LoRA pair
        for base_name, components in lora_pairs.items():
            if 'lora_down' in components and 'lora_up' in components and 'original_key' in components:
                original_key = components['original_key']
                if original_key in original_deltas:
                    lora_down = components['lora_down']
                    lora_up = components['lora_up']
                    original_delta = original_deltas[original_key]
                    
                    # Get effective rank from the actual tensor dimensions
                    effective_rank = min(lora_up.shape[1], lora_down.shape[0])
                    
                    # Reconstruct: ΔW = lora_up @ lora_down (no additional scaling needed since it's built into lora_up)
                    reconstructed = lora_up @ lora_down
                    
                    # Compute reconstruction error
                    mse_error = torch.mean((original_delta - reconstructed) ** 2).item()
                    reconstruction_errors[base_name] = mse_error
        
        return reconstruction_errors

def compute_reconstruction_errors(
    original_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor, 
    target_tensor: torch.Tensor
) -> Dict[str, float]:
    """
    Compute various error metrics between original, reconstructed, and target tensors.
    
    Args:
        original_tensor: Original tensor before fine-tuning
        reconstructed_tensor: Reconstructed tensor from LoRA (original + LoRA_reconstruction)  
        target_tensor: Target tensor (fine-tuned)
        
    Returns:
        Dictionary with error metrics
    """
    # Ensure all tensors are on the same device and have the same shape
    device = original_tensor.device
    reconstructed_tensor = reconstructed_tensor.to(device)
    target_tensor = target_tensor.to(device)
    
    # Compute differences
    delta_original = target_tensor - original_tensor  # True fine-tuning difference
    delta_reconstructed = reconstructed_tensor - original_tensor  # LoRA reconstructed difference
    reconstruction_error = target_tensor - reconstructed_tensor  # Final reconstruction error
    
    # Compute various error metrics
    errors = {}
    
    # Mean Squared Error (MSE)
    errors['mse_delta'] = torch.mean((delta_original - delta_reconstructed) ** 2).item()
    errors['mse_final'] = torch.mean(reconstruction_error ** 2).item()
    
    # Mean Absolute Error (MAE)  
    errors['mae_delta'] = torch.mean(torch.abs(delta_original - delta_reconstructed)).item()
    errors['mae_final'] = torch.mean(torch.abs(reconstruction_error)).item()
    
    # Relative errors (as percentages)
    original_norm = torch.norm(original_tensor).item()
    target_norm = torch.norm(target_tensor).item()
    delta_norm = torch.norm(delta_original).item()
    
    if original_norm > 0:
        errors['relative_error_original'] = (torch.norm(reconstruction_error).item() / original_norm) * 100
    if target_norm > 0:
        errors['relative_error_target'] = (torch.norm(reconstruction_error).item() / target_norm) * 100  
    if delta_norm > 0:
        errors['relative_error_delta'] = (torch.norm(delta_original - delta_reconstructed).item() / delta_norm) * 100
        
    # Cosine similarity (higher is better, 1.0 = perfect)
    delta_flat = delta_original.flatten()
    reconstructed_flat = delta_reconstructed.flatten()
    
    if torch.norm(delta_flat) > 0 and torch.norm(reconstructed_flat) > 0:
        cosine_sim = torch.nn.functional.cosine_similarity(
            delta_flat.unsqueeze(0), 
            reconstructed_flat.unsqueeze(0)
        ).item()
        errors['cosine_similarity'] = cosine_sim
    else:
        errors['cosine_similarity'] = 0.0
        
    # Signal-to-noise ratio (SNR) in dB
    if errors['mse_final'] > 0:
        signal_power = torch.mean(target_tensor ** 2).item()
        errors['snr_db'] = 10 * torch.log10(signal_power / errors['mse_final']).item()
    else:
        errors['snr_db'] = float('inf')
    
    return errors

# Example usage and utility functions
def load_and_extract_lora(
    original_model_path: str,
    finetuned_model_path: str,
    rank: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    show_progress: bool = True,
    test_mode: bool = False,
    show_reconstruction_errors: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load models and extract LoRA tensors with GPU acceleration.
    
    Args:
        original_model_path: Path to original model state dict
        finetuned_model_path: Path to fine-tuned model state dict
        rank: Target LoRA rank (default: 128)
        device: Device for computation (defaults to GPU if available)
        show_progress: Whether to display progress information
        test_mode: If True, creates zero tensors without computation for format testing
        show_reconstruction_errors: If True, calculates and displays reconstruction error for each LoRA pair
        
    Returns:
        Dictionary of LoRA tensors with modified parameter names as keys
    """
    # Load state dictionaries directly to CPU first (safetensors loads to CPU by default)
    if show_progress:
        print(f"Loading original model from: {original_model_path}")
    original_state_dict = torch.load(original_model_path, map_location='cpu')
    
    if show_progress:
        print(f"Loading fine-tuned model from: {finetuned_model_path}")
    finetuned_state_dict = torch.load(finetuned_model_path, map_location='cpu')
    
    # Handle nested state dicts (if wrapped in 'model' key or similar)
    if 'state_dict' in original_state_dict:
        original_state_dict = original_state_dict['state_dict']
    if 'state_dict' in finetuned_state_dict:
        finetuned_state_dict = finetuned_state_dict['state_dict']
    
    # Extract LoRA tensors with GPU acceleration
    extractor = LoRAExtractor(rank=rank, test_mode=test_mode, show_reconstruction_errors=show_reconstruction_errors)
    lora_tensors = extractor.extract_lora_from_state_dicts(
        original_state_dict, 
        finetuned_state_dict, 
        device=device,
        show_progress=show_progress
    )
    
    return lora_tensors

def save_lora_tensors(lora_tensors: Dict[str, torch.Tensor], save_path: str):
    """Save extracted LoRA tensors to disk."""
    torch.save(lora_tensors, save_path)
    print(f"LoRA tensors saved to {save_path}")

def save_lora_safetensors(lora_tensors: Dict[str, torch.Tensor], save_path: str, rank: int = None):
    """Save extracted LoRA tensors as safetensors format with metadata."""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not available. Install with: pip install safetensors")
    
    # Ensure all tensors are contiguous for safetensors
    contiguous_tensors = {k: v.contiguous() if v.is_floating_point() else v.contiguous() 
                         for k, v in lora_tensors.items()}
    
    # Add rank as metadata if provided
    metadata = {}
    if rank is not None:
        metadata["rank"] = str(rank)
    
    save_safetensors(contiguous_tensors, save_path, metadata=metadata if metadata else None)
    print(f"LoRA tensors saved as safetensors to {save_path}")
    if metadata:
        print(f"Metadata: {metadata}")

def analyze_lora_tensors(lora_tensors: Dict[str, torch.Tensor]):
    """Analyze the extracted LoRA tensors."""
    print(f"Extracted LoRA tensors ({len(lora_tensors)} components):")
    
    # Group by type for better organization
    lora_down_tensors = {k: v for k, v in lora_tensors.items() if k.endswith('.lora_down.weight')}
    lora_up_tensors = {k: v for k, v in lora_tensors.items() if k.endswith('.lora_up.weight')}
    diff_b_tensors = {k: v for k, v in lora_tensors.items() if k.endswith('.diff_b')}
    diff_tensors = {k: v for k, v in lora_tensors.items() if k.endswith('.diff')}
    
    if lora_down_tensors:
        print(f"\nLinear LoRA down matrices ({len(lora_down_tensors)}):")
        for name, tensor in lora_down_tensors.items():
            print(f"  {name}: {tensor.shape}")
    
    if lora_up_tensors:
        print(f"\nLinear LoRA up matrices ({len(lora_up_tensors)}):")
        for name, tensor in lora_up_tensors.items():
            print(f"  {name}: {tensor.shape}")
    
    if diff_b_tensors:
        print(f"\nBias differences ({len(diff_b_tensors)}):")
        for name, tensor in diff_b_tensors.items():
            print(f"  {name}: {tensor.shape}")
    
    if diff_tensors:
        print(f"\nFull weight differences ({len(diff_tensors)}):")
        print("  (Includes conv, modulation, and other multi-dimensional tensors)")
        for name, tensor in diff_tensors.items():
            print(f"  {name}: {tensor.shape}")

# Example usage
if __name__ == "__main__":


    from safetensors.torch import load_file as load_safetensors
    
    # Load original and fine-tuned models from safetensors files
    
    original_state_dict = load_safetensors("ckpts/hunyuan_video_1.5_i2v_480_bf16.safetensors")
    finetuned_state_dict = load_safetensors("ckpts/hunyuan_video_1.5_i2v_480_step_distilled_bf16.safetensors")

    # original_state_dict = load_safetensors("ckpts/flux1-dev_bf16.safetensors")
    # finetuned_state_dict = load_safetensors("ckpts/flux1-schnell_bf16.safetensors")

    print(f"Loaded original model with {len(original_state_dict)} parameters")
    print(f"Loaded fine-tuned model with {len(finetuned_state_dict)} parameters")
    
    # extractor_test = LoRAExtractor(test_mode=True)

    extractor_test = LoRAExtractor(show_reconstruction_errors=True, rank=32)
    
    lora_tensors_test = extractor_test.extract_lora_from_state_dicts(
        original_state_dict, 
        finetuned_state_dict,
        device='cuda',
        show_progress=True
    )
    
    print("\nTest mode tensor keys (first 10):")
    for i, key in enumerate(sorted(lora_tensors_test.keys())):
        if i < 10:
            print(f"  {key}: {lora_tensors_test[key].shape}")
        elif i == 10:
            print(f"  ... and {len(lora_tensors_test) - 10} more")
            break
    
    # Always save as extracted_lora.safetensors for easier testing
    save_lora_safetensors(lora_tensors_test, "extracted_lora.safetensors")
    
