"""
TaskConfig - the main typed configuration object.

This combines all parameter groups and handles:
- Parsing from DB tasks
- Parsing from segment parameters
- Converting to WGP format
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import logging

from .base import ParamGroup
from .lora import LoRAConfig
from .vace import VACEConfig
from .generation import GenerationConfig
from .phase import PhaseConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig(ParamGroup):
    """
    Complete task configuration combining all parameter groups.
    
    This is the main entry point for typed parameter handling.
    """
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    vace: VACEConfig = field(default_factory=VACEConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    
    # Task metadata
    task_id: str = ""
    task_type: str = ""
    model: str = ""
    
    # Extra params that don't fit elsewhere (passthrough)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'TaskConfig':
        """
        Parse TaskConfig from a flat parameter dict.
        
        Use from_db_task() for DB tasks that need flattening first.
        """
        task_id = context.get('task_id', '')
        task_type = context.get('task_type', '')
        model = context.get('model', '') or params.get('model_name', '')
        
        # Parse each component
        generation = GenerationConfig.from_params(params, **context)
        vace = VACEConfig.from_params(params, **context)
        phase = PhaseConfig.from_params(params, **context)
        
        # Parse LoRA config
        lora_context = {**context, 'model': model}
        lora = LoRAConfig.from_params(params, **lora_context)
        
        # If phase config has LoRAs, merge them in
        if not phase.is_empty():
            lora_info = phase.get_lora_info()
            if lora_info.get('additional_loras'):
                phase_lora = LoRAConfig.from_params({
                    'additional_loras': lora_info['additional_loras'],
                    'lora_names': lora_info.get('lora_names', []),
                    'lora_multipliers': lora_info.get('lora_multipliers', []),
                }, **lora_context)
                lora = lora.merge(phase_lora)
        
        # Collect extra params (everything not handled by specific groups)
        handled_keys = {
            'prompt', 'negative_prompt', 'resolution', 'video_length',
            'num_inference_steps', 'steps', 'guidance_scale', 'seed',
            'flow_shift', 'sample_solver', 'image_start', 'image_end', 'image_refs',
            'embedded_guidance_scale', 'denoising_strength', 'denoise_strength',
            'video_guide', 'video_mask', 'video_prompt_type',
            'control_net_weight', 'control_net_weight2',
            'activated_loras', 'lora_names', 'loras_multipliers', 'lora_multipliers',
            'additional_loras', 'phase_config', 'model_name',
            # Image-to-image parameter aliases
            'image', 'image_url', 'strength',
        }
        extra_params = {k: v for k, v in params.items() if k not in handled_keys}
        
        return cls(
            generation=generation,
            lora=lora,
            vace=vace,
            phase=phase,
            task_id=task_id,
            task_type=task_type,
            model=model,
            extra_params=extra_params,
        )
    
    @classmethod
    def from_db_task(cls, db_params: Dict[str, Any], **context) -> 'TaskConfig':
        """
        Parse TaskConfig from DB task parameters.
        
        Handles nested orchestrator_payload/orchestrator_details with proper precedence.
        """
        # Flatten with precedence
        flat_params = ParamGroup.flatten_params(db_params, context.get('task_id', ''))
        return cls.from_params(flat_params, **context)
    
    @classmethod
    def from_segment_params(
        cls,
        segment_params: Dict[str, Any],
        orchestrator_payload: Dict[str, Any],
        individual_params: Optional[Dict[str, Any]] = None,
        **context
    ) -> 'TaskConfig':
        """
        Parse TaskConfig from segment parameters with proper precedence.
        
        Precedence: individual_params > segment_params > orchestrator_payload
        """
        # Build merged params with precedence
        merged = {}
        
        # Start with orchestrator_payload (lowest)
        if orchestrator_payload:
            merged.update(orchestrator_payload)
        
        # Then segment_params (medium)
        if segment_params:
            merged.update(segment_params)
        
        # Finally individual_params (highest)
        if individual_params:
            merged.update(individual_params)
        
        return cls.from_params(merged, **context)
    
    def to_wgp_format(self) -> Dict[str, Any]:
        """
        Convert to WGP-compatible parameter dict.
        
        This is the single point of conversion to WGP format.
        """
        result = {}
        
        # Add generation params
        result.update(self.generation.to_wgp_format())
        
        # Add LoRA params
        result.update(self.lora.to_wgp_format())
        
        # CRITICAL: Also include pending downloads so downstream can handle them
        if self.lora.has_pending_downloads():
            result['additional_loras'] = self.lora.get_pending_downloads()
        
        # Add VACE params
        result.update(self.vace.to_wgp_format())
        
        # Add phase config params
        if not self.phase.is_empty():
            phase_params = self.phase.to_wgp_format()
            
            # Don't duplicate LoRA params from phase (already handled above)
            for key in ['lora_names', 'lora_multipliers', 'additional_loras']:
                phase_params.pop(key, None)
            
            result.update(phase_params)
            
            # If phase config requires patching, pass the info through
            if self.phase.has_patch_config():
                result['_parsed_phase_config'] = self.phase.parsed_output
                result['_phase_config_model_name'] = self.phase.patch_model_name or self.model
        
        # Add model
        if self.model:
            result['model'] = self.model
        
        # Add extra params (passthrough)
        result.update(self.extra_params)
        
        return result
    
    def validate(self) -> List[str]:
        """Validate entire task configuration."""
        errors = []
        
        errors.extend(self.generation.validate())
        errors.extend(self.lora.validate())
        errors.extend(self.vace.validate())
        errors.extend(self.phase.validate())
        
        # Check LoRA/multiplier count match
        lora_format = self.lora.to_wgp_format()
        lora_count = len(lora_format.get('activated_loras', []))
        mult_str = lora_format.get('loras_multipliers', '')
        
        if lora_count > 0 and mult_str:
            mult_str_stripped = mult_str.strip()
            if mult_str_stripped:
                # Count multipliers based on format
                if ';' in mult_str_stripped:
                    mult_count = len(mult_str_stripped.split())
                elif ',' in mult_str_stripped:
                    mult_count = len(mult_str_stripped.split(','))
                else:
                    mult_count = len(mult_str_stripped.split())
                
                if mult_count != lora_count:
                    errors.append(f"LoRA count ({lora_count}) != multiplier count ({mult_count})")
        
        return errors
    
    def log_summary(self, log_func: Callable = None):
        """Log a summary of the configuration."""
        if log_func is None:
            log_func = logger.info
        
        log_func(f"[TaskConfig] Task {self.task_id} ({self.task_type})")
        log_func(f"  Model: {self.model}")
        log_func(f"  Resolution: {self.generation.resolution}")
        log_func(f"  Video length: {self.generation.video_length}")
        log_func(f"  Steps: {self.generation.num_inference_steps}")
        log_func(f"  LoRAs: {len(self.lora.entries)} entries, {len([e for e in self.lora.entries if e.status.value == 'pending'])} pending")
        if not self.phase.is_empty():
            log_func(f"  Phase config: {self.phase.num_phases} phases, {self.phase.total_steps} steps")
