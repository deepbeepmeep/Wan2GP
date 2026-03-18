"""
Typed parameter dataclasses for clean parameter handling.

Usage:
    from source.core.params import TaskConfig, LoRAConfig

    # Parse from DB task
    config = TaskConfig.from_db_task(task.parameters, task_id=task.id, model=task.model)

    # Convert to WGP format at the boundary
    wgp_params = config.to_wgp_format()
"""

from .base import ParamGroup
from .lora import LoRAConfig, LoRAEntry, LoRAStatus
from .vace import VACEConfig
from .generation import GenerationConfig
from .phase import PhaseConfig
from .task import TaskConfig
from .structure_guidance import StructureGuidanceConfig, StructureVideoEntry
from .travel_guidance import TravelGuidanceConfig
from .contracts import (
    TaskDispatchContext,
    OrchestratorDetails,
    validate_orchestrator_details,
)

__all__ = [
    'ParamGroup',
    'LoRAConfig', 'LoRAEntry', 'LoRAStatus',
    'VACEConfig',
    'GenerationConfig',
    'PhaseConfig',
    'TaskConfig',
    'TravelGuidanceConfig',
    'StructureGuidanceConfig', 'StructureVideoEntry',
    'TaskDispatchContext', 'OrchestratorDetails',
    'validate_orchestrator_details',
]
