"""
LoRA configuration handling with typed entries.

Uses LoRAEntry objects instead of parallel arrays to preserve ordering
and ensure filename/multiplier pairs stay in sync.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import os
import logging

from .base import ParamGroup
from source.models.lora.sanitizer import sanitize_lora_entries

logger = logging.getLogger(__name__)


class LoRAStatus(Enum):
    """Status of a LoRA entry."""
    PENDING = "pending"      # URL needs to be downloaded
    DOWNLOADED = "downloaded"  # Downloaded, has local_path
    LOCAL = "local"          # Local file reference


@dataclass
class LoRAEntry:
    """
    Single LoRA entry with all its metadata.
    
    Using objects instead of parallel arrays ensures:
    - Filename and multiplier stay paired
    - Order is preserved during operations
    - Clear status tracking
    """
    url: Optional[str] = None
    local_path: Optional[str] = None
    filename: Optional[str] = None
    multiplier: Union[float, str] = 1.0  # str for phase-config format like "1.0;0.5"
    status: LoRAStatus = LoRAStatus.LOCAL
    source: str = ""  # Where this entry came from (for debugging)
    
    def mark_downloaded(self, local_path: str):
        """Mark this entry as downloaded with the given local path."""
        self.local_path = local_path
        self.filename = os.path.basename(local_path)
        self.status = LoRAStatus.DOWNLOADED
    
    def get_effective_path(self) -> Optional[str]:
        """Get the path to use for this LoRA (local_path or filename)."""
        return self.local_path or self.filename
    
    def is_phase_config_multiplier(self) -> bool:
        """Check if multiplier is phase-config format (contains semicolons)."""
        return isinstance(self.multiplier, str) and ';' in str(self.multiplier)


@dataclass  
class LoRAConfig(ParamGroup):
    """
    Typed LoRA configuration.
    
    Handles all LoRA naming conventions:
    - WGP: activated_loras, loras_multipliers (string)
    - Internal: lora_names, lora_multipliers (list)
    - Phase config: additional_loras dict with URL keys
    """
    entries: List[LoRAEntry] = field(default_factory=list)
    
    @classmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'LoRAConfig':
        """
        Parse LoRA config from various input formats.
        
        Handles:
        - activated_loras / lora_names (list of filenames)
        - loras_multipliers / lora_multipliers (string or list)
        - additional_loras (dict of URL -> multiplier for downloads)
        """
        entries = []
        
        # Get filenames from either naming convention
        filenames = (
            params.get('activated_loras') or 
            params.get('lora_names') or 
            []
        )
        if isinstance(filenames, str):
            filenames = [f.strip() for f in filenames.split(',') if f.strip()]
        
        # Get multipliers from either naming convention
        multipliers_raw = (
            params.get('loras_multipliers') or 
            params.get('lora_multipliers') or
            []
        )
        
        # Parse multipliers
        if isinstance(multipliers_raw, str):
            # Could be space-separated or comma-separated
            if ';' in multipliers_raw:
                # Phase-config format: space-separated strings with semicolons
                multipliers = multipliers_raw.split()
            elif ',' in multipliers_raw:
                multipliers = [m.strip() for m in multipliers_raw.split(',')]
            else:
                multipliers = multipliers_raw.split()
        elif isinstance(multipliers_raw, list):
            multipliers = multipliers_raw
        else:
            multipliers = []
        
        # Create entries from filenames (could be URLs, absolute paths, or filenames)
        for i, filename in enumerate(filenames):
            mult = multipliers[i] if i < len(multipliers) else 1.0
            # Try to convert to float if it's a simple number
            if isinstance(mult, str) and ';' not in mult:
                try:
                    mult = float(mult)
                except ValueError:
                    pass
            
            # Detect if this is a URL that needs downloading
            is_url = filename.startswith(('http://', 'https://'))
            
            if is_url:
                # URL needs downloading
                entries.append(LoRAEntry(
                    url=filename,
                    filename=os.path.basename(filename),
                    multiplier=mult,
                    status=LoRAStatus.PENDING,
                    source='params_url'
                ))
            else:
                # Local file or filename
                entries.append(LoRAEntry(
                    filename=filename,
                    local_path=filename if os.path.isabs(filename) else None,
                    multiplier=mult,
                    status=LoRAStatus.LOCAL,
                    source='params'
                ))
        
        # Handle additional_loras (URLs needing download)
        additional_loras = params.get('additional_loras', {})
        if isinstance(additional_loras, dict):
            for url, mult in additional_loras.items():
                # Check if we already have this URL's file
                filename = os.path.basename(url) if url else None
                existing = next((e for e in entries if e.filename == filename), None)
                
                if existing:
                    # Update existing entry with URL info
                    existing.url = url
                    if mult and ';' in str(mult):
                        # Phase-config multiplier takes precedence
                        existing.multiplier = mult
                    existing.status = LoRAStatus.PENDING
                else:
                    # New entry for download
                    entries.append(LoRAEntry(
                        url=url,
                        filename=filename,
                        multiplier=mult if mult else 1.0,
                        status=LoRAStatus.PENDING,
                        source='additional_loras'
                    ))
        
        architecture = (
            context.get('model')
            or context.get('model_name')
            or params.get('model')
            or params.get('model_name')
        )
        if architecture and entries:
            entries = sanitize_lora_entries(
                entries,
                architecture=architecture,
                task_id=context.get('task_id'),
            ).entries

        return cls(entries=entries)
    
    @classmethod
    def from_phase_config(cls, phase_config: Dict[str, Any], **context) -> 'LoRAConfig':
        """
        Parse LoRA config from phase_config structure.

        Phase config has LoRAs per phase with per-phase multipliers.
        We combine into single entries with semicolon-separated multipliers.
        """
        entries = []
        phases = phase_config.get('phases', [])

        # Collect all unique LoRAs with their per-phase multipliers
        lora_phases: Dict[str, List[float]] = {}  # url -> [mult_phase1, mult_phase2, ...]

        for phase_idx, phase in enumerate(phases):
            phase_loras = phase.get('loras', [])
            for lora in phase_loras:
                url = lora.get('url', '')
                mult = lora.get('multiplier', 1.0)

                if url not in lora_phases:
                    lora_phases[url] = [0.0] * len(phases)
                lora_phases[url][phase_idx] = mult

        # Create entries with combined multipliers
        for url, mults in lora_phases.items():
            mult_str = ';'.join(str(m) for m in mults)
            entries.append(LoRAEntry(
                url=url,
                filename=os.path.basename(url) if url else None,
                multiplier=mult_str,
                status=LoRAStatus.PENDING,
                source='phase_config'
            ))

        architecture = context.get('model') or context.get('model_name')
        if architecture and entries:
            entries = sanitize_lora_entries(
                entries,
                architecture=architecture,
                task_id=context.get('task_id'),
            ).entries

        return cls(entries=entries)

    @classmethod
    def from_segment_loras(cls, segment_loras: List[Dict[str, Any]], **context) -> 'LoRAConfig':
        """
        Parse LoRA config from per-segment LoRA override format.

        Args:
            segment_loras: List of LoRA dicts from frontend:
                [{"id": "...", "path": "...", "strength": 0.8, "name": "..."}, ...]

        Returns:
            LoRAConfig with entries for each LoRA
        """
        entries = []

        for lora_dict in segment_loras:
            # Frontend sends: {id, path, strength, name?}
            # path could be a URL or a local path/filename
            lora_path = lora_dict.get('path', '')
            strength = lora_dict.get('strength', 1.0)
            if not lora_path:
                continue

            # Detect if this is a URL that needs downloading
            is_url = lora_path.startswith(('http://', 'https://'))

            if is_url:
                # URL needs downloading
                entries.append(LoRAEntry(
                    url=lora_path,
                    filename=os.path.basename(lora_path),
                    multiplier=strength,
                    status=LoRAStatus.PENDING,
                    source='segment_loras'
                ))
            else:
                # Local file or filename
                entries.append(LoRAEntry(
                    filename=lora_path,
                    local_path=lora_path if os.path.isabs(lora_path) else None,
                    multiplier=strength,
                    status=LoRAStatus.LOCAL,
                    source='segment_loras'
                ))

        architecture = context.get('model') or context.get('model_name')
        if architecture and entries:
            entries = sanitize_lora_entries(
                entries,
                architecture=architecture,
                task_id=context.get('task_id'),
            ).entries

        return cls(entries=entries)
    
    def to_wgp_format(self) -> Dict[str, Any]:
        """
        Convert to WGP-compatible format.
        
        Returns:
            activated_loras: list of filenames/paths
            loras_multipliers: space-separated string (for phase-config) or comma-separated
        """
        if not self.entries:
            return {'activated_loras': [], 'loras_multipliers': ''}
        
        # Only include downloaded/local entries
        ready_entries = [e for e in self.entries if e.status != LoRAStatus.PENDING]
        
        activated_loras = []
        multipliers = []
        
        for entry in ready_entries:
            path = entry.get_effective_path()
            if path:
                activated_loras.append(path)
                multipliers.append(str(entry.multiplier))
        
        # Format multipliers as space-separated string
        # WGP's preparse_loras_multipliers() splits on spaces, so both
        # phase-config ("0.9;0.5 1.1;1.1") and regular ("0.9 1.1 0.5")
        # must be space-separated
        loras_multipliers = ' '.join(multipliers)
        
        return {
            'activated_loras': activated_loras,
            'loras_multipliers': loras_multipliers
        }
    
    def has_pending_downloads(self) -> bool:
        """Check if any entries need downloading."""
        return any(e.status == LoRAStatus.PENDING and e.url for e in self.entries)
    
    def get_pending_downloads(self) -> Dict[str, Any]:
        """Get dict of URL -> multiplier for pending downloads."""
        return {
            e.url: e.multiplier 
            for e in self.entries 
            if e.status == LoRAStatus.PENDING and e.url
        }
    
    def mark_downloaded(self, url: str, local_path: str):
        """Mark a URL as downloaded with the given local path."""
        for entry in self.entries:
            if entry.url == url:
                entry.mark_downloaded(local_path)
                return
    
    def merge(self, other: 'LoRAConfig') -> 'LoRAConfig':
        """
        Merge another LoRAConfig, with other taking precedence for duplicates.
        Phase-config multipliers take precedence over regular ones.
        """
        merged_entries = list(self.entries)
        
        for other_entry in other.entries:
            # Find existing entry by filename
            existing_idx = next(
                (i for i, e in enumerate(merged_entries) 
                 if e.filename == other_entry.filename),
                None
            )
            
            if existing_idx is not None:
                existing = merged_entries[existing_idx]
                # Phase-config multipliers take precedence
                if other_entry.is_phase_config_multiplier():
                    merged_entries[existing_idx] = other_entry
                elif not existing.is_phase_config_multiplier():
                    # Other takes precedence for non-phase-config
                    merged_entries[existing_idx] = other_entry
            else:
                merged_entries.append(other_entry)
        
        return LoRAConfig(entries=merged_entries)
    
    def validate(self) -> List[str]:
        """Validate the LoRA configuration."""
        errors = []
        
        for entry in self.entries:
            if entry.status == LoRAStatus.LOCAL and entry.filename:
                # Could add file existence check here if needed
                pass
            if entry.status == LoRAStatus.PENDING and not entry.url:
                errors.append(f"Pending LoRA entry has no URL: {entry.filename}")
        
        return errors
