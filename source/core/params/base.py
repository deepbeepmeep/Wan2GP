"""
Base classes and utilities for typed parameter handling.

PRECEDENCE RULES (documented here as single source of truth):
- DB Tasks: top_level > orchestrator_details > orchestrator_payload
- Segments: individual_params > segment_params > orchestrator_payload
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


PARAM_MISSING = object()


class ParamGroup(ABC):
    """
    Base class for parameter groups.
    
    Subclasses must implement:
    - from_params(params, **context) -> cls
    - to_wgp_format() -> Dict[str, Any]
    """
    
    @classmethod
    @abstractmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'ParamGroup':
        """Parse parameters from a dict. Context can include task_id, model, etc."""
        pass
    
    @abstractmethod
    def to_wgp_format(self) -> Dict[str, Any]:
        """Convert to WGP-compatible format."""
        pass
    
    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        return []
    
    @staticmethod
    def _get_first_of(params: Dict[str, Any], *keys, default=None):
        """Get first non-None value from a list of possible keys."""
        for key in keys:
            if key in params and params[key] is not None:
                return params[key]
        return default
    
    @staticmethod
    def _parse_list(value, separator=',') -> List[str]:
        """Parse a value that could be a list or comma-separated string."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [x.strip() for x in value.split(separator) if x.strip()]
        return [str(value)]
    
    @staticmethod
    def flatten_params(db_params: Dict[str, Any], task_id: str = "") -> Dict[str, Any]:
        """
        Flatten nested DB params with documented precedence.
        
        Precedence: top_level > orchestrator_details > full_orchestrator_payload (legacy)
        """
        result = {}
        
        # Start with full_orchestrator_payload (legacy name, lowest precedence)
        if "full_orchestrator_payload" in db_params and isinstance(db_params["full_orchestrator_payload"], dict):
            result.update(db_params["full_orchestrator_payload"])
        
        # Then orchestrator_details (canonical name, medium precedence)
        if "orchestrator_details" in db_params and isinstance(db_params["orchestrator_details"], dict):
            result.update(db_params["orchestrator_details"])
        
        # Finally top-level params (highest precedence)
        for key, value in db_params.items():
            if key not in ("full_orchestrator_payload", "orchestrator_details"):
                result[key] = value
        
        return result


def warn_similar_key(params: Dict[str, Any], expected: str, alternatives: List[str], task_id: str = ""):
    """Log a warning if we find a similar but not exact key."""
    for alt in alternatives:
        if alt in params and expected not in params:
            logger.warning(f"Task {task_id}: Found '{alt}' but expected '{expected}' - using '{alt}'")


def resolve_param(key: str, *sources: Dict[str, Any], default=PARAM_MISSING, prefer_truthy: bool = False):
    """Resolve a key from the first source that contains a usable value."""
    for source in sources:
        if not isinstance(source, dict) or key not in source:
            continue
        value = source[key]
        if value is None:
            continue
        if prefer_truthy and not isinstance(value, bool) and not value:
            continue
        return value
    return None if default is PARAM_MISSING else default


def resolve_segment_param(
    key: str,
    individual_segment_params: Dict[str, Any] | None,
    segment_params: Dict[str, Any] | None,
    orchestrator_details: Dict[str, Any] | None,
    *,
    default=PARAM_MISSING,
    prefer_truthy: bool = False,
):
    return resolve_param(
        key,
        individual_segment_params or {},
        segment_params or {},
        orchestrator_details or {},
        default=default,
        prefer_truthy=prefer_truthy,
    )


def get_individual_segment_params(params: Dict[str, Any]) -> Dict[str, Any]:
    value = params.get("individual_segment_params")
    return value if isinstance(value, dict) else {}


def resolve_orchestrator_details(
    params: Dict[str, Any],
    *,
    prefer_truthy_canonical: bool = True,
) -> Dict[str, Any] | None:
    canonical = params.get("orchestrator_details")
    legacy = params.get("full_orchestrator_payload")

    if isinstance(canonical, dict):
        if canonical or not prefer_truthy_canonical:
            return canonical
        return canonical

    allow_legacy = (os.getenv("HEADLESS_ALLOW_LEGACY_ORCHESTRATOR_PAYLOAD") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if allow_legacy and isinstance(legacy, dict):
        return legacy
    return None
