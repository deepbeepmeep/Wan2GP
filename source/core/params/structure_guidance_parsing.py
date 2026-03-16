"""Compatibility helpers for the extracted structure-guidance parsing API."""

from source.core.params.structure_guidance import StructureGuidanceConfig


def parse_new_structure_guidance_format(
    config: StructureGuidanceConfig,
    payload: dict,
) -> StructureGuidanceConfig:
    """Return a parsed config from the new ``structure_guidance`` payload shape."""
    del config
    return StructureGuidanceConfig._from_new_format(payload)


def parse_legacy_structure_guidance_format(
    config: StructureGuidanceConfig,
    payload: dict,
) -> StructureGuidanceConfig:
    """Return a parsed config from the legacy structure-guidance parameter shape."""
    del config
    return StructureGuidanceConfig._from_legacy_format(payload)

