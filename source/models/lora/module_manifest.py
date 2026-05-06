"""LoRA module manifest loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_DIR = Path(__file__).with_name("module_manifests")


class LoRAModuleManifestError(ValueError):
    """Raised when a LoRA module manifest is missing or malformed."""


@dataclass(frozen=True)
class LoRAModuleManifest:
    """Validated LoRA module names for one architecture."""

    architecture: str
    module_names: tuple[str, ...]
    source_path: Path
    schema_version: int = MANIFEST_SCHEMA_VERSION


def manifest_filename_for_architecture(architecture: str) -> str:
    if not architecture or not architecture.strip():
        raise LoRAModuleManifestError("architecture is required")
    return f"module_names_{architecture}.json"


def manifest_path_for_architecture(
    architecture: str,
    manifest_dir: Path | None = None,
) -> Path:
    directory = manifest_dir or MANIFEST_DIR
    return directory / manifest_filename_for_architecture(architecture)


def architecture_from_manifest_filename(path: Path) -> str:
    name = path.name
    if not name.startswith("module_names_") or not name.endswith(".json"):
        raise LoRAModuleManifestError(f"invalid manifest filename: {name}")
    return name.removeprefix("module_names_").removesuffix(".json")


def _validate_module_names(value: object, *, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise LoRAModuleManifestError(f"{path}: module_names must be a non-empty list")

    names: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LoRAModuleManifestError(f"{path}: module_names entries must be non-empty strings")
        name = item.strip()
        if name in seen:
            raise LoRAModuleManifestError(f"{path}: duplicate module name '{name}'")
        names.append(name)
        seen.add(name)
    return tuple(names)


def parse_module_manifest(data: object, *, source_path: Path) -> LoRAModuleManifest:
    if not isinstance(data, dict):
        raise LoRAModuleManifestError(f"{source_path}: manifest must be a JSON object")

    schema_version = data.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise LoRAModuleManifestError(
            f"{source_path}: schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )

    architecture = data.get("architecture")
    if not isinstance(architecture, str) or not architecture.strip():
        raise LoRAModuleManifestError(f"{source_path}: architecture must be a non-empty string")
    architecture = architecture.strip()

    expected_architecture = architecture_from_manifest_filename(source_path)
    if architecture != expected_architecture:
        raise LoRAModuleManifestError(
            f"{source_path}: architecture '{architecture}' does not match filename "
            f"'{expected_architecture}'"
        )

    return LoRAModuleManifest(
        architecture=architecture,
        module_names=_validate_module_names(data.get("module_names"), path=source_path),
        source_path=source_path,
        schema_version=schema_version,
    )


def load_module_manifest(
    architecture: str,
    manifest_dir: Path | None = None,
) -> LoRAModuleManifest:
    path = manifest_path_for_architecture(architecture, manifest_dir)
    if not path.is_file():
        raise LoRAModuleManifestError(f"missing LoRA module manifest for architecture '{architecture}'")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoRAModuleManifestError(f"{path}: invalid JSON: {exc}") from exc
    return parse_module_manifest(data, source_path=path)


def iter_module_manifest_paths(manifest_dir: Path | None = None) -> Iterable[Path]:
    directory = manifest_dir or MANIFEST_DIR
    if not directory.is_dir():
        return ()
    return tuple(sorted(directory.glob("module_names_*.json")))


def load_all_module_manifests(
    manifest_dir: Path | None = None,
) -> dict[str, LoRAModuleManifest]:
    manifests: dict[str, LoRAModuleManifest] = {}
    for path in iter_module_manifest_paths(manifest_dir):
        manifest = parse_module_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            source_path=path,
        )
        manifests[manifest.architecture] = manifest
    return manifests
