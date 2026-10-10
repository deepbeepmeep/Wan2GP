from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTaskSpec:
    name: str
    sampling_group: str
    output_field: str


@dataclass(frozen=True)
class TokenBlockSpec:
    name: str
    labels: tuple[str, ...]
    output_field: str | None = None


@dataclass(frozen=True)
class PromptMultitaskSchema:
    version: str
    tasks: tuple[PromptTaskSpec, ...]
    event_field_order: tuple[str, ...]
    appended_token_blocks: tuple[TokenBlockSpec, ...] = ()


V1_TASKS = (
    PromptTaskSpec("timestamp", "timestamp", "timestamp"),
    PromptTaskSpec("downbeat_meter", "rhythm", "rhythm"),
    PromptTaskSpec("structure", "structure", "structure"),
    PromptTaskSpec("key", "key", "key"),
    PromptTaskSpec("chord_majmin", "chord", "chord"),
    PromptTaskSpec("chord_full", "chord", "chord"),
    PromptTaskSpec("melody_vocal", "melody", "melody"),
    PromptTaskSpec("melody_full", "melody", "melody"),
)


PROMPT_MULTITASK_SCHEMAS = {
    "v1": PromptMultitaskSchema(
        version="v1",
        tasks=V1_TASKS,
        event_field_order=(
            "timestamp",
            "rhythm",
            "structure",
            "key",
            "chord",
            "melody",
        ),
    ),
}


def _validate_schema(schema):
    task_names = tuple(task.name for task in schema.tasks)
    if len(task_names) != len(set(task_names)):
        raise ValueError(f"Duplicate task name in schema {schema.version!r}")
    field_names = tuple(schema.event_field_order)
    if len(field_names) != len(set(field_names)):
        raise ValueError(f"Duplicate output field in schema {schema.version!r}")
    missing_fields = {
        task.output_field for task in schema.tasks if task.output_field not in field_names
    }
    if missing_fields:
        raise ValueError(
            f"Tasks in schema {schema.version!r} use unknown output fields: "
            f"{tuple(sorted(missing_fields))}"
        )
    block_names = tuple(block.name for block in schema.appended_token_blocks)
    if len(block_names) != len(set(block_names)):
        raise ValueError(f"Duplicate token block in schema {schema.version!r}")
    for block in schema.appended_token_blocks:
        if not block.labels or len(block.labels) != len(set(block.labels)):
            raise ValueError(
                f"Token block {block.name!r} must contain unique labels"
            )


def register_prompt_multitask_schema(schema):
    """Register a frozen schema version without permitting redefinition."""
    if not isinstance(schema, PromptMultitaskSchema):
        raise TypeError("schema must be a PromptMultitaskSchema")
    _validate_schema(schema)
    existing = PROMPT_MULTITASK_SCHEMAS.get(schema.version)
    if existing is not None and existing != schema:
        raise ValueError(f"Schema {schema.version!r} is already registered")
    PROMPT_MULTITASK_SCHEMAS[schema.version] = schema
    return schema


def extend_prompt_multitask_schema(
    base_version,
    version,
    tasks=(),
    event_fields=(),
    token_blocks=(),
):
    """Append tasks and vocabulary blocks while preserving every base token id."""
    base = get_prompt_multitask_schema(base_version)
    schema = PromptMultitaskSchema(
        version=str(version),
        tasks=base.tasks + tuple(tasks),
        event_field_order=base.event_field_order + tuple(event_fields),
        appended_token_blocks=base.appended_token_blocks + tuple(token_blocks),
    )
    return register_prompt_multitask_schema(schema)


def get_prompt_multitask_schema(version):
    version = str(version)
    try:
        return PROMPT_MULTITASK_SCHEMAS[version]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt multitask schema {version!r}; "
            f"available={tuple(PROMPT_MULTITASK_SCHEMAS)}"
        ) from exc


_validate_schema(PROMPT_MULTITASK_SCHEMAS["v1"])
