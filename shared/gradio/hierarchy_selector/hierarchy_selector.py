from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gradio.components.base import Component, FormComponent
from gradio.events import Events

if TYPE_CHECKING:
    from gradio.components import Timer


HierarchyNode = dict[str, Any]


def _clean_rel_path(path: str) -> str:
    rel_path = os.path.normpath(str(path)).replace(os.sep, "/")
    if rel_path == ".":
        return ""
    while rel_path.startswith("./"):
        rel_path = rel_path[2:]
    return rel_path


def _item_path_without_suffix(rel_path: str) -> str:
    return os.path.splitext(_clean_rel_path(rel_path))[0]


def _normalize_suffixes(suffix: str | Sequence[str]) -> tuple[str, ...]:
    return (suffix,) if isinstance(suffix, str) else tuple(str(one_suffix) for one_suffix in suffix)


def _item_display_path(rel_path: str, suffix: str | Sequence[str]) -> str:
    rel_path = _clean_rel_path(rel_path)
    for one_suffix in _normalize_suffixes(suffix):
        if rel_path.casefold().endswith(one_suffix.casefold()):
            return rel_path[: -len(one_suffix)]
    return _item_path_without_suffix(rel_path)


def _sort_key(node: HierarchyNode) -> str:
    return str(node.get("name") or node.get("path") or "").casefold()


def _sorted_hierarchy(node: HierarchyNode) -> HierarchyNode:
    folders = [_sorted_hierarchy(folder) for folder in node.get("folders", [])]
    items = [dict(item) for item in node.get("items", [])]
    folders.sort(key=_sort_key)
    items.sort(key=_sort_key)
    sorted_node = dict(node)
    sorted_node["folders"] = folders
    sorted_node["items"] = items
    return sorted_node


def build_file_hierarchy(root_dir: str | os.PathLike | None, suffix: str | Sequence[str] = (".safetensors", ".sft")) -> HierarchyNode:
    root = Path(root_dir) if root_dir else None
    tree: HierarchyNode = {"folders": [], "items": []}
    if root is None or not root.is_dir():
        return tree

    suffixes = tuple(one_suffix.casefold() for one_suffix in _normalize_suffixes(suffix))
    file_paths = [path for path in root.rglob("*") if path.is_file() and path.name.casefold().endswith(suffixes)]
    folder_index: dict[str, HierarchyNode] = {"": tree}
    for file_path in sorted(file_paths, key=lambda p: str(p.relative_to(root)).casefold()):
        rel_value = _clean_rel_path(str(file_path.relative_to(root)))
        rel_display = _item_display_path(rel_value, suffix)
        parts = rel_display.split("/")
        current = tree
        current_path = ""
        for folder_name in parts[:-1]:
            current_path = f"{current_path}/{folder_name}" if current_path else folder_name
            folder = folder_index.get(current_path)
            if folder is None:
                folder = {"name": folder_name, "path": current_path, "folders": [], "items": []}
                current.setdefault("folders", []).append(folder)
                folder_index[current_path] = folder
            current = folder
        current.setdefault("items", []).append({"name": parts[-1], "path": rel_display, "value": rel_value})
    return _sorted_hierarchy(tree)


def build_choices_hierarchy(choices: Sequence[str], suffix: str | Sequence[str] = (".safetensors", ".sft")) -> HierarchyNode:
    tree: HierarchyNode = {"folders": [], "items": []}
    folder_index: dict[str, HierarchyNode] = {"": tree}
    for value in sorted(dict.fromkeys(str(choice) for choice in choices if choice), key=str.casefold):
        rel_value = _clean_rel_path(value)
        rel_display = _item_display_path(rel_value, suffix)
        parts = rel_display.split("/")
        current = tree
        current_path = ""
        for folder_name in parts[:-1]:
            current_path = f"{current_path}/{folder_name}" if current_path else folder_name
            folder = folder_index.get(current_path)
            if folder is None:
                folder = {"name": folder_name, "path": current_path, "folders": [], "items": []}
                current.setdefault("folders", []).append(folder)
                folder_index[current_path] = folder
            current = folder
        current.setdefault("items", []).append({"name": parts[-1], "path": rel_display, "value": rel_value})
    return _sorted_hierarchy(tree)


class HierarchySelector(FormComponent):
    """
    Ordered multi-selection component backed by a folder/item hierarchy.
    """

    EVENTS = [Events.change, Events.input, Events.select, Events.focus, Events.blur]
    TEMPLATE_DIR = "templates/"
    FRONTEND_DIR = "frontend/"

    def __init__(
        self,
        hierarchy: HierarchyNode | None = None,
        *,
        value: Sequence[str] | str | Callable | None = None,
        height: int = 10,
        display_mode: str = "file",
        breadcrumb_separator: str = " ",
        sort_hierarchy: bool = True,
        search_empty_label: str = "No matches",
        show_placeholder: bool = False,
        label: str | None = None,
        info: str | None = None,
        every: "Timer | float | None" = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | None = None,
    ):
        self.display_mode = display_mode
        self.breadcrumb_separator = breadcrumb_separator
        self.sort_hierarchy = sort_hierarchy
        self.search_empty_label = search_empty_label
        self.show_placeholder = show_placeholder
        self.hierarchy = _sorted_hierarchy(hierarchy or {"folders": [], "items": []}) if sort_hierarchy else hierarchy or {"folders": [], "items": []}
        self.height = int(height)
        super().__init__(
            label=label,
            info=info,
            every=every,
            inputs=inputs,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            value=value,
        )

    def preprocess(self, payload: list[str] | str | None) -> list[str]:
        if payload is None:
            return []
        return payload if isinstance(payload, list) else [payload]

    def postprocess(self, value: Sequence[str] | str | None) -> list[str]:
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def example_payload(self) -> list[str]:
        return []

    def example_value(self) -> list[str]:
        return []

    def api_info(self) -> dict[str, Any]:
        return {"type": "array", "items": {"type": "string"}}
