from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PLUGIN_DIR = Path(__file__).resolve().parent
APP_ROOT_DIR = PLUGIN_DIR.parent.parent
PACKS_DIR = APP_ROOT_DIR / "character_packs"


MODE_LABELS = {
    "bernini_ingredients": "Bernini-R 14B - multi-reference ingredients",
    "bernini_ingredients_1_3b": "Bernini-R 1.3B - low VRAM ingredients",
    "vace_standin": "VACE Stand-In 14B - face identity + control",
    "standin_face": "Stand-In 14B - face identity",
    "scail2_animate": "SCAIL-2 14B - animate from reference/control video",
    "joyai_echo_memory": "JoyAI-Echo 22B - multi-shot memory story",
}


def slugify(value: str, fallback: str = "character") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return text[:80] if text else fallback


def split_lines(value: str) -> list[str]:
    lines = []
    for raw_line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if line:
            lines.append(line)
    return lines


def split_shots(value: str) -> list[str]:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    return split_lines(text)


def coerce_uploaded_paths(uploaded_files: Any) -> list[str]:
    if uploaded_files is None:
        return []
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    paths: list[str] = []
    for item in uploaded_files:
        candidate = None
        if isinstance(item, (str, Path)):
            candidate = str(item)
        elif isinstance(item, dict):
            candidate = item.get("path") or item.get("name") or item.get("value")
        else:
            candidate = getattr(item, "name", None) or getattr(item, "path", None)
        if candidate:
            paths.append(str(candidate))
    return paths


def coerce_reference_paths(path_text: str = "", uploaded_files: Any = None) -> list[str]:
    paths = split_lines(path_text)
    paths.extend(coerce_uploaded_paths(uploaded_files))
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path).strip().strip('"')
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)
    return deduped


def validate_pack(mode: str, refs: list[str], identity_prompt: str, shot_prompts: list[str]) -> list[str]:
    issues: list[str] = []
    if not identity_prompt.strip():
        issues.append("Add a locked identity prompt.")
    if not shot_prompts:
        issues.append("Add at least one shot prompt.")
    if mode != "joyai_echo_memory" and not refs:
        issues.append("Add at least one reference image path or upload.")
    if mode == "standin_face" and len(refs) > 1:
        issues.append("Stand-In uses one close-up face reference; extra references will be ignored.")
    if mode == "vace_standin" and refs and len(refs) < 2:
        issues.append("VACE Stand-In works best when the last reference is a close-up face and earlier refs show outfit/body.")
    if mode.startswith("bernini") and len(refs) > 6:
        issues.append("Bernini can accept many references, but 2-4 strong images are usually more stable than a large mixed set.")
    return issues


def _base_prompt(identity_prompt: str, shot_prompt: str, style_prompt: str = "", environment_prompt: str = "") -> str:
    parts = [
        "Use the provided character references as the source of truth for identity.",
        f"LOCKED CHARACTER IDENTITY: {identity_prompt.strip()}",
    ]
    if environment_prompt.strip():
        parts.append(f"LOCKED ENVIRONMENT/WORLD: {environment_prompt.strip()}")
    if style_prompt.strip():
        parts.append(f"LOCKED VISUAL STYLE: {style_prompt.strip()}")
    parts.append(f"SHOT DIRECTION: {shot_prompt.strip()}")
    parts.append(
        "Preserve the same face, hair, body proportions, age, signature clothing details, and recognizable silhouette. "
        "Do not redesign the character. Keep recurring environment details, color palette, lighting logic, and camera movement "
        "cinematic and temporally coherent."
    )
    return "\n".join(parts)


def _common_settings(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    video_length: int,
    seed: int,
    refs: list[str],
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": resolution,
        "video_length": int(video_length),
        "seed": int(seed),
        "repeat_generation": 1,
        "prompt_enhancer": "",
    }
    if refs:
        settings["image_refs"] = refs
    return settings


def build_settings(
    *,
    mode: str,
    identity_prompt: str,
    shot_prompt: str,
    refs: list[str],
    negative_prompt: str = "",
    style_prompt: str = "",
    environment_prompt: str = "",
    resolution: str = "1280x720",
    video_length: int = 121,
    seed: int = -1,
    control_video: str = "",
) -> dict[str, Any]:
    prompt = _base_prompt(identity_prompt, shot_prompt, style_prompt, environment_prompt)
    settings = _common_settings(prompt, negative_prompt, resolution, video_length, seed, refs)

    if mode == "bernini_ingredients":
        settings.update(
            {
                "model_type": "bernini",
                "video_prompt_type": "I",
                "num_inference_steps": 24,
                "sample_solver": "unipc",
                "flow_shift": 5,
                "guidance_phases": 2,
                "model_switch_phase": 1,
                "switch_threshold": 875,
                "guidance_scale": 4,
                "guidance2_scale": 4,
                "control_net_weight": 1.15,
                "alt_guidance_scale": 4.5,
                "remove_background_images_ref": 1,
            }
        )
    elif mode == "bernini_ingredients_1_3b":
        settings.update(
            {
                "model_type": "bernini_1.3B",
                "video_prompt_type": "I",
                "num_inference_steps": 28,
                "sample_solver": "unipc",
                "flow_shift": 5,
                "guidance_phases": 1,
                "switch_threshold": 0,
                "guidance_scale": 4,
                "control_net_weight": 1.1,
                "alt_guidance_scale": 4.5,
                "remove_background_images_ref": 1,
            }
        )
    elif mode == "vace_standin":
        settings.update(
            {
                "model_type": "vace_standin_14B",
                "video_prompt_type": "I",
                "num_inference_steps": 18,
                "guidance_scale": 5,
                "flow_shift": 3,
                "embedded_guidance_scale": 6,
                "remove_background_images_ref": 1,
                "sliding_window_size": 81,
                "sliding_window_overlap": 9,
            }
        )
    elif mode == "standin_face":
        settings.update(
            {
                "model_type": "standin",
                "video_prompt_type": "I",
                "image_refs": refs[:1],
                "num_inference_steps": 28,
                "guidance_scale": 5,
                "flow_shift": 3,
                "remove_background_images_ref": 1,
            }
        )
    elif mode == "scail2_animate":
        settings.update(
            {
                "model_type": "scail2_14B",
                "video_prompt_type": "I" + ("V" if control_video.strip() else ""),
                "video_source": control_video.strip() or None,
                "num_inference_steps": 32,
                "flow_shift": 3,
                "guidance_scale": 5,
                "custom_settings": {
                    "scail2_animate_preprocessing": "raw",
                    "image_ref_keyword_content": "human character",
                },
                "sliding_window_size": 81,
                "sliding_window_overlap": 5,
                "sliding_window_color_correction_strength": 0,
            }
        )
        if settings["video_source"] is None:
            settings.pop("video_source", None)
    elif mode == "joyai_echo_memory":
        settings.update(
            {
                "model_type": "joyai_echo",
                "num_inference_steps": 8,
                "guidance_scale": 1.0,
                "audio_guidance_scale": 1.0,
                "alt_guidance_scale": 1.0,
                "alt_scale": 0.0,
                "guidance_phases": 1,
                "multi_prompts_gen_type": "PW",
                "custom_settings": {"joyai_control_memory_positions": ""},
            }
        )
        settings.pop("image_refs", None)
    else:
        raise ValueError(f"Unsupported character consistency mode: {mode}")
    return {key: value for key, value in settings.items() if value is not None}


def build_manifest(
    *,
    character_name: str,
    mode: str,
    identity_prompt: str,
    shot_prompts: list[str],
    refs: list[str],
    negative_prompt: str = "",
    style_prompt: str = "",
    environment_prompt: str = "",
    resolution: str = "1280x720",
    video_length: int = 121,
    seed: int = -1,
    control_video: str = "",
) -> list[dict[str, Any]]:
    manifest = []
    for index, shot_prompt in enumerate(shot_prompts, start=1):
        settings = build_settings(
            mode=mode,
            identity_prompt=identity_prompt,
            shot_prompt=shot_prompt,
            refs=refs,
            negative_prompt=negative_prompt,
            style_prompt=style_prompt,
            environment_prompt=environment_prompt,
            resolution=resolution,
            video_length=video_length,
            seed=seed,
            control_video=control_video,
        )
        settings["character_pack"] = {
            "name": character_name.strip() or "Character",
            "mode": mode,
            "mode_label": MODE_LABELS.get(mode, mode),
            "shot_index": index,
            "source": "Character Consistency Studio",
        }
        manifest.append(settings)
    return manifest


def save_pack_json(character_name: str, payload: dict[str, Any]) -> Path:
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = PACKS_DIR / f"{slugify(character_name)}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def save_manifest_json(character_name: str, manifest: list[dict[str, Any]]) -> Path:
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = PACKS_DIR / f"{slugify(character_name)}-wan2gp-settings-{stamp}.json"
    path.write_text(json.dumps(manifest if len(manifest) > 1 else manifest[0], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def research_summary_markdown() -> str:
    return (
        "Commercial consistency tools use three layers: curated multi-angle references, reusable identity/world anchors, "
        "and shot-by-shot prompting that repeats those anchors. In WanGP, the closest open workflow is to save a character "
        "and environment pack, then re-inject it through Bernini, VACE Stand-In, Stand-In, SCAIL-2, or JoyAI-Echo depending on the shot."
    )
