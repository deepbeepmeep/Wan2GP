import os
import shutil
from pathlib import Path

from safetensors import safe_open


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _move_with_collision_handling(src: Path, dst_dir: Path):
    _ensure_dir(dst_dir)
    target = dst_dir / src.name
    if target.exists():
        stem, suffix = os.path.splitext(src.name)
        idx = 1
        while True:
            candidate = dst_dir / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            idx += 1
    shutil.move(str(src), str(target))


def _classify_wan_lora(file_path: Path, subfolder_hint: str | None = None) -> str:
    if subfolder_hint in {"1.3B"}:
        return "wan_1.3B"
    if subfolder_hint in {"5B"}:
        return "wan_5B"
    if subfolder_hint in {"14B"}:
        return "wan"

    candidate_tokens = [
        "to_q",
        "to_k",
        "to_v",
        "proj_in",
        "proj_out",
        "ff.net",
        "ffn",
        "transformer",
        "attn",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lora_down",
        "lora_up",
    ]
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            if not keys:
                return "wan"
            max_dim = 0
            # Prefer typical attention/ffn LoRA tensors; fall back to any tensor
            for key in keys:
                k_lower = key.lower()
                if not any(token in k_lower for token in candidate_tokens):
                    continue
                tensor = f.get_tensor(key)
                max_dim = max(max_dim, max(tensor.shape) if tensor.ndim > 0 else 0)
            if max_dim == 0:
                tensor = f.get_tensor(keys[0])
                max_dim = max(tensor.shape) if tensor.ndim > 0 else 0
            if max_dim >= 5000:
                return "wan"
            if max_dim >= 3000:
                return "wan_5B"
            return "wan_1.3B"
    except Exception:
        return "wan"


def _move_dir_contents(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        return
    for item in src_dir.iterdir():
        _move_with_collision_handling(item, dst_dir)
    try:
        src_dir.rmdir()
    except OSError:
        pass


def migrate_loras_layout(root_dir: Path | str | None = None):
    """
    Reorganize lora folders into a single 'loras' tree.
    Migration is skipped if root/loras_i2v is already gone.
    """
    root = Path(root_dir or ".").resolve()
    marker = root / "loras_i2v"
    if not marker.exists():
        return

    loras_root = root / "loras"
    _ensure_dir(loras_root)

    # Move dedicated per-family folders (loras_foo -> loras/foo)
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name in {"loras", "loras_i2v"}:
            continue
        if entry.name.startswith("loras_"):
            target_name = entry.name[len("loras_") :]
            _move_dir_contents(entry, loras_root / target_name)

    # Move Wan i2v
    _move_dir_contents(marker, loras_root / "wan_i2v")

    # Handle Wan core folders inside loras
    wan_dir = loras_root / "wan"
    wan_1_3b_dir = loras_root / "wan_1.3B"
    wan_5b_dir = loras_root / "wan_5B"
    wan_i2v_dir = loras_root / "wan_i2v"
    optional_hints = {"1.3B", "5B", "14B"}
    moved_wan_by_signature = {}
    for legacy_name, target in [
        ("14B", wan_dir),
        ("1.3B", wan_1_3b_dir),
        ("5B", wan_5b_dir),
    ]:
        _move_dir_contents(loras_root / legacy_name, target)

    # Classify remaining loose Wan loras at the old loras root
    for item in list(loras_root.iterdir()):
        if item.is_dir() and item.name.startswith("wan"):
            continue
        if item.is_dir():
            # Non-Wan folders already handled above
            continue
        if not item.is_file():
            continue
        subfolder_hint = item.parent.name
        used_hint = subfolder_hint in optional_hints
        target_bucket = _classify_wan_lora(item, subfolder_hint=subfolder_hint if used_hint else None)
        target_dir = {"wan": wan_dir, "wan_1.3B": wan_1_3b_dir, "wan_5B": wan_5b_dir}.get(
            target_bucket, wan_dir
        )
        _move_with_collision_handling(item, target_dir)
        if target_bucket.startswith("wan") and not used_hint:
            moved_wan_by_signature[item.name] = target_dir

    # Move any remaining files under loras root (unlikely) into wan
    for item in list(loras_root.iterdir()):
        if item.is_file():
            _move_with_collision_handling(item, wan_dir)

    # Move .lset files referencing Wan loras that were reclassified by tensor signature
    for lset_file in loras_root.glob("*.lset"):
        try:
            content = lset_file.read_text(errors="ignore")
        except Exception:
            continue
        for lora_name, dest_dir in moved_wan_by_signature.items():
            if lora_name in content:
                _move_with_collision_handling(lset_file, dest_dir)
                break

    for folder in [wan_dir, wan_1_3b_dir, wan_5b_dir, wan_i2v_dir]:
        _ensure_dir(folder)
