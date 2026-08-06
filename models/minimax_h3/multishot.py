"""MiniMax H3 Multishot generation for WanGP.

Chains multiple shots into one continuous video with audio. Each shot starts
from the last frame of the previous one; duplicated seam frames and their
matching audio are trimmed automatically. Ported from ComfyUI-H3-Multishot.
"""

import json
import math
import re

import torch
import numpy as np
from PIL import Image

from .pipeline import FPS, AUDIO_SAMPLE_RATE


def parse_script(text):
    """Parse script text into list of shot prompts.

    Supports:
      - JSON: {"prompts": ["shot 1 ...", "shot 2 ..."]}
      - Plain text with --- separators between shots
    """
    text = (text or "").strip()
    if not text:
        return []

    shots = []
    if text.startswith("{") or text.startswith("["):
        data, repaired = _repair_json(text)
        if data is None:
            raise ValueError(
                f"H3 script looks like JSON but does not parse ({repaired}). "
                f"Fix the script or use plain prompts separated by --- lines."
            )
        if repaired:
            print(f"[Multishot] script JSON was incomplete; auto-repaired "
                  f"({repaired}).", flush=True)
        if isinstance(data, dict):
            shots = [str(p) for p in data.get("prompts", [])]
        elif isinstance(data, list):
            shots = [str(p) for p in data]

    if not shots:
        raw_segments = [b.strip() for b in re.split(r"(?m)^---\s*$", text) if b.strip()]
        shots = []
        for seg in raw_segments:
            try:
                parsed = json.loads(seg)
                if isinstance(parsed, dict) and "prompt" in parsed:
                    shots.append(parsed)
                else:
                    shots.append(seg)
            except (json.JSONDecodeError, ValueError):
                shots.append(seg)

    if not shots:
        shots = [text]

    return shots


def _repair_json(text):
    """Parse JSON, auto-closing unterminated brackets/quotes."""
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as e:
        first_err = str(e)

    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or
                          (ch == "]" and stack[-1] == "[")):
                stack.pop()

    candidate = text.rstrip()
    fixes = []
    if in_str:
        candidate += '"'
        fixes.append("closed an open string")
    if candidate.endswith(","):
        candidate = candidate[:-1]
        fixes.append("dropped a trailing comma")
    cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
    if cleaned != candidate:
        candidate = cleaned
        fixes.append("removed comma(s) before a closing bracket")
    for opener in reversed(stack):
        candidate += "}" if opener == "{" else "]"
    if stack:
        fixes.append("added " + "".join("}" if o == "{" else "]"
                                        for o in reversed(stack)))
    if not fixes:
        return None, first_err
    try:
        return json.loads(candidate), ", ".join(fixes)
    except json.JSONDecodeError as e:
        return None, str(e)


def xfade_audio(parts, sr, ms=40):
    """Concatenate shot audio with equal-power crossfade at seams."""
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    n = max(1, int(sr * ms / 1000.0))
    out = parts[0]
    for nxt in parts[1:]:
        k = min(n, out.shape[-1], nxt.shape[-1])
        if k < 8:
            out = torch.cat([out, nxt], dim=-1)
            continue
        t = torch.linspace(0, 1, k, dtype=out.dtype, device=out.device)
        fade_out = torch.cos(t * math.pi / 2)
        fade_in = torch.sin(t * math.pi / 2)
        head, tail = out[..., :-k], out[..., -k:]
        seam = tail * fade_out + nxt[..., :k] * fade_in
        out = torch.cat([head, seam, nxt[..., k:]], dim=-1)
    return out


def _pil_to_tensor(image):
    """Convert PIL Image to CTHW tensor."""
    arr = np.asarray(image.convert("RGB")).copy()
    return (torch.from_numpy(arr).permute(2, 0, 1).float()
            .div_(127.5).sub_(1.0).unsqueeze(1))




def _parse_keyframe_positions(positions_str, frame_count):
    """Parse comma-separated keyframe positions (percentage, absolute, or range).
    
    Examples: "50%", "120", "30%-60%", "0%,50%,100%", "2-5"
    Returns list of integer frame indices.
    """
    if not positions_str:
        return []
    
    def parse_value(tok):
        tok = tok.strip()
        is_pct = tok.endswith("%")
        if is_pct:
            tok = tok[:-1]
        try:
            value = float(tok)
        except ValueError:
            raise ValueError(f"'{tok}' in positions is not a number")
        if value < 0:
            raise ValueError(f"position {tok} is negative")
        return value, is_pct
    
    def value_to_index(value, is_pct):
        if is_pct:
            idx = int(round((value / 100.0) * (frame_count - 1)))
        else:
            idx = int(round(value))
        return max(0, min(frame_count - 1, idx))
    
    out = []
    raw = positions_str.replace(";", ",")
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok and not tok.startswith("-"):
            parts = tok.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid range '{tok}'")
            left, right = (p.strip() for p in parts)
            sv, sp = parse_value(left)
            ev, ep = parse_value(right)
            if sp != ep:
                raise ValueError(f"range '{tok}' mixes pct and absolute")
            start = value_to_index(sv, sp)
            end = value_to_index(ev, ep)
            step = 1 if start <= end else -1
            out.extend(range(start, end + step, step))
        else:
            v, p = parse_value(tok)
            out.append(value_to_index(v, p))
    return sorted(set(out))


def _load_and_encode_interior_keyframes(pipeline, kf_specs, frame_num, height, width):
    """Load user-provided interior keyframe images and prepare them.
    
    Args:
        pipeline: MiniMaxH3Pipeline instance
        kf_specs: list of {"image": path, "position": str} dicts
        frame_num: total frames in this shot
        height, width: target resolution
    
    Returns: list of (image_tensor, latent_frame_index) tuples
    """
    import torchvision.io as tv_io
    import torch.nn.functional as F
    
    results = []
    for spec in kf_specs:
        img_path = spec.get("image")
        pos_str = spec.get("position", "50%")
        
        # Load image as [C, H, W] float 0-1
        img = tv_io.read_image(img_path).float() / 255.0  # [C, H, W]
        
        # Resize to target (need batch dim for interpolate, then squeeze back)
        img = F.interpolate(img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        # img is now [C, H, W] — _as_video will handle CTHW conversion
        
        # Normalize to [-1, 1] range expected by VAE
        img = img * 2.0 - 1.0
        
        # Parse position to latent frame index
        pixel_indices = _parse_keyframe_positions(pos_str, frame_num)
        for pixel_idx in pixel_indices:
            latent_idx = max(1, round(pixel_idx / 3.375))
            results.append((img, latent_idx))
            print(f"  [InteriorKF] Loaded {img_path} at pixel frame {pixel_idx} -> latent {latent_idx}", flush=True)
    
    return results


def _inject_mid_keyframes(pipeline, keyframes_spec, height, width):
    """Process keyframe specs into (frame_index, tensor) pairs."""
    if not keyframes_spec:
        return None
    
    from PIL import Image
    
    processed = []
    for kf in keyframes_spec:
        frame_idx = int(kf.get("frame", 0))
        img = kf.get("image")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
            target_h, target_w = height, width
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_tensor = _pil_to_tensor(img)
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            continue
        processed.append((frame_idx, img_tensor))
        print(f"  [Keyframe] Anchoring frame {frame_idx}", flush=True)
    
    return processed


def _generate_with_keyframes(pipeline, gen_kwargs, mid_keyframes):
    """Generate with keyframe anchoring including interior (mid-frame) keyframes.
    
    Supports image_start (frame 0), image_end (last frame), AND interior
    keyframes at any frame position via the patched packing.py.
    Interior keyframes are passed as (image_tensor, latent_frame_index) tuples.
    """
    if not mid_keyframes:
        return pipeline.generate(**gen_kwargs)
    
    frame_num = gen_kwargs.get("frame_num", 124)
    
    # Separate into boundary vs interior keyframes
    start_kfs = [kf for kf in mid_keyframes if kf[0] == 0]
    end_kfs = [kf for kf in mid_keyframes if kf[0] >= frame_num - 2]
    interior_kfs = [kf for kf in mid_keyframes if 0 < kf[0] < frame_num - 2]
    
    # Boundary keyframes use existing image_start/image_end
    if start_kfs and gen_kwargs.get("image_start") is None:
        gen_kwargs["image_start"] = start_kfs[0][1]
        print(f"  [Keyframe] Using frame {start_kfs[0][0]} as image_start anchor", flush=True)
    
    if end_kfs and gen_kwargs.get("image_end") is None:
        gen_kwargs["image_end"] = end_kfs[-1][1]
        print(f"  [Keyframe] Using frame {end_kfs[-1][0]} as image_end anchor", flush=True)
    
    # Interior keyframes: convert pixel frame index to latent frame index
    # H3 uses ~3.375 pixel frames per latent frame (FRAME_PER_TOKEN pattern)
    # Approximate: latent_idx = pixel_frame // 3 (rounded)
    if interior_kfs:
        latent_interior = []
        for pixel_frame, image_tensor in interior_kfs:
            latent_idx = max(1, round(pixel_frame / 3.375))
            latent_interior.append((image_tensor, latent_idx))
            print(f"  [Keyframe] Interior anchor: pixel frame {pixel_frame} -> latent frame {latent_idx}", flush=True)
        gen_kwargs["interior_keyframes"] = latent_interior
    
    return pipeline.generate(**gen_kwargs)


def generate_multishot(pipeline, settings, callback=None, set_progress_status=None):
    """Generate a multishot video by chaining shots together.

    Args:
        pipeline: MiniMaxH3Pipeline instance
        settings: dict with keys:
            - script (str): multi-shot prompt (--- separated or JSON)
            - shot_count (int): 0 = auto from script, 1-8 forces count
            - width, height (int): output resolution
            - frames_per_shot (int): frames per shot (default 243)
            - seed (int): base seed
            - steps (int): inference steps (default 20)
            - image_start (optional): first frame for shot 1
            - shift (float): flow shift (default 12.0)
            - seed_per_shot (bool): vary seed per shot (default True)
        callback: optional progress callback
        set_progress_status: optional status setter

    Returns:
        dict with "x" (video tensor), "audio" (numpy array), "audio_sampling_rate"
    """
    script = settings.get("script", "")
    shots = parse_script(script)
    if not shots:
        raise ValueError("Multishot requires a non-empty script")

    shot_count = int(settings.get("shot_count", 0))
    n = shot_count if shot_count > 0 else len(shots)
    if len(shots) > n:
        print(f"[Multishot] dropping {len(shots) - n} extra script prompt(s) "
              f"(shot_count={n}).", flush=True)
        shots = shots[:n]
    while len(shots) < n:
        print(f"[Multishot] shot {len(shots) + 1} continues the last prompt.",
              flush=True)
        shots.append(shots[-1])

    width = int(settings.get("width", 1344))
    height = int(settings.get("height", 768))
    frames_per_shot = int(settings.get("frames_per_shot", 243))
    base_seed = int(settings.get("seed", 0))
    
    # Memory/presenter mode for long chains
    memory_mode = bool(settings.get("memory_mode", False))
    memory_frames_count = int(settings.get("memory_frames", 2))
    anchor_frames_count = int(settings.get("anchor_frames", 1))
    
    # Condition strength controls
    visual_cond_strength = float(settings.get("visual_cond_strength", 0.999))
    audio_cond_strength = float(settings.get("audio_cond_strength", 1.0))
    
    # User-provided interior keyframes (single-pass)
    user_interior_kfs = settings.get("interior_keyframes", [])
    steps = int(settings.get("steps", 20))
    shift = float(settings.get("shift", 12.0))
    seed_per_shot = settings.get("seed_per_shot", True)

    # Optional start image for shot 1
    image_start = settings.get("image_start")
    if isinstance(image_start, str) and image_start:
        img = Image.open(image_start).convert("RGB")
        target_h, target_w = height, width
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        image_start = _pil_to_tensor(img)

    # Pre-parse all shots to check if any have keyframes
    parsed_shots = []
    has_keyframes = False
    for prompt in shots:
        if isinstance(prompt, dict):
            raw_kfs = prompt.get("keyframes", [])
            prompt_text = prompt.get("prompt", "")
            kfs = [{"frame": int(kf.get("frame", 0)), "image": kf.get("image")} for kf in raw_kfs]
            parsed_shots.append({"prompt": prompt_text, "keyframes": kfs})
            if kfs:
                has_keyframes = True
        else:
            parsed_shots.append({"prompt": prompt, "keyframes": []})

    # === PASS 1: Generate all shots (no keyframes) ===
    pass_label = "Pass 1/2" if has_keyframes else ""
    frames_parts = []
    audio_parts = []
    pass1_videos = []  # Cache per-shot videos for keyframe extraction
    prev_last_frame = image_start

    # Memory bank for presenter mode
    memory_bank = []  # recent shot-end frames
    identity_anchor = None  # persistent first-shot frame

    for si, shot_data in enumerate(parsed_shots):
        prompt = shot_data["prompt"]
        print(f"[Multishot] {pass_label} shot {si + 1}/{n} ({frames_per_shot}f @ "
              f"{width}x{height})...", flush=True)

        if set_progress_status:
            set_progress_status(f"{'Pass 1: ' if has_keyframes else ''}Shot {si + 1}/{n}")

        shot_seed = (base_seed + si) if seed_per_shot else base_seed

        gen_kwargs = dict(
            input_prompt=prompt,
            image_start=prev_last_frame,
            frame_num=frames_per_shot,
            height=height,
            width=width,
            shift=shift,
            sampling_steps=steps,
            seed=shot_seed,
            callback=callback,
            set_progress_status=set_progress_status,
        )
        
        # Inject user-provided interior keyframes (single-pass)
        if user_interior_kfs:
            interior = _load_and_encode_interior_keyframes(
                pipeline, user_interior_kfs, frames_per_shot, height, width)
            if interior:
                gen_kwargs["interior_keyframes"] = interior
        
        # Memory mode: augment prompt with identity consistency instruction
        # FL2VA doesn't support input_ref_images (needs Ref2VA), so we use
        # prompt augmentation + image_start chaining for identity preservation
        if memory_mode and si > 0 and identity_anchor is not None:
            anchor_prefix = "Maintain exact same character appearance, clothing, voice, and setting as previous shots. "
            gen_kwargs["input_prompt"] = anchor_prefix + gen_kwargs.get("input_prompt", prompt)
            print(f"  [Memory] Shot {si+1}: identity anchor in prompt "
                  f"(bank={len(memory_bank)} frames)", flush=True)

        result = pipeline.generate(**gen_kwargs)
        if result is None:
            raise RuntimeError(f"Shot {si + 1} generation was interrupted")

        decoded_video = result["x"]  # CTHW tensor
        decoded_audio = result["audio"]  # numpy (samples, channels)

        # Cache raw video for keyframe extraction (before seam trimming)
        pass1_videos.append(decoded_video.clone())

        # Convert audio to torch for processing
        audio_tensor = torch.from_numpy(decoded_audio).transpose(0, 1).float()

        # Trim seam: remove first frame + matching audio for shots after the first
        if si > 0:
            decoded_video = decoded_video[:, 1:]  # drop duplicated first frame
            trim_samples = int(round(AUDIO_SAMPLE_RATE / FPS))
            audio_tensor = audio_tensor[..., trim_samples:]

        frames_parts.append(decoded_video.cpu())
        audio_parts.append(audio_tensor.cpu())

        # Last frame becomes next shot's start image
        prev_last_frame = decoded_video[:, -1:].clone()

        # Update memory bank for presenter mode
        if memory_mode:
            last_frame = decoded_video[:, -1:]  # [B, 1, H, W, C] or similar
            memory_bank.append(last_frame.clone())
            if si == 0 and identity_anchor is None:
                identity_anchor = decoded_video[:, 0:1].clone()
                print(f"  [Memory] Identity anchor set from shot 1 frame 0", flush=True)

        print(f"[Multishot] {pass_label} shot {si + 1} done: {decoded_video.shape[1]} frames",
              flush=True)

    # === PASS 2: Re-generate shots with keyframes using Pass 1 anchors ===
    if has_keyframes:
        print("[Multishot] === Pass 2/2: Re-generating anchored shots ===", flush=True)
        
        # Reset for pass 2 — rebuild from scratch with anchors
        frames_parts_p2 = []
        audio_parts_p2 = []
        prev_last_frame_p2 = image_start

        for si, shot_data in enumerate(parsed_shots):
            prompt = shot_data["prompt"]
            kfs = shot_data["keyframes"]

            if not kfs:
                # No keyframes — reuse Pass 1 output
                # But we need to recalculate prev_last_frame chain
                p1_vid = pass1_videos[si]
                trimmed = p1_vid[:, 1:] if si > 0 else p1_vid
                audio_t = audio_parts[si]
                
                frames_parts_p2.append(trimmed.cpu())
                audio_parts_p2.append(audio_t.cpu())
                prev_last_frame_p2 = trimmed[:, -1:].clone()
                print(f"[Multishot] Pass 2: shot {si + 1}/{n} reused from Pass 1", flush=True)
                continue

            # Extract anchor frames from Pass 1 video for this shot
            p1_video = pass1_videos[si]
            total_frames = p1_video.shape[1]
            
            processed_kfs = []
            for kf in kfs:
                frame_idx = min(int(kf["frame"]), total_frames - 1)
                anchor_frame = p1_video[:, frame_idx:frame_idx+1].clone()
                processed_kfs.append((frame_idx, anchor_frame))
                print(f"  [Keyframe] Extracted anchor from Pass 1 frame {frame_idx}", flush=True)

            shot_seed = (base_seed + si) if seed_per_shot else base_seed

            gen_kwargs = dict(
                input_prompt=prompt,
                image_start=prev_last_frame_p2,
                frame_num=frames_per_shot,
                height=height,
                width=width,
                shift=shift,
                sampling_steps=steps,
                seed=shot_seed,
                callback=callback,
                set_progress_status=set_progress_status,
            )

            # Inject keyframes from Pass 1
            result = _generate_with_keyframes(pipeline, gen_kwargs, processed_kfs)
            if result is None:
                raise RuntimeError(f"Pass 2 shot {si + 1} generation was interrupted")

            decoded_video = result["x"]
            decoded_audio = result["audio"]
            audio_tensor = torch.from_numpy(decoded_audio).transpose(0, 1).float()

            if si > 0:
                decoded_video = decoded_video[:, 1:]
                trim_samples = int(round(AUDIO_SAMPLE_RATE / FPS))
                audio_tensor = audio_tensor[..., trim_samples:]

            frames_parts_p2.append(decoded_video.cpu())
            audio_parts_p2.append(audio_tensor.cpu())
            prev_last_frame_p2 = decoded_video[:, -1:].clone()

            print(f"[Multishot] Pass 2: shot {si + 1}/{n} re-generated with {len(processed_kfs)} anchor(s)",
                  flush=True)

        # Use Pass 2 results
        frames_parts = frames_parts_p2
        audio_parts = audio_parts_p2

    # Concatenate all frames
    master_frames = torch.cat(frames_parts, dim=1)

    # Crossfade audio at seams
    master_audio = xfade_audio(audio_parts, AUDIO_SAMPLE_RATE)

    # Convert back to numpy for output
    total_samples = round(master_frames.shape[1] / FPS * AUDIO_SAMPLE_RATE)
    if master_audio.shape[-1] > total_samples:
        master_audio = master_audio[..., :total_samples]
    elif master_audio.shape[-1] < total_samples:
        master_audio = torch.nn.functional.pad(
            master_audio, (0, total_samples - master_audio.shape[-1]))

    audio_numpy = master_audio.transpose(0, 1).float().cpu().numpy()

    print(f"[Multishot] done: {n} shots, {master_frames.shape[1]} frames "
          f"(~{master_frames.shape[1] / FPS:.1f}s).", flush=True)

    return {
        "x": master_frames,
        "audio": audio_numpy,
        "audio_sampling_rate": AUDIO_SAMPLE_RATE,
    }
