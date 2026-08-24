from __future__ import annotations

import os
import sys
from typing import Any, Callable

from PIL import Image

from postprocessing.spatial_upsamplers import PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_POSTPROCESSING, POSTPROCESSING_CATEGORY_REFINER, SimpleScaleSuffixMixin, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING

from .runtime import DEFAULT_MODEL_TYPE, DEFAULT_WINDOW_FRAMES, DEFAULT_WINDOW_OVERLAP, MAX_WINDOW_FRAMES, RUNTIME_NAME, TEMPORAL_STRIDE, TEXT_ENCODER_CONFIG


REFERENCE_BINDING = """subject_definitions:
<Subject 1> is the person in <Picture 1>, preserving their exact identity, facial features, skin tone, hairstyle, and expression."""
DEFAULT_PROMPT = """A stabilized close-up crop of the single tracked person already visible in the control video. Preserve that exact person's identity, facial proportions, expression, gaze, head pose, hair, clothing, motion, framing, lighting, and background. Restore natural, sharp, temporally stable eyes, mouth, teeth, skin, and fine facial detail. Do not introduce any person, duplicate head, object, or scene element."""
INJECT_REFERENCE_IMAGE = True
METHOD = "h3facerefine"
ASSET_FOLDER = "buffalo_l"
DETECTOR_FILE = "face_yolov8m.pt"
FALLBACK_DETECTOR_FILE = "person_yolov8m-seg.safetensors"
DETECTOR_PATH = f"{ASSET_FOLDER}/{DETECTOR_FILE}"
FALLBACK_DETECTOR_PATH = f"{ASSET_FOLDER}/{FALLBACK_DETECTOR_FILE}"
INSIGHTFACE_FILES = ("det_10g.onnx", "2d106det.onnx", "w600k_r50.onnx")


def _face_progress_callback(callback, face_index: int, face_count: int):
    if face_count <= 1 or not callable(callback):
        return callback
    prefix = f"{face_index + 1}/{face_count} Faces, "
    return lambda phase, current=None, total=None: callback(prefix + phase, current, total)


class H3FaceRefinerBridge(SimpleScaleSuffixMixin):
    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "model_mode": "h3",
            "detector_confidence": 0.25,
            "crop_factor": 1.6,
            "canvas_mode": "auto_no_downscale",
            "canvas_width": 512,
            "canvas_height": 512,
            "smooth_window": 21,
            "size_smooth_window": 51,
            "smooth_method": "gaussian",
            "size_mode": "per_frame",
            "identity_threshold": 0.28,
            "auto_min_face_height": 32,
            "auto_min_presence": 0.2,
            "fallback_detector": FALLBACK_DETECTOR_FILE,
            "fallback_head_frac": 0.5,
            "denoising_strength": 0.45,
            "sampling_steps": 4,
            "flow_shift": 12.0,
            "sample_solver": "er_sde",
            "turbo_lora_strength": 1.0,
            "strength": 0.75,
            "paste_region": "face_only",
            "mask_dilation": 24,
            "feather": 24,
            "colour_match": 1.0,
            "blend": 1.0,
            "undetected_frames": "fade_out",
            "feather_scales_with_crop": False,
            "window_size": DEFAULT_WINDOW_FRAMES,
            "window_overlap": DEFAULT_WINDOW_OVERLAP,
        }

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        values = cls.default_config()
        values.update(config or {})
        values.pop("version", None)

        def number(key, cast):
            try:
                values[key] = cast(values[key])
            except (TypeError, ValueError):
                values[key] = cls.default_config()[key]

        for key in ("canvas_width", "canvas_height", "smooth_window", "size_smooth_window", "auto_min_face_height", "sampling_steps",
                    "mask_dilation", "feather", "window_size", "window_overlap"):
            number(key, int)
        for key in ("detector_confidence", "crop_factor", "identity_threshold", "auto_min_presence", "denoising_strength", "flow_shift",
                    "turbo_lora_strength", "strength", "fallback_head_frac",
                    "colour_match", "blend"):
            number(key, float)
        if values["model_mode"] not in ("h3", "current"):
            values["model_mode"] = "h3"
        if values["canvas_mode"] not in ("manual", "auto_no_downscale", "auto_capped_768"):
            values["canvas_mode"] = "manual"
        if values["smooth_method"] not in ("gaussian", "savgol", "moving_average"):
            values["smooth_method"] = "gaussian"
        if values["size_mode"] not in ("per_frame", "max_of_clip"):
            values["size_mode"] = "per_frame"
        if values["fallback_detector"] not in ("none", FALLBACK_DETECTOR_FILE):
            values["fallback_detector"] = "none"
        if values["sample_solver"] not in ("er_sde", "euler", "res_multistep", "ralston_2s"):
            values["sample_solver"] = "er_sde"
        if values["paste_region"] not in ("face_only", "face_ellipse", "full_crop"):
            values["paste_region"] = "face_only"
        if values["undetected_frames"] not in ("fade_out", "skip", "composite_anyway"):
            values["undetected_frames"] = "fade_out"
        for key in ("identity_threshold", "denoising_strength", "turbo_lora_strength", "strength", "colour_match", "blend"):
            values[key] = max(0.0, min(1.0, values[key]))
        values["detector_confidence"] = max(0.05, min(0.95, values["detector_confidence"]))
        values["crop_factor"] = max(1.2, min(8.0, values["crop_factor"]))
        values["auto_min_face_height"] = max(8, min(512, values["auto_min_face_height"]))
        values["auto_min_presence"] = max(0.0, min(1.0, values["auto_min_presence"]))
        values["flow_shift"] = max(0.01, values["flow_shift"])
        values["fallback_head_frac"] = max(0.0, min(1.5, values["fallback_head_frac"]))
        for key in ("canvas_width", "canvas_height"):
            values[key] = max(128, min(1344, round(int(values[key]) / 32) * 32))
        for key in ("smooth_window", "size_smooth_window"):
            values[key] = max(1, values[key])
        values["mask_dilation"] = max(0, min(256, values["mask_dilation"]))
        values["feather"] = max(0, min(256, values["feather"]))
        values["window_size"] = max(5, min(MAX_WINDOW_FRAMES, int(values["window_size"])))
        values["window_size"] = ((values["window_size"] - 5) // TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 5
        values["window_overlap"] = max(1, min(int(values["window_overlap"]), values["window_size"] - 1))
        values["sampling_steps"] = max(1, int(values["sampling_steps"]))
        return values

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api

        return upsampler_api.read_config_section(self.server_config, self)

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": RUNTIME_NAME,
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video",),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "h3_face_refiner",
            "pos": 44,
            "methods": [("H3 Face Refiner", METHOD)],
            "vae_methods": [],
            "default_spatial_upsampling": METHOD,
            "postprocessing_category": POSTPROCESSING_CATEGORY_REFINER,
            "default_prompt": DEFAULT_PROMPT,
            "source_audio_conditioning": True,
            "description": "Detect and identity-track up to five faces through a video, refine stabilized crops with MiniMax H3 Ref2VA, and feather-stitch them back without changing output resolution. A uniform refinement strength controls source fidelity, while sliding windows keep longer clips temporally coherent.",
            "method_parameters": {METHOD: [
                {"name": "spatial_upsampler_face_count", "setting": "face_count", "type": "integer", "component": "slider", "ui": (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING), "required": False, "default": 1, "minimum": 0, "maximum": 5, "step": 1, "label": "Faces to Refine (0 = Auto, up to 5)", "description": "Choose an exact maximum from 1 to 5, or 0 for automatic selection. Runtime increases approximately linearly with the selected face count."},
                {"name": "spatial_upsampler_prompt", "setting": "prompt", "type": "string", "component": "textbox", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": False, "default": "", "label": "Refiner Prompt", "description": "Optional description of the source clip and desired facial restoration. The source metadata prompt or a neutral restoration prompt is used when omitted.", "lines": 1},
                {"name": "spatial_upsampler_reference_images", "setting": "reference_images", "type": "array", "component": "images", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": False, "default": [], "label": "Optional Face Reference Images", "description": "Optional reference-image media ids. InsightFace matches them to detected identities irrespective of order; unmatched tracks use an automatically selected source frame.", "multiple": True, "media_type": "image"},
            ]},
        }

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Group():
            model_mode = gr.Dropdown(choices=[("Force/reuse H3", "h3"), ("Use current compatible model, fall back to H3", "current")], value=config["model_mode"], label="Face Refiner Model", interactive=not lock_config)
            with gr.Row():
                detector_confidence = gr.Slider(0.05, 0.95, value=config["detector_confidence"], step=0.05, label="Face Detection Confidence", interactive=not lock_config)
                identity_threshold = gr.Slider(0.0, 1.0, value=config["identity_threshold"], step=0.01, label="Identity Similarity Threshold", interactive=not lock_config)
                auto_min_face_height = gr.Slider(8, 512, value=config["auto_min_face_height"], step=1, label="Auto Minimum Face Height", info="Minimum maximum detected face-box height, in source pixels, for automatic selection.", interactive=not lock_config)
                auto_min_presence = gr.Slider(0.0, 1.0, value=config["auto_min_presence"], step=0.05, label="Auto Minimum Frame Presence", info="Minimum fraction of source frames in which the identity must be detected.", interactive=not lock_config)
            with gr.Row():
                crop_factor = gr.Slider(1.2, 8.0, value=config["crop_factor"], step=0.1, label="Face Crop Factor", interactive=not lock_config)
                canvas_mode = gr.Dropdown(choices=[("Manual", "manual"), ("Auto, no downscale", "auto_no_downscale"), ("Auto, capped at 768", "auto_capped_768")], value=config["canvas_mode"], label="Crop Canvas", interactive=not lock_config)
                canvas_width = gr.Slider(128, 1344, value=config["canvas_width"], step=32, label="Manual Canvas Width", interactive=not lock_config)
                canvas_height = gr.Slider(128, 1344, value=config["canvas_height"], step=32, label="Manual Canvas Height", interactive=not lock_config)
            with gr.Row():
                smooth_window = gr.Slider(1, 201, value=config["smooth_window"], step=2, label="Position Smoothing", interactive=not lock_config)
                size_smooth_window = gr.Slider(1, 201, value=config["size_smooth_window"], step=2, label="Size Smoothing", interactive=not lock_config)
                smooth_method = gr.Dropdown(choices=[("Gaussian", "gaussian"), ("Savitzky-Golay", "savgol"), ("Moving average", "moving_average")], value=config["smooth_method"], label="Smoothing Method", interactive=not lock_config)
                size_mode = gr.Dropdown(choices=[("Per frame", "per_frame"), ("Maximum of clip", "max_of_clip")], value=config["size_mode"], label="Crop Size Mode", interactive=not lock_config)
            with gr.Row():
                fallback_detector = gr.Dropdown(choices=[("None", "none"), ("Person YOLOv8 fallback", FALLBACK_DETECTOR_FILE)], value=config["fallback_detector"], label="Missing-face Fallback", interactive=not lock_config)
                fallback_head_frac = gr.Slider(0.0, 1.5, value=config["fallback_head_frac"], step=0.05, label="Fallback Head Position", interactive=not lock_config)
            with gr.Row():
                denoising_strength = gr.Slider(0.0, 1.0, value=config["denoising_strength"], step=0.01, label="BasicScheduler Denoise", interactive=not lock_config)
                sampling_steps = gr.Slider(1, 30, value=config["sampling_steps"], step=1, label="Refinement Steps", interactive=not lock_config)
                flow_shift = gr.Slider(1.0, 20.0, value=config["flow_shift"], step=0.1, label="Flow Shift", interactive=not lock_config)
                sample_solver = gr.Dropdown(choices=[("ER-SDE (upstream workflow)", "er_sde"), ("Euler", "euler"), ("RES Multistep", "res_multistep"), ("Ralston 2S", "ralston_2s")], value=config["sample_solver"], label="Sampler", interactive=not lock_config)
                turbo_lora_strength = gr.Slider(0.0, 2.0, value=config["turbo_lora_strength"], step=0.05, label="Turbo LoRA Strength", interactive=not lock_config)
                strength = gr.Slider(0.0, 1.0, value=config["strength"], step=0.01, label="Refinement Strength", interactive=not lock_config)
            with gr.Row():
                paste_region = gr.Dropdown(choices=[("Face rectangle", "face_only"), ("Face ellipse", "face_ellipse"), ("Full crop", "full_crop")], value=config["paste_region"], label="Paste Region", interactive=not lock_config)
                mask_dilation = gr.Slider(0, 256, value=config["mask_dilation"], step=2, label="Mask Dilation", interactive=not lock_config)
                feather = gr.Slider(0, 256, value=config["feather"], step=2, label="Mask Feather", interactive=not lock_config)
                colour_match = gr.Slider(0.0, 1.0, value=config["colour_match"], step=0.05, label="Colour Match", interactive=not lock_config)
                blend = gr.Slider(0.0, 1.0, value=config["blend"], step=0.05, label="Blend", interactive=not lock_config)
            with gr.Row():
                undetected_frames = gr.Dropdown(choices=[("Fade out", "fade_out"), ("Skip", "skip"), ("Composite anyway", "composite_anyway")], value=config["undetected_frames"], label="Undetected Frames", interactive=not lock_config)
                feather_scales_with_crop = gr.Checkbox(value=config["feather_scales_with_crop"], label="Legacy Canvas-relative Feather", interactive=not lock_config)
            with gr.Row():
                window_size = gr.Slider(5, MAX_WINDOW_FRAMES, value=config["window_size"], step=TEMPORAL_STRIDE, label="H3 Refiner Sliding Window Size", info="Frames processed per H3 refinement window. Larger windows use more VRAM.", interactive=not lock_config)
                window_overlap = gr.Slider(1, config["window_size"] - 1, value=config["window_overlap"], step=1, label="H3 Refiner Sliding Window Overlap", info="Frames shared and crossfaded between consecutive windows.", interactive=not lock_config)

        def update_overlap_limit(new_window_size, current_overlap):
            maximum = int(new_window_size) - 1
            return gr.update(maximum=maximum, value=min(int(current_overlap), maximum))

        window_size.change(update_overlap_limit, inputs=[window_size, window_overlap], outputs=window_overlap)
        controls = locals().copy()
        field_names = tuple(self.default_config())
        return [(name, controls[name]) for name in field_names if name in controls]

    def enabled(self) -> bool:
        return True

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        if image_mode:
            return "H3 Face Refiner is available for videos only"
        if split is None or split[0] != METHOD:
            return "Invalid H3 Face Refiner selection"
        return ""

    def supports_loaded_model(self, spatial_upsampling, context, *, reference_images=None, **_kwargs) -> bool:
        if self.validate_upsampling(spatial_upsampling, 0) or getattr(context.model, "refinement_api", None) != "masked_video_sigma_v1":
            return False
        if self.config()["model_mode"] == "current":
            return True
        return context.model_family == "minimax_h3"

    def query_download_defs(self) -> list[dict[str, Any]]:
        from models.minimax_h3.minimax_h3_handler import REF2VA_PRUNED_ARCHITECTURE, REPO_ID, family_handler

        model_def = family_handler.query_model_def(REF2VA_PRUNED_ARCHITECTURE, {})
        return family_handler.query_model_files(lambda filename: [os.path.basename(filename)], REF2VA_PRUNED_ARCHITECTURE, model_def) + [{
            "repoId": REPO_ID,
            "sourceFolderList": [ASSET_FOLDER],
            "fileList": [[DETECTOR_FILE, FALLBACK_DETECTOR_FILE, *INSIGHTFACE_FILES]],
        }]

    @staticmethod
    def _private_model_def(wgp):
        model_def = wgp.get_model_def(DEFAULT_MODEL_TYPE).copy()
        model_def.update(model_def["system_configs"][TEXT_ENCODER_CONFIG])
        return model_def

    def _turbo_path(self, wgp, model_type=DEFAULT_MODEL_TYPE):
        from models.minimax_h3.minimax_h3_handler import REF_TURBO_LORA_KEY

        model_def = wgp.get_model_def(model_type)
        return wgp.get_lora_local_path(wgp.get_lora_dir(model_type), model_def[REF_TURBO_LORA_KEY])

    def _ensure_turbo(self, wgp):
        from models.minimax_h3.minimax_h3_handler import REF_TURBO_LORA_KEY

        model_def = self._private_model_def(wgp)
        path = self._turbo_path(wgp)
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            wgp.download_file(model_def[REF_TURBO_LORA_KEY], path)
        return path

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        wgp = sys.modules.get("wgp")
        if wgp is None:
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        for download_def in self.query_download_defs():
            process_files(**download_def)
        model_def = self._private_model_def(wgp)
        transformer = wgp.get_model_filename(DEFAULT_MODEL_TYPE, wgp.transformer_quantization, wgp.transformer_dtype_policy, model_def=model_def)
        wgp.download_models(transformer, DEFAULT_MODEL_TYPE, 0, -1, model_def=model_def)
        text_encoder_urls = wgp.get_model_recursive_prop(DEFAULT_MODEL_TYPE, "text_encoder_URLs", return_list=True, model_def=model_def)
        text_encoder = wgp.get_model_filename(DEFAULT_MODEL_TYPE, wgp.text_encoder_quantization, wgp.transformer_dtype_policy, URLs=text_encoder_urls)
        wgp.download_models(text_encoder, DEFAULT_MODEL_TYPE, 2, -1, force_path=model_def["text_encoder_folder"], model_def=model_def)
        self._ensure_turbo(wgp)
        return True

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], profile, **kwargs):
        error = self.validate_upsampling(spatial_upsampling, 0)
        if error:
            raise ValueError(error)
        self.download(process_files, spatial_upsampling=spatial_upsampling)

    def upscale(self, sample, spatial_upsampling, *, loaded_model_context=None, prompt="", seed=0, fps=24.0, frame_offset=0,
                audio_waveform=None, audio_sample_rate=0, source_audio_path=None, vae_tile_size=None, still_image=False,
                reference_images=None, image_refs_relative_size=100.0, face_count=1,
                abort_callback=None, progress_callback=None, profile=-1, **kwargs):
        if still_image:
            raise ValueError("H3 Face Refiner is available for videos only")
        error = self.validate_upsampling(spatial_upsampling, 0)
        if error:
            raise ValueError(error)
        import wgp
        from .face import crop_face_track, frames_to_sample, sample_to_frames, select_reference_frame, stitch, track_faces
        from .runtime import RUNTIME, load_model, refine_video

        config = self.config()
        if config["denoising_strength"] == 0.0 or config["blend"] == 0.0 or config["strength"] == 0.0:
            return sample, None
        frames, source_format = sample_to_frames(sample)
        reference_images = wgp.clean_image_list(reference_images or []) or []
        detector_path = self.files_locator.locate_file(DETECTOR_PATH)
        fallback_detector_path = None if config["fallback_detector"] == "none" else self.files_locator.locate_file(FALLBACK_DETECTOR_PATH)
        insightface_model_path = self.files_locator.locate_file(f"{ASSET_FOLDER}/{INSIGHTFACE_FILES[0]}")
        insightface_model_dir = os.path.dirname(os.path.abspath(insightface_model_path))
        tracked_faces = track_faces(frames, detector_path, face_count=max(0, min(5, int(face_count))), reference_images=reference_images,
                                    confidence=config["detector_confidence"], crop_factor=config["crop_factor"],
                                    canvas_width=config["canvas_width"], canvas_height=config["canvas_height"], canvas_mode=config["canvas_mode"],
                                    smooth_window=config["smooth_window"], size_smooth_window=config["size_smooth_window"], smooth_method=config["smooth_method"],
                                    size_mode=config["size_mode"], identity_threshold=config["identity_threshold"], auto_min_face_height=config["auto_min_face_height"],
                                    auto_min_presence=config["auto_min_presence"], fallback_detector_path=fallback_detector_path, insightface_model_dir=insightface_model_dir,
                                    fallback_head_frac=config["fallback_head_frac"], strength_small_face=config["strength"],
                                    strength_large_face=config["strength"], strength_smooth_frames=1, abort_callback=abort_callback,
                                    progress_callback=progress_callback)
        if tracked_faces is None:
            return None, None
        if not tracked_faces:
            print("[H3FaceRefine] Auto selection found no relevant face tracks; leaving the source video unchanged")
            return sample, None
        hybrid_h3 = loaded_model_context is not None and loaded_model_context.model_family == "minimax_h3" and not loaded_model_context.model.reference_mode
        if loaded_model_context is None or hybrid_h3:
            load_model(loaded_model_context if hybrid_h3 else None)
            pipeline = RUNTIME.model
            model_type = DEFAULT_MODEL_TYPE
        else:
            pipeline = loaded_model_context.model
            model_type = loaded_model_context.base_model_type
            if loaded_model_context.model_family == "minimax_h3":
                print(f"[H3FaceRefine] Reusing loaded H3 Ref2VA model '{loaded_model_context.model_type}' and its MMGP offload object")
        reference_mode = bool(getattr(pipeline, "reference_mode", False))
        if reference_mode and not INJECT_REFERENCE_IMAGE:
            print("[H3FaceRefine] Reference image injection disabled; conditioning Ref2VA from the tracked control video and its audio track")
        turbo_path = self._ensure_turbo(wgp) if loaded_model_context is None or loaded_model_context.model_family == "minimax_h3" else None
        refinement_prompt = prompt or DEFAULT_PROMPT
        print(f"[H3FaceRefine] Prompt: {refinement_prompt}")
        try:
            track_jobs = []
            for track_index, (transform, strengths, matched_reference) in enumerate(tracked_faces):
                track_references = [matched_reference] if INJECT_REFERENCE_IMAGE and matched_reference is not None else None
                if INJECT_REFERENCE_IMAGE and track_references is None and reference_mode:
                    reference_frame = select_reference_frame(frames, transform["raw_face_boxes"], insightface_model_dir=insightface_model_dir)
                    reference_crop = crop_face_track(frames, transform, reference_frame, reference_frame + 1, uint8_storage=True)[:, 0].permute(1, 2, 0).numpy()
                    track_references = [Image.fromarray(reference_crop)]
                    print(f"[H3FaceRefine] Face track {track_index + 1}: using stabilized single-face crop from pose-aware source frame {reference_frame + 1} as reference")
                track_prompt = refinement_prompt if not track_references or "<Picture 1>" in refinement_prompt else REFERENCE_BINDING + "\n" + refinement_prompt
                track_jobs.append((transform, strengths, track_references, track_prompt))
            if hasattr(pipeline, "prewarm_refinement_prompt"):
                print(f"[H3FaceRefine] Pre-encoding prompt/reference conditioning for {len(track_jobs)} face track(s) before denoising")
                for transform, _strengths, track_references, track_prompt in track_jobs:
                    canvas_width, canvas_height = transform["canvas"]
                    pipeline.prewarm_refinement_prompt(track_prompt, track_references if reference_mode else None, canvas_width, canvas_height,
                                                       image_refs_relative_size=image_refs_relative_size)

            source_frames = frames
            output = frames.clone()
            for track_index, (transform, strengths, track_references, track_prompt) in enumerate(track_jobs):
                track_progress_callback = _face_progress_callback(progress_callback, track_index, len(track_jobs))
                segments = transform["segments"]
                excluded = len(transform["active"]) - sum(stop - start for start, stop in segments)
                print(f"[H3FaceRefine] Face track {track_index + 1}: refining {len(segments)} presence segment(s); excluded {excluded} absent frame(s) from H3 temporal attention")
                for segment_index, (start, stop) in enumerate(segments, start=1):
                    print(f"[H3FaceRefine] Face track {track_index + 1}, segment {segment_index}/{len(segments)}: source frames {start + 1}-{stop}")
                    uniform_strengths = strengths.new_full((stop - start,), config["strength"])
                    def load_crop_window(window_start, window_stop):
                        return crop_face_track(source_frames, transform, start + window_start, start + window_stop)

                    refined = refine_video(None, uniform_strengths, pipeline=pipeline, model_type=model_type, turbo_path=turbo_path,
                                           turbo_multiplier=config["turbo_lora_strength"], prompt=track_prompt,
                                           denoising_strength=config["denoising_strength"], sampling_steps=config["sampling_steps"], shift=config["flow_shift"],
                                           sample_solver=config["sample_solver"], seed=int(seed) + track_index, fps=fps, window_size=config["window_size"],
                                           window_overlap=config["window_overlap"], frame_offset=frame_offset + start, audio_waveform=audio_waveform,
                                           audio_sample_rate=audio_sample_rate, source_audio_path=source_audio_path,
                                           reference_images=track_references if reference_mode else None,
                                           image_refs_relative_size=image_refs_relative_size, vae_tile_size=vae_tile_size,
                                           abort_callback=abort_callback, progress_callback=track_progress_callback,
                                           frame_count=stop - start, video_window_loader=load_crop_window)
                    if refined is None:
                        return None, None
                    segment_transform = transform.copy()
                    for key in ("boxes", "face_rect", "weights", "detected", "active"):
                        segment_transform[key] = transform[key][start:stop]
                    segment_transform["frames"] = stop - start
                    stitched = stitch(output[start:stop], refined, segment_transform, paste_region=config["paste_region"], mask_dilation=config["mask_dilation"],
                                      feather=config["feather"], colour_match=config["colour_match"], blend=config["blend"],
                                      undetected_frames=config["undetected_frames"], feather_scales_with_crop=config["feather_scales_with_crop"],
                                      abort_callback=abort_callback, progress_callback=track_progress_callback)
                    if stitched is None:
                        return None, None
                    refined = stitched = None
                if track_index + 1 == len(track_jobs):
                    source_frames = frames = None
        finally:
            if loaded_model_context is None or hybrid_h3:
                RUNTIME.unload_all()
            else:
                loaded_model_context.offloadobj.unload_all()
        return frames_to_sample(output, source_format, sample.dtype), None

    def release_private_runtime(self) -> None:
        from .runtime import release_model

        release_model()

    def release_vram(self) -> None:
        self.release_private_runtime()


__all__ = ["H3FaceRefinerBridge"]
