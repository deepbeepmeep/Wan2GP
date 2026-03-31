"""
Travel Segment Processor - Orchestrator for Travel Segment Generation

Coordinates guide video creation, mask video creation, and video_prompt_type
construction for travel segments. The heavy lifting is delegated to:
- guide_builder: guide video creation and supporting helpers
- mask_builder: mask video creation and frame control

Key Components:
- TravelSegmentContext: dataclass holding all parameters for segment processing
- TravelSegmentProcessor: orchestrator that delegates to builders and constructs results
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

from source.core.log import travel_logger
from source.core.params.generation_policy import GenerationPolicy
from source.core.params.travel_guidance import TravelGuidanceConfig
from source.media.video.video_info import get_video_frame_count_and_fps
from source.utils.download_utils import download_file, download_image_if_url
from source.utils.subprocess_utils import run_subprocess

from source.task_handlers.travel.guide_builder import (
    create_guide_video as _build_guide_video,
)
from source.task_handlers.travel.mask_builder import (
    create_mask_video as _build_mask_video,
)


@dataclass
class TravelSegmentContext:
    """Contains all parameters needed for travel segment processing."""
    task_id: str
    segment_idx: int
    model_name: str
    total_frames_for_segment: int
    parsed_res_wh: Tuple[int, int]
    segment_processing_dir: Path
    main_output_dir_base: Path  # Base output directory from worker
    orchestrator_details: Dict[str, Any]
    segment_params: Dict[str, Any]
    mask_active_frames: bool
    debug_enabled: bool


class TravelSegmentProcessor:
    """
    Orchestrator for travel segment generation logic.

    Eliminates code duplication between blocking and queue-based handlers
    by providing a single implementation of guide video creation, mask video
    creation, and video_prompt_type construction.

    Guide and mask creation are delegated to guide_builder and mask_builder
    respectively; this class handles VACE detection, video_prompt_type
    construction, and the top-level process_segment workflow.
    """

    def __init__(self, ctx: TravelSegmentContext):
        self.ctx = ctx
        self.is_vace_model = self._detect_vace_model()
        self.generation_policy = GenerationPolicy.from_payload({
            **ctx.orchestrator_details,
            **ctx.segment_params,
        })
        self._detected_structure_type = None
        self._structure_config = None
        self._travel_guidance_config = None

    def _detect_vace_model(self) -> bool:
        """Detect if this is a VACE model that requires guide videos."""
        model_name = self.ctx.model_name.lower()

        # Only models with "vace" in the name are VACE models.
        # Other indicators like "lightning", "cocktail" appear in non-VACE models too
        # (e.g. wan_2_2_i2v_lightning_baseline_2_2_2 is I2V, not VACE).
        is_vace = "vace" in model_name

        travel_logger.debug(f"[VACE_DEBUG] Seg {self.ctx.segment_idx}: Model '{self.ctx.model_name}' -> is_vace_model = {is_vace}", task_id=self.ctx.task_id)
        return is_vace

    def _is_ltx2_model(self) -> bool:
        """Detect if this is an LTX-2 model."""
        return "ltx2" in self.ctx.model_name.lower()

    def create_guide_video(self) -> Optional[Path]:
        """Create guide video for VACE models or debug mode.

        Delegates to guide_builder.create_guide_video.
        """
        return _build_guide_video(self)

    def create_mask_video(self) -> Optional[Path]:
        """Create mask video for frame control.

        Delegates to mask_builder.create_mask_video.
        """
        return _build_mask_video(self)

    def _get_travel_guidance_config(self) -> Optional[TravelGuidanceConfig]:
        ctx = self.ctx
        travel_guidance_config = self._travel_guidance_config
        if travel_guidance_config is None and isinstance(ctx.segment_params.get("travel_guidance"), dict):
            travel_guidance_config = TravelGuidanceConfig.from_payload(
                ctx.segment_params["travel_guidance"],
                ctx.model_name,
            )
            self._travel_guidance_config = travel_guidance_config
        return travel_guidance_config

    def _build_hybrid_anchor_payload(self) -> Dict[str, Any]:
        ctx = self.ctx
        travel_guidance_config = self._get_travel_guidance_config()
        if travel_guidance_config is None or not travel_guidance_config.is_ltx_hybrid or not travel_guidance_config.has_anchors:
            return {
                "image_refs_paths": None,
                "frames_positions": None,
                "image_refs_strengths": None,
            }

        image_refs_paths: list[str] = []
        frames_positions: list[str] = []
        image_refs_strengths: list[float] = []
        for anchor_index, anchor in enumerate(travel_guidance_config.anchors):
            resolved_path = download_image_if_url(
                anchor.image_url,
                ctx.segment_processing_dir,
                ctx.task_id,
                debug_mode=ctx.debug_enabled,
                descriptive_name=f"hybrid_anchor_seg{ctx.segment_idx}_{anchor_index}",
            )
            image_refs_paths.append(resolved_path)
            frames_positions.append(str(int(anchor.frame_position) + 1))
            image_refs_strengths.append(float(anchor.strength))

        return {
            "image_refs_paths": image_refs_paths,
            "frames_positions": ",".join(frames_positions),
            "image_refs_strengths": image_refs_strengths,
        }

    def _resolve_external_audio_path(self, audio_url: str) -> str:
        ctx = self.ctx
        parsed = urlparse(audio_url)
        suffix = Path(parsed.path).suffix or ".wav"
        local_name = f"hybrid_audio_src_seg{ctx.segment_idx}{suffix}"
        if audio_url.startswith(("http://", "https://")):
            ok = download_file(audio_url, ctx.segment_processing_dir, local_name)
            if not ok:
                raise ValueError(f"Failed to download hybrid audio guide: {audio_url}")
            return str((ctx.segment_processing_dir / local_name).resolve())
        return str(Path(audio_url).resolve())

    def _slice_external_audio(self, audio_url: str) -> str:
        ctx = self.ctx
        source_audio = self._resolve_external_audio_path(audio_url)
        fps = float(ctx.orchestrator_details.get("fps_helpers", 16) or 16)
        start_seconds = max(0.0, float(self._get_travel_guidance_config().frame_offset) / fps)
        duration_seconds = max(0.0, float(ctx.total_frames_for_segment) / fps)
        output_path = ctx.segment_processing_dir / f"hybrid_audio_seg{ctx.segment_idx}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            source_audio,
            "-t",
            f"{duration_seconds:.6f}",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
        result = run_subprocess(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            raise ValueError(
                "Failed to slice hybrid audio guide: "
                f"{audio_url} (rc={result.returncode})"
            )
        return str(output_path.resolve())

    def _build_hybrid_audio_payload(self) -> Dict[str, Any]:
        travel_guidance_config = self._get_travel_guidance_config()
        if travel_guidance_config is None or not travel_guidance_config.is_ltx_hybrid or not travel_guidance_config.has_audio:
            return {
                "audio_guide": None,
                "audio_prompt_type": "",
                "audio_scale": None,
            }

        assert travel_guidance_config.audio is not None
        if travel_guidance_config.audio.source == "control_track":
            if not travel_guidance_config.has_control:
                raise ValueError(
                    "travel_guidance audio.source='control_track' requires control overlap "
                    "for the current segment"
                )
            return {
                "audio_guide": None,
                "audio_prompt_type": "K",
                "audio_scale": float(travel_guidance_config.audio.strength),
            }

        return {
            "audio_guide": self._slice_external_audio(travel_guidance_config.audio.audio_url or ""),
            "audio_prompt_type": "A",
            "audio_scale": float(travel_guidance_config.audio.strength),
        }

    def create_video_prompt_type(self, mask_video_path: Optional[Path], guide_video_path: Optional[Path] = None) -> str:
        """
        Create video_prompt_type string for VACE compatibility.

        Args:
            mask_video_path: Path to mask video, if created
            guide_video_path: Path to guide video, if created

        Returns:
            video_prompt_type string (e.g., "VM", "VIM", "UM")
        """
        ctx = self.ctx

        travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Starting video_prompt_type construction", task_id=ctx.task_id)
        travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: is_vace_model = {self.is_vace_model}", task_id=ctx.task_id)
        travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: mask_video_path exists = {mask_video_path is not None}", task_id=ctx.task_id)

        travel_guidance_config = self._get_travel_guidance_config()

        if travel_guidance_config is not None:
            if travel_guidance_config.is_ltx_hybrid:
                parts = []
                if travel_guidance_config.has_control:
                    mode_map = {
                        "video": "VG",
                        "pose": "PVG",
                        "depth": "DVG",
                        "canny": "EVG",
                    }
                    parts.append(mode_map.get(travel_guidance_config.mode, ""))
                if travel_guidance_config.has_anchors:
                    parts.append("KFI")
                video_prompt_type_str = "".join(parts)
                travel_logger.debug(
                    f"[VPT_DEBUG] Seg {ctx.segment_idx}: travel_guidance ltx_hybrid -> video_prompt_type='{video_prompt_type_str}'",
                    task_id=ctx.task_id,
                )
            elif travel_guidance_config.is_ltx_control:
                video_prompt_type_str = "VG"
                travel_logger.debug(
                    f"[VPT_DEBUG] Seg {ctx.segment_idx}: travel_guidance ltx_control -> video_prompt_type='VG'",
                    task_id=ctx.task_id,
                )
            elif travel_guidance_config.kind == "uni3c":
                # Uni3C needs VG for LTX-2, but VACE must still suppress the
                # raw video guide to avoid double-feeding motion guidance.
                video_prompt_type_str = "VG" if self._is_ltx2_model() else ""
                travel_logger.debug(
                    f"[VPT_DEBUG] Seg {ctx.segment_idx}: travel_guidance uni3c -> video_prompt_type='{video_prompt_type_str}'",
                    task_id=ctx.task_id,
                )
            elif travel_guidance_config.kind == "none":
                video_prompt_type_str = ""
            else:
                video_prompt_type_str = ""

            if video_prompt_type_str:
                travel_logger.debug(
                    f"[VPT_DEBUG] Seg {ctx.segment_idx}: travel_guidance override result = '{video_prompt_type_str}'",
                    task_id=ctx.task_id,
                )
                return video_prompt_type_str

        if self.is_vace_model:
            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: ENTERING VACE MODEL PATH", task_id=ctx.task_id)
            vpt_components = []

            # Check if uni3c is handling motion guidance - if so, skip VACE video guide
            # Use config if available, fallback to legacy detection
            is_uni3c_mode = self._structure_config.is_uni3c
            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: structure_type={self._detected_structure_type}, is_uni3c_mode={is_uni3c_mode}", task_id=ctx.task_id)

            if is_uni3c_mode:
                # Uni3C provides motion guidance - don't double-feed with VACE video guide
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: SKIPPING 'V' - uni3c handles motion guidance", task_id=ctx.task_id)
                travel_logger.debug(f"[UNI3C_VPT] Seg {ctx.segment_idx}: Uni3C mode - video_guide will NOT be used by VACE", task_id=ctx.task_id)
            else:
                # Non-uni3c: use VACE video guide as normal
                vpt_components.append("V")
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Added video 'V', vpt_components = {vpt_components}", task_id=ctx.task_id)
                travel_logger.debug(f"[VACEActivated] Seg {ctx.segment_idx}: Using VACE with raw video guide (no preprocessing)", task_id=ctx.task_id)

            # Add mask component if mask video exists
            if mask_video_path:
                vpt_components.append("M")
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Added mask 'M', vpt_components = {vpt_components}", task_id=ctx.task_id)
                travel_logger.debug(f"[VACEActivated] Seg {ctx.segment_idx}: Adding mask control - mask video: {mask_video_path}", task_id=ctx.task_id)
            else:
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: No mask video, vpt_components = {vpt_components}", task_id=ctx.task_id)

            video_prompt_type_str = "".join(vpt_components)
            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: VACE PATH RESULT: video_prompt_type = '{video_prompt_type_str}'", task_id=ctx.task_id)
            travel_logger.debug(f"[VACEActivated] Seg {ctx.segment_idx}: Final video_prompt_type: '{video_prompt_type_str}'", task_id=ctx.task_id)
        elif self._is_ltx2_model():
            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: ENTERING LTX-2 MODEL PATH", task_id=ctx.task_id)
            travel_logger.debug(
                f"[LTX_GUIDANCE] Seg {ctx.segment_idx}: "
                f"guide_video={'YES' if guide_video_path else 'NO'}, "
                f"mask_video={'YES' if mask_video_path else 'NO'}, "
                f"guidance_kind={getattr(travel_guidance_config, 'kind', None)}, "
                f"guidance_mode={getattr(travel_guidance_config, 'mode', None)}, "
                f"needs_ic_lora={getattr(travel_guidance_config, 'needs_ic_lora', lambda: False)() if travel_guidance_config else False}, "
                f"is_uni3c={getattr(travel_guidance_config, 'is_uni3c', False) if travel_guidance_config else False}",
                task_id=ctx.task_id,
            )
            # LTX-2 supports TSEV: T=text, S=start image, E=end image, V=video guide
            # Use guide_preprocessing from orchestrator if a guide video is provided
            if mask_video_path:
                # LTX-2 supports mask preprocessing (A, NA, XA, XNA)
                video_prompt_type_str = "VG" + "M" if mask_video_path else "VG"
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: LTX-2 with guide+mask → '{video_prompt_type_str}'", task_id=ctx.task_id)
            elif guide_video_path:
                # Guide video exists but no mask — use VG for video-guided generation
                video_prompt_type_str = "VG"
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: LTX-2 with guide (no mask) → 'VG'", task_id=ctx.task_id)
            else:
                # No guide video — use start image mode for keyframe-based generation
                video_prompt_type_str = "S"
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: LTX-2 start-image mode → 'S'", task_id=ctx.task_id)

            # Allow orchestrator to override via explicit guide_preprocessing
            guide_preprocessing = ctx.orchestrator_details.get("guide_preprocessing")
            if guide_preprocessing and travel_guidance_config is None:
                video_prompt_type_str = guide_preprocessing
                travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Using explicit guide_preprocessing: {guide_preprocessing}", task_id=ctx.task_id)

            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: LTX-2 PATH RESULT: video_prompt_type = '{video_prompt_type_str}'", task_id=ctx.task_id)
        else:
            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: ENTERING NON-VACE MODEL PATH", task_id=ctx.task_id)
            # Fallback for non-VACE models: use 'U' for unprocessed RGB to provide direct pixel-level control.
            u_component = "U"
            m_component = "M" if mask_video_path else ""

            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: Non-VACE components: U='{u_component}', M='{m_component}'", task_id=ctx.task_id)

            video_prompt_type_str = u_component + m_component
            travel_logger.debug(f"[VPT_DEBUG] Seg {ctx.segment_idx}: NON-VACE PATH RESULT: video_prompt_type = '{video_prompt_type_str}'", task_id=ctx.task_id)
            travel_logger.debug(f"[VACESkipped] Seg {ctx.segment_idx}: Using non-VACE model -> video_prompt_type: '{video_prompt_type_str}'", task_id=ctx.task_id)

        # Final debug logging
        travel_logger.debug(f"[VPT_FINAL] Seg {ctx.segment_idx}: ===== FINAL VIDEO_PROMPT_TYPE SUMMARY =====", task_id=ctx.task_id)
        travel_logger.debug(f"[VPT_FINAL] Seg {ctx.segment_idx}: Model: '{ctx.model_name}'", task_id=ctx.task_id)
        travel_logger.debug(f"[VPT_FINAL] Seg {ctx.segment_idx}: Is VACE: {self.is_vace_model}", task_id=ctx.task_id)
        travel_logger.debug(f"[VPT_FINAL] Seg {ctx.segment_idx}: Final video_prompt_type: '{video_prompt_type_str}'", task_id=ctx.task_id)
        travel_logger.debug(f"[VPT_FINAL] Seg {ctx.segment_idx}: =========================================", task_id=ctx.task_id)

        return video_prompt_type_str

    def process_segment(self) -> Dict[str, Any]:
        """Main processing method that orchestrates all segment operations."""
        ctx = self.ctx

        # Create guide video
        guide_video_path = self.create_guide_video()

        # Create mask video
        mask_video_path = self.create_mask_video()

        # Create video_prompt_type
        video_prompt_type = self.create_video_prompt_type(mask_video_path, guide_video_path)
        hybrid_anchor_payload = self._build_hybrid_anchor_payload()
        hybrid_audio_payload = self._build_hybrid_audio_payload()
        travel_guidance_config = self._get_travel_guidance_config()

        # === FRAME COUNT DIAGNOSTIC SUMMARY ===
        # This consolidated log helps quickly diagnose frame count mismatches
        travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: ========== FRAME COUNT SUMMARY ==========", task_id=ctx.task_id)
        travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Target frames for segment: {ctx.total_frames_for_segment}", task_id=ctx.task_id)
        travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Valid 4N+1 check: {(ctx.total_frames_for_segment - 1) % 4 == 0}", task_id=ctx.task_id)

        # Verify guide video frame count
        guide_frames = None
        if guide_video_path:
            try:
                guide_frames, _ = get_video_frame_count_and_fps(str(guide_video_path))
                match_status = "\u2713" if guide_frames == ctx.total_frames_for_segment else "\u2717 MISMATCH"
                travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Guide video frames: {guide_frames} {match_status}", task_id=ctx.task_id)
            except (OSError, ValueError, RuntimeError) as e:
                travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Guide video frames: ERROR ({e})", task_id=ctx.task_id)
        else:
            travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Guide video: None", task_id=ctx.task_id)

        # Verify mask video frame count
        mask_frames = None
        if mask_video_path:
            try:
                mask_frames, _ = get_video_frame_count_and_fps(str(mask_video_path))
                match_status = "\u2713" if mask_frames == ctx.total_frames_for_segment else "\u2717 MISMATCH"
                travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Mask video frames: {mask_frames} {match_status}", task_id=ctx.task_id)
            except (OSError, ValueError, RuntimeError) as e:
                travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Mask video frames: ERROR ({e})", task_id=ctx.task_id)
        else:
            travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: Mask video: None", task_id=ctx.task_id)

        # Warn if any mismatch detected
        if guide_frames and guide_frames != ctx.total_frames_for_segment:
            travel_logger.warning(f"[FRAME_COUNTS] Guide frames ({guide_frames}) != target ({ctx.total_frames_for_segment})", task_id=ctx.task_id)
        if mask_frames and mask_frames != ctx.total_frames_for_segment:
            travel_logger.warning(f"[FRAME_COUNTS] Mask frames ({mask_frames}) != target ({ctx.total_frames_for_segment})", task_id=ctx.task_id)

        travel_logger.debug(f"[FRAME_COUNTS] Seg {ctx.segment_idx}: ==========================================", task_id=ctx.task_id)

        return {
            "video_guide": str(guide_video_path) if guide_video_path else None,
            "video_mask": str(mask_video_path) if mask_video_path else None,
            "video_prompt_type": video_prompt_type,
            "structure_type": self._detected_structure_type,  # Pass through for uni3c handling
            "image_refs_paths": hybrid_anchor_payload["image_refs_paths"],
            "frames_positions": hybrid_anchor_payload["frames_positions"],
            "image_refs_strengths": hybrid_anchor_payload["image_refs_strengths"],
            "audio_guide": hybrid_audio_payload["audio_guide"],
            "audio_prompt_type": hybrid_audio_payload["audio_prompt_type"],
            "audio_scale": hybrid_audio_payload["audio_scale"],
            "denoising_strength": (
                travel_guidance_config.control_strength
                if travel_guidance_config is not None
                and travel_guidance_config.is_ltx_hybrid
                and travel_guidance_config.has_control
                else None
            ),
        }

    def _detect_single_image_journey(self) -> bool:
        """
        Detect if this is a single image journey (I2V mode, no end anchor).

        A single image journey is when:
        - Only one input image is provided for this segment
        - Not continuing from a video
        - AND EITHER:
          - This segment is both first and last (entire project is single image)
          - OR this is a trailing segment (is_last_segment=True, has no end target)

        Returns:
            True if this is a single image journey (use I2V mode), False otherwise
        """
        ctx = self.ctx

        input_images_for_check = ctx.orchestrator_details.get("input_image_paths_resolved", [])
        is_first_segment = ctx.segment_params.get("is_first_segment", False)
        is_last_segment = ctx.segment_params.get("is_last_segment", False)
        has_continue_video = ctx.orchestrator_details.get("continue_from_video_resolved_path") is not None

        # Single image journey detection:
        # 1. Must have exactly 1 input image
        # 2. Must not be continuing from a video
        # 3. Either: (a) entire project is one segment, OR (b) this is a trailing segment
        is_single_image_journey = (
            len(input_images_for_check) == 1
            and not has_continue_video
            and (
                (is_first_segment and is_last_segment)  # Single image project
                or is_last_segment  # Trailing segment (no end target image)
            )
        )

        travel_logger.debug(f"[SINGLE_IMAGE_DEBUG] Seg {ctx.segment_idx}: Single image journey detection:", task_id=ctx.task_id)
        travel_logger.debug(f"[SINGLE_IMAGE_DEBUG]   Input images count: {len(input_images_for_check)}", task_id=ctx.task_id)
        travel_logger.debug(f"[SINGLE_IMAGE_DEBUG]   Continue from video: {has_continue_video}", task_id=ctx.task_id)
        travel_logger.debug(f"[SINGLE_IMAGE_DEBUG]   Is first segment: {is_first_segment}", task_id=ctx.task_id)
        travel_logger.debug(f"[SINGLE_IMAGE_DEBUG]   Is last segment: {is_last_segment}", task_id=ctx.task_id)
        travel_logger.debug(f"[SINGLE_IMAGE_DEBUG]   Result: {is_single_image_journey}", task_id=ctx.task_id)

        return is_single_image_journey
