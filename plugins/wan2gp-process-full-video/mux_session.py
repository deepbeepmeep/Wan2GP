from __future__ import annotations

import os
from dataclasses import dataclass

from . import media_io as media
from . import video_buffers as video


@dataclass
class MuxSession:
    mux_process: object | None = None
    output_path_for_write: str = ""
    video_only_output_path: str = ""
    reserved_metadata_path: str | None = None
    stopped: bool = False
    mux_finished: bool = False
    owns_output_path_for_write: bool = False
    owns_video_only_output: bool = False

    def set_output_path(self, output_path: str) -> None:
        self.output_path_for_write = output_path
        self.video_only_output_path = ""

    def ensure_started(
        self,
        *,
        server_config: dict,
        ffmpeg_path: str,
        process_is_hdr: bool,
        use_live_av_mux: bool,
        output_container: str,
        source_path: str,
        exact_start_seconds: float,
        selected_audio_track: int | None,
        resolved_width: int,
        resolved_height: int,
        fps_float: float,
    ) -> None:
        if self.mux_process is not None:
            return
        self.reserved_metadata_path = media.create_reserved_metadata_file(self.output_path_for_write)
        self.owns_output_path_for_write = use_live_av_mux
        self.owns_video_only_output = not use_live_av_mux
        if self.owns_video_only_output:
            self.video_only_output_path = media.reserve_video_only_output_path(self.output_path_for_write)
        if process_is_hdr:
            hdr_video_crf = server_config.get("hdr_video_crf", 8)
            self.mux_process = media.start_hdr_av_mux_process(ffmpeg_path, self.output_path_for_write, resolved_width, resolved_height, fps_float, hdr_video_crf, output_container, source_path, exact_start_seconds, selected_audio_track, self.reserved_metadata_path) if use_live_av_mux else media.start_hdr_video_mux_process(ffmpeg_path, self.video_only_output_path, resolved_width, resolved_height, fps_float, hdr_video_crf, output_container, self.reserved_metadata_path)
            return
        video_codec = server_config.get("video_output_codec", "libx264_8")
        self.mux_process = media.start_av_mux_process(ffmpeg_path, self.output_path_for_write, resolved_width, resolved_height, fps_float, video_codec, output_container, source_path, exact_start_seconds, selected_audio_track, self.reserved_metadata_path) if use_live_av_mux else media.start_video_mux_process(ffmpeg_path, self.video_only_output_path, resolved_width, resolved_height, fps_float, video_codec, output_container, self.reserved_metadata_path)

    def write_chunk(self, *, process_is_hdr: bool, video_tensor_hdr, video_tensor_uint8, start_frame: int, frame_count: int):
        if process_is_hdr and video_tensor_hdr is not None:
            return video.write_hdr_video_chunk(self.mux_process, video_tensor_hdr, start_frame=start_frame, frame_count=frame_count)
        return video.write_video_chunk(self.mux_process, video_tensor_uint8, start_frame=start_frame, frame_count=frame_count)

    def finalize(self) -> tuple[int, str, bool]:
        return_code, stderr, forced_termination = media.finalize_mux_process(self.mux_process)
        self.mux_finished = True
        return return_code, stderr, forced_termination

    def cleanup_partial_outputs(self) -> None:
        if self.mux_process is not None and not self.mux_finished and self.mux_process.poll() is None:
            try:
                media.finalize_mux_process(self.mux_process)
            except Exception:
                pass
        if self.owns_output_path_for_write and self.mux_process is not None and not self.stopped and self.mux_process.returncode not in (0, None) and os.path.isfile(self.output_path_for_write):
            media.delete_file_if_exists(self.output_path_for_write, label="continuation output")
        if self.owns_video_only_output:
            media.delete_file_if_exists(self.video_only_output_path, label="video-only output")
        media.delete_file_if_exists(self.reserved_metadata_path, label="reserved metadata file")
