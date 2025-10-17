import ffmpeg
import subprocess
import re
from PyQt6.QtCore import QObject, pyqtSignal, QThread

class _ExportRunner(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, ffmpeg_cmd, total_duration_ms, parent=None):
        super().__init__(parent)
        self.ffmpeg_cmd = ffmpeg_cmd
        self.total_duration_ms = total_duration_ms
        self.process = None

    def run(self):
        try:
            startupinfo = None
            if hasattr(subprocess, 'STARTUPINFO'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            self.process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding="utf-8",
                errors='ignore',
                startupinfo=startupinfo
            )

            full_output = []
            time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")

            for line in iter(self.process.stdout.readline, ""):
                full_output.append(line)
                match = time_pattern.search(line)
                if match:
                    h, m, s, cs = [int(g) for g in match.groups()]
                    processed_ms = (h * 3600 + m * 60 + s) * 1000 + cs * 10
                    if self.total_duration_ms > 0:
                        percentage = int((processed_ms / self.total_duration_ms) * 100)
                        self.progress.emit(min(100, percentage))

            self.process.stdout.close()
            return_code = self.process.wait()

            if return_code == 0:
                self.progress.emit(100)
                self.finished.emit(True, "Export completed successfully!")
            else:
                print("--- FFmpeg Export FAILED ---")
                print("Command: " + " ".join(self.ffmpeg_cmd))
                print("".join(full_output))
                self.finished.emit(False, f"Export failed with code {return_code}. Check console.")

        except FileNotFoundError:
            self.finished.emit(False, "Export failed: ffmpeg.exe not found in your system's PATH.")
        except Exception as e:
            self.finished.emit(False, f"An exception occurred during export: {e}")

    def get_process(self):
        return self.process

class Encoder(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self.worker = None
        self._is_running = False

    def start_export(self, timeline, project_settings, export_settings):
        if self._is_running:
            self.finished.emit(False, "An export is already in progress.")
            return

        self._is_running = True

        try:
            total_dur_ms = timeline.get_total_duration()
            total_dur_sec = total_dur_ms / 1000.0
            w, h, fps = project_settings['width'], project_settings['height'], project_settings['fps']
            sample_rate, channel_layout = '44100', 'stereo'

            # --- VIDEO GRAPH CONSTRUCTION (Definitive Solution) ---

            all_video_clips = sorted(
                [c for c in timeline.clips if c.track_type == 'video'],
                key=lambda c: c.track_index
            )

            # 1. Start with a base black canvas that defines the project's duration.
            # This is our master clock and bottom layer.
            final_video = ffmpeg.input(f'color=c=black:s={w}x{h}:r={fps}:d={total_dur_sec}', f='lavfi')

            # 2. Process each clip individually and overlay it.
            for clip in all_video_clips:
                # A new, separate input for every single clip guarantees no conflicts.
                if clip.media_type == 'image':
                    clip_input = ffmpeg.input(clip.source_path, loop=1, framerate=fps)
                else:
                    clip_input = ffmpeg.input(clip.source_path)

                # a) Calculate the time shift to align the clip's content with the timeline.
                #    This ensures the correct frame of the source is shown at the start of the clip.
                timeline_start_sec = clip.timeline_start_ms / 1000.0
                clip_start_sec = clip.clip_start_ms / 1000.0
                time_shift_sec = timeline_start_sec - clip_start_sec

                # b) Prepare the layer: apply the time shift, then scale and pad.
                timed_layer = (
                    clip_input.video
                    .setpts(f'PTS+{time_shift_sec}/TB')
                    .filter('scale', w, h, force_original_aspect_ratio='decrease')
                    .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
                )
                
                # c) Define the visibility window for the overlay on the master timeline.
                timeline_end_sec = (clip.timeline_start_ms + clip.duration_ms) / 1000.0
                enable_expression = f'between(t,{timeline_start_sec:.6f},{timeline_end_sec:.6f})'

                # d) Overlay the prepared layer onto the composition, enabling it only during its time window.
                #    eof_action='pass' handles finite streams gracefully.
                final_video = ffmpeg.overlay(final_video, timed_layer, enable=enable_expression, eof_action='pass')
                
            # 3. Set final output format and framerate.
            final_video = final_video.filter('format', pix_fmts='yuv420p').filter('fps', fps=fps)


            # --- AUDIO GRAPH CONSTRUCTION (UNCHANGED and CORRECT) ---
            track_audio_streams = []
            for i in range(1, timeline.num_audio_tracks + 1):
                track_clips = sorted([c for c in timeline.clips if c.track_type == 'audio' and c.track_index == i], key=lambda c: c.timeline_start_ms)
                if not track_clips:
                    continue

                track_segments = []
                last_end_ms = 0
                for clip in track_clips:
                    gap_ms = clip.timeline_start_ms - last_end_ms
                    if gap_ms > 10:
                        track_segments.append(ffmpeg.input(f'anullsrc=r={sample_rate}:cl={channel_layout}:d={gap_ms/1000.0}', f='lavfi'))

                    clip_start_sec = clip.clip_start_ms / 1000.0
                    clip_duration_sec = clip.duration_ms / 1000.0
                    audio_source_node = ffmpeg.input(clip.source_path)
                    a_seg = audio_source_node.audio.filter('atrim', start=clip_start_sec, duration=clip_duration_sec).filter('asetpts', 'PTS-STARTPTS')
                    track_segments.append(a_seg)
                    last_end_ms = clip.timeline_start_ms + clip.duration_ms

                if track_segments:
                    track_audio_streams.append(ffmpeg.concat(*track_segments, v=0, a=1))

            # --- FINAL OUTPUT ASSEMBLY ---
            output_args = {}
            stream_args = []
            has_audio = bool(track_audio_streams) and export_settings.get('acodec')

            if export_settings.get('vcodec'):
                stream_args.append(final_video)
                output_args['vcodec'] = export_settings['vcodec']
                if export_settings.get('v_bitrate'): output_args['b:v'] = export_settings['v_bitrate']

            if has_audio:
                final_audio = ffmpeg.filter(track_audio_streams, 'amix', inputs=len(track_audio_streams), duration='longest')
                stream_args.append(final_audio)
                output_args['acodec'] = export_settings['acodec']
                if export_settings.get('a_bitrate'): output_args['b:a'] = export_settings['a_bitrate']
            
            if not has_audio:
                output_args['an'] = None

            if not stream_args:
                raise ValueError("No streams to output. Check export settings.")
                
            ffmpeg_cmd = ffmpeg.output(*stream_args, export_settings['output_path'], **output_args).overwrite_output().compile()

        except Exception as e:
            self.finished.emit(False, f"Error building FFmpeg command: {e}")
            self._is_running = False
            return

        self.worker_thread = QThread()
        self.worker = _ExportRunner(ffmpeg_cmd, total_dur_ms)
        self.worker.moveToThread(self.worker_thread)

        self.worker.progress.connect(self.progress.emit)
        self.worker.finished.connect(self._on_export_runner_finished)

        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def _on_export_runner_finished(self, success, message):
        self._is_running = False
        self.finished.emit(success, message)
        
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker_thread = None
        self.worker = None

    def cancel_export(self):
        if self.worker and self.worker.get_process() and self.worker.get_process().poll() is None:
            self.worker.get_process().terminate()
            print("Export cancelled by user.")