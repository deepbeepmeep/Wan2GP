import sys
import os
import tempfile
import shutil
import requests
from pathlib import Path
import ffmpeg
import uuid

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QLabel, QMessageBox, QCheckBox, QListWidget,
    QListWidgetItem, QGroupBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, Qt, QSize, QRectF, QUrl, QTimer
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget

sys.path.append(str(Path(__file__).parent.parent.parent))
from plugins import VideoEditorPlugin

API_BASE_URL = "http://127.0.0.1:5100"

class VideoResultItemWidget(QWidget):
    def __init__(self, video_path, plugin, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.plugin = plugin
        self.app = plugin.app
        self.duration = 0.0
        self.has_audio = False

        self.setMinimumSize(200, 180)
        self.setMaximumHeight(190)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.video_widget.setFixedSize(160, 90)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
        self.media_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.info_label = QLabel(os.path.basename(video_path))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)

        self.insert_button = QPushButton("Insert into Timeline")
        self.insert_button.clicked.connect(self.on_insert)

        h_layout = QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(self.video_widget)
        h_layout.addStretch()
        
        layout.addLayout(h_layout)
        layout.addWidget(self.info_label)
        layout.addWidget(self.insert_button)

        self.probe_video()

    def probe_video(self):
        try:
            probe = ffmpeg.probe(self.video_path)
            self.duration = float(probe['format']['duration'])
            self.has_audio = any(s['codec_type'] == 'audio' for s in probe.get('streams', []))

            self.info_label.setText(f"{os.path.basename(self.video_path)}\n({self.duration:.2f}s)")

        except Exception as e:
            self.info_label.setText(f"Error probing:\n{os.path.basename(self.video_path)}")
            print(f"Error probing video {self.video_path}: {e}")
    
    def enterEvent(self, event):
        super().enterEvent(event)
        self.media_player.play()

        if not self.plugin.active_region or self.duration == 0:
            return

        start_sec, _ = self.plugin.active_region
        timeline = self.app.timeline_widget
        
        video_rect, audio_rect = None, None
        
        x = timeline.sec_to_x(start_sec)
        w = int(self.duration * timeline.pixels_per_second)
        
        if self.plugin.insert_on_new_track:
            video_y = timeline.TIMESCALE_HEIGHT
            video_rect = QRectF(x, video_y, w, timeline.TRACK_HEIGHT)
            if self.has_audio:
                audio_y = timeline.audio_tracks_y_start + self.app.timeline.num_audio_tracks * timeline.TRACK_HEIGHT
                audio_rect = QRectF(x, audio_y, w, timeline.TRACK_HEIGHT)
        else:
            v_track_idx = 1
            visual_v_idx = self.app.timeline.num_video_tracks - v_track_idx
            video_y = timeline.video_tracks_y_start + visual_v_idx * timeline.TRACK_HEIGHT
            video_rect = QRectF(x, video_y, w, timeline.TRACK_HEIGHT)
            if self.has_audio:
                a_track_idx = 1
                visual_a_idx = a_track_idx - 1
                audio_y = timeline.audio_tracks_y_start + visual_a_idx * timeline.TRACK_HEIGHT
                audio_rect = QRectF(x, audio_y, w, timeline.TRACK_HEIGHT)

        timeline.set_hover_preview_rects(video_rect, audio_rect)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.media_player.pause()
        self.media_player.setPosition(0)
        self.app.timeline_widget.set_hover_preview_rects(None, None)
    
    def on_insert(self):
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self.media_player.setVideoOutput(None)
        self.app.timeline_widget.set_hover_preview_rects(None, None)
        self.plugin.insert_generated_clip(self.video_path)

class WgpClientWidget(QWidget):
    status_updated = pyqtSignal(str)
    
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin
        self.processed_files = set()

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.previews_widget = QWidget()
        previews_layout = QHBoxLayout(self.previews_widget)
        previews_layout.setContentsMargins(0, 0, 0, 0)
        
        start_preview_layout = QVBoxLayout()
        start_preview_layout.addWidget(QLabel("<b>Start Frame</b>"))
        self.start_frame_preview = QLabel("N/A")
        self.start_frame_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_frame_preview.setFixedSize(160, 90)
        self.start_frame_preview.setStyleSheet("background-color: #222; color: #888;")
        start_preview_layout.addWidget(self.start_frame_preview)
        previews_layout.addLayout(start_preview_layout)

        end_preview_layout = QVBoxLayout()
        end_preview_layout.addWidget(QLabel("<b>End Frame</b>"))
        self.end_frame_preview = QLabel("N/A")
        self.end_frame_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.end_frame_preview.setFixedSize(160, 90)
        self.end_frame_preview.setStyleSheet("background-color: #222; color: #888;")
        end_preview_layout.addWidget(self.end_frame_preview)
        previews_layout.addLayout(end_preview_layout)
        
        self.start_frame_input = QLineEdit()
        self.end_frame_input = QLineEdit()
        self.duration_input = QLineEdit()
        
        self.autostart_checkbox = QCheckBox("Start Generation Immediately")
        self.autostart_checkbox.setChecked(True)

        self.generate_button = QPushButton("Generate")

        layout.addWidget(self.previews_widget)
        form_layout.addRow(self.autostart_checkbox)
        form_layout.addRow(self.generate_button)
        
        layout.addLayout(form_layout)
        
        # --- Results Area ---
        results_group = QGroupBox("Generated Clips (Hover to play, Click button to insert)")
        results_layout = QVBoxLayout()
        self.results_list = QListWidget()
        self.results_list.setFlow(QListWidget.Flow.LeftToRight)
        self.results_list.setWrapping(True)
        self.results_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.results_list.setSpacing(10)
        results_layout.addWidget(self.results_list)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch() 

        self.generate_button.clicked.connect(self.generate)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(3000)
        self.poll_timer.timeout.connect(self.poll_for_output)

    def set_previews(self, start_pixmap, end_pixmap):
        if start_pixmap:
            self.start_frame_preview.setPixmap(start_pixmap)
        else:
            self.start_frame_preview.setText("N/A")
            self.start_frame_preview.setPixmap(QPixmap())
            
        if end_pixmap:
            self.end_frame_preview.setPixmap(end_pixmap)
        else:
            self.end_frame_preview.setText("N/A")
            self.end_frame_preview.setPixmap(QPixmap())

    def start_polling(self):
        self.poll_timer.start()
        
    def stop_polling(self):
        self.poll_timer.stop()

    def handle_api_error(self, response, action="performing action"):
        try:
            error_msg = response.json().get("error", "Unknown error")
        except requests.exceptions.JSONDecodeError:
            error_msg = response.text
        self.status_updated.emit(f"AI Joiner Error: {error_msg}")
        QMessageBox.warning(self, "API Error", f"Failed while {action}.\n\nServer response:\n{error_msg}")

    def check_server_status(self):
        try:
            requests.get(f"{API_BASE_URL}/api/outputs", timeout=1)
            self.status_updated.emit("AI Joiner: Connected to WanGP server.")
            return True
        except requests.exceptions.ConnectionError:
            self.status_updated.emit("AI Joiner Error: Cannot connect to WanGP server.")
            QMessageBox.critical(self, "Connection Error", f"Could not connect to the WanGP API at {API_BASE_URL}.\n\nPlease ensure wgptool.py is running.")
            return False

    def generate(self):
        payload = {}
        
        model_type = self.model_type_to_use
        if model_type:
            payload['model_type'] = model_type

        if self.start_frame_input.text():
            payload['start_frame'] = self.start_frame_input.text()
        if self.end_frame_input.text():
            payload['end_frame'] = self.end_frame_input.text()
        
        if self.duration_input.text():
            payload['duration_sec'] = self.duration_input.text()
        
        payload['start_generation'] = self.autostart_checkbox.isChecked()
        
        self.status_updated.emit("AI Joiner: Sending parameters...")
        try:
            response = requests.post(f"{API_BASE_URL}/api/generate", json=payload)
            if response.status_code == 200:
                if payload['start_generation']:
                    self.status_updated.emit("AI Joiner: Generation sent to server. Polling for output...")
                else:
                    self.status_updated.emit("AI Joiner: Parameters set. Polling for manually started generation...")
                self.start_polling()
            else:
                self.handle_api_error(response, "setting parameters")
        except requests.exceptions.RequestException as e:
            self.status_updated.emit(f"AI Joiner Error: Connection error: {e}")

    def poll_for_output(self):
        try:
            response = requests.get(f"{API_BASE_URL}/api/outputs")
            if response.status_code == 200:
                data = response.json()
                output_files = data.get("outputs", [])
                
                new_files = set(output_files) - self.processed_files
                if new_files:
                    self.status_updated.emit(f"AI Joiner: Received {len(new_files)} new clip(s).")
                    for file_path in sorted(list(new_files)):
                        self.add_result_item(file_path)
                        self.processed_files.add(file_path)
                else:
                    self.status_updated.emit("AI Joiner: Polling for output...")
            else:
                 self.status_updated.emit("AI Joiner: Polling for output...")
        except requests.exceptions.RequestException:
             self.status_updated.emit("AI Joiner: Polling... (Connection issue)")
    
    def add_result_item(self, video_path):
        item_widget = VideoResultItemWidget(video_path, self.plugin)
        list_item = QListWidgetItem(self.results_list)
        list_item.setSizeHint(item_widget.sizeHint())
        self.results_list.addItem(list_item)
        self.results_list.setItemWidget(list_item, item_widget)

    def clear_results(self):
        self.results_list.clear()
        self.processed_files.clear()

class Plugin(VideoEditorPlugin):
    def initialize(self):
        self.name = "AI Frame Joiner"
        self.description = "Uses a local AI server to generate a video between two frames."
        self.client_widget = WgpClientWidget(self)
        self.dock_widget = None
        self.active_region = None
        self.temp_dir = None
        self.insert_on_new_track = False
        self.client_widget.status_updated.connect(self.update_main_status)

    def enable(self):
        if not self.dock_widget:
            self.dock_widget = self.app.add_dock_widget(self, self.client_widget, "AI Frame Joiner", show_on_creation=False)
        
        self.dock_widget.hide()
        
        self.app.timeline_widget.context_menu_requested.connect(self.on_timeline_context_menu)

    def disable(self):
        try:
            self.app.timeline_widget.context_menu_requested.disconnect(self.on_timeline_context_menu)
        except TypeError:
            pass
        self._cleanup_temp_dir()
        self.client_widget.stop_polling()

    def update_main_status(self, message):
        self.app.status_label.setText(message)

    def _cleanup_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            
    def _reset_state(self):
        self.active_region = None
        self.insert_on_new_track = False
        self.client_widget.stop_polling()
        self.client_widget.clear_results()
        self._cleanup_temp_dir()
        self.update_main_status("AI Joiner: Idle")
        self.client_widget.set_previews(None, None)
        self.client_widget.model_type_to_use = None

    def on_timeline_context_menu(self, menu, event):
        region = self.app.timeline_widget.get_region_at_pos(event.pos())
        if region:
            menu.addSeparator()

            start_sec, end_sec = region
            start_data, _, _ = self.app.get_frame_data_at_time(start_sec)
            end_data, _, _ = self.app.get_frame_data_at_time(end_sec)

            if start_data and end_data:
                join_action = menu.addAction("Join Frames With AI")
                join_action.triggered.connect(lambda: self.setup_generator_for_region(region, on_new_track=False))
                
                join_action_new_track = menu.addAction("Join Frames With AI (New Track)")
                join_action_new_track.triggered.connect(lambda: self.setup_generator_for_region(region, on_new_track=True))

            create_action = menu.addAction("Create Frames With AI")
            create_action.triggered.connect(lambda: self.setup_creator_for_region(region, on_new_track=False))

            create_action_new_track = menu.addAction("Create Frames With AI (New Track)")
            create_action_new_track.triggered.connect(lambda: self.setup_creator_for_region(region, on_new_track=True))

    def setup_generator_for_region(self, region, on_new_track=False):
        self._reset_state()
        self.client_widget.model_type_to_use = "i2v_2_2"
        self.active_region = region
        self.insert_on_new_track = on_new_track
        self.client_widget.previews_widget.setVisible(True)
        
        start_sec, end_sec = region
        if not self.client_widget.check_server_status():
            return
            
        start_data, w, h = self.app.get_frame_data_at_time(start_sec)
        end_data, _, _ = self.app.get_frame_data_at_time(end_sec)

        if not start_data or not end_data:
            QMessageBox.warning(self.app, "Frame Error", "Could not extract start and/or end frames for the selected region.")
            return

        preview_size = QSize(160, 90)
        start_pixmap, end_pixmap = None, None
        
        try:
            start_img = QImage(start_data, w, h, QImage.Format.Format_RGB888)
            start_pixmap = QPixmap.fromImage(start_img).scaled(preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            end_img = QImage(end_data, w, h, QImage.Format.Format_RGB888)
            end_pixmap = QPixmap.fromImage(end_img).scaled(preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        except Exception as e:
             QMessageBox.critical(self.app, "Image Error", f"Could not create preview images: {e}")

        self.client_widget.set_previews(start_pixmap, end_pixmap)

        try:
            self.temp_dir = tempfile.mkdtemp(prefix="ai_joiner_")
            start_img_path = os.path.join(self.temp_dir, "start_frame.png")
            end_img_path = os.path.join(self.temp_dir, "end_frame.png")
            
            QImage(start_data, w, h, QImage.Format.Format_RGB888).save(start_img_path)
            QImage(end_data, w, h, QImage.Format.Format_RGB888).save(end_img_path)
        except Exception as e:
            QMessageBox.critical(self.app, "File Error", f"Could not save temporary frame images: {e}")
            self._cleanup_temp_dir()
            return

        duration_sec = end_sec - start_sec
        self.client_widget.duration_input.setText(str(duration_sec))

        self.client_widget.start_frame_input.setText(start_img_path)
        self.client_widget.end_frame_input.setText(end_img_path)
        self.update_main_status(f"AI Joiner: Ready for region {start_sec:.2f}s - {end_sec:.2f}s")
        
        self.dock_widget.show()
        self.dock_widget.raise_()

    def setup_creator_for_region(self, region, on_new_track=False):
        self._reset_state()
        self.client_widget.model_type_to_use = "t2v_2_2"
        self.active_region = region
        self.insert_on_new_track = on_new_track
        self.client_widget.previews_widget.setVisible(False)
        
        start_sec, end_sec = region
        if not self.client_widget.check_server_status():
            return

        self.client_widget.set_previews(None, None)
        
        duration_sec = end_sec - start_sec
        self.client_widget.duration_input.setText(str(duration_sec))

        self.client_widget.start_frame_input.clear()
        self.client_widget.end_frame_input.clear()

        self.update_main_status(f"AI Creator: Ready for region {start_sec:.2f}s - {end_sec:.2f}s")
        
        self.dock_widget.show()
        self.dock_widget.raise_()

    def insert_generated_clip(self, video_path):
        from main import TimelineClip

        if not self.active_region:
            self.update_main_status("AI Joiner Error: No active region to insert into.")
            return

        if not os.path.exists(video_path):
            self.update_main_status(f"AI Joiner Error: Output file not found: {video_path}")
            return
            
        start_sec, end_sec = self.active_region
        is_new_track_mode = self.insert_on_new_track
        
        self.update_main_status(f"AI Joiner: Inserting clip {os.path.basename(video_path)}")

        def complex_insertion_action():
            probe = ffmpeg.probe(video_path)
            actual_duration = float(probe['format']['duration'])

            if is_new_track_mode:
                self.app.timeline.num_video_tracks += 1
                new_track_index = self.app.timeline.num_video_tracks
                
                new_clip = TimelineClip(
                    source_path=video_path,
                    timeline_start_sec=start_sec,
                    clip_start_sec=0,
                    duration_sec=actual_duration,
                    track_index=new_track_index,
                    track_type='video',
                    media_type='video',
                    group_id=str(uuid.uuid4())
                )
                self.app.timeline.add_clip(new_clip)
            else:
                for clip in list(self.app.timeline.clips): self.app._split_at_time(clip, start_sec)
                for clip in list(self.app.timeline.clips): self.app._split_at_time(clip, end_sec)

                clips_to_remove = [
                    c for c in self.app.timeline.clips 
                    if c.timeline_start_sec >= start_sec and c.timeline_end_sec <= end_sec
                ]
                for clip in clips_to_remove:
                    if clip in self.app.timeline.clips:
                        self.app.timeline.clips.remove(clip)
                
                new_clip = TimelineClip(
                    source_path=video_path,
                    timeline_start_sec=start_sec,
                    clip_start_sec=0,
                    duration_sec=actual_duration,
                    track_index=1,
                    track_type='video',
                    media_type='video',
                    group_id=str(uuid.uuid4())
                )
                self.app.timeline.add_clip(new_clip)

        try:
            self.app._perform_complex_timeline_change("Insert AI Clip", complex_insertion_action)
            
            self.app.prune_empty_tracks()
            self.update_main_status("AI clip inserted successfully.")

            for i in range(self.client_widget.results_list.count()):
                item = self.client_widget.results_list.item(i)
                widget = self.client_widget.results_list.itemWidget(item)
                if widget and widget.video_path == video_path:
                    self.client_widget.results_list.takeItem(i)
                    break

        except Exception as e:
            error_message = f"AI Joiner Error during clip insertion/probing: {e}"
            self.update_main_status(error_message)
            print(error_message)