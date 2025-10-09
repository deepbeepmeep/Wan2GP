import sys
import os
import tempfile
import shutil
import requests
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QLabel, QMessageBox, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, pyqtSignal, Qt, QSize

sys.path.append(str(Path(__file__).parent.parent.parent))
from plugins import VideoEditorPlugin


API_BASE_URL = "http://127.0.0.1:5100"

class WgpClientWidget(QWidget):
    generation_complete = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        self.last_known_output = None

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Optional (default: i2v_2_2)")

        previews_layout = QHBoxLayout()
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
        
        self.status_label = QLabel("Status: Idle")
        self.output_label = QLabel("Latest Output File: None")
        self.output_label.setWordWrap(True)

        layout.addLayout(previews_layout) 
        form_layout.addRow("Model Type:", self.model_input)
        form_layout.addRow(self.autostart_checkbox)
        form_layout.addRow(self.generate_button)
        
        layout.addLayout(form_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.output_label)

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
        self.check_server_status()
        
    def stop_polling(self):
        self.poll_timer.stop()

    def handle_api_error(self, response, action="performing action"):
        try:
            error_msg = response.json().get("error", "Unknown error")
        except requests.exceptions.JSONDecodeError:
            error_msg = response.text
        self.status_label.setText(f"Status: Error {action}: {error_msg}")
        QMessageBox.warning(self, "API Error", f"Failed while {action}.\n\nServer response:\n{error_msg}")

    def check_server_status(self):
        try:
            requests.get(f"{API_BASE_URL}/api/latest_output", timeout=1)
            self.status_label.setText("Status: Connected to WanGP server.")
        except requests.exceptions.ConnectionError:
            self.status_label.setText("Status: Error - Cannot connect to WanGP server.")
            QMessageBox.critical(self, "Connection Error", f"Could not connect to the WanGP API at {API_BASE_URL}.\n\nPlease ensure wgptool.py is running.")


    def generate(self):
        payload = {}
        
        model_type = self.model_input.text().strip()
        if model_type:
            payload['model_type'] = model_type

        if self.start_frame_input.text():
            payload['start_frame'] = self.start_frame_input.text()
        if self.end_frame_input.text():
            payload['end_frame'] = self.end_frame_input.text()
        
        if self.duration_input.text():
            payload['duration_sec'] = self.duration_input.text()
        
        payload['start_generation'] = self.autostart_checkbox.isChecked()
        
        self.status_label.setText("Status: Sending parameters...")
        try:
            response = requests.post(f"{API_BASE_URL}/api/generate", json=payload)
            if response.status_code == 200:
                if payload['start_generation']:
                    self.status_label.setText("Status: Parameters set. Generation sent. Polling...")
                    self.start_polling()
                else:
                    self.status_label.setText("Status: Parameters set. Waiting for manual start.")
            else:
                self.handle_api_error(response, "setting parameters")
        except requests.exceptions.RequestException as e:
            self.status_label.setText(f"Status: Connection error: {e}")

    def poll_for_output(self):
        try:
            response = requests.get(f"{API_BASE_URL}/api/latest_output")
            if response.status_code == 200:
                data = response.json()
                latest_path = data.get("latest_output_path")
                
                if latest_path and latest_path != self.last_known_output:
                    self.stop_polling()
                    self.last_known_output = latest_path
                    self.output_label.setText(f"Latest Output File:\n{latest_path}")
                    self.status_label.setText("Status: New output received! Inserting clip...")
                    self.generation_complete.emit(latest_path)
            else:
                 if "Error" not in self.status_label.text() and "waiting" not in self.status_label.text().lower():
                    self.status_label.setText("Status: Polling for output...")
        except requests.exceptions.RequestException:
            if "Error" not in self.status_label.text():
                 self.status_label.setText("Status: Polling... (Connection issue)")

class Plugin(VideoEditorPlugin):
    def initialize(self):
        self.name = "AI Frame Joiner"
        self.description = "Uses a local AI server to generate a video between two frames."
        self.client_widget = WgpClientWidget()
        self.dock_widget = None
        self.active_region = None
        self.temp_dir = None
        self.client_widget.generation_complete.connect(self.insert_generated_clip)

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

    def _cleanup_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            
    def _reset_state(self):
        self.active_region = None
        self._cleanup_temp_dir()
        self.client_widget.status_label.setText("Status: Idle")
        self.client_widget.output_label.setText("Latest Output File: None")
        self.client_widget.set_previews(None, None)

    def on_timeline_context_menu(self, menu, event):
        region = self.app.timeline_widget.get_region_at_pos(event.pos())
        if region:
            menu.addSeparator()
            action = menu.addAction("Join Frames With AI")
            action.triggered.connect(lambda: self.setup_generator_for_region(region))

    def setup_generator_for_region(self, region):
        self._reset_state()
        self.active_region = region
        start_sec, end_sec = region
        
        self.client_widget.check_server_status()
        
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
        self.client_widget.status_label.setText(f"Status: Ready for region {start_sec:.2f}s - {end_sec:.2f}s")
        
        self.dock_widget.show()
        self.dock_widget.raise_()

    def insert_generated_clip(self, video_path):
        if not self.active_region:
            self.client_widget.status_label.setText("Status: Error - No active region to insert into.")
            return

        if not os.path.exists(video_path):
            self.client_widget.status_label.setText(f"Status: Error - Output file not found: {video_path}")
            return
            
        start_sec, end_sec = self.active_region
        duration = end_sec - start_sec
        self.app.status_label.setText(f"Inserting AI clip: {os.path.basename(video_path)}")

        try:
            for clip in list(self.app.timeline.clips): self.app._split_at_time(clip, start_sec)
            for clip in list(self.app.timeline.clips): self.app._split_at_time(clip, end_sec)

            clips_to_remove = [
                c for c in self.app.timeline.clips 
                if c.timeline_start_sec >= start_sec and c.timeline_end_sec <= end_sec
            ]
            for clip in clips_to_remove:
                if clip in self.app.timeline.clips:
                    self.app.timeline.clips.remove(clip)
            
            self.app._add_clip_to_timeline(
                source_path=video_path,
                timeline_start_sec=start_sec,
                clip_start_sec=0,
                duration_sec=duration,
                video_track_index=1,
                audio_track_index=None
            )
            
            self.app.status_label.setText("AI clip inserted successfully.")
            self.client_widget.status_label.setText("Status: Success! Clip inserted.")
            self.app.prune_empty_tracks()

        except Exception as e:
            error_message = f"Error during clip insertion: {e}"
            self.app.status_label.setText(error_message)
            self.client_widget.status_label.setText("Status: Failed to insert clip.")
            print(error_message)
        
        finally:
            self._cleanup_temp_dir()
            self.active_region = None