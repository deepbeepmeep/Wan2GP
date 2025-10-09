import sys
import os
import uuid
import subprocess
import re
import json
import ffmpeg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QScrollArea, QFrame, QProgressBar, QDialog,
                             QCheckBox, QDialogButtonBox, QMenu, QSplitter, QDockWidget)
from PyQt6.QtGui import (QPainter, QColor, QPen, QFont, QFontMetrics, QMouseEvent, QAction,
                         QPixmap, QImage)
from PyQt6.QtCore import (Qt, QPoint, QRect, QRectF, QSize, QPointF, QObject, QThread,
                          pyqtSignal, QTimer, QByteArray)

from plugins import PluginManager, ManagePluginsDialog
import requests
import zipfile
from tqdm import tqdm

def download_ffmpeg():
    if os.name != 'nt': return
    exes = ['ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe']
    if all(os.path.exists(e) for e in exes): return
    api_url = 'https://api.github.com/repos/GyanD/codexffmpeg/releases/latest'
    r = requests.get(api_url, headers={'Accept': 'application/vnd.github+json'})
    assets = r.json().get('assets', [])
    zip_asset = next((a for a in assets if 'essentials_build.zip' in a['name']), None)
    if not zip_asset: return
    zip_url = zip_asset['browser_download_url']
    zip_name = zip_asset['name']
    with requests.get(zip_url, stream=True) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        with open(zip_name, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    with zipfile.ZipFile(zip_name) as z:
        for f in z.namelist():
            if f.endswith(tuple(exes)) and '/bin/' in f:
                z.extract(f)
                os.rename(f, os.path.basename(f))
    os.remove(zip_name)

download_ffmpeg()

class TimelineClip:
    def __init__(self, source_path, timeline_start_sec, clip_start_sec, duration_sec, track_index, track_type, group_id):
        self.id = str(uuid.uuid4())
        self.source_path = source_path
        self.timeline_start_sec = timeline_start_sec
        self.clip_start_sec = clip_start_sec
        self.duration_sec = duration_sec
        self.track_index = track_index
        self.track_type = track_type
        self.group_id = group_id

    @property
    def timeline_end_sec(self):
        return self.timeline_start_sec + self.duration_sec

class Timeline:
    def __init__(self):
        self.clips = []
        self.num_video_tracks = 1
        self.num_audio_tracks = 1

    def add_clip(self, clip):
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.timeline_start_sec)

    def get_total_duration(self):
        if not self.clips: return 0
        return max(c.timeline_end_sec for c in self.clips)

class ExportWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    def __init__(self, ffmpeg_cmd, total_duration):
        super().__init__()
        self.ffmpeg_cmd = ffmpeg_cmd
        self.total_duration_secs = total_duration
    def run_export(self):
        try:
            process = subprocess.Popen(self.ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding="utf-8")
            time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
            for line in iter(process.stdout.readline, ""):
                match = time_pattern.search(line)
                if match:
                    hours, minutes, seconds = [int(g) for g in match.groups()[:3]]
                    processed_time = hours * 3600 + minutes * 60 + seconds
                    if self.total_duration_secs > 0:
                        percentage = int((processed_time / self.total_duration_secs) * 100)
                        self.progress.emit(percentage)
            process.stdout.close()
            return_code = process.wait()
            if return_code == 0:
                self.progress.emit(100)
                self.finished.emit("Export completed successfully!")
            else:
                self.finished.emit(f"Export failed! Check console for FFmpeg errors.")
        except Exception as e:
            self.finished.emit(f"An exception occurred during export: {e}")

class TimelineWidget(QWidget):
    PIXELS_PER_SECOND = 50
    TIMESCALE_HEIGHT = 30
    HEADER_WIDTH = 120
    TRACK_HEIGHT = 50
    AUDIO_TRACKS_SEPARATOR_Y = 15

    split_requested = pyqtSignal(object)
    playhead_moved = pyqtSignal(float)
    split_region_requested = pyqtSignal(list)
    split_all_regions_requested = pyqtSignal(list)
    join_region_requested = pyqtSignal(list)
    join_all_regions_requested = pyqtSignal(list)
    add_track = pyqtSignal(str)
    remove_track = pyqtSignal(str)
    operation_finished = pyqtSignal()
    context_menu_requested = pyqtSignal(QMenu, 'QContextMenuEvent') 

    def __init__(self, timeline_model, settings, parent=None):
        super().__init__(parent)
        self.timeline = timeline_model
        self.settings = settings
        self.playhead_pos_sec = 0.0
        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self.selection_regions = []
        self.dragging_clip = None
        self.dragging_linked_clip = None
        self.dragging_playhead = False
        self.creating_selection_region = False
        self.dragging_selection_region = None
        self.drag_start_pos = QPoint()
        self.drag_original_timeline_start = 0
        self.selection_drag_start_sec = 0.0
        self.drag_selection_start_values = None

        self.highlighted_track_info = None
        self.add_video_track_btn_rect = QRect()
        self.remove_video_track_btn_rect = QRect()
        self.add_audio_track_btn_rect = QRect()
        self.remove_audio_track_btn_rect = QRect()
        
        self.video_tracks_y_start = 0
        self.audio_tracks_y_start = 0

    def sec_to_x(self, sec): return self.HEADER_WIDTH + int(sec * self.PIXELS_PER_SECOND)
    def x_to_sec(self, x): return float(x - self.HEADER_WIDTH) / self.PIXELS_PER_SECOND if x > self.HEADER_WIDTH else 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#333"))

        self.draw_headers(painter)
        self.draw_timescale(painter)
        self.draw_tracks_and_clips(painter)
        self.draw_selections(painter)
        self.draw_playhead(painter)

        total_width = self.sec_to_x(self.timeline.get_total_duration()) + 100
        total_height = self.calculate_total_height()
        self.setMinimumSize(max(self.parent().width(), total_width), total_height)

    def calculate_total_height(self):
        video_tracks_height = (self.timeline.num_video_tracks + 1) * self.TRACK_HEIGHT
        audio_tracks_height = (self.timeline.num_audio_tracks + 1) * self.TRACK_HEIGHT
        return self.TIMESCALE_HEIGHT + video_tracks_height + self.AUDIO_TRACKS_SEPARATOR_Y + audio_tracks_height + 20

    def draw_headers(self, painter):
        painter.save()
        painter.setPen(QColor("#AAA"))
        header_font = QFont("Arial", 9, QFont.Weight.Bold)
        button_font = QFont("Arial", 8)

        y_cursor = self.TIMESCALE_HEIGHT
        
        rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
        painter.fillRect(rect, QColor("#3a3a3a"))
        painter.drawRect(rect)
        self.add_video_track_btn_rect = QRect(rect.left() + 10, rect.top() + (rect.height() - 22)//2, self.HEADER_WIDTH - 20, 22)
        painter.setFont(button_font)
        painter.fillRect(self.add_video_track_btn_rect, QColor("#454"))
        painter.drawText(self.add_video_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "Add Track (+)")
        y_cursor += self.TRACK_HEIGHT
        self.video_tracks_y_start = y_cursor

        for i in range(self.timeline.num_video_tracks):
            track_number = self.timeline.num_video_tracks - i
            rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
            painter.fillRect(rect, QColor("#444"))
            painter.drawRect(rect)
            painter.setFont(header_font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"Video {track_number}")

            if track_number == self.timeline.num_video_tracks and self.timeline.num_video_tracks > 1:
                self.remove_video_track_btn_rect = QRect(rect.right() - 25, rect.top() + 5, 20, 20)
                painter.setFont(button_font)
                painter.fillRect(self.remove_video_track_btn_rect, QColor("#833"))
                painter.drawText(self.remove_video_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "-")
            y_cursor += self.TRACK_HEIGHT

        y_cursor += self.AUDIO_TRACKS_SEPARATOR_Y

        self.audio_tracks_y_start = y_cursor
        for i in range(self.timeline.num_audio_tracks):
            track_number = i + 1
            rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
            painter.fillRect(rect, QColor("#444"))
            painter.drawRect(rect)
            painter.setFont(header_font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"Audio {track_number}")

            if track_number == self.timeline.num_audio_tracks and self.timeline.num_audio_tracks > 1:
                self.remove_audio_track_btn_rect = QRect(rect.right() - 25, rect.top() + 5, 20, 20)
                painter.setFont(button_font)
                painter.fillRect(self.remove_audio_track_btn_rect, QColor("#833"))
                painter.drawText(self.remove_audio_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "-")
            y_cursor += self.TRACK_HEIGHT
        
        rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
        painter.fillRect(rect, QColor("#3a3a3a"))
        painter.drawRect(rect)
        self.add_audio_track_btn_rect = QRect(rect.left() + 10, rect.top() + (rect.height() - 22)//2, self.HEADER_WIDTH - 20, 22)
        painter.setFont(button_font)
        painter.fillRect(self.add_audio_track_btn_rect, QColor("#454"))
        painter.drawText(self.add_audio_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "Add Track (+)")
        
        painter.restore()

    def draw_timescale(self, painter):
        painter.save()
        painter.setPen(QColor("#AAA"))
        painter.setFont(QFont("Arial", 8))
        font_metrics = QFontMetrics(painter.font())
        
        painter.fillRect(QRect(self.HEADER_WIDTH, 0, self.width() - self.HEADER_WIDTH, self.TIMESCALE_HEIGHT), QColor("#222"))
        painter.drawLine(self.HEADER_WIDTH, self.TIMESCALE_HEIGHT - 1, self.width(), self.TIMESCALE_HEIGHT - 1)

        max_time_to_draw = self.x_to_sec(self.width())
        major_interval_sec = 5
        minor_interval_sec = 1

        for t_sec in range(int(max_time_to_draw) + 2):
            x = self.sec_to_x(t_sec)

            if t_sec % major_interval_sec == 0:
                painter.drawLine(x, self.TIMESCALE_HEIGHT - 10, x, self.TIMESCALE_HEIGHT - 1)
                label = f"{t_sec}s"
                label_width = font_metrics.horizontalAdvance(label)
                painter.drawText(x - label_width // 2, self.TIMESCALE_HEIGHT - 12, label)
            elif t_sec % minor_interval_sec == 0:
                painter.drawLine(x, self.TIMESCALE_HEIGHT - 5, x, self.TIMESCALE_HEIGHT - 1)
        painter.restore()

    def get_clip_rect(self, clip):
        if clip.track_type == 'video':
            visual_index = self.timeline.num_video_tracks - clip.track_index
            y = self.video_tracks_y_start + visual_index * self.TRACK_HEIGHT
        else:
            visual_index = clip.track_index - 1
            y = self.audio_tracks_y_start + visual_index * self.TRACK_HEIGHT
        
        x = self.sec_to_x(clip.timeline_start_sec)
        w = int(clip.duration_sec * self.PIXELS_PER_SECOND)
        clip_height = self.TRACK_HEIGHT - 10
        y += (self.TRACK_HEIGHT - clip_height) / 2
        return QRectF(x, y, w, clip_height)

    def draw_tracks_and_clips(self, painter):
        painter.save()
        y_cursor = self.video_tracks_y_start
        for i in range(self.timeline.num_video_tracks):
            rect = QRect(self.HEADER_WIDTH, y_cursor, self.width() - self.HEADER_WIDTH, self.TRACK_HEIGHT)
            painter.fillRect(rect, QColor("#444") if i % 2 == 0 else QColor("#404040"))
            y_cursor += self.TRACK_HEIGHT

        y_cursor = self.audio_tracks_y_start
        for i in range(self.timeline.num_audio_tracks):
            rect = QRect(self.HEADER_WIDTH, y_cursor, self.width() - self.HEADER_WIDTH, self.TRACK_HEIGHT)
            painter.fillRect(rect, QColor("#444") if i % 2 == 0 else QColor("#404040"))
            y_cursor += self.TRACK_HEIGHT

        if self.highlighted_track_info:
            track_type, track_index = self.highlighted_track_info
            y = -1
            if track_type == 'video' and track_index <= self.timeline.num_video_tracks:
                visual_index = self.timeline.num_video_tracks - track_index
                y = self.video_tracks_y_start + visual_index * self.TRACK_HEIGHT
            elif track_type == 'audio' and track_index <= self.timeline.num_audio_tracks:
                visual_index = track_index - 1
                y = self.audio_tracks_y_start + visual_index * self.TRACK_HEIGHT

            if y != -1:
                highlight_rect = QRect(self.HEADER_WIDTH, y, self.width() - self.HEADER_WIDTH, self.TRACK_HEIGHT)
                painter.fillRect(highlight_rect, QColor(255, 255, 0, 40))

        # Draw clips
        for clip in self.timeline.clips:
            clip_rect = self.get_clip_rect(clip)
            base_color = QColor("#46A") if clip.track_type == 'video' else QColor("#48C")
            color = QColor("#5A9") if self.dragging_clip and self.dragging_clip.id == clip.id else base_color
            painter.fillRect(clip_rect, color)
            painter.setPen(QPen(QColor("#FFF"), 1))
            font = QFont("Arial", 10)
            painter.setFont(font)
            text = os.path.basename(clip.source_path)
            font_metrics = QFontMetrics(font)
            text_width = font_metrics.horizontalAdvance(text)
            if text_width > clip_rect.width() - 10: text = font_metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(clip_rect.width() - 10))
            painter.drawText(QPoint(int(clip_rect.left() + 5), int(clip_rect.center().y() + 5)), text)
        painter.restore()

    def draw_selections(self, painter):
        for start_sec, end_sec in self.selection_regions:
            x = self.sec_to_x(start_sec)
            w = int((end_sec - start_sec) * self.PIXELS_PER_SECOND)
            selection_rect = QRectF(x, self.TIMESCALE_HEIGHT, w, self.height() - self.TIMESCALE_HEIGHT)
            painter.fillRect(selection_rect, QColor(100, 100, 255, 80))
            painter.setPen(QColor(150, 150, 255, 150))
            painter.drawRect(selection_rect)

    def draw_playhead(self, painter):
        playhead_x = self.sec_to_x(self.playhead_pos_sec)
        painter.setPen(QPen(QColor("red"), 2))
        painter.drawLine(playhead_x, 0, playhead_x, self.height())

    def y_to_track_info(self, y):
        video_tracks_end_y = self.video_tracks_y_start + self.timeline.num_video_tracks * self.TRACK_HEIGHT
        if self.video_tracks_y_start <= y < video_tracks_end_y:
            visual_index = (y - self.video_tracks_y_start) // self.TRACK_HEIGHT
            track_index = self.timeline.num_video_tracks - visual_index
            return ('video', track_index)
        
        audio_tracks_end_y = self.audio_tracks_y_start + self.timeline.num_audio_tracks * self.TRACK_HEIGHT
        if self.audio_tracks_y_start <= y < audio_tracks_end_y:
            visual_index = (y - self.audio_tracks_y_start) // self.TRACK_HEIGHT
            track_index = visual_index + 1
            return ('audio', track_index)
        return None

    def get_region_at_pos(self, pos: QPoint):
        if pos.y() <= self.TIMESCALE_HEIGHT or pos.x() <= self.HEADER_WIDTH:
            return None
        
        clicked_sec = self.x_to_sec(pos.x())
        for region in reversed(self.selection_regions):
            if region[0] <= clicked_sec <= region[1]:
                return region
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.pos().x() < self.HEADER_WIDTH:
                # Click in headers area
                if self.add_video_track_btn_rect.contains(event.pos()): self.add_track.emit('video')
                elif self.remove_video_track_btn_rect.contains(event.pos()): self.remove_track.emit('video')
                elif self.add_audio_track_btn_rect.contains(event.pos()): self.add_track.emit('audio')
                elif self.remove_audio_track_btn_rect.contains(event.pos()): self.remove_track.emit('audio')
                return

            self.dragging_clip = None
            self.dragging_linked_clip = None
            self.dragging_playhead = False
            self.dragging_selection_region = None
            self.creating_selection_region = False

            for clip in reversed(self.timeline.clips):
                clip_rect = self.get_clip_rect(clip)
                if clip_rect.contains(QPointF(event.pos())):
                    self.dragging_clip = clip
                    self.dragging_linked_clip = next((c for c in self.timeline.clips if c.group_id == clip.group_id and c.id != clip.id), None)
                    self.drag_start_pos = event.pos()
                    self.drag_original_timeline_start = clip.timeline_start_sec
                    break
            
            if not self.dragging_clip:
                region_to_drag = self.get_region_at_pos(event.pos())
                if region_to_drag:
                    self.dragging_selection_region = region_to_drag
                    self.drag_start_pos = event.pos()
                    self.drag_selection_start_values = tuple(region_to_drag)
                else:
                    is_on_timescale = event.pos().y() <= self.TIMESCALE_HEIGHT
                    is_in_track_area = event.pos().y() > self.TIMESCALE_HEIGHT and event.pos().x() > self.HEADER_WIDTH

                    if is_in_track_area:
                        self.creating_selection_region = True
                        self.selection_drag_start_sec = self.x_to_sec(event.pos().x())
                        self.selection_regions.append([self.selection_drag_start_sec, self.selection_drag_start_sec])
                    elif is_on_timescale:
                        self.playhead_pos_sec = max(0, self.x_to_sec(event.pos().x()))
                        self.playhead_moved.emit(self.playhead_pos_sec)
                        self.dragging_playhead = True
            
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.creating_selection_region:
            current_sec = self.x_to_sec(event.pos().x())
            start = min(self.selection_drag_start_sec, current_sec)
            end = max(self.selection_drag_start_sec, current_sec)
            self.selection_regions[-1] = [start, end]
            self.update()
            return
        
        if self.dragging_selection_region:
            delta_x = event.pos().x() - self.drag_start_pos.x()
            time_delta = delta_x / self.PIXELS_PER_SECOND
            
            original_start, original_end = self.drag_selection_start_values
            duration = original_end - original_start
            new_start = max(0, original_start + time_delta)
            
            self.dragging_selection_region[0] = new_start
            self.dragging_selection_region[1] = new_start + duration
            
            self.update()
            return

        if self.dragging_playhead:
            self.playhead_pos_sec = max(0, self.x_to_sec(event.pos().x()))
            self.playhead_moved.emit(self.playhead_pos_sec)
            self.update()
        elif self.dragging_clip:
            self.highlighted_track_info = None
            new_track_info = self.y_to_track_info(event.pos().y())
            if new_track_info:
                new_track_type, new_track_index = new_track_info
                if new_track_type == self.dragging_clip.track_type:
                    self.dragging_clip.track_index = new_track_index
                    if self.dragging_linked_clip:
                        self.dragging_linked_clip.track_index = new_track_index
                        if self.dragging_clip.track_type == 'video':
                            if new_track_index > self.timeline.num_audio_tracks:
                                self.timeline.num_audio_tracks = new_track_index
                            self.highlighted_track_info = ('audio', new_track_index)
                        else:
                            if new_track_index > self.timeline.num_video_tracks:
                                self.timeline.num_video_tracks = new_track_index
                            self.highlighted_track_info = ('video', new_track_index)

            delta_x = event.pos().x() - self.drag_start_pos.x()
            time_delta = delta_x / self.PIXELS_PER_SECOND
            new_start_time = self.drag_original_timeline_start + time_delta

            for other_clip in self.timeline.clips:
                if other_clip.id == self.dragging_clip.id: continue
                if self.dragging_linked_clip and other_clip.id == self.dragging_linked_clip.id: continue
                if (other_clip.track_type != self.dragging_clip.track_type or 
                    other_clip.track_index != self.dragging_clip.track_index):
                    continue

                is_overlapping = (new_start_time < other_clip.timeline_end_sec and
                                  new_start_time + self.dragging_clip.duration_sec > other_clip.timeline_start_sec)
                
                if is_overlapping:
                    if time_delta > 0: new_start_time = other_clip.timeline_start_sec - self.dragging_clip.duration_sec
                    else: new_start_time = other_clip.timeline_end_sec
                    break 

            final_start_time = max(0, new_start_time)
            self.dragging_clip.timeline_start_sec = final_start_time
            if self.dragging_linked_clip:
                self.dragging_linked_clip.timeline_start_sec = final_start_time
            
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.creating_selection_region:
                self.creating_selection_region = False
                if self.selection_regions:
                    start, end = self.selection_regions[-1]
                    if (end - start) * self.PIXELS_PER_SECOND < 2:
                        self.selection_regions.pop()
            
            if self.dragging_selection_region:
                self.dragging_selection_region = None
                self.drag_selection_start_values = None

            self.dragging_playhead = False
            if self.dragging_clip:
                self.timeline.clips.sort(key=lambda c: c.timeline_start_sec)
                self.highlighted_track_info = None
                self.operation_finished.emit()
            self.dragging_clip = None
            self.dragging_linked_clip = None
            
            self.update()

    def contextMenuEvent(self, event: 'QContextMenuEvent'):
        menu = QMenu(self)
        
        region_at_pos = self.get_region_at_pos(event.pos())
        if region_at_pos:
            split_this_action = menu.addAction("Split This Region")
            split_all_action = menu.addAction("Split All Regions")
            join_this_action = menu.addAction("Join This Region")
            join_all_action = menu.addAction("Join All Regions")
            menu.addSeparator()
            clear_this_action = menu.addAction("Clear This Region")
            clear_all_action = menu.addAction("Clear All Regions")
            split_this_action.triggered.connect(lambda: self.split_region_requested.emit(region_at_pos))
            split_all_action.triggered.connect(lambda: self.split_all_regions_requested.emit(self.selection_regions))
            join_this_action.triggered.connect(lambda: self.join_region_requested.emit(region_at_pos))
            join_all_action.triggered.connect(lambda: self.join_all_regions_requested.emit(self.selection_regions))
            clear_this_action.triggered.connect(lambda: self.clear_region(region_at_pos))
            clear_all_action.triggered.connect(self.clear_all_regions)

        clip_at_pos = None
        for clip in self.timeline.clips:
            if self.get_clip_rect(clip).contains(QPointF(event.pos())):
                clip_at_pos = clip
                break
        
        if clip_at_pos:
            if not menu.isEmpty(): menu.addSeparator()
            split_action = menu.addAction("Split Clip")
            playhead_time = self.playhead_pos_sec
            is_playhead_over_clip = (clip_at_pos.timeline_start_sec < playhead_time < clip_at_pos.timeline_end_sec)
            split_action.setEnabled(is_playhead_over_clip)
            split_action.triggered.connect(lambda: self.split_requested.emit(clip_at_pos))

        self.context_menu_requested.emit(menu, event)

        if not menu.isEmpty():
            menu.exec(self.mapToGlobal(event.pos()))

    def clear_region(self, region_to_clear):
        if region_to_clear in self.selection_regions:
            self.selection_regions.remove(region_to_clear)
            self.update()
    
    def clear_all_regions(self):
        self.selection_regions.clear()
        self.update()

class SettingsDialog(QDialog):
    def __init__(self, parent_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)
        self.seek_anywhere_checkbox = QCheckBox("Allow seeking by clicking anywhere on the timeline")
        self.seek_anywhere_checkbox.setChecked(parent_settings.get("allow_seek_anywhere", False))
        layout.addWidget(self.seek_anywhere_checkbox)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_settings(self):
        return {"allow_seek_anywhere": self.seek_anywhere_checkbox.isChecked()}

class MainWindow(QMainWindow):
    def __init__(self, project_to_load=None):
        super().__init__()
        self.setWindowTitle("Inline AI Video Editor")
        self.setGeometry(100, 100, 1200, 800)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks)

        self.timeline = Timeline()
        self.export_thread = None
        self.export_worker = None
        self.current_project_path = None
        self.settings = {}
        self.settings_file = "settings.json"
        self.is_shutting_down = False
        self._load_settings()

        self.plugin_manager = PluginManager(self)
        self.plugin_manager.discover_and_load_plugins()

        self.project_fps = 25.0
        self.project_width = 1280
        self.project_height = 720
        self.playback_timer = QTimer(self)
        self.playback_process = None
        self.playback_clip = None

        self.splitter = QSplitter(Qt.Orientation.Vertical)

        self.preview_widget = QLabel()
        self.preview_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_widget.setMinimumSize(640, 360)
        self.preview_widget.setFrameShape(QFrame.Shape.Box)
        self.preview_widget.setStyleSheet("background-color: black; color: white;")
        self.splitter.addWidget(self.preview_widget)

        self.timeline_widget = TimelineWidget(self.timeline, self.settings)
        self.timeline_scroll_area = QScrollArea()
        self.timeline_scroll_area.setWidgetResizable(True)
        self.timeline_scroll_area.setWidget(self.timeline_widget)
        self.timeline_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.timeline_scroll_area.setMinimumHeight(250)
        self.splitter.addWidget(self.timeline_scroll_area)

        container_widget = QWidget()
        main_layout = QVBoxLayout(container_widget)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(self.splitter, 1)

        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 5, 0, 5)
        self.play_pause_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.frame_back_button = QPushButton("<")
        self.frame_forward_button = QPushButton(">")
        controls_layout.addStretch()
        controls_layout.addWidget(self.frame_back_button)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.frame_forward_button)
        controls_layout.addStretch()
        main_layout.addWidget(controls_widget)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready. Create or open a project from the File menu.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar, 1)
        main_layout.addLayout(status_layout)
        
        self.setCentralWidget(container_widget)

        self.managed_widgets = {
            'preview': {'widget': self.preview_widget, 'name': 'Video Preview', 'action': None},
            'timeline': {'widget': self.timeline_scroll_area, 'name': 'Timeline', 'action': None}
        }
        self.plugin_menu_actions = {}
        self.windows_menu = None
        self._create_menu_bar()

        self.splitter_save_timer = QTimer(self)
        self.splitter_save_timer.setSingleShot(True)
        self.splitter_save_timer.timeout.connect(self._save_settings)
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        self.timeline_widget.split_requested.connect(self.split_clip_at_playhead)
        self.timeline_widget.playhead_moved.connect(self.seek_preview)
        self.timeline_widget.split_region_requested.connect(self.on_split_region)
        self.timeline_widget.split_all_regions_requested.connect(self.on_split_all_regions)
        self.timeline_widget.join_region_requested.connect(self.on_join_region)
        self.timeline_widget.join_all_regions_requested.connect(self.on_join_all_regions)
        self.timeline_widget.add_track.connect(self.add_track)
        self.timeline_widget.remove_track.connect(self.remove_track)
        self.timeline_widget.operation_finished.connect(self.prune_empty_tracks)

        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.stop_button.clicked.connect(self.stop_playback)
        self.frame_back_button.clicked.connect(lambda: self.step_frame(-1))
        self.frame_forward_button.clicked.connect(lambda: self.step_frame(1))
        self.playback_timer.timeout.connect(self.advance_playback_frame)
        
        self.plugin_manager.load_enabled_plugins_from_settings(self.settings.get("enabled_plugins", []))
        self._apply_loaded_settings()
        self.seek_preview(0)
        
        if not self.settings_file_was_loaded: self._save_settings()
        if project_to_load: QTimer.singleShot(100, lambda: self._load_project_from_path(project_to_load))

    def prune_empty_tracks(self):
        pruned_something = False

        while self.timeline.num_video_tracks > 1:
            highest_track_index = self.timeline.num_video_tracks
            is_track_occupied = any(c for c in self.timeline.clips 
                                    if c.track_type == 'video' and c.track_index == highest_track_index)
            if is_track_occupied:
                break
            else:
                self.timeline.num_video_tracks -= 1
                pruned_something = True

        while self.timeline.num_audio_tracks > 1:
            highest_track_index = self.timeline.num_audio_tracks
            is_track_occupied = any(c for c in self.timeline.clips 
                                    if c.track_type == 'audio' and c.track_index == highest_track_index)
            if is_track_occupied:
                break
            else:
                self.timeline.num_audio_tracks -= 1
                pruned_something = True
        
        if pruned_something:
            self.timeline_widget.update()


    def add_track(self, track_type):
        if track_type == 'video':
            self.timeline.num_video_tracks += 1
        elif track_type == 'audio':
            self.timeline.num_audio_tracks += 1
        self.timeline_widget.update()
    
    def remove_track(self, track_type):
        if track_type == 'video' and self.timeline.num_video_tracks > 1:
            self.timeline.num_video_tracks -= 1
        elif track_type == 'audio' and self.timeline.num_audio_tracks > 1:
            self.timeline.num_audio_tracks -= 1
        self.timeline_widget.update()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        new_action = QAction("&New Project", self); new_action.triggered.connect(self.new_project)
        open_action = QAction("&Open Project...", self); open_action.triggered.connect(self.open_project)
        save_action = QAction("&Save Project As...", self); save_action.triggered.connect(self.save_project_as)
        add_video_action = QAction("&Add Video...", self); add_video_action.triggered.connect(self.add_video_clip)
        export_action = QAction("&Export Video...", self); export_action.triggered.connect(self.export_video)
        settings_action = QAction("Se&ttings...", self); settings_action.triggered.connect(self.open_settings_dialog)
        exit_action = QAction("E&xit", self); exit_action.triggered.connect(self.close)
        file_menu.addAction(new_action); file_menu.addAction(open_action); file_menu.addAction(save_action)
        file_menu.addSeparator(); file_menu.addAction(add_video_action); file_menu.addAction(export_action)
        file_menu.addSeparator(); file_menu.addAction(settings_action); file_menu.addSeparator(); file_menu.addAction(exit_action)
        
        edit_menu = menu_bar.addMenu("&Edit")
        split_action = QAction("Split Clip at Playhead", self); split_action.triggered.connect(self.split_clip_at_playhead)
        edit_menu.addAction(split_action)
        
        plugins_menu = menu_bar.addMenu("&Plugins")
        for name, data in self.plugin_manager.plugins.items():
            plugin_action = QAction(name, self, checkable=True)
            plugin_action.setChecked(data['enabled'])
            plugin_action.toggled.connect(lambda checked, n=name: self.toggle_plugin(n, checked))
            plugins_menu.addAction(plugin_action)
            self.plugin_menu_actions[name] = plugin_action

        plugins_menu.addSeparator()
        manage_action = QAction("Manage plugins...", self)
        manage_action.triggered.connect(self.open_manage_plugins_dialog)
        plugins_menu.addAction(manage_action)
        
        self.windows_menu = menu_bar.addMenu("&Windows")
        for key, data in self.managed_widgets.items():
            action = QAction(data['name'], self, checkable=True)
            action.toggled.connect(lambda checked, k=key: self.toggle_widget_visibility(k, checked))
            data['action'] = action
            self.windows_menu.addAction(action)

    def _start_playback_stream_at(self, time_sec):
        self._stop_playback_stream()
        clip = next((c for c in self.timeline.clips if c.track_type == 'video' and c.timeline_start_sec <= time_sec < c.timeline_end_sec), None)
        if not clip: return
        self.playback_clip = clip
        clip_time = time_sec - clip.timeline_start_sec + clip.clip_start_sec
        try:
            args = (ffmpeg.input(self.playback_clip.source_path, ss=clip_time).output('pipe:', format='rawvideo', pix_fmt='rgb24', r=self.project_fps).compile())
            self.playback_process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Failed to start playback stream: {e}"); self._stop_playback_stream()

    def _stop_playback_stream(self):
        if self.playback_process:
            if self.playback_process.poll() is None:
                self.playback_process.terminate()
                try: self.playback_process.wait(timeout=0.5)
                except subprocess.TimeoutExpired: self.playback_process.kill(); self.playback_process.wait()
            self.playback_process = None
        self.playback_clip = None

    def _set_project_properties_from_clip(self, source_path):
        try:
            probe = ffmpeg.probe(source_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if video_stream:
                self.project_width = int(video_stream['width']); self.project_height = int(video_stream['height'])
                if 'r_frame_rate' in video_stream and video_stream['r_frame_rate'] != '0/0':
                    num, den = map(int, video_stream['r_frame_rate'].split('/'))
                    if den > 0: self.project_fps = num / den
                print(f"Project properties set: {self.project_width}x{self.project_height} @ {self.project_fps:.2f} FPS")
                return True
        except Exception as e: print(f"Could not probe for project properties: {e}")
        return False

    def get_frame_data_at_time(self, time_sec):
        clip_at_time = next((c for c in self.timeline.clips if c.track_type == 'video' and c.timeline_start_sec <= time_sec < c.timeline_end_sec), None)
        if not clip_at_time:
            return (None, 0, 0)
        try:
            clip_time = time_sec - clip_at_time.timeline_start_sec + clip_at_time.clip_start_sec
            out, _ = (
                ffmpeg
                .input(clip_at_time.source_path, ss=clip_time)
                .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            return (out, self.project_width, self.project_height)
        except ffmpeg.Error as e:
            print(f"Error extracting frame data: {e.stderr}")
            return (None, 0, 0)

    def get_frame_at_time(self, time_sec):
        clip_at_time = next((c for c in self.timeline.clips if c.track_type == 'video' and c.timeline_start_sec <= time_sec < c.timeline_end_sec), None)
        black_pixmap = QPixmap(self.project_width, self.project_height); black_pixmap.fill(QColor("black"))
        if not clip_at_time: return black_pixmap
        try:
            clip_time = time_sec - clip_at_time.timeline_start_sec + clip_at_time.clip_start_sec
            out, _ = (ffmpeg.input(clip_at_time.source_path, ss=clip_time).output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True))
            image = QImage(out, self.project_width, self.project_height, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(image)
        except ffmpeg.Error as e: print(f"Error extracting frame: {e.stderr}"); return black_pixmap

    def seek_preview(self, time_sec):
        self._stop_playback_stream()
        self.timeline_widget.playhead_pos_sec = time_sec
        self.timeline_widget.update()
        frame_pixmap = self.get_frame_at_time(time_sec)
        if frame_pixmap:
            scaled_pixmap = frame_pixmap.scaled(self.preview_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_widget.setPixmap(scaled_pixmap)

    def toggle_playback(self):
        if self.playback_timer.isActive(): self.playback_timer.stop(); self._stop_playback_stream(); self.play_pause_button.setText("Play")
        else:
            if not self.timeline.clips: return
            if self.timeline_widget.playhead_pos_sec >= self.timeline.get_total_duration(): self.timeline_widget.playhead_pos_sec = 0.0
            self.playback_timer.start(int(1000 / self.project_fps)); self.play_pause_button.setText("Pause")

    def stop_playback(self): self.playback_timer.stop(); self._stop_playback_stream(); self.play_pause_button.setText("Play"); self.seek_preview(0.0)
    def step_frame(self, direction):
        if not self.timeline.clips: return
        self.playback_timer.stop(); self.play_pause_button.setText("Play"); self._stop_playback_stream()
        frame_duration = 1.0 / self.project_fps
        new_time = self.timeline_widget.playhead_pos_sec + (direction * frame_duration)
        self.seek_preview(max(0, min(new_time, self.timeline.get_total_duration())))

    def advance_playback_frame(self):
        frame_duration = 1.0 / self.project_fps
        new_time = self.timeline_widget.playhead_pos_sec + frame_duration
        if new_time > self.timeline.get_total_duration(): self.stop_playback(); return
        self.timeline_widget.playhead_pos_sec = new_time; self.timeline_widget.update()
        clip_at_new_time = next((c for c in self.timeline.clips if c.track_type == 'video' and c.timeline_start_sec <= new_time < c.timeline_end_sec), None)
        if not clip_at_new_time:
            self._stop_playback_stream()
            black_pixmap = QPixmap(self.project_width, self.project_height); black_pixmap.fill(QColor("black"))
            scaled_pixmap = black_pixmap.scaled(self.preview_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_widget.setPixmap(scaled_pixmap); return
        if self.playback_clip is None or self.playback_clip.id != clip_at_new_time.id: self._start_playback_stream_at(new_time)
        if self.playback_process:
            frame_size = self.project_width * self.project_height * 3
            frame_bytes = self.playback_process.stdout.read(frame_size)
            if len(frame_bytes) == frame_size:
                image = QImage(frame_bytes, self.project_width, self.project_height, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(self.preview_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_widget.setPixmap(scaled_pixmap)
            else: self._stop_playback_stream()

    def _load_settings(self):
        self.settings_file_was_loaded = False
        defaults = {"allow_seek_anywhere": False, "window_visibility": {"preview": True, "timeline": True}, "splitter_state": None, "enabled_plugins": []}
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f: self.settings = json.load(f)
                self.settings_file_was_loaded = True
                for key, value in defaults.items():
                    if key not in self.settings: self.settings[key] = value
            except (json.JSONDecodeError, IOError): self.settings = defaults
        else: self.settings = defaults

    def _save_settings(self):
        self.settings["splitter_state"] = self.splitter.saveState().toHex().data().decode('ascii')
        visibility_to_save = {
            key: data['action'].isChecked()
            for key, data in self.managed_widgets.items() if data.get('action')
        }

        self.settings["window_visibility"] = visibility_to_save
        self.settings['enabled_plugins'] = self.plugin_manager.get_enabled_plugin_names()
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=4)
        except IOError as e:
            print(f"Error saving settings: {e}")

    def _apply_loaded_settings(self):
        visibility_settings = self.settings.get("window_visibility", {})
        for key, data in self.managed_widgets.items():
            if data.get('plugin'):
                continue

            is_visible = visibility_settings.get(key, True)
            data['widget'].setVisible(is_visible)
            if data['action']: data['action'].setChecked(is_visible)
            
        splitter_state = self.settings.get("splitter_state")
        if splitter_state: self.splitter.restoreState(QByteArray.fromHex(splitter_state.encode('ascii')))

    def on_splitter_moved(self, pos, index): self.splitter_save_timer.start(500)
    
    def toggle_widget_visibility(self, key, checked):
        if self.is_shutting_down:
            return
        if key in self.managed_widgets:
            self.managed_widgets[key]['widget'].setVisible(checked)
            self._save_settings()

    def open_settings_dialog(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dialog.get_settings()); self._save_settings(); self.status_label.setText("Settings updated.")

    def new_project(self):
        self.timeline.clips.clear(); self.timeline.num_video_tracks = 1; self.timeline.num_audio_tracks = 1
        self.current_project_path = None; self.stop_playback(); self.timeline_widget.update()
        self.status_label.setText("New project created. Add video clips to begin.")

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Project Files (*.json)")
        if not path: return
        project_data = {
            "clips": [{"source_path": c.source_path, "timeline_start_sec": c.timeline_start_sec, "clip_start_sec": c.clip_start_sec, "duration_sec": c.duration_sec, "track_index": c.track_index, "track_type": c.track_type, "group_id": c.group_id} for c in self.timeline.clips],
            "settings": {"num_video_tracks": self.timeline.num_video_tracks, "num_audio_tracks": self.timeline.num_audio_tracks}
        }
        try:
            with open(path, "w") as f: json.dump(project_data, f, indent=4)
            self.current_project_path = path; self.status_label.setText(f"Project saved to {os.path.basename(path)}")
        except Exception as e: self.status_label.setText(f"Error saving project: {e}")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Project Files (*.json)")
        if path: self._load_project_from_path(path)

    def _load_project_from_path(self, path):
        try:
            with open(path, "r") as f: project_data = json.load(f)
            self.timeline.clips.clear()
            for clip_data in project_data["clips"]:
                if not os.path.exists(clip_data["source_path"]):
                    self.status_label.setText(f"Error: Missing media file {clip_data['source_path']}"); self.timeline.clips.clear(); self.timeline_widget.update(); return
                self.timeline.add_clip(TimelineClip(**clip_data))
            
            project_settings = project_data.get("settings", {})
            self.timeline.num_video_tracks = project_settings.get("num_video_tracks", 1)
            self.timeline.num_audio_tracks = project_settings.get("num_audio_tracks", 1)

            self.current_project_path = path
            if self.timeline.clips: self._set_project_properties_from_clip(self.timeline.clips[0].source_path)
            self.prune_empty_tracks()
            self.timeline_widget.update(); self.stop_playback()
            self.status_label.setText(f"Project '{os.path.basename(path)}' loaded.")
        except Exception as e: self.status_label.setText(f"Error opening project: {e}")

    def add_video_clip(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.mov *.avi)")
        if not file_path: return
        if not self.timeline.clips:
            if not self._set_project_properties_from_clip(file_path): self.status_label.setText("Error: Could not determine video properties from file."); return
        try:
            self.status_label.setText(f"Probing {os.path.basename(file_path)}..."); QApplication.processEvents()
            probe = ffmpeg.probe(file_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if not video_stream: raise ValueError("No video stream found.")
            duration = float(video_stream.get('duration', probe['format'].get('duration', 0)))
            has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])
            timeline_start = self.timeline.get_total_duration()

            self._add_clip_to_timeline(
                source_path=file_path,
                timeline_start_sec=timeline_start,
                duration_sec=duration,
                clip_start_sec=0,
                video_track_index=1,
                audio_track_index=1 if has_audio else None
            )

            self.timeline_widget.update(); self.seek_preview(self.timeline_widget.playhead_pos_sec)
        except Exception as e: self.status_label.setText(f"Error adding file: {e}")

    def _add_clip_to_timeline(self, source_path, timeline_start_sec, duration_sec, clip_start_sec=0.0, video_track_index=None, audio_track_index=None):
        group_id = str(uuid.uuid4())
        
        if video_track_index is not None:
             video_clip = TimelineClip(source_path, timeline_start_sec, clip_start_sec, duration_sec, video_track_index, 'video', group_id)
             self.timeline.add_clip(video_clip)
        
        if audio_track_index is not None:
             audio_clip = TimelineClip(source_path, timeline_start_sec, clip_start_sec, duration_sec, audio_track_index, 'audio', group_id)
             self.timeline.add_clip(audio_clip)

        self.timeline_widget.update()
        self.status_label.setText(f"Added {os.path.basename(source_path)}.")

    def _split_at_time(self, clip_to_split, time_sec, new_group_id=None):
        if not (clip_to_split.timeline_start_sec < time_sec < clip_to_split.timeline_end_sec): return False
        split_point = time_sec - clip_to_split.timeline_start_sec
        orig_dur = clip_to_split.duration_sec
        
        group_id_for_new_clip = new_group_id if new_group_id is not None else clip_to_split.group_id
        
        new_clip = TimelineClip(clip_to_split.source_path, time_sec, clip_to_split.clip_start_sec + split_point, orig_dur - split_point, clip_to_split.track_index, clip_to_split.track_type, group_id_for_new_clip)
        clip_to_split.duration_sec = split_point
        self.timeline.add_clip(new_clip)
        return True

    def split_clip_at_playhead(self, clip_to_split=None):
        playhead_time = self.timeline_widget.playhead_pos_sec
        if not clip_to_split:
            clip_to_split = next((c for c in self.timeline.clips if c.timeline_start_sec < playhead_time < c.timeline_end_sec), None)
        
        if not clip_to_split:
            self.status_label.setText("Playhead is not over a clip to split.")
            return
        
        linked_clip = next((c for c in self.timeline.clips if c.group_id == clip_to_split.group_id and c.id != clip_to_split.id), None)
        
        new_right_side_group_id = str(uuid.uuid4())
        
        split1 = self._split_at_time(clip_to_split, playhead_time, new_group_id=new_right_side_group_id)
        split2 = True
        if linked_clip:
            split2 = self._split_at_time(linked_clip, playhead_time, new_group_id=new_right_side_group_id)

        if split1 and split2:
            self.timeline_widget.update()
            self.status_label.setText("Clip split.")
        else:
            self.status_label.setText("Failed to split clip.")

    def on_split_region(self, region):
        start_sec, end_sec = region;
        clips = list(self.timeline.clips)
        for clip in clips: self._split_at_time(clip, end_sec)
        for clip in clips: self._split_at_time(clip, start_sec)
        self.timeline_widget.clear_region(region); self.timeline_widget.update(); self.status_label.setText("Region split.")

    def on_split_all_regions(self, regions):
        split_points = set()
        for start, end in regions: split_points.add(start); split_points.add(end)
        for point in sorted(list(split_points)):
            group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
            new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
            
            for clip in list(self.timeline.clips):
                if clip.group_id in new_group_ids:
                    self._split_at_time(clip, point, new_group_ids[clip.group_id])

        self.timeline_widget.clear_all_regions(); self.timeline_widget.update(); self.status_label.setText("All regions split.")

    def on_join_region(self, region):
        start_sec, end_sec = region; duration_to_remove = end_sec - start_sec
        if duration_to_remove <= 0.01: return
        
        for point in [start_sec, end_sec]:
             group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
             new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
             for clip in list(self.timeline.clips):
                 if clip.group_id in new_group_ids:
                     self._split_at_time(clip, point, new_group_ids[clip.group_id])

        clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_sec >= start_sec and c.timeline_start_sec < end_sec]
        for clip in clips_to_remove: self.timeline.clips.remove(clip)
        for clip in self.timeline.clips:
            if clip.timeline_start_sec >= end_sec: clip.timeline_start_sec -= duration_to_remove
        self.timeline.clips.sort(key=lambda c: c.timeline_start_sec); self.timeline_widget.clear_region(region)
        self.timeline_widget.update(); self.status_label.setText("Region joined (content removed).")
        self.prune_empty_tracks()

    def on_join_all_regions(self, regions):
        for region in sorted(regions, key=lambda r: r[0], reverse=True):
            start_sec, end_sec = region; duration_to_remove = end_sec - start_sec
            if duration_to_remove <= 0.01: continue

            for point in [start_sec, end_sec]:
                group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
                new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                for clip in list(self.timeline.clips):
                    if clip.group_id in new_group_ids:
                        self._split_at_time(clip, point, new_group_ids[clip.group_id])
            
            clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_sec >= start_sec and c.timeline_start_sec < end_sec]
            for clip in clips_to_remove:
                try: self.timeline.clips.remove(clip)
                except ValueError: pass
            for clip in self.timeline.clips:
                if clip.timeline_start_sec >= end_sec: clip.timeline_start_sec -= duration_to_remove

        self.timeline.clips.sort(key=lambda c: c.timeline_start_sec); self.timeline_widget.clear_all_regions()
        self.timeline_widget.update(); self.status_label.setText("All regions joined.")
        self.prune_empty_tracks()

    def export_video(self):
        if not self.timeline.clips: self.status_label.setText("Timeline is empty."); return
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "MP4 Files (*.mp4)")
        if not output_path: return

        w, h, fr_str, total_dur = self.project_width, self.project_height, str(self.project_fps), self.timeline.get_total_duration()
        sample_rate, channel_layout = '44100', 'stereo'
        
        input_files = {clip.source_path: ffmpeg.input(clip.source_path) for clip in self.timeline.clips}
        
        last_video_stream = ffmpeg.input(f'color=c=black:s={w}x{h}:r={fr_str}:d={total_dur}', f='lavfi')
        for i in range(self.timeline.num_video_tracks):
            track_clips = sorted([c for c in self.timeline.clips if c.track_type == 'video' and c.track_index == i + 1], key=lambda c: c.timeline_start_sec)
            if not track_clips: continue
            
            track_segments = []
            last_end = track_clips[0].timeline_start_sec 
            
            for clip in track_clips:
                gap = clip.timeline_start_sec - last_end
                if gap > 0.01: 
                    track_segments.append(ffmpeg.input(f'color=c=black@0.0:s={w}x{h}:r={fr_str}:d={gap}', f='lavfi').filter('format', pix_fmts='rgba'))
                
                v_seg = (input_files[clip.source_path].video.trim(start=clip.clip_start_sec, duration=clip.duration_sec).setpts('PTS-STARTPTS')
                        .filter('scale', w, h, force_original_aspect_ratio='decrease').filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black').filter('format', pix_fmts='rgba'))
                track_segments.append(v_seg)
                last_end = clip.timeline_end_sec
            
            track_stream = ffmpeg.concat(*track_segments, v=1, a=0).filter('setpts', f'PTS-STARTPTS+{track_clips[0].timeline_start_sec}/TB')
            last_video_stream = ffmpeg.overlay(last_video_stream, track_stream)
        final_video = last_video_stream.filter('format', pix_fmts='yuv420p').filter('fps', fps=self.project_fps)

        track_audio_streams = []
        for i in range(self.timeline.num_audio_tracks):
            track_clips = sorted([c for c in self.timeline.clips if c.track_type == 'audio' and c.track_index == i + 1], key=lambda c: c.timeline_start_sec)
            if not track_clips: continue

            track_segments = []
            last_end = track_clips[0].timeline_start_sec

            for clip in track_clips:
                gap = clip.timeline_start_sec - last_end
                if gap > 0.01: 
                    track_segments.append(ffmpeg.input(f'anullsrc=r={sample_rate}:cl={channel_layout}:d={gap}', f='lavfi'))
                
                a_seg = input_files[clip.source_path].audio.filter('atrim', start=clip.clip_start_sec, duration=clip.duration_sec).filter('asetpts', 'PTS-STARTPTS')
                track_segments.append(a_seg)
                last_end = clip.timeline_end_sec

            track_audio_streams.append(ffmpeg.concat(*track_segments, v=0, a=1).filter('adelay', f'{int(track_clips[0].timeline_start_sec * 1000)}ms', all=True))
        
        if track_audio_streams:
            final_audio = ffmpeg.filter(track_audio_streams, 'amix', inputs=len(track_audio_streams), duration='longest')
        else:
            final_audio = ffmpeg.input(f'anullsrc=r={sample_rate}:cl={channel_layout}:d={total_dur}', f='lavfi')

        output_args = {'vcodec': 'libx264', 'acodec': 'aac', 'pix_fmt': 'yuv420p', 'b:v': '5M'}
        try:
            ffmpeg_cmd = ffmpeg.output(final_video, final_audio, output_path, **output_args).overwrite_output().compile()
            self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.status_label.setText("Exporting...")
            self.export_thread = QThread()
            self.export_worker = ExportWorker(ffmpeg_cmd, total_dur)
            self.export_worker.moveToThread(self.export_thread)
            self.export_thread.started.connect(self.export_worker.run_export)
            self.export_worker.finished.connect(self.on_export_finished)
            self.export_worker.progress.connect(self.progress_bar.setValue)
            self.export_worker.finished.connect(self.export_thread.quit)
            self.export_worker.finished.connect(self.export_worker.deleteLater)
            self.export_thread.finished.connect(self.export_thread.deleteLater)
            self.export_thread.finished.connect(self.on_thread_finished_cleanup)
            self.export_thread.start()
        except ffmpeg.Error as e:
            self.status_label.setText(f"FFmpeg error: {e.stderr}")
            print(e.stderr)

    def on_export_finished(self, message):
        self.status_label.setText(message)
        self.progress_bar.setVisible(False)

    def on_thread_finished_cleanup(self):
        self.export_thread = None
        self.export_worker = None
    
    def add_dock_widget(self, plugin_instance, widget, title, area=Qt.DockWidgetArea.RightDockWidgetArea, show_on_creation=True):
        widget_key = f"plugin_{plugin_instance.name}_{title}".replace(' ', '_').lower()

        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)

        visibility_settings = self.settings.get("window_visibility", {})
        initial_visibility = visibility_settings.get(widget_key, show_on_creation)
        
        dock.setVisible(initial_visibility)

        action = QAction(title, self, checkable=True)
        action.toggled.connect(lambda checked, k=widget_key: self.toggle_widget_visibility(k, checked))
        dock.visibilityChanged.connect(action.setChecked)
        
        action.setChecked(dock.isVisible()) 

        self.windows_menu.addAction(action)

        self.managed_widgets[widget_key] = {
            'widget': dock,
            'name': title,
            'action': action,
            'plugin': plugin_instance.name
        }

        return dock
        
    def update_plugin_ui_visibility(self, plugin_name, is_enabled):
        for key, data in self.managed_widgets.items():
            if data.get('plugin') == plugin_name:
                data['action'].setVisible(is_enabled)
                if not is_enabled:
                    data['widget'].hide()

    def toggle_plugin(self, name, checked):
        if checked:
            self.plugin_manager.enable_plugin(name)
        else:
            self.plugin_manager.disable_plugin(name)
        self._save_settings()

    def toggle_plugin_action(self, name, checked):
        if name in self.plugin_menu_actions:
            action = self.plugin_menu_actions[name]
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def open_manage_plugins_dialog(self):
        dialog = ManagePluginsDialog(self.plugin_manager, self)
        dialog.app = self
        dialog.exec()

    def closeEvent(self, event):
        self.is_shutting_down = True
        self._save_settings()
        self._stop_playback_stream()
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.quit()
            self.export_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    project_to_load_on_startup = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path) and path.lower().endswith('.json'):
            project_to_load_on_startup = path
            print(f"Loading project: {path}")
    window = MainWindow(project_to_load=project_to_load_on_startup)
    window.show()
    sys.exit(app.exec())