import sys
import os
import uuid
import subprocess
import re
import json
import ffmpeg
import copy
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QScrollArea, QFrame, QProgressBar, QDialog,
                             QCheckBox, QDialogButtonBox, QMenu, QSplitter, QDockWidget, QListWidget, QListWidgetItem, QMessageBox)
from PyQt6.QtGui import (QPainter, QColor, QPen, QFont, QFontMetrics, QMouseEvent, QAction,
                         QPixmap, QImage, QDrag, QCursor)
from PyQt6.QtCore import (Qt, QPoint, QRect, QRectF, QSize, QPointF, QObject, QThread,
                          pyqtSignal, QTimer, QByteArray, QMimeData)

from plugins import PluginManager, ManagePluginsDialog
from undo import UndoStack, TimelineStateChangeCommand, MoveClipsCommand

class TimelineClip:
    def __init__(self, source_path, timeline_start_sec, clip_start_sec, duration_sec, track_index, track_type, media_type, group_id):
        self.id = str(uuid.uuid4())
        self.source_path = source_path
        self.timeline_start_sec = timeline_start_sec
        self.clip_start_sec = clip_start_sec
        self.duration_sec = duration_sec
        self.track_index = track_index
        self.track_type = track_type
        self.media_type = media_type
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
    TIMESCALE_HEIGHT = 30
    HEADER_WIDTH = 120
    TRACK_HEIGHT = 50
    AUDIO_TRACKS_SEPARATOR_Y = 15
    RESIZE_HANDLE_WIDTH = 8
    SNAP_THRESHOLD_PIXELS = 8

    split_requested = pyqtSignal(object)
    delete_clip_requested = pyqtSignal(object)
    playhead_moved = pyqtSignal(float)
    split_region_requested = pyqtSignal(list)
    split_all_regions_requested = pyqtSignal(list)
    join_region_requested = pyqtSignal(list)
    join_all_regions_requested = pyqtSignal(list)
    delete_region_requested = pyqtSignal(list)
    delete_all_regions_requested = pyqtSignal(list)
    add_track = pyqtSignal(str)
    remove_track = pyqtSignal(str)
    operation_finished = pyqtSignal()
    context_menu_requested = pyqtSignal(QMenu, 'QContextMenuEvent')

    def __init__(self, timeline_model, settings, project_fps, parent=None):
        super().__init__(parent)
        self.timeline = timeline_model
        self.settings = settings
        self.playhead_pos_sec = 0.0
        self.scroll_area = None
        
        self.pixels_per_second = 50.0 
        self.max_pixels_per_second = 1.0 # Will be updated by set_project_fps
        self.project_fps = 25.0
        self.set_project_fps(project_fps)

        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.selection_regions = []
        self.dragging_clip = None
        self.dragging_linked_clip = None
        self.dragging_playhead = False
        self.creating_selection_region = False
        self.dragging_selection_region = None
        self.drag_start_pos = QPoint()
        self.drag_original_clip_states = {} # Store {'clip_id': (start_sec, track_index)}
        self.selection_drag_start_sec = 0.0
        self.drag_selection_start_values = None
        self.drag_start_state = None

        self.resizing_clip = None
        self.resize_edge = None # 'left' or 'right'
        self.resize_start_pos = QPoint()

        self.highlighted_track_info = None
        self.highlighted_ghost_track_info = None
        self.add_video_track_btn_rect = QRect()
        self.remove_video_track_btn_rect = QRect()
        self.add_audio_track_btn_rect = QRect()
        self.remove_audio_track_btn_rect = QRect()
        
        self.video_tracks_y_start = 0
        self.audio_tracks_y_start = 0
        
        self.drag_over_active = False
        self.drag_over_rect = QRectF()
        self.drag_over_audio_rect = QRectF()

    def set_project_fps(self, fps):
        self.project_fps = fps if fps > 0 else 25.0
        # Set max zoom to be 10 pixels per frame
        self.max_pixels_per_second = self.project_fps * 20
        self.pixels_per_second = min(self.pixels_per_second, self.max_pixels_per_second)
        self.update()

    def sec_to_x(self, sec): return self.HEADER_WIDTH + int(sec * self.pixels_per_second)
    def x_to_sec(self, x): return float(x - self.HEADER_WIDTH) / self.pixels_per_second if x > self.HEADER_WIDTH and self.pixels_per_second > 0 else 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#333"))

        h_offset = 0
        if self.scroll_area and self.scroll_area.horizontalScrollBar():
            h_offset = self.scroll_area.horizontalScrollBar().value()

        self.draw_headers(painter, h_offset)
        self.draw_timescale(painter)
        self.draw_tracks_and_clips(painter)
        self.draw_selections(painter)
        
        if self.drag_over_active:
            painter.setPen(QColor(0, 255, 0, 150))
            if not self.drag_over_rect.isNull():
                painter.fillRect(self.drag_over_rect, QColor(0, 255, 0, 80))
                painter.drawRect(self.drag_over_rect)
            if not self.drag_over_audio_rect.isNull():
                painter.fillRect(self.drag_over_audio_rect, QColor(0, 255, 0, 80))
                painter.drawRect(self.drag_over_audio_rect)

        self.draw_playhead(painter)

        total_width = self.sec_to_x(self.timeline.get_total_duration()) + 200
        total_height = self.calculate_total_height()
        self.setMinimumSize(max(self.parent().width(), total_width), total_height)

    def calculate_total_height(self):
        video_tracks_height = (self.timeline.num_video_tracks + 1) * self.TRACK_HEIGHT
        audio_tracks_height = (self.timeline.num_audio_tracks + 1) * self.TRACK_HEIGHT
        return self.TIMESCALE_HEIGHT + video_tracks_height + self.AUDIO_TRACKS_SEPARATOR_Y + audio_tracks_height + 20

    def draw_headers(self, painter, h_offset):
        painter.save()
        painter.setPen(QColor("#AAA"))
        header_font = QFont("Arial", 9, QFont.Weight.Bold)
        button_font = QFont("Arial", 8)

        y_cursor = self.TIMESCALE_HEIGHT
        
        rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
        painter.fillRect(rect.translated(h_offset, 0), QColor("#3a3a3a"))
        painter.drawRect(rect.translated(h_offset, 0))
        self.add_video_track_btn_rect = QRect(h_offset + rect.left() + 10, rect.top() + (rect.height() - 22)//2, self.HEADER_WIDTH - 20, 22)
        painter.setFont(button_font)
        painter.fillRect(self.add_video_track_btn_rect, QColor("#454"))
        painter.drawText(self.add_video_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "Add Track (+)")
        y_cursor += self.TRACK_HEIGHT
        self.video_tracks_y_start = y_cursor

        for i in range(self.timeline.num_video_tracks):
            track_number = self.timeline.num_video_tracks - i
            rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
            painter.fillRect(rect.translated(h_offset, 0), QColor("#444"))
            painter.drawRect(rect.translated(h_offset, 0))
            painter.setFont(header_font)
            painter.drawText(rect.translated(h_offset, 0), Qt.AlignmentFlag.AlignCenter, f"Video {track_number}")

            if track_number == self.timeline.num_video_tracks and self.timeline.num_video_tracks > 1:
                self.remove_video_track_btn_rect = QRect(h_offset + rect.right() - 25, rect.top() + 5, 20, 20)
                painter.setFont(button_font)
                painter.fillRect(self.remove_video_track_btn_rect, QColor("#833"))
                painter.drawText(self.remove_video_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "-")
            y_cursor += self.TRACK_HEIGHT

        y_cursor += self.AUDIO_TRACKS_SEPARATOR_Y

        self.audio_tracks_y_start = y_cursor
        for i in range(self.timeline.num_audio_tracks):
            track_number = i + 1
            rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
            painter.fillRect(rect.translated(h_offset, 0), QColor("#444"))
            painter.drawRect(rect.translated(h_offset, 0))
            painter.setFont(header_font)
            painter.drawText(rect.translated(h_offset, 0), Qt.AlignmentFlag.AlignCenter, f"Audio {track_number}")

            if track_number == self.timeline.num_audio_tracks and self.timeline.num_audio_tracks > 1:
                self.remove_audio_track_btn_rect = QRect(h_offset + rect.right() - 25, rect.top() + 5, 20, 20)
                painter.setFont(button_font)
                painter.fillRect(self.remove_audio_track_btn_rect, QColor("#833"))
                painter.drawText(self.remove_audio_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "-")
            y_cursor += self.TRACK_HEIGHT
        
        rect = QRect(0, y_cursor, self.HEADER_WIDTH, self.TRACK_HEIGHT)
        painter.fillRect(rect.translated(h_offset, 0), QColor("#3a3a3a"))
        painter.drawRect(rect.translated(h_offset, 0))
        self.add_audio_track_btn_rect = QRect(h_offset + rect.left() + 10, rect.top() + (rect.height() - 22)//2, self.HEADER_WIDTH - 20, 22)
        painter.setFont(button_font)
        painter.fillRect(self.add_audio_track_btn_rect, QColor("#454"))
        painter.drawText(self.add_audio_track_btn_rect, Qt.AlignmentFlag.AlignCenter, "Add Track (+)")
        
        painter.restore()
        
    def _format_timecode(self, seconds):
        if abs(seconds) < 1e-9: seconds = 0
        sign = "-" if seconds < 0 else ""
        seconds = abs(seconds)
        
        h = int(seconds / 3600)
        m = int((seconds % 3600) / 60)
        s = seconds % 60
        
        if h > 0: return f"{sign}{h}h:{m:02d}m"
        if m > 0 or seconds >= 59.99: return f"{sign}{m}m:{int(round(s)):02d}s"
        
        precision = 2 if s < 1 else 1 if s < 10 else 0
        val = f"{s:.{precision}f}"
        # Remove trailing .0 or .00
        if '.' in val: val = val.rstrip('0').rstrip('.')
        return f"{sign}{val}s"

    def draw_timescale(self, painter):
        painter.save()
        painter.setPen(QColor("#AAA"))
        painter.setFont(QFont("Arial", 8))
        font_metrics = QFontMetrics(painter.font())

        painter.fillRect(QRect(self.HEADER_WIDTH, 0, self.width() - self.HEADER_WIDTH, self.TIMESCALE_HEIGHT), QColor("#222"))
        painter.drawLine(self.HEADER_WIDTH, self.TIMESCALE_HEIGHT - 1, self.width(), self.TIMESCALE_HEIGHT - 1)

        frame_dur = 1.0 / self.project_fps
        intervals = [
            frame_dur, 2*frame_dur, 5*frame_dur, 10*frame_dur,
            0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30,
            60, 120, 300, 600, 900, 1800,
            3600, 2*3600, 5*3600, 10*3600
        ]
        
        min_pixel_dist = 70
        major_interval = next((i for i in intervals if i * self.pixels_per_second > min_pixel_dist), intervals[-1])

        minor_interval = 0
        for divisor in [5, 4, 2]:
            if (major_interval / divisor) * self.pixels_per_second > 10:
                minor_interval = major_interval / divisor
                break
        
        start_sec = self.x_to_sec(self.HEADER_WIDTH)
        end_sec = self.x_to_sec(self.width())

        def draw_ticks(interval, height):
            if interval <= 1e-6: return
            start_tick_num = int(start_sec / interval)
            end_tick_num = int(end_sec / interval) + 1
            for i in range(start_tick_num, end_tick_num + 1):
                t_sec = i * interval
                x = self.sec_to_x(t_sec)
                if x > self.width(): break
                if x >= self.HEADER_WIDTH:
                    painter.drawLine(x, self.TIMESCALE_HEIGHT - height, x, self.TIMESCALE_HEIGHT)
        
        if frame_dur * self.pixels_per_second > 4:
            draw_ticks(frame_dur, 3)
        if minor_interval > 0:
            draw_ticks(minor_interval, 6)

        start_major_tick = int(start_sec / major_interval)
        end_major_tick = int(end_sec / major_interval) + 1
        for i in range(start_major_tick, end_major_tick + 1):
            t_sec = i * major_interval
            x = self.sec_to_x(t_sec)
            if x > self.width() + 50: break
            if x >= self.HEADER_WIDTH - 50:
                painter.drawLine(x, self.TIMESCALE_HEIGHT - 12, x, self.TIMESCALE_HEIGHT)
                label = self._format_timecode(t_sec)
                label_width = font_metrics.horizontalAdvance(label)
                painter.drawText(x - label_width // 2, self.TIMESCALE_HEIGHT - 14, label)

        painter.restore()

    def get_clip_rect(self, clip):
        if clip.track_type == 'video':
            visual_index = self.timeline.num_video_tracks - clip.track_index
            y = self.video_tracks_y_start + visual_index * self.TRACK_HEIGHT
        else:
            visual_index = clip.track_index - 1
            y = self.audio_tracks_y_start + visual_index * self.TRACK_HEIGHT
        
        x = self.sec_to_x(clip.timeline_start_sec)
        w = int(clip.duration_sec * self.pixels_per_second)
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
                highlight_rect = QRect(self.HEADER_WIDTH, int(y), self.width() - self.HEADER_WIDTH, self.TRACK_HEIGHT)
                painter.fillRect(highlight_rect, QColor(255, 255, 0, 40))

        if self.highlighted_ghost_track_info:
            track_type, track_index = self.highlighted_ghost_track_info
            y = -1
            if track_type == 'video':
                y = self.TIMESCALE_HEIGHT
            elif track_type == 'audio':
                y = self.audio_tracks_y_start + self.timeline.num_audio_tracks * self.TRACK_HEIGHT

            if y != -1:
                highlight_rect = QRect(self.HEADER_WIDTH, int(y), self.width() - self.HEADER_WIDTH, self.TRACK_HEIGHT)
                painter.fillRect(highlight_rect, QColor(255, 255, 0, 40))

        for clip in self.timeline.clips:
            clip_rect = self.get_clip_rect(clip)
            base_color = QColor("#46A") # Default video
            if clip.media_type == 'image':
                base_color = QColor("#4A6") # Greenish for images
            elif clip.track_type == 'audio':
                base_color = QColor("#48C") # Bluish for audio
            
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
            w = int((end_sec - start_sec) * self.pixels_per_second)
            selection_rect = QRectF(x, self.TIMESCALE_HEIGHT, w, self.height() - self.TIMESCALE_HEIGHT)
            painter.fillRect(selection_rect, QColor(100, 100, 255, 80))
            painter.setPen(QColor(150, 150, 255, 150))
            painter.drawRect(selection_rect)

    def draw_playhead(self, painter):
        playhead_x = self.sec_to_x(self.playhead_pos_sec)
        painter.setPen(QPen(QColor("red"), 2))
        painter.drawLine(playhead_x, 0, playhead_x, self.height())

    def y_to_track_info(self, y):
        # Check for ghost video track
        if self.TIMESCALE_HEIGHT <= y < self.video_tracks_y_start:
            return ('video', self.timeline.num_video_tracks + 1)

        # Check for existing video tracks
        video_tracks_end_y = self.video_tracks_y_start + self.timeline.num_video_tracks * self.TRACK_HEIGHT
        if self.video_tracks_y_start <= y < video_tracks_end_y:
            visual_index = (y - self.video_tracks_y_start) // self.TRACK_HEIGHT
            track_index = self.timeline.num_video_tracks - visual_index
            return ('video', track_index)
        
        # Check for existing audio tracks
        audio_tracks_end_y = self.audio_tracks_y_start + self.timeline.num_audio_tracks * self.TRACK_HEIGHT
        if self.audio_tracks_y_start <= y < audio_tracks_end_y:
            visual_index = (y - self.audio_tracks_y_start) // self.TRACK_HEIGHT
            track_index = visual_index + 1
            return ('audio', track_index)
            
        # Check for ghost audio track
        add_audio_btn_y_start = self.audio_tracks_y_start + self.timeline.num_audio_tracks * self.TRACK_HEIGHT
        add_audio_btn_y_end = add_audio_btn_y_start + self.TRACK_HEIGHT
        if add_audio_btn_y_start <= y < add_audio_btn_y_end:
            return ('audio', self.timeline.num_audio_tracks + 1)
            
        return None

    def get_region_at_pos(self, pos: QPoint):
        if pos.y() <= self.TIMESCALE_HEIGHT or pos.x() <= self.HEADER_WIDTH:
            return None
        
        clicked_sec = self.x_to_sec(pos.x())
        for region in reversed(self.selection_regions):
            if region[0] <= clicked_sec <= region[1]:
                return region
        return None

    def wheelEvent(self, event: QMouseEvent):
        if not self.scroll_area or event.position().x() < self.HEADER_WIDTH:
            event.ignore()
            return

        scrollbar = self.scroll_area.horizontalScrollBar()
        mouse_x_abs = event.position().x()
        
        time_at_cursor = self.x_to_sec(mouse_x_abs)

        delta = event.angleDelta().y()
        zoom_factor = 1.15
        old_pps = self.pixels_per_second

        if delta > 0: new_pps = old_pps * zoom_factor
        else: new_pps = old_pps / zoom_factor
        
        min_pps = 1 / (3600 * 10) # Min zoom of 1px per 10 hours
        new_pps = max(min_pps, min(new_pps, self.max_pixels_per_second))
        
        if abs(new_pps - old_pps) < 1e-9:
            return
            
        self.pixels_per_second = new_pps
        # This will trigger a paintEvent, which resizes the widget.
        # The scroll area will update its scrollbar ranges in response.
        self.update() 
        
        # The new absolute x-coordinate for the time under the cursor
        new_mouse_x_abs = self.sec_to_x(time_at_cursor)
        
        # The amount the content "shifted" at the cursor's location
        shift_amount = new_mouse_x_abs - mouse_x_abs
        
        # Adjust the scrollbar by this shift amount to keep the content under the cursor
        new_scroll_value = scrollbar.value() + shift_amount
        scrollbar.setValue(int(new_scroll_value))
        
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.pos().x() < self.HEADER_WIDTH + self.scroll_area.horizontalScrollBar().value():
            if self.add_video_track_btn_rect.contains(event.pos()): self.add_track.emit('video')
            elif self.remove_video_track_btn_rect.contains(event.pos()): self.remove_track.emit('video')
            elif self.add_audio_track_btn_rect.contains(event.pos()): self.add_track.emit('audio')
            elif self.remove_audio_track_btn_rect.contains(event.pos()): self.remove_track.emit('audio')
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging_clip = None
            self.dragging_linked_clip = None
            self.dragging_playhead = False
            self.dragging_selection_region = None
            self.creating_selection_region = False
            self.resizing_clip = None
            self.resize_edge = None
            self.drag_original_clip_states.clear()

            # Check for resize handles first
            for clip in reversed(self.timeline.clips):
                clip_rect = self.get_clip_rect(clip)
                if abs(event.pos().x() - clip_rect.left()) < self.RESIZE_HANDLE_WIDTH and clip_rect.contains(QPointF(clip_rect.left(), event.pos().y())):
                    self.resizing_clip = clip
                    self.resize_edge = 'left'
                    break
                elif abs(event.pos().x() - clip_rect.right()) < self.RESIZE_HANDLE_WIDTH and clip_rect.contains(QPointF(clip_rect.right(), event.pos().y())):
                    self.resizing_clip = clip
                    self.resize_edge = 'right'
                    break
            
            if self.resizing_clip:
                self.drag_start_state = self.window()._get_current_timeline_state()
                self.resize_start_pos = event.pos()
                self.update()
                return

            for clip in reversed(self.timeline.clips):
                clip_rect = self.get_clip_rect(clip)
                if clip_rect.contains(QPointF(event.pos())):
                    self.dragging_clip = clip
                    self.drag_start_state = self.window()._get_current_timeline_state()
                    self.drag_original_clip_states[clip.id] = (clip.timeline_start_sec, clip.track_index)
                    
                    self.dragging_linked_clip = next((c for c in self.timeline.clips if c.group_id == clip.group_id and c.id != clip.id), None)
                    if self.dragging_linked_clip:
                        self.drag_original_clip_states[self.dragging_linked_clip.id] = \
                            (self.dragging_linked_clip.timeline_start_sec, self.dragging_linked_clip.track_index)

                    self.drag_start_pos = event.pos()
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
                        playhead_x = self.sec_to_x(self.playhead_pos_sec)
                        if abs(event.pos().x() - playhead_x) < self.SNAP_THRESHOLD_PIXELS:
                            self.selection_drag_start_sec = self.playhead_pos_sec
                        else:
                            self.selection_drag_start_sec = self.x_to_sec(event.pos().x())
                        self.selection_regions.append([self.selection_drag_start_sec, self.selection_drag_start_sec])
                    elif is_on_timescale:
                        self.playhead_pos_sec = max(0, self.x_to_sec(event.pos().x()))
                        self.playhead_moved.emit(self.playhead_pos_sec)
                        self.dragging_playhead = True
            
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.resizing_clip:
            delta_x = event.pos().x() - self.resize_start_pos.x()
            time_delta = delta_x / self.pixels_per_second
            min_duration = 1.0 / self.project_fps
            playhead_time = self.playhead_pos_sec
            snap_time_delta = self.SNAP_THRESHOLD_PIXELS / self.pixels_per_second

            media_props = self.window().media_properties.get(self.resizing_clip.source_path)
            source_duration = media_props['duration'] if media_props else float('inf')

            if self.resize_edge == 'left':
                original_start = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].timeline_start_sec
                original_duration = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].duration_sec
                original_clip_start = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].clip_start_sec
                true_new_start_sec = original_start + time_delta
                if abs(true_new_start_sec - playhead_time) < snap_time_delta:
                    new_start_sec = playhead_time
                else:
                    new_start_sec = true_new_start_sec

                if new_start_sec > original_start + original_duration - min_duration:
                    new_start_sec = original_start + original_duration - min_duration

                new_start_sec = max(0, new_start_sec)

                if self.resizing_clip.media_type != 'image':
                    if new_start_sec < original_start - original_clip_start:
                         new_start_sec = original_start - original_clip_start

                new_duration = (original_start + original_duration) - new_start_sec
                new_clip_start = original_clip_start + (new_start_sec - original_start)
                
                if new_duration < min_duration:
                    new_duration = min_duration
                    new_start_sec = (original_start + original_duration) - new_duration
                    new_clip_start = original_clip_start + (new_start_sec - original_start)

                self.resizing_clip.timeline_start_sec = new_start_sec
                self.resizing_clip.duration_sec = new_duration
                self.resizing_clip.clip_start_sec = new_clip_start

            elif self.resize_edge == 'right':
                original_start = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].timeline_start_sec
                original_duration = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].duration_sec
                
                true_new_duration = original_duration + time_delta
                true_new_end_time = original_start + true_new_duration

                if abs(true_new_end_time - playhead_time) < snap_time_delta:
                    new_duration = playhead_time - original_start
                else:
                    new_duration = true_new_duration
                
                if new_duration < min_duration:
                    new_duration = min_duration

                if self.resizing_clip.media_type != 'image':
                    if self.resizing_clip.clip_start_sec + new_duration > source_duration:
                        new_duration = source_duration - self.resizing_clip.clip_start_sec
                
                self.resizing_clip.duration_sec = new_duration

            self.update()
            return

        if not self.dragging_clip and not self.dragging_playhead and not self.creating_selection_region:
            cursor_set = False
            ### ADD BEGIN ###
            # Check for playhead proximity for selection snapping
            playhead_x = self.sec_to_x(self.playhead_pos_sec)
            is_in_track_area = event.pos().y() > self.TIMESCALE_HEIGHT and event.pos().x() > self.HEADER_WIDTH
            if is_in_track_area and abs(event.pos().x() - playhead_x) < self.SNAP_THRESHOLD_PIXELS:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                cursor_set = True
            
            if not cursor_set:
            ### ADD END ###
                for clip in self.timeline.clips:
                    clip_rect = self.get_clip_rect(clip)
                    if (abs(event.pos().x() - clip_rect.left()) < self.RESIZE_HANDLE_WIDTH and clip_rect.contains(QPointF(clip_rect.left(), event.pos().y()))) or \
                       (abs(event.pos().x() - clip_rect.right()) < self.RESIZE_HANDLE_WIDTH and clip_rect.contains(QPointF(clip_rect.right(), event.pos().y()))):
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                        cursor_set = True
                        break
            if not cursor_set:
                self.unsetCursor()

        if self.creating_selection_region:
            current_sec = self.x_to_sec(event.pos().x())
            start = min(self.selection_drag_start_sec, current_sec)
            end = max(self.selection_drag_start_sec, current_sec)
            self.selection_regions[-1] = [start, end]
            self.update()
            return
        
        if self.dragging_selection_region:
            delta_x = event.pos().x() - self.drag_start_pos.x()
            time_delta = delta_x / self.pixels_per_second
            
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
            self.highlighted_ghost_track_info = None
            new_track_info = self.y_to_track_info(event.pos().y())
            
            original_start_sec, _ = self.drag_original_clip_states[self.dragging_clip.id]

            if new_track_info:
                new_track_type, new_track_index = new_track_info

                is_ghost_track = (new_track_type == 'video' and new_track_index > self.timeline.num_video_tracks) or \
                                 (new_track_type == 'audio' and new_track_index > self.timeline.num_audio_tracks)
                
                if is_ghost_track:
                    self.highlighted_ghost_track_info = new_track_info
                else:
                    self.highlighted_track_info = new_track_info

                if new_track_type == self.dragging_clip.track_type:
                    self.dragging_clip.track_index = new_track_index

            delta_x = event.pos().x() - self.drag_start_pos.x()
            time_delta = delta_x / self.pixels_per_second
            true_new_start_time = original_start_sec + time_delta

            playhead_time = self.playhead_pos_sec
            snap_time_delta = self.SNAP_THRESHOLD_PIXELS / self.pixels_per_second
            
            new_start_time = true_new_start_time
            true_new_end_time = true_new_start_time + self.dragging_clip.duration_sec
            
            if abs(true_new_start_time - playhead_time) < snap_time_delta:
                new_start_time = playhead_time
            elif abs(true_new_end_time - playhead_time) < snap_time_delta:
                new_start_time = playhead_time - self.dragging_clip.duration_sec

            for other_clip in self.timeline.clips:
                if other_clip.id == self.dragging_clip.id: continue
                if self.dragging_linked_clip and other_clip.id == self.dragging_linked_clip.id: continue
                if (other_clip.track_type != self.dragging_clip.track_type or 
                    other_clip.track_index != self.dragging_clip.track_index):
                    continue

                is_overlapping = (new_start_time < other_clip.timeline_end_sec and
                                  new_start_time + self.dragging_clip.duration_sec > other_clip.timeline_start_sec)
                
                if is_overlapping:
                    movement_direction = true_new_start_time - original_start_sec
                    if movement_direction > 0:
                        new_start_time = other_clip.timeline_start_sec - self.dragging_clip.duration_sec
                    else:
                        new_start_time = other_clip.timeline_end_sec
                    break 

            final_start_time = max(0, new_start_time)
            self.dragging_clip.timeline_start_sec = final_start_time
            if self.dragging_linked_clip:
                self.dragging_linked_clip.timeline_start_sec = final_start_time
            
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.resizing_clip:
                new_state = self.window()._get_current_timeline_state()
                command = TimelineStateChangeCommand("Resize Clip", self.timeline, *self.drag_start_state, *new_state)
                command.undo()
                self.window().undo_stack.push(command)
                self.resizing_clip = None
                self.resize_edge = None
                self.drag_start_state = None
                self.update()
                return

            if self.creating_selection_region:
                self.creating_selection_region = False
                if self.selection_regions:
                    start, end = self.selection_regions[-1]
                    if (end - start) * self.pixels_per_second < 2:
                        self.clear_all_regions()
            
            if self.dragging_selection_region:
                self.dragging_selection_region = None
                self.drag_selection_start_values = None

            self.dragging_playhead = False
            if self.dragging_clip:
                orig_start, orig_track = self.drag_original_clip_states[self.dragging_clip.id]
                moved = (orig_start != self.dragging_clip.timeline_start_sec or 
                         orig_track != self.dragging_clip.track_index)

                if self.dragging_linked_clip:
                    orig_start_link, orig_track_link = self.drag_original_clip_states[self.dragging_linked_clip.id]
                    moved = moved or (orig_start_link != self.dragging_linked_clip.timeline_start_sec or 
                                      orig_track_link != self.dragging_linked_clip.track_index)
                
                if moved:
                    self.window().finalize_clip_drag(self.drag_start_state)
                
                self.timeline.clips.sort(key=lambda c: c.timeline_start_sec)
                self.highlighted_track_info = None
                self.highlighted_ghost_track_info = None
                self.operation_finished.emit()

            self.dragging_clip = None
            self.dragging_linked_clip = None
            self.drag_original_clip_states.clear()
            self.drag_start_state = None
            
            self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-vnd.video.filepath'):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drag_over_active = False
        self.highlighted_ghost_track_info = None
        self.highlighted_track_info = None
        self.update()

    def dragMoveEvent(self, event):
        mime_data = event.mimeData()
        if not mime_data.hasFormat('application/x-vnd.video.filepath'):
            event.ignore()
            return
        
        event.acceptProposedAction()
        
        json_data = json.loads(mime_data.data('application/x-vnd.video.filepath').data().decode('utf-8'))
        duration = json_data['duration']
        media_type = json_data['media_type']
        has_audio = json_data['has_audio']

        pos = event.position()
        start_sec = self.x_to_sec(pos.x())
        track_info = self.y_to_track_info(pos.y())

        self.drag_over_rect = QRectF()
        self.drag_over_audio_rect = QRectF()
        self.drag_over_active = False
        self.highlighted_ghost_track_info = None
        self.highlighted_track_info = None

        if track_info:
            self.drag_over_active = True
            track_type, track_index = track_info
            
            is_ghost_track = (track_type == 'video' and track_index > self.timeline.num_video_tracks) or \
                             (track_type == 'audio' and track_index > self.timeline.num_audio_tracks)
            if is_ghost_track:
                self.highlighted_ghost_track_info = track_info
            else:
                self.highlighted_track_info = track_info
            
            width = int(duration * self.pixels_per_second)
            x = self.sec_to_x(start_sec)
            
            video_y, audio_y = -1, -1

            if media_type in ['video', 'image']:
                if track_type == 'video':
                    visual_index = self.timeline.num_video_tracks - track_index
                    video_y = self.video_tracks_y_start + visual_index * self.TRACK_HEIGHT
                    if has_audio:
                        audio_y = self.audio_tracks_y_start
                elif track_type == 'audio' and has_audio:
                    visual_index = track_index - 1
                    audio_y = self.audio_tracks_y_start + visual_index * self.TRACK_HEIGHT
                    video_y = self.video_tracks_y_start + (self.timeline.num_video_tracks - 1) * self.TRACK_HEIGHT
            
            elif media_type == 'audio':
                if track_type == 'audio':
                    visual_index = track_index - 1
                    audio_y = self.audio_tracks_y_start + visual_index * self.TRACK_HEIGHT

            if video_y != -1:
                self.drag_over_rect = QRectF(x, video_y, width, self.TRACK_HEIGHT)
            if audio_y != -1:
                self.drag_over_audio_rect = QRectF(x, audio_y, width, self.TRACK_HEIGHT)
        
        self.update()

    def dropEvent(self, event):
        self.drag_over_active = False
        self.highlighted_ghost_track_info = None
        self.highlighted_track_info = None
        self.update()
        
        mime_data = event.mimeData()
        if not mime_data.hasFormat('application/x-vnd.video.filepath'):
            return

        json_data = json.loads(mime_data.data('application/x-vnd.video.filepath').data().decode('utf-8'))
        file_path = json_data['path']
        duration = json_data['duration']
        has_audio = json_data['has_audio']
        media_type = json_data['media_type']

        pos = event.position()
        start_sec = self.x_to_sec(pos.x())
        track_info = self.y_to_track_info(pos.y())

        if not track_info:
            return

        drop_track_type, drop_track_index = track_info
        video_track_idx = None
        audio_track_idx = None

        if media_type == 'image':
            if drop_track_type == 'video':
                video_track_idx = drop_track_index
        elif media_type == 'audio':
            if drop_track_type == 'audio':
                audio_track_idx = drop_track_index
        elif media_type == 'video':
            if drop_track_type == 'video':
                video_track_idx = drop_track_index
                if has_audio: audio_track_idx = 1
            elif drop_track_type == 'audio' and has_audio:
                audio_track_idx = drop_track_index
                video_track_idx = 1
        
        if video_track_idx is None and audio_track_idx is None:
            return

        main_window = self.window()
        main_window._add_clip_to_timeline(
            source_path=file_path,
            timeline_start_sec=start_sec,
            duration_sec=duration,
            media_type=media_type,
            clip_start_sec=0,
            video_track_index=video_track_idx,
            audio_track_index=audio_track_idx
        )

    def contextMenuEvent(self, event: 'QContextMenuEvent'):
        menu = QMenu(self)
        
        region_at_pos = self.get_region_at_pos(event.pos())
        if region_at_pos:
            split_this_action = menu.addAction("Split This Region")
            split_all_action = menu.addAction("Split All Regions")
            join_this_action = menu.addAction("Join This Region")
            join_all_action = menu.addAction("Join All Regions")
            delete_this_action = menu.addAction("Delete This Region")
            delete_all_action = menu.addAction("Delete All Regions")
            menu.addSeparator()
            clear_this_action = menu.addAction("Clear This Region")
            clear_all_action = menu.addAction("Clear All Regions")
            split_this_action.triggered.connect(lambda: self.split_region_requested.emit(region_at_pos))
            split_all_action.triggered.connect(lambda: self.split_all_regions_requested.emit(self.selection_regions))
            join_this_action.triggered.connect(lambda: self.join_region_requested.emit(region_at_pos))
            join_all_action.triggered.connect(lambda: self.join_all_regions_requested.emit(self.selection_regions))
            delete_this_action.triggered.connect(lambda: self.delete_region_requested.emit(region_at_pos))
            delete_all_action.triggered.connect(lambda: self.delete_all_regions_requested.emit(self.selection_regions))
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
            delete_action = menu.addAction("Delete Clip")
            playhead_time = self.playhead_pos_sec
            is_playhead_over_clip = (clip_at_pos.timeline_start_sec < playhead_time < clip_at_pos.timeline_end_sec)
            split_action.setEnabled(is_playhead_over_clip)
            split_action.triggered.connect(lambda: self.split_requested.emit(clip_at_pos))
            delete_action.triggered.connect(lambda: self.delete_clip_requested.emit(clip_at_pos))

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
        self.confirm_on_exit_checkbox = QCheckBox("Confirm before exiting")
        self.confirm_on_exit_checkbox.setChecked(parent_settings.get("confirm_on_exit", True))
        layout.addWidget(self.confirm_on_exit_checkbox)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_settings(self):
        return {
            "confirm_on_exit": self.confirm_on_exit_checkbox.isChecked()
        }

class MediaListWidget(QListWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setDragEnabled(True)
        self.setAcceptDrops(False)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def startDrag(self, supportedActions):
        drag = QDrag(self)
        mime_data = QMimeData()
        
        item = self.currentItem()
        if not item: return

        path = item.data(Qt.ItemDataRole.UserRole)
        media_info = self.main_window.media_properties.get(path)
        if not media_info: return

        payload = {
            "path": path,
            "duration": media_info['duration'],
            "has_audio": media_info['has_audio'],
            "media_type": media_info['media_type']
        }
        
        mime_data.setData('application/x-vnd.video.filepath', QByteArray(json.dumps(payload).encode('utf-8')))
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)

class ProjectMediaWidget(QWidget):
    media_removed = pyqtSignal(str)
    add_media_requested = pyqtSignal()
    add_to_timeline_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        self.media_list = MediaListWidget(self.main_window, self)
        self.media_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.media_list.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.media_list)
        
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add")
        remove_button = QPushButton("Remove")
        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)
        layout.addLayout(button_layout)
        
        add_button.clicked.connect(self.add_media_requested.emit)
        remove_button.clicked.connect(self.remove_selected_media)
    def show_context_menu(self, pos):
        item = self.media_list.itemAt(pos)
        if not item:
            return

        menu = QMenu()
        add_action = menu.addAction("Add to timeline at playhead")
        
        action = menu.exec(self.media_list.mapToGlobal(pos))

        if action == add_action:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            self.add_to_timeline_requested.emit(file_path)

    def add_media_item(self, file_path):
        if not any(self.media_list.item(i).data(Qt.ItemDataRole.UserRole) == file_path for i in range(self.media_list.count())):
            item = QListWidgetItem(os.path.basename(file_path))
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            self.media_list.addItem(item)

    def remove_selected_media(self):
        selected_items = self.media_list.selectedItems()
        if not selected_items: return

        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            self.media_removed.emit(file_path)
            self.media_list.takeItem(self.media_list.row(item))

    def clear_list(self):
        self.media_list.clear()

class MainWindow(QMainWindow):
    def __init__(self, project_to_load=None):
        super().__init__()
        self.setWindowTitle("Inline AI Video Editor")
        self.setGeometry(100, 100, 1200, 800)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks)

        self.timeline = Timeline()
        self.undo_stack = UndoStack()
        self.media_pool = []
        self.media_properties = {}
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

        self._setup_ui()
        self._connect_signals()

        self.plugin_manager.load_enabled_plugins_from_settings(self.settings.get("enabled_plugins", []))
        self._apply_loaded_settings()
        self.seek_preview(0)
        
        if not self.settings_file_was_loaded: self._save_settings()
        if project_to_load: QTimer.singleShot(100, lambda: self._load_project_from_path(project_to_load))

    def _get_current_timeline_state(self):
        return (
            copy.deepcopy(self.timeline.clips),
            self.timeline.num_video_tracks,
            self.timeline.num_audio_tracks
        )

    def _setup_ui(self):
        self.media_dock = QDockWidget("Project Media", self)
        self.project_media_widget = ProjectMediaWidget(self)
        self.media_dock.setWidget(self.project_media_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.media_dock)

        self.splitter = QSplitter(Qt.Orientation.Vertical)

        self.preview_widget = QLabel()
        self.preview_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_widget.setMinimumSize(640, 360)
        self.preview_widget.setFrameShape(QFrame.Shape.Box)
        self.preview_widget.setStyleSheet("background-color: black; color: white;")
        self.splitter.addWidget(self.preview_widget)

        self.timeline_widget = TimelineWidget(self.timeline, self.settings, self.project_fps, self)
        self.timeline_scroll_area = QScrollArea()
        self.timeline_widget.scroll_area = self.timeline_scroll_area
        self.timeline_scroll_area.setWidgetResizable(False)
        self.timeline_widget.setMinimumWidth(2000)
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
            'timeline': {'widget': self.timeline_scroll_area, 'name': 'Timeline', 'action': None},
            'project_media': {'widget': self.media_dock, 'name': 'Project Media', 'action': None}
        }
        self.plugin_menu_actions = {}
        self.windows_menu = None
        self._create_menu_bar()

        self.splitter_save_timer = QTimer(self)
        self.splitter_save_timer.setSingleShot(True)
        self.splitter_save_timer.timeout.connect(self._save_settings)

    def _connect_signals(self):
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        self.timeline_widget.split_requested.connect(self.split_clip_at_playhead)
        self.timeline_widget.delete_clip_requested.connect(self.delete_clip)
        self.timeline_widget.playhead_moved.connect(self.seek_preview)
        self.timeline_widget.split_region_requested.connect(self.on_split_region)
        self.timeline_widget.split_all_regions_requested.connect(self.on_split_all_regions)
        self.timeline_widget.join_region_requested.connect(self.on_join_region)
        self.timeline_widget.join_all_regions_requested.connect(self.on_join_all_regions)
        self.timeline_widget.delete_region_requested.connect(self.on_delete_region)
        self.timeline_widget.delete_all_regions_requested.connect(self.on_delete_all_regions)
        self.timeline_widget.add_track.connect(self.add_track)
        self.timeline_widget.remove_track.connect(self.remove_track)
        self.timeline_widget.operation_finished.connect(self.prune_empty_tracks)

        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.stop_button.clicked.connect(self.stop_playback)
        self.frame_back_button.clicked.connect(lambda: self.step_frame(-1))
        self.frame_forward_button.clicked.connect(lambda: self.step_frame(1))
        self.playback_timer.timeout.connect(self.advance_playback_frame)
        
        self.project_media_widget.add_media_requested.connect(self.add_media_files)
        self.project_media_widget.media_removed.connect(self.on_media_removed_from_pool)
        self.project_media_widget.add_to_timeline_requested.connect(self.on_add_to_timeline_at_playhead)
        
        self.undo_stack.history_changed.connect(self.update_undo_redo_actions)
        self.undo_stack.timeline_changed.connect(self.on_timeline_changed_by_undo)

    def on_timeline_changed_by_undo(self):
        self.prune_empty_tracks()
        self.timeline_widget.update()
        self.status_label.setText("Operation undone/redone.")

    def update_undo_redo_actions(self):
        self.undo_action.setEnabled(self.undo_stack.can_undo())
        self.undo_action.setText(f"Undo {self.undo_stack.undo_text()}" if self.undo_stack.can_undo() else "Undo")
        
        self.redo_action.setEnabled(self.undo_stack.can_redo())
        self.redo_action.setText(f"Redo {self.undo_stack.redo_text()}" if self.undo_stack.can_redo() else "Redo")

    def finalize_clip_drag(self, old_state_tuple):
        current_clips, _, _ = self._get_current_timeline_state()
        
        max_v_idx = max([c.track_index for c in current_clips if c.track_type == 'video'] + [1])
        max_a_idx = max([c.track_index for c in current_clips if c.track_type == 'audio'] + [1])

        if max_v_idx > self.timeline.num_video_tracks:
            self.timeline.num_video_tracks = max_v_idx
        
        if max_a_idx > self.timeline.num_audio_tracks:
            self.timeline.num_audio_tracks = max_a_idx
            
        new_state_tuple = self._get_current_timeline_state()
        
        command = TimelineStateChangeCommand("Move Clip", self.timeline, *old_state_tuple, *new_state_tuple)
        command.undo()
        self.undo_stack.push(command)

    def on_add_to_timeline_at_playhead(self, file_path):
        media_info = self.media_properties.get(file_path)
        if not media_info:
            self.status_label.setText(f"Error: Could not find properties for {os.path.basename(file_path)}")
            return

        playhead_pos = self.timeline_widget.playhead_pos_sec
        duration = media_info['duration']
        has_audio = media_info['has_audio']
        media_type = media_info['media_type']

        video_track = 1 if media_type in ['video', 'image'] else None
        audio_track = 1 if has_audio else None

        self._add_clip_to_timeline(
            source_path=file_path,
            timeline_start_sec=playhead_pos,
            duration_sec=duration,
            media_type=media_type,
            clip_start_sec=0.0,
            video_track_index=video_track,
            audio_track_index=audio_track
        )

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
        old_state = self._get_current_timeline_state()
        if track_type == 'video':
            self.timeline.num_video_tracks += 1
        elif track_type == 'audio':
            self.timeline.num_audio_tracks += 1
        new_state = self._get_current_timeline_state()
        
        command = TimelineStateChangeCommand(f"Add {track_type.capitalize()} Track", self.timeline, *old_state, *new_state)
        self.undo_stack.push(command)
        command.undo()
        self.undo_stack.push(command)
    
    def remove_track(self, track_type):
        old_state = self._get_current_timeline_state()
        if track_type == 'video' and self.timeline.num_video_tracks > 1:
            self.timeline.num_video_tracks -= 1
        elif track_type == 'audio' and self.timeline.num_audio_tracks > 1:
            self.timeline.num_audio_tracks -= 1
        else:
            return
        new_state = self._get_current_timeline_state()
        
        command = TimelineStateChangeCommand(f"Remove {track_type.capitalize()} Track", self.timeline, *old_state, *new_state)
        command.undo()
        self.undo_stack.push(command)


    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        new_action = QAction("&New Project", self); new_action.triggered.connect(self.new_project)
        open_action = QAction("&Open Project...", self); open_action.triggered.connect(self.open_project)
        self.recent_menu = file_menu.addMenu("Recent")
        save_action = QAction("&Save Project As...", self); save_action.triggered.connect(self.save_project_as)
        add_media_action = QAction("&Add Media to Project...", self); add_media_action.triggered.connect(self.add_media_files)
        export_action = QAction("&Export Video...", self); export_action.triggered.connect(self.export_video)
        settings_action = QAction("Se&ttings...", self); settings_action.triggered.connect(self.open_settings_dialog)
        exit_action = QAction("E&xit", self); exit_action.triggered.connect(self.close)
        file_menu.addAction(new_action); file_menu.addAction(open_action); file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addSeparator(); file_menu.addAction(add_media_action); file_menu.addAction(export_action)
        file_menu.addSeparator(); file_menu.addAction(settings_action); file_menu.addSeparator(); file_menu.addAction(exit_action)
        self._update_recent_files_menu()

        edit_menu = menu_bar.addMenu("&Edit")
        self.undo_action = QAction("Undo", self); self.undo_action.setShortcut("Ctrl+Z"); self.undo_action.triggered.connect(self.undo_stack.undo)
        self.redo_action = QAction("Redo", self); self.redo_action.setShortcut("Ctrl+Y"); self.redo_action.triggered.connect(self.undo_stack.redo)
        edit_menu.addAction(self.undo_action); edit_menu.addAction(self.redo_action); edit_menu.addSeparator()

        split_action = QAction("Split Clip at Playhead", self); split_action.triggered.connect(self.split_clip_at_playhead)
        edit_menu.addAction(split_action)
        self.update_undo_redo_actions()
        
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
            if data['widget'] is self.preview_widget: continue
            action = QAction(data['name'], self, checkable=True)
            if hasattr(data['widget'], 'visibilityChanged'):
                action.toggled.connect(data['widget'].setVisible)
                data['widget'].visibilityChanged.connect(action.setChecked)
            else:
                action.toggled.connect(lambda checked, k=key: self.toggle_widget_visibility(k, checked))

            data['action'] = action
            self.windows_menu.addAction(action)

    def _start_playback_stream_at(self, time_sec):
        self._stop_playback_stream()
        clip = next((c for c in self.timeline.clips if c.track_type == 'video' and c.timeline_start_sec <= time_sec < c.timeline_end_sec), None)
        if not clip: return
        self.playback_clip = clip
        clip_time = time_sec - clip.timeline_start_sec + clip.clip_start_sec
        w, h = self.project_width, self.project_height
        try:
            args = (ffmpeg.input(self.playback_clip.source_path, ss=clip_time)
                    .filter('scale', w, h, force_original_aspect_ratio='decrease')
                    .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=self.project_fps).compile())
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
                    if den > 0:
                        self.project_fps = num / den
                        self.timeline_widget.set_project_fps(self.project_fps)
                print(f"Project properties set: {self.project_width}x{self.project_height} @ {self.project_fps:.2f} FPS")
                return True
        except Exception as e: print(f"Could not probe for project properties: {e}")
        return False

    def get_frame_data_at_time(self, time_sec):
        clip_at_time = next((c for c in self.timeline.clips if c.track_type == 'video' and c.timeline_start_sec <= time_sec < c.timeline_end_sec), None)
        if not clip_at_time:
            return (None, 0, 0)
        try:
            w, h = self.project_width, self.project_height
            if clip_at_time.media_type == 'image':
                out, _ = (
                    ffmpeg
                    .input(clip_at_time.source_path)
                    .filter('scale', w, h, force_original_aspect_ratio='decrease')
                    .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
                    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
            else:
                clip_time = time_sec - clip_at_time.timeline_start_sec + clip_at_time.clip_start_sec
                out, _ = (
                    ffmpeg
                    .input(clip_at_time.source_path, ss=clip_time)
                    .filter('scale', w, h, force_original_aspect_ratio='decrease')
                    .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
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
            w, h = self.project_width, self.project_height
            if clip_at_time.media_type == 'image':
                out, _ = (
                    ffmpeg.input(clip_at_time.source_path)
                    .filter('scale', w, h, force_original_aspect_ratio='decrease')
                    .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
                    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
            else:
                clip_time = time_sec - clip_at_time.timeline_start_sec + clip_at_time.clip_start_sec
                out, _ = (ffmpeg.input(clip_at_time.source_path, ss=clip_time)
                          .filter('scale', w, h, force_original_aspect_ratio='decrease')
                          .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
                          .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                          .run(capture_stdout=True, quiet=True))
            
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
        
        self.timeline_widget.playhead_pos_sec = new_time
        self.timeline_widget.update()
        
        clip_at_new_time = next((c for c in self.timeline.clips if c.track_type == 'video' and c.timeline_start_sec <= new_time < c.timeline_end_sec), None)
        
        if not clip_at_new_time:
            self._stop_playback_stream()
            black_pixmap = QPixmap(self.project_width, self.project_height); black_pixmap.fill(QColor("black"))
            scaled_pixmap = black_pixmap.scaled(self.preview_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_widget.setPixmap(scaled_pixmap)
            return

        if clip_at_new_time.media_type == 'image':
            if self.playback_clip is None or self.playback_clip.id != clip_at_new_time.id:
                self._stop_playback_stream()
                self.playback_clip = clip_at_new_time
                frame_pixmap = self.get_frame_at_time(new_time)
                if frame_pixmap:
                    scaled_pixmap = frame_pixmap.scaled(self.preview_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.preview_widget.setPixmap(scaled_pixmap)
            return

        if self.playback_clip is None or self.playback_clip.id != clip_at_new_time.id:
            self._start_playback_stream_at(new_time)
        
        if self.playback_process:
            frame_size = self.project_width * self.project_height * 3
            frame_bytes = self.playback_process.stdout.read(frame_size)
            if len(frame_bytes) == frame_size:
                image = QImage(frame_bytes, self.project_width, self.project_height, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(self.preview_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_widget.setPixmap(scaled_pixmap)
            else:
                self._stop_playback_stream()

    def _load_settings(self):
        self.settings_file_was_loaded = False
        defaults = {"window_visibility": {"project_media": True}, "splitter_state": None, "enabled_plugins": [], "recent_files": [], "confirm_on_exit": True}
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
            with open(self.settings_file, "w") as f: json.dump(self.settings, f, indent=4)
        except IOError as e: print(f"Error saving settings: {e}")

    def _apply_loaded_settings(self):
        visibility_settings = self.settings.get("window_visibility", {})
        for key, data in self.managed_widgets.items():
            if data.get('plugin'): continue
            is_visible = visibility_settings.get(key, True)
            if data['widget'] is not self.preview_widget:
                data['widget'].setVisible(is_visible)
            if data['action']: data['action'].setChecked(is_visible)
        splitter_state = self.settings.get("splitter_state")
        if splitter_state: self.splitter.restoreState(QByteArray.fromHex(splitter_state.encode('ascii')))

    def on_splitter_moved(self, pos, index): self.splitter_save_timer.start(500)
    
    def toggle_widget_visibility(self, key, checked):
        if self.is_shutting_down: return
        if key in self.managed_widgets:
            self.managed_widgets[key]['widget'].setVisible(checked)
            self._save_settings()

    def open_settings_dialog(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dialog.get_settings()); self._save_settings(); self.status_label.setText("Settings updated.")

    def new_project(self):
        self.timeline.clips.clear(); self.timeline.num_video_tracks = 1; self.timeline.num_audio_tracks = 1
        self.media_pool.clear(); self.media_properties.clear(); self.project_media_widget.clear_list()
        self.current_project_path = None; self.stop_playback()
        self.project_fps = 25.0
        self.timeline_widget.set_project_fps(self.project_fps)
        self.timeline_widget.update()
        self.undo_stack = UndoStack()
        self.undo_stack.history_changed.connect(self.update_undo_redo_actions)
        self.update_undo_redo_actions()
        self.status_label.setText("New project created. Add media to begin.")

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Project Files (*.json)")
        if not path: return
        project_data = {
            "media_pool": self.media_pool,
            "clips": [{"source_path": c.source_path, "timeline_start_sec": c.timeline_start_sec, "clip_start_sec": c.clip_start_sec, "duration_sec": c.duration_sec, "track_index": c.track_index, "track_type": c.track_type, "media_type": c.media_type, "group_id": c.group_id} for c in self.timeline.clips],
            "settings": {"num_video_tracks": self.timeline.num_video_tracks, "num_audio_tracks": self.timeline.num_audio_tracks}
        }
        try:
            with open(path, "w") as f: json.dump(project_data, f, indent=4)
            self.current_project_path = path; self.status_label.setText(f"Project saved to {os.path.basename(path)}")
            self._add_to_recent_files(path)
        except Exception as e: self.status_label.setText(f"Error saving project: {e}")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Project Files (*.json)")
        if path: self._load_project_from_path(path)

    def _load_project_from_path(self, path):
        try:
            with open(path, "r") as f: project_data = json.load(f)
            self.new_project()
            
            self.media_pool = project_data.get("media_pool", [])
            for p in self.media_pool: self._add_media_to_pool(p)
            
            for clip_data in project_data["clips"]:
                if not os.path.exists(clip_data["source_path"]):
                    self.status_label.setText(f"Error: Missing media file {clip_data['source_path']}"); self.new_project(); return
                
                if 'media_type' not in clip_data:
                    ext = os.path.splitext(clip_data['source_path'])[1].lower()
                    if ext in ['.mp3', '.wav', '.m4a', '.aac']:
                         clip_data['media_type'] = 'audio'
                    else:
                         clip_data['media_type'] = 'video'
                
                self.timeline.add_clip(TimelineClip(**clip_data))
            
            project_settings = project_data.get("settings", {})
            self.timeline.num_video_tracks = project_settings.get("num_video_tracks", 1)
            self.timeline.num_audio_tracks = project_settings.get("num_audio_tracks", 1)

            self.current_project_path = path
            video_clips = [c for c in self.timeline.clips if c.media_type == 'video']
            if video_clips:
                self._set_project_properties_from_clip(video_clips[0].source_path)
            self.prune_empty_tracks()
            self.timeline_widget.update(); self.stop_playback()
            self.status_label.setText(f"Project '{os.path.basename(path)}' loaded.")
            self._add_to_recent_files(path)
        except Exception as e: self.status_label.setText(f"Error opening project: {e}")

    def _add_to_recent_files(self, path):
        recent = self.settings.get("recent_files", [])
        if path in recent: recent.remove(path)
        recent.insert(0, path)
        self.settings["recent_files"] = recent[:10]
        self._update_recent_files_menu()
        self._save_settings()

    def _update_recent_files_menu(self):
        self.recent_menu.clear()
        recent_files = self.settings.get("recent_files", [])
        for path in recent_files:
            if os.path.exists(path):
                action = QAction(os.path.basename(path), self)
                action.triggered.connect(lambda checked, p=path: self._load_project_from_path(p))
                self.recent_menu.addAction(action)

    def _add_media_to_pool(self, file_path):
        if file_path in self.media_pool: return True
        try:
            self.status_label.setText(f"Probing {os.path.basename(file_path)}..."); QApplication.processEvents()
            
            file_ext = os.path.splitext(file_path)[1].lower()
            media_info = {}
            
            if file_ext in ['.png', '.jpg', '.jpeg']:
                media_info['media_type'] = 'image'
                media_info['duration'] = 5.0
                media_info['has_audio'] = False
            else:
                probe = ffmpeg.probe(file_path)
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

                if video_stream:
                    media_info['media_type'] = 'video'
                    media_info['duration'] = float(video_stream.get('duration', probe['format'].get('duration', 0)))
                    media_info['has_audio'] = audio_stream is not None
                elif audio_stream:
                    media_info['media_type'] = 'audio'
                    media_info['duration'] = float(audio_stream.get('duration', probe['format'].get('duration', 0)))
                    media_info['has_audio'] = True
                else:
                    raise ValueError("No video or audio stream found.")

            self.media_properties[file_path] = media_info
            self.media_pool.append(file_path)
            self.project_media_widget.add_media_item(file_path)
            
            self.status_label.setText(f"Added {os.path.basename(file_path)} to project.")
            return True
        except Exception as e:
            self.status_label.setText(f"Error probing file: {e}")
            return False

    def on_media_removed_from_pool(self, file_path):
        old_state = self._get_current_timeline_state()
        
        if file_path in self.media_pool: self.media_pool.remove(file_path)
        if file_path in self.media_properties: del self.media_properties[file_path]
        
        clips_to_remove = [c for c in self.timeline.clips if c.source_path == file_path]
        for clip in clips_to_remove: self.timeline.clips.remove(clip)

        new_state = self._get_current_timeline_state()
        command = TimelineStateChangeCommand("Remove Media From Project", self.timeline, *old_state, *new_state)
        command.undo()
        self.undo_stack.push(command)


    def add_media_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Media Files", "", "All Supported Files (*.mp4 *.mov *.avi *.png *.jpg *.jpeg *.mp3 *.wav);;Video Files (*.mp4 *.mov *.avi);;Image Files (*.png *.jpg *.jpeg);;Audio Files (*.mp3 *.wav)")
        if not file_paths: return
        self.media_dock.show()

        first_video_added = any(c.media_type == 'video' for c in self.timeline.clips)

        for file_path in file_paths:
            if not self.timeline.clips and not self.media_pool and not first_video_added:
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in ['.png', '.jpg', '.jpeg', '.mp3', '.wav']:
                    if self._set_project_properties_from_clip(file_path):
                        first_video_added = True
                    else:
                        self.status_label.setText("Error: Could not determine video properties from file."); 
                        continue
            
            if self._add_media_to_pool(file_path):
                 self.status_label.setText(f"Added {os.path.basename(file_path)} to project media.")
            else:
                 self.status_label.setText(f"Failed to add {os.path.basename(file_path)}.")

    def _add_clip_to_timeline(self, source_path, timeline_start_sec, duration_sec, media_type, clip_start_sec=0.0, video_track_index=None, audio_track_index=None):
        old_state = self._get_current_timeline_state()
        group_id = str(uuid.uuid4())
        
        if video_track_index is not None:
             if video_track_index > self.timeline.num_video_tracks:
                 self.timeline.num_video_tracks = video_track_index
             video_clip = TimelineClip(source_path, timeline_start_sec, clip_start_sec, duration_sec, video_track_index, 'video', media_type, group_id)
             self.timeline.add_clip(video_clip)
        
        if audio_track_index is not None:
             if audio_track_index > self.timeline.num_audio_tracks:
                 self.timeline.num_audio_tracks = audio_track_index
             audio_clip = TimelineClip(source_path, timeline_start_sec, clip_start_sec, duration_sec, audio_track_index, 'audio', media_type, group_id)
             self.timeline.add_clip(audio_clip)

        new_state = self._get_current_timeline_state()
        command = TimelineStateChangeCommand("Add Clip", self.timeline, *old_state, *new_state)
        command.undo()
        self.undo_stack.push(command)

    def _split_at_time(self, clip_to_split, time_sec, new_group_id=None):
        if not (clip_to_split.timeline_start_sec < time_sec < clip_to_split.timeline_end_sec): return False
        split_point = time_sec - clip_to_split.timeline_start_sec
        orig_dur = clip_to_split.duration_sec
        group_id_for_new_clip = new_group_id if new_group_id is not None else clip_to_split.group_id
        
        new_clip = TimelineClip(clip_to_split.source_path, time_sec, clip_to_split.clip_start_sec + split_point, orig_dur - split_point, clip_to_split.track_index, clip_to_split.track_type, clip_to_split.media_type, group_id_for_new_clip)
        clip_to_split.duration_sec = split_point
        self.timeline.add_clip(new_clip)
        return True

    def split_clip_at_playhead(self, clip_to_split=None):
        playhead_time = self.timeline_widget.playhead_pos_sec
        if not clip_to_split:
            clips_at_playhead = [c for c in self.timeline.clips if c.timeline_start_sec < playhead_time < c.timeline_end_sec]
            if not clips_at_playhead:
                self.status_label.setText("Playhead is not over a clip to split.")
                return
            clip_to_split = clips_at_playhead[0]

        old_state = self._get_current_timeline_state()

        linked_clip = next((c for c in self.timeline.clips if c.group_id == clip_to_split.group_id and c.id != clip_to_split.id), None)
        new_right_side_group_id = str(uuid.uuid4())
        
        split1 = self._split_at_time(clip_to_split, playhead_time, new_group_id=new_right_side_group_id)
        if linked_clip:
            self._split_at_time(linked_clip, playhead_time, new_group_id=new_right_side_group_id)

        if split1:
            new_state = self._get_current_timeline_state()
            command = TimelineStateChangeCommand("Split Clip", self.timeline, *old_state, *new_state)
            command.undo()
            self.undo_stack.push(command)
        else:
            self.status_label.setText("Failed to split clip.")


    def delete_clip(self, clip_to_delete):
        old_state = self._get_current_timeline_state()

        linked_clip = next((c for c in self.timeline.clips if c.group_id == clip_to_delete.group_id and c.id != clip_to_delete.id), None)
        
        if clip_to_delete in self.timeline.clips: self.timeline.clips.remove(clip_to_delete)
        if linked_clip and linked_clip in self.timeline.clips: self.timeline.clips.remove(linked_clip)
            
        new_state = self._get_current_timeline_state()
        command = TimelineStateChangeCommand("Delete Clip", self.timeline, *old_state, *new_state)
        command.undo()
        self.undo_stack.push(command)
        self.prune_empty_tracks()

    def _perform_complex_timeline_change(self, description, change_function):
        old_state = self._get_current_timeline_state()
        change_function()
        
        new_state = self._get_current_timeline_state()
        if old_state[0] == new_state[0] and old_state[1] == new_state[1] and old_state[2] == new_state[2]:
            return
            
        command = TimelineStateChangeCommand(description, self.timeline, *old_state, *new_state)
        command.undo()
        self.undo_stack.push(command)

    def on_split_region(self, region):
        def action():
            start_sec, end_sec = region
            clips = list(self.timeline.clips)
            for clip in clips: self._split_at_time(clip, end_sec)
            for clip in clips: self._split_at_time(clip, start_sec)
            self.timeline_widget.clear_region(region)
        self._perform_complex_timeline_change("Split Region", action)

    def on_split_all_regions(self, regions):
        def action():
            split_points = set()
            for start, end in regions:
                split_points.add(start)
                split_points.add(end)

            for point in sorted(list(split_points)):
                group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
                new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                for clip in list(self.timeline.clips):
                    if clip.group_id in new_group_ids:
                        self._split_at_time(clip, point, new_group_ids[clip.group_id])
            self.timeline_widget.clear_all_regions()
        self._perform_complex_timeline_change("Split All Regions", action)

    def on_join_region(self, region):
        def action():
            start_sec, end_sec = region
            duration_to_remove = end_sec - start_sec
            if duration_to_remove <= 0.01: return

            for point in [start_sec, end_sec]:
                 group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
                 new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                 for clip in list(self.timeline.clips):
                     if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])

            clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_sec >= start_sec and c.timeline_start_sec < end_sec]
            for clip in clips_to_remove: self.timeline.clips.remove(clip)

            for clip in self.timeline.clips:
                if clip.timeline_start_sec >= end_sec:
                    clip.timeline_start_sec -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_sec)
            self.timeline_widget.clear_region(region)
        self._perform_complex_timeline_change("Join Region", action)

    def on_join_all_regions(self, regions):
        def action():
            for region in sorted(regions, key=lambda r: r[0], reverse=True):
                start_sec, end_sec = region
                duration_to_remove = end_sec - start_sec
                if duration_to_remove <= 0.01: continue

                for point in [start_sec, end_sec]:
                    group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
                    new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                    for clip in list(self.timeline.clips):
                        if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])
                clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_sec >= start_sec and c.timeline_start_sec < end_sec]
                for clip in clips_to_remove:
                    try: self.timeline.clips.remove(clip)
                    except ValueError: pass 

                for clip in self.timeline.clips:
                    if clip.timeline_start_sec >= end_sec:
                        clip.timeline_start_sec -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_sec)
            self.timeline_widget.clear_all_regions()
        self._perform_complex_timeline_change("Join All Regions", action)

    def on_delete_region(self, region):
        def action():
            start_sec, end_sec = region
            duration_to_remove = end_sec - start_sec
            if duration_to_remove <= 0.01: return

            for point in [start_sec, end_sec]:
                 group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
                 new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                 for clip in list(self.timeline.clips):
                     if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])

            clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_sec >= start_sec and c.timeline_start_sec < end_sec]
            for clip in clips_to_remove: self.timeline.clips.remove(clip)

            for clip in self.timeline.clips:
                if clip.timeline_start_sec >= end_sec:
                    clip.timeline_start_sec -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_sec)
            self.timeline_widget.clear_region(region)
        self._perform_complex_timeline_change("Delete Region", action)

    def on_delete_all_regions(self, regions):
        def action():
            for region in sorted(regions, key=lambda r: r[0], reverse=True):
                start_sec, end_sec = region
                duration_to_remove = end_sec - start_sec
                if duration_to_remove <= 0.01: continue

                for point in [start_sec, end_sec]:
                    group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_sec < point < c.timeline_end_sec}
                    new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                    for clip in list(self.timeline.clips):
                        if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])

                clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_sec >= start_sec and c.timeline_start_sec < end_sec]
                for clip in clips_to_remove:
                    try: self.timeline.clips.remove(clip)
                    except ValueError: pass
                
                # Ripple
                for clip in self.timeline.clips:
                    if clip.timeline_start_sec >= end_sec:
                        clip.timeline_start_sec -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_sec)
            self.timeline_widget.clear_all_regions()
        self._perform_complex_timeline_change("Delete All Regions", action)


    def export_video(self):
        if not self.timeline.clips: self.status_label.setText("Timeline is empty."); return
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "MP4 Files (*.mp4)")
        if not output_path: return

        w, h, fr_str, total_dur = self.project_width, self.project_height, str(self.project_fps), self.timeline.get_total_duration()
        sample_rate, channel_layout = '44100', 'stereo'
        
        input_streams = {}

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
                
                if clip.source_path not in input_streams:
                    if clip.media_type == 'image':
                        input_streams[clip.source_path] = ffmpeg.input(clip.source_path, loop=1, framerate=self.project_fps)
                    else:
                        input_streams[clip.source_path] = ffmpeg.input(clip.source_path)

                if clip.media_type == 'image':
                    v_seg = (input_streams[clip.source_path].video.trim(duration=clip.duration_sec).setpts('PTS-STARTPTS'))
                else: # video
                    v_seg = (input_streams[clip.source_path].video.trim(start=clip.clip_start_sec, duration=clip.duration_sec).setpts('PTS-STARTPTS'))

                v_seg = (v_seg.filter('scale', w, h, force_original_aspect_ratio='decrease').filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black').filter('format', pix_fmts='rgba'))
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
                if clip.source_path not in input_streams:
                    input_streams[clip.source_path] = ffmpeg.input(clip.source_path)

                gap = clip.timeline_start_sec - last_end
                if gap > 0.01: 
                    track_segments.append(ffmpeg.input(f'anullsrc=r={sample_rate}:cl={channel_layout}:d={gap}', f='lavfi'))
                
                a_seg = input_streams[clip.source_path].audio.filter('atrim', start=clip.clip_start_sec, duration=clip.duration_sec).filter('asetpts', 'PTS-STARTPTS')
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
        action.toggled.connect(dock.setVisible)
        dock.visibilityChanged.connect(action.setChecked)
        action.setChecked(dock.isVisible()) 
        self.windows_menu.addAction(action)
        self.managed_widgets[widget_key] = {'widget': dock, 'name': title, 'action': action, 'plugin': plugin_instance.name}
        return dock
        
    def update_plugin_ui_visibility(self, plugin_name, is_enabled):
        for key, data in self.managed_widgets.items():
            if data.get('plugin') == plugin_name:
                data['action'].setVisible(is_enabled)
                if not is_enabled: data['widget'].hide()

    def toggle_plugin(self, name, checked):
        if checked: self.plugin_manager.enable_plugin(name)
        else: self.plugin_manager.disable_plugin(name)
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
        if self.settings.get("confirm_on_exit", True):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Confirm Exit")
            msg_box.setText("Are you sure you want to exit?")
            msg_box.setInformativeText("Any unsaved changes will be lost.")
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)
            
            dont_ask_cb = QCheckBox("Don't ask again")
            msg_box.setCheckBox(dont_ask_cb)
            
            reply = msg_box.exec()

            if dont_ask_cb.isChecked():
                self.settings['confirm_on_exit'] = False

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

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