import wgp
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
                             QCheckBox, QDialogButtonBox, QMenu, QSplitter, QDockWidget,
                             QListWidget, QListWidgetItem, QMessageBox, QComboBox,
                             QFormLayout, QGroupBox, QLineEdit)
from PyQt6.QtGui import (QPainter, QColor, QPen, QFont, QFontMetrics, QMouseEvent, QAction,
                         QPixmap, QImage, QDrag, QCursor, QKeyEvent)
from PyQt6.QtCore import (Qt, QPoint, QRect, QRectF, QSize, QPointF, QObject, QThread,
                          pyqtSignal, QTimer, QByteArray, QMimeData)

from plugins import PluginManager, ManagePluginsDialog
from undo import UndoStack, TimelineStateChangeCommand, MoveClipsCommand

CONTAINER_PRESETS = {
    'mp4': {
        'vcodec': 'libx264', 'acodec': 'aac', 
        'allowed_vcodecs': ['libx264', 'libx265', 'mpeg4'],
        'allowed_acodecs': ['aac', 'libmp3lame'],
        'v_bitrate': '5M', 'a_bitrate': '192k'
    },
    'matroska': {
        'vcodec': 'libx264', 'acodec': 'aac',
        'allowed_vcodecs': ['libx264', 'libx265', 'libvpx-vp9'],
        'allowed_acodecs': ['aac', 'libopus', 'libvorbis', 'flac'],
        'v_bitrate': '5M', 'a_bitrate': '192k'
    },
    'mov': {
        'vcodec': 'libx264', 'acodec': 'aac',
        'allowed_vcodecs': ['libx264', 'prores_ks', 'mpeg4'],
        'allowed_acodecs': ['aac', 'pcm_s16le'],
        'v_bitrate': '8M', 'a_bitrate': '256k'
    },
    'avi': {
        'vcodec': 'mpeg4', 'acodec': 'libmp3lame',
        'allowed_vcodecs': ['mpeg4', 'msmpeg4'],
        'allowed_acodecs': ['libmp3lame'],
        'v_bitrate': '5M', 'a_bitrate': '192k'
    },
    'webm': {
        'vcodec': 'libvpx-vp9', 'acodec': 'libopus',
        'allowed_vcodecs': ['libvpx-vp9'],
        'allowed_acodecs': ['libopus', 'libvorbis'],
        'v_bitrate': '4M', 'a_bitrate': '192k'
    },
    'wav': {
        'vcodec': None, 'acodec': 'pcm_s16le',
        'allowed_vcodecs': [], 'allowed_acodecs': ['pcm_s16le', 'pcm_s24le'],
        'v_bitrate': None, 'a_bitrate': None
    },
    'mp3': {
        'vcodec': None, 'acodec': 'libmp3lame',
        'allowed_vcodecs': [], 'allowed_acodecs': ['libmp3lame'],
        'v_bitrate': None, 'a_bitrate': '192k'
    },
    'flac': {
        'vcodec': None, 'acodec': 'flac',
        'allowed_vcodecs': [], 'allowed_acodecs': ['flac'],
        'v_bitrate': None, 'a_bitrate': None
    },
    'gif': {
        'vcodec': 'gif', 'acodec': None,
        'allowed_vcodecs': ['gif'], 'allowed_acodecs': [],
        'v_bitrate': None, 'a_bitrate': None
    },
    'oga': { # Using oga for ogg audio
        'vcodec': None, 'acodec': 'libvorbis',
        'allowed_vcodecs': [], 'allowed_acodecs': ['libvorbis', 'libopus'],
        'v_bitrate': None, 'a_bitrate': '192k'
    }
}

_cached_formats = None
_cached_video_codecs = None
_cached_audio_codecs = None

def run_ffmpeg_command(args):
    try:
        startupinfo = None
        if hasattr(subprocess, 'STARTUPINFO'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        
        result = subprocess.run(
            ['ffmpeg'] + args,
            capture_output=True, text=True, encoding='utf-8',
            errors='ignore', startupinfo=startupinfo
        )
        if result.returncode != 0 and "Unrecognized option" not in result.stderr:
             print(f"FFmpeg command failed: {' '.join(args)}\n{result.stderr}")
             return ""
        return result.stdout
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure it is in your system's PATH.")
        return None
    except Exception as e:
        print(f"An error occurred while running ffmpeg: {e}")
        return None

def get_available_formats():
    global _cached_formats
    if _cached_formats is not None:
        return _cached_formats

    output = run_ffmpeg_command(['-formats'])
    if not output:
        _cached_formats = {}
        return {}

    formats = {}
    lines = output.split('\n')
    header_found = False
    for line in lines:
        if "---" in line:
            header_found = True
            continue
        if not header_found or not line.strip():
            continue

        if line[2] == 'E':
            parts = line[4:].strip().split(None, 1)
            if len(parts) == 2:
                names, description = parts
                primary_name = names.split(',')[0].strip()
                formats[primary_name] = description.strip()

    _cached_formats = dict(sorted(formats.items()))
    return _cached_formats

def get_available_codecs(codec_type='video'):
    global _cached_video_codecs, _cached_audio_codecs
    
    if codec_type == 'video' and _cached_video_codecs is not None: return _cached_video_codecs
    if codec_type == 'audio' and _cached_audio_codecs is not None: return _cached_audio_codecs

    output = run_ffmpeg_command(['-encoders'])
    if not output:
        if codec_type == 'video': _cached_video_codecs = {}
        else: _cached_audio_codecs = {}
        return {}

    video_codecs = {}
    audio_codecs = {}
    lines = output.split('\n')
    
    header_found = False
    for line in lines:
        if "------" in line:
            header_found = True
            continue

        if not header_found or not line.strip():
            continue

        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue

        flags, name, description = parts
        type_flag = flags[0]
        clean_description = re.sub(r'\s*\(codec .*\)$', '', description).strip()

        if type_flag == 'V':
            video_codecs[name] = clean_description
        elif type_flag == 'A':
            audio_codecs[name] = clean_description

    _cached_video_codecs = dict(sorted(video_codecs.items()))
    _cached_audio_codecs = dict(sorted(audio_codecs.items()))

    return _cached_video_codecs if codec_type == 'video' else _cached_audio_codecs

class TimelineClip:
    def __init__(self, source_path, timeline_start_ms, clip_start_ms, duration_ms, track_index, track_type, media_type, group_id):
        self.id = str(uuid.uuid4())
        self.source_path = source_path
        self.timeline_start_ms = int(timeline_start_ms)
        self.clip_start_ms = int(clip_start_ms)
        self.duration_ms = int(duration_ms)
        self.track_index = track_index
        self.track_type = track_type
        self.media_type = media_type
        self.group_id = group_id

    @property
    def timeline_end_ms(self):
        return self.timeline_start_ms + self.duration_ms

class Timeline:
    def __init__(self):
        self.clips = []
        self.num_video_tracks = 1
        self.num_audio_tracks = 1

    def add_clip(self, clip):
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.timeline_start_ms)

    def get_total_duration(self):
        if not self.clips: return 0
        return max(c.timeline_end_ms for c in self.clips)

class ExportWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    def __init__(self, ffmpeg_cmd, total_duration_ms):
        super().__init__()
        self.ffmpeg_cmd = ffmpeg_cmd
        self.total_duration_ms = total_duration_ms

    def run_export(self):
        try:
            process = subprocess.Popen(self.ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding="utf-8")
            time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
            for line in iter(process.stdout.readline, ""):
                match = time_pattern.search(line)
                if match:
                    h, m, s, cs = [int(g) for g in match.groups()]
                    processed_ms = (h * 3600 + m * 60 + s) * 1000 + cs * 10
                    if self.total_duration_ms > 0:
                        percentage = int((processed_ms / self.total_duration_ms) * 100)
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
    delete_clips_requested = pyqtSignal(list)
    playhead_moved = pyqtSignal(int)
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
        self.playhead_pos_ms = 0
        self.view_start_ms = 0
        self.panning = False
        self.pan_start_pos = QPoint()
        self.pan_start_view_ms = 0
        
        self.pixels_per_ms = 0.05
        self.max_pixels_per_ms = 1.0
        self.project_fps = 25.0
        self.set_project_fps(project_fps)

        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.selection_regions = []
        self.selected_clips = set()
        self.dragging_clip = None
        self.dragging_linked_clip = None
        self.dragging_playhead = False
        self.creating_selection_region = False
        self.dragging_selection_region = None
        self.drag_start_pos = QPoint()
        self.drag_original_clip_states = {}
        self.selection_drag_start_ms = 0
        self.drag_selection_start_values = None
        self.drag_start_state = None

        self.resizing_clip = None
        self.resize_edge = None
        self.resize_start_pos = QPoint()

        self.resizing_selection_region = None
        self.resize_selection_edge = None
        self.resize_selection_start_values = None

        self.highlighted_track_info = None
        self.highlighted_ghost_track_info = None
        self.add_video_track_btn_rect = QRect()
        self.remove_video_track_btn_rect = QRect()
        self.add_audio_track_btn_rect = QRect()
        self.remove_audio_track_btn_rect = QRect()
        
        self.video_tracks_y_start = 0
        self.audio_tracks_y_start = 0
        self.hover_preview_rect = None
        self.hover_preview_audio_rect = None
        self.drag_over_active = False
        self.drag_over_rect = QRectF()
        self.drag_over_audio_rect = QRectF()
        self.drag_url_cache = {}

    def set_hover_preview_rects(self, video_rect, audio_rect):
        self.hover_preview_rect = video_rect
        self.hover_preview_audio_rect = audio_rect
        self.update()

    def set_project_fps(self, fps):
        self.project_fps = fps if fps > 0 else 25.0
        self.max_pixels_per_ms = (self.project_fps * 20) / 1000.0
        self.pixels_per_ms = min(self.pixels_per_ms, self.max_pixels_per_ms)
        self.update()

    def ms_to_x(self, ms): return self.HEADER_WIDTH + int((ms - self.view_start_ms) * self.pixels_per_ms)
    def x_to_ms(self, x): return self.view_start_ms + int(float(x - self.HEADER_WIDTH) / self.pixels_per_ms) if x > self.HEADER_WIDTH and self.pixels_per_ms > 0 else self.view_start_ms

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#333"))

        self.draw_headers(painter)

        painter.save()
        painter.setClipRect(self.HEADER_WIDTH, 0, self.width() - self.HEADER_WIDTH, self.height())
        
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

        if self.hover_preview_rect:
            painter.setPen(QPen(QColor(0, 255, 255, 180), 2, Qt.PenStyle.DashLine))
            painter.fillRect(self.hover_preview_rect, QColor(0, 255, 255, 60))
            painter.drawRect(self.hover_preview_rect)
        if self.hover_preview_audio_rect:
            painter.setPen(QPen(QColor(0, 255, 255, 180), 2, Qt.PenStyle.DashLine))
            painter.fillRect(self.hover_preview_audio_rect, QColor(0, 255, 255, 60))
            painter.drawRect(self.hover_preview_audio_rect)

        self.draw_playhead(painter)
        
        painter.restore()

        total_height = self.calculate_total_height()
        if self.minimumHeight() != total_height:
            self.setMinimumHeight(total_height)

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
        
    def _format_timecode(self, total_ms):
        if abs(total_ms) < 1: total_ms = 0
        sign = "-" if total_ms < 0 else ""
        total_ms = abs(total_ms)
        
        seconds = total_ms / 1000.0
        h = int(seconds / 3600)
        m = int((seconds % 3600) / 60)
        s = seconds % 60
        
        if h > 0: return f"{sign}{h}h:{m:02d}m"
        if m > 0 or seconds >= 59.99: return f"{sign}{m}m:{int(round(s)):02d}s"
        
        precision = 2 if s < 1 else 1 if s < 10 else 0
        val = f"{s:.{precision}f}"
        if '.' in val: val = val.rstrip('0').rstrip('.')
        return f"{sign}{val}s"

    def draw_timescale(self, painter):
        painter.save()
        painter.setPen(QColor("#AAA"))
        painter.setFont(QFont("Arial", 8))
        font_metrics = QFontMetrics(painter.font())

        painter.fillRect(QRect(self.HEADER_WIDTH, 0, self.width() - self.HEADER_WIDTH, self.TIMESCALE_HEIGHT), QColor("#222"))
        painter.drawLine(self.HEADER_WIDTH, self.TIMESCALE_HEIGHT - 1, self.width(), self.TIMESCALE_HEIGHT - 1)

        frame_dur_ms = 1000.0 / self.project_fps
        intervals_ms = [
            frame_dur_ms, 2*frame_dur_ms, 5*frame_dur_ms, 10*frame_dur_ms,
            100, 200, 500, 1000, 2000, 5000, 10000, 15000, 30000,
            60000, 120000, 300000, 600000, 900000, 1800000,
            3600000, 2*3600000, 5*3600000, 10*3600000
        ]
        
        min_pixel_dist = 70
        major_interval = next((i for i in intervals_ms if i * self.pixels_per_ms > min_pixel_dist), intervals_ms[-1])

        minor_interval = 0
        for divisor in [5, 4, 2]:
            if (major_interval / divisor) * self.pixels_per_ms > 10:
                minor_interval = major_interval / divisor
                break
        
        start_ms = self.x_to_ms(self.HEADER_WIDTH)
        end_ms = self.x_to_ms(self.width())

        def draw_ticks(interval_ms, height):
            if interval_ms < 1: return
            start_tick_num = int(start_ms / interval_ms)
            end_tick_num = int(end_ms / interval_ms) + 1
            for i in range(start_tick_num, end_tick_num + 1):
                t_ms = i * interval_ms
                x = self.ms_to_x(t_ms)
                if x > self.width(): break
                if x >= self.HEADER_WIDTH:
                    painter.drawLine(x, self.TIMESCALE_HEIGHT - height, x, self.TIMESCALE_HEIGHT)
        
        if frame_dur_ms * self.pixels_per_ms > 4:
            draw_ticks(frame_dur_ms, 3)
        if minor_interval > 0:
            draw_ticks(minor_interval, 6)

        start_major_tick = int(start_ms / major_interval)
        end_major_tick = int(end_ms / major_interval) + 1
        for i in range(start_major_tick, end_major_tick + 1):
            t_ms = i * major_interval
            x = self.ms_to_x(t_ms)
            if x > self.width() + 50: break
            if x >= self.HEADER_WIDTH - 50:
                painter.drawLine(x, self.TIMESCALE_HEIGHT - 12, x, self.TIMESCALE_HEIGHT)
                label = self._format_timecode(t_ms)
                label_width = font_metrics.horizontalAdvance(label)
                label_x = x - label_width // 2
                if label_x < self.HEADER_WIDTH:
                    label_x = self.HEADER_WIDTH
                painter.drawText(label_x, self.TIMESCALE_HEIGHT - 14, label)

        painter.restore()

    def get_clip_rect(self, clip):
        if clip.track_type == 'video':
            visual_index = self.timeline.num_video_tracks - clip.track_index
            y = self.video_tracks_y_start + visual_index * self.TRACK_HEIGHT
        else:
            visual_index = clip.track_index - 1
            y = self.audio_tracks_y_start + visual_index * self.TRACK_HEIGHT
        
        x = self.ms_to_x(clip.timeline_start_ms)
        w = int(clip.duration_ms * self.pixels_per_ms)
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
            base_color = QColor("#46A")
            if clip.media_type == 'image':
                base_color = QColor("#4A6")
            elif clip.track_type == 'audio':
                base_color = QColor("#48C")
            
            color = QColor("#5A9") if self.dragging_clip and self.dragging_clip.id == clip.id else base_color
            painter.fillRect(clip_rect, color)

            if clip.id in self.selected_clips:
                pen = QPen(QColor(255, 255, 0, 220), 2)
                painter.setPen(pen)
                painter.drawRect(clip_rect)

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
        for start_ms, end_ms in self.selection_regions:
            x = self.ms_to_x(start_ms)
            w = int((end_ms - start_ms) * self.pixels_per_ms)
            selection_rect = QRectF(x, self.TIMESCALE_HEIGHT, w, self.height() - self.TIMESCALE_HEIGHT)
            painter.fillRect(selection_rect, QColor(100, 100, 255, 80))
            painter.setPen(QColor(150, 150, 255, 150))
            painter.drawRect(selection_rect)

    def draw_playhead(self, painter):
        playhead_x = self.ms_to_x(self.playhead_pos_ms)
        painter.setPen(QPen(QColor("red"), 2))
        painter.drawLine(playhead_x, 0, playhead_x, self.height())

    def y_to_track_info(self, y):
        if self.TIMESCALE_HEIGHT <= y < self.video_tracks_y_start:
            return ('video', self.timeline.num_video_tracks + 1)

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

        add_audio_btn_y_start = self.audio_tracks_y_start + self.timeline.num_audio_tracks * self.TRACK_HEIGHT
        add_audio_btn_y_end = add_audio_btn_y_start + self.TRACK_HEIGHT
        if add_audio_btn_y_start <= y < add_audio_btn_y_end:
            return ('audio', self.timeline.num_audio_tracks + 1)
            
        return None

    def _snap_time_if_needed(self, time_ms):
        frame_duration_ms = 1000.0 / self.project_fps
        if frame_duration_ms > 0 and frame_duration_ms * self.pixels_per_ms > 4:
            frame_number = round(time_ms / frame_duration_ms)
            return int(frame_number * frame_duration_ms)
        return int(time_ms)

    def get_region_at_pos(self, pos: QPoint):
        if pos.y() <= self.TIMESCALE_HEIGHT or pos.x() <= self.HEADER_WIDTH:
            return None
        
        clicked_ms = self.x_to_ms(pos.x())
        for region in reversed(self.selection_regions):
            if region[0] <= clicked_ms <= region[1]:
                return region
        return None

    def wheelEvent(self, event: QMouseEvent):
        delta = event.angleDelta().y()
        zoom_factor = 1.15
        old_pps = self.pixels_per_ms

        if delta > 0:
            new_pps = old_pps * zoom_factor
        else:
            new_pps = old_pps / zoom_factor

        min_pps = 1 / (3600 * 10 * 1000)
        new_pps = max(min_pps, min(new_pps, self.max_pixels_per_ms))

        if abs(new_pps - old_pps) < 1e-9:
            return

        if event.position().x() < self.HEADER_WIDTH:
            new_view_start_ms = self.view_start_ms * (old_pps / new_pps)
        else:
            mouse_x = event.position().x()
            time_at_cursor = self.x_to_ms(mouse_x)
            new_view_start_ms = time_at_cursor - (mouse_x - self.HEADER_WIDTH) / new_pps

        self.pixels_per_ms = new_pps
        self.view_start_ms = int(max(0, new_view_start_ms))

        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.pan_start_view_ms = self.view_start_ms
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.pos().x() < self.HEADER_WIDTH:
            if self.add_video_track_btn_rect.contains(event.pos()): self.add_track.emit('video')
            elif self.remove_video_track_btn_rect.contains(event.pos()): self.remove_track.emit('video')
            elif self.add_audio_track_btn_rect.contains(event.pos()): self.add_track.emit('audio')
            elif self.remove_audio_track_btn_rect.contains(event.pos()): self.remove_track.emit('audio')
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()

            self.dragging_clip = None
            self.dragging_linked_clip = None
            self.dragging_playhead = False
            self.creating_selection_region = False
            self.dragging_selection_region = None
            self.resizing_clip = None
            self.resize_edge = None
            self.drag_original_clip_states.clear()
            self.resizing_selection_region = None
            self.resize_selection_edge = None
            self.resize_selection_start_values = None

            for region in self.selection_regions:
                if not region: continue
                x_start = self.ms_to_x(region[0])
                x_end = self.ms_to_x(region[1])
                if event.pos().y() > self.TIMESCALE_HEIGHT:
                    if abs(event.pos().x() - x_start) < self.RESIZE_HANDLE_WIDTH:
                        self.resizing_selection_region = region
                        self.resize_selection_edge = 'left'
                        break
                    elif abs(event.pos().x() - x_end) < self.RESIZE_HANDLE_WIDTH:
                        self.resizing_selection_region = region
                        self.resize_selection_edge = 'right'
                        break
            
            if self.resizing_selection_region:
                self.resize_selection_start_values = tuple(self.resizing_selection_region)
                self.drag_start_pos = event.pos()
                self.update()
                return

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

            clicked_clip = None
            for clip in reversed(self.timeline.clips):
                if self.get_clip_rect(clip).contains(QPointF(event.pos())):
                    clicked_clip = clip
                    break
            
            if clicked_clip:
                is_ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                
                if clicked_clip.id in self.selected_clips:
                    if is_ctrl_pressed:
                        self.selected_clips.remove(clicked_clip.id)
                else:
                    if not is_ctrl_pressed:
                        self.selected_clips.clear()
                    self.selected_clips.add(clicked_clip.id)

                if clicked_clip.id in self.selected_clips:
                    self.dragging_clip = clicked_clip
                    self.drag_start_state = self.window()._get_current_timeline_state()
                    self.drag_original_clip_states[clicked_clip.id] = (clicked_clip.timeline_start_ms, clicked_clip.track_index)
                    
                    self.dragging_linked_clip = next((c for c in self.timeline.clips if c.group_id == clicked_clip.group_id and c.id != clicked_clip.id), None)
                    if self.dragging_linked_clip:
                        self.drag_original_clip_states[self.dragging_linked_clip.id] = \
                            (self.dragging_linked_clip.timeline_start_ms, self.dragging_linked_clip.track_index)
                    self.drag_start_pos = event.pos()

            else:
                self.selected_clips.clear()
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
                        playhead_x = self.ms_to_x(self.playhead_pos_ms)
                        if abs(event.pos().x() - playhead_x) < self.SNAP_THRESHOLD_PIXELS:
                            self.selection_drag_start_ms = self.playhead_pos_ms
                        else:
                            self.selection_drag_start_ms = self.x_to_ms(event.pos().x())
                        self.selection_regions.append([self.selection_drag_start_ms, self.selection_drag_start_ms])
                    elif is_on_timescale:
                        time_ms = max(0, self.x_to_ms(event.pos().x()))
                        self.playhead_pos_ms = self._snap_time_if_needed(time_ms)
                        self.playhead_moved.emit(self.playhead_pos_ms)
                        self.dragging_playhead = True
            
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.panning:
            delta_x = event.pos().x() - self.pan_start_pos.x()
            time_delta = delta_x / self.pixels_per_ms
            new_view_start = self.pan_start_view_ms - time_delta
            self.view_start_ms = int(max(0, new_view_start))
            self.update()
            return

        if self.resizing_selection_region:
            current_ms = max(0, self.x_to_ms(event.pos().x()))
            original_start, original_end = self.resize_selection_start_values

            if self.resize_selection_edge == 'left':
                new_start = current_ms
                new_end = original_end
            else: # right
                new_start = original_start
                new_end = current_ms

            self.resizing_selection_region[0] = min(new_start, new_end)
            self.resizing_selection_region[1] = max(new_start, new_end)

            if (self.resize_selection_edge == 'left' and new_start > new_end) or \
               (self.resize_selection_edge == 'right' and new_end < new_start):
                self.resize_selection_edge = 'right' if self.resize_selection_edge == 'left' else 'left'
                self.resize_selection_start_values = (original_end, original_start)

            self.update()
            return
        if self.resizing_clip:
            linked_clip = next((c for c in self.timeline.clips if c.group_id == self.resizing_clip.group_id and c.id != self.resizing_clip.id), None)
            delta_x = event.pos().x() - self.resize_start_pos.x()
            time_delta = delta_x / self.pixels_per_ms
            min_duration_ms = int(1000 / self.project_fps)
            snap_time_delta = self.SNAP_THRESHOLD_PIXELS / self.pixels_per_ms

            snap_points = [self.playhead_pos_ms]
            for clip in self.timeline.clips:
                if clip.id == self.resizing_clip.id: continue
                if linked_clip and clip.id == linked_clip.id: continue
                snap_points.append(clip.timeline_start_ms)
                snap_points.append(clip.timeline_end_ms)

            media_props = self.window().media_properties.get(self.resizing_clip.source_path)
            source_duration_ms = media_props['duration_ms'] if media_props else float('inf')

            if self.resize_edge == 'left':
                original_start = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].timeline_start_ms
                original_duration = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].duration_ms
                original_clip_start = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].clip_start_ms
                true_new_start_ms = original_start + time_delta
                
                new_start_ms = true_new_start_ms
                for snap_point in snap_points:
                    if abs(true_new_start_ms - snap_point) < snap_time_delta:
                        new_start_ms = snap_point
                        break

                if new_start_ms > original_start + original_duration - min_duration_ms:
                    new_start_ms = original_start + original_duration - min_duration_ms

                new_start_ms = max(0, new_start_ms)

                if self.resizing_clip.media_type != 'image':
                    if new_start_ms < original_start - original_clip_start:
                         new_start_ms = original_start - original_clip_start

                new_duration = (original_start + original_duration) - new_start_ms
                new_clip_start = original_clip_start + (new_start_ms - original_start)
                
                if new_duration < min_duration_ms:
                    new_duration = min_duration_ms
                    new_start_ms = (original_start + original_duration) - new_duration
                    new_clip_start = original_clip_start + (new_start_ms - original_start)

                self.resizing_clip.timeline_start_ms = int(new_start_ms)
                self.resizing_clip.duration_ms = int(new_duration)
                self.resizing_clip.clip_start_ms = int(new_clip_start)
                if linked_clip:
                    linked_clip.timeline_start_ms = int(new_start_ms)
                    linked_clip.duration_ms = int(new_duration)
                    linked_clip.clip_start_ms = int(new_clip_start)

            elif self.resize_edge == 'right':
                original_start = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].timeline_start_ms
                original_duration = self.drag_start_state[0][[c.id for c in self.drag_start_state[0]].index(self.resizing_clip.id)].duration_ms
                
                true_new_duration = original_duration + time_delta
                true_new_end_time = original_start + true_new_duration
                
                new_end_time = true_new_end_time
                for snap_point in snap_points:
                    if abs(true_new_end_time - snap_point) < snap_time_delta:
                        new_end_time = snap_point
                        break
                
                new_duration = new_end_time - original_start
                
                if new_duration < min_duration_ms:
                    new_duration = min_duration_ms

                if self.resizing_clip.media_type != 'image':
                    if self.resizing_clip.clip_start_ms + new_duration > source_duration_ms:
                        new_duration = source_duration_ms - self.resizing_clip.clip_start_ms
                
                self.resizing_clip.duration_ms = int(new_duration)
                if linked_clip:
                    linked_clip.duration_ms = int(new_duration)

            self.update()
            return

        if not self.dragging_clip and not self.dragging_playhead and not self.creating_selection_region:
            cursor_set = False
            playhead_x = self.ms_to_x(self.playhead_pos_ms)
            is_in_track_area = event.pos().y() > self.TIMESCALE_HEIGHT and event.pos().x() > self.HEADER_WIDTH
            if is_in_track_area and abs(event.pos().x() - playhead_x) < self.SNAP_THRESHOLD_PIXELS:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                cursor_set = True
            
            if not cursor_set and is_in_track_area:
                for region in self.selection_regions:
                    x_start = self.ms_to_x(region[0])
                    x_end = self.ms_to_x(region[1])
                    if abs(event.pos().x() - x_start) < self.RESIZE_HANDLE_WIDTH or \
                       abs(event.pos().x() - x_end) < self.RESIZE_HANDLE_WIDTH:
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                        cursor_set = True
                        break
            if not cursor_set:
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
            current_ms = self.x_to_ms(event.pos().x())
            start = min(self.selection_drag_start_ms, current_ms)
            end = max(self.selection_drag_start_ms, current_ms)
            self.selection_regions[-1] = [start, end]
            self.update()
            return
        
        if self.dragging_selection_region:
            delta_x = event.pos().x() - self.drag_start_pos.x()
            time_delta = int(delta_x / self.pixels_per_ms)
            
            original_start, original_end = self.drag_selection_start_values
            duration = original_end - original_start
            new_start = max(0, original_start + time_delta)
            
            self.dragging_selection_region[0] = new_start
            self.dragging_selection_region[1] = new_start + duration
            
            self.update()
            return

        if self.dragging_playhead:
            time_ms = max(0, self.x_to_ms(event.pos().x()))
            self.playhead_pos_ms = self._snap_time_if_needed(time_ms)
            self.playhead_moved.emit(self.playhead_pos_ms)
            self.update()
        elif self.dragging_clip:
            self.highlighted_track_info = None
            self.highlighted_ghost_track_info = None
            new_track_info = self.y_to_track_info(event.pos().y())
            
            original_start_ms, _ = self.drag_original_clip_states[self.dragging_clip.id]

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
            time_delta = delta_x / self.pixels_per_ms
            true_new_start_time = original_start_ms + time_delta

            playhead_time = self.playhead_pos_ms
            snap_time_delta = self.SNAP_THRESHOLD_PIXELS / self.pixels_per_ms
            
            new_start_time = true_new_start_time
            true_new_end_time = true_new_start_time + self.dragging_clip.duration_ms
            
            if abs(true_new_start_time - playhead_time) < snap_time_delta:
                new_start_time = playhead_time
            elif abs(true_new_end_time - playhead_time) < snap_time_delta:
                new_start_time = playhead_time - self.dragging_clip.duration_ms

            for other_clip in self.timeline.clips:
                if other_clip.id == self.dragging_clip.id: continue
                if self.dragging_linked_clip and other_clip.id == self.dragging_linked_clip.id: continue
                if (other_clip.track_type != self.dragging_clip.track_type or 
                    other_clip.track_index != self.dragging_clip.track_index):
                    continue

                is_overlapping = (new_start_time < other_clip.timeline_end_ms and
                                  new_start_time + self.dragging_clip.duration_ms > other_clip.timeline_start_ms)
                
                if is_overlapping:
                    movement_direction = true_new_start_time - original_start_ms
                    if movement_direction > 0:
                        new_start_time = other_clip.timeline_start_ms - self.dragging_clip.duration_ms
                    else:
                        new_start_time = other_clip.timeline_end_ms
                    break 

            final_start_time = max(0, new_start_time)
            self.dragging_clip.timeline_start_ms = int(final_start_time)
            if self.dragging_linked_clip:
                self.dragging_linked_clip.timeline_start_ms = int(final_start_time)
            
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.unsetCursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.resizing_selection_region:
                self.resizing_selection_region = None
                self.resize_selection_edge = None
                self.resize_selection_start_values = None
                self.update()
                return
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
                    if (end - start) * self.pixels_per_ms < 2:
                        self.clear_all_regions()
            
            if self.dragging_selection_region:
                self.dragging_selection_region = None
                self.drag_selection_start_values = None

            self.dragging_playhead = False
            if self.dragging_clip:
                orig_start, orig_track = self.drag_original_clip_states[self.dragging_clip.id]
                moved = (orig_start != self.dragging_clip.timeline_start_ms or 
                         orig_track != self.dragging_clip.track_index)

                if self.dragging_linked_clip:
                    orig_start_link, orig_track_link = self.drag_original_clip_states[self.dragging_linked_clip.id]
                    moved = moved or (orig_start_link != self.dragging_linked_clip.timeline_start_ms or 
                                      orig_track_link != self.dragging_linked_clip.track_index)
                
                if moved:
                    self.window().finalize_clip_drag(self.drag_start_state)
                
                self.timeline.clips.sort(key=lambda c: c.timeline_start_ms)
                self.highlighted_track_info = None
                self.highlighted_ghost_track_info = None
                self.operation_finished.emit()

            self.dragging_clip = None
            self.dragging_linked_clip = None
            self.drag_original_clip_states.clear()
            self.drag_start_state = None
            
            self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-vnd.video.filepath') or event.mimeData().hasUrls():
            if event.mimeData().hasUrls():
                self.drag_url_cache.clear()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drag_over_active = False
        self.highlighted_ghost_track_info = None
        self.highlighted_track_info = None
        self.drag_url_cache.clear()
        self.update()

    def dragMoveEvent(self, event):
        mime_data = event.mimeData()
        media_props = None
        
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if not urls:
                event.ignore()
                return
            
            file_path = urls[0].toLocalFile()
            
            if file_path in self.drag_url_cache:
                media_props = self.drag_url_cache[file_path]
            else:
                probed_props = self.window()._probe_for_drag(file_path)
                if probed_props:
                    self.drag_url_cache[file_path] = probed_props
                    media_props = probed_props
        
        elif mime_data.hasFormat('application/x-vnd.video.filepath'):
            json_data_bytes = mime_data.data('application/x-vnd.video.filepath').data()
            media_props = json.loads(json_data_bytes.decode('utf-8'))
        
        if not media_props:
            if mime_data.hasUrls():
                event.acceptProposedAction()
                pos = event.position()
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
                
                self.update()
            else:
                event.ignore()
            return

        event.acceptProposedAction()
        
        duration_ms = media_props['duration_ms']
        media_type = media_props['media_type']
        has_audio = media_props['has_audio']

        pos = event.position()
        start_ms = self.x_to_ms(pos.x())
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
            
            width = int(duration_ms * self.pixels_per_ms)
            x = self.ms_to_x(start_ms)
            
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
        self.drag_url_cache.clear()
        self.update()
        
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            file_paths = [url.toLocalFile() for url in mime_data.urls()]
            main_window = self.window()
            added_files = main_window._add_media_files_to_project(file_paths)
            if not added_files:
                event.ignore()
                return

            pos = event.position()
            start_ms = self.x_to_ms(pos.x())
            track_info = self.y_to_track_info(pos.y())
            if not track_info:
                event.ignore()
                return
            
            current_timeline_pos = start_ms

            for file_path in added_files:
                media_info = main_window.media_properties.get(file_path)
                if not media_info: continue

                duration_ms = media_info['duration_ms']
                has_audio = media_info['has_audio']
                media_type = media_info['media_type']

                drop_track_type, drop_track_index = track_info
                video_track_idx = None
                audio_track_idx = None

                if media_type == 'image':
                    if drop_track_type == 'video': video_track_idx = drop_track_index
                elif media_type == 'audio':
                    if drop_track_type == 'audio': audio_track_idx = drop_track_index
                elif media_type == 'video':
                    if drop_track_type == 'video':
                        video_track_idx = drop_track_index
                        if has_audio: audio_track_idx = 1
                    elif drop_track_type == 'audio' and has_audio:
                        audio_track_idx = drop_track_index
                        video_track_idx = 1
                
                if video_track_idx is None and audio_track_idx is None:
                    continue

                main_window._add_clip_to_timeline(
                    source_path=file_path,
                    timeline_start_ms=current_timeline_pos,
                    duration_ms=duration_ms,
                    media_type=media_type,
                    clip_start_ms=0,
                    video_track_index=video_track_idx,
                    audio_track_index=audio_track_idx
                )
                current_timeline_pos += duration_ms
            
            event.acceptProposedAction()
            return

        if not mime_data.hasFormat('application/x-vnd.video.filepath'):
            return

        json_data = json.loads(mime_data.data('application/x-vnd.video.filepath').data().decode('utf-8'))
        file_path = json_data['path']
        duration_ms = json_data['duration_ms']
        has_audio = json_data['has_audio']
        media_type = json_data['media_type']

        pos = event.position()
        start_ms = self.x_to_ms(pos.x())
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
            timeline_start_ms=start_ms,
            duration_ms=duration_ms,
            media_type=media_type,
            clip_start_ms=0,
            video_track_index=video_track_idx,
            audio_track_index=audio_track_idx
        )

    def contextMenuEvent(self, event: 'QContextMenuEvent'):
        menu = QMenu(self)
        
        region_at_pos = self.get_region_at_pos(event.pos())
        if region_at_pos:
            num_regions = len(self.selection_regions)
            
            if num_regions == 1:
                split_this_action = menu.addAction("Split Region")
                join_this_action = menu.addAction("Join Region")
                delete_this_action = menu.addAction("Delete Region")
                menu.addSeparator()
                clear_this_action = menu.addAction("Clear Region")

                split_this_action.triggered.connect(lambda: self.split_region_requested.emit(region_at_pos))
                join_this_action.triggered.connect(lambda: self.join_region_requested.emit(region_at_pos))
                delete_this_action.triggered.connect(lambda: self.delete_region_requested.emit(region_at_pos))
                clear_this_action.triggered.connect(lambda: self.clear_region(region_at_pos))

            elif num_regions > 1:
                split_all_action = menu.addAction("Split All Regions")
                join_all_action = menu.addAction("Join All Regions")
                delete_all_action = menu.addAction("Delete All Regions")
                menu.addSeparator()
                clear_all_action = menu.addAction("Clear All Regions")

                split_all_action.triggered.connect(lambda: self.split_all_regions_requested.emit(self.selection_regions))
                join_all_action.triggered.connect(lambda: self.join_all_regions_requested.emit(self.selection_regions))
                delete_all_action.triggered.connect(lambda: self.delete_all_regions_requested.emit(self.selection_regions))
                clear_all_action.triggered.connect(self.clear_all_regions)

        clip_at_pos = None
        for clip in self.timeline.clips:
            if self.get_clip_rect(clip).contains(QPointF(event.pos())):
                clip_at_pos = clip
                break
        
        if clip_at_pos:
            if not menu.isEmpty(): menu.addSeparator()

            linked_clip = next((c for c in self.timeline.clips if c.group_id == clip_at_pos.group_id and c.id != clip_at_pos.id), None)
            if linked_clip:
                unlink_action = menu.addAction("Unlink Audio Track")
                unlink_action.triggered.connect(lambda: self.window().unlink_clip_pair(clip_at_pos))
            else:
                media_info = self.window().media_properties.get(clip_at_pos.source_path)
                if (clip_at_pos.track_type == 'video' and
                    media_info and media_info.get('has_audio')):
                    relink_action = menu.addAction("Relink Audio Track")
                    relink_action.triggered.connect(lambda: self.window().relink_clip_audio(clip_at_pos))

            split_action = menu.addAction("Split Clip")
            delete_action = menu.addAction("Delete Clip")
            playhead_time = self.playhead_pos_ms
            is_playhead_over_clip = (clip_at_pos.timeline_start_ms < playhead_time < clip_at_pos.timeline_end_ms)
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

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self.selected_clips:
                clips_to_delete = [c for c in self.timeline.clips if c.id in self.selected_clips]
                if clips_to_delete:
                    self.delete_clips_requested.emit(clips_to_delete)
        elif event.key() == Qt.Key.Key_Left:
            self.window().step_frame(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.window().step_frame(1)
        else:
            super().keyPressEvent(event)

    
class SettingsDialog(QDialog):
    def __init__(self, parent_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)
        self.confirm_on_exit_checkbox = QCheckBox("Confirm before exiting")
        self.confirm_on_exit_checkbox.setChecked(parent_settings.get("confirm_on_exit", True))
        layout.addWidget(self.confirm_on_exit_checkbox)

        export_path_group = QGroupBox("Default Export Path (for new projects)")
        export_path_layout = QHBoxLayout()
        self.default_export_path_edit = QLineEdit()
        self.default_export_path_edit.setPlaceholderText("Optional: e.g., C:/Users/YourUser/Videos/Exports")
        self.default_export_path_edit.setText(parent_settings.get("default_export_path", ""))
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_default_export_path)
        export_path_layout.addWidget(self.default_export_path_edit)
        export_path_layout.addWidget(browse_button)
        export_path_group.setLayout(export_path_layout)
        layout.addWidget(export_path_group)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def browse_default_export_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Default Export Folder", self.default_export_path_edit.text())
        if path:
            self.default_export_path_edit.setText(path)

    def get_settings(self):
        return {
            "confirm_on_exit": self.confirm_on_exit_checkbox.isChecked(),
            "default_export_path": self.default_export_path_edit.text(),
        }

class ExportDialog(QDialog):
    def __init__(self, default_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Settings")
        self.setMinimumWidth(550)

        self.video_bitrate_options = ["500k", "1M", "2.5M", "5M", "8M", "15M", "Custom..."]
        self.audio_bitrate_options = ["96k", "128k", "192k", "256k", "320k", "Custom..."]
        self.display_ext_map = {'matroska': 'mkv', 'oga': 'ogg'}

        self.layout = QVBoxLayout(self)
        self.formats = get_available_formats()
        self.video_codecs = get_available_codecs('video')
        self.audio_codecs = get_available_codecs('audio')

        self._setup_ui()
        
        self.path_edit.setText(default_path)
        self.on_advanced_toggled(False)

    def _setup_ui(self):
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.path_edit)
        output_layout.addWidget(browse_button)
        output_group.setLayout(output_layout)
        self.layout.addWidget(output_group)

        self.container_combo = QComboBox()
        self.container_combo.currentIndexChanged.connect(self.on_container_changed)
        
        self.advanced_formats_checkbox = QCheckBox("Show Advanced Options")
        self.advanced_formats_checkbox.toggled.connect(self.on_advanced_toggled)

        container_layout = QFormLayout()
        container_layout.addRow("Format Preset:", self.container_combo)
        container_layout.addRow(self.advanced_formats_checkbox)
        self.layout.addLayout(container_layout)

        self.video_group = QGroupBox("Video Settings")
        video_layout = QFormLayout()
        self.video_codec_combo = QComboBox()
        video_layout.addRow("Video Codec:", self.video_codec_combo)
        self.v_bitrate_combo = QComboBox()
        self.v_bitrate_combo.addItems(self.video_bitrate_options)
        self.v_bitrate_custom_edit = QLineEdit()
        self.v_bitrate_custom_edit.setPlaceholderText("e.g., 6500k")
        self.v_bitrate_custom_edit.hide()
        self.v_bitrate_combo.currentTextChanged.connect(self.on_v_bitrate_changed)
        video_layout.addRow("Video Bitrate:", self.v_bitrate_combo)
        video_layout.addRow(self.v_bitrate_custom_edit)
        self.video_group.setLayout(video_layout)
        self.layout.addWidget(self.video_group)

        self.audio_group = QGroupBox("Audio Settings")
        audio_layout = QFormLayout()
        self.audio_codec_combo = QComboBox()
        audio_layout.addRow("Audio Codec:", self.audio_codec_combo)
        self.a_bitrate_combo = QComboBox()
        self.a_bitrate_combo.addItems(self.audio_bitrate_options)
        self.a_bitrate_custom_edit = QLineEdit()
        self.a_bitrate_custom_edit.setPlaceholderText("e.g., 256k")
        self.a_bitrate_custom_edit.hide()
        self.a_bitrate_combo.currentTextChanged.connect(self.on_a_bitrate_changed)
        audio_layout.addRow("Audio Bitrate:", self.a_bitrate_combo)
        audio_layout.addRow(self.a_bitrate_custom_edit)
        self.audio_group.setLayout(audio_layout)
        self.layout.addWidget(self.audio_group)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def _populate_combo(self, combo, data_dict, filter_keys=None):
        current_selection = combo.currentData()
        combo.blockSignals(True)
        combo.clear()

        keys_to_show = filter_keys if filter_keys is not None else data_dict.keys()
        for codename in keys_to_show:
            if codename in data_dict:
                desc = data_dict[codename]
                display_name = self.display_ext_map.get(codename, codename) if combo is self.container_combo else codename
                combo.addItem(f"{desc} ({display_name})", codename)

        new_index = combo.findData(current_selection)
        combo.setCurrentIndex(new_index if new_index != -1 else 0)
        combo.blockSignals(False)

    def on_advanced_toggled(self, checked):
        self._populate_combo(self.container_combo, self.formats, None if checked else CONTAINER_PRESETS.keys())

        if not checked:
            mp4_index = self.container_combo.findData("mp4")
            if mp4_index != -1: self.container_combo.setCurrentIndex(mp4_index)

        self.on_container_changed(self.container_combo.currentIndex())

    def apply_preset(self, container_codename):
        preset = CONTAINER_PRESETS.get(container_codename, {})
        
        vcodec = preset.get('vcodec')
        self.video_group.setEnabled(vcodec is not None)
        if vcodec:
            vcodec_idx = self.video_codec_combo.findData(vcodec)
            self.video_codec_combo.setCurrentIndex(vcodec_idx if vcodec_idx != -1 else 0)

        v_bitrate = preset.get('v_bitrate')
        self.v_bitrate_combo.setEnabled(v_bitrate is not None)
        if v_bitrate:
            v_bitrate_idx = self.v_bitrate_combo.findText(v_bitrate)
            if v_bitrate_idx != -1: self.v_bitrate_combo.setCurrentIndex(v_bitrate_idx)
            else:
                self.v_bitrate_combo.setCurrentText("Custom...")
                self.v_bitrate_custom_edit.setText(v_bitrate)
        
        acodec = preset.get('acodec')
        self.audio_group.setEnabled(acodec is not None)
        if acodec:
            acodec_idx = self.audio_codec_combo.findData(acodec)
            self.audio_codec_combo.setCurrentIndex(acodec_idx if acodec_idx != -1 else 0)

        a_bitrate = preset.get('a_bitrate')
        self.a_bitrate_combo.setEnabled(a_bitrate is not None)
        if a_bitrate:
            a_bitrate_idx = self.a_bitrate_combo.findText(a_bitrate)
            if a_bitrate_idx != -1: self.a_bitrate_combo.setCurrentIndex(a_bitrate_idx)
            else:
                self.a_bitrate_combo.setCurrentText("Custom...")
                self.a_bitrate_custom_edit.setText(a_bitrate)

    def on_v_bitrate_changed(self, text):
        self.v_bitrate_custom_edit.setVisible(text == "Custom...")

    def on_a_bitrate_changed(self, text):
        self.a_bitrate_custom_edit.setVisible(text == "Custom...")

    def browse_output_path(self):
        container_codename = self.container_combo.currentData()
        all_formats_desc = [f"{desc} (*.{name})" for name, desc in self.formats.items()]
        filter_str = ";;".join(all_formats_desc)
        current_desc = self.formats.get(container_codename, "Custom Format")
        specific_filter = f"{current_desc} (*.{container_codename})"
        final_filter = f"{specific_filter};;{filter_str};;All Files (*)"
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Video As", self.path_edit.text(), final_filter)
        if path: self.path_edit.setText(path)

    def on_container_changed(self, index):
        if index == -1: return
        new_container_codename = self.container_combo.itemData(index)
        
        is_advanced = self.advanced_formats_checkbox.isChecked()
        preset = CONTAINER_PRESETS.get(new_container_codename)

        if not is_advanced and preset:
            v_filter = preset.get('allowed_vcodecs')
            a_filter = preset.get('allowed_acodecs')
            self._populate_combo(self.video_codec_combo, self.video_codecs, v_filter)
            self._populate_combo(self.audio_codec_combo, self.audio_codecs, a_filter)
        else:
            self._populate_combo(self.video_codec_combo, self.video_codecs)
            self._populate_combo(self.audio_codec_combo, self.audio_codecs)

        if new_container_codename:
            self.update_output_path_extension(new_container_codename)
            self.apply_preset(new_container_codename)

    def update_output_path_extension(self, new_container_codename):
        current_path = self.path_edit.text()
        if not current_path: return
        directory, filename = os.path.split(current_path)
        basename, _ = os.path.splitext(filename)
        ext = self.display_ext_map.get(new_container_codename, new_container_codename)
        new_path = os.path.join(directory, f"{basename}.{ext}")
        self.path_edit.setText(new_path)
        
    def get_export_settings(self):
        v_bitrate = self.v_bitrate_combo.currentText()
        if v_bitrate == "Custom...": v_bitrate = self.v_bitrate_custom_edit.text()

        a_bitrate = self.a_bitrate_combo.currentText()
        if a_bitrate == "Custom...": a_bitrate = self.a_bitrate_custom_edit.text()

        return {
            "output_path": self.path_edit.text(),
            "container": self.container_combo.currentData(),
            "vcodec": self.video_codec_combo.currentData() if self.video_group.isEnabled() else None,
            "v_bitrate": v_bitrate if self.v_bitrate_combo.isEnabled() else None,
            "acodec": self.audio_codec_combo.currentData() if self.audio_group.isEnabled() else None,
            "a_bitrate": a_bitrate if self.a_bitrate_combo.isEnabled() else None,
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
            "duration_ms": media_info['duration_ms'],
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
        self.setAcceptDrops(True)
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
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.main_window._add_media_files_to_project(file_paths)
            event.acceptProposedAction()
        else:
            event.ignore()

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
        self.last_export_path = None
        self.settings = {}
        self.settings_file = "settings.json"
        self.is_shutting_down = False
        self._load_settings()

        self.plugin_manager = PluginManager(self, wgp)
        self.plugin_manager.discover_and_load_plugins()

        # Project settings with defaults
        self.project_fps = 25.0
        self.project_width = 1280
        self.project_height = 720
        
        # Preview settings
        self.scale_to_fit = True
        self.current_preview_pixmap = None

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_display()

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

        # --- PREVIEW WIDGET SETUP (CHANGED) ---
        self.preview_scroll_area = QScrollArea()
        self.preview_scroll_area.setWidgetResizable(False) # Important for 1:1 scaling
        self.preview_scroll_area.setStyleSheet("background-color: black; border: 0px;")
        
        self.preview_widget = QLabel()
        self.preview_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_widget.setMinimumSize(640, 360)
        self.preview_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        
        self.preview_scroll_area.setWidget(self.preview_widget)
        self.splitter.addWidget(self.preview_scroll_area)

        self.timeline_widget = TimelineWidget(self.timeline, self.settings, self.project_fps, self)
        self.timeline_widget.setMinimumHeight(250)
        self.splitter.addWidget(self.timeline_widget)
        
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

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
            'preview': {'widget': self.preview_scroll_area, 'name': 'Video Preview', 'action': None},
            'timeline': {'widget': self.timeline_widget, 'name': 'Timeline', 'action': None},
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
        self.preview_widget.customContextMenuRequested.connect(self._show_preview_context_menu)

        self.timeline_widget.split_requested.connect(self.split_clip_at_playhead)
        self.timeline_widget.delete_clip_requested.connect(self.delete_clip)
        self.timeline_widget.delete_clips_requested.connect(self.delete_clips)
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

    def _show_preview_context_menu(self, pos):
        menu = QMenu(self)
        scale_action = QAction("Scale to Fit", self, checkable=True)
        scale_action.setChecked(self.scale_to_fit)
        scale_action.toggled.connect(self._toggle_scale_to_fit)
        menu.addAction(scale_action)
        menu.exec(self.preview_widget.mapToGlobal(pos))

    def _toggle_scale_to_fit(self, checked):
        self.scale_to_fit = checked
        self._update_preview_display()

    def on_timeline_changed_by_undo(self):
        self.prune_empty_tracks()
        self.timeline_widget.update()
        self.seek_preview(self.timeline_widget.playhead_pos_ms)
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

        playhead_pos = self.timeline_widget.playhead_pos_ms
        duration_ms = media_info['duration_ms']
        has_audio = media_info['has_audio']
        media_type = media_info['media_type']

        video_track = 1 if media_type in ['video', 'image'] else None
        audio_track = 1 if has_audio else None

        self._add_clip_to_timeline(
            source_path=file_path,
            timeline_start_ms=playhead_pos,
            duration_ms=duration_ms,
            media_type=media_type,
            clip_start_ms=0,
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
        else:
            return

        new_state = self._get_current_timeline_state()
        command = TimelineStateChangeCommand(f"Add {track_type.capitalize()} Track", self.timeline, *old_state, *new_state)

        self.undo_stack.blockSignals(True)
        self.undo_stack.push(command)
        self.undo_stack.blockSignals(False)

        self.update_undo_redo_actions()
        self.timeline_widget.update()
    
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

    def on_dock_visibility_changed(self, action, visible):
        if self.isMinimized():
            return
        action.setChecked(visible)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        new_action = QAction("&New Project", self); new_action.triggered.connect(self.new_project)
        open_action = QAction("&Open Project...", self); open_action.triggered.connect(self.open_project)
        self.recent_menu = file_menu.addMenu("Recent")
        save_action = QAction("&Save Project As...", self); save_action.triggered.connect(self.save_project_as)
        add_media_to_timeline_action = QAction("Add Media to &Timeline...", self)
        add_media_to_timeline_action.triggered.connect(self.add_media_to_timeline)
        add_media_action = QAction("&Add Media to Project...", self); add_media_action.triggered.connect(self.add_media_files)
        export_action = QAction("&Export Video...", self); export_action.triggered.connect(self.export_video)
        settings_action = QAction("Se&ttings...", self); settings_action.triggered.connect(self.open_settings_dialog)
        exit_action = QAction("E&xit", self); exit_action.triggered.connect(self.close)
        file_menu.addAction(new_action); file_menu.addAction(open_action); file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(add_media_to_timeline_action)
        file_menu.addAction(add_media_action)
        file_menu.addAction(export_action)
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
            if data['widget'] is self.preview_scroll_area or data['widget'] is self.timeline_widget: continue
            action = QAction(data['name'], self, checkable=True)
            if hasattr(data['widget'], 'visibilityChanged'):
                action.toggled.connect(data['widget'].setVisible)
                data['widget'].visibilityChanged.connect(lambda visible, a=action: self.on_dock_visibility_changed(a, visible))
            else:
                action.toggled.connect(lambda checked, k=key: self.toggle_widget_visibility(k, checked))

            data['action'] = action
            self.windows_menu.addAction(action)

    def _get_topmost_video_clip_at(self, time_ms):
        """Finds the video clip on the highest track at a specific time."""
        top_clip = None
        for c in self.timeline.clips:
            if c.track_type == 'video' and c.timeline_start_ms <= time_ms < c.timeline_end_ms:
                if top_clip is None or c.track_index > top_clip.track_index:
                    top_clip = c
        return top_clip

    def _start_playback_stream_at(self, time_ms):
        self._stop_playback_stream()
        clip = self._get_topmost_video_clip_at(time_ms)
        if not clip: return

        self.playback_clip = clip
        clip_time_sec = (time_ms - clip.timeline_start_ms + clip.clip_start_ms) / 1000.0
        w, h = self.project_width, self.project_height
        try:
            args = (ffmpeg.input(self.playback_clip.source_path, ss=f"{clip_time_sec:.6f}")
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

    def _get_media_properties(self, file_path):
        """Probes a file to get its media properties. Returns a dict or None."""
        if file_path in self.media_properties:
            return self.media_properties[file_path]
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            media_info = {}
            
            if file_ext in ['.png', '.jpg', '.jpeg']:
                img = QImage(file_path)
                media_info['media_type'] = 'image'
                media_info['duration_ms'] = 5000
                media_info['has_audio'] = False
                media_info['width'] = img.width()
                media_info['height'] = img.height()
            else:
                probe = ffmpeg.probe(file_path)
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

                if video_stream:
                    media_info['media_type'] = 'video'
                    duration_sec = float(video_stream.get('duration', probe['format'].get('duration', 0)))
                    media_info['duration_ms'] = int(duration_sec * 1000)
                    media_info['has_audio'] = audio_stream is not None
                    media_info['width'] = int(video_stream['width'])
                    media_info['height'] = int(video_stream['height'])
                    if 'r_frame_rate' in video_stream and video_stream['r_frame_rate'] != '0/0':
                        num, den = map(int, video_stream['r_frame_rate'].split('/'))
                        if den > 0: media_info['fps'] = num / den
                elif audio_stream:
                    media_info['media_type'] = 'audio'
                    duration_sec = float(audio_stream.get('duration', probe['format'].get('duration', 0)))
                    media_info['duration_ms'] = int(duration_sec * 1000)
                    media_info['has_audio'] = True
                else:
                    return None
            
            return media_info
        except Exception as e:
            print(f"Failed to probe file {os.path.basename(file_path)}: {e}")
            return None

    def _update_project_properties_from_clip(self, source_path):
        try:
            media_info = self._get_media_properties(source_path)
            if not media_info or media_info['media_type'] not in ['video', 'image']:
                return False

            new_w = media_info.get('width')
            new_h = media_info.get('height')
            new_fps = media_info.get('fps')
            
            if not new_w or not new_h:
                return False

            is_first_video = not any(c.media_type in ['video', 'image'] for c in self.timeline.clips if c.source_path != source_path)
            
            if is_first_video:
                self.project_width = new_w
                self.project_height = new_h
                if new_fps: self.project_fps = new_fps
                self.timeline_widget.set_project_fps(self.project_fps)
                print(f"Project properties set from first clip: {self.project_width}x{self.project_height} @ {self.project_fps:.2f} FPS")
            else:
                current_area = self.project_width * self.project_height
                new_area = new_w * new_h
                if new_area > current_area:
                    self.project_width = new_w
                    self.project_height = new_h
                    print(f"Project resolution updated to: {self.project_width}x{self.project_height}")
            return True
        except Exception as e:
            print(f"Could not probe for project properties: {e}")
        return False

    def _probe_for_drag(self, file_path):
        return self._get_media_properties(file_path)

    def get_frame_data_at_time(self, time_ms):
        clip_at_time = self._get_topmost_video_clip_at(time_ms)
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
                clip_time_sec = (time_ms - clip_at_time.timeline_start_ms + clip_at_time.clip_start_ms) / 1000.0
                out, _ = (
                    ffmpeg
                    .input(clip_at_time.source_path, ss=f"{clip_time_sec:.6f}")
                    .filter('scale', w, h, force_original_aspect_ratio='decrease')
                    .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
                    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
            return (out, self.project_width, self.project_height)
        except ffmpeg.Error as e:
            print(f"Error extracting frame data: {e.stderr}")
            return (None, 0, 0)

    def get_frame_at_time(self, time_ms):
        clip_at_time = self._get_topmost_video_clip_at(time_ms)
        if not clip_at_time: return None
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
                clip_time_sec = (time_ms - clip_at_time.timeline_start_ms + clip_at_time.clip_start_ms) / 1000.0
                out, _ = (ffmpeg.input(clip_at_time.source_path, ss=f"{clip_time_sec:.6f}")
                          .filter('scale', w, h, force_original_aspect_ratio='decrease')
                          .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black')
                          .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                          .run(capture_stdout=True, quiet=True))
            
            image = QImage(out, self.project_width, self.project_height, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(image)
        except ffmpeg.Error as e: print(f"Error extracting frame: {e.stderr}"); return None

    def seek_preview(self, time_ms):
        self._stop_playback_stream()
        self.timeline_widget.playhead_pos_ms = int(time_ms)
        self.timeline_widget.update()
        self.current_preview_pixmap = self.get_frame_at_time(time_ms)
        self._update_preview_display()

    def _update_preview_display(self):
        pixmap_to_show = self.current_preview_pixmap
        if not pixmap_to_show:
            pixmap_to_show = QPixmap(self.project_width, self.project_height)
            pixmap_to_show.fill(QColor("black"))

        if self.scale_to_fit:
            # When fitting, we want the label to resize with the scroll area
            self.preview_scroll_area.setWidgetResizable(True)
            scaled_pixmap = pixmap_to_show.scaled(
                self.preview_scroll_area.viewport().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_widget.setPixmap(scaled_pixmap)
        else:
            # For 1:1, the label takes the size of the pixmap, and the scroll area handles overflow
            self.preview_scroll_area.setWidgetResizable(False)
            self.preview_widget.setPixmap(pixmap_to_show)
            self.preview_widget.adjustSize()
        

    def toggle_playback(self):
        if self.playback_timer.isActive(): self.playback_timer.stop(); self._stop_playback_stream(); self.play_pause_button.setText("Play")
        else:
            if not self.timeline.clips: return
            if self.timeline_widget.playhead_pos_ms >= self.timeline.get_total_duration(): self.timeline_widget.playhead_pos_ms = 0
            self.playback_timer.start(int(1000 / self.project_fps)); self.play_pause_button.setText("Pause")

    def stop_playback(self): self.playback_timer.stop(); self._stop_playback_stream(); self.play_pause_button.setText("Play"); self.seek_preview(0)
    def step_frame(self, direction):
        if not self.timeline.clips: return
        self.playback_timer.stop(); self.play_pause_button.setText("Play"); self._stop_playback_stream()
        frame_duration_ms = 1000.0 / self.project_fps
        new_time = self.timeline_widget.playhead_pos_ms + (direction * frame_duration_ms)
        self.seek_preview(int(max(0, min(new_time, self.timeline.get_total_duration()))))

    def advance_playback_frame(self):
        frame_duration_ms = 1000.0 / self.project_fps
        new_time_ms = self.timeline_widget.playhead_pos_ms + frame_duration_ms
        if new_time_ms > self.timeline.get_total_duration(): self.stop_playback(); return
        
        self.timeline_widget.playhead_pos_ms = round(new_time_ms)
        self.timeline_widget.update()
        
        clip_at_new_time = self._get_topmost_video_clip_at(new_time_ms)
        
        if not clip_at_new_time:
            self._stop_playback_stream()
            self.current_preview_pixmap = None
            self._update_preview_display()
            return

        if clip_at_new_time.media_type == 'image':
            if self.playback_clip is None or self.playback_clip.id != clip_at_new_time.id:
                self._stop_playback_stream()
                self.playback_clip = clip_at_new_time
                self.current_preview_pixmap = self.get_frame_at_time(new_time_ms)
                self._update_preview_display()
            return

        if self.playback_clip is None or self.playback_clip.id != clip_at_new_time.id:
            self._start_playback_stream_at(new_time_ms)
        
        if self.playback_process:
            frame_size = self.project_width * self.project_height * 3
            frame_bytes = self.playback_process.stdout.read(frame_size)
            if len(frame_bytes) == frame_size:
                image = QImage(frame_bytes, self.project_width, self.project_height, QImage.Format.Format_RGB888)
                self.current_preview_pixmap = QPixmap.fromImage(image)
                self._update_preview_display()
            else:
                self._stop_playback_stream()

    def _load_settings(self):
        self.settings_file_was_loaded = False
        defaults = {"window_visibility": {"project_media": False}, "splitter_state": None, "enabled_plugins": [], "recent_files": [], "confirm_on_exit": True, "default_export_path": ""}
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
            is_visible = visibility_settings.get(key, False if key == 'project_media' else True)
            if data['widget'] is not self.preview_scroll_area:
                data['widget'].setVisible(is_visible)
            if data['action']: data['action'].setChecked(is_visible)
        splitter_state = self.settings.get("splitter_state")
        if splitter_state: self.splitter.restoreState(QByteArray.fromHex(splitter_state.encode('ascii')))

    def on_splitter_moved(self, pos, index): 
        self.splitter_save_timer.start(500)
        self._update_preview_display()
    
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
        self.last_export_path = None
        self.project_fps = 25.0
        self.project_width = 1280
        self.project_height = 720
        self.timeline_widget.set_project_fps(self.project_fps)
        self.timeline_widget.clear_all_regions()
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
            "clips": [{"source_path": c.source_path, "timeline_start_ms": c.timeline_start_ms, "clip_start_ms": c.clip_start_ms, "duration_ms": c.duration_ms, "track_index": c.track_index, "track_type": c.track_type, "media_type": c.media_type, "group_id": c.group_id} for c in self.timeline.clips],
            "selection_regions": self.timeline_widget.selection_regions,
            "last_export_path": self.last_export_path,
            "settings": {
                "num_video_tracks": self.timeline.num_video_tracks, 
                "num_audio_tracks": self.timeline.num_audio_tracks,
                "project_width": self.project_width,
                "project_height": self.project_height,
                "project_fps": self.project_fps
            }
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
            
            project_settings = project_data.get("settings", {})
            self.timeline.num_video_tracks = project_settings.get("num_video_tracks", 1)
            self.timeline.num_audio_tracks = project_settings.get("num_audio_tracks", 1)
            self.project_width = project_settings.get("project_width", 1280)
            self.project_height = project_settings.get("project_height", 720)
            self.project_fps = project_settings.get("project_fps", 25.0)
            self.timeline_widget.set_project_fps(self.project_fps)
            self.last_export_path = project_data.get("last_export_path")
            self.timeline_widget.selection_regions = project_data.get("selection_regions", [])

            media_pool_paths = project_data.get("media_pool", [])
            for p in media_pool_paths: self._add_media_to_pool(p)
            
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
            
            self.current_project_path = path
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
        if file_path in self.media_pool:
            return True
        
        self.status_label.setText(f"Probing {os.path.basename(file_path)}..."); QApplication.processEvents()
        
        media_info = self._get_media_properties(file_path)
        
        if media_info:
            self.media_properties[file_path] = media_info
            self.media_pool.append(file_path)
            self.project_media_widget.add_media_item(file_path)
            self.status_label.setText(f"Added {os.path.basename(file_path)} to project.")
            return True
        else:
            self.status_label.setText(f"Error probing file: {os.path.basename(file_path)}")
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

    def _add_media_files_to_project(self, file_paths):
        if not file_paths:
            return []

        self.media_dock.show()
        added_files = []

        for file_path in file_paths:
            self._update_project_properties_from_clip(file_path)
            if self._add_media_to_pool(file_path):
                added_files.append(file_path)
        
        return added_files

    def add_media_to_timeline(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Add Media to Timeline", "", "All Supported Files (*.mp4 *.mov *.avi *.png *.jpg *.jpeg *.mp3 *.wav);;Video Files (*.mp4 *.mov *.avi);;Image Files (*.png *.jpg *.jpeg);;Audio Files (*.mp3 *.wav)")
        if not file_paths:
            return

        added_files = self._add_media_files_to_project(file_paths)
        if not added_files:
            return

        playhead_pos = self.timeline_widget.playhead_pos_ms

        def add_clips_action():
            for file_path in added_files:
                media_info = self.media_properties.get(file_path)
                if not media_info: continue

                duration_ms = media_info['duration_ms']
                media_type = media_info['media_type']
                has_audio = media_info['has_audio']

                clip_start_time = playhead_pos
                clip_end_time = playhead_pos + duration_ms

                video_track_index = None
                audio_track_index = None

                if media_type in ['video', 'image']:
                    for i in range(1, self.timeline.num_video_tracks + 2):
                        is_occupied = any(
                            c.timeline_start_ms < clip_end_time and c.timeline_end_ms > clip_start_time
                            for c in self.timeline.clips if c.track_type == 'video' and c.track_index == i
                        )
                        if not is_occupied:
                            video_track_index = i
                            break
                
                if has_audio:
                    for i in range(1, self.timeline.num_audio_tracks + 2):
                        is_occupied = any(
                            c.timeline_start_ms < clip_end_time and c.timeline_end_ms > clip_start_time
                            for c in self.timeline.clips if c.track_type == 'audio' and c.track_index == i
                        )
                        if not is_occupied:
                            audio_track_index = i
                            break
                
                group_id = str(uuid.uuid4())
                if video_track_index is not None:
                    if video_track_index > self.timeline.num_video_tracks:
                        self.timeline.num_video_tracks = video_track_index
                    video_clip = TimelineClip(file_path, clip_start_time, 0, duration_ms, video_track_index, 'video', media_type, group_id)
                    self.timeline.add_clip(video_clip)
                
                if audio_track_index is not None:
                    if audio_track_index > self.timeline.num_audio_tracks:
                        self.timeline.num_audio_tracks = audio_track_index
                    audio_clip = TimelineClip(file_path, clip_start_time, 0, duration_ms, audio_track_index, 'audio', media_type, group_id)
                    self.timeline.add_clip(audio_clip)

            self.status_label.setText(f"Added {len(added_files)} file(s) to timeline.")

        self._perform_complex_timeline_change("Add Media to Timeline", add_clips_action)

    def add_media_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Media Files", "", "All Supported Files (*.mp4 *.mov *.avi *.png *.jpg *.jpeg *.mp3 *.wav);;Video Files (*.mp4 *.mov *.avi);;Image Files (*.png *.jpg *.jpeg);;Audio Files (*.mp3 *.wav)")
        if file_paths:
            self._add_media_files_to_project(file_paths)

    def _add_clip_to_timeline(self, source_path, timeline_start_ms, duration_ms, media_type, clip_start_ms=0, video_track_index=None, audio_track_index=None):
        if media_type in ['video', 'image']:
            self._update_project_properties_from_clip(source_path)

        old_state = self._get_current_timeline_state()
        group_id = str(uuid.uuid4())
        
        if video_track_index is not None:
             if video_track_index > self.timeline.num_video_tracks:
                 self.timeline.num_video_tracks = video_track_index
             video_clip = TimelineClip(source_path, timeline_start_ms, clip_start_ms, duration_ms, video_track_index, 'video', media_type, group_id)
             self.timeline.add_clip(video_clip)
        
        if audio_track_index is not None:
             if audio_track_index > self.timeline.num_audio_tracks:
                 self.timeline.num_audio_tracks = audio_track_index
             audio_clip = TimelineClip(source_path, timeline_start_ms, clip_start_ms, duration_ms, audio_track_index, 'audio', media_type, group_id)
             self.timeline.add_clip(audio_clip)

        new_state = self._get_current_timeline_state()
        command = TimelineStateChangeCommand("Add Clip", self.timeline, *old_state, *new_state)
        command.undo()
        self.undo_stack.push(command)

    def _split_at_time(self, clip_to_split, time_ms, new_group_id=None):
        if not (clip_to_split.timeline_start_ms < time_ms < clip_to_split.timeline_end_ms): return False
        split_point = time_ms - clip_to_split.timeline_start_ms
        orig_dur = clip_to_split.duration_ms
        group_id_for_new_clip = new_group_id if new_group_id is not None else clip_to_split.group_id
        
        new_clip = TimelineClip(clip_to_split.source_path, time_ms, clip_to_split.clip_start_ms + split_point, orig_dur - split_point, clip_to_split.track_index, clip_to_split.track_type, clip_to_split.media_type, group_id_for_new_clip)
        clip_to_split.duration_ms = split_point
        self.timeline.add_clip(new_clip)
        return True

    def split_clip_at_playhead(self, clip_to_split=None):
        playhead_time = self.timeline_widget.playhead_pos_ms
        if not clip_to_split:
            clips_at_playhead = [c for c in self.timeline.clips if c.timeline_start_ms < playhead_time < c.timeline_end_ms]
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
        self.delete_clips([clip_to_delete])

    def delete_clips(self, clips_to_delete):
        if not clips_to_delete: return

        old_state = self._get_current_timeline_state()

        ids_to_remove = set()
        for clip in clips_to_delete:
            ids_to_remove.add(clip.id)
            linked_clips = [c for c in self.timeline.clips if c.group_id == clip.group_id and c.id != clip.id]
            for lc in linked_clips:
                ids_to_remove.add(lc.id)
        
        self.timeline.clips = [c for c in self.timeline.clips if c.id not in ids_to_remove]
        self.timeline_widget.selected_clips.clear()
        
        new_state = self._get_current_timeline_state()
        command = TimelineStateChangeCommand(f"Delete {len(clips_to_delete)} Clip(s)", self.timeline, *old_state, *new_state)
        command.undo()
        self.undo_stack.push(command)
        self.prune_empty_tracks()
        self.timeline_widget.update()

    def unlink_clip_pair(self, clip_to_unlink):
        old_state = self._get_current_timeline_state()

        linked_clip = next((c for c in self.timeline.clips if c.group_id == clip_to_unlink.group_id and c.id != clip_to_unlink.id), None)
        
        if linked_clip:
            clip_to_unlink.group_id = str(uuid.uuid4())
            linked_clip.group_id = str(uuid.uuid4())
            
            new_state = self._get_current_timeline_state()
            command = TimelineStateChangeCommand("Unlink Clips", self.timeline, *old_state, *new_state)
            command.undo()
            self.undo_stack.push(command)
            self.status_label.setText("Clips unlinked.")
        else:
            self.status_label.setText("Could not find a clip to unlink.")

    def relink_clip_audio(self, video_clip):
        def action():
            media_info = self.media_properties.get(video_clip.source_path)
            if not media_info or not media_info.get('has_audio'):
                self.status_label.setText("Source media has no audio to relink.")
                return

            target_audio_track = 1
            new_audio_start = video_clip.timeline_start_ms
            new_audio_end = video_clip.timeline_end_ms
            
            conflicting_clips = [
                c for c in self.timeline.clips
                if c.track_type == 'audio' and c.track_index == target_audio_track and
                c.timeline_start_ms < new_audio_end and c.timeline_end_ms > new_audio_start
            ]

            for conflict_clip in conflicting_clips:
                found_spot = False
                for check_track_idx in range(target_audio_track + 1, self.timeline.num_audio_tracks + 2):
                    is_occupied = any(
                        other.timeline_start_ms < conflict_clip.timeline_end_ms and other.timeline_end_ms > conflict_clip.timeline_start_ms
                        for other in self.timeline.clips
                        if other.id != conflict_clip.id and other.track_type == 'audio' and other.track_index == check_track_idx
                    )
                    
                    if not is_occupied:
                        if check_track_idx > self.timeline.num_audio_tracks:
                            self.timeline.num_audio_tracks = check_track_idx
                        
                        conflict_clip.track_index = check_track_idx
                        found_spot = True
                        break
            
            new_audio_clip = TimelineClip(
                source_path=video_clip.source_path,
                timeline_start_ms=video_clip.timeline_start_ms,
                clip_start_ms=video_clip.clip_start_ms,
                duration_ms=video_clip.duration_ms,
                track_index=target_audio_track,
                track_type='audio',
                media_type=video_clip.media_type,
                group_id=video_clip.group_id
            )
            self.timeline.add_clip(new_audio_clip)
            self.status_label.setText("Audio relinked.")

        self._perform_complex_timeline_change("Relink Audio", action)

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
            start_ms, end_ms = region
            clips = list(self.timeline.clips)
            for clip in clips: self._split_at_time(clip, end_ms)
            for clip in clips: self._split_at_time(clip, start_ms)
            self.timeline_widget.clear_region(region)
        self._perform_complex_timeline_change("Split Region", action)

    def on_split_all_regions(self, regions):
        def action():
            split_points = set()
            for start, end in regions:
                split_points.add(start)
                split_points.add(end)

            for point in sorted(list(split_points)):
                group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_ms < point < c.timeline_end_ms}
                new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                for clip in list(self.timeline.clips):
                    if clip.group_id in new_group_ids:
                        self._split_at_time(clip, point, new_group_ids[clip.group_id])
            self.timeline_widget.clear_all_regions()
        self._perform_complex_timeline_change("Split All Regions", action)

    def on_join_region(self, region):
        def action():
            start_ms, end_ms = region
            duration_to_remove = end_ms - start_ms
            if duration_to_remove <= 10: return

            for point in [start_ms, end_ms]:
                 group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_ms < point < c.timeline_end_ms}
                 new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                 for clip in list(self.timeline.clips):
                     if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])

            clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_ms >= start_ms and c.timeline_start_ms < end_ms]
            for clip in clips_to_remove: self.timeline.clips.remove(clip)

            for clip in self.timeline.clips:
                if clip.timeline_start_ms >= end_ms:
                    clip.timeline_start_ms -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_ms)
            self.timeline_widget.clear_region(region)
        self._perform_complex_timeline_change("Join Region", action)

    def on_join_all_regions(self, regions):
        def action():
            for region in sorted(regions, key=lambda r: r[0], reverse=True):
                start_ms, end_ms = region
                duration_to_remove = end_ms - start_ms
                if duration_to_remove <= 10: continue

                for point in [start_ms, end_ms]:
                    group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_ms < point < c.timeline_end_ms}
                    new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                    for clip in list(self.timeline.clips):
                        if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])
                clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_ms >= start_ms and c.timeline_start_ms < end_ms]
                for clip in clips_to_remove:
                    try: self.timeline.clips.remove(clip)
                    except ValueError: pass 

                for clip in self.timeline.clips:
                    if clip.timeline_start_ms >= end_ms:
                        clip.timeline_start_ms -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_ms)
            self.timeline_widget.clear_all_regions()
        self._perform_complex_timeline_change("Join All Regions", action)

    def on_delete_region(self, region):
        def action():
            start_ms, end_ms = region
            duration_to_remove = end_ms - start_ms
            if duration_to_remove <= 10: return

            for point in [start_ms, end_ms]:
                 group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_ms < point < c.timeline_end_ms}
                 new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                 for clip in list(self.timeline.clips):
                     if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])

            clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_ms >= start_ms and c.timeline_start_ms < end_ms]
            for clip in clips_to_remove: self.timeline.clips.remove(clip)

            for clip in self.timeline.clips:
                if clip.timeline_start_ms >= end_ms:
                    clip.timeline_start_ms -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_ms)
            self.timeline_widget.clear_region(region)
        self._perform_complex_timeline_change("Delete Region", action)

    def on_delete_all_regions(self, regions):
        def action():
            for region in sorted(regions, key=lambda r: r[0], reverse=True):
                start_ms, end_ms = region
                duration_to_remove = end_ms - start_ms
                if duration_to_remove <= 10: continue

                for point in [start_ms, end_ms]:
                    group_ids_at_point = {c.group_id for c in self.timeline.clips if c.timeline_start_ms < point < c.timeline_end_ms}
                    new_group_ids = {gid: str(uuid.uuid4()) for gid in group_ids_at_point}
                    for clip in list(self.timeline.clips):
                        if clip.group_id in new_group_ids: self._split_at_time(clip, point, new_group_ids[clip.group_id])

                clips_to_remove = [c for c in self.timeline.clips if c.timeline_start_ms >= start_ms and c.timeline_start_ms < end_ms]
                for clip in clips_to_remove:
                    try: self.timeline.clips.remove(clip)
                    except ValueError: pass

                for clip in self.timeline.clips:
                    if clip.timeline_start_ms >= end_ms:
                        clip.timeline_start_ms -= duration_to_remove
            
            self.timeline.clips.sort(key=lambda c: c.timeline_start_ms)
            self.timeline_widget.clear_all_regions()
        self._perform_complex_timeline_change("Delete All Regions", action)


    def export_video(self):
        if not self.timeline.clips:
            self.status_label.setText("Timeline is empty.")
            return

        default_path = ""
        if self.last_export_path and os.path.isdir(os.path.dirname(self.last_export_path)):
            default_path = self.last_export_path
        elif self.settings.get("default_export_path") and os.path.isdir(self.settings.get("default_export_path")):
            proj_basename = "output"
            if self.current_project_path:
                _, proj_file = os.path.split(self.current_project_path)
                proj_basename, _ = os.path.splitext(proj_file)
            
            default_path = os.path.join(self.settings["default_export_path"], f"{proj_basename}_export.mp4")
        elif self.current_project_path:
            proj_dir, proj_file = os.path.split(self.current_project_path)
            proj_basename, _ = os.path.splitext(proj_file)
            default_path = os.path.join(proj_dir, f"{proj_basename}_export.mp4")
        else:
            default_path = "output.mp4"

        default_path = os.path.normpath(default_path)

        dialog = ExportDialog(default_path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.status_label.setText("Export canceled.")
            return

        settings = dialog.get_export_settings()
        output_path = settings["output_path"]
        if not output_path:
            self.status_label.setText("Export failed: No output path specified.")
            return

        self.last_export_path = output_path

        total_dur_ms = self.timeline.get_total_duration()
        total_dur_sec = total_dur_ms / 1000.0
        w, h, fr_str = self.project_width, self.project_height, str(self.project_fps)
        sample_rate, channel_layout = '44100', 'stereo'

        video_stream = ffmpeg.input(f'color=c=black:s={w}x{h}:r={fr_str}:d={total_dur_sec}', f='lavfi')

        all_video_clips = sorted(
            [c for c in self.timeline.clips if c.track_type == 'video'],
            key=lambda c: c.track_index
        )
        input_nodes = {}
        for clip in all_video_clips:
            if clip.source_path not in input_nodes:
                if clip.media_type == 'image':
                    input_nodes[clip.source_path] = ffmpeg.input(clip.source_path, loop=1, framerate=self.project_fps)
                else:
                    input_nodes[clip.source_path] = ffmpeg.input(clip.source_path)
            
            clip_source_node = input_nodes[clip.source_path]
            clip_duration_sec = clip.duration_ms / 1000.0
            clip_start_sec = clip.clip_start_ms / 1000.0
            timeline_start_sec = clip.timeline_start_ms / 1000.0
            timeline_end_sec = clip.timeline_end_ms / 1000.0

            if clip.media_type == 'image':
                segment_stream = (clip_source_node.video.trim(duration=clip_duration_sec).setpts('PTS-STARTPTS'))
            else:
                segment_stream = (clip_source_node.video.trim(start=clip_start_sec, duration=clip_duration_sec).setpts('PTS-STARTPTS'))

            processed_segment = (segment_stream.filter('scale', w, h, force_original_aspect_ratio='decrease').filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', 'black'))
            video_stream = ffmpeg.overlay(video_stream, processed_segment, enable=f'between(t,{timeline_start_sec},{timeline_end_sec})')

        final_video = video_stream.filter('format', pix_fmts='yuv420p').filter('fps', fps=self.project_fps)

        track_audio_streams = []
        for i in range(self.timeline.num_audio_tracks):
            track_clips = sorted([c for c in self.timeline.clips if c.track_type == 'audio' and c.track_index == i + 1], key=lambda c: c.timeline_start_ms)
            if not track_clips: continue
            track_segments = []
            last_end_ms = track_clips[0].timeline_start_ms
            for clip in track_clips:
                if clip.source_path not in input_nodes:
                    input_nodes[clip.source_path] = ffmpeg.input(clip.source_path)
                gap_ms = clip.timeline_start_ms - last_end_ms
                if gap_ms > 10: 
                    track_segments.append(ffmpeg.input(f'anullsrc=r={sample_rate}:cl={channel_layout}:d={gap_ms/1000.0}', f='lavfi'))
                
                clip_start_sec = clip.clip_start_ms / 1000.0
                clip_duration_sec = clip.duration_ms / 1000.0
                a_seg = input_nodes[clip.source_path].audio.filter('atrim', start=clip_start_sec, duration=clip_duration_sec).filter('asetpts', 'PTS-STARTPTS')
                track_segments.append(a_seg)
                last_end_ms = clip.timeline_end_ms
            track_audio_streams.append(ffmpeg.concat(*track_segments, v=0, a=1).filter('adelay', f'{int(track_clips[0].timeline_start_ms)}ms', all=True))
        
        if track_audio_streams:
            final_audio = ffmpeg.filter(track_audio_streams, 'amix', inputs=len(track_audio_streams), duration='longest')
        else:
            final_audio = ffmpeg.input(f'anullsrc=r={sample_rate}:cl={channel_layout}:d={total_dur_sec}', f='lavfi')

        output_args = {'vcodec': settings['vcodec'], 'acodec': settings['acodec'], 'pix_fmt': 'yuv420p'}
        if settings['v_bitrate']: output_args['b:v'] = settings['v_bitrate']
        if settings['a_bitrate']: output_args['b:a'] = settings['a_bitrate']
        
        try:
            ffmpeg_cmd = ffmpeg.output(final_video, final_audio, output_path, **output_args).overwrite_output().compile()
            self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.status_label.setText("Exporting...")
            self.export_thread = QThread()
            self.export_worker = ExportWorker(ffmpeg_cmd, total_dur_ms)
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
        dock.visibilityChanged.connect(lambda visible, a=action: self.on_dock_visibility_changed(a, visible))
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