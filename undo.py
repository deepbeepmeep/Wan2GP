import copy
from PyQt6.QtCore import QObject, pyqtSignal

class UndoCommand:
    def __init__(self, description=""):
        self.description = description

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError

class CompositeCommand(UndoCommand):
    def __init__(self, description, commands):
        super().__init__(description)
        self.commands = commands

    def undo(self):
        for cmd in reversed(self.commands):
            cmd.undo()

    def redo(self):
        for cmd in self.commands:
            cmd.redo()

class UndoStack(QObject):
    history_changed = pyqtSignal()
    timeline_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.undo_stack = []
        self.redo_stack = []

    def push(self, command):
        self.undo_stack.append(command)
        self.redo_stack.clear()
        command.redo()
        self.history_changed.emit()
        self.timeline_changed.emit()

    def undo(self):
        if not self.can_undo():
            return
        command = self.undo_stack.pop()
        self.redo_stack.append(command)
        command.undo()
        self.history_changed.emit()
        self.timeline_changed.emit()

    def redo(self):
        if not self.can_redo():
            return
        command = self.redo_stack.pop()
        self.undo_stack.append(command)
        command.redo()
        self.history_changed.emit()
        self.timeline_changed.emit()

    def can_undo(self):
        return bool(self.undo_stack)

    def can_redo(self):
        return bool(self.redo_stack)
    
    def undo_text(self):
        return self.undo_stack[-1].description if self.can_undo() else ""

    def redo_text(self):
        return self.redo_stack[-1].description if self.can_redo() else ""

class TimelineStateChangeCommand(UndoCommand):
    def __init__(self, description, timeline_model, old_clips, old_v_tracks, old_a_tracks, new_clips, new_v_tracks, new_a_tracks):
        super().__init__(description)
        self.timeline = timeline_model
        self.old_clips_state = old_clips
        self.old_v_tracks = old_v_tracks
        self.old_a_tracks = old_a_tracks
        self.new_clips_state = new_clips
        self.new_v_tracks = new_v_tracks
        self.new_a_tracks = new_a_tracks
    
    def undo(self):
        self.timeline.clips = self.old_clips_state
        self.timeline.num_video_tracks = self.old_v_tracks
        self.timeline.num_audio_tracks = self.old_a_tracks
    
    def redo(self):
        self.timeline.clips = self.new_clips_state
        self.timeline.num_video_tracks = self.new_v_tracks
        self.timeline.num_audio_tracks = self.new_a_tracks


class MoveClipsCommand(UndoCommand):
    def __init__(self, description, timeline_model, move_data):
        super().__init__(description)
        self.timeline = timeline_model
        self.move_data = move_data

    def _apply_state(self, state_key_prefix):
        for data in self.move_data:
            clip_id = data['clip_id']
            clip = next((c for c in self.timeline.clips if c.id == clip_id), None)
            if clip:
                clip.timeline_start_sec = data[f'{state_key_prefix}_start']
                clip.track_index = data[f'{state_key_prefix}_track']
        self.timeline.clips.sort(key=lambda c: c.timeline_start_sec)

    def undo(self):
        self._apply_state('old')

    def redo(self):
        self._apply_state('new')