"""
Storyboard data management - handles project persistence and scene management.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class Transition:
    """Transition between scenes"""
    type: str = "cut"  # cut, crossfade, fade_black, wipe_left, wipe_right
    duration_frames: int = 0


@dataclass
class Scene:
    """A single scene in the storyboard"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    order: int = 0
    name: str = "Untitled Scene"
    prompt: str = ""
    negative_prompt: str = ""
    duration_frames: int = 81  # ~5 seconds at 16fps
    reference_image: Optional[str] = None
    settings_override: Dict[str, Any] = field(default_factory=dict)
    transition_to_next: Transition = field(default_factory=Transition)
    status: str = "pending"  # pending, queued, generating, complete, failed
    output_path: Optional[str] = None
    thumbnail: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "Scene":
        if "transition_to_next" in data and isinstance(data["transition_to_next"], dict):
            data["transition_to_next"] = Transition(**data["transition_to_next"])
        return cls(**data)


@dataclass
class GlobalSettings:
    """Global settings applied to all scenes"""
    resolution: str = "1280x720"
    fps: int = 16
    model_type: str = "wan2.1-t2v-14B"
    style_lora: Optional[str] = None
    style_lora_weight: float = 1.0
    seed_mode: str = "fixed"  # fixed, increment, random
    seed: int = 42
    guidance_scale: float = 7.5
    num_inference_steps: int = 30

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "GlobalSettings":
        return cls(**data)


@dataclass
class StoryboardProject:
    """A complete storyboard project"""
    version: str = "1.0.0"
    name: str = "Untitled Project"
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    global_settings: GlobalSettings = field(default_factory=GlobalSettings)
    scenes: List[Scene] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "name": self.name,
            "created": self.created,
            "modified": self.modified,
            "global_settings": self.global_settings.to_dict(),
            "scenes": [s.to_dict() for s in self.scenes]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StoryboardProject":
        global_settings = GlobalSettings.from_dict(data.get("global_settings", {}))
        scenes = [Scene.from_dict(s) for s in data.get("scenes", [])]
        return cls(
            version=data.get("version", "1.0.0"),
            name=data.get("name", "Untitled Project"),
            created=data.get("created", datetime.now().isoformat()),
            modified=data.get("modified", datetime.now().isoformat()),
            global_settings=global_settings,
            scenes=scenes
        )

    def add_scene(self, scene: Optional[Scene] = None) -> Scene:
        """Add a new scene to the project"""
        if scene is None:
            scene = Scene()
        scene.order = len(self.scenes)
        scene.name = f"Scene {scene.order + 1}"
        self.scenes.append(scene)
        self.modified = datetime.now().isoformat()
        return scene

    def remove_scene(self, scene_id: str) -> bool:
        """Remove a scene by ID"""
        for i, scene in enumerate(self.scenes):
            if scene.id == scene_id:
                self.scenes.pop(i)
                self._reorder_scenes()
                self.modified = datetime.now().isoformat()
                return True
        return False

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """Get a scene by ID"""
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        return None

    def update_scene(self, scene_id: str, **kwargs) -> bool:
        """Update scene properties"""
        scene = self.get_scene(scene_id)
        if scene is None:
            return False

        for key, value in kwargs.items():
            if hasattr(scene, key):
                setattr(scene, key, value)

        self.modified = datetime.now().isoformat()
        return True

    def move_scene(self, scene_id: str, new_order: int) -> bool:
        """Move a scene to a new position"""
        scene = self.get_scene(scene_id)
        if scene is None:
            return False

        old_order = scene.order
        if old_order == new_order:
            return True

        # Remove from old position
        self.scenes.pop(old_order)

        # Insert at new position
        new_order = max(0, min(new_order, len(self.scenes)))
        self.scenes.insert(new_order, scene)

        self._reorder_scenes()
        self.modified = datetime.now().isoformat()
        return True

    def duplicate_scene(self, scene_id: str) -> Optional[Scene]:
        """Duplicate a scene"""
        original = self.get_scene(scene_id)
        if original is None:
            return None

        new_scene = Scene(
            name=f"{original.name} (copy)",
            prompt=original.prompt,
            negative_prompt=original.negative_prompt,
            duration_frames=original.duration_frames,
            reference_image=original.reference_image,
            settings_override=original.settings_override.copy(),
            transition_to_next=Transition(
                type=original.transition_to_next.type,
                duration_frames=original.transition_to_next.duration_frames
            )
        )

        # Insert after original
        new_scene.order = original.order + 1
        self.scenes.insert(new_scene.order, new_scene)
        self._reorder_scenes()
        self.modified = datetime.now().isoformat()
        return new_scene

    def _reorder_scenes(self):
        """Reindex scene order after changes"""
        for i, scene in enumerate(self.scenes):
            scene.order = i

    def get_total_duration_frames(self) -> int:
        """Get total duration in frames (excluding transitions)"""
        return sum(s.duration_frames for s in self.scenes)

    def get_total_duration_seconds(self) -> float:
        """Get total duration in seconds"""
        fps = self.global_settings.fps
        return self.get_total_duration_frames() / fps

    def get_completed_scenes(self) -> List[Scene]:
        """Get all completed scenes in order"""
        return [s for s in sorted(self.scenes, key=lambda x: x.order)
                if s.status == "complete" and s.output_path]

    def get_pending_scenes(self) -> List[Scene]:
        """Get scenes that haven't been generated yet"""
        return [s for s in sorted(self.scenes, key=lambda x: x.order)
                if s.status in ("pending", "failed")]


class StoryboardManager:
    """Manages storyboard project files"""

    def __init__(self, projects_dir: str = "storyboards"):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.current_project: Optional[StoryboardProject] = None

    def new_project(self, name: str = "Untitled Project") -> StoryboardProject:
        """Create a new project"""
        self.current_project = StoryboardProject(name=name)
        return self.current_project

    def save_project(self, filepath: Optional[str] = None) -> str:
        """Save current project to file"""
        if self.current_project is None:
            raise ValueError("No project to save")

        if filepath is None:
            safe_name = "".join(c for c in self.current_project.name if c.isalnum() or c in " -_").strip()
            safe_name = safe_name.replace(" ", "_")
            filepath = self.projects_dir / f"{safe_name}.json"

        filepath = Path(filepath)
        self.current_project.modified = datetime.now().isoformat()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.current_project.to_dict(), f, indent=2)

        return str(filepath)

    def load_project(self, filepath: str) -> StoryboardProject:
        """Load a project from file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.current_project = StoryboardProject.from_dict(data)
        return self.current_project

    def list_projects(self) -> List[Dict[str, str]]:
        """List all saved projects"""
        projects = []
        for f in self.projects_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    projects.append({
                        "path": str(f),
                        "name": data.get("name", f.stem),
                        "modified": data.get("modified", "Unknown"),
                        "scene_count": len(data.get("scenes", []))
                    })
            except Exception:
                continue

        return sorted(projects, key=lambda x: x["modified"], reverse=True)

    def export_to_csv(self, filepath: str) -> str:
        """Export project scenes to CSV"""
        if self.current_project is None:
            raise ValueError("No project to export")

        import csv

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Order", "Name", "Prompt", "Negative Prompt",
                           "Duration (s)", "Transition", "Transition Duration (s)"])

            fps = self.current_project.global_settings.fps
            for scene in sorted(self.current_project.scenes, key=lambda x: x.order):
                writer.writerow([
                    scene.order + 1,
                    scene.name,
                    scene.prompt,
                    scene.negative_prompt,
                    scene.duration_frames / fps,
                    scene.transition_to_next.type,
                    scene.transition_to_next.duration_frames / fps
                ])

        return filepath

    def import_from_csv(self, filepath: str) -> int:
        """Import scenes from CSV, returns number of scenes imported"""
        if self.current_project is None:
            self.new_project()

        import csv

        imported = 0
        fps = self.current_project.global_settings.fps

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene = Scene(
                    name=row.get("Name", f"Scene {imported + 1}"),
                    prompt=row.get("Prompt", ""),
                    negative_prompt=row.get("Negative Prompt", ""),
                    duration_frames=int(float(row.get("Duration (s)", 5)) * fps),
                    transition_to_next=Transition(
                        type=row.get("Transition", "cut"),
                        duration_frames=int(float(row.get("Transition Duration (s)", 0)) * fps)
                    )
                )
                self.current_project.add_scene(scene)
                imported += 1

        return imported
