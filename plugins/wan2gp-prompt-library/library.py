"""
Prompt Library - Storage and retrieval logic for prompt templates
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class PromptLibrary:
    """Manages prompt storage, retrieval, and organization"""

    def __init__(self, library_path: Optional[str] = None):
        """Initialize the prompt library

        Args:
            library_path: Custom path for library file. If None, uses ~/.wan2gp/prompt_library.json
        """
        if library_path is None:
            home = Path.home()
            wan2gp_dir = home / ".wan2gp"
            wan2gp_dir.mkdir(exist_ok=True)
            self.library_path = wan2gp_dir / "prompt_library.json"
        else:
            self.library_path = Path(library_path)

        self.data = self._load_library()

    def _load_library(self) -> Dict[str, Any]:
        """Load library from disk or create default structure"""
        if self.library_path.exists():
            try:
                with open(self.library_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading prompt library: {e}")
                return self._create_default_library()
        else:
            return self._create_default_library()

    def _create_default_library(self) -> Dict[str, Any]:
        """Create default library structure with built-in collections"""
        return {
            "version": "1.0.0",
            "collections": {
                "favorites": {
                    "name": "Favorites",
                    "icon": "⭐",
                    "prompts": []
                },
                "cinematic": {
                    "name": "Cinematic",
                    "icon": "🎬",
                    "prompts": []
                },
                "anime": {
                    "name": "Anime",
                    "icon": "🎨",
                    "prompts": []
                },
                "realistic": {
                    "name": "Realistic",
                    "icon": "📷",
                    "prompts": []
                },
                "character": {
                    "name": "Character",
                    "icon": "🎭",
                    "prompts": []
                }
            }
        }

    def save_library(self) -> bool:
        """Save library to disk"""
        try:
            with open(self.library_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving prompt library: {e}")
            return False

    def get_collection_names(self) -> List[tuple]:
        """Get list of collection names with icons for display

        Returns:
            List of (display_name, collection_id) tuples
        """
        collections = []
        for coll_id, coll_data in self.data["collections"].items():
            icon = coll_data.get("icon", "📁")
            name = coll_data.get("name", coll_id)
            display = f"{icon} {name}"
            collections.append((display, coll_id))
        return collections

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get a collection by ID"""
        return self.data["collections"].get(collection_id)

    def create_collection(self, collection_id: str, name: str, icon: str = "📁") -> bool:
        """Create a new collection

        Args:
            collection_id: Unique identifier for the collection
            name: Display name
            icon: Emoji icon

        Returns:
            True if created successfully
        """
        if collection_id in self.data["collections"]:
            return False

        self.data["collections"][collection_id] = {
            "name": name,
            "icon": icon,
            "prompts": []
        }
        return self.save_library()

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection

        Args:
            collection_id: Collection to delete

        Returns:
            True if deleted successfully
        """
        # Don't allow deletion of favorites
        if collection_id == "favorites":
            return False

        if collection_id in self.data["collections"]:
            del self.data["collections"][collection_id]
            return self.save_library()
        return False

    def add_prompt(
        self,
        collection_id: str,
        name: str,
        prompt: str,
        negative_prompt: str = "",
        tags: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Add a new prompt to a collection

        Args:
            collection_id: Collection to add to
            name: Prompt name/title
            prompt: The prompt text
            negative_prompt: Negative prompt text
            tags: List of tags
            settings: Generation settings (model, resolution, etc.)

        Returns:
            Prompt ID if successful, None otherwise
        """
        collection = self.get_collection(collection_id)
        if not collection:
            return None

        # Generate unique ID
        prompt_id = str(uuid.uuid4())

        # Extract variables from prompt (words in {curly braces})
        import re
        variables = re.findall(r'\{(\w+)\}', prompt)

        prompt_data = {
            "id": prompt_id,
            "name": name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "tags": tags or [],
            "variables": list(set(variables)),  # Unique variables
            "settings": settings or {},
            "created": datetime.utcnow().isoformat() + "Z",
            "last_used": None,
            "use_count": 0
        }

        collection["prompts"].append(prompt_data)

        if self.save_library():
            return prompt_id
        return None

    def update_prompt(
        self,
        prompt_id: str,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        tags: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing prompt

        Args:
            prompt_id: Prompt to update
            name: New name (if provided)
            prompt: New prompt text (if provided)
            negative_prompt: New negative prompt (if provided)
            tags: New tags (if provided)
            settings: New settings (if provided)

        Returns:
            True if updated successfully
        """
        prompt_data = self.get_prompt(prompt_id)
        if not prompt_data:
            return False

        if name is not None:
            prompt_data["name"] = name
        if prompt is not None:
            prompt_data["prompt"] = prompt
            # Recalculate variables
            import re
            variables = re.findall(r'\{(\w+)\}', prompt)
            prompt_data["variables"] = list(set(variables))
        if negative_prompt is not None:
            prompt_data["negative_prompt"] = negative_prompt
        if tags is not None:
            prompt_data["tags"] = tags
        if settings is not None:
            prompt_data["settings"] = settings

        return self.save_library()

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt from all collections

        Args:
            prompt_id: Prompt to delete

        Returns:
            True if deleted successfully
        """
        deleted = False
        for collection in self.data["collections"].values():
            prompts = collection["prompts"]
            original_length = len(prompts)
            collection["prompts"] = [p for p in prompts if p["id"] != prompt_id]
            if len(collection["prompts"]) < original_length:
                deleted = True

        if deleted:
            return self.save_library()
        return False

    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a prompt by ID from any collection

        Args:
            prompt_id: Prompt ID to find

        Returns:
            Prompt data dict or None
        """
        for collection in self.data["collections"].values():
            for prompt in collection["prompts"]:
                if prompt["id"] == prompt_id:
                    return prompt
        return None

    def get_prompts_in_collection(
        self,
        collection_id: str,
        search: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get prompts from a collection with optional filtering

        Args:
            collection_id: Collection to search
            search: Search string (matches name and prompt text)
            tags: Filter by tags (must have at least one matching tag)

        Returns:
            List of matching prompts
        """
        collection = self.get_collection(collection_id)
        if not collection:
            return []

        prompts = collection["prompts"]

        # Apply search filter
        if search and search.strip():
            search_lower = search.lower()
            prompts = [
                p for p in prompts
                if search_lower in p["name"].lower() or
                   search_lower in p["prompt"].lower()
            ]

        # Apply tag filter
        if tags:
            prompts = [
                p for p in prompts
                if any(tag in p.get("tags", []) for tag in tags)
            ]

        # Sort by usage (most used first), then by last used
        prompts = sorted(
            prompts,
            key=lambda p: (p.get("use_count", 0), p.get("last_used") or ""),
            reverse=True
        )

        return prompts

    def record_usage(self, prompt_id: str) -> bool:
        """Record that a prompt was used

        Args:
            prompt_id: Prompt that was used

        Returns:
            True if recorded successfully
        """
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            return False

        prompt["use_count"] = prompt.get("use_count", 0) + 1
        prompt["last_used"] = datetime.utcnow().isoformat() + "Z"

        return self.save_library()

    def find_by_prompt(self, prompt_text: str) -> Optional[Dict[str, Any]]:
        """Find a prompt by exact prompt text match

        Args:
            prompt_text: Prompt text to search for

        Returns:
            First matching prompt or None
        """
        for collection in self.data["collections"].values():
            for prompt in collection["prompts"]:
                if prompt["prompt"] == prompt_text:
                    return prompt
        return None

    def get_all_tags(self) -> List[str]:
        """Get all unique tags across all prompts

        Returns:
            Sorted list of unique tags
        """
        tags = set()
        for collection in self.data["collections"].values():
            for prompt in collection["prompts"]:
                tags.update(prompt.get("tags", []))
        return sorted(tags)

    def add_to_favorites(self, prompt_id: str) -> bool:
        """Add a prompt to favorites collection

        Args:
            prompt_id: Prompt to add

        Returns:
            True if added successfully
        """
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            return False

        favorites = self.get_collection("favorites")
        if not favorites:
            return False

        # Check if already in favorites
        for fav_prompt in favorites["prompts"]:
            if fav_prompt["id"] == prompt_id:
                return True  # Already in favorites

        # Add to favorites (by reference)
        favorites["prompts"].append(prompt)
        return self.save_library()

    def remove_from_favorites(self, prompt_id: str) -> bool:
        """Remove a prompt from favorites

        Args:
            prompt_id: Prompt to remove

        Returns:
            True if removed successfully
        """
        favorites = self.get_collection("favorites")
        if not favorites:
            return False

        original_length = len(favorites["prompts"])
        favorites["prompts"] = [
            p for p in favorites["prompts"]
            if p["id"] != prompt_id
        ]

        if len(favorites["prompts"]) < original_length:
            return self.save_library()
        return False

    def is_in_favorites(self, prompt_id: str) -> bool:
        """Check if a prompt is in favorites

        Args:
            prompt_id: Prompt to check

        Returns:
            True if in favorites
        """
        favorites = self.get_collection("favorites")
        if not favorites:
            return False

        return any(p["id"] == prompt_id for p in favorites["prompts"])

    def export_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Export a collection for sharing

        Args:
            collection_id: Collection to export

        Returns:
            Collection data dict or None
        """
        collection = self.get_collection(collection_id)
        if not collection:
            return None

        return {
            "version": self.data["version"],
            "collection": {
                collection_id: collection
            }
        }

    def import_collection(self, collection_data: Dict[str, Any], merge: bool = False) -> bool:
        """Import a collection from exported data

        Args:
            collection_data: Exported collection data
            merge: If True, merge with existing. If False, replace.

        Returns:
            True if imported successfully
        """
        try:
            if "collection" not in collection_data:
                return False

            for coll_id, coll_data in collection_data["collection"].items():
                if coll_id in self.data["collections"] and not merge:
                    # Replace existing
                    self.data["collections"][coll_id] = coll_data
                elif coll_id in self.data["collections"] and merge:
                    # Merge prompts
                    existing = self.data["collections"][coll_id]
                    existing_ids = {p["id"] for p in existing["prompts"]}

                    for prompt in coll_data["prompts"]:
                        if prompt["id"] not in existing_ids:
                            existing["prompts"].append(prompt)
                else:
                    # New collection
                    self.data["collections"][coll_id] = coll_data

            return self.save_library()
        except Exception as e:
            print(f"Error importing collection: {e}")
            return False
