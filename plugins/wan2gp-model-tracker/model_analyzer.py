"""
Model Analyzer for Wan2GP

Detects which models are downloaded, analyzes their properties,
and provides information about model files.
"""

import os
from typing import Dict, List, Optional, Callable, Any

try:
    from performance_db import PerformanceDatabase
except ImportError:
    from .performance_db import PerformanceDatabase


class ModelAnalyzer:
    """Analyzes Wan2GP models to determine download status and properties"""

    def __init__(self, models_def: Dict, files_locator: Any, get_local_model_filename_func: Callable):
        """
        Initialize with references to Wan2GP's model system

        Args:
            models_def: Dictionary of model definitions from Wan2GP
            files_locator: Wan2GP's files_locator module for path resolution
            get_local_model_filename_func: Function to get local model filename
        """
        self.models_def = models_def
        self.files_locator = files_locator
        self.get_local_model_filename = get_local_model_filename_func

    def get_model_status(self, model_type: str) -> str:
        """
        Returns download status for a model

        Args:
            model_type: Model type identifier (e.g., "vace_14B", "t2v")

        Returns:
            'downloaded' - all files exist
            'partial' - some files exist
            'missing' - no files exist
            'unknown' - model not in definitions
        """
        if model_type not in self.models_def:
            return "unknown"

        model_def = self.models_def[model_type]
        urls = self._get_all_urls(model_def)

        if not urls:
            return "unknown"

        existing_count = 0
        total_count = len(urls)

        for url in urls:
            if self._check_file_exists(url):
                existing_count += 1

        if existing_count == total_count:
            return "downloaded"
        elif existing_count > 0:
            return "partial"
        else:
            return "missing"

    def get_model_files(self, model_type: str) -> List[Dict]:
        """
        Returns list of file dictionaries with status

        Args:
            model_type: Model type identifier

        Returns:
            List of dictionaries containing file information:
            [
                {
                    "filename": "model.safetensors",
                    "url": "https://...",
                    "status": "downloaded" | "missing",
                    "size_gb": 4.2,
                    "path": "/path/to/file" or None
                },
                ...
            ]
        """
        if model_type not in self.models_def:
            return []

        model_def = self.models_def[model_type]
        urls = self._get_all_urls(model_def)
        files = []

        for url in urls:
            filename = url.split('/')[-1]
            local_path = self._get_local_path(url)
            exists = os.path.exists(local_path) if local_path else False

            file_info = {
                "filename": filename,
                "url": url,
                "status": "downloaded" if exists else "missing",
                "size_gb": self._get_file_size_gb(local_path) if exists else None,
                "path": local_path if exists else None
            }
            files.append(file_info)

        return files

    def get_model_size_gb(self, model_type: str) -> float:
        """
        Returns total size in GB (actual or estimated)

        Args:
            model_type: Model type identifier

        Returns:
            Total size in GB
        """
        files = self.get_model_files(model_type)
        total_size = 0.0
        has_actual_sizes = False

        for file_info in files:
            if file_info["size_gb"] is not None:
                total_size += file_info["size_gb"]
                has_actual_sizes = True

        # If no actual sizes, estimate from model type
        if not has_actual_sizes:
            total_size = self._estimate_model_size(model_type)

        return round(total_size, 2)

    def estimate_vram_usage(self, model_type: str, quantization: str = "bf16") -> Dict:
        """
        Estimate VRAM requirements

        Args:
            model_type: Model type identifier
            quantization: Quantization type ("int8", "bf16", "fp32")

        Returns:
            {
                "min_gb": 8,
                "recommended_gb": 12,
                "notes": "Based on model type with bf16"
            }
        """
        # Get metrics from performance database
        metrics = PerformanceDatabase.get_metrics(model_type)

        # Apply quantization multiplier
        multiplier = {
            "int8": 0.5,
            "bf16": 1.0,
            "fp32": 2.0
        }.get(quantization, 1.0)

        min_gb = int(metrics["vram_min_gb"] * multiplier)
        recommended_gb = int(metrics["vram_recommended_gb"] * multiplier)

        return {
            "min_gb": min_gb,
            "recommended_gb": recommended_gb,
            "notes": f"Based on {model_type} with {quantization} quantization"
        }

    def get_download_urls(self, model_type: str) -> List[str]:
        """
        Extract all URLs from model definition

        Args:
            model_type: Model type identifier

        Returns:
            List of download URLs
        """
        if model_type not in self.models_def:
            return []

        model_def = self.models_def[model_type]
        return self._get_all_urls(model_def)

    def get_all_models(self) -> List[str]:
        """
        Get list of all model types in models_def

        Returns:
            List of model type identifiers
        """
        return list(self.models_def.keys())

    def get_model_name(self, model_type: str) -> str:
        """
        Get human-readable model name

        Args:
            model_type: Model type identifier

        Returns:
            Human-readable name from model definition or formatted type
        """
        if model_type not in self.models_def:
            return self._format_model_name(model_type)

        model_def = self.models_def[model_type]

        # Try to get name from various fields
        if "name" in model_def:
            return model_def["name"]
        elif "description" in model_def:
            # Use first line of description
            desc = model_def["description"]
            if isinstance(desc, str):
                return desc.split('\n')[0]

        return self._format_model_name(model_type)

    def get_model_description(self, model_type: str) -> str:
        """
        Get model description

        Args:
            model_type: Model type identifier

        Returns:
            Description text or empty string
        """
        if model_type not in self.models_def:
            return ""

        model_def = self.models_def[model_type]
        return model_def.get("description", "")

    # Private helper methods

    def _get_all_urls(self, model_def: Dict) -> List[str]:
        """Extract all URLs from model definition"""
        urls = []

        # Get URLs from "URLs" field
        if "URLs" in model_def:
            urls_field = model_def["URLs"]
            if isinstance(urls_field, list):
                urls.extend(urls_field)
            elif isinstance(urls_field, str):
                urls.append(urls_field)

        # Get URLs from "URLs2" field
        if "URLs2" in model_def:
            urls2_field = model_def["URLs2"]
            if isinstance(urls2_field, list):
                urls.extend(urls2_field)
            elif isinstance(urls2_field, str):
                urls.append(urls2_field)

        return urls

    def _normalize_for_matching(self, name: str) -> str:
        """Normalize filename for fuzzy matching"""
        import re
        # Remove extension
        name = os.path.splitext(name)[0]
        # Remove common suffixes/patterns
        name = re.sub(r'_quanto.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'_bf16.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'_fp16.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'_int8.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'_q[0-9]+.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[-_]', '', name)  # Remove separators
        return name.lower()

    def _fuzzy_match_filename(self, url_filename: str, disk_filename: str) -> bool:
        """Check if two filenames likely refer to the same model"""
        normalized_url = self._normalize_for_matching(url_filename)
        normalized_disk = self._normalize_for_matching(disk_filename)

        # Check if one contains the other (flexible matching)
        return (normalized_url in normalized_disk or
                normalized_disk in normalized_url or
                normalized_url == normalized_disk)

    def _get_local_path(self, url: str) -> Optional[str]:
        """Get local file path for a URL"""
        try:
            # Try using the provided function
            local_path = self.get_local_model_filename(url)
            if local_path and os.path.exists(local_path):
                return local_path
        except Exception:
            pass

        # Fallback: search for the file manually with fuzzy matching
        url_filename = url.split('/')[-1]

        # Try common checkpoint directories
        possible_dirs = ["checkpoints", "ckpts", "models"]

        for dir_name in possible_dirs:
            if not os.path.exists(dir_name):
                continue

            # First try exact match
            direct_path = os.path.join(dir_name, url_filename)
            if os.path.exists(direct_path):
                return direct_path

            # Then search with fuzzy matching
            for root, dirs, files in os.walk(dir_name):
                for disk_file in files:
                    # Only check model files
                    if not disk_file.endswith(('.safetensors', '.pth', '.ckpt', '.pt', '.bin')):
                        continue

                    if self._fuzzy_match_filename(url_filename, disk_file):
                        return os.path.join(root, disk_file)

        return None

    def _check_file_exists(self, url: str) -> bool:
        """Check if file for URL exists locally"""
        local_path = self._get_local_path(url)
        return os.path.exists(local_path) if local_path else False

    def _get_file_size_gb(self, file_path: str) -> float:
        """Get file size in GB"""
        try:
            size_bytes = os.path.getsize(file_path)
            size_gb = size_bytes / (1024 ** 3)
            return round(size_gb, 2)
        except Exception:
            return 0.0

    def _estimate_model_size(self, model_type: str) -> float:
        """Estimate model size based on type"""
        # Use VRAM as a rough proxy (model size ≈ VRAM requirement / 2)
        metrics = PerformanceDatabase.get_metrics(model_type)
        estimated_size = metrics["vram_recommended_gb"] / 2
        return round(estimated_size, 1)

    def _format_model_name(self, model_type: str) -> str:
        """Format model type into human-readable name"""
        # Replace underscores with spaces and capitalize
        name = model_type.replace('_', ' ')
        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())
        return name

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for all models

        Returns:
            {
                "total": 15,
                "downloaded": 5,
                "missing": 8,
                "partial": 2,
                "total_size_gb": 120.5
            }
        """
        all_models = self.get_all_models()
        stats = {
            "total": len(all_models),
            "downloaded": 0,
            "missing": 0,
            "partial": 0,
            "unknown": 0,
            "total_size_gb": 0.0
        }

        for model_type in all_models:
            status = self.get_model_status(model_type)
            stats[status] = stats.get(status, 0) + 1

            if status == "downloaded":
                stats["total_size_gb"] += self.get_model_size_gb(model_type)

        stats["total_size_gb"] = round(stats["total_size_gb"], 2)
        return stats
