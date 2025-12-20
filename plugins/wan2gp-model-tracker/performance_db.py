"""
Performance Database for Wan2GP Models

Stores and retrieves performance metrics (VRAM, speed, quality) for known models.
Provides estimation fallback for unknown models.
"""

from typing import Dict, List, Optional


class PerformanceDatabase:
    """Database of model performance metrics with estimation fallback"""

    # Hardcoded metrics for known models
    METRICS = {
        "vace_14B": {
            "speed_tier": "fast",
            "speed_multiplier": 2.0,
            "vram_min_gb": 8,
            "vram_recommended_gb": 12,
            "vram_max_observed_gb": 14,
            "quality_tier": "medium",
            "best_for": "Quick iterations and prompt testing",
            "notes": "2x faster than base T2V with slightly lower quality",
            "similar_faster": ["vace_1.3"],
            "similar_higher_quality": ["t2v", "t2v_2_2"]
        },
        "vace_1.3": {
            "speed_tier": "fast",
            "speed_multiplier": 2.5,
            "vram_min_gb": 6,
            "vram_recommended_gb": 10,
            "vram_max_observed_gb": 12,
            "quality_tier": "medium",
            "best_for": "Fastest iterations, lower quality acceptable",
            "notes": "Fastest VACE variant, good for testing",
            "similar_faster": [],
            "similar_higher_quality": ["vace_14B", "t2v"]
        },
        "t2v": {
            "speed_tier": "medium",
            "speed_multiplier": 1.0,
            "vram_min_gb": 16,
            "vram_recommended_gb": 24,
            "vram_max_observed_gb": 28,
            "quality_tier": "high",
            "best_for": "Balanced quality and speed for production",
            "notes": "Base Wan2.1 model, good all-around choice",
            "similar_faster": ["vace_14B"],
            "similar_higher_quality": ["t2v_2_2"]
        },
        "t2v_2_2": {
            "speed_tier": "medium",
            "speed_multiplier": 0.9,
            "vram_min_gb": 18,
            "vram_recommended_gb": 26,
            "vram_max_observed_gb": 30,
            "quality_tier": "high",
            "best_for": "Higher quality text-to-video generation",
            "notes": "Improved version of base T2V model",
            "similar_faster": ["t2v"],
            "similar_higher_quality": []
        },
        "i2v": {
            "speed_tier": "medium",
            "speed_multiplier": 1.1,
            "vram_min_gb": 16,
            "vram_recommended_gb": 24,
            "vram_max_observed_gb": 28,
            "quality_tier": "high",
            "best_for": "Image-to-video generation",
            "notes": "Specialized for converting images to video",
            "similar_faster": [],
            "similar_higher_quality": []
        },
        "flux_dev": {
            "speed_tier": "slow",
            "speed_multiplier": 0.3,
            "vram_min_gb": 20,
            "vram_recommended_gb": 32,
            "vram_max_observed_gb": 40,
            "quality_tier": "highest",
            "best_for": "Highest quality image generation",
            "notes": "Slower but exceptional quality for images",
            "similar_faster": ["flux_schnell"],
            "similar_higher_quality": []
        },
        "flux_schnell": {
            "speed_tier": "medium",
            "speed_multiplier": 0.8,
            "vram_min_gb": 16,
            "vram_recommended_gb": 24,
            "vram_max_observed_gb": 32,
            "quality_tier": "high",
            "best_for": "Faster high-quality image generation",
            "notes": "Faster Flux variant with good quality",
            "similar_faster": [],
            "similar_higher_quality": ["flux_dev"]
        },
        "hunyuan_1_5_t2v": {
            "speed_tier": "medium",
            "speed_multiplier": 1.0,
            "vram_min_gb": 16,
            "vram_recommended_gb": 24,
            "vram_max_observed_gb": 28,
            "quality_tier": "high",
            "best_for": "Alternative high-quality video generation",
            "notes": "Hunyuan Video model for text-to-video",
            "similar_faster": [],
            "similar_higher_quality": []
        },
        "ltxv": {
            "speed_tier": "fast",
            "speed_multiplier": 1.8,
            "vram_min_gb": 10,
            "vram_recommended_gb": 16,
            "vram_max_observed_gb": 20,
            "quality_tier": "medium",
            "best_for": "Fast video generation with good quality",
            "notes": "LTX Video model, efficient architecture",
            "similar_faster": [],
            "similar_higher_quality": ["ltxv_hq"]
        },
        "ltxv_hq": {
            "speed_tier": "medium",
            "speed_multiplier": 1.2,
            "vram_min_gb": 14,
            "vram_recommended_gb": 20,
            "vram_max_observed_gb": 24,
            "quality_tier": "high",
            "best_for": "Higher quality LTX video generation",
            "notes": "LTX Video high-quality variant",
            "similar_faster": ["ltxv"],
            "similar_higher_quality": []
        },
        "qwen2vl": {
            "speed_tier": "fast",
            "speed_multiplier": 2.2,
            "vram_min_gb": 8,
            "vram_recommended_gb": 12,
            "vram_max_observed_gb": 16,
            "quality_tier": "medium",
            "best_for": "Vision-language tasks",
            "notes": "Qwen2-VL model for vision-language understanding",
            "similar_faster": [],
            "similar_higher_quality": []
        },
        "chatterbox": {
            "speed_tier": "fast",
            "speed_multiplier": 3.0,
            "vram_min_gb": 4,
            "vram_recommended_gb": 8,
            "vram_max_observed_gb": 10,
            "quality_tier": "low",
            "best_for": "Quick audio/speech generation",
            "notes": "Fast audio generation model",
            "similar_faster": [],
            "similar_higher_quality": []
        },
        "Alpha": {
            "speed_tier": "medium",
            "speed_multiplier": 1.0,
            "vram_min_gb": 12,
            "vram_recommended_gb": 18,
            "vram_max_observed_gb": 22,
            "quality_tier": "medium",
            "best_for": "Experimental features",
            "notes": "Alpha variant model",
            "similar_faster": [],
            "similar_higher_quality": ["Omega"]
        },
        "Omega": {
            "speed_tier": "slow",
            "speed_multiplier": 0.7,
            "vram_min_gb": 16,
            "vram_recommended_gb": 24,
            "vram_max_observed_gb": 30,
            "quality_tier": "high",
            "best_for": "Advanced experimental features",
            "notes": "Omega variant model with higher quality",
            "similar_faster": ["Alpha"],
            "similar_higher_quality": []
        }
    }

    @classmethod
    def get_metrics(cls, model_type: str) -> Dict:
        """
        Get metrics for a model, with fallback to estimation

        Args:
            model_type: Model type identifier (e.g., "vace_14B", "t2v")

        Returns:
            Dictionary containing performance metrics

        Logic:
        1. Try exact match in METRICS
        2. Try base model type match (remove version numbers)
        3. Estimate from model name patterns if no match
        """
        # Try exact match
        if model_type in cls.METRICS:
            return cls.METRICS[model_type].copy()

        # Try base model type (remove version numbers and variants)
        base_type = cls._get_base_type(model_type)
        if base_type in cls.METRICS:
            metrics = cls.METRICS[base_type].copy()
            metrics["notes"] = f"Metrics estimated from base model '{base_type}'"
            return metrics

        # Estimate from model name patterns
        return cls.estimate_from_name(model_type)

    @classmethod
    def _get_base_type(cls, model_type: str) -> str:
        """Extract base model type by removing version numbers"""
        # Remove common version patterns
        import re
        base = re.sub(r'_v?\d+(\.\d+)*$', '', model_type)
        base = re.sub(r'_\d+B$', '', base)  # Remove parameter count like _14B
        return base

    @classmethod
    def estimate_from_name(cls, model_type: str) -> Dict:
        """
        Estimate metrics from model name patterns

        Heuristics:
        - VACE models → fast tier
        - Flux models → slow tier (dev) or medium (schnell)
        - T2V models → medium tier
        - I2V models → medium tier
        - LTX models → fast tier
        """
        model_lower = model_type.lower()

        # Default estimates
        metrics = {
            "speed_tier": "medium",
            "speed_multiplier": 1.0,
            "vram_min_gb": 12,
            "vram_recommended_gb": 20,
            "vram_max_observed_gb": 24,
            "quality_tier": "medium",
            "best_for": "General purpose",
            "notes": f"Estimated metrics for unknown model '{model_type}'",
            "similar_faster": [],
            "similar_higher_quality": []
        }

        # Pattern-based estimation
        if "vace" in model_lower:
            metrics.update({
                "speed_tier": "fast",
                "speed_multiplier": 2.0,
                "vram_min_gb": 8,
                "vram_recommended_gb": 12,
                "vram_max_observed_gb": 14,
                "quality_tier": "medium",
                "best_for": "Quick iterations"
            })
        elif "flux" in model_lower:
            if "dev" in model_lower:
                metrics.update({
                    "speed_tier": "slow",
                    "speed_multiplier": 0.3,
                    "vram_min_gb": 20,
                    "vram_recommended_gb": 32,
                    "vram_max_observed_gb": 40,
                    "quality_tier": "highest",
                    "best_for": "Highest quality image generation"
                })
            else:  # schnell or other
                metrics.update({
                    "speed_tier": "medium",
                    "speed_multiplier": 0.8,
                    "vram_min_gb": 16,
                    "vram_recommended_gb": 24,
                    "vram_max_observed_gb": 32,
                    "quality_tier": "high",
                    "best_for": "Fast high-quality images"
                })
        elif "ltx" in model_lower or "ltxv" in model_lower:
            metrics.update({
                "speed_tier": "fast",
                "speed_multiplier": 1.8,
                "vram_min_gb": 10,
                "vram_recommended_gb": 16,
                "vram_max_observed_gb": 20,
                "quality_tier": "medium",
                "best_for": "Fast video generation"
            })
        elif "hunyuan" in model_lower:
            metrics.update({
                "speed_tier": "medium",
                "speed_multiplier": 1.0,
                "vram_min_gb": 16,
                "vram_recommended_gb": 24,
                "vram_max_observed_gb": 28,
                "quality_tier": "high",
                "best_for": "High-quality video generation"
            })
        elif "t2v" in model_lower:
            metrics.update({
                "speed_tier": "medium",
                "speed_multiplier": 1.0,
                "vram_min_gb": 16,
                "vram_recommended_gb": 24,
                "vram_max_observed_gb": 28,
                "quality_tier": "high",
                "best_for": "Balanced quality and speed"
            })
        elif "i2v" in model_lower:
            metrics.update({
                "speed_tier": "medium",
                "speed_multiplier": 1.1,
                "vram_min_gb": 16,
                "vram_recommended_gb": 24,
                "vram_max_observed_gb": 28,
                "quality_tier": "high",
                "best_for": "Image-to-video generation"
            })

        return metrics

    @classmethod
    def get_all_models_sorted(cls, sort_by: str = "name") -> List[str]:
        """
        Return sorted list of model types from database

        Args:
            sort_by: Sort key - "name", "speed", "vram", or "quality"

        Returns:
            Sorted list of model type identifiers
        """
        models = list(cls.METRICS.keys())

        if sort_by == "name":
            return sorted(models)
        elif sort_by == "speed":
            # Fast → Medium → Slow
            speed_order = {"fast": 0, "medium": 1, "slow": 2}
            return sorted(models, key=lambda m: speed_order.get(
                cls.METRICS[m]["speed_tier"], 1
            ))
        elif sort_by == "vram":
            return sorted(models, key=lambda m: cls.METRICS[m]["vram_recommended_gb"])
        elif sort_by == "quality":
            # Low → Medium → High → Highest
            quality_order = {"low": 0, "medium": 1, "high": 2, "highest": 3}
            return sorted(models, key=lambda m: quality_order.get(
                cls.METRICS[m]["quality_tier"], 1
            ))
        else:
            return sorted(models)

    @classmethod
    def get_speed_tier_emoji(cls, speed_tier: str) -> str:
        """Get emoji representation of speed tier"""
        return {
            "fast": "🚀",
            "medium": "⚡",
            "slow": "🐢"
        }.get(speed_tier, "❓")

    @classmethod
    def get_quality_tier_emoji(cls, quality_tier: str) -> str:
        """Get emoji representation of quality tier"""
        return {
            "low": "⭐",
            "medium": "⭐⭐",
            "high": "⭐⭐⭐",
            "highest": "⭐⭐⭐⭐"
        }.get(quality_tier, "❓")
