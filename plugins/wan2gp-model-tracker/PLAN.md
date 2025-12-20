# wan2gp-model-tracker - Implementation Plan

## Overview
A Wan2GP plugin that tracks which models are downloaded vs. missing, displays performance metrics (VRAM/speed/quality), and helps users choose the right model for their needs.

---

## Problem Statement

### User Pain Points
1. **Model Status Unknown** - Can't tell which models are downloaded until generation starts (50/50 chance of triggering download)
2. **Performance Confusion** - Hard to remember which models are faster/slower, use more/less VRAM
3. **Suboptimal Choices** - Users pick wrong model for their hardware/use case

### Solution
Dashboard plugin that shows:
- ✓/✗ Downloaded status for all models
- VRAM requirements and speed tiers
- Quality ratings and recommended use cases
- Smart filtering and comparison tools

---

## Technical Architecture

### File Structure
```
wan2gp-model-tracker/
├── __init__.py                 # Empty Python package marker
├── plugin.py                   # Main WAN2GPPlugin subclass
├── model_analyzer.py           # Model detection and file analysis
├── performance_db.py           # Performance metrics database
├── ui_components.py            # Reusable Gradio UI components
├── README.md                   # Installation and usage docs
└── requirements.txt            # Dependencies (likely empty)
```

---

## Core Components

### 1. ModelAnalyzer (`model_analyzer.py`)

**Purpose**: Detect which models are downloaded and analyze their properties

**Key Methods**:
```python
class ModelAnalyzer:
    def __init__(self, models_def, files_locator, get_local_model_filename_func):
        """Initialize with references to Wan2GP's model system"""

    def get_model_status(self, model_type: str) -> str:
        """Returns: 'downloaded', 'missing', or 'partial'

        Logic:
        - Use get_local_model_filename() from wgp.py (line 2871)
        - Check all URLs in model definition
        - Return 'downloaded' if all files exist
        - Return 'partial' if some exist
        - Return 'missing' if none exist
        """

    def get_model_files(self, model_type: str) -> List[Dict]:
        """Returns list of file dictionaries with status

        Returns:
        [
            {
                "filename": "model.safetensors",
                "url": "https://...",
                "status": "downloaded",
                "size_gb": 4.2,
                "path": "C:\\...\\ckpts\\model.safetensors"
            },
            ...
        ]
        """

    def get_model_size_gb(self, model_type: str) -> float:
        """Returns total size in GB (estimated or actual)

        Logic:
        - If downloaded: use os.path.getsize()
        - If missing: estimate from architecture
        """

    def estimate_vram_usage(self, model_type: str, quantization: str = "bf16") -> Dict:
        """Estimate VRAM requirements

        Returns:
        {
            "min_gb": 8,
            "recommended_gb": 12,
            "notes": "Based on 14B parameter count with bf16"
        }

        Logic:
        - Parse model architecture from model_def
        - Apply quantization multiplier (int8 = 0.5x, bf16 = 1.0x, fp32 = 2.0x)
        - Add overhead estimates (15-20%)
        """

    def get_download_urls(self, model_type: str) -> List[str]:
        """Extract all URLs from model definition

        From model_def["URLs"] and model_def.get("URLs2", [])
        """
```

**Integration Points**:
- Access Wan2GP's `models_def` dictionary (wgp.py line 2209)
- Use `get_local_model_filename()` function (wgp.py line 2871)
- Use `files_locator` module for path resolution (shared/utils/files_locator.py)

---

### 2. PerformanceDatabase (`performance_db.py`)

**Purpose**: Store and retrieve performance metrics for models

**Data Structure**:
```python
class PerformanceDatabase:
    # Hardcoded metrics for known models
    METRICS = {
        "vace_14B": {
            "speed_tier": "fast",           # fast/medium/slow
            "speed_multiplier": 2.0,        # vs base t2v
            "vram_min_gb": 8,
            "vram_recommended_gb": 12,
            "vram_max_observed_gb": 14,
            "quality_tier": "medium",       # low/medium/high/highest
            "best_for": "Quick iterations and prompt testing",
            "notes": "2x faster than base T2V with slightly lower quality",
            "similar_faster": ["vace_1.3"],
            "similar_higher_quality": ["t2v", "t2v_2_2"]
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
        # ... more models (populate from model analysis)
    }

    @classmethod
    def get_metrics(cls, model_type: str) -> Dict:
        """Get metrics for a model, with fallback to estimation

        Logic:
        1. Try exact match in METRICS
        2. Try base model type match
        3. Estimate from architecture if no match
        """

    @classmethod
    def estimate_from_architecture(cls, model_def: Dict) -> Dict:
        """Estimate metrics from model definition

        Heuristics:
        - VACE models → fast tier
        - Flux models → slow tier
        - T2V models → medium tier
        - Parameter count → VRAM estimate (1B params ≈ 2-4GB)
        - Parse description for quality hints
        """

    @classmethod
    def get_all_models_sorted(cls, sort_by: str = "name") -> List[str]:
        """Return sorted list of model types

        Sort options: name, speed, vram, quality
        """
```

**Initial Population Strategy**:
1. Start with ~10-15 most common models
2. Add estimation fallback for unknown models
3. Can be expanded later with community contributions

---

### 3. Main Plugin (`plugin.py`)

**Purpose**: Gradio UI and event handling

**Key Features**:
- Summary statistics (total/downloaded/missing counts)
- Filter by: Status (all/downloaded/missing), Speed tier, VRAM budget
- Sort by: Name, Status, Speed, VRAM, Quality
- Detail view: Full model info with file paths, recommendations
- Export report: Generate markdown inventory

**Gradio Components**:
- Dataframe for model table
- Radio/Dropdown for filters
- Accordion for expandable details
- Buttons for refresh and export

---

## Implementation Phases

### Phase 1: Core Functionality (MVP)
1. Create plugin file structure
2. Implement ModelAnalyzer with basic status detection
3. Create PerformanceDatabase with ~10 common models
4. Build basic Gradio UI (table view only)
5. Test with local Wan2GP installation
6. Verify model detection works correctly

### Phase 2: Enhanced Features
7. Add filtering (status, speed, VRAM)
8. Add sorting (name, speed, VRAM, quality)
9. Implement detail view with recommendations
10. Add summary statistics
11. Test with multiple models (downloaded and missing)

### Phase 3: Polish
12. Add export report functionality
13. Improve UI layout and styling
14. Add tooltips and help text
15. Write comprehensive README
16. Create screenshots for documentation

---

## Data Requirements

### Performance Metrics to Populate

**Priority 1 (Most Common Models)**:
- vace_14B
- vace_1.3
- t2v
- t2v_2_2
- i2v
- flux_dev
- flux_schnell
- hunyuan_1_5_t2v

**Priority 2 (Secondary Models)**:
- ltxv
- ltxv_hq
- qwen2vl
- chatterbox
- Alpha
- Omega

**Data Sources**:
1. Model JSON descriptions (defaults/*.json)
2. Architecture analysis (parameter counts)
3. Community benchmarks (if available)
4. Conservative estimates (when no data available)

---

## Testing Strategy

### Test Cases
1. **Model Detection**
   - Correctly identifies downloaded models
   - Correctly identifies missing models
   - Handles multi-file models (URLs and URLs2)
   - Works with different checkpoint paths

2. **Filtering**
   - "Downloaded" shows only ✓ models
   - "Missing" shows only ✗ models
   - Speed filter works correctly
   - VRAM filter works correctly

3. **Sorting**
   - Name sort (alphabetical)
   - Status sort (downloaded first)
   - Speed sort (fast → slow)
   - VRAM sort (low → high)
   - Quality sort (low → high)

4. **Detail View**
   - Shows correct file paths
   - Shows correct file sizes
   - Recommendations are relevant

---

## UI Example

### Table View
```
Status | Model Name              | Speed  | VRAM (GB) | Quality | Size (GB) | Best For
-------|-------------------------|--------|-----------|---------|-----------|-------------------------
✓      | Wan2.1 VACE 14B        | Fast   | 8-12      | Medium  | 4.2       | Quick iterations
✗      | Wan2.1 Text2Video      | Medium | 16-24     | High    | ~12       | Balanced production
✓      | Flux Dev               | Slow   | 20-32     | Highest | 23.8      | Highest quality images
```

### Detail View
```markdown
## Wan2.1 VACE 14B

**Status:** Downloaded
**Architecture:** vace_14B
**Speed Tier:** Fast (2x base T2V)
**VRAM Required:** 8GB minimum, 12GB recommended
**Quality:** Medium
**Best For:** Quick iterations and prompt testing

### Files
- ✓ `vace_14B.safetensors` (4.23GB) - C:\...\ckpts\vace_14B.safetensors

### Recommendations
**Faster alternatives:** vace_1.3
**Higher quality alternatives:** t2v, t2v_2_2
```

---

## Future Enhancements (Post-MVP)

### v1.1 - Download Queue
- Select multiple missing models
- Batch download with progress bar

### v1.2 - Disk Management
- Show total disk usage
- Identify unused models
- One-click cleanup

### v1.3 - Benchmarking
- Run actual speed tests
- Store results in local database

### v1.4 - Smart Recommendations
- Track usage patterns
- Suggest removing unused models

---

## Success Metrics

1. User can answer "Do I have model X?" in <5 seconds
2. User can find "fastest model I have downloaded" in <10 seconds
3. 100% accurate model detection
4. <1s table refresh time
5. No crashes on edge cases

---

## Conclusion

This plugin will significantly improve user experience by:
1. Eliminating surprise model downloads
2. Helping users choose optimal models for their hardware
3. Providing clear performance trade-offs
4. Enabling informed decision-making

Implementation is straightforward using Wan2GP's existing plugin API with minimal dependencies and maximum value.
