# Model Storage Tracker Plugin

A Wan2GP plugin to help you track and manage your downloaded models, checkpoints, and LoRAs with comprehensive LoRA metadata management.

## Features

### Storage Tracking
- **Visual Storage Breakdown**: See at a glance how much space each category (Models, LoRAs, Checkpoints) is using
- **Detailed File List**: Browse all your model files with sizes, dates, and paths
- **Search & Filter**: Quickly find specific files or filter by category and size
- **Export**: Export your storage inventory to JSON for analysis

### LoRA Library (NEW in v2.0)
- **LoRA Cards**: Visual card-based display of all your LoRAs with metadata
- **Trigger Words**: Store trigger words/keywords for each LoRA
- **Tags**: Organize LoRAs with custom tags (style, character, motion, etc.)
- **Star Ratings**: Rate your LoRAs from 1-5 stars
- **Model Compatibility**: Track which base model each LoRA works with (Wan T2V, I2V, Hunyuan, Flux, LTXV, Qwen, TTS)
- **Usage Tracking**: Automatic tracking of how often each LoRA is used
- **Notes**: Add personal notes about each LoRA (recommended settings, tips, etc.)
- **Source URLs**: Store where you downloaded each LoRA from

### LoRA Filtering & Search
- Filter by model type (Wan T2V, Hunyuan, Flux, etc.)
- Filter by rating (5 stars, 4+ stars, 3+ stars, unrated)
- Filter by usage (Most Used, Recently Used, Never Used)
- Search across names, trigger words, tags, and notes
- Sort by name, rating, usage count, or file size

### Usage Statistics
- Total LoRA count and usage stats
- Most used LoRAs leaderboard
- Highest rated LoRAs
- Breakdown by model type

## Usage

### Basic Storage Tracking

1. Click on the **Storage** tab in Wan2GP
2. The plugin will automatically scan your model directories on first visit
3. Use the **Scan Storage** button to refresh the data
4. Filter files by category or size
5. Sort by size, name, or date
6. Use the search box to find specific files

### Managing LoRA Metadata

1. Open the **LoRA Library** accordion
2. Browse your LoRAs in the card view
3. To edit a LoRA's metadata:
   - Expand the **Edit LoRA Metadata** section
   - Select a LoRA from the dropdown
   - Click **Load** to populate the form
   - Add trigger words (comma-separated)
   - Add tags for organization
   - Set the model type
   - Rate it 1-5 stars
   - Add source URL and notes
   - Click **Save Metadata**

### Viewing Usage Statistics

1. Expand the **LoRA Usage Statistics** accordion
2. See your most used and highest rated LoRAs
3. View distribution by model type

## What Gets Scanned

The plugin automatically scans all LoRA directories:
- `loras/` - General T2V LoRAs
- `loras_i2v/` - Image-to-Video LoRAs
- `loras_hunyuan/` - Hunyuan Video T2V LoRAs
- `loras_hunyuan_i2v/` - Hunyuan Video I2V LoRAs
- `loras_flux/` - Flux LoRAs
- `loras_ltxv/` - LTX Video LoRAs
- `loras_qwen/` - Qwen LoRAs
- `loras_tts/` - TTS LoRAs
- `models/` directory (all subdirectories)
- Custom checkpoint paths from server settings

## Supported File Types

- `.safetensors`
- `.ckpt`
- `.pth`
- `.pt`
- `.bin`
- `.sft`

## Data Storage

- **LoRA Metadata**: Stored in `~/.wan2gp/lora_metadata.json`
- **Exports**: Saved to `~/.wan2gp/exports/`

The metadata persists across sessions, so you won't lose your trigger words, tags, notes, or ratings.

## API for Other Plugins

Other plugins can record LoRA usage by calling:
```python
storage_tracker_plugin.record_lora_usage("/path/to/lora.safetensors")
```

This automatically increments the usage count and updates the last used timestamp.

## Tips

- **Trigger Words**: Store activation keywords from CivitAI or HuggingFace
- **Tags**: Use consistent tags like `anime`, `realistic`, `motion`, `style` for easy filtering
- **Notes**: Record optimal strength values, compatible models, or any quirks
- **Ratings**: Use ratings to quickly find your best LoRAs
- **Source URLs**: Save links to update pages for future reference
- **Regular Scans**: Rescan after downloading new LoRAs to catalog them

## Version History

- **2.0.0** - Added comprehensive LoRA tracking with metadata, ratings, tags, usage stats
- **1.0.0** - Initial release with storage tracking
