# wan2gp-storyboard

A plugin for [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) that enables multi-scene video planning and generation with visual timeline, transitions, and automatic concatenation.

## Features

- **Visual Timeline**: Interactive scene organization with clickable cards
- **Scene Management**: Add, edit, duplicate, delete, and reorder scenes
- **12 Transition Types**: Crossfade, wipe, dissolve, slide, fade, zoom, and more
- **Global Settings**: Configure model, resolution, FPS, seed mode for all scenes
- **Queue Integration**: Queue all scenes or only pending ones for generation
- **Auto-Concatenation**: FFmpeg-powered scene joining with smooth transitions
- **Import/Export**: Save projects as JSON, bulk import scenes from CSV
- **Per-Scene Settings**: Override global settings for individual scenes
- **Reference Images**: Support for I2V with per-scene reference images

## Installation

### Method 1: Direct Copy (Easiest)

1. Download or copy these plugin files to your Wan2GP installation:
   ```
   plugins/wan2gp-storyboard/__init__.py
   plugins/wan2gp-storyboard/plugin.py
   plugins/wan2gp-storyboard/storyboard.py
   plugins/wan2gp-storyboard/transitions.py
   plugins/wan2gp-storyboard/README.md
   ```

2. Enable the plugin:
   - Open `wgp_config.json` in your Wan2GP root directory
   - Add `"wan2gp-storyboard"` to the `enabled_plugins` list:
     ```json
     {
       "enabled_plugins": ["wan2gp-storyboard"]
     }
     ```

3. Restart WanGP

### Method 2: Git Clone from Subdirectory

```bash
cd /path/to/your/wan2gp/installation/plugins
git clone --no-checkout https://github.com/hnethery/Wan2GP.git temp-storyboard
cd temp-storyboard
git sparse-checkout init --cone
git sparse-checkout set plugins/wan2gp-storyboard
git checkout claude/propose-plugins-013hbhRw8dBzVERAr2LTKhtK
mv plugins/wan2gp-storyboard ../
cd ..
rm -rf temp-storyboard
```

Then enable the plugin as described in Method 1, step 2.

## Usage

### Creating a Project

1. Open the **Storyboard** tab
2. Enter a project name and click **New**
3. Configure **Global Settings** (model, resolution, FPS, etc.)

### Adding Scenes

1. Click **+ Add Scene**
2. Fill in scene details:
   - **Name**: Descriptive scene name
   - **Prompt**: What happens in this scene
   - **Duration**: Length in seconds
   - **Reference Image**: (Optional) For I2V generation
   - **Transition**: How this scene transitions to the next

### Timeline Controls

- **Click a scene card** to edit it
- **Move Up/Down** to reorder scenes
- **Duplicate** to copy a scene
- **Delete** to remove a scene

### Generating Videos

1. **Queue All Scenes**: Add all scenes to the generation queue
2. **Queue Remaining**: Only queue pending/failed scenes
3. Navigate to the **Video Generator** tab to process the queue
4. Return to **Storyboard** when scenes complete

### Final Assembly

1. Once scenes are complete, click **Concatenate Completed**
2. The plugin will join all scenes with your chosen transitions
3. Find the final video in `outputs/storyboard_[project]_[timestamp].mp4`

## Transition Types

- **Cut**: Hard cut, no transition
- **Crossfade**: Smooth blend
- **Fade Black/White**: Fade through solid color
- **Wipe Left/Right/Up/Down**: Directional wipe
- **Slide Left/Right**: Sliding motion
- **Dissolve**: Pixelated dissolve
- **Zoom In**: Zoom into next scene

## Global Settings

### Seed Modes

- **Fixed**: Same seed for all scenes (consistent style)
- **Increment**: Seed+1 per scene (slight variation)
- **Random**: Different seed each scene (maximum variety)

### Per-Scene Overrides

Each scene can override global settings for guidance scale and inference steps.

## Import/Export

### Export Project
- Click **Save** to save as JSON in `storyboards/`
- Click **Export CSV** to export scene list

### Import Scenes
- Click **Import CSV** to bulk-add scenes from CSV
- CSV format: `Order, Name, Prompt, Negative Prompt, Duration (s), Transition, Transition Duration (s)`

## Requirements

- Wan2GP v9.52+
- FFmpeg (for concatenation)
- Python 3.10+

## Troubleshooting

### Plugin doesn't appear
- Check `enabled_plugins` in `wgp_config.json`
- Verify plugin is in `plugins/wan2gp-storyboard/`
- Check console for Python errors
- Restart WanGP

### Concatenation fails
- Ensure FFmpeg is installed and accessible
- Check that all scene output files exist
- Try **Queue Remaining** to regenerate failed scenes

### Transitions don't work
- Some transitions require specific FFmpeg versions
- Falls back to simple concatenation on error
- Check console for FFmpeg error messages

## Version

**v1.0.0** - Initial release

## License

This plugin follows the same license as Wan2GP.

## Credits

Created as a community plugin for Wan2GP.
