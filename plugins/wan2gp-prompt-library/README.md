# Wan2GP Prompt Library Plugin

A comprehensive prompt management system that lets you save, organize, and reuse prompts with their associated generation settings.

## Features

- 📚 **Organize prompts** in collections (Favorites, Cinematic, Anime, Realistic, Character)
- 🔍 **Search and filter** prompts by name, text, or tags
- 📝 **Variable substitution** - use placeholders like `{location}` in prompts
- ⚙️ **Save generation settings** - capture model, resolution, LoRAs, and other parameters
- ⭐ **Favorites system** - mark your best prompts for quick access
- 📊 **Usage tracking** - see which prompts you use most
- 📤 **Import/Export** - share collections with the community
- 🎨 **Starter templates** - pre-loaded examples to get you started

## Installation

### Enable the Plugin

1. **Using Plugin Manager (Recommended)**:
   - Open Wan2GP and go to the "Plugin Manager" tab
   - Find "wan2gp-prompt-library" in the list
   - Check the box to enable it
   - Click "Save & Restart"

2. **Manual Configuration**:
   - Find your server config file (usually in the Wan2GP directory)
   - Add `"wan2gp-prompt-library"` to the `enabled_plugins` array:
   ```json
   {
     "enabled_plugins": [
       "wan2gp-prompt-library"
     ]
   }
   ```
   - Restart Wan2GP

## Usage

### Browsing Prompts

1. Click the "📚 Prompt Library" tab
2. Select a collection from the left panel
3. Use the search box or tag filters to find prompts
4. Click on a prompt card to select it

### Using a Prompt

When you click a prompt card, you have two options:

- **📝 Use Prompt Only**: Copies just the prompt text to the main generator
- **⚙️ Use with Settings**: Copies the prompt AND applies all saved settings (model, resolution, LoRAs, etc.)

If the prompt contains variables like `{location}`, you'll see a field to fill them in before use.

### Saving a Prompt

1. Generate a video with the settings you want
2. In the Prompt Library tab, expand "💾 Save Current Prompt to Library"
3. Enter a name and select a collection
4. Add tags (comma-separated) for easier searching
5. Check "Save current generation settings" if you want to capture model/resolution/LoRAs
6. Click "💾 Save to Library"

### Variable Substitution

Create reusable prompt templates with variables:

```
Cinematic drone shot of {location}, {time_of_day} lighting, 4K quality
```

When you use this prompt, you'll be asked to fill in values:
```
location=mountains, time_of_day=golden hour
```

### Managing Collections

- **Create**: Click "➕ New" in the Collections panel (requires manual JSON editing currently)
- **Delete**: Select a collection and click "🗑️ Delete"
- **Export**: Use the Import/Export section to save a collection as JSON
- **Import**: Upload a JSON file to add prompts from another collection

### Favorites

Click the ⭐ button to add/remove a prompt from your Favorites collection for quick access.

## File Storage

Prompts are stored in: `~/.wan2gp/prompt_library.json`

You can manually edit this file to:
- Create new collections
- Bulk edit prompts
- Backup your library

## Data Structure

```json
{
  "version": "1.0.0",
  "collections": {
    "favorites": {
      "name": "Favorites",
      "icon": "⭐",
      "prompts": [
        {
          "id": "uuid-here",
          "name": "Epic Drone Shot",
          "prompt": "Cinematic drone shot flying over {location}...",
          "negative_prompt": "blurry, distorted",
          "tags": ["aerial", "landscape", "cinematic"],
          "variables": ["location"],
          "settings": {
            "model_type": "wan2.1-t2v-14B",
            "resolution": "1280x720",
            "steps": 30,
            "guidance_scale": 7.5
          },
          "created": "2025-01-15T10:30:00Z",
          "last_used": "2025-01-20T14:00:00Z",
          "use_count": 12
        }
      ]
    }
  }
}
```

## Starter Templates

The plugin comes with pre-loaded templates in these collections:

- **🎬 Cinematic**: Epic drone shots, slow motion, establishing shots, close-up portraits
- **🎨 Anime**: Character actions, background scenes, magical girl transformations
- **📷 Realistic**: Nature documentary, urban street scenes, product showcases, time lapses
- **🎭 Character**: Walk cycles, talking animations, emotions, action sequences

Feel free to modify or delete these templates as needed!

## Sharing Collections

To share your prompt collection with others:

1. Go to the Import/Export section
2. Select the collection to export
3. Click "📤 Export Collection"
4. Find the exported JSON in `~/.wan2gp/exports/`
5. Share the JSON file

To import someone else's collection:

1. Download their JSON file
2. Go to Import/Export section
3. Upload the file
4. Choose whether to merge or replace
5. Click "📥 Import"

## Tips

- Use descriptive names for your prompts
- Add multiple tags to make prompts easier to find
- Save settings with prompts you've refined to perfection
- Use variables for prompts you use often with different subjects
- Check usage stats to see which prompts work best
- Export your favorites regularly as backup

## Troubleshooting

**Plugin doesn't appear**: Make sure it's enabled in the server config or Plugin Manager

**Prompts not saving**: Check that `~/.wan2gp/` directory exists and is writable

**Variables not substituting**: Make sure you use the format `key=value, key2=value2` with commas between pairs

**Settings not applying**: Try "Use with Settings" instead of "Use Prompt Only"

## Version

Current version: 1.0.0

## License

Same license as Wan2GP project.
