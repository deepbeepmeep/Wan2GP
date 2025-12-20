# wan2gp-model-tracker

A Wan2GP plugin that tracks which models are downloaded vs. missing, displays performance metrics (VRAM/speed/quality), and helps you choose the right model for your needs.

## Features

- **Model Status Dashboard** - See at a glance which models are downloaded (✓) and which are missing (✗)
- **Performance Metrics** - View VRAM requirements, speed tiers, and quality ratings for each model
- **Smart Filtering** - Filter by download status, speed tier, or VRAM budget
- **Comparison Tools** - Sort and compare models to find the best fit for your hardware
- **Detailed Info** - Expandable details showing file paths, sizes, and recommendations
- **Export Reports** - Generate markdown inventory of your models

## Installation

### From Wan2GP Plugin Manager

1. Open Wan2GP
2. Go to the **Plugins** tab
3. Click **Install from URL**
4. Enter: `https://github.com/[your-username]/wan2gp-model-tracker`
5. Check the box to enable the plugin
6. Click **Save Settings**
7. Restart Wan2GP

### Manual Installation

1. Clone this repository into your Wan2GP `plugins/` directory:
   ```bash
   cd /path/to/wan2gp/plugins
   git clone https://github.com/[your-username]/wan2gp-model-tracker.git
   ```
2. Enable the plugin in Wan2GP's Plugins tab
3. Restart Wan2GP

## Usage

After installation, a new **📊 Models** tab will appear in Wan2GP.

### Quick Start

1. Click the **📊 Models** tab
2. View summary: Total models, Downloaded count, Missing count
3. Browse the model table to see status of all models
4. Use filters to find specific models (e.g., "Downloaded only", "Fast models", "≤12GB VRAM")
5. Click a model in the details dropdown for in-depth information

### Features in Detail

#### Model Table
- **Status**: ✓ (downloaded) or ✗ (missing)
- **Model Name**: Human-readable model name
- **Speed**: Fast/Medium/Slow tier
- **VRAM**: Minimum-Recommended GB range
- **Quality**: Low/Medium/High/Highest tier
- **Size**: File size in GB (or estimated)
- **Best For**: Recommended use case

#### Filters
- **Status**: Show all, downloaded only, or missing only
- **Speed**: Filter by Fast/Medium/Slow tiers
- **VRAM**: Filter by VRAM budget (≤12GB, ≤24GB, >24GB)
- **Sort By**: Sort table by Name, Status, Speed, VRAM, or Quality

#### Model Details
- Select a model from the dropdown to view:
  - Download status and file locations
  - Architecture and description
  - Detailed performance metrics
  - File list with paths and sizes
  - Recommendations for similar faster/higher quality models

#### Export Report
- Click **📄 Export Report** to generate a markdown file with your complete model inventory
- File is saved in current directory with timestamp: `model_inventory_YYYYMMDD_HHMMSS.md`

## Screenshots

*(Screenshots will be added after implementation)*

## Troubleshooting

### Plugin doesn't appear after installation
- Make sure you enabled the plugin checkbox in the Plugins tab
- Restart Wan2GP completely
- Check console for error messages

### Model status shows incorrect
- Click **🔄 Refresh** to re-scan your checkpoint directories
- Verify your checkpoint paths are configured correctly in Wan2GP settings

### Missing performance data for a model
- The plugin uses a database of known models plus estimation for unknown ones
- You can contribute metrics by opening an issue or pull request

## Development

### Project Structure
```
wan2gp-model-tracker/
├── __init__.py           # Python package marker
├── plugin.py             # Main plugin class
├── model_analyzer.py     # Model detection logic
├── performance_db.py     # Performance metrics database
├── PLAN.md              # Detailed implementation plan
└── README.md            # This file
```

### Contributing

Contributions are welcome! Areas where you can help:

1. **Performance Data** - Add metrics for more models to `performance_db.py`
2. **UI Improvements** - Enhance the Gradio interface
3. **Features** - Implement items from the roadmap below
4. **Bug Fixes** - Report and fix issues

### Roadmap

- [ ] v1.0: MVP (status tracking, filtering, details view)
- [ ] v1.1: Download queue (batch download missing models)
- [ ] v1.2: Disk management (cleanup unused models)
- [ ] v1.3: Benchmarking (run actual speed tests)
- [ ] v1.4: Smart recommendations (usage-based suggestions)

## License

MIT

## Credits

Created for the Wan2GP community to solve the "which models do I have?" problem.

## Support

- Report issues: [GitHub Issues](https://github.com/[your-username]/wan2gp-model-tracker/issues)
- Wan2GP Discord: *(link if available)*
- Wan2GP Documentation: *(link if available)*
