# Installation Troubleshooting Guide

## Issue: "Plugin already exists" but doesn't appear in UI

If you're getting an error that the plugin already exists but don't see it in the Plugins menu or as a tab, follow these manual installation steps:

### Manual Installation Steps

1. **Locate your Wan2GP plugins directory**
   - Usually: `<wan2gp-installation>/plugins/`
   - On Windows: `C:\...\wan2gp\plugins\`
   - On Linux/Mac: `~/wan2gp/plugins/` or `/opt/wan2gp/plugins/`

2. **Check if partial installation exists**
   ```bash
   cd <wan2gp-path>/plugins/
   ls -la wan2gp-model-tracker/
   ```

3. **Remove any existing installation**
   ```bash
   rm -rf wan2gp-model-tracker/
   ```

4. **Clone the plugin fresh**
   ```bash
   cd <wan2gp-path>/plugins/
   git clone https://github.com/hnethery/wan2gp-model-tracker.git
   ```

5. **Verify files are present**
   The directory should contain:
   - `__init__.py`
   - `plugin.py`
   - `model_analyzer.py`
   - `performance_db.py`
   - `ui_components.py`
   - `manifest.json`
   - `requirements.txt`

6. **Restart Wan2GP completely**
   - Close all Wan2GP windows
   - Restart the application

7. **Enable the plugin**
   - Go to Plugins tab
   - Find "Model Tracker" in the list
   - Check the enable checkbox
   - Click "Save Settings"
   - Restart Wan2GP again

8. **Look for the tab**
   - After restart, you should see "📊 Models" tab

## Alternative: Direct File Copy

If git clone doesn't work:

1. Download the repository as ZIP
2. Extract it
3. Copy the entire `wan2gp-model-tracker` folder to `<wan2gp-path>/plugins/`
4. Follow steps 6-8 above

## Still Not Working?

If the plugin still doesn't appear after manual installation:

1. **Check Wan2GP console/logs for errors**
   - Look for any Python import errors
   - Check for missing dependencies

2. **Verify Python dependencies**
   - Ensure `gradio` and `pandas` are installed in your Wan2GP environment

3. **Check plugin compatibility**
   - This plugin requires Wan2GP to expose:
     - `models_def` dictionary
     - `get_local_model_filename()` function
     - `files_locator` module

4. **Contact for support**
   - Open an issue at: https://github.com/hnethery/wan2gp-model-tracker/issues
   - Include any error messages from Wan2GP console
