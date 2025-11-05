# Wan2GP Plugin System

This system allows you to extend and customize the Wan2GP user interface and functionality without modifying the core application code. This document will guide you through the process of creating and installing your own plugins.

## Table of Contents
1.  [Plugin Structure](#plugin-structure)
2.  [Getting Started: Creating a Plugin](#getting-started-creating-a-plugin)
3.  [Plugin Distribution and Installation](#plugin-distribution-and-installation)
4.  [Plugin API Reference](#plugin-api-reference)
    *   [The `WAN2GPPlugin` Class](#the-wan2gpplugin-class)
    *   [Core Methods](#core-methods)
5.  [Examples](#examples)
    *   [Example 1: Creating a New Tab](#example-1-creating-a-new-tab)
    *   [Example 2: Injecting UI Elements](#example-2-injecting-ui-elements)
    *   [Example 3: Advanced UI Injection and Interaction](#example-3-advanced-ui-injection-and-interaction)
    *   [Example 4: Accessing Global Functions and Variables](#example-4-accessing-global-functions-and-variables)
    *   [Example 5: Using Helper Modules (Relative Imports)](#example-5-using-helper-modules-relative-imports)
6.  [Finding Component IDs](#finding-component-ids)

## Plugin Structure

Plugins are standard Python packages (folders) located within the main `plugins/` directory. This structure allows for multiple files, dependencies, and proper packaging.

Don't hesitate to have a look at the Sample PlugIn "wan2gp_sample" as it illustrates:
-How to get Settings from the Main Form and then Modify them
-How to suspend the Video Gen (and release VRAM) to execute your own GPU intensive process.
-How to switch back automatically to the Main Tab

A valid plugin folder must contain at a minimum:
*   `__init__.py`: An empty file that tells Python to treat the directory as a package.
*   `plugin.py`: The main file containing your class that inherits from `WAN2GPPlugin`.


A complete plugin folder typically looks like this:

```
plugins/
└── my-awesome-plugin/
    ├── __init__.py         # (Required, can be empty) Makes this a Python package.
    ├── plugin.py           # (Required) Main plugin logic and class definition.
    ├── requirements.txt    # (Optional) Lists pip dependencies for your plugin.
    └── ...                 # Other helper .py files, assets, etc.
```

## Getting Started: Creating a Plugin

1.  **Create a Plugin Folder**: Inside the main `plugins/` directory, create a new folder for your plugin (e.g., `my-awesome-plugin`).

2.  **Create Core Files**:
    *   Inside `my-awesome-plugin/`, create an empty file named `__init__.py`.
    *   Create another file named `plugin.py`. This will be the entry point for your plugin.

3.  **Define a Plugin Class**: In `plugin.py`, create a class that inherits from `WAN2GPPlugin` and set its metadata attributes.

    ```python
    from shared.utils.plugins import WAN2GPPlugin

    class MyPlugin(WAN2GPPlugin):
        def __init__(self):
            super().__init__()
            self.name = "My Awesome Plugin"
            self.version = "1.0.0"
            self.description = "This plugin adds awesome new features."
    ```

4.  **Add Dependencies (Optional)**: If your plugin requires external Python libraries (e.g., `numpy`), list them in a `requirements.txt` file inside your plugin folder. These will be installed automatically when a user installs your plugin via the UI.

5.  **Enable and Test**:
    *   Start Wan2GP.
    *   Go to the **Plugins** tab.
    *   You should see your new plugin (`my-awesome-plugin`) in the list.
    *   Check the box to enable it and click "Save Settings".
    *   **Restart the Wan2GP application.** Your plugin will now be active.

## Plugin Distribution and Installation

#### Packaging for Distribution
To share your plugin, simply upload your entire plugin folder (e.g., `my-awesome-plugin/`) to a public GitHub repository.

#### Installing from the UI
Users can install your plugin directly from the Wan2GP interface:
1.  Go to the **Plugins** tab.
2.  Under "Install New Plugin," paste the full URL of your plugin's GitHub repository.
3.  Click "Download and Install Plugin."
4.  The system will clone the repository and install any dependencies from `requirements.txt`.
5.  The new plugin will appear in the "Available Plugins" list. The user must then enable it and restart the application to activate it.

The plugin manager also supports updating plugins (if installed from git) and uninstalling them.

## Plugin API Reference

### The `WAN2GPPlugin` Class
Every plugin must define its main class in `plugin.py` inheriting from `WAN2GPPlugin`.

```python
# in plugins/my-awesome-plugin/plugin.py
from shared.utils.plugins import WAN2GPPlugin
import gradio as gr

class MyAwesomePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        # Metadata for the Plugin Manager UI
        self.name = "My Awesome Plugin"
        self.version = "1.0.0"
        self.description = "A short description of what my plugin does."
        
    def setup_ui(self):
        # UI setup calls go here
        pass
        
    def post_ui_setup(self, components: dict):
        # Event wiring and UI injection calls go here
        pass
```

### Core Methods
These are the methods you can override or call to build your plugin.

#### `setup_ui(self)`
This method is called when your plugin is first loaded. It's the place to declare new tabs or request access to components and globals before the main UI is built.

*   **`self.add_tab(tab_id, label, component_constructor, position)`**: Adds a new top-level tab to the UI.
*   **`self.request_component(component_id)`**: Requests access to an existing Gradio component by its `elem_id`. The component will be available as an attribute (e.g., `self.loras_multipliers`) in `post_ui_setup`.
*   **`self.request_global(global_name)`**: Requests access to a global variable or function from the main `wgp.py` application. The global will be available as an attribute (e.g., `self.server_config`).

#### `post_ui_setup(self, components)`
This method runs after the entire main UI has been built. Use it to wire up events for your custom UI and to inject new components into the existing layout.

*   `components` (dict): A dictionary of the components you requested via `request_component`.
*   **`self.insert_after(target_component_id, new_component_constructor)`**: A powerful method to dynamically inject new UI elements into the page.

#### `on_tab_select(self, state)` and `on_tab_deselect(self, state)`
If you used `add_tab`, these methods will be called automatically when your tab is selected or deselected, respectively. This is useful for loading data or managing resources.

#### `set_global(self, variable_name, new_value)`
Allows your plugin to safely modify a global variable in the main `wgp.py` application.

#### `register_data_hook(self, hook_name, callback)`
Allows you to intercept and modify data at key points. For example, the `before_metadata_save` hook lets you add custom data to the metadata before it's saved to a file.

## Examples

### Example 1: Creating a New Tab

**File Structure:**
```
plugins/
└── greeter_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/greeter_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class GreeterPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Greeter Plugin"
        self.version = "1.0.0"
        self.description = "Adds a simple 'Greeter' tab."

    def setup_ui(self):
        self.add_tab(
            tab_id="greeter_tab",
            label="Greeter",
            component_constructor=self.create_greeter_ui,
            position=2 # Place it as the 3rd tab (0-indexed)
        )
        
    def create_greeter_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## A Simple Greeter")
            with gr.Row():
                name_input = gr.Textbox(label="Enter your name")
                output_text = gr.Textbox(label="Output")
            greet_btn = gr.Button("Greet")
            
            greet_btn.click(
                fn=lambda name: f"Hello, {name}!",
                inputs=[name_input],
                outputs=output_text
            )
        return demo
```

### Example 2: Injecting UI Elements

This example adds a simple HTML element right after the "Loras Multipliers" textbox.

**File Structure:**
```
plugins/
└── injector_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/injector_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class InjectorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "UI Injector"
        self.version = "1.0.0"
        self.description = "Injects a message into the main UI."

    def post_ui_setup(self, components: dict):
        def create_inserted_component():
            return gr.HTML(value="<div style='padding: 10px; color: gray; text-align: center;'>--- Injected by a plugin! ---</div>")

        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_inserted_component
        )
```

### Example 3: Advanced UI Injection and Interaction

This plugin injects a button that interacts with other components on the page.

**File Structure:**
```
plugins/
└── advanced_ui_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/advanced_ui_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class AdvancedUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "LoRA Helper"
        self.description = "Adds a button to copy selected LoRAs."
        
    def setup_ui(self):
        # Request access to the components we want to read from and write to.
        self.request_component("loras_multipliers")
        self.request_component("loras_choices")

    def post_ui_setup(self, components: dict):
        # This function will create our new UI and wire its events.
        def create_and_wire_advanced_ui():
            with gr.Accordion("LoRA Helper Panel (Plugin)", open=False):
                copy_btn = gr.Button("Copy selected LoRA names to Multipliers")

            # Define the function for the button's click event.
            def copy_lora_names(selected_loras):
                return " ".join(selected_loras)

            # Wire the event. We can access the components as attributes of `self`.
            copy_btn.click(
                fn=copy_lora_names,
                inputs=[self.loras_choices],
                outputs=[self.loras_multipliers]
            )
            return panel # Return the top-level component to be inserted.

        # Tell the manager to insert our UI after the 'loras_multipliers' textbox.
        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_and_wire_advanced_ui
        )
```

### Example 4: Accessing Global Functions and Variables

**File Structure:**
```
plugins/
└── global_access_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/global_access_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class GlobalAccessPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Global Access Plugin"
        self.description = "Demonstrates reading and writing global state."

    def setup_ui(self):
        # Request read access to globals
        self.request_global("server_config")
        self.request_global("get_video_info")
        
        # Add a tab to host our UI
        self.add_tab("global_access_tab", "Global Access", self.create_ui)
        
    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("### Read Globals")
            video_input = gr.Video(label="Upload a video to analyze")
            info_output = gr.JSON(label="Video Info")
            
            def analyze_video(video_path):
                if not video_path: return "Upload a video."
                # Access globals as attributes of `self`
                save_path = self.server_config.get("save_path", "outputs")
                fps, w, h, frames = self.get_video_info(video_path)
                return {"save_path": save_path, "fps": fps, "dimensions": f"{w}x{h}"}

            analyze_btn = gr.Button("Analyze Video")
            analyze_btn.click(fn=analyze_video, inputs=[video_input], outputs=[info_output])

            gr.Markdown("--- \n ### Write Globals")
            theme_changer = gr.Dropdown(choices=["default", "gradio"], label="Change UI Theme (Requires Restart)")
            save_theme_btn = gr.Button("Save Theme Change")

            def save_theme(theme_choice):
                # Use the safe `set_global` method
                self.set_global("UI_theme", theme_choice)
                gr.Info(f"Theme set to '{theme_choice}'. Restart required.")

            save_theme_btn.click(fn=save_theme, inputs=[theme_changer])

        return demo
```

### Example 5: Using Helper Modules (Relative Imports)
This example shows how to organize your code into multiple files within your plugin package.

**File Structure:**
```
plugins/
└── helper_plugin/
    ├── __init__.py
    ├── plugin.py
    └── helpers.py
```

**Code:**
```python
# in plugins/helper_plugin/helpers.py
def format_greeting(name: str) -> str:
    """A helper function in a separate file."""
    if not name:
        return "Hello, mystery person!"
    return f"A very special hello to {name.upper()}!"

# in plugins/helper_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from .helpers import format_greeting # <-- Relative import works!

class HelperPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Helper Module Example"
        self.description = "Shows how to use relative imports."

    def setup_ui(self):
        self.add_tab("helper_tab", "Helper Example", self.create_ui)

    def create_ui(self):
        with gr.Blocks() as demo:
            name_input = gr.Textbox(label="Name")
            output = gr.Textbox(label="Formatted Greeting")
            btn = gr.Button("Greet with Helper")
            
            btn.click(fn=format_greeting, inputs=[name_input], outputs=[output])
        return demo
```

## Finding Component IDs

To interact with an existing component using `request_component` or `insert_after`, you need its `elem_id`. You can find these IDs by:

1.  **Inspecting the Source Code**: Look through `wgp.py` for Gradio components defined with an `elem_id`.
2.  **Browser Developer Tools**: Run Wan2GP, open your browser's developer tools (F12), and use the "Inspect Element" tool to find the `id` of the HTML element you want to target.

Some common `elem_id`s include:
*   `loras_multipliers`
*   `loras_choices`
*   `main_tabs`
*   `gallery`
*   `family_list`, `model_base_types_list`, `model_list`