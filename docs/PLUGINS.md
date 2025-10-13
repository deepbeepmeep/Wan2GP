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
    *   [Example 2: Modifying an Existing Component](#example-2-modifying-an-existing-component)
    *   [Example 3: Advanced UI Injection and Interaction](#example-3-advanced-ui-injection-and-interaction)
    *   [Example 4: Accessing Global Functions and Variables](#example-4-accessing-global-functions-and-variables)
    *   [Example 5: Using Helper Modules (Relative Imports)](#example-5-using-helper-modules-relative-imports)
6.  [Finding Component IDs](#finding-component-ids)

## Plugin Structure

Plugins are no longer single files. They are now standard Python packages (folders) located within the main `plugins/` directory. This structure allows for multiple files, dependencies, and proper packaging.

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
    ├── setup.py            # (Optional) For more complex installation or packaging.
    └── ...                 # Other helper .py files, assets, etc.
```

## Getting Started: Creating a Plugin

1.  **Create a Plugin Folder**: Inside the main `plugins/` directory, create a new folder for your plugin (e.g., `my-awesome-plugin`).

2.  **Create Core Files**:
    *   Inside `my-awesome-plugin/`, create an empty file named `__init__.py`.
    *   Create another file named `plugin.py`. This will be the entry point for your plugin.

3.  **Define a Plugin Class**: In `plugin.py`, create a class that inherits from `WAN2GPPlugin`.

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
4.  The system will clone the repository, install any dependencies from `requirements.txt`, and run `setup.py` if present.
5.  The new plugin will appear in the "Available Plugins" list. The user must then enable it and restart the application.

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
        self.name = "My Awesome Plugin"
        self.version = "1.0.0"
        # ... your initialization code here
```

### Core Methods
These are the methods you can override or call to build your plugin.

#### `setup_ui(self)`
This method is called when your plugin is first loaded. It's the place to declare new tabs or request access to components and globals.

*   **`self.add_tab(tab_id, label, component_constructor, position)`**: Adds a new top-level tab.
*   **`self.request_component(component_id)`**: Requests access to an existing Gradio component. The component will be available as an attribute (e.g., `self.loras_multipliers`) in `post_ui_setup`.
*   **`self.request_global(global_name)`**: Requests access to a global variable or function from the main application. The global will be available as an attribute (e.g., `self.server_config`).

#### `post_ui_setup(self, components)`
This method runs after the entire main UI has been built. Use it to wire up events and make initial modifications.

*   `components` (dict): A dictionary of the components you requested.
*   **Return Value**: Can be used to update components on load, but it's now recommended to use `self.insert_after` for UI changes and wire events directly.

#### `self.insert_after(target_component_id, new_component_constructor)`
Call this inside `post_ui_setup` to dynamically inject UI elements.

*   `target_component_id` (str): The `elem_id` of the existing component after which your new UI will be inserted.
*   `new_component_constructor` (callable): A function that creates and returns the new Gradio component(s).

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

    def setup_ui(self):
        self.add_tab(
            tab_id="greeter_tab",
            label="Greeter",
            component_constructor=self.create_greeter_ui,
            position=2 # Place it as the 3rd tab
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

### Example 2: Modifying an Existing Component

**File Structure:**
```
plugins/
└── simple_modifier_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/simple_modifier_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class SimpleModifierPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Simple Modifier"
        self.version = "1.0.0"

    def setup_ui(self):
        self.request_component("loras_multipliers")

    def post_ui_setup(self, components: dict):
        # The component is now an attribute of self
        if not hasattr(self, 'loras_multipliers'):
            return {}
            
        def create_inserted_component():
            return gr.HTML(value="<div style='padding: 10px; background: #eee;'>Inserted by a plugin!</div>")

        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_inserted_component
        )

        # To update the component on load, you can also wire into the main 'load' event
        # but for simplicity, the original return method is shown here.
        return {
            self.loras_multipliers: gr.update(value="Hello from the Simple Modifier plugin!")
        }
```

### Example 3: Advanced UI Injection and Interaction

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
    def setup_ui(self):
        self.request_component("loras_multipliers")
        self.request_component("loras_choices")
        self.request_component("main") 

    def post_ui_setup(self, components: dict):
        def create_and_wire_advanced_ui():
            with gr.Accordion("Advanced Plugin Panel", open=True) as panel:
                info_md = gr.Markdown("This panel is inserted by a plugin.")
                copy_btn = gr.Button("Copy selected LoRAs to multiplier textbox")

            def copy_lora_names(selected_loras):
                return ", ".join(selected_loras)

            copy_btn.click(
                fn=copy_lora_names,
                inputs=[self.loras_choices],
                outputs=[self.loras_multipliers]
            )
            return panel

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
    def setup_ui(self):
        self.request_global("server_config")
        self.request_global("get_video_info")
        self.add_tab("global_access_tab", "Global Access", self.create_ui)
        
    def create_ui(self):
        with gr.Blocks() as demo:
            video_input = gr.Video(label="Upload a video")
            info_output = gr.JSON(label="Video Info")
            
            def analyze_video(video_path):
                if not video_path: return "Upload a video."
                save_path = self.server_config.get("save_path", "outputs")
                fps, w, h, frames = self.get_video_info(video_path)
                return {"save_path": save_path, "fps": fps, "dimensions": f"{w}x{h}"}

            analyze_btn = gr.Button("Analyze Video")
            analyze_btn.click(fn=analyze_video, inputs=[video_input], outputs=[info_output])
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

To interact with an existing component, you need its `elem_id`. You can find these IDs by:

1.  **Inspecting the Source Code**: Look through `wgp.py` and other UI-related files for Gradio components defined with an `elem_id`.
2.  **Browser Developer Tools**: If an `elem_id` is not set, run Wan2GP, open your browser's developer tools (F12), and inspect the HTML to find the ID of the element you want to target.