# Wan2GP Plugin System

This system allows you to extend and customize the Wan2GP user interface and functionality without modifying the core application code. This document will guide you through the process of creating your own plugins.

## Table of Contents
1.  [Getting Started](#getting-started)
2.  [Plugin API Reference](#plugin-api-reference)
    *   [The `WAN2GPPlugin` Class](#the-wan2gpplugin-class)
    *   [Core Methods](#core-methods)
3.  [Examples](#examples)
    *   [Example 1: Creating a New Tab](#example-1-creating-a-new-tab)
    *   [Example 2: Modifying an Existing Component](#example-2-modifying-an-existing-component)
    *   [Example 3: Advanced UI Injection and Interaction](#example-3-advanced-ui-injection-and-interaction)
    *   [Example 4: Accessing Global Functions and Variables](#example-4-accessing-global-functions-and-variables)
4.  [Finding Component IDs](#finding-component-ids)

## Getting Started

1.  **Create a Python File**: Your plugin will be a single Python file located in the `plugins/` directory.
2.  **Define a Plugin Class**: Inside your file, create a class that inherits from `WAN2GPPlugin`.
3.  **Implement Your Logic**: Use the methods provided by the `WAN2GPPlugin` base class to add tabs, interact with existing components, and define new functionality.

The application will automatically discover and load any valid plugin file from the `plugins/` directory upon startup.

## Plugin API Reference

### The `WAN2GPPlugin` Class

Every plugin must define a class that inherits from `WAN2GPPlugin`.

```python
# in plugins/my_awesome_plugin.py
from plugin_system import WAN2GPPlugin
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
This method is called during the initial UI construction phase. It's the primary place to declare new tabs or request access to components and globals you'll need later.

*   **`self.add_tab(tab_id, label, component_constructor, position)`**: Adds a new top-level tab to the main UI.
    *   `tab_id` (str): A unique identifier for your tab.
    *   `label` (str): The text that will appear on the tab.
    *   `component_constructor` (callable): A function that creates the Gradio interface for this tab. It should return a Gradio `Blocks` instance.
    *   `position` (int, optional): The desired position of the tab (e.g., `1` for the second position).

*   **`self.request_component(component_id)`**: Requests access to an existing Gradio component from the main application.
    *   `component_id` (str): The `elem_id` of the Gradio component you want to interact with.
    *   The requested component will be injected as an attribute of your plugin instance (e.g., `self.loras_multipliers`) and will be available in the `post_ui_setup` method.

*   **`self.request_global(global_name)`**: Requests access to a global variable or function from the main application scope (`wgp.py`).
    *   `global_name` (str): The name of the global variable/function.
    *   The requested global will be injected as an attribute of your plugin instance (e.g., `self.server_config`).

#### `post_ui_setup(self, components)`
This method is executed after the entire main UI has been built. It's the place to wire up event listeners and make initial modifications to components.

*   `components` (dict): A dictionary containing the Gradio components you requested with `self.request_component`.
*   **Return Value**: The method should return a dictionary where keys are Gradio component objects and values are `gr.update(...)` instances. This is used to modify the initial state of components when the UI loads.

```python
def post_ui_setup(self, components: dict):
    # Get the component you requested
    loras_multipliers_textbox = components.get("loras_multipliers")

    # Change the initial value of the textbox on UI load
    return {
        loras_multipliers_textbox: gr.update(value="Value set by a plugin!")
    }
```

#### `self.insert_after(target_component_id, new_component_constructor)`
Call this method inside `post_ui_setup` to dynamically inject your own UI elements into the main application's layout.

*   `target_component_id` (str): The `elem_id` of the existing component after which your new UI will be inserted.
*   `new_component_constructor` (callable): A function that creates and returns the new Gradio component(s). This function will be executed within the proper Gradio `Blocks` context.

## Examples

### Example 1: Creating a New Tab
This plugin adds a new tab with a simple "Hello World" interface. This is based on `example_plugin.py`.

```python
# in plugins/greeter_plugin.py
import gradio as gr
from plugin_system import WAN2GPPlugin

class GreeterPlugin(WAN2GPPlugin):
    
    def __init__(self):
        super().__init__()
        self.name = "Greeter Plugin"
        self.version = "1.0.0"

    # setup_ui is called first
    def setup_ui(self):
        # We declare that we want to add a tab
        self.add_tab(
            tab_id="greeter_tab",
            label="Greeter",
            component_constructor=self.create_greeter_ui,
            position=2 # Place it as the 3rd tab
        )
        
    # This function builds the UI for our tab
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
This plugin demonstrates how to change the value of an existing component (`loras_multipliers`) and insert a new HTML component directly after it. This is based on `lora_setter_plugin.py`.

```python
# in plugins/simple_modifier_plugin.py
import gradio as gr
from plugin_system import WAN2GPPlugin

class SimpleModifierPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Simple Modifier"
        self.version = "1.0.0"

    def setup_ui(self):
        # Request access to the loras_multipliers textbox
        self.request_component("loras_multipliers")

    def post_ui_setup(self, components: dict):
        # Get the component we requested in setup_ui
        loras_multipliers_textbox = components.get("loras_multipliers")
        if loras_multipliers_textbox is None:
            return {}
            
        # This function will create our new component
        def create_inserted_component():
            return gr.HTML(
                value="<div style='padding: 10px; background: #eee; border: 1px solid #ccc;'>"
                      "This HTML block was inserted by a plugin!"
                      "</div>"
            )

        # Schedule the insertion to happen after the UI is built
        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_inserted_component
        )

        # Return a dictionary to update the component's initial value
        return {
            loras_multipliers_textbox: gr.update(value="Hello from the Simple Modifier plugin!")
        }
```

### Example 3: Advanced UI Injection and Interaction
This example, inspired by `lora_multipliers_ui.py`, shows a more complex use case: injecting a full UI panel and wiring its events to both existing and new components.

```python
# in plugins/advanced_ui_plugin.py
import gradio as gr
from plugin_system import WAN2GPPlugin

class AdvancedUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Advanced UI Plugin"
        self.version = "1.0.0"

    def setup_ui(self):
        # Request components we want to interact with
        self.request_component("loras_multipliers")
        self.request_component("loras_choices")
        self.request_component("main") # The main Gradio Blocks instance

    def post_ui_setup(self, components: dict):
        loras_multipliers_textbox = components.get("loras_multipliers")
        loras_choices_dropdown = components.get("loras_choices")
        main_ui = components.get("main")

        # This function defines our new UI and its logic
        def create_and_wire_advanced_ui():
            with gr.Accordion("Advanced Plugin Panel", open=True):
                info_md = gr.Markdown("This panel is inserted by a plugin.")
                copy_btn = gr.Button("Copy selected LoRAs to multiplier textbox")

            # Event handling for our new component
            def copy_lora_names(selected_loras):
                return ", ".join(selected_loras)

            copy_btn.click(
                fn=copy_lora_names,
                inputs=[loras_choices_dropdown],
                outputs=[loras_multipliers_textbox]
            )
            
            # Event handling for an existing component
            def on_main_load(choices):
                print(f"Plugin sees that UI has loaded with these choices: {choices}")

            main_ui.load(fn=on_main_load, inputs=[loras_choices_dropdown])

            return info_md.parent # Return the Accordion

        # Insert our new UI
        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_and_wire_advanced_ui
        )
        
        return {} # No initial updates needed
```

### Example 4: Accessing Global Functions and Variables
This example, inspired by `gallery.py`, shows how to request and use functions and variables from the main application.

```python
# in plugins/global_access_plugin.py
import gradio as gr
from plugin_system import WAN2GPPlugin

class GlobalAccessPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Global Access Plugin"

    def setup_ui(self):
        # Request access to a global variable (a dictionary)
        self.request_global("server_config")
        # Request access to a global function
        self.request_global("get_video_info")
        
        self.add_tab(
            tab_id="global_access_tab",
            label="Global Access",
            component_constructor=self.create_ui
        )
        
    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Accessing Globals")
            video_input = gr.Video(label="Upload a video")
            info_output = gr.JSON(label="Video Info")
            
            # This function will be called by the button click
            def analyze_video(video_path):
                if not video_path:
                    return "Upload a video first."
                
                # Access the globals requested in setup_ui
                save_path = self.server_config.get("save_path", "outputs")
                fps, w, h, frames = self.get_video_info(video_path)
                
                return {
                    "save_path_from_config": save_path,
                    "fps": fps,
                    "dimensions": f"{w}x{h}",
                    "frame_count": frames
                }

            analyze_btn = gr.Button("Analyze Video")
            analyze_btn.click(fn=analyze_video, inputs=[video_input], outputs=[info_output])
            
        return demo
```

## Finding Component IDs

To interact with an existing component, you need its `elem_id`. You can find these IDs by:

1.  **Inspecting the Source Code**: Look through `wgp.py` and other UI-related files for Gradio components defined with an `elem_id`, for example:
    ```python
    self.loras_multipliers = gr.Textbox(elem_id="loras_multipliers", ...)
    ```
2.  **Browser Developer Tools**: If an `elem_id` is not explicitly set, you can run Wan2GP, open your browser's developer tools (usually by pressing F12), and inspect the HTML to find the ID of the element you want to target. Gradio components are often wrapped in a `<div>` with an ID.