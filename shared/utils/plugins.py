import os
import sys
import importlib.util
import inspect
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import gradio as gr
import traceback

@dataclass
class InsertAfterRequest:
    target_component_id: str
    new_component_constructor: callable

@dataclass
class PluginTab:
    id: str
    label: str
    component_constructor: callable
    position: int = -1

class WAN2GPPlugin:
    def __init__(self):
        self.tabs: Dict[str, PluginTab] = {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self._component_requests: List[str] = []
        self._global_requests: List[str] = []
        self._insert_after_requests: List[InsertAfterRequest] = []
        self._setup_complete = False
        
    def setup_ui(self) -> None:
        pass
        
    def add_tab(self, tab_id: str, label: str, component_constructor: callable, position: int = -1):
        self.tabs[tab_id] = PluginTab(id=tab_id, label=label, component_constructor=component_constructor, position=position)

    def request_component(self, component_id: str) -> None:
        if component_id not in self._component_requests:
            self._component_requests.append(component_id)
    def request_global(self, global_name: str) -> None:
        if global_name not in self._global_requests:
            self._global_requests.append(global_name)
    @property
    def component_requests(self) -> List[str]:
        return self._component_requests.copy()

    @property
    def global_requests(self) -> List[str]:
        return self._global_requests.copy()

    def insert_after(self, target_component_id: str, new_component_constructor: callable) -> None:
        if not hasattr(self, '_insert_after_requests'):
            self._insert_after_requests = []
        self._insert_after_requests.append(
            InsertAfterRequest(
                target_component_id=target_component_id,
                new_component_constructor=new_component_constructor
            )
        )
        
    def post_ui_setup(self, components: Dict[str, gr.components.Component]) -> Dict[gr.components.Component, Union[gr.update, Any]]:
        return {}

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, WAN2GPPlugin] = {}
        self._load_builtin_plugins()
        
    def _load_builtin_plugins(self) -> None:
        pass
    
    def load_plugin(self, plugin_path: str) -> bool:
        try:
            module_name = os.path.splitext(os.path.basename(plugin_path))[0]
            
            spec = importlib.util.spec_from_file_location(f"wan2gp_plugins.{module_name}", plugin_path)
            if spec is None or spec.loader is None:
                print(f"Error: Could not load plugin {plugin_path}")
                return False
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                    plugin = obj()
                    plugin.setup_ui()
                    self.plugins[module_name] = plugin
                    print(f"Loaded plugin: {plugin.name}")
                    return True
            return False
            
        except Exception as e:
            print(f"Error loading plugin {plugin_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_plugins_from_directory(self, directory: str) -> None:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
            return
            
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                self.load_plugin(os.path.join(directory, filename))
    
    def get_plugin(self, plugin_id: str) -> Optional[WAN2GPPlugin]:
        return self.plugins.get(plugin_id)
    
    def get_all_plugins(self) -> Dict[str, WAN2GPPlugin]:
        return self.plugins.copy()

    def inject_globals(self, global_references: Dict[str, Any]) -> None:
        for plugin_id, plugin in self.plugins.items():
            try:
                for global_name in plugin.global_requests:
                    if global_name in global_references:
                        setattr(plugin, global_name, global_references[global_name])
            except Exception as e:
                print(f"  [!] ERROR injecting globals for {plugin_id}: {str(e)}")

    def setup_ui(self) -> Dict[str, Dict[str, Any]]:
        tabs = {}
        
        for plugin_id, plugin in self.plugins.items():
            try:
                for tab_id, tab in plugin.tabs.items():
                    tabs[tab_id] = {
                        'label': tab.label,
                        'component_constructor': tab.component_constructor,
                        'position': tab.position
                    }
            except Exception as e:
                print(f"Error in setup_ui for plugin {plugin_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        return {'tabs': tabs}

    def run_post_ui_setup(self, all_components: Dict[str, gr.components.Component]) -> None:
        all_insert_requests: List[InsertAfterRequest] = []

        for plugin_id, plugin in self.plugins.items():
            try:
                for comp_id in plugin.component_requests:
                    if comp_id in all_components:
                        setattr(plugin, comp_id, all_components[comp_id])

                requested_components = {
                    comp_id: all_components[comp_id]
                    for comp_id in plugin.component_requests
                    if comp_id in all_components
                }

                updates = plugin.post_ui_setup(requested_components)
                if updates:
                    for component, update_instruction in updates.items():
                        new_value = None
                        if isinstance(update_instruction, dict) and update_instruction.get("__type__") == "update":
                            if 'value' in update_instruction:
                                new_value = update_instruction['value']
                        elif isinstance(update_instruction, gr.components.Component):
                            if hasattr(update_instruction, 'value'):
                                print(f"Warning: Plugin '{plugin.name}' is returning a new component instance for a value update. Please use gr.update(value=...) instead.")
                                new_value = getattr(update_instruction, 'value')
                        else:
                            new_value = update_instruction

                        if new_value is not None:
                            component.value = new_value
                
                all_insert_requests.extend(getattr(plugin, '_insert_after_requests', []))
                getattr(plugin, '_insert_after_requests', []).clear()

            except Exception as e:
                print(f"[PluginManager] Error in post_ui_setup for {plugin_id}: {str(e)}")

        if all_insert_requests:
            for request in all_insert_requests:
                target_id = request.target_component_id
                constructor = request.new_component_constructor
                
                try:
                    if target_id not in all_components:
                        print(f"[PluginManager] ERROR: Target '{target_id}' for insertion not found.")
                        continue
                        
                    target = all_components[target_id]
                    parent = getattr(target, 'parent', None)
                    if parent is None or not hasattr(parent, 'children'):
                        continue

                    children = list(parent.children)
                    target_index = children.index(target)

                    with parent:
                        new_component = constructor()

                    newly_added_component = parent.children.pop(-1)
                    parent.children.insert(target_index + 1, newly_added_component)

                    print(f"[PluginManager] Successfully inserted {type(new_component).__name__} after {target_id}")

                except Exception as e:
                    print(f"[PluginManager] Error processing insert_after for {target_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()

class WAN2GPApplication:
    def __init__(self):
        self.plugin_manager = None
        self.ui_components = {}

    def initialize_plugin_manager(self) -> None:
        try:
            self.plugin_manager = PluginManager()
            plugins_dir = os.path.join(os.getcwd(), "plugins")
            os.makedirs(plugins_dir, exist_ok=True)
            self.plugin_manager.load_plugins_from_directory(plugins_dir)
        except Exception as e:
            print(f"Error initializing plugin manager: {str(e)}")
            import traceback
            traceback.print_exc()

    def setup_plugins_ui(self, main_module_globals: Dict[str, Any]):
        self.initialize_plugin_manager()
        if self.plugin_manager:
            self.plugin_manager.inject_globals(main_module_globals)
            plugin_ui = self.plugin_manager.setup_ui()
            plugin_tabs = plugin_ui.get('tabs', {})
            sorted_plugin_tabs = sorted(
                plugin_tabs.items(),
                key=lambda x: x[1].get('position', -1)
            )
            return sorted_plugin_tabs
        return []

    def run_post_ui_setup(self):
        if hasattr(self, 'ui_components') and self.plugin_manager:
            self.plugin_manager.run_post_ui_setup(self.ui_components)