import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import gradio as gr
import traceback
import subprocess
import git
import shutil

SYSTEM_PLUGINS = [
    "wan2gp-about",
    "wan2gp-configuration",
    "wan2gp-downloads",
    "wan2gp-guides",
    "wan2gp-plugin-manager",
    "wan2gp-video-mask-creator",
]

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
        self.description = "No description provided."
        self._component_requests: List[str] = []
        self._global_requests: List[str] = []
        self._insert_after_requests: List[InsertAfterRequest] = []
        self._setup_complete = False
        self._data_hooks: Dict[str, List[callable]] = {}
        
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
        
    def register_data_hook(self, hook_name: str, callback: callable):
        if hook_name not in self._data_hooks:
            self._data_hooks[hook_name] = []
        self._data_hooks[hook_name].append(callback)

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
    def __init__(self, plugins_dir="plugins"):
        self.plugins: Dict[str, WAN2GPPlugin] = {}
        self.plugins_dir = plugins_dir
        os.makedirs(self.plugins_dir, exist_ok=True)
        if self.plugins_dir not in sys.path:
            sys.path.insert(0, self.plugins_dir)
        self.data_hooks: Dict[str, List[callable]] = {}

    def get_plugins_info(self) -> List[Dict[str, str]]:
        plugins_info = []
        for dir_name in self.discover_plugins():
            plugin_path = os.path.join(self.plugins_dir, dir_name)
            is_system = dir_name in SYSTEM_PLUGINS
            info = {'id': dir_name, 'name': dir_name, 'version': 'N/A', 'description': 'No description provided.', 'path': plugin_path, 'system': is_system}
            try:
                # Use import_module, which correctly handles packages and relative imports
                module = importlib.import_module(f"{dir_name}.plugin")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                        instance = obj()
                        info['name'] = instance.name
                        info['version'] = instance.version
                        info['description'] = instance.description
                        break
            except Exception as e:
                print(f"Could not load metadata for plugin {dir_name}: {e}")
            plugins_info.append(info)
        
        plugins_info.sort(key=lambda p: (not p['system'], p['name']))
        return plugins_info

    def uninstall_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for uninstallation."
        
        if plugin_id in SYSTEM_PLUGINS:
            return f"[Error] Cannot uninstall system plugin '{plugin_id}'."

        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' directory not found."

        try:
            shutil.rmtree(target_dir)
            return f"[Success] Plugin '{plugin_id}' uninstalled. Please restart WanGP."
        except Exception as e:
            return f"[Error] Failed to remove plugin '{plugin_id}': {e}"

    def update_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for update."
            
        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(os.path.join(target_dir, '.git')):
            return f"[Error] '{plugin_id}' is not a git repository and cannot be updated automatically."

        try:
            gr.Info(f"Attempting to update '{plugin_id}'...")
            repo = git.Repo(target_dir)
            origin = repo.remotes.origin
            origin.fetch()
            
            local_commit = repo.head.commit
            remote_commit = origin.refs[repo.active_branch.name].commit

            if local_commit == remote_commit:
                 return f"[Info] Plugin '{plugin_id}' is already up to date."

            origin.pull()
            
            requirements_path = os.path.join(target_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                gr.Info(f"Re-installing dependencies for '{plugin_id}'...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

            return f"[Success] Plugin '{plugin_id}' updated. Please restart WanGP for changes to take effect."
        except git.exc.GitCommandError as e:
            return f"[Error] Git update failed for '{plugin_id}': {e.stderr}"
        except Exception as e:
            return f"[Error] An unexpected error occurred during update of '{plugin_id}': {str(e)}"

    def reinstall_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for reinstallation."

        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' not found."

        git_url = None
        if os.path.isdir(os.path.join(target_dir, '.git')):
            try:
                repo = git.Repo(target_dir)
                git_url = repo.remotes.origin.url
            except Exception as e:
                return f"[Error] Could not get remote URL for '{plugin_id}': {e}"
        
        if not git_url:
            return f"[Error] Could not determine remote URL for '{plugin_id}'. Cannot reinstall."

        gr.Info(f"Reinstalling '{plugin_id}'...")
        uninstall_msg = self.uninstall_plugin(plugin_id)
        if "[Error]" in uninstall_msg:
            return uninstall_msg
        
        install_msg = self.install_plugin_from_url(git_url)
        if "[Success]" in install_msg:
            return f"[Success] Plugin '{plugin_id}' reinstalled. Please restart WanGP."
        else:
            return f"[Error] Reinstallation failed during install step: {install_msg}"

    def install_plugin_from_url(self, git_url: str):
        if not git_url or not git_url.startswith("https://github.com/"):
            return "[Error] Invalid GitHub URL."

        try:
            repo_name = git_url.split('/')[-1].replace('.git', '')
            target_dir = os.path.join(self.plugins_dir, repo_name)

            if os.path.exists(target_dir):
                return f"[Warning] Plugin '{repo_name}' already exists. Please remove it manually to reinstall."

            gr.Info(f"Cloning '{repo_name}' into '{target_dir}'...")
            git.Repo.clone_from(git_url, target_dir)

            requirements_path = os.path.join(target_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                gr.Info(f"Installing dependencies for '{repo_name}'...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
                except subprocess.CalledProcessError as e:
                    return f"[Error] Failed to install dependencies for {repo_name}. Check console for details. Error: {e}"

            setup_path = os.path.join(target_dir, 'setup.py')
            if os.path.exists(setup_path):
                gr.Info(f"Running setup for '{repo_name}'...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', target_dir])
                except subprocess.CalledProcessError as e:
                    return f"[Error] Failed to run setup.py for {repo_name}. Check console for details. Error: {e}"
            
            init_path = os.path.join(target_dir, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    pass

            return f"[Success] Plugin '{repo_name}' installed. Please enable it in the list and restart WanGP."

        except git.exc.GitCommandError as e:
            return f"[Error] Git clone failed: {e.stderr}"
        except Exception as e:
            return f"[Error] An unexpected error occurred: {str(e)}"

    def discover_plugins(self) -> List[str]:
        discovered = []
        for item in os.listdir(self.plugins_dir):
            path = os.path.join(self.plugins_dir, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, '__init__.py')):
                discovered.append(item)
        return sorted(discovered)

    def load_plugins_from_directory(self, enabled_user_plugins: List[str]) -> None:
        plugins_to_load = SYSTEM_PLUGINS + [p for p in enabled_user_plugins if p not in SYSTEM_PLUGINS]

        for plugin_dir_name in self.discover_plugins():
            if plugin_dir_name not in plugins_to_load:
                continue
            try:
                module = importlib.import_module(f"{plugin_dir_name}.plugin")

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                        plugin = obj()
                        plugin.setup_ui()
                        self.plugins[plugin_dir_name] = plugin
                        for hook_name, callbacks in plugin._data_hooks.items():
                            if hook_name not in self.data_hooks:
                                self.data_hooks[hook_name] = []
                            self.data_hooks[hook_name].extend(callbacks)
                        print(f"Loaded plugin: {plugin.name} (from {plugin_dir_name})")
                        break
            except Exception as e:
                print(f"Error loading plugin from directory {plugin_dir_name}: {e}")
                traceback.print_exc()

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
        return {'tabs': tabs}
        
    def run_data_hooks(self, hook_name: str, *args, **kwargs):
        if hook_name not in self.data_hooks:
            return kwargs.get('configs')

        callbacks = self.data_hooks[hook_name]
        data = kwargs.get('configs')

        if 'configs' in kwargs:
            kwargs.pop('configs')

        for callback in callbacks:
            try:
                data = callback(data, **kwargs)
            except Exception as e:
                print(f"[PluginManager] Error running hook '{hook_name}' from {callback.__module__}: {e}")
                traceback.print_exc()
        return data

    def run_post_ui_setup(self, all_components: Dict[str, gr.components.Component]) -> None:
        if 'tab_state' not in all_components and 'main_tabs' in all_components:
            main_tabs = all_components.get('main_tabs')
            root_blocks = getattr(main_tabs, 'parent', None)
            if root_blocks:
                for component in getattr(root_blocks, 'children', []):
                    if isinstance(component, gr.State):
                        all_components['tab_state'] = component
                        break
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
                plugin.post_ui_setup(requested_components)
                all_insert_requests.extend(getattr(plugin, '_insert_after_requests', []))
                
            except Exception as e:
                print(f"[PluginManager] Error in post_ui_setup for {plugin_id}: {str(e)}")
                traceback.print_exc()

        if all_insert_requests:
            for request in all_insert_requests:
                try:
                    target = all_components.get(request.target_component_id)
                    parent = getattr(target, 'parent', None)
                    if not target or not parent or not hasattr(parent, 'children'):
                        print(f"[PluginManager] ERROR: Target '{request.target_component_id}' for insertion not found or invalid.")
                        continue
                        
                    target_index = parent.children.index(target)
                    with parent:
                        new_component = request.new_component_constructor()
                    
                    newly_added = parent.children.pop(-1)
                    parent.children.insert(target_index + 1, newly_added)
                    print(f"[PluginManager] Successfully inserted component after '{request.target_component_id}'")

                except Exception as e:
                    print(f"[PluginManager] Error processing insert_after for {request.target_component_id}: {str(e)}")

class WAN2GPApplication:
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.ui_components = {}

    def _create_plugin_tabs(self, main_module_globals: Dict[str, Any]):
        if not self.plugin_manager:
            return
        server_config = main_module_globals.get('server_config', {})
        enabled_user_plugins = server_config.get("enabled_plugins", [])
        
        self.plugin_manager.load_plugins_from_directory(enabled_user_plugins)
        self.plugin_manager.inject_globals(main_module_globals)
        
        loaded_plugins = self.plugin_manager.get_all_plugins()

        system_tabs = []
        user_tabs = []

        for plugin_id, plugin in loaded_plugins.items():
            for tab_id, tab in plugin.tabs.items():
                tab_info = {
                    'id': tab_id,
                    'label': tab.label,
                    'component_constructor': tab.component_constructor,
                    'position': tab.position
                }
                if plugin_id in SYSTEM_PLUGINS:
                    system_tabs.append(tab_info)
                else:
                    user_tabs.append((plugin_id, tab_info))

        system_tabs.sort(key=lambda t: (t.get('position', -1), t['label']))
        
        sorted_user_tabs = []
        for plugin_id in enabled_user_plugins:
            for pid, tab_info in user_tabs:
                if pid == plugin_id:
                    sorted_user_tabs.append(tab_info)
        
        all_tabs_to_render = system_tabs + sorted_user_tabs
        
        for tab_info in all_tabs_to_render:
            with gr.Tab(tab_info['label'], id=f"plugin_{tab_info['id']}"):
                tab_info['component_constructor']()

    def finalize_ui_setup(self, main_module_globals: Dict[str, Any], all_ui_components: Dict[str, Any]):
        self._create_plugin_tabs(main_module_globals)
        self.ui_components = {
            name: obj for name, obj in all_ui_components.items() 
            if isinstance(obj, (gr.Blocks, gr.components.Component, gr.Row, gr.Column, gr.Tabs, gr.Group, gr.Accordion))
        }
        if self.plugin_manager:
            self.plugin_manager.run_post_ui_setup(self.ui_components)