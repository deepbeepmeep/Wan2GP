import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass
import gradio as gr
import traceback
import subprocess
import git
import shutil
import stat
import json
video_gen_label = "Video Generator"
def auto_install_and_enable_default_plugins(manager: 'PluginManager', wgp_globals: dict):
    server_config = wgp_globals.get("server_config")
    server_config_filename = wgp_globals.get("server_config_filename")

    if not server_config or not server_config_filename:
        print("[Plugins] WARNING: Cannot auto-install/enable default plugins. Server config not found.")
        return

    default_plugins = {
        "wan2gp-gallery": "https://github.com/Tophness/wan2gp-gallery.git",
        "wan2gp-lora-multipliers-ui": "https://github.com/Tophness/wan2gp-lora-multipliers-ui.git"
    }
    
    config_modified = False
    enabled_plugins = server_config.get("enabled_plugins", [])

    for repo_name, url in default_plugins.items():
        target_dir = os.path.join(manager.plugins_dir, repo_name)
        if not os.path.isdir(target_dir):
            print(f"[Plugins] Auto-installing default plugin: {repo_name}...")
            result = manager.install_plugin_from_url(url)
            print(f"[Plugins] Install result for {repo_name}: {result}")
            
            if "[Success]" in result:
                if repo_name not in enabled_plugins:
                    enabled_plugins.append(repo_name)
                    config_modified = True
    
    if config_modified:
        print("[Plugins] Disabling newly installed default plugins...")
        server_config["enabled_plugins"] = []
        try:
            with open(server_config_filename, 'w', encoding='utf-8') as f:
                json.dump(server_config, f, indent=4)
        except Exception as e:
            print(f"[Plugins] ERROR: Failed to update config file '{server_config_filename}': {e}")


SYSTEM_PLUGINS = [
    "wan2gp-video-mask-creator",
    "wan2gp-motion-designer",
    "wan2gp-guides",
    "wan2gp-downloads",
    "wan2gp-configuration",
    "wan2gp-plugin-manager",
    "wan2gp-about",
]

USER_PLUGIN_INSERT_POSITION = 3

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
        self.tab_ids: List[str] = []
        self._set_wgp_global_func = None
        self._custom_js_snippets: List[str] = []
        
    def setup_ui(self) -> None:
        pass
        
    def add_tab(self, tab_id: str, label: str, component_constructor: callable, position: int = -1):
        self.tabs[tab_id] = PluginTab(id=tab_id, label=label, component_constructor=component_constructor, position=position)

    def post_ui_setup(self, components: Dict[str, gr.components.Component]) -> Dict[gr.components.Component, Union[gr.update, Any]]:
        return {}

    def on_tab_select(self, state: Dict[str, Any]) -> None:
        pass

    def on_tab_deselect(self, state: Dict[str, Any]) -> None:
        pass

    def request_component(self, component_id: str) -> None:
        if component_id not in self._component_requests:
            self._component_requests.append(component_id)
            
    def request_global(self, global_name: str) -> None:
        if global_name not in self._global_requests:
            self._global_requests.append(global_name)

    def set_global(self, variable_name: str, new_value: Any):
        if self._set_wgp_global_func:
            return self._set_wgp_global_func(variable_name, new_value)

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

    def add_custom_js(self, js_code: str) -> None:
        if isinstance(js_code, str) and js_code.strip():
            self._custom_js_snippets.append(js_code)

    @property
    def custom_js_snippets(self) -> List[str]:
        return self._custom_js_snippets.copy()

    def insert_after(self, target_component_id: str, new_component_constructor: callable) -> None:
        if not hasattr(self, '_insert_after_requests'):
            self._insert_after_requests = []
        self._insert_after_requests.append(
            InsertAfterRequest(
                target_component_id=target_component_id,
                new_component_constructor=new_component_constructor
            )
        )

class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        self.plugins: Dict[str, WAN2GPPlugin] = {}
        self.plugins_dir = plugins_dir
        os.makedirs(self.plugins_dir, exist_ok=True)
        if self.plugins_dir not in sys.path:
            sys.path.insert(0, self.plugins_dir)
        self.data_hooks: Dict[str, List[callable]] = {}
        self.restricted_globals: Set[str] = set()
        self.custom_js_snippets: List[str] = []

    def get_plugins_info(self) -> List[Dict[str, str]]:
        plugins_info = []
        for dir_name in self.discover_plugins():
            plugin_path = os.path.join(self.plugins_dir, dir_name)
            is_system = dir_name in SYSTEM_PLUGINS
            info = {'id': dir_name, 'name': dir_name, 'version': 'N/A', 'description': 'No description provided.', 'path': plugin_path, 'system': is_system}
            try:
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

    def _remove_readonly(self, func, path, exc_info):
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        else:
            raise

    def uninstall_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for uninstallation."
        
        if plugin_id in SYSTEM_PLUGINS:
            return f"[Error] Cannot uninstall system plugin '{plugin_id}'."

        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' directory not found."

        try:
            shutil.rmtree(target_dir, onerror=self._remove_readonly)
            return f"[Success] Plugin '{plugin_id}' uninstalled. Please restart WanGP."
        except Exception as e:
            return f"[Error] Failed to remove plugin '{plugin_id}': {e}"

    def update_plugin(self, plugin_id: str, progress=None):
        if not plugin_id:
            return "[Error] No plugin selected for update."
            
        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(os.path.join(target_dir, '.git')):
            return f"[Error] '{plugin_id}' is not a git repository and cannot be updated automatically."

        try:
            if progress is not None: progress(0, desc=f"Updating '{plugin_id}'...")
            repo = git.Repo(target_dir)
            origin = repo.remotes.origin
            
            if progress is not None: progress(0.2, desc=f"Fetching updates for '{plugin_id}'...")
            origin.fetch()
            
            local_commit = repo.head.commit
            remote_commit = origin.refs[repo.active_branch.name].commit

            if local_commit == remote_commit:
                 return f"[Info] Plugin '{plugin_id}' is already up to date."

            if progress is not None: progress(0.6, desc=f"Pulling updates for '{plugin_id}'...")
            origin.pull()
            
            requirements_path = os.path.join(target_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                if progress is not None: progress(0.8, desc=f"Re-installing dependencies for '{plugin_id}'...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

            if progress is not None: progress(1.0, desc="Update complete.")
            return f"[Success] Plugin '{plugin_id}' updated. Please restart WanGP for changes to take effect."
        except git.exc.GitCommandError as e:
            traceback.print_exc()
            return f"[Error] Git update failed for '{plugin_id}': {e.stderr}"
        except Exception as e:
            traceback.print_exc()
            return f"[Error] An unexpected error occurred during update of '{plugin_id}': {str(e)}"

    def reinstall_plugin(self, plugin_id: str, progress=None):
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
                traceback.print_exc()
                return f"[Error] Could not get remote URL for '{plugin_id}': {e}"
        
        if not git_url:
            return f"[Error] Could not determine remote URL for '{plugin_id}'. Cannot reinstall."

        if progress is not None: progress(0, desc=f"Reinstalling '{plugin_id}'...")

        backup_dir = f"{target_dir}.bak"
        if os.path.exists(backup_dir):
            try:
                shutil.rmtree(backup_dir, onerror=self._remove_readonly)
            except Exception as e:
                return f"[Error] Could not remove old backup directory '{backup_dir}'. Please remove it manually and try again. Error: {e}"

        try:
            if progress is not None: progress(0.2, desc=f"Moving old version of '{plugin_id}' aside...")
            os.rename(target_dir, backup_dir)
        except OSError as e:
            traceback.print_exc()
            return f"[Error] Could not move the existing plugin directory for '{plugin_id}'. It may be in use by another process. Please close any file explorers or editors in that folder and try again. Error: {e}"
        
        install_msg = self.install_plugin_from_url(git_url, progress=progress)
        
        if "[Success]" in install_msg:
            try:
                shutil.rmtree(backup_dir, onerror=self._remove_readonly)
            except Exception:
                pass
            return f"[Success] Plugin '{plugin_id}' reinstalled. Please restart WanGP."
        else:
            try:
                os.rename(backup_dir, target_dir)
                return f"[Error] Reinstallation failed during install step: {install_msg}. The original plugin has been restored."
            except Exception as restore_e:
                return f"[CRITICAL ERROR] Reinstallation failed AND could not restore backup. Plugin '{plugin_id}' is now in a broken state. Please manually rename '{backup_dir}' back to '{target_dir}'. Original error: {install_msg}. Restore error: {restore_e}"

    def install_plugin_from_url(self, git_url: str, progress=None):
        if not git_url or not git_url.startswith("https://github.com/"):
            return "[Error] Invalid GitHub URL."

        try:
            repo_name = git_url.split('/')[-1].replace('.git', '')
            target_dir = os.path.join(self.plugins_dir, repo_name)

            if os.path.exists(target_dir):
                return f"[Warning] Plugin '{repo_name}' already exists. Please remove it manually to reinstall."

            if progress is not None: progress(0.1, desc=f"Cloning '{repo_name}'...")
            git.Repo.clone_from(git_url, target_dir)

            requirements_path = os.path.join(target_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                if progress is not None: progress(0.5, desc=f"Installing dependencies for '{repo_name}'...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
                except subprocess.CalledProcessError as e:
                    traceback.print_exc()
                    return f"[Error] Failed to install dependencies for {repo_name}. Check console for details. Error: {e}"

            setup_path = os.path.join(target_dir, 'setup.py')
            if os.path.exists(setup_path):
                if progress is not None: progress(0.8, desc=f"Running setup for '{repo_name}'...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', target_dir])
                except subprocess.CalledProcessError as e:
                    traceback.print_exc()
                    return f"[Error] Failed to run setup.py for {repo_name}. Check console for details. Error: {e}"
            
            init_path = os.path.join(target_dir, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    pass
            
            if progress is not None: progress(1.0, desc="Installation complete.")
            return f"[Success] Plugin '{repo_name}' installed. Please enable it in the list and restart WanGP."

        except git.exc.GitCommandError as e:
            traceback.print_exc()
            return f"[Error] Git clone failed: {e.stderr}"
        except Exception as e:
            traceback.print_exc()
            return f"[Error] An unexpected error occurred: {str(e)}"

    def discover_plugins(self) -> List[str]:
        discovered = []
        for item in os.listdir(self.plugins_dir):
            path = os.path.join(self.plugins_dir, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, '__init__.py')):
                discovered.append(item)
        return sorted(discovered)

    def load_plugins_from_directory(self, enabled_user_plugins: List[str]) -> None:
        self.custom_js_snippets = []
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
                        if plugin.custom_js_snippets:
                            self.custom_js_snippets.extend(plugin.custom_js_snippets)
                        for hook_name, callbacks in plugin._data_hooks.items():
                            if hook_name not in self.data_hooks:
                                self.data_hooks[hook_name] = []
                            self.data_hooks[hook_name].extend(callbacks)
                        if plugin_dir_name not in SYSTEM_PLUGINS:
                            print(f"Loaded plugin: {plugin.name} (from {plugin_dir_name})")
                        break
            except Exception as e:
                print(f"Error loading plugin from directory {plugin_dir_name}: {e}")
                traceback.print_exc()

    def get_all_plugins(self) -> Dict[str, WAN2GPPlugin]:
        return self.plugins.copy()

    def get_custom_js(self) -> str:
        if not self.custom_js_snippets:
            return ""
        return "\n".join(self.custom_js_snippets)

    def inject_globals(self, global_references: Dict[str, Any]) -> None:
        for plugin_id, plugin in self.plugins.items():
            try:
                if 'set_wgp_global' in global_references:
                    plugin._set_wgp_global_func = global_references['set_wgp_global']
                for global_name in plugin.global_requests:
                    if global_name in self.restricted_globals:
                        setattr(plugin, global_name, None)
                    elif global_name in global_references:
                        setattr(plugin, global_name, global_references[global_name])
            except Exception as e:
                print(f"  [!] ERROR injecting globals for {plugin_id}: {str(e)}")

    def update_global_reference(self, global_name: str, new_value: Any) -> None:
        safe_value = None if global_name in self.restricted_globals else new_value
        for plugin_id, plugin in self.plugins.items():
            try:
                if hasattr(plugin, '_global_requests') and global_name in plugin._global_requests:
                    setattr(plugin, global_name, safe_value)
            except Exception as e:
                print(f"  [!] ERROR updating global '{global_name}' for plugin {plugin_id}: {str(e)}")

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
        
    def run_component_insertion_and_setup(self, all_components: Dict[str, Any]):
        all_insert_requests: List[InsertAfterRequest] = []

        for plugin_id, plugin in self.plugins.items():
            try:
                for comp_id in plugin.component_requests:
                    if comp_id in all_components and (not hasattr(plugin, comp_id) or getattr(plugin, comp_id) is None):
                        setattr(plugin, comp_id, all_components[comp_id])

                requested_components = {
                    comp_id: all_components[comp_id]
                    for comp_id in plugin.component_requests
                    if comp_id in all_components
                }
                
                plugin.post_ui_setup(requested_components)
                
                insert_requests = getattr(plugin, '_insert_after_requests', [])
                if insert_requests:
                    all_insert_requests.extend(insert_requests)
                    plugin._insert_after_requests.clear()
                
            except Exception as e:
                print(f"[PluginManager] ERROR in post_ui_setup for {plugin_id}: {str(e)}")
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

                except Exception as e:
                    print(f"[PluginManager] ERROR processing insert_after for {request.target_component_id}: {str(e)}")
                    traceback.print_exc()

class WAN2GPApplication:
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.tab_to_plugin_map: Dict[str, WAN2GPPlugin] = {}
        self.all_rendered_tabs: List[gr.Tab] = []
        self.enabled_plugins: List[str] = []

    def initialize_plugins(self, wgp_globals: dict):
        if not hasattr(self, 'plugin_manager'):
            return
        
        auto_install_and_enable_default_plugins(self.plugin_manager, wgp_globals)
        
        server_config = wgp_globals.get("server_config")
        if not server_config:
            print("[PluginManager] ERROR: server_config not found in globals.")
            return

        self.enabled_plugins = server_config.get("enabled_plugins", [])
        self.plugin_manager.load_plugins_from_directory(self.enabled_plugins)
        self.plugin_manager.inject_globals(wgp_globals)

    def setup_ui_tabs(self, main_tabs_component: gr.Tabs, state_component: gr.State, set_save_form_event):
        self._create_plugin_tabs(main_tabs_component, state_component)
        self._setup_tab_events(main_tabs_component, state_component, set_save_form_event)
    
    def _create_plugin_tabs(self, main_tabs, state):
        if not hasattr(self, 'plugin_manager'):
            return
        
        loaded_plugins = self.plugin_manager.get_all_plugins()
        system_tabs, user_tabs = [], []
        system_order = {pid: idx for idx, pid in enumerate(SYSTEM_PLUGINS)}

        for plugin_id, plugin in loaded_plugins.items():
            for tab_id, tab in plugin.tabs.items():
                self.tab_to_plugin_map[tab.label] = plugin
                tab_info = {
                    'id': tab_id,
                    'label': tab.label,
                    'component_constructor': tab.component_constructor,
                    'position': system_order.get(plugin_id, tab.position),
                    'plugin_id': plugin_id,
                }
                if plugin_id in SYSTEM_PLUGINS:
                    system_tabs.append(tab_info)
                else:
                    user_tabs.append((plugin_id, tab_info))

        # Respect the declared system order, then splice user tabs after the configured index.
        system_tabs_sorted = sorted(
            system_tabs,
            key=lambda t: (system_order.get(t['plugin_id'], 1_000_000), t['label']),
        )
        pre_user_tabs = system_tabs_sorted[:USER_PLUGIN_INSERT_POSITION]
        post_user_tabs = system_tabs_sorted[USER_PLUGIN_INSERT_POSITION:]

        sorted_user_tabs = [tab_info for plugin_id in self.enabled_plugins for pid, tab_info in user_tabs if pid == plugin_id]

        all_tabs_to_render = pre_user_tabs + sorted_user_tabs + post_user_tabs

        def goto_video_tab(state):
            self._handle_tab_selection(state, None)
            return  gr.Tabs(selected="video_gen")
        

        for tab_info in all_tabs_to_render:
            with gr.Tab(tab_info['label'], id=f"plugin_{tab_info['id']}") as new_tab:
                self.all_rendered_tabs.append(new_tab)
                plugin = self.tab_to_plugin_map[new_tab.label]
                plugin.goto_video_tab = goto_video_tab 
                tab_info['component_constructor']()


    def _setup_tab_events(self, main_tabs_component: gr.Tabs, state_component: gr.State, set_save_form_event):
        if main_tabs_component and state_component:
            main_tabs_component.select(
                fn=self._handle_tab_selection,
                inputs=[state_component],
                outputs=None,
                show_progress="hidden",
            )


            for tab in self.all_rendered_tabs:
                # def test_tab(state_component, evt: gr.SelectData):
                #     last_save_form = state_component.get("last_save_form", video_gen_label)
                #     if last_save_form != video_gen_label :
                #         state_component["ignore_save_form"] = True
                #     else:
                #         state_component["last_save_form"] = evt.value


                plugin = self.tab_to_plugin_map[tab.label]
                # event = tab.select(fn=test_tab, inputs=[state_component])
                # event = set_save_form_event(event.then)
                event = set_save_form_event(tab.select)
                event.then(
                        fn=self._handle_one_tab_selection,
                        inputs=[state_component, gr.State(tab.label)],
                        outputs=plugin.on_tab_outputs if hasattr(plugin, "on_tab_outputs") else None,
                        show_progress="hidden",
                        trigger_mode="multiple",
                    )

            
    def _handle_tab_selection(self, state: dict, evt: gr.SelectData):
        if not hasattr(self, 'previous_tab_id'):
            self.previous_tab_id = video_gen_label
        
        new_tab_id = video_gen_label if evt is None else evt.value
        
        if self.previous_tab_id == new_tab_id:
            return

        if self.previous_tab_id and self.previous_tab_id in self.tab_to_plugin_map:
            plugin_to_deselect = self.tab_to_plugin_map[self.previous_tab_id]
            try:
                plugin_to_deselect.on_tab_deselect(state)
            except Exception as e:
                print(f"[PluginManager] Error in on_tab_deselect for plugin {plugin_to_deselect.name}: {e}")
                traceback.print_exc()

        # if new_tab_id and new_tab_id in self.tab_to_plugin_map:
            # plugin_to_select = self.tab_to_plugin_map[new_tab_id]
            # if not hasattr(plugin_to_select, "on_tab_outputs"):
            #     try:
            #         plugin_to_select.on_tab_select(state)
            #     except Exception as e:
            #         print(f"[PluginManager] Error in on_tab_select for plugin {plugin_to_select.name}: {e}")
            #         traceback.print_exc()

        self.previous_tab_id = new_tab_id

    def _handle_one_tab_selection(self, state: dict, new_tab_id): #, evt: gr.SelectData
        plugin_to_select = self.tab_to_plugin_map.get(new_tab_id, None)
        try:
            ret = plugin_to_select.on_tab_select(state)
        except Exception as e:
            print(f"[PluginManager] Error in on_tab_select for plugin {plugin_to_select.name}: {e}")
            traceback.print_exc()
            ret = None
        return ret
    
    def run_component_insertion(self, components_dict: Dict[str, Any]):
        if hasattr(self, 'plugin_manager'):
            self.plugin_manager.run_component_insertion_and_setup(components_dict)
