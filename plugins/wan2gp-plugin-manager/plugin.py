import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import json
import traceback
from wgp import quit_application
import requests

COMMUNITY_PLUGINS_URL = "https://github.com/deepbeepmeep/Wan2GP/raw/refs/heads/main/plugins.json"

class PluginManagerUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Plugin Manager UI"
        self.version = "1.8.0"
        self.description = "A built-in UI for managing, installing, and updating Wan2GP plugins"

    def setup_ui(self):
        self.request_global("app")
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.request_component("main")
        self.request_component("main_tabs")
        
        self.add_tab(
            tab_id="plugin_manager_tab",
            label="Plugins",
            component_constructor=self.create_plugin_manager_ui,
        )

    def _get_js_script_html(self):
        js_code = """
            () => {
                function updateGradioInput(elem_id, value) {
                    const gradio_app = document.querySelector('gradio-app') || document;
                    const input = gradio_app.querySelector(`#${elem_id} textarea`);
                    if (input) {
                        input.value = value;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                    return false;
                }

                function makeSortable() {
                    const userPluginList = document.querySelector('#user-plugin-list');
                    if (!userPluginList) return;

                    let draggedItem = null;

                    userPluginList.addEventListener('dragstart', e => {
                        draggedItem = e.target.closest('.plugin-item');
                        if (!draggedItem) return;
                        setTimeout(() => {
                            if (draggedItem) draggedItem.style.opacity = '0.5';
                        }, 0);
                    });

                    userPluginList.addEventListener('dragend', e => {
                        setTimeout(() => {
                             if (draggedItem) {
                                draggedItem.style.opacity = '1';
                                draggedItem = null;
                             }
                        }, 0);
                    });

                    userPluginList.addEventListener('dragover', e => {
                        e.preventDefault();
                        const afterElement = getDragAfterElement(userPluginList, e.clientY);
                        if (draggedItem) {
                            if (afterElement == null) {
                                userPluginList.appendChild(draggedItem);
                            } else {
                                userPluginList.insertBefore(draggedItem, afterElement);
                            }
                        }
                    });

                    function getDragAfterElement(container, y) {
                        const draggableElements = [...container.querySelectorAll('.plugin-item:not(.dragging)')];
                        return draggableElements.reduce((closest, child) => {
                            const box = child.getBoundingClientRect();
                            const offset = y - box.top - box.height / 2;
                            if (offset < 0 && offset > closest.offset) {
                                return { offset: offset, element: child };
                            } else {
                                return closest;
                            }
                        }, { offset: Number.NEGATIVE_INFINITY }).element;
                    }
                }
                
                setTimeout(makeSortable, 500);

                window.handlePluginAction = function(button, action) {
                    const pluginItem = button.closest('.plugin-item');
                    const pluginId = pluginItem.dataset.pluginId;
                    const payload = JSON.stringify({ action: action, plugin_id: pluginId });
                    updateGradioInput('plugin_action_input', payload);
                };
                
                window.handleStoreInstall = function(button, url) {
                    const payload = JSON.stringify({ action: 'install_from_store', url: url });
                    updateGradioInput('plugin_action_input', payload);
                };

                window.handleSave = function(restart) {
                    const user_container = document.querySelector('#user-plugin-list');
                    if (!user_container) return;
                    
                    const user_plugins = user_container.querySelectorAll('.plugin-item');
                    const enabledUserPlugins = Array.from(user_plugins)
                        .filter(item => item.querySelector('.plugin-enable-checkbox').checked)
                        .map(item => item.dataset.pluginId);
                    
                    const payload = JSON.stringify({ restart: restart, enabled_plugins: enabledUserPlugins });
                    updateGradioInput('save_action_input', payload);
                };
            }
        """
        return f"{js_code}"
    
    def _get_community_plugins_info(self):
        if hasattr(self, '_community_plugins_cache') and self._community_plugins_cache is not None:
            return self._community_plugins_cache
        try:
            response = requests.get(COMMUNITY_PLUGINS_URL, timeout=10)
            response.raise_for_status()
            plugins = response.json()
            self._community_plugins_cache = {
                p.get('url', '').split('/')[-1].replace('.git', ''): p
                for p in plugins if p.get('url')
            }
            return self._community_plugins_cache
        except Exception as e:
            print(f"[PluginManager] Could not fetch community plugins info: {e}")
            self._community_plugins_cache = {}
            return {}

    def _build_community_plugins_html(self):
        try:
            installed_plugin_ids = {p['id'] for p in self.app.plugin_manager.get_plugins_info()}
            remote_plugins = self._get_community_plugins_info()
            community_plugins = [
                p for plugin_id, p in remote_plugins.items()
                if plugin_id not in installed_plugin_ids
            ]

        except Exception as e:
            gr.Warning(f"Could not process community plugins list: {e}")
            return "<p style='text-align:center; color: var(--color-accent-soft);'>Failed to load community plugins.</p>"

        if not community_plugins:
            return "<p style='text-align:center; color: var(--text-color-secondary);'>All available community plugins are already installed.</p>"

        items_html = ""
        for plugin in community_plugins:
            name = plugin.get('name')
            author = plugin.get('author')
            version = plugin.get('version', 'N/A')
            description = plugin.get('description')
            url = plugin.get('url')

            if not all([name, author, description, url]):
                continue
            
            safe_url = url.replace("'", "\\'")

            items_html += f"""
            <div class="plugin-item">
                <div class="plugin-item-info">
                    <div class="plugin-header">
                        <span class="name">{name}</span>
                        <span class="version">version {version} by {author}</span>
                    </div>
                    <span class="description">{description}</span>
                </div>
                <div class="plugin-item-actions">
                    <button class="plugin-action-btn" onclick="handleStoreInstall(this, '{safe_url}')">Install</button>
                </div>
            </div>
            """
        
        return f"<div class='plugin-list'>{items_html}</div>"

    def _build_plugins_html(self):
        plugins_info = self.app.plugin_manager.get_plugins_info()
        enabled_user_plugins = self.server_config.get("enabled_plugins", [])
        all_user_plugins_info = [p for p in plugins_info if not p.get('system')]
        remote_plugins_info = self._get_community_plugins_info()
        
        css = """
        <style>
            .plugin-list { display: flex; flex-direction: column; gap: 12px; }
            .plugin-item { display: flex; flex-wrap: column; gap: 12px; justify-content: space-between; align-items: center; padding: 16px; border: 1px solid var(--border-color-primary); border-radius: 12px; background-color: var(--background-fill-secondary); transition: box-shadow 0.2s ease-in-out; }
            .plugin-item:hover { box-shadow: var(--shadow-drop-lg); }
            .plugin-item[draggable="true"] { cursor: grab; }
            .plugin-item[draggable="true"]:active { cursor: grabbing; }
            .plugin-info-container { display: flex; align-items: center; gap: 16px; flex-grow: 1; }
            .plugin-item-info { display: flex; flex-direction: column; gap: 4px; }
            .plugin-item-info .name { font-weight: 600; font-size: 1.1em; color: var(--text-color-primary); font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif; }
            .plugin-item-info .version { font-size: 0.9em; color: var(--text-color-secondary); }
            .plugin-item-info .description { font-size: 0.95em; color: var(--text-color-secondary); margin-top: 4px; }
            .plugin-item-actions { display: flex; gap: 8px; flex-shrink: 0; align-items: center; }
            .plugin-action-btn { display: inline-flex; align-items: center; justify-content: center; margin: 0 !important; border: 1px solid var(--button-secondary-border-color, #ccc) !important; background: var(--button-secondary-background-fill, #f0f0f0) !important; color: var(--button-secondary-text-color) !important; padding: 4px 12px !important; border-radius: 4px !important; cursor: pointer; font-weight: 500; }
            .plugin-action-btn:hover { background: var(--button-secondary-background-fill-hover, #e0e0e0) !important; transform: translateY(-1px); box-shadow: var(--shadow-drop); }
            .plugin-enable-checkbox { -webkit-appearance: none; appearance: none; position: relative; width: 22px; height: 22px; border-radius: 4px; border: 2px solid var(--border-color-primary); background-color: var(--background-fill-primary); cursor: pointer; display: inline-block; vertical-align: middle; box-sizing: border-box; transition: all 0.2s ease; }
            .plugin-enable-checkbox:hover { border-color: var(--color-accent); }
            .plugin-enable-checkbox:checked { background-color: var(--color-accent); border-color: var(--color-accent); }
            .plugin-enable-checkbox:checked::after { content: '✔'; position: absolute; color: white; font-size: 16px; font-weight: bold; top: 50%; left: 50%; transform: translate(-50%, -50%); }
            .save-buttons-container { justify-content: flex-start; margin-top: 20px !important; gap: 12px; }
            .stylish-save-btn { font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important; font-weight: 600 !important; font-size: 1.05em !important; padding: 10px 20px !important; }
            .update-available-notice {
                font-size: 0.9em;
                font-weight: 600;
                color: var(--color-accent);
                background-color: var(--color-accent-soft);
                padding: 2px 8px;
                border-radius: 4px;
                margin-left: 8px;
                white-space: nowrap;
            }
            .plugin-item.update-available {
                border-left: 4px solid var(--color-accent);
            }
        </style>
        """

        if not all_user_plugins_info:
            user_html = "<p style='text-align:center; color: var(--text-color-secondary);'>No user-installed plugins found.</p>"
        else:
            user_plugins_map = {p['id']: p for p in all_user_plugins_info}
            user_plugins = []
            for plugin_id in enabled_user_plugins:
                if plugin_id in user_plugins_map:
                    user_plugins.append(user_plugins_map.pop(plugin_id))
            user_plugins.extend(sorted(user_plugins_map.values(), key=lambda p: p['name']))

            user_items_html = ""
            for plugin in user_plugins:
                plugin_id = plugin['id']
                checked = "checked" if plugin_id in enabled_user_plugins else ""
                
                update_notice_html = ''
                item_update_class = ''
                if plugin_id in remote_plugins_info:
                    remote_version = remote_plugins_info[plugin_id].get('version', '0.0.0')
                    local_version = plugin.get('version', '0.0.0')
                    if remote_version > local_version:
                        update_notice_html = f'<span class="update-available-notice">v{remote_version} update available</span>'
                        item_update_class = 'update-available'
                
                user_items_html += f"""
                <div class="plugin-item {item_update_class}" data-plugin-id="{plugin_id}" draggable="true">
                    <div class="plugin-info-container">
                        <input type="checkbox" class="plugin-enable-checkbox" {checked}>
                        <div class="plugin-item-info">
                            <div class="plugin-header">
                                <span class="name">{plugin['name']}</span>
                                {update_notice_html}
                            </div>
                            <span class="version">version {plugin['version']} (id: {plugin['id']})</span>
                            <span class="description">{plugin.get('description', 'No description provided.')}</span>
                        </div>
                    </div>
                    <div class="plugin-item-actions">
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'update')">Update</button>
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'reinstall')">Reinstall</button>
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'uninstall')">Uninstall</button>
                    </div>
                </div>
                """
            user_html = f'<div id="user-plugin-list">{user_items_html}</div>'

        return f"{css}<div class='plugin-list'>{user_html}</div>"

    def create_plugin_manager_ui(self):
        with gr.Blocks() as plugin_blocks:
            with gr.Row(equal_height=False, variant='panel'):
                with gr.Column(scale=2, min_width=600):
                    gr.Markdown("### Installed Plugins (Drag to reorder tabs)")
                    self.plugins_html_display = gr.HTML()
                    with gr.Row(elem_classes="save-buttons-container"):
                        self.save_plugins_button = gr.Button("Save", variant="secondary", size="sm", scale=0, elem_classes="stylish-save-btn")
                        self.save_and_restart_button = gr.Button("Save and Restart", variant="primary", size="sm", scale=0, elem_classes="stylish-save-btn")
                with gr.Column(scale=2, min_width=300):
                    gr.Markdown("### Discover & Install")
                    
                    self.community_plugins_html = gr.HTML()
                    
                    with gr.Accordion("Install from URL", open=True):
                        with gr.Group():
                            self.plugin_url_textbox = gr.Textbox(label="GitHub URL", placeholder="https://github.com/user/wan2gp-plugin-repo")
                            self.install_plugin_button = gr.Button("Download and Install from URL")

            with gr.Column(visible=False):
                self.plugin_action_input = gr.Textbox(elem_id="plugin_action_input")
                self.save_action_input = gr.Textbox(elem_id="save_action_input")

        js = self._get_js_script_html()
        plugin_blocks.load(fn=None, js=js)

        self.main_tabs.select(
            self._on_tab_select_refresh,
            None,
            [self.plugins_html_display, self.community_plugins_html],
            show_progress="hidden"
        )
        
        self.save_plugins_button.click(fn=None, js="handleSave(false)")
        self.save_and_restart_button.click(fn=None, js="handleSave(true)")

        self.save_action_input.change(
            fn=self._handle_save_action,
            inputs=[self.save_action_input],
            outputs=[self.plugins_html_display]
        )
        
        self.plugin_action_input.change(
            fn=self._handle_plugin_action_from_json,
            inputs=[self.plugin_action_input],
            outputs=[self.plugins_html_display, self.community_plugins_html],
            show_progress="full"
        )

        self.install_plugin_button.click(
            fn=self._install_plugin_and_refresh,
            inputs=[self.plugin_url_textbox],
            outputs=[self.plugins_html_display, self.community_plugins_html, self.plugin_url_textbox],
            show_progress="full"
        )

        return plugin_blocks

    def _on_tab_select_refresh(self, evt: gr.SelectData):
        if evt.value != "Plugins":
            return gr.update(), gr.update()
        if hasattr(self, '_community_plugins_cache'):
            del self._community_plugins_cache
            
        installed_html = self._build_plugins_html()
        community_html = self._build_community_plugins_html()
        return gr.update(value=installed_html), gr.update(value=community_html)

    def _enable_plugin_after_install(self, url: str):
        try:
            plugin_id = url.split('/')[-1].replace('.git', '')
            enabled_plugins = self.server_config.get("enabled_plugins", [])
            if plugin_id not in enabled_plugins:
                enabled_plugins.append(plugin_id)
                self.server_config["enabled_plugins"] = enabled_plugins
                with open(self.server_config_filename, "w", encoding="utf-8") as writer:
                    writer.write(json.dumps(self.server_config, indent=4))
                return True
        except Exception as e:
            gr.Warning(f"Failed to auto-enable plugin {plugin_id}: {e}")
        return False

    def _save_plugin_settings(self, enabled_plugins: list):
        self.server_config["enabled_plugins"] = enabled_plugins
        with open(self.server_config_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.server_config, indent=4))
        gr.Info("Plugin settings saved. Please restart WanGP for changes to take effect.")
        return gr.update(value=self._build_plugins_html())

    def _save_and_restart(self, enabled_plugins: list):
        self.server_config["enabled_plugins"] = enabled_plugins
        with open(self.server_config_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.server_config, indent=4))
        gr.Info("Settings saved. Restarting application...")
        quit_application()

    def _handle_save_action(self, payload_str: str):
        if not payload_str:
            return gr.update(value=self._build_plugins_html())
        try:
            payload = json.loads(payload_str)
            enabled_plugins = payload.get("enabled_plugins", [])
            if payload.get("restart", False):
                self._save_and_restart(enabled_plugins)
                return gr.update(value=self._build_plugins_html())
            else:
                return self._save_plugin_settings(enabled_plugins)
        except (json.JSONDecodeError, TypeError):
            gr.Warning("Could not process save action due to invalid data.")
            return gr.update(value=self._build_plugins_html())

    def _install_plugin_and_refresh(self, url, progress=gr.Progress()):
        progress(0, desc="Starting installation...")
        result_message = self.app.plugin_manager.install_plugin_from_url(url, progress=progress)
        if "[Success]" in result_message:
            was_enabled = self._enable_plugin_after_install(url)
            if was_enabled:
                result_message = result_message.replace("Please enable it", "It has been auto-enabled")
            gr.Info(result_message)
        else:
            gr.Warning(result_message)
        return self._build_plugins_html(), self._build_community_plugins_html(), ""

    def _handle_plugin_action_from_json(self, payload_str: str, progress=gr.Progress()):
        if not payload_str:
            return gr.update(), gr.update()
        try:
            payload = json.loads(payload_str)
            action = payload.get("action")
            plugin_id = payload.get("plugin_id")
            
            if action == 'install_from_store':
                url = payload.get("url")
                if not url:
                    raise ValueError("URL is required for install_from_store action.")
                result_message = self.app.plugin_manager.install_plugin_from_url(url, progress=progress)
                if "[Success]" in result_message:
                    was_enabled = self._enable_plugin_after_install(url)
                    if was_enabled:
                         result_message = result_message.replace("Please enable it", "It has been auto-enabled")
            else:
                if not action or not plugin_id:
                     raise ValueError("Action and plugin_id are required.")
                result_message = ""
                if action == 'uninstall':
                    result_message = self.app.plugin_manager.uninstall_plugin(plugin_id)
                    current_enabled = self.server_config.get("enabled_plugins", [])
                    if plugin_id in current_enabled:
                        current_enabled.remove(plugin_id)
                        self.server_config["enabled_plugins"] = current_enabled
                        with open(self.server_config_filename, "w", encoding="utf-8") as writer:
                            writer.write(json.dumps(self.server_config, indent=4))
                elif action == 'update':
                    result_message = self.app.plugin_manager.update_plugin(plugin_id, progress=progress)
                elif action == 'reinstall':
                    result_message = self.app.plugin_manager.reinstall_plugin(plugin_id, progress=progress)
            
            if "[Success]" in result_message:
                gr.Info(result_message)
            elif "[Error]" in result_message or "[Warning]" in result_message:
                gr.Warning(result_message)
            else:
                gr.Info(result_message)
        except (json.JSONDecodeError, ValueError) as e:
            gr.Warning(f"Could not perform plugin action: {e}")
            traceback.print_exc()

        if hasattr(self, '_community_plugins_cache'):
            del self._community_plugins_cache

        return self._build_plugins_html(), self._build_community_plugins_html()