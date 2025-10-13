import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import json
import traceback
from wgp import quit_application

class PluginManagerUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Plugin Manager UI"
        self.version = "1.5.0"
        self.description = "A built-in UI for managing, installing, and updating Wan2GP plugins"

    def setup_ui(self):
        self.request_global("app")
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.request_component("main")
        
        self.add_tab(
            tab_id="plugin_manager_tab",
            label="Plugins",
            component_constructor=self.create_plugin_manager_ui,
            position=6
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

    def _build_plugins_html(self):
        plugins_info = self.app.plugin_manager.get_plugins_info()
        enabled_user_plugins = self.server_config.get("enabled_plugins", [])
        
        if not plugins_info:
            return "<p style='text-align:center; color: var(--text-color-secondary);'>No plugins found.</p>"

        system_plugins = [p for p in plugins_info if p.get('system')]
        user_plugins_map = {p['id']: p for p in plugins_info if not p.get('system')}

        user_plugins = []
        for plugin_id in enabled_user_plugins:
            if plugin_id in user_plugins_map:
                user_plugins.append(user_plugins_map.pop(plugin_id))
        user_plugins.extend(user_plugins_map.values())

        system_items_html = ""
        for plugin in system_plugins:
            system_items_html += f"""
            <div class="plugin-item" data-plugin-id="{plugin['id']}">
                <div class="plugin-info-container">
                    <input type="checkbox" class="plugin-enable-checkbox" checked disabled>
                    <div class="plugin-item-info">
                        <div class="plugin-header">
                            <span class="name">{plugin['name']}</span>
                            <span class="version">version {plugin['version']} (id: {plugin['id']})</span>
                        </div>
                        <span class="description">{plugin['description']}</span>
                    </div>
                </div>
                 <div class="plugin-item-actions">
                    <span class="system-plugin-tag">System Plugin</span>
                </div>
            </div>
            """

        user_items_html = ""
        for plugin in user_plugins:
            plugin_id = plugin['id']
            checked = "checked" if plugin_id in enabled_user_plugins else ""
            user_items_html += f"""
            <div class="plugin-item" data-plugin-id="{plugin_id}" draggable="true">
                <div class="plugin-info-container">
                    <input type="checkbox" class="plugin-enable-checkbox" {checked}>
                    <div class="plugin-item-info">
                        <div class="plugin-header">
                            <span class="name">{plugin['name']}</span>
                            <span class="version">version {plugin['version']} (id: {plugin['id']})</span>
                        </div>
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

        css = """
        <style>
            .plugin-list { display: flex; flex-direction: column; gap: 12px; }
            .plugin-item { display: flex; flex-wrap: column; gap: 12px; justify-content: space-between; align-items: center; padding: 16px; border: 1px solid var(--border-color-primary); border-radius: 12px; background-color: var(--background-fill-secondary); transition: box-shadow 0.2s ease-in-out; }
            .plugin-item:hover { box-shadow: var(--shadow-drop-lg); }
            .plugin-item[draggable="true"] { cursor: grab; }
            .plugin-item[draggable="true"]:active { cursor: grabbing; }
            .plugin-section-header { margin-top: 20px; margin-bottom: 10px; font-size: 1.2em; font-weight: bold; color: var(--text-color-primary); border-bottom: 1px solid var(--border-color-primary); padding-bottom: 5px; }
            .system-plugin-tag { font-size: 0.9em; font-style: italic; color: var(--text-color-secondary); }
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
            .plugin-enable-checkbox:disabled { cursor: not-allowed; border-color: var(--border-color-accent-subdued); background-color: var(--background-fill-primary-hover); }
            .plugin-enable-checkbox:disabled:checked { background-color: var(--color-accent-soft); border-color: var(--color-accent-soft); }
            .save-buttons-container { justify-content: flex-start; margin-top: 20px !important; gap: 12px; }
            .stylish-save-btn { font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important; font-weight: 600 !important; font-size: 1.05em !important; padding: 10px 20px !important; }
        </style>
        """

        full_html = f"""
        {css}
        <div class="plugin-list">
            <h3 class="plugin-section-header">System Plugins</h3>
            <div id="system-plugin-list">{system_items_html}</div>
            <h3 class="plugin-section-header">User Plugins (Drag to reorder tabs)</h3>
            <div id="user-plugin-list">{user_items_html}</div>
        </div>
        """
        return full_html

    def create_plugin_manager_ui(self):
        with gr.Blocks() as plugin_blocks:
            with gr.Row(equal_height=False, variant='panel'):
                with gr.Column(scale=2, min_width=600):
                    self.plugins_html_display = gr.HTML(self._build_plugins_html)
                    with gr.Row(elem_classes="save-buttons-container"):
                        self.save_plugins_button = gr.Button("Save", variant="secondary", size="sm", scale=0, elem_classes="stylish-save-btn")
                        self.save_and_restart_button = gr.Button("Save and Restart", variant="primary", size="sm", scale=0, elem_classes="stylish-save-btn")
                with gr.Column(scale=2, min_width=300):
                    gr.Markdown("### Install New Plugin")
                    gr.Markdown("Enter the URL of a GitHub repository containing a Wan2GP plugin.")
                    with gr.Group():
                        self.plugin_url_textbox = gr.Textbox(label="GitHub URL", placeholder="https://github.com/user/wan2gp-plugin-repo")
                        self.install_plugin_button = gr.Button("Download and Install")
            with gr.Column(visible=False):
                self.plugin_action_input = gr.Textbox(elem_id="plugin_action_input")
                self.save_action_input = gr.Textbox(elem_id="save_action_input")

        return plugin_blocks

    def _refresh_ui(self):
        return gr.update(value=self._build_plugins_html())

    def _save_plugin_settings(self, enabled_plugins: list):
        self.server_config["enabled_plugins"] = enabled_plugins
        with open(self.server_config_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.server_config, indent=4))
        gr.Info("Plugin settings saved. Please restart WanGP for changes to take effect.")
        return self._refresh_ui()

    def _save_and_restart(self, enabled_plugins: list):
        self.server_config["enabled_plugins"] = enabled_plugins
        with open(self.server_config_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.server_config, indent=4))
        gr.Info("Settings saved. Restarting application...")
        quit_application()

    def _handle_save_action(self, payload_str: str):
        if not payload_str:
            return self._refresh_ui()
        try:
            payload = json.loads(payload_str)
            enabled_plugins = payload.get("enabled_plugins", [])
            if payload.get("restart", False):
                self._save_and_restart(enabled_plugins)
                return self._refresh_ui()
            else:
                return self._save_plugin_settings(enabled_plugins)
        except (json.JSONDecodeError, TypeError):
            gr.Warning("Could not process save action due to invalid data.")
            return self._refresh_ui()

    def _install_plugin_and_refresh(self, url, progress=gr.Progress()):
        progress(0, desc="Starting installation...")
        result_message = self.app.plugin_manager.install_plugin_from_url(url)
        if "[Success]" in result_message:
            gr.Info(result_message)
        else:
            gr.Warning(result_message)
        return self._refresh_ui(), ""

    def _handle_plugin_action_from_json(self, payload_str: str):
        if not payload_str:
            return self._refresh_ui()
        try:
            payload = json.loads(payload_str)
            action = payload.get("action")
            plugin_id = payload.get("plugin_id")
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
                result_message = self.app.plugin_manager.update_plugin(plugin_id)
            elif action == 'reinstall':
                result_message = self.app.plugin_manager.reinstall_plugin(plugin_id)
            
            if "[Success]" in result_message:
                gr.Info(result_message)
            elif "[Error]" in result_message or "[Warning]" in result_message:
                gr.Warning(result_message)
            else:
                gr.Info(result_message)
        except (json.JSONDecodeError, ValueError) as e:
            gr.Warning(f"Could not perform plugin action: {e}")
            traceback.print_exc()
        return self._refresh_ui()

    def post_ui_setup(self, components: dict):
        js=self._get_js_script_html()
        self.main.load(
            fn=self._refresh_ui,
            inputs=[],
            outputs=[self.plugins_html_display],
            js=js
        )
        self.save_plugins_button.click(
            fn=None, js="handleSave(false)"
        )
        self.save_and_restart_button.click(
            fn=None, js="handleSave(true)"
        )

        self.save_action_input.change(
            fn=self._handle_save_action,
            inputs=[self.save_action_input],
            outputs=[self.plugins_html_display]
        )
        
        self.plugin_action_input.change(
            fn=self._handle_plugin_action_from_json,
            inputs=[self.plugin_action_input],
            outputs=[self.plugins_html_display]
        )

        self.install_plugin_button.click(
            fn=self._install_plugin_and_refresh,
            inputs=[self.plugin_url_textbox],
            outputs=[self.plugins_html_display, self.plugin_url_textbox]
        )
        
        return {}