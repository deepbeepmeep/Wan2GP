import gradio as gr
import json
import traceback
from wgp import quit_application, WanGP_version
from shared.utils.plugins import WAN2GPPlugin, compare_release_metadata, is_wangp_compatible, plugin_id_from_url

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
                function pluginRoot() {
                    if (window.gradioApp) {
                        return window.gradioApp();
                    }
                    const app = document.querySelector('gradio-app');
                    return app ? (app.shadowRoot || app) : document;
                }

                function updateGradioInput(elem_id, value) {
                    const root = pluginRoot();
                    const input = root.querySelector(`#${elem_id} textarea, #${elem_id} input`);
                    if (input) {
                        input.value = value;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                    return false;
                }

                function makeSortable() {
                    const root = pluginRoot();
                    const userPluginList = root.querySelector('#user-plugin-list');
                    if (!userPluginList) return;
                    if (userPluginList.dataset.sortableBound === '1') return;
                    userPluginList.dataset.sortableBound = '1';

                    let draggedItem = null;

                    userPluginList.addEventListener('dragstart', e => {
                        draggedItem = e.target.closest('.plugin-item');
                        if (!draggedItem) return;
                        draggedItem.classList.add('dragging');
                        if (e.dataTransfer) {
                            e.dataTransfer.effectAllowed = 'move';
                            e.dataTransfer.setData('text/plain', draggedItem.dataset.pluginId || '');
                        }
                        setTimeout(() => {
                            if (draggedItem) draggedItem.style.opacity = '0.5';
                        }, 0);
                    });

                    userPluginList.addEventListener('dragend', e => {
                        setTimeout(() => {
                             if (draggedItem) {
                                draggedItem.style.opacity = '1';
                                draggedItem.classList.remove('dragging');
                                draggedItem = null;
                             }
                        }, 0);
                    });

                    userPluginList.addEventListener('dragover', e => {
                        e.preventDefault();
                        if (e.dataTransfer) {
                            e.dataTransfer.dropEffect = 'move';
                        }
                        const afterElement = getDragAfterElement(userPluginList, e.clientY);
                        if (draggedItem) {
                            if (afterElement === draggedItem) return;
                            if (afterElement == null) {
                                userPluginList.appendChild(draggedItem);
                            } else {
                                userPluginList.insertBefore(draggedItem, afterElement);
                            }
                        }
                    });

                    userPluginList.addEventListener('drop', e => {
                        e.preventDefault();
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
                
                function observeUserPluginList() {
                    const root = pluginRoot();
                    if (!root) {
                        setTimeout(observeUserPluginList, 400);
                        return;
                    }
                    if (root.dataset.pluginSortableObserver === '1') {
                        makeSortable();
                        return;
                    }
                    root.dataset.pluginSortableObserver = '1';
                    const observer = new MutationObserver(() => {
                        makeSortable();
                    });
                    observer.observe(root, { childList: true, subtree: true });
                    makeSortable();
                }
                
                setTimeout(observeUserPluginList, 200);
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
                    const root = pluginRoot();
                    const user_container = root.querySelector('#user-plugin-list');
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
            self._community_plugins_cache = self.app.plugin_manager.get_merged_catalog_entries(use_remote=True)
            return self._community_plugins_cache
        except Exception as e:
            print(f"[PluginManager] Could not fetch community plugins info: {e}")
            self._community_plugins_cache = {}
            return {}

    def _build_community_plugins_html(self):
        try:
            installed_plugin_ids = {p['id'] for p in self.app.plugin_manager.get_plugins_info()}
            remote_plugins = self._get_community_plugins_info()
            base_entries = self.app.plugin_manager.load_catalog_entries(use_remote=False)
            base_ids = {
                plugin_id_from_url(entry.get('url', ''))
                for entry in base_entries
                if entry.get('url')
            }
            community_plugins = [
                p for plugin_id, p in remote_plugins.items()
                if plugin_id not in installed_plugin_ids
            ]
            community_plugins.sort(
                key=lambda p: (
                    plugin_id_from_url(p.get('url', '')) not in base_ids,
                    (p.get('name') or '').lower()
                )
            )

        except Exception as e:
            gr.Warning(f"Could not process community plugins list: {e}")
            return "<p style='text-align:center; color: var(--color-accent-soft);'>Failed to load community plugins.</p>"

        if not community_plugins:
            return "<p style='text-align:center; color: var(--text-color-secondary);'>All available community plugins are already installed.</p>"

        items_html = ""
        for plugin in community_plugins:
            name = plugin.get('name')
            author = plugin.get('author') or "Unknown"
            version = plugin.get('version', 'N/A')
            description = plugin.get('description') or "No description provided."
            url = plugin.get('url')
            wan2gp_version = plugin.get('wan2gp_version') or plugin.get('wangp_version', '')

            if not url:
                continue
            if not name:
                name = plugin_id_from_url(url) or "Unknown Plugin"
            
            safe_url = url.replace("'", "\\'")
            incompatible = not is_wangp_compatible(wan2gp_version, WanGP_version)
            incompat_html = ""
            if incompatible and wan2gp_version:
                incompat_html = (
                    f"<span class='plugin-incompatible-badge' "
                    f"title='Requires WanGP v{wan2gp_version}+'>"
                    f"Requires WanGP v{wan2gp_version}+"
                    "</span>"
                )

            items_html += f"""
            <div class="plugin-item">
                <div class="plugin-item-info">
                    <div class="plugin-header">
                        <span class="name">{name}</span>
                        <span class="version">version {version} by {author}</span>
                        {incompat_html}
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
        remote_plugins_info = self.app.plugin_manager.get_merged_catalog_entries(use_remote=False)
        
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
            .stylish-save-btn { font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important; font-weight: 600 !important; font-size: 1.05em !important; padding: 10px 20px !important; white-space: nowrap; }
            .plugin-instruction { font-size: 0.95em; color: var(--text-color-secondary); margin-bottom: 6px; }
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
            .plugin-bundled-badge {
                font-size: 0.85em;
                font-weight: 600;
                color: var(--text-color-secondary);
                background-color: var(--background-fill-primary);
                padding: 2px 8px;
                border-radius: 4px;
                margin-left: 8px;
                border: 1px solid var(--border-color-primary);
                white-space: nowrap;
            }
            .plugin-incompatible-badge {
                font-size: 0.85em;
                font-weight: 600;
                color: var(--color-error, #b91c1c);
                background-color: rgba(185, 28, 28, 0.1);
                padding: 2px 8px;
                border-radius: 4px;
                margin-left: 8px;
                border: 1px solid rgba(185, 28, 28, 0.35);
                white-space: nowrap;
            }
            .plugin-item.update-available {
                border-left: 4px solid var(--color-accent);
            }
            .plugin-item.incompatible {
                border-left: 4px solid var(--color-error, #b91c1c);
            }
        </style>
        """

        instruction_html = "<div class='plugin-instruction'>Please Select the Plugins you want to enable</div>"
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
                uninstallable = plugin.get('uninstallable', True)
                author = plugin.get('author') or "Unknown"
                
                update_notice_html = ''
                item_classes = []
                if uninstallable and plugin_id in remote_plugins_info:
                    remote_entry = remote_plugins_info[plugin_id]
                    if compare_release_metadata(remote_entry, plugin) > 0:
                        remote_version = remote_entry.get('version') or remote_entry.get('date') or "unknown"
                        update_notice_html = (
                            f'<span class="update-available-notice">New version {remote_version} is available !</span>'
                        )
                        item_classes.append('update-available')

                bundled_badge_html = ''
                if not uninstallable:
                    bundled_badge_html = '<span class="plugin-bundled-badge" title="Bundled plugin, cannot be uninstalled">Bundled</span>'

                wan2gp_version = plugin.get('wan2gp_version') or plugin.get('wangp_version', '')
                incompatible = not is_wangp_compatible(wan2gp_version, WanGP_version)
                incompat_html = ''
                if incompatible and wan2gp_version:
                    incompat_html = (
                        f"<span class='plugin-incompatible-badge' "
                        f"title='Requires WanGP v{wan2gp_version}+'>"
                        f"Requires WanGP v{wan2gp_version}+"
                        "</span>"
                    )
                    item_classes.append('incompatible')

                actions_html = ""
                if uninstallable:
                    actions_html = """
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'update')">Update</button>
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'reinstall')">Reinstall</button>
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'uninstall')">Uninstall</button>
                    """
                actions_container_html = f'<div class="plugin-item-actions">{actions_html}</div>' if actions_html else ""
                
                user_items_html += f"""
                <div class="plugin-item {' '.join(item_classes)}" data-plugin-id="{plugin_id}" draggable="true">
                    <div class="plugin-info-container">
                        <input type="checkbox" class="plugin-enable-checkbox" {checked}>
                        <div class="plugin-item-info">
                            <div class="plugin-header">
                                <span class="name">{plugin['name']}</span>
                                {update_notice_html}
                                {bundled_badge_html}
                                {incompat_html}
                            </div>
                            <span class="version">version {plugin['version']} by {author} (id: {plugin['id']})</span>
                            <span class="description">{plugin.get('description', 'No description provided.')}</span>
                        </div>
                    </div>
                    {actions_container_html}
                </div>
                """
            user_html = f'<div id="user-plugin-list">{user_items_html}</div>'

        return f"{css}<div class='plugin-list'>{instruction_html}{user_html}</div>"

    def create_plugin_manager_ui(self):
        with gr.Blocks() as plugin_blocks:
            with gr.Row(equal_height=False, variant='panel'):
                with gr.Column(scale=2, min_width=600):
                    gr.Markdown("### Plugins Available Locally (Drag to reorder tabs)")
                    self.plugins_html_display = gr.HTML()
                    with gr.Row(elem_classes="save-buttons-container"):
                        self.save_plugins_button = gr.Button("Save", variant="secondary", size="sm", scale=0, elem_classes="stylish-save-btn")
                        self.save_and_restart_button = gr.Button("Save and Restart", variant="primary", size="sm", scale=0, elem_classes="stylish-save-btn")
                        self.refresh_catalog_button = gr.Button("Check for Updates", variant="secondary", size="sm", scale=0, elem_classes="stylish-save-btn")
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
        self.refresh_catalog_button.click(
            fn=self._refresh_catalog,
            inputs=[],
            outputs=[self.plugins_html_display, self.community_plugins_html],
            show_progress="full"
        )

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

    def _refresh_catalog(self, progress=gr.Progress()):
        self.app.plugin_manager.refresh_catalog(installed_only=True, use_remote=False)
        if hasattr(self, '_community_plugins_cache'):
            del self._community_plugins_cache
        updates_available = self._count_available_updates()
        if updates_available <= 0:
            gr.Info("No Plugin Update is available")
        elif updates_available == 1:
            gr.Info("One Plugin Update is available")
        else:
            gr.Info(f"{updates_available} Plugin Updates are available")
        return self._build_plugins_html(), self._build_community_plugins_html()

    def _count_available_updates(self) -> int:
        try:
            plugins_info = self.app.plugin_manager.get_plugins_info()
            remote_plugins_info = self.app.plugin_manager.get_merged_catalog_entries(use_remote=False)
            count = 0
            for plugin in plugins_info:
                if plugin.get('system'):
                    continue
                if not plugin.get('uninstallable', True):
                    continue
                plugin_id = plugin.get('id')
                if not plugin_id or plugin_id not in remote_plugins_info:
                    continue
                remote_entry = remote_plugins_info[plugin_id]
                if compare_release_metadata(remote_entry, plugin) > 0:
                    count += 1
            return count
        except Exception:
            return 0

    def _enable_plugin_after_install(self, url: str):
        try:
            plugin_id = plugin_id_from_url(url)
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
            plugin_id = plugin_id_from_url(url)
            if plugin_id:
                self.app.plugin_manager.record_plugin_metadata(plugin_id, url=url)
            if hasattr(self, '_community_plugins_cache'):
                del self._community_plugins_cache
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
