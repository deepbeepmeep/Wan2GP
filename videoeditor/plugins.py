import os
import sys
import importlib.util
import subprocess
import shutil
import git
import json
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                             QPushButton, QLabel, QLineEdit, QMessageBox, QProgressBar,
                             QDialogButtonBox, QWidget, QCheckBox)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread

class VideoEditorPlugin:
    """
    Base class for all plugins.
    Plugins should inherit from this class and be located in 'plugins/plugin_name/main.py'.
    The main class in main.py must be named 'Plugin'.
    """
    def __init__(self, app_instance):
        self.app = app_instance
        self.name = "Unnamed Plugin"
        self.description = "No description provided."

    def initialize(self):
        """Called once when the plugin is loaded by the PluginManager."""
        pass

    def enable(self):
        """Called when the plugin is enabled by the user (e.g., checking the box in the menu)."""
        pass

    def disable(self):
        """Called when the plugin is disabled by the user."""
        pass

class PluginManager:
    """Manages the discovery, loading, and lifecycle of plugins."""
    def __init__(self, main_app):
        self.app = main_app
        self.plugins_dir = "plugins"
        self.plugins = {}  # { 'plugin_name': {'instance': plugin_instance, 'enabled': False, 'module_path': path} }
        if not os.path.exists(self.plugins_dir):
            os.makedirs(self.plugins_dir)

    def discover_and_load_plugins(self):
        """Scans the plugins directory, loads valid plugins, and calls their initialize method."""
        for plugin_name in os.listdir(self.plugins_dir):
            plugin_path = os.path.join(self.plugins_dir, plugin_name)
            main_py_path = os.path.join(plugin_path, 'main.py')
            if os.path.isdir(plugin_path) and os.path.exists(main_py_path):
                try:
                    spec = importlib.util.spec_from_file_location(f"plugins.{plugin_name}.main", main_py_path)
                    module = importlib.util.module_from_spec(spec)
                    
                    sys.path.insert(0, plugin_path)
                    spec.loader.exec_module(module)
                    sys.path.pop(0)

                    if hasattr(module, 'Plugin'):
                        plugin_class = getattr(module, 'Plugin')
                        instance = plugin_class(self.app)
                        instance.initialize()
                        self.plugins[instance.name] = {
                            'instance': instance,
                            'enabled': False,
                            'module_path': plugin_path
                        }
                        print(f"Discovered and loaded plugin: {instance.name}")
                    else:
                        print(f"Warning: {main_py_path} does not have a 'Plugin' class.")
                except Exception as e:
                    print(f"Error loading plugin {plugin_name}: {e}")

    def load_enabled_plugins_from_settings(self, enabled_plugins_list):
        """Enables plugins based on the loaded settings."""
        for name in enabled_plugins_list:
            if name in self.plugins:
                self.enable_plugin(name)

    def get_enabled_plugin_names(self):
        """Returns a list of names of all enabled plugins."""
        return [name for name, data in self.plugins.items() if data['enabled']]

    def enable_plugin(self, name):
        """Enables a specific plugin by name and updates the UI."""
        if name in self.plugins and not self.plugins[name]['enabled']:
            self.plugins[name]['instance'].enable()
            self.plugins[name]['enabled'] = True
            # Notify the app to update the menu's checkmark
            self.app.toggle_plugin_action(name, True)
            self.app.update_plugin_ui_visibility(name, True)

    def disable_plugin(self, name):
        """Disables a specific plugin by name and updates the UI."""
        if name in self.plugins and self.plugins[name]['enabled']:
            self.plugins[name]['instance'].disable()
            self.plugins[name]['enabled'] = False
            # Notify the app to update the menu's checkmark
            self.app.toggle_plugin_action(name, False)
            self.app.update_plugin_ui_visibility(name, False)

    def uninstall_plugin(self, name):
        """Uninstalls (deletes) a plugin by name."""
        if name in self.plugins:
            path = self.plugins[name]['module_path']
            if self.plugins[name]['enabled']:
                self.disable_plugin(name)
            
            try:
                shutil.rmtree(path)
                del self.plugins[name]
                return True
            except OSError as e:
                print(f"Error removing plugin directory {path}: {e}")
                return False
        return False


class InstallWorker(QObject):
    finished = pyqtSignal(str, bool)
    
    def __init__(self, url, target_dir):
        super().__init__()
        self.url = url
        self.target_dir = target_dir
        
    def run(self):
        try:
            repo_name = self.url.split('/')[-1].replace('.git', '')
            clone_path = os.path.join(self.target_dir, repo_name)
            if os.path.exists(clone_path):
                self.finished.emit(f"Directory '{repo_name}' already exists.", False)
                return

            git.Repo.clone_from(self.url, clone_path)
            
            req_path = os.path.join(clone_path, 'requirements.txt')
            if os.path.exists(req_path):
                print("Installing plugin requirements...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])

            self.finished.emit(f"Plugin '{repo_name}' installed successfully. Please restart the application.", True)
        except Exception as e:
            self.finished.emit(f"Installation failed: {e}", False)


class ManagePluginsDialog(QDialog):
    def __init__(self, plugin_manager, parent=None):
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.setWindowTitle("Manage Plugins")
        self.setMinimumSize(500, 400)

        self.plugin_checkboxes = {}

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Installed Plugins:"))
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        install_layout = QVBoxLayout()
        install_layout.addWidget(QLabel("Install new plugin from GitHub URL:"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("e.g., https://github.com/user/repo.git")
        self.install_btn = QPushButton("Install")
        install_layout.addWidget(self.url_input)
        install_layout.addWidget(self.install_btn)
        layout.addLayout(install_layout)
        
        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)
        
        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.button_box)

        # Connections
        self.install_btn.clicked.connect(self.install_plugin)
        self.button_box.accepted.connect(self.save_changes)
        self.button_box.rejected.connect(self.reject)
        
        self.populate_list()

    def populate_list(self):
        self.list_widget.clear()
        self.plugin_checkboxes.clear()

        for name, data in sorted(self.plugin_manager.plugins.items()):
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 5, 5, 5)

            checkbox = QCheckBox(name)
            checkbox.setChecked(data['enabled'])
            checkbox.setToolTip(data['instance'].description)
            self.plugin_checkboxes[name] = checkbox
            item_layout.addWidget(checkbox, 1)

            uninstall_btn = QPushButton("Uninstall")
            uninstall_btn.setFixedWidth(80)
            uninstall_btn.clicked.connect(lambda _, n=name: self.handle_uninstall(n))
            item_layout.addWidget(uninstall_btn)
            
            item_widget.setLayout(item_layout)

            list_item = QListWidgetItem(self.list_widget)
            list_item.setSizeHint(item_widget.sizeHint())
            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, item_widget)
        
    def save_changes(self):
        for name, checkbox in self.plugin_checkboxes.items():
            if name not in self.plugin_manager.plugins:
                continue
            
            is_checked = checkbox.isChecked()
            is_currently_enabled = self.plugin_manager.plugins[name]['enabled']
            
            if is_checked and not is_currently_enabled:
                self.plugin_manager.enable_plugin(name)
            elif not is_checked and is_currently_enabled:
                self.plugin_manager.disable_plugin(name)
        
        self.plugin_manager.app._save_settings()
        self.accept()

    def handle_uninstall(self, name):
        reply = QMessageBox.question(self, "Confirm Uninstall", 
                                     f"Are you sure you want to permanently delete the plugin '{name}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.plugin_manager.uninstall_plugin(name):
                self.status_label.setText(f"Uninstalled '{name}'. Restart the application to fully remove it from the menu.")
                self.populate_list()
            else:
                self.status_label.setText(f"Failed to uninstall '{name}'.")

    def install_plugin(self):
        url = self.url_input.text().strip()
        if not url.endswith(".git"):
            QMessageBox.warning(self, "Invalid URL", "Please provide a valid git repository URL (ending in .git).")
            return
        
        self.install_btn.setEnabled(False)
        self.status_label.setText(f"Cloning from {url}...")
        
        self.install_thread = QThread()
        self.install_worker = InstallWorker(url, self.plugin_manager.plugins_dir)
        self.install_worker.moveToThread(self.install_thread)
        
        self.install_thread.started.connect(self.install_worker.run)
        self.install_worker.finished.connect(self.on_install_finished)
        self.install_worker.finished.connect(self.install_thread.quit)
        self.install_worker.finished.connect(self.install_worker.deleteLater)
        self.install_thread.finished.connect(self.install_thread.deleteLater)
        
        self.install_thread.start()
        
    def on_install_finished(self, message, success):
        self.status_label.setText(message)
        self.install_btn.setEnabled(True)
        if success:
            self.url_input.clear()