import sys
import os
import threading
import time
import json
import re
from unittest.mock import MagicMock

# --- Start of Gradio Hijacking ---
# This block creates a mock Gradio module. When wgp.py is imported,
# all calls to `gr.*` will be intercepted by these mock objects,
# preventing any UI from being built and allowing us to use the
# backend logic directly.

class MockGradioComponent(MagicMock):
    """A smarter mock that captures constructor arguments."""
    def __init__(self, *args, **kwargs):
        super().__init__(name=f"gr.{kwargs.get('elem_id', 'component')}")
        # Store the kwargs so we can inspect them later
        self.kwargs = kwargs
        self.value = kwargs.get('value')
        self.choices = kwargs.get('choices')
        
        # Mock chaining methods like .click(), .then(), etc.
        for method in ['then', 'change', 'click', 'input', 'select', 'upload', 'mount', 'launch', 'on', 'release']:
            setattr(self, method, lambda *a, **kw: self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockGradioError(Exception):
    pass

class MockGradioModule: # No longer inherits from MagicMock
    def __getattr__(self, name):
        if name == 'Error':
            return lambda *args, **kwargs: MockGradioError(*args)

        # Nullify functions that show pop-ups
        if name in ['Info', 'Warning']:
            return lambda *args, **kwargs: print(f"Intercepted gr.{name}:", *args)

        return lambda *args, **kwargs: MockGradioComponent(*args, **kwargs)

sys.modules['gradio'] = MockGradioModule()
sys.modules['gradio.gallery'] = MockGradioModule() # Also mock any submodules used
sys.modules['shared.gradio.gallery'] = MockGradioModule()
# --- End of Gradio Hijacking ---

# Global placeholder for the wgp module. Will be None if import fails.
wgp = None

# Load configuration and attempt to import wgp
MAIN_CONFIG_FILE = 'main_config.json'
main_config = {}

def load_main_config():
    global main_config
    try:
        with open(MAIN_CONFIG_FILE, 'r') as f:
            main_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_folder = os.path.dirname(os.path.abspath(sys.argv[0]))
        main_config = {'wgp_path': current_folder}

def save_main_config():
    global main_config
    try:
        with open(MAIN_CONFIG_FILE, 'w') as f:
            json.dump(main_config, f, indent=4)
    except Exception as e:
        print(f"Error saving main_config.json: {e}")

def setup_and_import_wgp():
    """Adds configured path to sys.path and tries to import wgp."""
    global wgp
    wgp_path = main_config.get('wgp_path')
    if wgp_path and os.path.isdir(wgp_path) and os.path.isfile(os.path.join(wgp_path, 'wgp.py')):
        if wgp_path not in sys.path:
            sys.path.insert(0, wgp_path)
        try:
            import wgp as wgp_module
            wgp = wgp_module
            return True
        except ImportError as e:
            print(f"Error: Failed to import wgp.py from the configured path '{wgp_path}'.\nDetails: {e}")
            wgp = None
            return False
    else:
        print("Info: WAN2GP folder path not set or invalid. Please configure it in File > Settings.")
        wgp = None
        return False

# Load config and attempt import at script start
load_main_config()
wgp_loaded = setup_and_import_wgp()

# Now import PyQt6 components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QFileDialog, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QScrollArea, QListWidget, QListWidgetItem,
    QMessageBox, QRadioButton, QDialog
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap, QDropEvent, QAction
from PIL.ImageQt import ImageQt


class SettingsDialog(QDialog):
    """Dialog to configure application settings like the wgp.py path."""
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        path_layout = QHBoxLayout()
        self.wgp_path_edit = QLineEdit(self.config.get('wgp_path', ''))
        self.wgp_path_edit.setPlaceholderText("Path to the folder containing wgp.py")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_for_wgp_folder)
        path_layout.addWidget(self.wgp_path_edit)
        path_layout.addWidget(browse_btn)

        form_layout.addRow("WAN2GP Folder Path:", path_layout)
        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_restart_btn = QPushButton("Save and Restart")
        cancel_btn = QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(save_restart_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        save_btn.clicked.connect(self.save_and_close)
        save_restart_btn.clicked.connect(self.save_and_restart)
        cancel_btn.clicked.connect(self.reject)

    def browse_for_wgp_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "Select WAN2GP Folder")
        if directory:
            self.wgp_path_edit.setText(directory)

    def validate_path(self, path):
        if not path or not os.path.isdir(path) or not os.path.isfile(os.path.join(path, 'wgp.py')):
            QMessageBox.warning(self, "Invalid Path", "The selected folder does not contain 'wgp.py'. Please select the correct WAN2GP folder.")
            return False
        return True

    def _save_config(self):
        path = self.wgp_path_edit.text()
        if not self.validate_path(path):
            return False
        self.config['wgp_path'] = path
        save_main_config()
        return True

    def save_and_close(self):
        if self._save_config():
            QMessageBox.information(self, "Settings Saved", "Settings have been saved. Please restart the application for changes to take effect.")
            self.accept()

    def save_and_restart(self):
        if self._save_config():
            self.parent().close()  # Close the main window before restarting
            os.execv(sys.executable, [sys.executable] + sys.argv)


class QueueTableWidget(QTableWidget):
    """A QTableWidget with drag-and-drop reordering for rows."""
    rowsMoved = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.DragDropMode.InternalMove)
        self.setSelectionBehavior(self.SelectionBehavior.SelectRows)
        self.setSelectionMode(self.SelectionMode.SingleSelection)

    def dropEvent(self, event: QDropEvent):
        if event.source() == self and event.dropAction() == Qt.DropAction.MoveAction:
            source_row = self.currentRow()
            target_item = self.itemAt(event.position().toPoint())
            dest_row = target_item.row() if target_item else self.rowCount()

            # Adjust destination row if moving down
            if source_row < dest_row:
                dest_row -=1

            if source_row != dest_row:
                self.rowsMoved.emit(source_row, dest_row)
            
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class Worker(QObject):
    progress = pyqtSignal(list)
    status = pyqtSignal(str)
    preview = pyqtSignal(object)
    output = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._is_running = True
        self._last_progress_phase = None
        self._last_preview = None

    def send_cmd(self, cmd, data=None):
        if not self._is_running:
            return

    def run(self):
        def generation_target():
            try:
                for _ in wgp.process_tasks(self.state):
                    if self._is_running:
                        self.output.emit()
                    else:
                        break
            except Exception as e:
                import traceback
                print("Error in generation thread:")
                traceback.print_exc()
                if "gradio.Error" in str(type(e)):
                    self.error.emit(str(e))
                else:
                    self.error.emit(f"An unexpected error occurred: {e}")
            finally:
                self._is_running = False

        gen_thread = threading.Thread(target=generation_target, daemon=True)
        gen_thread.start()

        while self._is_running:
            gen = self.state.get('gen', {})
            
            current_phase = gen.get("progress_phase")
            if current_phase and current_phase != self._last_progress_phase:
                self._last_progress_phase = current_phase
                
                phase_name, step = current_phase
                total_steps = gen.get("num_inference_steps", 1)
                high_level_status = gen.get("progress_status", "")

                status_msg = wgp.merge_status_context(high_level_status, phase_name)
                
                progress_args = [(step, total_steps), status_msg]
                self.progress.emit(progress_args)

            preview_img = gen.get('preview')
            if preview_img is not None and preview_img is not self._last_preview:
                self._last_preview = preview_img
                self.preview.emit(preview_img)
                gen['preview'] = None

            time.sleep(0.1)

        gen_thread.join()
        self.finished.emit()

class ApiBridge(QObject):
    """
    An object that lives in the main thread to receive signals from the API thread
    and forward them to the MainWindow's slots.
    """
    # --- CHANGE: Signal now includes model_type and duration_sec ---
    generateSignal = pyqtSignal(object, object, object, object, bool)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.api_bridge = ApiBridge()

        self.widgets = {}
        self.state = {}
        self.worker = None
        self.thread = None
        self.lora_map = {}
        self.full_resolution_choices = []
        self.latest_output_path = None
        
        self.setup_menu()
        
        if not wgp:
            self.setWindowTitle("WanGP - Setup Required")
            self.setGeometry(100, 100, 600, 200)
            self.setup_placeholder_ui()
            QTimer.singleShot(100, lambda: self.show_settings_dialog(first_time=True))
        else:
            self.setWindowTitle(f"WanGP v{wgp.WanGP_version} - Qt Interface")
            self.setGeometry(100, 100, 1400, 950)
            self.setup_full_ui()
            self.apply_initial_config()
            self.connect_signals()
            self.init_wgp_state()
            
            # --- CHANGE: Removed setModelSignal as it's no longer needed ---
            self.api_bridge.generateSignal.connect(self._api_generate)
    
    @pyqtSlot(str)
    def _api_set_model(self, model_type):
        """This slot is executed in the main GUI thread."""
        if not model_type or not wgp: return
        
        # 1. Check if the model is valid by looking it up in the master model definition dictionary.
        if model_type not in wgp.models_def:
            print(f"API Error: Model type '{model_type}' is not a valid model.")
            return

        # 2. Check if already selected to avoid unnecessary UI refreshes.
        if self.state.get('model_type') == model_type:
            print(f"API: Model is already set to {model_type}.")
            return

        # 3. Redraw all model dropdowns to ensure the correct hierarchy is displayed
        #    and the target model is selected. This function handles finding the
        #    correct Family and Base model for the given finetune model_type.
        self.update_model_dropdowns(model_type)
        
        # 4. Manually trigger the logic that normally runs when the user selects a model.
        #    This is necessary because update_model_dropdowns blocks signals.
        self._on_model_changed()

        # 5. Final check to see if the model was actually set.
        if self.state.get('model_type') == model_type:
            print(f"API: Successfully set model to {model_type}.")
        else:
            # This could happen if update_model_dropdowns silently fails to find the model.
            print(f"API Error: Failed to set model to '{model_type}'. The model might be hidden by your current configuration.")
    
    # --- CHANGE: Slot now accepts model_type and duration_sec, and calculates frame count ---
    @pyqtSlot(object, object, object, object, bool)
    def _api_generate(self, start_frame, end_frame, duration_sec, model_type, start_generation):
        """This slot is executed in the main GUI thread."""
        # 1. Set model if a new one is provided
        if model_type:
            self._api_set_model(model_type)

        # 2. Set frame inputs
        if start_frame:
            self.widgets['mode_s'].setChecked(True)
            self.widgets['image_start'].setText(start_frame)

        if end_frame:
            self.widgets['image_end_checkbox'].setChecked(True)
            self.widgets['image_end'].setText(end_frame)
        
        # 3. Calculate video length in frames based on duration and model FPS
        if duration_sec is not None:
            try:
                duration = float(duration_sec)
                
                # Get base FPS by parsing the "Force FPS" dropdown's default text (e.g., "Model Default (16 fps)")
                base_fps = 16 # Fallback
                fps_text = self.widgets['force_fps'].itemText(0) 
                match = re.search(r'\((\d+)\s*fps\)', fps_text)
                if match:
                    base_fps = int(match.group(1))

                # Temporal upsampling creates more frames in post-processing, so we must account for it here.
                upsample_setting = self.widgets['temporal_upsampling'].currentData()
                multiplier = 1.0
                if upsample_setting == "rife2":
                    multiplier = 2.0
                elif upsample_setting == "rife4":
                    multiplier = 4.0

                # The number of frames the model needs to generate
                video_length_frames = int(duration * base_fps * multiplier)
                
                self.widgets['video_length'].setValue(video_length_frames)
                print(f"API: Calculated video length: {video_length_frames} frames for {duration:.2f}s @ {base_fps*multiplier:.0f} effective FPS.")

            except (ValueError, TypeError) as e:
                print(f"API Error: Invalid duration_sec '{duration_sec}': {e}")
        
        # 4. Conditionally start generation
        if start_generation:
            self.generate_btn.click()
            print("API: Generation started.")
        else:
            print("API: Parameters set without starting generation.")


    def setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)

    def show_settings_dialog(self, first_time=False):
        dialog = SettingsDialog(main_config, self)
        if first_time:
            dialog.setWindowTitle("Initial Setup: Configure WAN2GP Path")
        dialog.exec()

    def setup_placeholder_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        placeholder_label = QLabel(
            "<h1>Welcome to WanGP</h1>"
            "<p>The path to your WAN2GP installation (the folder containing wgp.py) is not set.</p>"
            "<p>Please go to <b>File > Settings</b> to configure the path.</p>"
        )
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setWordWrap(True)
        layout.addWidget(placeholder_label)

    def setup_full_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.header_info = QLabel("Header Info")
        main_layout.addWidget(self.header_info)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.setup_generator_tab()
        self.setup_config_tab()

    def create_widget(self, widget_class, name, *args, **kwargs):
        widget = widget_class(*args, **kwargs)
        self.widgets[name] = widget
        return widget

    def _create_slider_with_label(self, name, min_val, max_val, initial_val, scale=1.0, precision=1):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)

        slider = self.create_widget(QSlider, name, Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(int(initial_val * scale))

        value_label = self.create_widget(QLabel, f"{name}_label", f"{initial_val:.{precision}f}")
        value_label.setMinimumWidth(50)
        
        slider.valueChanged.connect(
            lambda v, lbl=value_label, s=scale, p=precision: lbl.setText(f"{v/s:.{p}f}")
        )
        
        hbox.addWidget(slider)
        hbox.addWidget(value_label)
        return container

    def _create_file_input(self, name, label_text):
        container = self.create_widget(QWidget, f"{name}_container")
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        
        line_edit = self.create_widget(QLineEdit, name)
        line_edit.setReadOnly(False) # Allow user to paste paths
        line_edit.setPlaceholderText("No file selected or path pasted")
        
        button = QPushButton("Browse...")
        
        def open_dialog():
            # Allow selecting multiple files for reference images
            if "refs" in name:
                filenames, _ = QFileDialog.getOpenFileNames(self, f"Select {label_text}")
                if filenames:
                    line_edit.setText(";".join(filenames))
            else:
                filename, _ = QFileDialog.getOpenFileName(self, f"Select {label_text}")
                if filename:
                    line_edit.setText(filename)

        button.clicked.connect(open_dialog)
        
        clear_button = QPushButton("X")
        clear_button.setFixedWidth(30)
        clear_button.clicked.connect(lambda: line_edit.clear())

        hbox.addWidget(QLabel(f"{label_text}:"))
        hbox.addWidget(line_edit, 1)
        hbox.addWidget(button)
        hbox.addWidget(clear_button)
        return container
        
    def setup_generator_tab(self):
        gen_tab = QWidget()
        self.tabs.addTab(gen_tab, "Video Generator")
        gen_layout = QHBoxLayout(gen_tab)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        gen_layout.addWidget(left_panel, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        gen_layout.addWidget(right_panel, 1)
        
        # Left Panel (Inputs)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)
        
        options_widget = QWidget()
        scroll_area.setWidget(options_widget)
        options_layout = QVBoxLayout(options_widget)

        # Model Selection
        model_layout = QHBoxLayout()
        self.widgets['model_family'] = QComboBox()
        self.widgets['model_base_type_choice'] = QComboBox()
        self.widgets['model_choice'] = QComboBox()
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.widgets['model_family'], 2)
        model_layout.addWidget(self.widgets['model_base_type_choice'], 3)
        model_layout.addWidget(self.widgets['model_choice'], 3)
        options_layout.addLayout(model_layout)

        # Prompt
        options_layout.addWidget(QLabel("Prompt:"))
        self.create_widget(QTextEdit, 'prompt').setMinimumHeight(100)
        options_layout.addWidget(self.widgets['prompt'])

        options_layout.addWidget(QLabel("Negative Prompt:"))
        self.create_widget(QTextEdit, 'negative_prompt').setMinimumHeight(60)
        options_layout.addWidget(self.widgets['negative_prompt'])

        # Basic controls
        basic_group = QGroupBox("Basic Options")
        basic_layout = QFormLayout(basic_group)
        
        res_container = QWidget()
        res_hbox = QHBoxLayout(res_container)
        res_hbox.setContentsMargins(0, 0, 0, 0)
        res_hbox.addWidget(self.create_widget(QComboBox, 'resolution_group'), 2)
        res_hbox.addWidget(self.create_widget(QComboBox, 'resolution'), 3)
        basic_layout.addRow("Resolution:", res_container)

        basic_layout.addRow("Video Length:", self._create_slider_with_label('video_length', 1, 737, 81, 1.0, 0))
        basic_layout.addRow("Inference Steps:", self._create_slider_with_label('num_inference_steps', 1, 100, 30, 1.0, 0))
        basic_layout.addRow("Seed:", self.create_widget(QLineEdit, 'seed', '-1'))
        options_layout.addWidget(basic_group)
        
        # Generation Mode and Input Options
        mode_options_group = QGroupBox("Generation Mode & Input Options")
        mode_options_layout = QVBoxLayout(mode_options_group)
        
        mode_hbox = QHBoxLayout()
        mode_hbox.addWidget(self.create_widget(QRadioButton, 'mode_t', "Text Prompt Only"))
        mode_hbox.addWidget(self.create_widget(QRadioButton, 'mode_s', "Start with Image"))
        mode_hbox.addWidget(self.create_widget(QRadioButton, 'mode_v', "Continue Video"))
        mode_hbox.addWidget(self.create_widget(QRadioButton, 'mode_l', "Continue Last Video"))
        self.widgets['mode_t'].setChecked(True)
        mode_options_layout.addLayout(mode_hbox)

        options_hbox = QHBoxLayout()
        options_hbox.addWidget(self.create_widget(QCheckBox, 'image_end_checkbox', "Use End Image"))
        options_hbox.addWidget(self.create_widget(QCheckBox, 'control_video_checkbox', "Use Control Video"))
        options_hbox.addWidget(self.create_widget(QCheckBox, 'ref_image_checkbox', "Use Reference Image(s)"))
        mode_options_layout.addLayout(options_hbox)
        options_layout.addWidget(mode_options_group)

        # Dynamic Inputs
        inputs_group = QGroupBox("Inputs")
        inputs_layout = QVBoxLayout(inputs_group)
        inputs_layout.addWidget(self._create_file_input('image_start', "Start Image"))
        inputs_layout.addWidget(self._create_file_input('image_end', "End Image"))
        inputs_layout.addWidget(self._create_file_input('video_source', "Source Video"))
        inputs_layout.addWidget(self._create_file_input('video_guide', "Control Video"))
        inputs_layout.addWidget(self._create_file_input('video_mask', "Video Mask"))
        inputs_layout.addWidget(self._create_file_input('image_refs', "Reference Image(s)"))
        denoising_row = QFormLayout()
        denoising_row.addRow("Denoising Strength:", self._create_slider_with_label('denoising_strength', 0, 100, 50, 100.0, 2))
        inputs_layout.addLayout(denoising_row)
        options_layout.addWidget(inputs_group)

        # Advanced controls
        self.advanced_group = self.create_widget(QGroupBox, 'advanced_group', "Advanced Options")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced_group)
        
        advanced_tabs = self.create_widget(QTabWidget, 'advanced_tabs')
        advanced_layout.addWidget(advanced_tabs)

        self._setup_adv_tab_general(advanced_tabs)
        self._setup_adv_tab_loras(advanced_tabs)
        self._setup_adv_tab_speed(advanced_tabs)
        self._setup_adv_tab_postproc(advanced_tabs)
        self._setup_adv_tab_audio(advanced_tabs)
        self._setup_adv_tab_quality(advanced_tabs)
        self._setup_adv_tab_sliding_window(advanced_tabs)
        self._setup_adv_tab_misc(advanced_tabs)

        options_layout.addWidget(self.advanced_group)

        # Right Panel (Output & Queue)
        btn_layout = QHBoxLayout()
        self.generate_btn = self.create_widget(QPushButton, 'generate_btn', "Generate")
        self.add_to_queue_btn = self.create_widget(QPushButton, 'add_to_queue_btn', "Add to Queue")
        self.generate_btn.setEnabled(True)
        self.add_to_queue_btn.setEnabled(False)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.add_to_queue_btn)
        right_layout.addLayout(btn_layout)

        self.status_label = self.create_widget(QLabel, 'status_label', "Idle")
        right_layout.addWidget(self.status_label)
        self.progress_bar = self.create_widget(QProgressBar, 'progress_bar')
        right_layout.addWidget(self.progress_bar)

        preview_group = self.create_widget(QGroupBox, 'preview_group', "Preview")
        preview_group.setCheckable(True)
        preview_group.setStyleSheet("QGroupBox { border: 1px solid #cccccc; }")
        preview_group_layout = QVBoxLayout(preview_group)

        self.preview_image = self.create_widget(QLabel, 'preview_image', "")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(200, 200)
        
        preview_group_layout.addWidget(self.preview_image)
        right_layout.addWidget(preview_group)

        right_layout.addWidget(QLabel("Output:"))
        self.output_gallery = self.create_widget(QListWidget, 'output_gallery')
        right_layout.addWidget(self.output_gallery)

        right_layout.addWidget(QLabel("Queue:"))
        self.queue_table = self.create_widget(QueueTableWidget, 'queue_table')
        right_layout.addWidget(self.queue_table)
        
        queue_btn_layout = QHBoxLayout()
        self.remove_queue_btn = self.create_widget(QPushButton, 'remove_queue_btn', "Remove Selected")
        self.clear_queue_btn = self.create_widget(QPushButton, 'clear_queue_btn', "Clear Queue")
        self.abort_btn = self.create_widget(QPushButton, 'abort_btn', "Abort")
        queue_btn_layout.addWidget(self.remove_queue_btn)
        queue_btn_layout.addWidget(self.clear_queue_btn)
        queue_btn_layout.addWidget(self.abort_btn)
        right_layout.addLayout(queue_btn_layout)

    def _setup_adv_tab_general(self, tabs):
        tab = QWidget()
        tabs.addTab(tab, "General")
        layout = QFormLayout(tab)
        self.widgets['adv_general_layout'] = layout

        guidance_group = QGroupBox("Guidance")
        guidance_layout = self.create_widget(QFormLayout, 'guidance_layout', guidance_group)
        guidance_layout.addRow("Guidance (CFG):", self._create_slider_with_label('guidance_scale', 10, 200, 5.0, 10.0, 1))
        
        self.widgets['guidance_phases_row_index'] = guidance_layout.rowCount()
        guidance_layout.addRow("Guidance Phases:", self.create_widget(QComboBox, 'guidance_phases'))
        
        self.widgets['guidance2_row_index'] = guidance_layout.rowCount()
        guidance_layout.addRow("Guidance 2:", self._create_slider_with_label('guidance2_scale', 10, 200, 5.0, 10.0, 1))
        self.widgets['guidance3_row_index'] = guidance_layout.rowCount()
        guidance_layout.addRow("Guidance 3:", self._create_slider_with_label('guidance3_scale', 10, 200, 5.0, 10.0, 1))
        self.widgets['switch_thresh_row_index'] = guidance_layout.rowCount()
        guidance_layout.addRow("Switch Threshold:", self._create_slider_with_label('switch_threshold', 0, 1000, 0, 1.0, 0))
        layout.addRow(guidance_group)

        nag_group = self.create_widget(QGroupBox, 'nag_group', "NAG (Negative Adversarial Guidance)")
        nag_layout = QFormLayout(nag_group)
        nag_layout.addRow("NAG Scale:", self._create_slider_with_label('NAG_scale', 10, 200, 1.0, 10.0, 1))
        nag_layout.addRow("NAG Tau:", self._create_slider_with_label('NAG_tau', 10, 50, 3.5, 10.0, 1))
        nag_layout.addRow("NAG Alpha:", self._create_slider_with_label('NAG_alpha', 0, 20, 0.5, 10.0, 1))
        layout.addRow(nag_group)

        self.widgets['solver_row_container'] = QWidget()
        solver_hbox = QHBoxLayout(self.widgets['solver_row_container'])
        solver_hbox.setContentsMargins(0,0,0,0)
        solver_hbox.addWidget(QLabel("Sampler Solver:"))
        solver_hbox.addWidget(self.create_widget(QComboBox, 'sample_solver'))
        layout.addRow(self.widgets['solver_row_container'])
        
        self.widgets['flow_shift_row_index'] = layout.rowCount()
        layout.addRow("Shift Scale:", self._create_slider_with_label('flow_shift', 10, 250, 3.0, 10.0, 1))
        
        self.widgets['audio_guidance_row_index'] = layout.rowCount()
        layout.addRow("Audio Guidance:", self._create_slider_with_label('audio_guidance_scale', 10, 200, 4.0, 10.0, 1))
        
        self.widgets['repeat_generation_row_index'] = layout.rowCount()
        layout.addRow("Repeat Generations:", self._create_slider_with_label('repeat_generation', 1, 25, 1, 1.0, 0))
        
        combo = self.create_widget(QComboBox, 'multi_images_gen_type')
        combo.addItem("Generate all combinations", 0)
        combo.addItem("Match images and texts", 1)
        self.widgets['multi_images_gen_type_row_index'] = layout.rowCount()
        layout.addRow("Multi-Image Mode:", combo)

    def _setup_adv_tab_loras(self, tabs):
        tab = QWidget()
        tabs.addTab(tab, "Loras")
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Available Loras (Ctrl+Click to select multiple):"))
        lora_list = self.create_widget(QListWidget, 'activated_loras')
        lora_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(lora_list) 
        layout.addWidget(QLabel("Loras Multipliers:"))
        layout.addWidget(self.create_widget(QTextEdit, 'loras_multipliers'))

    def _setup_adv_tab_speed(self, tabs):
        tab = QWidget()
        tabs.addTab(tab, "Speed")
        layout = QFormLayout(tab)
        combo = self.create_widget(QComboBox, 'skip_steps_cache_type')
        combo.addItem("None", "")
        combo.addItem("Tea Cache", "tea")
        combo.addItem("Mag Cache", "mag")
        layout.addRow("Cache Type:", combo)
        
        combo = self.create_widget(QComboBox, 'skip_steps_multiplier')
        combo.addItem("x1.5 speed up", 1.5)
        combo.addItem("x1.75 speed up", 1.75)
        combo.addItem("x2.0 speed up", 2.0)
        combo.addItem("x2.25 speed up", 2.25)
        combo.addItem("x2.5 speed up", 2.5)
        layout.addRow("Acceleration:", combo)
        layout.addRow("Start %:", self._create_slider_with_label('skip_steps_start_step_perc', 0, 100, 0, 1.0, 0))

    def _setup_adv_tab_postproc(self, tabs):
        tab = QWidget()
        tabs.addTab(tab, "Post-Processing")
        layout = QFormLayout(tab)
        combo = self.create_widget(QComboBox, 'temporal_upsampling')
        combo.addItem("Disabled", "")
        combo.addItem("Rife x2 frames/s", "rife2")
        combo.addItem("Rife x4 frames/s", "rife4")
        layout.addRow("Temporal Upsampling:", combo)

        combo = self.create_widget(QComboBox, 'spatial_upsampling')
        combo.addItem("Disabled", "")
        combo.addItem("Lanczos x1.5", "lanczos1.5")
        combo.addItem("Lanczos x2.0", "lanczos2")
        layout.addRow("Spatial Upsampling:", combo)
        
        layout.addRow("Film Grain Intensity:", self._create_slider_with_label('film_grain_intensity', 0, 100, 0, 100.0, 2))
        layout.addRow("Film Grain Saturation:", self._create_slider_with_label('film_grain_saturation', 0, 100, 0.5, 100.0, 2))

    def _setup_adv_tab_audio(self, tabs):
        tab = QWidget()
        tabs.addTab(tab, "Audio")
        layout = QFormLayout(tab)
        combo = self.create_widget(QComboBox, 'MMAudio_setting')
        combo.addItem("Disabled", 0)
        combo.addItem("Enabled", 1)
        layout.addRow("MMAudio:", combo)
        layout.addWidget(self.create_widget(QLineEdit, 'MMAudio_prompt', placeholderText="MMAudio Prompt"))
        layout.addWidget(self.create_widget(QLineEdit, 'MMAudio_neg_prompt', placeholderText="MMAudio Negative Prompt"))
        layout.addRow(self._create_file_input('audio_source', "Custom Soundtrack"))

    def _setup_adv_tab_quality(self, tabs):
        tab = QWidget()
        tabs.addTab(tab, "Quality")
        layout = QVBoxLayout(tab)
        
        slg_group = self.create_widget(QGroupBox, 'slg_group', "Skip Layer Guidance")
        slg_layout = QFormLayout(slg_group)
        slg_combo = self.create_widget(QComboBox, 'slg_switch')
        slg_combo.addItem("OFF", 0)
        slg_combo.addItem("ON", 1)
        slg_layout.addRow("Enable SLG:", slg_combo)
        slg_layout.addRow("Start %:", self._create_slider_with_label('slg_start_perc', 0, 100, 10, 1.0, 0))
        slg_layout.addRow("End %:", self._create_slider_with_label('slg_end_perc', 0, 100, 90, 1.0, 0))
        layout.addWidget(slg_group)

        quality_form = QFormLayout()
        self.widgets['quality_form_layout'] = quality_form

        apg_combo = self.create_widget(QComboBox, 'apg_switch')
        apg_combo.addItem("OFF", 0)
        apg_combo.addItem("ON", 1)
        self.widgets['apg_switch_row_index'] = quality_form.rowCount()
        quality_form.addRow("Adaptive Projected Guidance:", apg_combo)

        cfg_star_combo = self.create_widget(QComboBox, 'cfg_star_switch')
        cfg_star_combo.addItem("OFF", 0)
        cfg_star_combo.addItem("ON", 1)
        self.widgets['cfg_star_switch_row_index'] = quality_form.rowCount()
        quality_form.addRow("Classifier-Free Guidance Star:", cfg_star_combo)

        self.widgets['cfg_zero_step_row_index'] = quality_form.rowCount()
        quality_form.addRow("CFG Zero below Layer:", self._create_slider_with_label('cfg_zero_step', -1, 39, -1, 1.0, 0))

        combo = self.create_widget(QComboBox, 'min_frames_if_references')
        combo.addItem("Disabled (1 frame)", 1)
        combo.addItem("Generate 5 frames", 5)
        combo.addItem("Generate 9 frames", 9)
        combo.addItem("Generate 13 frames", 13)
        combo.addItem("Generate 17 frames", 17)
        self.widgets['min_frames_if_references_row_index'] = quality_form.rowCount()
        quality_form.addRow("Min Frames for Quality:", combo)
        layout.addLayout(quality_form)

    def _setup_adv_tab_sliding_window(self, tabs):
        tab = QWidget()
        self.widgets['sliding_window_tab_index'] = tabs.count()
        tabs.addTab(tab, "Sliding Window")
        layout = QFormLayout(tab)

        layout.addRow("Window Size:", self._create_slider_with_label('sliding_window_size', 5, 257, 129, 1.0, 0))
        layout.addRow("Overlap:", self._create_slider_with_label('sliding_window_overlap', 1, 97, 5, 1.0, 0))
        layout.addRow("Color Correction:", self._create_slider_with_label('sliding_window_color_correction_strength', 0, 100, 0, 100.0, 2))
        layout.addRow("Overlap Noise:", self._create_slider_with_label('sliding_window_overlap_noise', 0, 150, 20, 1.0, 0))
        layout.addRow("Discard Last Frames:", self._create_slider_with_label('sliding_window_discard_last_frames', 0, 20, 0, 1.0, 0))

    def _setup_adv_tab_misc(self, tabs):
        tab = QWidget()
        tabs.addTab(tab, "Misc")
        layout = QFormLayout(tab)
        self.widgets['misc_layout'] = layout

        riflex_combo = self.create_widget(QComboBox, 'RIFLEx_setting')
        riflex_combo.addItem("Auto", 0)
        riflex_combo.addItem("Always ON", 1)
        riflex_combo.addItem("Always OFF", 2)
        self.widgets['riflex_row_index'] = layout.rowCount()
        layout.addRow("RIFLEx Setting:", riflex_combo)

        fps_combo = self.create_widget(QComboBox, 'force_fps')
        layout.addRow("Force FPS:", fps_combo)

        profile_combo = self.create_widget(QComboBox, 'override_profile')
        profile_combo.addItem("Default Profile", -1)
        for text, val in wgp.memory_profile_choices:
            profile_combo.addItem(text.split(':')[0], val)
        layout.addRow("Override Memory Profile:", profile_combo)

        combo = self.create_widget(QComboBox, 'multi_prompts_gen_type')
        combo.addItem("Generate new Video per line", 0)
        combo.addItem("Use line for new Sliding Window", 1)
        layout.addRow("Multi-Prompt Mode:", combo)

    def setup_config_tab(self):
        config_tab = QWidget()
        self.tabs.addTab(config_tab, "Configuration")
        main_layout = QVBoxLayout(config_tab)

        self.config_status_label = QLabel("Apply changes for them to take effect. Some may require a restart.")
        main_layout.addWidget(self.config_status_label)

        config_tabs = QTabWidget()
        main_layout.addWidget(config_tabs)

        config_tabs.addTab(self._create_general_config_tab(), "General")
        config_tabs.addTab(self._create_performance_config_tab(), "Performance")
        config_tabs.addTab(self._create_extensions_config_tab(), "Extensions")
        config_tabs.addTab(self._create_outputs_config_tab(), "Outputs")
        config_tabs.addTab(self._create_notifications_config_tab(), "Notifications")

        self.apply_config_btn = QPushButton("Apply Changes")
        self.apply_config_btn.clicked.connect(self._on_apply_config_changes)
        main_layout.addWidget(self.apply_config_btn)
    
    def _create_scrollable_form_tab(self):
        tab_widget = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout = QVBoxLayout(tab_widget)
        layout.addWidget(scroll_area)
        
        content_widget = QWidget()
        form_layout = QFormLayout(content_widget)
        scroll_area.setWidget(content_widget)
        
        return tab_widget, form_layout

    def _create_config_combo(self, form_layout, label, key, choices, default_value):
        combo = QComboBox()
        for text, data in choices:
            combo.addItem(text, data)
        index = combo.findData(wgp.server_config.get(key, default_value))
        if index != -1: combo.setCurrentIndex(index)
        self.widgets[f'config_{key}'] = combo
        form_layout.addRow(label, combo)

    def _create_config_slider(self, form_layout, label, key, min_val, max_val, default_value, step=1):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0,0,0,0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setSingleStep(step)
        slider.setValue(wgp.server_config.get(key, default_value))
        
        value_label = QLabel(str(slider.value()))
        value_label.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
        
        hbox.addWidget(slider)
        hbox.addWidget(value_label)
        
        self.widgets[f'config_{key}'] = slider
        form_layout.addRow(label, container)

    def _create_config_checklist(self, form_layout, label, key, choices, default_value):
        list_widget = QListWidget()
        list_widget.setMinimumHeight(100)
        current_values = wgp.server_config.get(key, default_value)
        for text, data in choices:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, data)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if data in current_values:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            list_widget.addItem(item)
        self.widgets[f'config_{key}'] = list_widget
        form_layout.addRow(label, list_widget)

    def _create_config_textbox(self, form_layout, label, key, default_value, multi_line=False):
        if multi_line:
            textbox = QTextEdit(default_value)
            textbox.setAcceptRichText(False)
        else:
            textbox = QLineEdit(default_value)
        self.widgets[f'config_{key}'] = textbox
        form_layout.addRow(label, textbox)

    def _create_general_config_tab(self):
        tab, form = self._create_scrollable_form_tab()
        
        _, _, dropdown_choices = wgp.get_sorted_dropdown(wgp.displayed_model_types, None, None, False)
        self._create_config_checklist(form, "Selectable Models:", "transformer_types", dropdown_choices, wgp.transformer_types)

        self._create_config_combo(form, "Model Hierarchy:", "model_hierarchy_type", [
            ("Two Levels (Family > Model)", 0),
            ("Three Levels (Family > Base > Finetune)", 1)
        ], 1)

        self._create_config_combo(form, "Video Dimensions:", "fit_canvas", [
            ("Dimensions are Pixels Budget", 0), ("Dimensions are Max Width/Height", 1),
            ("Dimensions are Output Width/Height (Cropped)", 2)], 0)
        
        self._create_config_combo(form, "Attention Type:", "attention_mode", [
            ("Auto (Recommended)", "auto"), ("SDPA", "sdpa"), ("Flash", "flash"),
            ("Xformers", "xformers"), ("Sage", "sage"), ("Sage2/2++", "sage2")], "auto")
        
        self._create_config_combo(form, "Metadata Handling:", "metadata_type", [
            ("Embed in file (Exif/Comment)", "metadata"), ("Export separate JSON", "json"), ("None", "none")], "metadata")

        self._create_config_checklist(form, "RAM Loading Policy:", "preload_model_policy", [
            ("Preload on App Launch", "P"), ("Preload on Model Switch", "S"), ("Unload when Queue is Done", "U")], [])
        
        self._create_config_combo(form, "Keep Previous Videos:", "clear_file_list", [
            ("None", 0), ("Keep last video", 1), ("Keep last 5", 5), ("Keep last 10", 10),
            ("Keep last 20", 20), ("Keep last 30", 30)], 5)
        
        self._create_config_combo(form, "Display RAM/VRAM Stats:", "display_stats", [("Disabled", 0), ("Enabled", 1)], 0)
        
        self._create_config_combo(form, "Max Frames Multiplier:", "max_frames_multiplier",
                                  [(f"x{i}", i) for i in range(1, 8)], 1)
        
        checkpoints_paths_text = "\n".join(wgp.server_config.get("checkpoints_paths", wgp.fl.default_checkpoints_paths))
        checkpoints_textbox = QTextEdit()
        checkpoints_textbox.setPlainText(checkpoints_paths_text)
        checkpoints_textbox.setAcceptRichText(False)
        checkpoints_textbox.setMinimumHeight(60)
        self.widgets['config_checkpoints_paths'] = checkpoints_textbox
        form.addRow("Checkpoints Paths:", checkpoints_textbox)
                                  
        self._create_config_combo(form, "UI Theme (requires restart):", "UI_theme", [("Blue Sky", "default"), ("Classic Gradio", "gradio")], "default")
        
        return tab

    def _create_performance_config_tab(self):
        tab, form = self._create_scrollable_form_tab()
        self._create_config_combo(form, "Transformer Quantization:", "transformer_quantization", [("Scaled Int8 (recommended)", "int8"), ("16-bit (no quantization)", "bf16")], "int8")
        self._create_config_combo(form, "Transformer Data Type:", "transformer_dtype_policy", [("Best Supported by Hardware", ""), ("FP16", "fp16"), ("BF16", "bf16")], "")
        self._create_config_combo(form, "Transformer Calculation:", "mixed_precision", [("16-bit only", "0"), ("Mixed 16/32-bit (better quality)", "1")], "0")
        self._create_config_combo(form, "Text Encoder:", "text_encoder_quantization", [("16-bit (more RAM, better quality)", "bf16"), ("8-bit (less RAM)", "int8")], "int8")
        self._create_config_combo(form, "VAE Precision:", "vae_precision", [("16-bit (faster, less VRAM)", "16"), ("32-bit (slower, better quality)", "32")], "16")
        self._create_config_combo(form, "Compile Transformer:", "compile", [("On (requires Triton)", "transformer"), ("Off", "")], "")
        self._create_config_combo(form, "DepthAnything v2 Variant:", "depth_anything_v2_variant", [("Large (more precise)", "vitl"), ("Big (faster)", "vitb")], "vitl")
        self._create_config_combo(form, "VAE Tiling:", "vae_config", [("Auto", 0), ("Disabled", 1), ("256x256 (~8GB VRAM)", 2), ("128x128 (~6GB VRAM)", 3)], 0)
        self._create_config_combo(form, "Boost:", "boost", [("On", 1), ("Off", 2)], 1)
        self._create_config_combo(form, "Memory Profile:", "profile", wgp.memory_profile_choices, wgp.profile_type.LowRAM_LowVRAM)
        self._create_config_slider(form, "Preload in VRAM (MB):", "preload_in_VRAM", 0, 40000, 0, 100)
        
        release_ram_btn = QPushButton("Force Release Models from RAM")
        release_ram_btn.clicked.connect(self._on_release_ram)
        form.addRow(release_ram_btn)
        
        return tab

    def _create_extensions_config_tab(self):
        tab, form = self._create_scrollable_form_tab()
        self._create_config_combo(form, "Prompt Enhancer:", "enhancer_enabled", [("Off", 0), ("Florence 2 + Llama 3.2", 1), ("Florence 2 + Joy Caption (uncensored)", 2)], 0)
        self._create_config_combo(form, "Enhancer Mode:", "enhancer_mode", [("Automatic on Generate", 0), ("On Demand Only", 1)], 0)
        self._create_config_combo(form, "MMAudio:", "mmaudio_enabled", [("Off", 0), ("Enabled (unloaded after use)", 1), ("Enabled (persistent in RAM)", 2)], 0)
        return tab

    def _create_outputs_config_tab(self):
        tab, form = self._create_scrollable_form_tab()
        self._create_config_combo(form, "Video Codec:", "video_output_codec", [("x265 Balanced", 'libx265_28'), ("x264 Balanced", 'libx264_8'), ("x265 High Quality", 'libx265_8'), ("x264 High Quality", 'libx264_10'), ("x264 Lossless", 'libx264_lossless')], 'libx264_8')
        self._create_config_combo(form, "Image Codec:", "image_output_codec", [("JPEG Q85", 'jpeg_85'), ("WEBP Q85", 'webp_85'), ("JPEG Q95", 'jpeg_95'), ("WEBP Q95", 'webp_95'), ("WEBP Lossless", 'webp_lossless'), ("PNG Lossless", 'png')], 'jpeg_95')
        self._create_config_textbox(form, "Video Output Folder:", "save_path", "outputs")
        self._create_config_textbox(form, "Image Output Folder:", "image_save_path", "outputs")
        return tab

    def _create_notifications_config_tab(self):
        tab, form = self._create_scrollable_form_tab()
        self._create_config_combo(form, "Notification Sound:", "notification_sound_enabled", [("On", 1), ("Off", 0)], 0)
        self._create_config_slider(form, "Sound Volume:", "notification_sound_volume", 0, 100, 50, 5)
        return tab
        
    def init_wgp_state(self):
        initial_model = wgp.server_config.get("last_model_type", wgp.transformer_type)
        all_models, _, _ = wgp.get_sorted_dropdown(wgp.displayed_model_types, None, None, False)
        all_model_ids = [m[1] for m in all_models]
        if initial_model not in all_model_ids:
            initial_model = wgp.transformer_type
        
        state_dict = {}
        state_dict["model_filename"] = wgp.get_model_filename(initial_model, wgp.transformer_quantization, wgp.transformer_dtype_policy)
        state_dict["model_type"] = initial_model
        state_dict["advanced"] = wgp.advanced
        state_dict["last_model_per_family"] = wgp.server_config.get("last_model_per_family", {})
        state_dict["last_model_per_type"] = wgp.server_config.get("last_model_per_type", {})
        state_dict["last_resolution_per_group"] = wgp.server_config.get("last_resolution_per_group", {})
        state_dict["gen"] = {"queue": []}
        
        self.state = state_dict
        self.advanced_group.setChecked(wgp.advanced)

        self.update_model_dropdowns(initial_model)
        self.refresh_ui_from_model_change(initial_model)
        self._update_input_visibility() # Set initial visibility

    def update_model_dropdowns(self, current_model_type):
        family_mock, base_type_mock, choice_mock = wgp.generate_dropdown_model_list(current_model_type)

        self.widgets['model_family'].blockSignals(True)
        self.widgets['model_base_type_choice'].blockSignals(True)
        self.widgets['model_choice'].blockSignals(True)
        
        self.widgets['model_family'].clear()
        if family_mock.choices:
            for display_name, internal_key in family_mock.choices:
                self.widgets['model_family'].addItem(display_name, internal_key)
        index = self.widgets['model_family'].findData(family_mock.value)
        if index != -1: self.widgets['model_family'].setCurrentIndex(index)
        
        self.widgets['model_base_type_choice'].clear()
        if base_type_mock.choices:
            for label, value in base_type_mock.choices:
                self.widgets['model_base_type_choice'].addItem(label, value)
        index = self.widgets['model_base_type_choice'].findData(base_type_mock.value)
        if index != -1: self.widgets['model_base_type_choice'].setCurrentIndex(index)
        self.widgets['model_base_type_choice'].setVisible(base_type_mock.kwargs.get('visible', True))

        self.widgets['model_choice'].clear()
        if choice_mock.choices:
            for label, value in choice_mock.choices: self.widgets['model_choice'].addItem(label, value)
        index = self.widgets['model_choice'].findData(choice_mock.value)
        if index != -1: self.widgets['model_choice'].setCurrentIndex(index)
        self.widgets['model_choice'].setVisible(choice_mock.kwargs.get('visible', True))

        self.widgets['model_family'].blockSignals(False)
        self.widgets['model_base_type_choice'].blockSignals(False)
        self.widgets['model_choice'].blockSignals(False)

    def refresh_ui_from_model_change(self, model_type):
        """Update UI controls with default settings when the model is changed."""
        self.header_info.setText(wgp.generate_header(model_type, wgp.compile, wgp.attention_mode))
        ui_defaults = wgp.get_default_settings(model_type)
        wgp.set_model_settings(self.state, model_type, ui_defaults)

        model_def = wgp.get_model_def(model_type)
        base_model_type = wgp.get_base_model_type(model_type)
        model_filename = self.state.get('model_filename', '')

        image_outputs = model_def.get("image_outputs", False)
        vace = wgp.test_vace_module(model_type)
        t2v = base_model_type in ['t2v', 't2v_2_2']
        i2v = wgp.test_class_i2v(model_type)
        fantasy = base_model_type in ["fantasy"]
        multitalk = model_def.get("multitalk_class", False)
        any_audio_guidance = fantasy or multitalk
        sliding_window_enabled = wgp.test_any_sliding_window(model_type)
        recammaster = base_model_type in ["recam_1.3B"]
        ltxv = "ltxv" in model_filename
        diffusion_forcing = "diffusion_forcing" in model_filename
        any_skip_layer_guidance = model_def.get("skip_layer_guidance", False)
        any_cfg_zero = model_def.get("cfg_zero", False)
        any_cfg_star = model_def.get("cfg_star", False)
        any_apg = model_def.get("adaptive_projected_guidance", False)
        v2i_switch_supported = model_def.get("v2i_switch_supported", False)

        self._update_generation_mode_visibility(model_def)

        for widget in self.widgets.values():
            if hasattr(widget, 'blockSignals'): widget.blockSignals(True)

        self.widgets['prompt'].setText(ui_defaults.get("prompt", ""))
        self.widgets['negative_prompt'].setText(ui_defaults.get("negative_prompt", ""))
        self.widgets['seed'].setText(str(ui_defaults.get("seed", -1)))
        
        video_length_val = ui_defaults.get("video_length", 81)
        self.widgets['video_length'].setValue(video_length_val)
        self.widgets['video_length_label'].setText(str(video_length_val))

        steps_val = ui_defaults.get("num_inference_steps", 30)
        self.widgets['num_inference_steps'].setValue(steps_val)
        self.widgets['num_inference_steps_label'].setText(str(steps_val))

        self.widgets['resolution_group'].blockSignals(True)
        self.widgets['resolution'].blockSignals(True)

        current_res_choice = ui_defaults.get("resolution")
        model_resolutions = model_def.get("resolutions", None)
        self.full_resolution_choices, current_res_choice = wgp.get_resolution_choices(current_res_choice, model_resolutions)
        available_groups, selected_group_resolutions, selected_group = wgp.group_resolutions(model_def, self.full_resolution_choices, current_res_choice)

        self.widgets['resolution_group'].clear()
        self.widgets['resolution_group'].addItems(available_groups)
        group_index = self.widgets['resolution_group'].findText(selected_group)
        if group_index != -1:
            self.widgets['resolution_group'].setCurrentIndex(group_index)
        
        self.widgets['resolution'].clear()
        for label, value in selected_group_resolutions:
            self.widgets['resolution'].addItem(label, value)
        res_index = self.widgets['resolution'].findData(current_res_choice)
        if res_index != -1:
            self.widgets['resolution'].setCurrentIndex(res_index)

        self.widgets['resolution_group'].blockSignals(False)
        self.widgets['resolution'].blockSignals(False)


        for name in ['video_source', 'image_start', 'image_end', 'video_guide', 'video_mask', 'audio_source']:
            if name in self.widgets:
                self.widgets[name].clear()

        guidance_layout = self.widgets['guidance_layout']
        guidance_max = model_def.get("guidance_max_phases", 1)
        guidance_layout.setRowVisible(self.widgets['guidance_phases_row_index'], guidance_max > 1)

        adv_general_layout = self.widgets['adv_general_layout']
        adv_general_layout.setRowVisible(self.widgets['flow_shift_row_index'], not image_outputs)
        adv_general_layout.setRowVisible(self.widgets['audio_guidance_row_index'], any_audio_guidance)
        adv_general_layout.setRowVisible(self.widgets['repeat_generation_row_index'], not image_outputs)
        adv_general_layout.setRowVisible(self.widgets['multi_images_gen_type_row_index'], i2v)
        
        self.widgets['slg_group'].setVisible(any_skip_layer_guidance)
        quality_form_layout = self.widgets['quality_form_layout']
        quality_form_layout.setRowVisible(self.widgets['apg_switch_row_index'], any_apg)
        quality_form_layout.setRowVisible(self.widgets['cfg_star_switch_row_index'], any_cfg_star)
        quality_form_layout.setRowVisible(self.widgets['cfg_zero_step_row_index'], any_cfg_zero)
        quality_form_layout.setRowVisible(self.widgets['min_frames_if_references_row_index'], v2i_switch_supported and image_outputs)

        self.widgets['advanced_tabs'].setTabVisible(self.widgets['sliding_window_tab_index'], sliding_window_enabled and not image_outputs)
        
        misc_layout = self.widgets['misc_layout']
        misc_layout.setRowVisible(self.widgets['riflex_row_index'], not (recammaster or ltxv or diffusion_forcing))


        index = self.widgets['multi_images_gen_type'].findData(ui_defaults.get('multi_images_gen_type', 0))
        if index != -1: self.widgets['multi_images_gen_type'].setCurrentIndex(index)

        guidance_val = ui_defaults.get("guidance_scale", 5.0)
        self.widgets['guidance_scale'].setValue(int(guidance_val * 10))
        self.widgets['guidance_scale_label'].setText(f"{guidance_val:.1f}")

        guidance2_val = ui_defaults.get("guidance2_scale", 5.0)
        self.widgets['guidance2_scale'].setValue(int(guidance2_val * 10))
        self.widgets['guidance2_scale_label'].setText(f"{guidance2_val:.1f}")
        
        guidance3_val = ui_defaults.get("guidance3_scale", 5.0)
        self.widgets['guidance3_scale'].setValue(int(guidance3_val * 10))
        self.widgets['guidance3_scale_label'].setText(f"{guidance3_val:.1f}")
        self.widgets['guidance_phases'].clear()
        
        if guidance_max >= 1: self.widgets['guidance_phases'].addItem("One Phase", 1)
        if guidance_max >= 2: self.widgets['guidance_phases'].addItem("Two Phases", 2)
        if guidance_max >= 3: self.widgets['guidance_phases'].addItem("Three Phases", 3)
        
        index = self.widgets['guidance_phases'].findData(ui_defaults.get("guidance_phases", 1))
        if index != -1: self.widgets['guidance_phases'].setCurrentIndex(index)
        
        switch_thresh_val = ui_defaults.get("switch_threshold", 0)
        self.widgets['switch_threshold'].setValue(switch_thresh_val)
        self.widgets['switch_threshold_label'].setText(str(switch_thresh_val))

        nag_scale_val = ui_defaults.get('NAG_scale', 1.0)
        self.widgets['NAG_scale'].setValue(int(nag_scale_val * 10))
        self.widgets['NAG_scale_label'].setText(f"{nag_scale_val:.1f}")

        nag_tau_val = ui_defaults.get('NAG_tau', 3.5)
        self.widgets['NAG_tau'].setValue(int(nag_tau_val * 10))
        self.widgets['NAG_tau_label'].setText(f"{nag_tau_val:.1f}")
        
        nag_alpha_val = ui_defaults.get('NAG_alpha', 0.5)
        self.widgets['NAG_alpha'].setValue(int(nag_alpha_val * 10))
        self.widgets['NAG_alpha_label'].setText(f"{nag_alpha_val:.1f}")

        self.widgets['nag_group'].setVisible(vace or t2v or i2v)

        self.widgets['sample_solver'].clear()
        sampler_choices = model_def.get("sample_solvers", [])
        self.widgets['solver_row_container'].setVisible(bool(sampler_choices))
        if sampler_choices:
            for label, value in sampler_choices: self.widgets['sample_solver'].addItem(label, value)
            solver_val = ui_defaults.get('sample_solver', sampler_choices[0][1])
            index = self.widgets['sample_solver'].findData(solver_val)
            if index != -1: self.widgets['sample_solver'].setCurrentIndex(index)

        flow_val = ui_defaults.get("flow_shift", 3.0)
        self.widgets['flow_shift'].setValue(int(flow_val * 10))
        self.widgets['flow_shift_label'].setText(f"{flow_val:.1f}")

        audio_guidance_val = ui_defaults.get("audio_guidance_scale", 4.0)
        self.widgets['audio_guidance_scale'].setValue(int(audio_guidance_val * 10))
        self.widgets['audio_guidance_scale_label'].setText(f"{audio_guidance_val:.1f}")
        
        repeat_val = ui_defaults.get("repeat_generation", 1)
        self.widgets['repeat_generation'].setValue(repeat_val)
        self.widgets['repeat_generation_label'].setText(str(repeat_val))

        available_loras, _, _, _, _, _ = wgp.setup_loras(model_type, None, wgp.get_lora_dir(model_type), "")
        self.state['loras'] = available_loras
        self.lora_map = {os.path.basename(p): p for p in available_loras}
        lora_list_widget = self.widgets['activated_loras']
        lora_list_widget.clear()
        lora_list_widget.addItems(sorted(self.lora_map.keys()))
        selected_loras = ui_defaults.get('activated_loras', [])
        for i in range(lora_list_widget.count()):
            item = lora_list_widget.item(i)
            is_selected = any(item.text() == os.path.basename(p) for p in selected_loras)
            if is_selected:
                item.setSelected(True)
        self.widgets['loras_multipliers'].setText(ui_defaults.get('loras_multipliers', ''))

        skip_cache_val = ui_defaults.get('skip_steps_cache_type', "")
        index = self.widgets['skip_steps_cache_type'].findData(skip_cache_val)
        if index != -1: self.widgets['skip_steps_cache_type'].setCurrentIndex(index)

        skip_mult = ui_defaults.get('skip_steps_multiplier', 1.5)
        index = self.widgets['skip_steps_multiplier'].findData(skip_mult)
        if index != -1: self.widgets['skip_steps_multiplier'].setCurrentIndex(index)

        skip_perc_val = ui_defaults.get('skip_steps_start_step_perc', 0)
        self.widgets['skip_steps_start_step_perc'].setValue(skip_perc_val)
        self.widgets['skip_steps_start_step_perc_label'].setText(str(skip_perc_val))

        temp_up_val = ui_defaults.get('temporal_upsampling', "")
        index = self.widgets['temporal_upsampling'].findData(temp_up_val)
        if index != -1: self.widgets['temporal_upsampling'].setCurrentIndex(index)

        spat_up_val = ui_defaults.get('spatial_upsampling', "")
        index = self.widgets['spatial_upsampling'].findData(spat_up_val)
        if index != -1: self.widgets['spatial_upsampling'].setCurrentIndex(index)

        film_grain_i = ui_defaults.get('film_grain_intensity', 0)
        self.widgets['film_grain_intensity'].setValue(int(film_grain_i * 100))
        self.widgets['film_grain_intensity_label'].setText(f"{film_grain_i:.2f}")

        film_grain_s = ui_defaults.get('film_grain_saturation', 0.5)
        self.widgets['film_grain_saturation'].setValue(int(film_grain_s * 100))
        self.widgets['film_grain_saturation_label'].setText(f"{film_grain_s:.2f}")

        self.widgets['MMAudio_setting'].setCurrentIndex(ui_defaults.get('MMAudio_setting', 0))
        self.widgets['MMAudio_prompt'].setText(ui_defaults.get('MMAudio_prompt', ''))
        self.widgets['MMAudio_neg_prompt'].setText(ui_defaults.get('MMAudio_neg_prompt', ''))

        self.widgets['slg_switch'].setCurrentIndex(ui_defaults.get('slg_switch', 0))
        slg_start_val = ui_defaults.get('slg_start_perc', 10)
        self.widgets['slg_start_perc'].setValue(slg_start_val)
        self.widgets['slg_start_perc_label'].setText(str(slg_start_val))
        slg_end_val = ui_defaults.get('slg_end_perc', 90)
        self.widgets['slg_end_perc'].setValue(slg_end_val)
        self.widgets['slg_end_perc_label'].setText(str(slg_end_val))

        self.widgets['apg_switch'].setCurrentIndex(ui_defaults.get('apg_switch', 0))
        self.widgets['cfg_star_switch'].setCurrentIndex(ui_defaults.get('cfg_star_switch', 0))

        cfg_zero_val = ui_defaults.get('cfg_zero_step', -1)
        self.widgets['cfg_zero_step'].setValue(cfg_zero_val)
        self.widgets['cfg_zero_step_label'].setText(str(cfg_zero_val))

        min_frames_val = ui_defaults.get('min_frames_if_references', 1)
        index = self.widgets['min_frames_if_references'].findData(min_frames_val)
        if index != -1: self.widgets['min_frames_if_references'].setCurrentIndex(index)

        self.widgets['RIFLEx_setting'].setCurrentIndex(ui_defaults.get('RIFLEx_setting', 0))

        fps = wgp.get_model_fps(model_type)
        force_fps_choices = [
            (f"Model Default ({fps} fps)", ""), ("Auto", "auto"), ("Control Video fps", "control"),
            ("Source Video fps", "source"), ("15", "15"), ("16", "16"), ("23", "23"),
            ("24", "24"), ("25", "25"), ("30", "30")
        ]
        self.widgets['force_fps'].clear()
        for label, value in force_fps_choices: self.widgets['force_fps'].addItem(label, value)
        force_fps_val = ui_defaults.get('force_fps', "")
        index = self.widgets['force_fps'].findData(force_fps_val)
        if index != -1: self.widgets['force_fps'].setCurrentIndex(index)

        override_prof_val = ui_defaults.get('override_profile', -1)
        index = self.widgets['override_profile'].findData(override_prof_val)
        if index != -1: self.widgets['override_profile'].setCurrentIndex(index)
        
        self.widgets['multi_prompts_gen_type'].setCurrentIndex(ui_defaults.get('multi_prompts_gen_type', 0))

        denoising_val = ui_defaults.get("denoising_strength", 0.5)
        self.widgets['denoising_strength'].setValue(int(denoising_val * 100))
        self.widgets['denoising_strength_label'].setText(f"{denoising_val:.2f}")

        sw_size = ui_defaults.get("sliding_window_size", 129)
        self.widgets['sliding_window_size'].setValue(sw_size)
        self.widgets['sliding_window_size_label'].setText(str(sw_size))

        sw_overlap = ui_defaults.get("sliding_window_overlap", 5)
        self.widgets['sliding_window_overlap'].setValue(sw_overlap)
        self.widgets['sliding_window_overlap_label'].setText(str(sw_overlap))

        sw_color = ui_defaults.get("sliding_window_color_correction_strength", 0)
        self.widgets['sliding_window_color_correction_strength'].setValue(int(sw_color * 100))
        self.widgets['sliding_window_color_correction_strength_label'].setText(f"{sw_color:.2f}")

        sw_noise = ui_defaults.get("sliding_window_overlap_noise", 20)
        self.widgets['sliding_window_overlap_noise'].setValue(sw_noise)
        self.widgets['sliding_window_overlap_noise_label'].setText(str(sw_noise))
        
        sw_discard = ui_defaults.get("sliding_window_discard_last_frames", 0)
        self.widgets['sliding_window_discard_last_frames'].setValue(sw_discard)
        self.widgets['sliding_window_discard_last_frames_label'].setText(str(sw_discard))

        for widget in self.widgets.values():
            if hasattr(widget, 'blockSignals'): widget.blockSignals(False)
        self._update_dynamic_ui()
        self._update_input_visibility()

    def _update_dynamic_ui(self):
        """Update UI visibility based on current selections."""
        phases = self.widgets['guidance_phases'].currentData() or 1
        guidance_layout = self.widgets['guidance_layout']
        guidance_layout.setRowVisible(self.widgets['guidance2_row_index'], phases >= 2)
        guidance_layout.setRowVisible(self.widgets['guidance3_row_index'], phases >= 3)
        guidance_layout.setRowVisible(self.widgets['switch_thresh_row_index'], phases >= 2)

    def _update_generation_mode_visibility(self, model_def):
        """Shows/hides the main generation mode options based on the selected model."""
        allowed = model_def.get("image_prompt_types_allowed", "")
        
        choices = []
        if "T" in allowed or not allowed:
            choices.append(("Text Prompt Only" if "S" in allowed else "New Video", "T"))
        if "S" in allowed:
            choices.append(("Start Video with Image", "S"))
        if "V" in allowed:
            choices.append(("Continue Video", "V"))
        if "L" in allowed:
            choices.append(("Continue Last Video", "L"))

        button_map = {
            "T": self.widgets['mode_t'],
            "S": self.widgets['mode_s'],
            "V": self.widgets['mode_v'],
            "L": self.widgets['mode_l'],
        }

        for btn in button_map.values():
            btn.setVisible(False)
        
        allowed_values = [c[1] for c in choices]
        for label, value in choices:
            if value in button_map:
                btn = button_map[value]
                btn.setText(label)
                btn.setVisible(True)

        current_checked_value = None
        for value, btn in button_map.items():
            if btn.isChecked():
                current_checked_value = value
                break
        
        # If the currently selected mode is now hidden, reset to a visible default
        if current_checked_value is None or not button_map[current_checked_value].isVisible():
            if allowed_values:
                button_map[allowed_values[0]].setChecked(True)


        end_image_visible = "E" in allowed
        self.widgets['image_end_checkbox'].setVisible(end_image_visible)
        if not end_image_visible:
            self.widgets['image_end_checkbox'].setChecked(False)

        # Control Video Checkbox (Based on model_def.get("guide_preprocessing"))
        control_video_visible = model_def.get("guide_preprocessing") is not None
        self.widgets['control_video_checkbox'].setVisible(control_video_visible)
        if not control_video_visible:
            self.widgets['control_video_checkbox'].setChecked(False)

        # Reference Image Checkbox (Based on model_def.get("image_ref_choices"))
        ref_image_visible = model_def.get("image_ref_choices") is not None
        self.widgets['ref_image_checkbox'].setVisible(ref_image_visible)
        if not ref_image_visible:
            self.widgets['ref_image_checkbox'].setChecked(False)


    def _update_input_visibility(self):
        """Shows/hides input fields based on the selected generation mode."""
        is_s_mode = self.widgets['mode_s'].isChecked()
        is_v_mode = self.widgets['mode_v'].isChecked()
        is_l_mode = self.widgets['mode_l'].isChecked()

        use_end = self.widgets['image_end_checkbox'].isChecked() and self.widgets['image_end_checkbox'].isVisible()
        use_control = self.widgets['control_video_checkbox'].isChecked() and self.widgets['control_video_checkbox'].isVisible()
        use_ref = self.widgets['ref_image_checkbox'].isChecked() and self.widgets['ref_image_checkbox'].isVisible()
        
        self.widgets['image_start_container'].setVisible(is_s_mode)
        self.widgets['video_source_container'].setVisible(is_v_mode)
        
        end_checkbox_enabled = is_s_mode or is_v_mode or is_l_mode
        self.widgets['image_end_checkbox'].setEnabled(end_checkbox_enabled)
        self.widgets['image_end_container'].setVisible(use_end and end_checkbox_enabled)
        
        self.widgets['video_guide_container'].setVisible(use_control)
        self.widgets['video_mask_container'].setVisible(use_control)
        self.widgets['image_refs_container'].setVisible(use_ref)

    def connect_signals(self):
        self.widgets['model_family'].currentIndexChanged.connect(self._on_family_changed)
        self.widgets['model_base_type_choice'].currentIndexChanged.connect(self._on_base_type_changed)
        self.widgets['model_choice'].currentIndexChanged.connect(self._on_model_changed)
        self.widgets['resolution_group'].currentIndexChanged.connect(self._on_resolution_group_changed)
        self.widgets['guidance_phases'].currentIndexChanged.connect(self._update_dynamic_ui)

        self.widgets['mode_t'].toggled.connect(self._update_input_visibility)
        self.widgets['mode_s'].toggled.connect(self._update_input_visibility)
        self.widgets['mode_v'].toggled.connect(self._update_input_visibility)
        self.widgets['mode_l'].toggled.connect(self._update_input_visibility)
        
        self.widgets['image_end_checkbox'].toggled.connect(self._update_input_visibility)
        self.widgets['control_video_checkbox'].toggled.connect(self._update_input_visibility)
        self.widgets['ref_image_checkbox'].toggled.connect(self._update_input_visibility)
        self.widgets['preview_group'].toggled.connect(self._on_preview_toggled)

        self.generate_btn.clicked.connect(self._on_generate)
        self.add_to_queue_btn.clicked.connect(self._on_add_to_queue)
        self.remove_queue_btn.clicked.connect(self._on_remove_selected_from_queue)
        self.clear_queue_btn.clicked.connect(self._on_clear_queue)
        self.abort_btn.clicked.connect(self._on_abort)
        self.queue_table.rowsMoved.connect(self._on_queue_rows_moved)

    def apply_initial_config(self):
        is_visible = main_config.get('preview_visible', True)
        self.widgets['preview_group'].setChecked(is_visible)
        self.widgets['preview_image'].setVisible(is_visible)

    def _on_preview_toggled(self, checked):
        self.widgets['preview_image'].setVisible(checked)
        main_config['preview_visible'] = checked
        save_main_config()

    def _on_family_changed(self, index):
        family = self.widgets['model_family'].currentData()
        if not family or not self.state: return
        
        base_type_mock, choice_mock = wgp.change_model_family(self.state, family)

        self.widgets['model_base_type_choice'].blockSignals(True)
        self.widgets['model_base_type_choice'].clear()
        if base_type_mock.choices:
            for label, value in base_type_mock.choices:
                self.widgets['model_base_type_choice'].addItem(label, value)
        index = self.widgets['model_base_type_choice'].findData(base_type_mock.value)
        if index != -1:
            self.widgets['model_base_type_choice'].setCurrentIndex(index)
        self.widgets['model_base_type_choice'].setVisible(base_type_mock.kwargs.get('visible', True))
        self.widgets['model_base_type_choice'].blockSignals(False)
        
        self.widgets['model_choice'].blockSignals(True)
        self.widgets['model_choice'].clear()
        if choice_mock.choices:
            for label, value in choice_mock.choices:
                self.widgets['model_choice'].addItem(label, value)
        index = self.widgets['model_choice'].findData(choice_mock.value)
        if index != -1:
            self.widgets['model_choice'].setCurrentIndex(index)
        self.widgets['model_choice'].setVisible(choice_mock.kwargs.get('visible', True))
        self.widgets['model_choice'].blockSignals(False)

        self._on_model_changed()

    def _on_base_type_changed(self, index):
        family = self.widgets['model_family'].currentData()
        base_type = self.widgets['model_base_type_choice'].currentData()
        if not family or not base_type or not self.state: return
        
        base_type_mock, choice_mock = wgp.change_model_base_types(self.state, family, base_type)
        
        self.widgets['model_choice'].blockSignals(True)
        self.widgets['model_choice'].clear()
        if choice_mock.choices:
            for label, value in choice_mock.choices:
                self.widgets['model_choice'].addItem(label, value)
        index = self.widgets['model_choice'].findData(choice_mock.value)
        if index != -1:
            self.widgets['model_choice'].setCurrentIndex(index)
        self.widgets['model_choice'].setVisible(choice_mock.kwargs.get('visible', True))
        self.widgets['model_choice'].blockSignals(False)
        
        self._on_model_changed()

    def _on_model_changed(self):
        model_type = self.widgets['model_choice'].currentData()
        if not model_type or model_type == self.state['model_type']: return
        wgp.change_model(self.state, model_type)
        self.refresh_ui_from_model_change(model_type)

    def _on_resolution_group_changed(self):
        selected_group = self.widgets['resolution_group'].currentText()
        if not selected_group or not hasattr(self, 'full_resolution_choices'):
            return

        model_type = self.state['model_type']
        model_def = wgp.get_model_def(model_type)
        model_resolutions = model_def.get("resolutions", None)

        group_resolution_choices = []
        if model_resolutions is None:
            group_resolution_choices = [res for res in self.full_resolution_choices if wgp.categorize_resolution(res[1]) == selected_group]
        else:
            return

        last_resolution_per_group = self.state.get("last_resolution_per_group", {})
        last_resolution = last_resolution_per_group.get(selected_group, "")

        is_last_res_valid = any(last_resolution == res[1] for res in group_resolution_choices)
        if not is_last_res_valid and group_resolution_choices:
            last_resolution = group_resolution_choices[0][1]

        self.widgets['resolution'].blockSignals(True)
        self.widgets['resolution'].clear()
        for label, value in group_resolution_choices:
            self.widgets['resolution'].addItem(label, value)
        
        index = self.widgets['resolution'].findData(last_resolution)
        if index != -1:
            self.widgets['resolution'].setCurrentIndex(index)
        self.widgets['resolution'].blockSignals(False)

    def collect_inputs(self):
        """Gather all settings from UI widgets into a dictionary."""
        # Start with all possible defaults. This dictionary will be modified and returned.
        full_inputs = wgp.get_current_model_settings(self.state).copy()

        # Add dummy/default values for UI elements present in Gradio but not yet in PyQt.
        # These are expected by the backend logic.
        full_inputs['lset_name'] = ""
        # The PyQt UI is focused on generating videos, so image_mode is 0.
        full_inputs['image_mode'] = 0

        # Defensively initialize keys that are accessed directly in wgp.py but may not
        # be in the saved model settings or fully implemented in the UI yet.
        # This prevents both KeyErrors and TypeErrors for missing arguments.
        expected_keys = {
            "audio_guide": None, "audio_guide2": None, "image_guide": None,
            "image_mask": None, "speakers_locations": "", "frames_positions": "",
            "keep_frames_video_guide": "", "keep_frames_video_source": "",
            "video_guide_outpainting": "", "switch_threshold2": 0,
            "model_switch_phase": 1, "batch_size": 1,
            "control_net_weight_alt": 1.0,
            "image_refs_relative_size": 50,
        }
        for key, default_value in expected_keys.items():
            if key not in full_inputs:
                full_inputs[key] = default_value

        # Overwrite defaults with values from the PyQt UI widgets
        full_inputs['prompt'] = self.widgets['prompt'].toPlainText()
        full_inputs['negative_prompt'] = self.widgets['negative_prompt'].toPlainText()
        full_inputs['resolution'] = self.widgets['resolution'].currentData()
        full_inputs['video_length'] = self.widgets['video_length'].value()
        full_inputs['num_inference_steps'] = self.widgets['num_inference_steps'].value()
        full_inputs['seed'] = int(self.widgets['seed'].text())

        # Build prompt_type strings based on mode selections
        image_prompt_type = ""
        video_prompt_type = ""

        if self.widgets['mode_s'].isChecked():
            image_prompt_type = 'S'
        elif self.widgets['mode_v'].isChecked():
            image_prompt_type = 'V'
        elif self.widgets['mode_l'].isChecked():
            image_prompt_type = 'L'
        else: # mode_t is checked
            image_prompt_type = ''

        if self.widgets['image_end_checkbox'].isVisible() and self.widgets['image_end_checkbox'].isChecked():
            image_prompt_type += 'E'

        if self.widgets['control_video_checkbox'].isVisible() and self.widgets['control_video_checkbox'].isChecked():
            video_prompt_type += 'V' # This 'V' is for Control Video (V2V)
        if self.widgets['ref_image_checkbox'].isVisible() and self.widgets['ref_image_checkbox'].isChecked():
            video_prompt_type += 'I' # 'I' for Reference Image

        full_inputs['image_prompt_type'] = image_prompt_type
        full_inputs['video_prompt_type'] = video_prompt_type

        # File Inputs
        for name in ['video_source', 'image_start', 'image_end', 'video_guide', 'video_mask', 'audio_source']:
            if name in self.widgets:
                path = self.widgets[name].text()
                full_inputs[name] = path if path else None

        paths = self.widgets['image_refs'].text().split(';')
        full_inputs['image_refs'] = [p.strip() for p in paths if p.strip()] if paths and paths[0] else None

        full_inputs['denoising_strength'] = self.widgets['denoising_strength'].value() / 100.0

        if self.advanced_group.isChecked():
            full_inputs['guidance_scale'] = self.widgets['guidance_scale'].value() / 10.0
            full_inputs['guidance_phases'] = self.widgets['guidance_phases'].currentData()
            full_inputs['guidance2_scale'] = self.widgets['guidance2_scale'].value() / 10.0
            full_inputs['guidance3_scale'] = self.widgets['guidance3_scale'].value() / 10.0
            full_inputs['switch_threshold'] = self.widgets['switch_threshold'].value()

            full_inputs['NAG_scale'] = self.widgets['NAG_scale'].value() / 10.0
            full_inputs['NAG_tau'] = self.widgets['NAG_tau'].value() / 10.0
            full_inputs['NAG_alpha'] = self.widgets['NAG_alpha'].value() / 10.0

            full_inputs['sample_solver'] = self.widgets['sample_solver'].currentData()
            full_inputs['flow_shift'] = self.widgets['flow_shift'].value() / 10.0
            full_inputs['audio_guidance_scale'] = self.widgets['audio_guidance_scale'].value() / 10.0
            full_inputs['repeat_generation'] = self.widgets['repeat_generation'].value()
            full_inputs['multi_images_gen_type'] = self.widgets['multi_images_gen_type'].currentData()

            lora_list_widget = self.widgets['activated_loras']
            selected_items = lora_list_widget.selectedItems()
            full_inputs['activated_loras'] = [self.lora_map[item.text()] for item in selected_items if item.text() in self.lora_map]
            full_inputs['loras_multipliers'] = self.widgets['loras_multipliers'].toPlainText()

            full_inputs['skip_steps_cache_type'] = self.widgets['skip_steps_cache_type'].currentData()
            full_inputs['skip_steps_multiplier'] = self.widgets['skip_steps_multiplier'].currentData()
            full_inputs['skip_steps_start_step_perc'] = self.widgets['skip_steps_start_step_perc'].value()

            full_inputs['temporal_upsampling'] = self.widgets['temporal_upsampling'].currentData()
            full_inputs['spatial_upsampling'] = self.widgets['spatial_upsampling'].currentData()
            full_inputs['film_grain_intensity'] = self.widgets['film_grain_intensity'].value() / 100.0
            full_inputs['film_grain_saturation'] = self.widgets['film_grain_saturation'].value() / 100.0

            full_inputs['MMAudio_setting'] = self.widgets['MMAudio_setting'].currentData()
            full_inputs['MMAudio_prompt'] = self.widgets['MMAudio_prompt'].text()
            full_inputs['MMAudio_neg_prompt'] = self.widgets['MMAudio_neg_prompt'].text()

            full_inputs['RIFLEx_setting'] = self.widgets['RIFLEx_setting'].currentData()
            full_inputs['force_fps'] = self.widgets['force_fps'].currentData()
            full_inputs['override_profile'] = self.widgets['override_profile'].currentData()
            full_inputs['multi_prompts_gen_type'] = self.widgets['multi_prompts_gen_type'].currentData()

            full_inputs['slg_switch'] = self.widgets['slg_switch'].currentData()
            full_inputs['slg_start_perc'] = self.widgets['slg_start_perc'].value()
            full_inputs['slg_end_perc'] = self.widgets['slg_end_perc'].value()
            full_inputs['apg_switch'] = self.widgets['apg_switch'].currentData()
            full_inputs['cfg_star_switch'] = self.widgets['cfg_star_switch'].currentData()
            full_inputs['cfg_zero_step'] = self.widgets['cfg_zero_step'].value()
            full_inputs['min_frames_if_references'] = self.widgets['min_frames_if_references'].currentData()

            full_inputs['sliding_window_size'] = self.widgets['sliding_window_size'].value()
            full_inputs['sliding_window_overlap'] = self.widgets['sliding_window_overlap'].value()
            full_inputs['sliding_window_color_correction_strength'] = self.widgets['sliding_window_color_correction_strength'].value() / 100.0
            full_inputs['sliding_window_overlap_noise'] = self.widgets['sliding_window_overlap_noise'].value()
            full_inputs['sliding_window_discard_last_frames'] = self.widgets['sliding_window_discard_last_frames'].value()

        return full_inputs

    def _prepare_state_for_generation(self):
        if 'gen' in self.state:
            self.state['gen'].pop('abort', None)
            self.state['gen'].pop('in_progress', None)

    def _on_generate(self):
        try:
            is_running = self.thread and self.thread.isRunning()
            self._add_task_to_queue_and_update_ui()
            if not is_running:
                self.start_generation()
        except Exception:
            import traceback
            traceback.print_exc()


    def _on_add_to_queue(self):
        try:
            self._add_task_to_queue_and_update_ui()
        except Exception:
            import traceback
            traceback.print_exc()

    def _add_task_to_queue_and_update_ui(self):
        self._add_task_to_queue()
        self.update_queue_table()

    def _add_task_to_queue(self):
        queue_size_before = len(self.state["gen"]["queue"])
        all_inputs = self.collect_inputs()
        keys_to_remove = ['type', 'settings_version', 'is_image', 'video_quality', 'image_quality']
        for key in keys_to_remove:
            all_inputs.pop(key, None)

        all_inputs['state'] = self.state
        wgp.set_model_settings(self.state, self.state['model_type'], all_inputs)
        
        self.state["validate_success"] = 1
        wgp.process_prompt_and_add_tasks(self.state, self.state['model_type'])

        
    def start_generation(self):
        if not self.state['gen']['queue']:
            return
        self._prepare_state_for_generation()
        self.generate_btn.setEnabled(False)
        self.add_to_queue_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = Worker(self.state)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_generation_finished)

        self.worker.status.connect(self.status_label.setText)
        self.worker.progress.connect(self.update_progress)
        self.worker.preview.connect(self.update_preview)
        self.worker.output.connect(self.update_queue_and_gallery)
        self.worker.error.connect(self.on_generation_error)

        self.thread.start()
        self.update_queue_table()

    def on_generation_finished(self):
        time.sleep(0.1)
        self.status_label.setText("Finished.")
        self.progress_bar.setValue(0)
        self.generate_btn.setEnabled(True)
        self.add_to_queue_btn.setEnabled(False)
        self.thread = None
        self.worker = None
        self.update_queue_table()

    def on_generation_error(self, err_msg):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText("Generation Error")
        msg_box.setInformativeText(str(err_msg))
        msg_box.setWindowTitle("Error")
        msg_box.exec()
        self.on_generation_finished()

    def update_progress(self, data):
        if len(data) > 1 and isinstance(data[0], tuple):
            step, total = data[0]
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(step)
            self.status_label.setText(str(data[1]))
            if step <= 1:
                self.update_queue_table()
        elif len(data) > 1:
             self.status_label.setText(str(data[1]))

    def update_preview(self, pil_image):
        if pil_image:
            q_image = ImageQt(pil_image)
            pixmap = QPixmap.fromImage(q_image)
            self.preview_image.setPixmap(pixmap.scaled(
                self.preview_image.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))

    def update_queue_and_gallery(self):
        self.update_queue_table()
        
        file_list = self.state.get('gen', {}).get('file_list', [])
        self.output_gallery.clear()
        self.output_gallery.addItems(file_list)
        if file_list:
            self.output_gallery.setCurrentRow(len(file_list) - 1)
            self.latest_output_path = file_list[-1] # <-- ADDED for API access

    def update_queue_table(self):
        with wgp.lock:
            queue = self.state.get('gen', {}).get('queue', [])
            is_running = self.thread and self.thread.isRunning()
            queue_to_display = queue if is_running else [None] + queue
            
            table_data = wgp.get_queue_table(queue_to_display)

            self.queue_table.setRowCount(0)
            self.queue_table.setRowCount(len(table_data))
            self.queue_table.setColumnCount(4) 
            self.queue_table.setHorizontalHeaderLabels(["Qty", "Prompt", "Length", "Steps"])
            
            for row_idx, row_data in enumerate(table_data):
                prompt_html = row_data[1]
                try:
                    prompt_text = prompt_html.split('>')[1].split('<')[0]
                except IndexError:
                    prompt_text = str(row_data[1])

                self.queue_table.setItem(row_idx, 0, QTableWidgetItem(str(row_data[0])))
                self.queue_table.setItem(row_idx, 1, QTableWidgetItem(prompt_text))
                self.queue_table.setItem(row_idx, 2, QTableWidgetItem(str(row_data[2])))
                self.queue_table.setItem(row_idx, 3, QTableWidgetItem(str(row_data[3])))
            
            self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self.queue_table.resizeColumnsToContents()

    def _on_remove_selected_from_queue(self):
        selected_row = self.queue_table.currentRow()
        if selected_row < 0:
            return

        with wgp.lock:
            is_running = self.thread and self.thread.isRunning()
            offset = 1 if is_running else 0
            
            queue = self.state.get('gen', {}).get('queue', [])
            if len(queue) > selected_row + offset:
                queue.pop(selected_row + offset)
        self.update_queue_table()

    def _on_queue_rows_moved(self, source_row, dest_row):
        with wgp.lock:
            queue = self.state.get('gen', {}).get('queue', [])
            is_running = self.thread and self.thread.isRunning()
            offset = 1 if is_running else 0

            real_source_idx = source_row + offset
            real_dest_idx = dest_row + offset

            moved_item = queue.pop(real_source_idx)
            queue.insert(real_dest_idx, moved_item)
        self.update_queue_table()

    def _on_clear_queue(self):
        wgp.clear_queue_action(self.state)
        self.update_queue_table()
        
    def _on_abort(self):
        if self.worker:
            wgp.abort_generation(self.state)
            self.status_label.setText("Aborting...")
            self.worker._is_running = False
    
    def _on_release_ram(self):
        wgp.release_RAM()
        QMessageBox.information(self, "RAM Released", "Models stored in RAM have been released.")

    def _on_apply_config_changes(self):
        changes = {}
        
        list_widget = self.widgets['config_transformer_types']
        checked_items = [item.data(Qt.ItemDataRole.UserRole) for i in range(list_widget.count()) if list_widget.item(i).checkState() == Qt.CheckState.Checked]
        changes['transformer_types_choices'] = checked_items

        list_widget = self.widgets['config_preload_model_policy']
        checked_items = [item.data(Qt.ItemDataRole.UserRole) for i in range(list_widget.count()) if list_widget.item(i).checkState() == Qt.CheckState.Checked]
        changes['preload_model_policy_choice'] = checked_items

        changes['model_hierarchy_type_choice'] = self.widgets['config_model_hierarchy_type'].currentData()
        changes['checkpoints_paths'] = self.widgets['config_checkpoints_paths'].toPlainText()

        for key in ["fit_canvas", "attention_mode", "metadata_type", "clear_file_list", "display_stats", "max_frames_multiplier", "UI_theme"]:
            changes[f'{key}_choice'] = self.widgets[f'config_{key}'].currentData()

        for key in ["transformer_quantization", "transformer_dtype_policy", "mixed_precision", "text_encoder_quantization", "vae_precision", "compile", "depth_anything_v2_variant", "vae_config", "boost", "profile"]:
             changes[f'{key}_choice'] = self.widgets[f'config_{key}'].currentData()
        changes['preload_in_VRAM_choice'] = self.widgets['config_preload_in_VRAM'].value()
        
        for key in ["enhancer_enabled", "enhancer_mode", "mmaudio_enabled"]:
             changes[f'{key}_choice'] = self.widgets[f'config_{key}'].currentData()

        for key in ["video_output_codec", "image_output_codec", "save_path", "image_save_path"]:
             widget = self.widgets[f'config_{key}']
             changes[f'{key}_choice'] = widget.currentData() if isinstance(widget, QComboBox) else widget.text()

        changes['notification_sound_enabled_choice'] = self.widgets['config_notification_sound_enabled'].currentData()
        changes['notification_sound_volume_choice'] = self.widgets['config_notification_sound_volume'].value()

        changes['last_resolution_choice'] = self.widgets['resolution'].currentData()

        try:
            msg, header_mock, family_mock, base_type_mock, choice_mock, refresh_trigger = wgp.apply_changes(self.state, **changes)
            self.config_status_label.setText("Changes applied successfully. Some settings may require a restart.")
            
            self.header_info.setText(wgp.generate_header(self.state['model_type'], wgp.compile, wgp.attention_mode))
            
            if family_mock.choices is not None or choice_mock.choices is not None:
                self.update_model_dropdowns(wgp.transformer_type)
                self.refresh_ui_from_model_change(wgp.transformer_type)

        except Exception as e:
            self.config_status_label.setText(f"Error applying changes: {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        if wgp:
            wgp.autosave_queue()
        save_main_config()
        event.accept()

# =====================================================================
# --- START OF API SERVER ADDITION ---
# =====================================================================
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not installed. API server will not be available. Please run: pip install Flask")

# Global reference to the main window, to be populated after instantiation.
main_window_instance = None
api_server = Flask(__name__) if FLASK_AVAILABLE else None

def run_api_server():
    """Function to run the Flask server."""
    if api_server:
        print("Starting API server on http://127.0.0.1:5100")
        api_server.run(port=5100, host='127.0.0.1', debug=False)

if FLASK_AVAILABLE:
    # --- CHANGE: Removed /api/set_model endpoint ---

    @api_server.route('/api/generate', methods=['POST'])
    def generate():
        if not main_window_instance:
            return jsonify({"error": "Application not ready"}), 503

        data = request.json
        start_frame = data.get('start_frame')
        end_frame = data.get('end_frame')
        # --- CHANGE: Get duration_sec and model_type ---
        duration_sec = data.get('duration_sec')
        model_type = data.get('model_type')
        start_generation = data.get('start_generation', False)

        # --- CHANGE: Emit the signal with the new parameters ---
        main_window_instance.api_bridge.generateSignal.emit(
            start_frame, end_frame, duration_sec, model_type, start_generation
        )

        if start_generation:
            return jsonify({"message": "Parameters set and generation request sent."})
        else:
            return jsonify({"message": "Parameters set without starting generation."})


    @api_server.route('/api/latest_output', methods=['GET'])
    def get_latest_output():
        if not main_window_instance:
            return jsonify({"error": "Application not ready"}), 503
        
        path = main_window_instance.latest_output_path
        return jsonify({"latest_output_path": path})

# =====================================================================
# --- END OF API SERVER ADDITION ---
# =====================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MainWindow()
    main_window_instance = window  # Assign to global for API access
    window.show()

    # Start the Flask API server in a separate thread
    if FLASK_AVAILABLE:
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()

    sys.exit(app.exec())