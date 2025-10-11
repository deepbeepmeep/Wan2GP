# Inline AI Video Editor

A simple, non-linear video editor built with Python, PyQt6, and FFmpeg. It provides a multi-track timeline, a media preview window, and basic clip manipulation capabilities, all wrapped in a dockable user interface. The editor is designed to be extensible through a plugin system.

## Features

-   **Inline AI Video Generation With WAN2GP**: Select a region to join and it will bring up a desktop port of WAN2GP for you to generate a video inline using the start and end frames in the selected region. You can also create frames in the selected region.
-   **Multi-Track Timeline**: Arrange video and audio clips on separate tracks.
-   **Project Management**: Create, save, and load projects in a `.json` format.
-   **Clip Operations**:
    -   Drag and drop clips to reposition them in the timeline.
    -   Split clips at the playhead.
    -   Create selection regions for advanced operations.
    -   Join/remove content within selected regions across all tracks.
	-   Link/Unlink audio tracks from video
-   **Real-time Preview**: A video preview window with playback controls (Play, Pause, Stop, Frame-by-frame stepping).
-   **Dynamic Track Management**: Add or remove video and audio tracks as needed.
-   **FFmpeg Integration**:
    -   Handles video processing for frame extraction, playback, and exporting.
    -   **Automatic FFmpeg Downloader (Windows)**: Automatically downloads the necessary FFmpeg executables on first run if they are not found.
-   **Extensible Plugin System**: Load custom plugins to add new features and dockable widgets.
-   **Customizable UI**: Features a dockable interface with resizable panels for the video preview and timeline.
-   **More coming soon..

## Installation

```bash
git clone https://github.com/Tophness/Wan2GP.git
cd Wan2GP
git checkout video_editor
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

## Usage

**Run the video editor:**
```bash
python videoeditor.py
```

## Screenshots
<img width="2341" height="1553" alt="image" src="https://github.com/user-attachments/assets/923a84db-5518-4a71-a852-977f1b3c31d5" />
<img width="2341" height="1553" alt="image" src="https://github.com/user-attachments/assets/863337f6-f03e-4ce9-93e1-40abd8578daf" />
<img width="2341" height="1553" alt="image" src="https://github.com/user-attachments/assets/0d8ca700-f1c7-44cb-87dc-aaef69c039e9" />

## Credits
The AI Video Generator plugin is built from a desktop port of WAN2GP by DeepBeepMeep.
See WAN2GP for more details.
https://github.com/deepbeepmeep/Wan2GP