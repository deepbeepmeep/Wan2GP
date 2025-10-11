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
<img width="1991" height="1050" alt="image" src="https://github.com/user-attachments/assets/d6613d40-af1a-4cf0-8180-1d2dcaa4fb8e" />
<img width="1982" height="1039" alt="image" src="https://github.com/user-attachments/assets/b15d1fe7-cf19-452e-8012-dac3d86b8cca" />
<img width="3729" height="1620" alt="image" src="https://github.com/user-attachments/assets/681d4758-4e79-4418-8a69-1649c07cf463" />
<img width="1502" height="1039" alt="image" src="https://github.com/user-attachments/assets/01e6fe75-126c-4474-a485-34995691bde0" />

## Credits
The AI Video Generator plugin is built from a desktop port of WAN2GP by DeepBeepMeep.
See WAN2GP for more details.
https://github.com/deepbeepmeep/Wan2GP