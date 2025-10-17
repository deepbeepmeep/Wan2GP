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


#### **Windows**

```bash
git clone https://github.com/Tophness/Wan2GP.git
cd Wan2GP
git checkout video_editor
python -m venv venv
venv\Scripts\activate
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

#### **Linux / macOS**

```
git clone https://github.com/Tophness/Wan2GP.git
cd Wan2GP
git checkout video_editor
python3 -m venv venv
source venv/bin/activate
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```


## Usage

**Run the video editor:**
```bash
python videoeditor.py
```

## Screenshots
![sc1_](https://github.com/user-attachments/assets/98247de8-613d-418a-b71e-fdf2d6b547f4)
![sc2_](https://github.com/user-attachments/assets/41c3f885-2fa9-4a81-911c-68c13da1e97b)
![sc3_](https://github.com/user-attachments/assets/ec129087-03f5-43e5-a102-934efc62b001)



## Credits
The AI Video Generator plugin is built from a desktop port of WAN2GP by DeepBeepMeep.
See WAN2GP for more details.
https://github.com/deepbeepmeep/Wan2GP
