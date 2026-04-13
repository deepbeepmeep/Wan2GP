"""
ComfyUI Utilities

Provides ComfyUI server management and API client for processing workflow tasks.
"""

from __future__ import annotations

import os
import time
import signal
import subprocess
from pathlib import Path
from typing import Dict
import httpx

from source.core.log import headless_logger

# Configuration
COMFY_PATH = os.getenv("COMFY_PATH", "/workspace/ComfyUI")
COMFY_HOST = "localhost"
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))


class ComfyUIManager:
    """Manages ComfyUI server process lifecycle."""

    def __init__(self, comfy_path: str = COMFY_PATH, port: int = COMFY_PORT):
        self.comfy_path = comfy_path
        self.port = port
        self.process = None

    def start(self):
        """Start ComfyUI server."""
        if self.process is not None:
            headless_logger.warning("ComfyUI already running")
            return True

        comfy_main = Path(self.comfy_path) / "main.py"
        if not comfy_main.exists():
            raise FileNotFoundError(f"ComfyUI not found at {self.comfy_path}")

        headless_logger.debug(f"Starting ComfyUI server at {self.comfy_path}")

        # On Windows, use CREATE_NEW_PROCESS_GROUP to isolate the child from
        # our console Ctrl-C group; on POSIX, use setsid for the same purpose.
        if os.name == "nt":
            _extra = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        else:
            _extra = {"preexec_fn": os.setsid}
        self.process = subprocess.Popen(
            ["python", "main.py", "--listen", "0.0.0.0", "--port", str(self.port)],
            cwd=self.comfy_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_extra,
        )

        headless_logger.debug(f"ComfyUI started with PID: {self.process.pid}")
        return True

    async def wait_for_ready(self, client: httpx.AsyncClient, timeout: int = 120) -> bool:
        """Wait for ComfyUI to be ready."""
        start_time = time.time()
        url = f"http://{COMFY_HOST}:{self.port}/system_stats"

        while time.time() - start_time < timeout:
            try:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    headless_logger.debug("ComfyUI is ready!")
                    return True
            except (httpx.HTTPError, OSError) as e:
                headless_logger.debug(f"ComfyUI not ready yet: {e}")

            time.sleep(2)

        headless_logger.error(f"ComfyUI did not become ready within {timeout}s")
        return False

    def stop(self):
        """Stop ComfyUI server."""
        if self.process:
            headless_logger.debug(f"Stopping ComfyUI (PID: {self.process.pid})")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError) as e:
                    headless_logger.error(f"Failed to stop ComfyUI: {e}")
            finally:
                self.process = None


class ComfyUIClient:
    """Client for ComfyUI API."""

    def __init__(self, host: str = COMFY_HOST, port: int = COMFY_PORT):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    async def upload_video(
        self,
        client: httpx.AsyncClient,
        video_bytes: bytes,
        filename: str
    ) -> str:
        """Upload video to ComfyUI."""
        files = {'image': (filename, video_bytes, 'video/mp4')}
        data = {'overwrite': 'true'}

        response = await client.post(
            f"{self.base_url}/upload/image",
            files=files,
            data=data,
            timeout=120.0
        )
        response.raise_for_status()

        result = response.json()
        return result.get('name', filename)

    async def queue_workflow(
        self,
        client: httpx.AsyncClient,
        workflow: Dict
    ) -> str:
        """Submit workflow to ComfyUI."""
        payload = {"prompt": workflow}

        response = await client.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()

        result = response.json()
        return result.get('prompt_id')

    async def wait_for_completion(
        self,
        client: httpx.AsyncClient,
        prompt_id: str,
        timeout: int = 600
    ) -> Dict:
        """Wait for workflow to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = await client.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=10.0
            )
            response.raise_for_status()

            history = response.json().get(prompt_id)
            if history and history.get('status', {}).get('completed'):
                return history

            time.sleep(5)

        raise TimeoutError(f"Workflow did not complete within {timeout}s")

    async def download_output(
        self,
        client: httpx.AsyncClient,
        history: Dict
    ) -> list:
        """Download output from completed workflow."""
        outputs = []

        for node_id, node_output in history.get('outputs', {}).items():
            # Check both 'videos' and 'gifs' (VHS_VideoCombine uses 'gifs')
            for output_key in ['videos', 'gifs']:
                if output_key in node_output:
                    for video in node_output[output_key]:
                        params = {
                            'filename': video['filename'],
                            'subfolder': video.get('subfolder', ''),
                            'type': video.get('type', 'output')
                        }

                        response = await client.get(
                            f"{self.base_url}/view",
                            params=params,
                            timeout=120.0
                        )
                        response.raise_for_status()

                        outputs.append({
                            'filename': video['filename'],
                            'content': response.content
                        })

        return outputs
