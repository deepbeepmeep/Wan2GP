import os
import sys
import time
import uuid
from urllib.request import urlretrieve

import cv2
from PIL import Image

windows_model = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
mac_model = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"

def upscale(image_path, scale_factor):
    print(f"Upscaling {image_path} with factor of x{scale_factor}...")

    if sys.platform == 'darwin':
        executable="realesrgan-ncnn-vulkan"

        urlretrieve(mac_model, "realesrgan-ncnn-vulkan-20220424-macos.zip")
        import zipfile
        with zipfile.ZipFile("realesrgan-ncnn-vulkan-20220424-macos.zip", 'r') as zip_ref:
            zip_ref.extractall("upscaler")

        # make it executable
        os.system(f"chmod a+x {executable}")

        # put in the parent folder outputs
        output_file = f"outputs/{str(uuid.uuid4())}.png"

        os.system(f"cd upscaler; ./{executable} -i {image_path} -o ../{output_file} -s {scale_factor}")

        # need to wait for the file to exist; on macOS this takes a bit
        timeout = 30  # seconds
        interval = 0.5
        waited = 0.0
        while not os.path.exists(output_file) and waited < timeout:
            time.sleep(interval)
            waited += interval

        if os.path.exists(output_file):
            print(f"Output file ready: {output_file}")
        else:
            print(f"WARNING: Output file not found after {timeout}s: {output_file}")
    else:
        executable="upscaler\\realesrgan-ncnn-vulkan.exe"

        urlretrieve(windows_model, "realesrgan-ncnn-vulkan-20220424-windows.zip")
        import zipfile
        with zipfile.ZipFile("realesrgan-ncnn-vulkan-20220424-windows.zip", 'r') as zip_ref:
            zip_ref.extractall("upscaler")

        output_file = f"outputs\\{str(uuid.uuid4())}.png"

        os.system(f"{executable} -i {image_path} -o {output_file} -s {scale_factor}")

    return output_file