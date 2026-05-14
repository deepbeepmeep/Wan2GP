import os
import uuid
from urllib.request import urlretrieve

import cv2
from PIL import Image

windows_model = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"

def upscale(image_path, scale_factor):
    print(f"Upscaling ${image_path} with factor of x{scale_factor}...")
    executable="upscaler\\realesrgan-ncnn-vulkan.exe"
    urlretrieve(windows_model, "realesrgan-ncnn-vulkan-20220424-windows.zip")

    import zipfile
    with zipfile.ZipFile("realesrgan-ncnn-vulkan-20220424-windows.zip", 'r') as zip_ref:
        zip_ref.extractall("upscaler")

    output_file = f"outputs\\{str(uuid.uuid4())}.png"
    os.system(f"{executable} -i {image_path} -o {output_file} -s {scale_factor}")

    return output_file