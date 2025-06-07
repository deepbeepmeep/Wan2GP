import os
import torch
from hyvideo.hunyuan import HunyuanVideoSampler
from datetime import datetime
import base64 # For potential future use
import io # For potential future use
import imageio # For saving video
import numpy as np
import random
import requests # Added for downloading files
from PIL import Image # Added for image processing
import tempfile # Added for temporary file handling

# Global variable for the sampler
SAMPLER = None

# Paths for model components (defined globally for clarity, used in load_model)
# Updated for Avatar model
MODEL_FILEPATH = "/app/ckpts/hunyuan_video_avatar_720_bf16.safetensors"
# Determine text_encoder_filepath based on quantization (empty means fp16)
TEXT_ENCODER_QUANTIZATION = "" # Default to fp16 as per Dockerfile download
if TEXT_ENCODER_QUANTIZATION == "int8":
    TEXT_ENCODER_FILEPATH = "/app/ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_int8.safetensors"
else:
    TEXT_ENCODER_FILEPATH = "/app/ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors"

VAE_FILEPATH = "/app/ckpts/hunyuan_video_VAE_fp32.safetensors"
VAE_CONFIG_FILEPATH = "/app/ckpts/hunyuan_video_VAE_config.json"
TOKENIZER_PATH = "/app/ckpts/llava-llama-3-8b/"
CLIP_TEXT_ENCODER_FILEPATH = "/app/ckpts/clip_vit_large_patch14/"
CLIP_TOKENIZER_PATH = "/app/ckpts/clip_vit_large_patch14/"
SPEECH_ENCODER_PATH = "/app/ckpts/whisper-tiny/" # Added for speech encoder (e.g., Whisper)


def load_model():
    """
    Loads the HunyuanVideoSampler model and moves it to the GPU.
    """
    global SAMPLER
    if SAMPLER is not None:
        return SAMPLER

    print("Loading HunyuanVideoSampler for Avatar/Lip-sync...")

    sampler = HunyuanVideoSampler.from_pretrained(
        model_filepath=MODEL_FILEPATH,
        text_encoder_filepath=TEXT_ENCODER_FILEPATH,
        vae_filepath=VAE_FILEPATH,
        vae_config_filepath=VAE_CONFIG_FILEPATH,
        tokenizer_path=TOKENIZER_PATH,
        clip_text_encoder_filepath=CLIP_TEXT_ENCODER_FILEPATH,
        clip_tokenizer_path=CLIP_TOKENIZER_PATH,
        speech_encoder_path=SPEECH_ENCODER_PATH, # Added for avatar model
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        text_encoder_quantization=TEXT_ENCODER_QUANTIZATION,
    )

    print("Moving sampler to CUDA device...")
    sampler.to(torch.device('cuda'))
    print("Model loaded and moved to CUDA.")
    return sampler

def save_video_frames(frames_tensor, output_path, fps):
    """
    Saves a tensor of video frames as an MP4 video file.
    """
    print(f"Saving video to {output_path} with FPS {fps}...")
    if frames_tensor.ndim == 4 and frames_tensor.shape[1] == 3: # T, C, H, W
        frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # To T, H, W, C

    if frames_tensor.min() < 0:
        frames_tensor = (frames_tensor + 1) / 2

    frames_uint8 = (frames_tensor.clamp(0, 1) * 255).byte().cpu().numpy()

    imageio.mimsave(output_path, frames_uint8, fps=fps, quality=8)
    print(f"Video saved successfully to {output_path}")

def download_file(url, suffix):
    """
    Downloads a file from a URL and saves it to a temporary file.
    Returns the path to the temporary file.
    """
    print(f"Downloading file from {url} with suffix {suffix}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with temp_file as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully to {temp_file.name}")
        return temp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        raise # Re-raise the exception to be caught by the main handler


def handler(job):
    """
    RunPod serverless handler function for lip-sync video generation.
    """
    global SAMPLER

    image_path_temp = None
    audio_path_temp = None

    try:
        if SAMPLER is None:
            SAMPLER = load_model()

        job_input = job.get('input', {})

        # Extract parameters for lip-sync
        image_ref_url = job_input.get('image_ref_url')
        audio_guide_url = job_input.get('audio_guide_url')

        if not image_ref_url:
            return {"error": "image_ref_url is a required parameter."}
        if not audio_guide_url:
            return {"error": "audio_guide_url is a required parameter."}

        # Download image and audio
        # Determine suffix from URL if possible, or use generic ones
        img_suffix = os.path.splitext(image_ref_url)[1] or ".jpg"
        audio_suffix = os.path.splitext(audio_guide_url)[1] or ".wav"

        image_path_temp = download_file(image_ref_url, suffix=img_suffix)
        audio_path_temp = download_file(audio_guide_url, suffix=audio_suffix)

        # Load image using PIL
        pil_image_ref = Image.open(image_path_temp).convert("RGB")
        print(f"Loaded reference image: {image_path_temp}, Mode: {pil_image_ref.mode}, Size: {pil_image_ref.size}")


        height = job_input.get('height', 720) # Default or derive from image?
        width = job_input.get('width', 720)  # Default or derive from image? Often square for avatars
        num_inference_steps = job_input.get('num_inference_steps', 30)
        guidance_scale = job_input.get('guidance_scale', 7.5) # Adjusted default
        seed = job_input.get('seed', random.randint(0, 2**32 - 1))
        # video_length for avatar model is often determined by audio length, or a max_frames param
        # Let's assume fps and SAMPLER handles video length based on audio.
        # If video_length is provided, it could act as max_frames.
        video_length = job_input.get('video_length', None) # Max frames, if applicable
        fps = job_input.get('fps', 25) # Adjusted default
        negative_prompt = job_input.get('negative_prompt', "") # May not be used by avatar model
        flow_shift = job_input.get('shift', 5.0) # Adjusted default, used as 'shift' in some models

        # For Avatar models, prompt might be optional or fixed
        input_prompt = job_input.get('prompt', "") # Use empty or default if not driving semantics

        print(f"Received job for lip-sync with Image URL: '{image_ref_url}', Audio URL: '{audio_guide_url}'")
        print(f"Parameters: H={height}, W={width}, Steps={num_inference_steps}, Scale={guidance_scale}, Seed={seed}, FPS={fps}, Shift={flow_shift}")

        generate_params = {
            "prompt": input_prompt, # May be ignored by avatar model if audio/image are primary drivers
            "negative_prompt": negative_prompt,
            "image_refs": [pil_image_ref], # Expects a list of PIL images
            "audio_guide": audio_path_temp, # Path to audio file
            "height": height,
            "width": width,
            "frame_num": video_length, # Or let model decide based on audio length
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "fps": fps, # Pass FPS to sampler if it uses it for audio sync
            "flow_shift": flow_shift, # If applicable for the avatar model's generate method
            # Add other specific parameters if the sampler expects them
        }
        # Remove frame_num if None, so sampler can use audio length
        if video_length is None:
            del generate_params["frame_num"]

        video_frames = SAMPLER.generate(**generate_params)

        print(f"Generated video frames tensor with shape: {video_frames.shape}")

        # Save the video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output_dir = "/app/output"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"video_lipsync_{timestamp}.mp4")

        actual_fps_for_saving = video_frames.shape[0] / (video_frames.shape[0] / fps) # fps used in generate
        save_video_frames(video_frames, output_filename, actual_fps_for_saving)


        return {"video_path": output_filename, "message": "Lip-sync video generated successfully."}

    except Exception as e:
        print(f"Error during handler execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        # Cleanup temporary files
        if image_path_temp and os.path.exists(image_path_temp):
            try:
                os.remove(image_path_temp)
                print(f"Cleaned up temporary image file: {image_path_temp}")
            except Exception as e_clean:
                print(f"Error cleaning up temp image file {image_path_temp}: {e_clean}")
        if audio_path_temp and os.path.exists(audio_path_temp):
            try:
                os.remove(audio_path_temp)
                print(f"Cleaned up temporary audio file: {audio_path_temp}")
            except Exception as e_clean:
                print(f"Error cleaning up temp audio file {audio_path_temp}: {e_clean}")


if __name__ == '__main__':
    print("Starting local test of lip-sync handler...")

    class MockSampler:
        def generate(self, **kwargs):
            print(f"MockSampler.generate called with:")
            for k, v in kwargs.items():
                if k == "image_refs":
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Image.Image):
                        print(f"  {k}: List of 1 PIL Image, mode={v[0].mode}, size={v[0].size}")
                    else:
                        print(f"  {k}: {type(v)}")
                elif isinstance(v, torch.Tensor):
                     print(f"  {k}: Tensor of shape {v.shape}")
                else:
                    print(f"  {k}: {v}")

            frame_num_val = kwargs.get('frame_num', 50) # default to 50 frames if not specified by audio
            if kwargs.get("audio_guide"):
                 # Simulate audio determining length, e.g. 2 seconds at 25 fps = 50 frames
                 print("MockSampler: Audio guide provided, simulating video length based on audio (e.g. 50 frames).")

            # T, H, W, C format for save_video_frames
            return torch.rand(frame_num_val, kwargs.get('height', 720)//8, kwargs.get('width', 720)//8, 3)

        def to(self, device):
            print(f"MockSampler.to({device}) called.")
            return self

    # Create dummy files for testing download_file if needed, or mock requests.get
    # For simplicity, we'll assume download_file works or is tested elsewhere.
    # Here, we primarily test the handler logic flow.

    # To truly test download_file, you might need to set up a local HTTP server
    # or use public URLs, but be mindful of network dependency in tests.
    # Example public URLs (replace with actual small files for testing):
    # IMG_URL = "https://raw.githubusercontent.com/runpod/runpod-python/main/docs/runpod-logo.png" # Example image
    # AUDIO_URL = "https://www.kozco.com/tech/piano2.wav" # Example small wav

    # For this test, we'll use placeholders and might not actually download if not in full test env.
    IMG_URL = "https://via.placeholder.com/150.png/09f/fff?text=TestImage"
    AUDIO_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" # Using mp3, suffix will be .mp3

    # Mock requests.get if you want to avoid network calls in unit tests
    original_requests_get = requests.get
    def mock_requests_get(url, stream=True):
        print(f"MOCK requests.get called for {url}")
        class MockResponse:
            def __init__(self, content, status_code):
                self.content = content
                self.status_code = status_code
                self.headers = {'content-type': 'application/octet-stream'}
            def raise_for_status(self):
                if self.status_code != 200: raise requests.exceptions.HTTPError(f"Mock Error {self.status_code}")
            def iter_content(self, chunk_size=8192):
                for i in range(0, len(self.content), chunk_size):
                    yield self.content[i:i+chunk_size]
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass

        if "placeholder.com" in url:
            # Serve a tiny fake PNG
            # (bytes for a 1x1 transparent PNG)
            fake_image_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            return MockResponse(fake_image_content, 200)
        elif "soundhelix.com" in url: # Fake some audio data
            return MockResponse(b"fake_audio_data_very_short", 200)
        return original_requests_get(url, stream=stream) # fallback to real requests for other URLs

    if os.environ.get("RUNPOD_TEST_ENV_LIPSYNC"):
        SAMPLER = MockSampler()
        requests.get = mock_requests_get # Patch requests.get for this test scope
        print("Using MockSampler and mocked downloads for lip-sync testing.")
    else:
        print("Attempting to load real model for lip-sync (may fail without models/GPU). If so, no test run.")
        try:
            SAMPLER = load_model()
            # For real model testing, ensure IMG_URL and AUDIO_URL point to valid small files
            # IMG_URL = "URL_TO_YOUR_TEST_IMAGE.jpg"
            # AUDIO_URL = "URL_TO_YOUR_TEST_AUDIO.wav"
            # Potentially skip requests.get mocking if you want to test real downloads
        except Exception as e:
            print(f"Failed to load real model for local lip-sync test: {e}. Exiting test.")
            SAMPLER = None # Ensure it's None so test doesn't run

    if SAMPLER:
        test_job_lipsync = {
            "input": {
                "image_ref_url": IMG_URL,
                "audio_guide_url": AUDIO_URL,
                "height": 256,
                "width": 256,
                "video_length": None, # Let audio decide or sampler default
                "fps": 25,
                "num_inference_steps": 5,
                "seed": 123,
                "guidance_scale": 7.5,
                "shift": 5.0
            }
        }
        result_lipsync = handler(test_job_lipsync)
        print(f"Lip-sync handler result: {result_lipsync}")

        # Test with an error case, e.g., missing URL
        test_job_error_lipsync = {
             "input": {
                "audio_guide_url": AUDIO_URL
            }
        }
        result_error_lipsync = handler(test_job_error_lipsync)
        print(f"Lip-sync handler error result: {result_error_lipsync}")

        # Restore original requests.get if it was mocked
        if os.environ.get("RUNPOD_TEST_ENV_LIPSYNC"):
            requests.get = original_requests_get
    else:
        print("SAMPLER not available (real or mock), skipping lip-sync handler test execution.")
