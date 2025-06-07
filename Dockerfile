# 1. Start from the runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04 base image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# 2. Install ffmpeg and git
RUN apt-get update && \
    apt-get install -y ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# 3. Create a working directory /app
WORKDIR /app

# 4. Copy the hyvideo directory from the repository into /app/hyvideo
COPY ./hyvideo /app/hyvideo

# 5. Copy the requirements.txt file from the repository root into /app/requirements.txt
COPY ./requirements.txt /app/requirements.txt

# 6. Install Python dependencies from the copied requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 7. Create directories for checkpoints
RUN mkdir -p /app/ckpts/llava-llama-3-8b && \
    mkdir -p /app/ckpts/clip_vit_large_patch14 && \
    mkdir -p /app/ckpts/whisper-tiny && \
    mkdir -p /app/ckpts/det_align

# 8. Download the Hunyuan video model files
# Note: Using a single RUN command with python -c "from huggingface_hub import hf_hub_download; hf_hub_download(...); hf_hub_download(...)"
# is generally preferred to minimize layers, but for readability and easier debugging of individual downloads,
# separate commands or a script might be used in other contexts. Here, combining them.
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='hunyuan_video_720_bf16.safetensors', local_dir='/app/ckpts/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='hunyuan_video_VAE_fp32.safetensors', local_dir='/app/ckpts/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='hunyuan_video_VAE_config.json', local_dir='/app/ckpts/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='llava-llama-3-8b/config.json', local_dir='/app/ckpts/llava-llama-3-8b/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='llava-llama-3-8b/special_tokens_map.json', local_dir='/app/ckpts/llava-llama-3-8b/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='llava-llama-3-8b/tokenizer.json', local_dir='/app/ckpts/llava-llama-3-8b/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='llava-llama-3-8b/tokenizer_config.json', local_dir='/app/ckpts/llava-llama-3-8b/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='llava-llama-3-8b/preprocessor_config.json', local_dir='/app/ckpts/llava-llama-3-8b/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors', local_dir='/app/ckpts/llava-llama-3-8b/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/config.json', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/merges.txt', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/model.safetensors', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/preprocessor_config.json', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/special_tokens_map.json', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/tokenizer.json', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/tokenizer_config.json', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='clip_vit_large_patch14/vocab.json', local_dir='/app/ckpts/clip_vit_large_patch14/', local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='DeepBeepMeep/HunyuanVideo', filename='det_align/detface.pt', local_dir='/app/ckpts/det_align/', local_dir_use_symlinks=False)"

# 9. Set the working directory to /app
WORKDIR /app

# Copy the handler script into the working directory
COPY handler.py /app/handler.py

# 10. Define the command to run the handler
CMD ["python", "handler.py"]
