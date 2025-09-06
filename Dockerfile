FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt update && apt install -y --no-install-recommends \
    software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install dependencies with PIP
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu124   \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sageattention==1.0.6

# Default entrypoint
ENTRYPOINT ["python3.10", "wgp.py", "--listen"]
