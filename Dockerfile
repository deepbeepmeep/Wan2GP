# Base image with Python 3.10 and CUDA 12.4
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Change to Wan2GP directory
WORKDIR /app/Wan2GP
COPY . .

# Install dependencies with PIP
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu124 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sageattention==1.0.6

# Default command
ENTRYPOINT ["python", "wgp.py", "--t2v-1-3B", "--listen"]