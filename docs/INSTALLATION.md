# Manual Installation Guide For Windows & Linux

This guide covers manual installation for different GPU generations and operating systems. Alternatively you may use the 1 click install / update scripts (please check the repo readme for instructions).

It is recommended to use Python 3.10.9, PyTorch 2.7.1 with Cuda 12.8 for GTX 10XX and Python 3.11.14, PyTorch 2.10 with Cuda 13.0/13.1 for RTX 20XX - RTX 50XX as both these configs are well-tested and stable.

It is not recommended to use either PytTorch 2.8.0 as some System RAM memory leaks have been observed when switching models or 2.9.0 which has some Convolution 3D perf issues (VAE VRAM requirements explode).

If you want to use the NV FP4 optimized kernels for RTX 50xx, you will need to upgrade to Python 3.11, PyTorch 2.10 with Cuda 13.0 if you are still using the old install setup based on cuda 12.8.

## Setup Conda

You need to install anaconda or miniconda first (https://www.anaconda.com/download/success?reg=skipped) 

## Minimal WanGP installation
### RTX 20xx - RTX 50xx Installation
you must install Cuda 13.1: https://developer.nvidia.com/cuda-13-1-0-download-archive

Then open a Terminal Window get in the parent folder where you would to install WanGP and then type in:
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### GTX 10xx Installation

you must install Cuda 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive

Then open a Terminal Window get in the parent folder where you would to install WanGP and then type in:
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

## Optional DLSS 5 upsamplers

WanGP can expose NVIDIA DLSS 5 Neural Rendering as a native-resolution refiner or spatial upsampler, and DLSS Frame Generation as a temporal upsampler. These optional Windows components are not installed by the normal WanGP installer and include closed-source third-party binaries with separate licenses and security implications. Close WanGP and run `scripts\install_dlss5.bat` for the checksum-verified automatic installation.

Read the full **[DLSS 5 runtime installation, directory layout, copyright, and safety instructions](DLSS5.md)** before downloading or running them. The WanGP worker release ZIP is extracted into the root `dlss5` folder; the guide provides version-pinned downloads for the tested community components and identifies which files are modified, unsigned, or unavailable from NVIDIA's public SDK.


## Triton Installation
The Triton library is required for Pytorch compilation and Sage Attention and by various kernels to accelerate tensors processing.

### Windows RTX 20XX -RTX 30xx
```
pip install -U "triton-windows<3.3"
```

### Windows RTX 40XX -RTX 50xx
```
pip install triton-windows
```

### Linux
Triton library should be automatically installed when installing pytorch.

## Sage Attention
Sage Attention accelerates Video / Image Generation up to x2 with very little quality loss. Sage does not support GTX 10XX in WanGP.

Use the version matching your GPU generation:

- **RTX 20XX (Turing):** SageAttention 1.0.6. SageAttention 2 is not supported.
- **RTX 30XX or newer (Ampere, Ada, and Blackwell):** SageAttention 2.2.0.

#### Windows: Install SageAttention 1 for RTX 20XX
```
pip install sageattention==1.0.6
```

#### Windows: Install SageAttention 2 for RTX 30XX-50XX
```
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```

#### Linux: Install SageAttention 1 for RTX 20XX
```
pip install sageattention==1.0.6
```

#### Linux: Install SageAttention 2 for RTX 30XX-50XX
```
python -m pip install "setuptools<=75.8.2" --force-reinstall
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install --no-build-isolation -e .
```

## Sparge Attention
Sparge Attention (`spas_sage_attn`) provides the optimized sparse attention kernels used by FlashVSR. Install it after Pytorch and Triton.

#### Windows Install Sparge Attention for Pytorch 2.10 / Python 3.11 / Cuda 13
```
pip install https://github.com/woct0rdho/SpargeAttn/releases/download/v0.1.0-windows.post4/spas_sage_attn-0.1.0%2Bcu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```

#### Windows Install Sparge Attention for Pytorch 2.7.1 / Python 3.10 / Cuda 12.8
```
pip install https://github.com/woct0rdho/SpargeAttn/releases/download/v0.1.0-windows.post3/spas_sage_attn-0.1.0%2Bcu128torch2.7.1.post3-cp39-abi3-win_amd64.whl
```

#### Linux Install Sparge Attention
```
python -m pip install ninja wheel packaging
python -m pip install --no-build-isolation git+https://github.com/woct0rdho/SpargeAttn.git
```


## Flash Attention
Flash attention is not as fast as Sage for Generating Videos or Images but it preserves quality. However when used with a Language Model (prompt enhancer, Text to Speech, Deepy) it can offer a significant speedup.

 
### Flash Attention Windows
#### Windows Pytorch 2.10 / Python 3.11
```
pip install https://github.com/deepbeepmeep/kernels/releases/download/Flash2/flash_attn-2.8.3-cp311-cp311-win_amd64.whl
```

#### Windows Pytorch 2.7.1 / Python 3.10
```
pip install https://github.com/Redtash1/Flash_Attention_2_Windows/releases/download/v2.7.0-v2.7.4/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

#### Linux
```
pip install flash-attn==2.7.2.post1
```


## GGUF llama.cpp CUDA Kernels

These kernels accelerate GGUF models with packed MMVQ/MMQ, direct FP16/BF16 activation quantization, CUDA-graph-safe workspaces and quantized KV-cache attention. Wheel **1.0.21** also contains precompiled RTX50xx (SM120) async-copy kernels for Q8 prefill and decode/verification. WanGP's vLLM backend selects them automatically on compatible GPUs; this async path needs no runtime Triton compilation. Other architectures retain the shared kernels.

Install the wheel matching your Python, PyTorch and CUDA stack. `--no-deps` preserves the installed PyTorch environment.

### Python 3.11 / PyTorch 2.10 / CUDA 13

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.21/llamacpp_gguf_cuda-1.0.21%2Btorch210cu130py311-cp311-cp311-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.21/llamacpp_gguf_cuda-1.0.21%2Btorch210cu130py311-cp311-cp311-linux_x86_64.whl
```

### Python 3.10 / PyTorch 2.7.1 / CUDA 12.8

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.21/llamacpp_gguf_cuda-1.0.21%2Btorch271cu128py310-cp310-cp310-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.21/llamacpp_gguf_cuda-1.0.21%2Btorch271cu128py310-cp310-cp310-linux_x86_64.whl
```

The CUDA 13 builds contain native GPU code for SM75 through the architectures supported by CUDA 13.1. CUDA 12.8 builds additionally contain pre-SM75 code, subject to PyTorch's own support. The release includes the exact architecture lists, source and build instructions. Hardware validation was performed on RTX5090; Linux wheels were built and tested under Ubuntu 22.04 in WSL.

### Matmul selection and CUDA graphs

The default keeps weights packed and selects MMVQ for decoding/short batches or MMQ for larger batches. To override it, set `WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE` before starting WanGP:

- `fast` or `low_vram`: packed MMVQ/MMQ, with no full dense weight materialization.
- `materialized` or `cublas`: materialize weights for cuBLAS.

For example, in PowerShell:
```powershell
$env:WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE = "low_vram"
python wgp.py
```

On Linux:
```bash
export WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE=low_vram
python wgp.py
```

The wrapper reads the setting for eager calls. Existing CUDA graphs must be recreated to change their recorded operations. MMQ reuses a Stream-K workspace sized from the GPU's SM count and rounded to 16 MiB; WanGP reserves it before graph capture. This version has no `refresh_env()` API or Stream-K environment controls. To disable the GGUF CUDA package, set `WGP_GGUF_LLAMACPP_CUDA=0` before starting WanGP.

## INT4 / FP4 quantized support

These kernels will offer optimized INT4 / FP4 dequantization.

**Please Note FP4 support is hardware dependent and will work only with RTX 50xx / sm120+ GPUs**


### Lightx2v NVP4 Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13 (RTX 50xx / sm120+ only !)
- Windows
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-win_amd64.whl
   ```

- Linux
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl
   ```


### Nunchaku INT4/FP4 Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13

- Windows 
   ```
  pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
   ```

- Linux 
   ```
  pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-linux_x86_64.whl
   ```


### Nunchaku INT4/FP4 Kernels Wheels for Python 3.10 / Pytorch 2.7.1 / Cuda 12.8  
- Windows
   ```
   pip install https://github.com/deepbeepmeep/kernels/releases/download/v1.2.0_Nunchaku/nunchaku-1.2.0+torch2.7-cp310-cp310-win_amd64.whl
   ```

- Linux (Pytorch 2.7.1 / Cuda 12.8) 
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/v1.2.0_Nunchaku/nunchaku-1.2.0+torch2.7-cp310-cp310-linux_x86_64.whl
   ```


### Bitsandbytes NF4 Kernels for Python 3.11 / Pytorch 2.10 / Cuda 13

These kernels accelerate bitsandbytes 4-bit / NF4 checkpoints. Install them after Pytorch; pip will pick the matching Windows or Linux wheel automatically.
```
pip install bitsandbytes==0.49.2
```
