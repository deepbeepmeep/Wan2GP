# Manual Installation Guide For Windows & Linux

This guide covers manual installation for different GPU generations and operating systems. Alternatively you may use the 1 click install / update scripts (please check the repo readme for instructions).

It is recommended to use Python 3.10.9, PyTorch 2.7.1 with Cuda 12.8 for GTX 10XX and Python 3.11.14, PyTorch 2.10 with Cuda 13.0/13.1 for RTX 30XX - RTX 50XX as both these configs are well-tested and stable.

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
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
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
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```


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
Sage Attention accelerates a Video / Image Generation up to x2 with very little quality loss. Sage doesnt support GTX 10xx.

#### Windows Install Sage Attention for RTX 30XX Only
Only Sage attention 1 is supported for these GPUs 
```
pip install sageattention==1.0.6
```
#### Windows Install Sage2 Attention for RTX 40XX-50xx 
```
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```

#### Linux Install Sage Attention for RTX 30XX Only
Only Sage attention 1 is supported for these GPUs 
```
pip install sageattention==1.0.6
```

#### Linux Install Sage Attention for RTX 40XX, 50XX Only. Make sure it's Sage 2.2.0
```
python -m pip install "setuptools<=75.8.2" --force-reinstall
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .
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

These kernels are used to accelerate GGUF models. 

### GGUF Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13

- Windows
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch210cu13py311-cp311-cp311-win_amd64.whl
   ```

- Linux
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch210cu13py311-cp311-cp311-linux_x86_64.whl
   ```

### GGUF Kernels Wheels for Python 3.10 / Pytorch 2.7.1 / Cuda 12.8

- Windows
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch271cu128py310-cp310-cp310-win_amd64.whl
   ```

- Linux
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch271cu128py310-cp310-cp310-linux_x86_64.whl
   ```

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

