# Manual Installation Guide For Windows & Linux

This guide covers manual installation for different GPU generations and operating systems. Alternatively you may use the 1 click install / update scripts (please check the repo readme for instructions).

It is recommended to use Python 3.10.9, PyTorch 2.7.1 with Cuda 12.8 for GTX 10XX and Python 3.11.14, PyTorch 2.10 with Cuda 13.0/13.1 for RTX 20XX - RTX 50XX. RTX 20XX requires Triton 3.2.

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

## INT8 Math Kernels

In the configuration menu, **INT8 Math Kernels** selects the backend for INT8 quantized linear layers, including INT8 ConvRot checkpoints:

- **Auto** (default): tries WGPKernels, then Comfy Kitchen CUDA on NVIDIA or HIP on AMD ROCm, then Triton, then PyTorch when no other backend is available.
- **WGPKernels**: selects WanGP's own INT8 kernels, with higher precision. See [WGPKernels Installation](#wgpkernels-installation) below.
- **Triton**: selects WanGP's Triton INT8 kernels. See [Triton Installation](#triton-installation) below.
- **Comfy Kitchen Kernels**: selects Kitchen's native CUDA or HIP kernels for the current PyTorch runtime.
- **Disabled: PyTorch**: uses the standard PyTorch/Quanto path.

An unavailable selection behaves as Auto. Existing `enable_int8_kernels` configurations migrate automatically: enabled becomes `int8_kernels: "auto"`, disabled becomes `int8_kernels: "disabled"`.

Comfy Kitchen is installed with `requirements.txt`; its prebuilt CUDA kernels require an NVIDIA R580+ driver. AMD INT8/ConvRot uses ROCm PyTorch and Kitchen's HIP extension on supported RDNA3/3.5/4 devices. RDNA2 and unavailable or failing HIP backends retain the Triton/PyTorch fallback. WanGP probes small plain and ConvRot linear operations before selecting Kitchen, and reports CUDA or HIP in the startup message. This option selects INT8 linear math; it does not change the attention backend or enable FP16 accumulation.

### WGPKernels Installation

WGPKernels are WanGP's own INT8 kernels, with higher precision than the other fast INT8 backends. Install the wheel matching the CUDA version of your PyTorch (NVIDIA RTX 30XX or newer):

For **PyTorch 2.10 / CUDA 13** (NVIDIA R580+ driver):

```bash
# Windows
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.1.0/wgpkernels-1.1.0%2Bcu130-py3-none-win_amd64.whl
# Linux (glibc 2.34+)
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.1.0/wgpkernels-1.1.0%2Bcu130-py3-none-linux_x86_64.whl
```

For **PyTorch 2.7.1 / CUDA 12.8** (NVIDIA R570+ driver):

```bash
# Windows
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.1.0/wgpkernels-1.1.0%2Bcu128-py3-none-win_amd64.whl
# Linux (glibc 2.34+)
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.1.0/wgpkernels-1.1.0%2Bcu128-py3-none-linux_x86_64.whl
```

For **AMD RX 9070 with PyTorch 2.10.0+rocm7.14.0** (Windows, experimental):

```powershell
python -m pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.0.0/wgpkernels-1.0.0%2Brocm714-py3-none-win_amd64.whl
```

### CUDA Kernels Optimized Ops Precision (When Available)

This dropdown controls optional CUDA kernel optimizations:

- **Preserve Precision**: uses kernels at least as accurate as standard PyTorch math; tiny output differences are still possible.
- **Allow Faster Approximate Kernels** (default, `kernel_precision: "fast"`): allows small precision trade-offs for potential speed or VRAM savings.

Changes apply between generations without reloading weights. Saved choices are preserved; missing values use the default. **INT8 Math Kernels** sets the preferred kernel family, with automatic fallback when unavailable. Additional H3 optimizations require validated RTX 50-series (SM120) GPUs. Checkpoint dtypes, FP16 accumulation and attention settings are unchanged.



### H3 Video VAE Selection

H3's **Video VAE** configuration defaults to **Auto**, which follows the current **Transformer Model Quantization** choice: 16-bit selects BF16, FP8 selects FP8 Mixed Precision, and INT8 selects INT8 ConvRot Decoder. Explicit **BF16**, **FP8 Mixed Precision**, and **INT8 ConvRot Decoder** choices override Auto. The selected VAE is downloaded automatically before loading; INT8 math still follows the separate **INT8 Math Kernels** selection.

## Optional DLSS 5 upsamplers

WanGP can expose NVIDIA DLSS 5 Neural Rendering as a native-resolution refiner or spatial upsampler, and DLSS Frame Generation as a temporal upsampler. These optional Windows components are not installed by the normal WanGP installer and include closed-source third-party binaries with separate licenses and security implications. Close WanGP and run `scripts\install_dlss5.bat` for the checksum-verified automatic installation.

Read the full **[DLSS 5 runtime installation, directory layout, copyright, and safety instructions](DLSS5.md)** before downloading or running them. The WanGP worker release ZIP is extracted into the root `dlss5` folder; the guide provides version-pinned downloads for the tested community components and identifies which files are modified, unsigned, or unavailable from NVIDIA's public SDK.


## Triton Installation
The Triton library is required for Pytorch compilation and Sage Attention and by various kernels to accelerate tensors processing.

### Windows RTX 30XX - RTX 50XX

Use the Triton version matching your PyTorch installation. For the recommended **PyTorch 2.10** environment:

```bash
python -m pip install -U "triton-windows>=3.6,<3.7"
```

For an existing **PyTorch 2.7 / 2.7.1** environment:

```bash
python -m pip install -U "triton-windows>=3.3,<3.4"
```

Triton 3.3 or newer is required for optimized ConvRot support.

### Windows RTX 20XX

RTX 20XX must use Triton 3.2:

```bash
python -m pip install -U "triton-windows>=3.2,<3.3"
```

### Linux

PyTorch normally installs the matching Triton version automatically: 3.3 for PyTorch 2.7, or 3.6 for PyTorch 2.10. RTX 20XX requires Triton 3.2.

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
For the recommended PyTorch 2.10 environment:
```
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl
```
This version (or a later one) is needed for the **CUDA Graph** compilation mode to keep SageAttention's fastest kernels on RTX 40XX/50XX: with older versions, CUDA Graph mode switches the attention to slower SageAttention kernels.

#### Linux: Install SageAttention 1 for RTX 20XX
```
pip install sageattention==1.0.6
```

#### Linux: Install SageAttention 2 for RTX 30XX-50XX
Prebuilt wheels come from the [WanGP SageAttention fork](https://github.com/deepbeepmeep/SageAttention), no compilation needed. They also keep SageAttention's fastest kernels in **CUDA Graph** mode.

For the recommended Python 3.11 / PyTorch 2.10 / CUDA 13 environment:
```
pip install https://github.com/deepbeepmeep/SageAttention/releases/download/v2.2.0-linux.cg1/sageattention-2.2.0+cu130torch2.10.0.cg1-cp311-cp311-linux_x86_64.whl
```
The `cp311` part of the file name (it appears twice) is the Python version (3.11) and must match your environment's Python: with Python 3.10 or 3.12, replace it with `cp310` or `cp312` (see the [release page](https://github.com/deepbeepmeep/SageAttention/releases/tag/v2.2.0-linux.cg1)). These wheels work only with PyTorch 2.10 and CUDA 13, and need a Linux distribution from 2018 or later (glibc 2.28). The 1 click installer picks the right wheel automatically.

For another PyTorch version, compile the fork instead (this takes a while):
```
python -m pip install "setuptools<=75.8.2" ninja wheel --force-reinstall
git clone https://github.com/deepbeepmeep/SageAttention
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

The FlashAttention 2 packages below require RTX 30XX or newer. RTX 20XX users should use SDPA or SageAttention 1 where supported, and the PyTorch/legacy LM decoder for autoregressive models.

 
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

These kernels accelerate GGUF models with packed MMVQ/MMQ, direct FP16/BF16 activation quantization, CUDA-graph-safe workspaces and quantized KV-cache attention. 

Install the wheel matching your Python, PyTorch and CUDA stack. `--no-deps` preserves the installed PyTorch environment.

### Python 3.11 / PyTorch 2.10 / CUDA 13

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.25/llamacpp_gguf_cuda-1.0.25%2Btorch210cu130py311-cp311-cp311-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.25/llamacpp_gguf_cuda-1.0.25%2Btorch210cu130py311-cp311-cp311-linux_x86_64.whl
```

### Python 3.10 / PyTorch 2.7.1 / CUDA 12.8

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.25/llamacpp_gguf_cuda-1.0.25%2Btorch271cu128py310-cp310-cp310-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.25/llamacpp_gguf_cuda-1.0.25%2Btorch271cu128py310-cp310-cp310-linux_x86_64.whl
```

The CUDA 13 builds contain native GPU code for SM75 through the architectures supported by CUDA 13.1. CUDA 12.8 builds additionally contain pre-SM75 code, subject to PyTorch's own support. The release includes the exact architecture lists, source and build instructions. Hardware validation was performed on RTX5090; Linux wheels were built and tested under Ubuntu 22.04 in WSL.

### Experimental AMD HIP wheel: Windows / Python 3.11 / PyTorch 2.10

For RX 9070-series GPUs (`gfx1201`), the separate HIP wheel targets exactly
**PyTorch 2.10.0+rocm7.14.0**. It does not use the NVIDIA PyTorch build and does not
load in the installer's AMD environment (PyTorch 2.13.0+rocm10.0.0). Install into a
separate AMD environment:

```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1201]==2.10.0+rocm7.14.0" "rocm[libraries,device-gfx1201]==7.14.0"
python -m pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.25/llamacpp_gguf_cuda-1.0.25%2Btorch210rocm714py311-cp311-cp311-win_amd64.whl
```

The HIP and CUDA wheels use the same package name; select the matching backend.
The HIP wheel rejects a different PyTorch build at import. It replaces the earlier
`+hip` asset compiled against PyTorch 2.12. Compilation and DLL import checks do
not establish AMD inference correctness: hardware validation remains pending.


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

---

> Applies to: Manual installation and dependencies on Windows and Linux. Requirements vary by operating system and GPU; the commands in this guide concern installation and launch.
