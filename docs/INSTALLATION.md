# Manual Installation Guide For Windows & Linux

This guide covers installation for different GPU generations and operating systems.

## Requirements

### - Compatible GPU (GTX 10XX - RTX 50XX)
- Git [Git Download](https://github.com/git-for-windows/git/releases/download/v2.51.2.windows.1/Git-2.51.2-64-bit.exe)
- Build Tools for Visual Studio 2022 with C++ Extentions [Vs2022 Download](https://aka.ms/vs/17/release/vs_BuildTools.exe)
- Cuda Toolkit 12.8 or higher [Cuda Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- Nvidia Drivers Up to Date [Nvidia Drivers Download](https://www.nvidia.com/en-us/software/nvidia-app/)
- FFMPEG downloaded, unzipped & the bin folder on PATH [FFMPEG Download](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-win64-gpl-8.0.zip)
- Python 3.10.9 [Python Download](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe)
- Miniconda [Miniconda Download](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) or Python venv


  <img width="1234" height="962" alt="miniconda_1234x962" src="https://github.com/user-attachments/assets/222650d9-77e1-4c9e-8319-dfba9bc409d3" />



## Installation for Nvidia GTX 10XX - RTX QUADRO - 50XX (Stable)

This installation uses PyTorch 2.6.0, Cuda 12.6 for GTX 10XX - RTX 30XX & PyTorch 2.7.1, Cuda 12.8 for RTX 40XX - 50XX which are well-tested and stable.

It is not recommeneded to use neither PytTorch 2.8.0 as some System RAM memory leaks have been observed when switching models nor 2.9.0 which has some Convolution 3D perf issues (VAE VRAM requirements explode)

If you want to use the NV FP4 optimized kernels for RTX 50xx, you will need to upgrade to Python 3.11, PyTorch 2.10 with Cuda 13.0

## Download Repo and Setup Conda Environment

 

### Clone the repository

#### First, Create a folder named Wan2GP, then open it, then right click & select "open in terminal", then copy & paste the following commands, one at a time.


```
git clone https://github.com/deepbeepmeep/Wan2GP.git
```

#### Create Python 3.10.9 environment using Conda
```
conda create -n wan2gp python=3.10.9
```
#### Activate Conda Environment
```
conda activate wan2gp
```


# NOW CHOOSE INSTALLATION ACCORDING TO YOUR GPU


## Windows Installation for GTX 10XX -16XX Only


#### Windows Install PyTorch 2.6.0 with CUDA 12.6 for GTX 10XX -16XX Only
```shell
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

#### Windows Install requirements.txt for GTX 10XX -16XX Only
```
pip install -r requirements.txt
```


## Windows Installation for RTX QUADRO - 20XX Only


#### Windows Install PyTorch 2.6.0 with CUDA 12.6 for RTX QUADRO - 20XX Only
```
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```
#### Windows Install Triton for RTX QUADRO - 20XX Only
```
pip install -U "triton-windows<3.3"
```
#### Windows Install Sage1 Attention for RTX QUADRO - 20XX Only
```
pip install sageattention==1.0.6
```
#### Windows Install requirements.txt for RTX QUADRO - 20XX Only
```
pip install -r requirements.txt
```


## Windows Installation for RTX 30XX Only


#### Windows Install PyTorch 2.6.0 with CUDA 12.6 for RTX 30XX Only
```
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```
#### Windows Install Triton for RTX 30XX Only
```
pip install -U "triton-windows<3.3"
```
#### Windows Install Sage2 Attention for RTX 30XX Only
```
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu126torch2.6.0-cp310-cp310-win_amd64.whl
```
#### Windows Install requirements.txt for RTX 30XX Only
```
pip install -r requirements.txt
```


## Installation for RTX 40XX, 50XX Only

#### Windows Install PyTorch 2.7.1 with CUDA 12.8 for RTX 40XX - 50XX Only
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
#### Windows Install Triton for RTX 40XX, 50XX Only
```
pip install -U "triton-windows<3.4"
```
#### Windows Install Sage2 Attention for RTX 40XX, 50XX Only
```
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.7.1-cp310-cp310-win_amd64.whl
```
#### Windows Install requirements.txt for RTX 40XX, 50XX Only
```
pip install -r requirements.txt
```

## Installation for 50XX Only Python 3.11, PyTorch 2.10.0 Cuda 13. for NVFP4 optimized kernels
#### Create Python 3.11 environment using Conda
```
conda create -n wan2gp python=3.11.14
```
#### Windows Install PyTorch 2.10.0 with CUDA 13.0 for RTX 50XX Only
```
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```
#### Windows Install Triton for RTX 50XX Only
```
pip install -U triton-windows
```
#### Windows Install Sage2 Attention for RTX 50XX Only
```
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```
#### Windows Install requirements.txt for RTX 50XX Only
```
pip install -r requirements.txt
```
## Optional

### Flash Attention Windows

#### Pytorch 2.7.1
```
pip install https://github.com/Redtash1/Flash_Attention_2_Windows/releases/download/v2.7.0-v2.7.4/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```
#### Pytorch 2.10 
https://github.com/deepbeepmeep/kernels/releases/download/Flash2/flash_attn-2.8.3-cp311-cp311-win_amd64.whl



# Linux Installation 

### Step 1: Download Repo and Setup Conda Environment

#### Clone the repository
```
git clone https://github.com/deepbeepmeep/Wan2GP.git
```
#### Change directory
```
cd Wan2GP
```

#### Create Python 3.10.9 environment using Conda
```
conda create -n wan2gp python=3.10.9
```
#### Activate Conda Environment
```
conda activate wan2gp
```

## Installation for RTX 10XX -16XX Only


#### Install PyTorch 2.6.0 with CUDA 12.6 for RTX 10XX -16XX Only
```shell
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

#### Install requirements.txt for RTX 30XX Only
```
pip install -r requirements.txt
```


## Installation for RTX QUADRO - 20XX Only


#### Install PyTorch 2.6.0 with CUDA 12.6 for RTX QUADRO - 20XX Only
```
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```
#### Install Triton for RTX QUADRO - 20XX Only
```
pip install -U "triton<3.3"
```
#### Install Sage1 Attention for RTX QUADRO - 20XX Only
```
pip install sageattention==1.0.6
```
#### Install requirements.txt for RTX QUADRO - 20XX Only
```
pip install -r requirements.txt
```


## Installation for RTX 30XX Only


#### Install PyTorch 2.6.0 with CUDA 12.6 for RTX 30XX Only
```
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```
#### Install Triton for RTX 30XX Only
```
pip install -U "triton<3.3"
```
#### Install Sage2 Attention for RTX 30XX Only. Make sure it's Sage 2.1.1
```
python -m pip install "setuptools<=75.8.2" --force-reinstall
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .
```
#### Install requirements.txt for RTX 30XX Only
```
pip install -r requirements.txt
```


## Installation for RTX 40XX, 50XX Only

#### Install PyTorch 2.7.1 with CUDA 12.8 for RTX 40XX - 50XX Only
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
#### Install Triton for RTX 40XX, 50XX Only
```
pip install -U "triton<3.4"
```
#### Install Sage Attention for RTX 40XX, 50XX Only. Make sure it's Sage 2.2.0
```
python -m pip install "setuptools<=75.8.2" --force-reinstall
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .
```
#### Install requirements.txt for RTX 40XX, 50XX Only
```
pip install -r requirements.txt
```
## Optional

### Flash Attention

#### Linux
```
pip install flash-attn==2.7.2.post1
```
 
## Attention Modes

### WanGP supports several attention implementations:

- **SDPA** (default): Available by default with PyTorch
- **Sage**: 30% speed boost with small quality cost
- **Sage2**: 40% speed boost 
- **Flash**: Good performance, may be complex to install on Windows

### Attention GPU Compatibility

- RTX 10XX: SDPA
- RTX 20XX: SPDA, Sage1
- RTX 30XX, 40XX: SDPA, Flash Attention, Xformers, Sage1, Sage2/Sage2++
- RTX 50XX: SDPA, Flash Attention, Xformers, Sage2/Sage2++ / Sage3

## Performance Profiles

Choose a profile based on your hardware:

- **Profile 3 (LowRAM_HighVRAM)**: Loads entire model in VRAM, requires 24GB VRAM for 8-bit quantized 14B model
- **Profile 4 (LowRAM_LowVRAM)**: Default, loads model parts as needed, slower but lower VRAM requirement

## Troubleshooting

### Sage Attention Issues

If Sage attention doesn't work:

1. Check if Triton is properly installed
2. Clear Triton cache
3. Fallback to SDPA attention:
   ```
   python wgp.py --attention sdpa
   ```

### Memory Issues

- Use lower resolution or shorter videos
- Enable quantization (default)
- Use Profile 4 for lower VRAM usage
- Consider using 1.3B models instead of 14B models


For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 

## Optional Kernels for INT4 / FP4 quantized support
These kernels will offer optimized INT4 / FP4 dequantization.

**Please Note FP4 support is hardware dependent and will work only with RTX 50xx / sm120+ GPUs**


### Light2xv NVP4 Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13 (RTX 50xx / sm120+ only !)
- Windows
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-win_amd64.whl
   ```

- Linux
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl
   ```



### Nunchaku INT4/FP4 Kernels Wheels for Python 3.10 / Pytorch 2.7.1 / Cuda 12.8  
- Windows () 
   ```
   pip install https://github.com/deepbeepmeep/kernels/releases/download/v1.2.0_Nunchaku/nunchaku-1.2.0+torch2.7-cp310-cp310-win_amd64.whl
   ```
- Linux (Pytorch 2.7.1 / Cuda 12.8) 
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/v1.2.0_Nunchaku/nunchaku-1.2.0+torch2.7-cp310-cp310-linux_x86_64.whl
   ```

### Nunchaku INT4/FP4 Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13

- Windows 
   ```
  pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
   ```

- Linux 
   ```
  pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp310-cp310-linux_x86_64.whl
   ```