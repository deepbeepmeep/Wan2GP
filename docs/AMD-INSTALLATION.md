# AMD Installation Guide for Windows (TheRock)

This guide covers installation for AMD GPUs and APUs running under Windows using TheRock's official PyTorch wheels.

## Supported GPUs

Based on [TheRock's official support matrix](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md), the following GPUs are supported on Windows:

### **gfx110X-all** (RDNA 3):
* AMD RX 7900 XTX (gfx1100)
* AMD RX 7800 XT (gfx1101)
* AMD RX 7700 XT (gfx1101)
* AMD RX 7700S / Framework Laptop 16 (gfx1102)
* AMD Radeon 780M Laptop iGPU (gfx1103)

### **gfx1150** (RDNA 3.5 APU) 
* AMD Radeon 890M (Ryzen AI 9 HX 370 - Strix Point)

### **gfx1151** (RDNA 3.5 APU):
* AMD Strix Halo APUs

### **gfx120X-all** (RDNA 4):
* AMD RX 9060 XT (gfx1200)
* AMD RX 9060 (gfx1200)
* AMD RX 9070 XT (gfx1201)
* AMD RX 9070 (gfx1201)

### Also supported:
* **gfx103X-dgpu**: (RDNA 2)

**Note:** If your GPU is not listed above, it is not supported by TheRock on Windows. Support status and future updates can be found in the [official documentation](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md).

## Requirements

- Python 3.11 (recommended for Wan2GP - TheRock currently supports Python 3.11, 3.12, and 3.13).
- Windows 10/11

## Installation Environment

This installation uses PyTorch wheels built by TheRock.

### Installing Python

Download Python 3.11 from [python.org/downloads/windows](https://www.python.org/downloads/windows/). Press Ctrl+F and search for "3.11.". 
Alternatively, you can use this direct link: [Python 3.11.9 (64-bit)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe).

After installing, make sure `python --version` works in your terminal and returns 3.11.x

If not, you probably need to fix your PATH. Go to:

* Windows + Pause/Break
* Advanced System Settings
* Environment Variables
* Edit your `Path` under User Variables

Example correct entries:

```cmd
C:\Users\YOURNAME\AppData\Local\Programs\Python\Launcher\
C:\Users\YOURNAME\AppData\Local\Programs\Python\Python311\Scripts\
C:\Users\YOURNAME\AppData\Local\Programs\Python\Python311\
```

### Installing Git

Download Git from [git-scm.com/downloads/windows](https://git-scm.com/downloads/windows) and install it. The default installation options are fine.


## Install (Windows, using a Python `venv`)

### Step 1: Download and set up Wan2GP Environment

```cmd
:: Navigate to your desired install directory
cd \your-path-to-wan2gp

:: Clone the repository
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP

:: Create virtual environment
python -m venv wan2gp-env

:: Activate the virtual environment
wan2gp-env\Scripts\activate
```

### Step 2: Install ROCm/PyTorch by TheRock

**IMPORTANT:** Choose the correct index URL for your GPU family!

#### For gfx110X-all (RX 7900 XTX, RX 7800 XT, etc.):

```cmd
pip install --pre torch torchaudio torchvision --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/
```

#### For gfx120X-all (RX 9060, RX 9070, etc.):

```cmd
pip install --pre torch torchaudio torchvision --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/
```

#### For gfx1151 (Strix Halo iGPU):

```cmd
pip install --pre torch torchaudio torchvision --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

#### For gfx1150 (Radeon 890M - Strix Point):

```cmd
pip install --pre torch torchaudio torchvision --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/
```

#### For gfx103X-dgpu (RDNA 2):

```cmd
pip install --pre torch torchaudio torchvision --index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/
```

This will automatically install the latest PyTorch, torchaudio, and torchvision wheels with ROCm support.

### Step 3: Install Wan2GP Dependencies

```cmd
:: Install core dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected output example:
```
PyTorch: 2.11.0+rocm7.12.0
ROCm available: True
Device: AMD Radeon RX 9070 XT
```

## Attention Modes

WanGP supports multiple attention implementations via [triton-windows](https://github.com/woct0rdho/triton-windows/).

First, install `triton-windows` in your virtual environment.  
If you have an older version of Triton installed, uninstall it first:

```cmd
pip uninstall triton
pip install triton-windows
```

### Supported attention implementations

- **Sageattention V1** (Requires the `.post26` wheel or newer):

```cmd
pip install "sageattention <2"
```

- **FlashAttention-2** (Only the Triton backend is supported): 
```cmd
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install ninja
# Install FlashAttention-2 with the Triton backend enabled
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE && python setup.py install
```

- **SDPA** (default): Available by default in PyTorch on post-RDNA3 GPUs.

## Running Wan2GP

For future sessions, activate the environment:

```cmd
cd \path-to\Wan2GP
wan2gp-env\Scripts\activate
python wgp.py
```

## Troubleshooting

### GPU Not Detected

If `torch.cuda.is_available()` returns `False`:

1. **Verify your GPU is supported** - Check the [Supported GPUs](#supported-gpus) list above
2. **Check AMD drivers** - Ensure you have the latest AMD Adrenalin drivers installed
3. **Verify correct index URL** - Make sure you used the right GPU family index URL

### Installation Errors

**"Could not find a version that satisfies the requirement":**
- Double-check that you're using the correct `--index-url` for your GPU family. You can also try adding the `--pre` flag or replacing `/v2/` in the URL with `/v2/staging/`
- Ensure you're using Python 3.11
- Try adding `--pre` flag if not already present

**"No matching distribution found":**
- Your GPU architecture may not be supported
- Check that you've activated your virtual environment

### Performance Issues

- **Monitor VRAM usage** - Reduce batch size if running out of memory
- **Close GPU-intensive apps** - Discord hardware acceleration, browsers, etc.

### Known Issues

Windows packages are new and may be unstable!

Known issues are tracked at: https://github.com/ROCm/TheRock/issues/808

## Additional Resources

- [TheRock GitHub Repository](https://github.com/ROCm/TheRock/)
- [TheRock Releases Documentation](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [Supported GPU Architectures](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [TheRock Roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)
