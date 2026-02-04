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

### **gfx120X-all** (RDNA 4):
* AMD RX 9060 XT (gfx1200)
* AMD RX 9060 (gfx1200)
* AMD RX 9070 XT (gfx1201)
* AMD RX 9070 (gfx1201)

### **gfx1151** (RDNA 3.5 APU):
* AMD Strix Halo APUs

### **gfx1150** (RDNA 3.5 APU) 
* AMD Radeon 890M (Ryzen AI 9 HX 370 - Strix Point)

### Also supported:
### **gfx103X-dgpu**: (RDNA 2)

<br>

> **Note:** If your GPU is not listed above, it is not supported by TheRock on Windows. Support status and future updates can be found in the [official documentation](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md).

## Requirements

- Python 3.11 (recommended for Wan2GP - TheRock currently supports Python 3.11, 3.12, and 3.13).
- Windows 10/11

## Installation Environment

This installation uses PyTorch wheels built by TheRock.

### Installing Python

Download Python 3.11 from [python.org/downloads/windows](https://www.python.org/downloads/windows/). Press Ctrl+F and search for "3.11." to find the newest version available for installation.

Alternatively, you can use this direct link: [Python 3.11.9 (64-bit)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe).

After installing, make sure `python --version` works in your terminal and returns `3.11.9`

If it doesn’t, you need to add Python to your PATH:

* Press the `Windows` key, type `Environment Variables`, and select `Edit the system environment variables`.
* In the `System Properties` window, click `Environment Variables…`.
* Under `User variables`, find `Path`, then click `Edit` → `New` and add the following entries (replace `<username>` with your Windows username):

```cmd
C:\Users\<username>\AppData\Local\Programs\Python\Launcher\
C:\Users\<username>\AppData\Local\Programs\Python\Python311\Scripts\
C:\Users\<username>\AppData\Local\Programs\Python\Python311\
```

> **Note:** If Python still doesn't show the correct version after updating PATH, try signing out and signing back in to Windows to apply the changes.

### Installing Git

Download Git from [git-scm.com/downloads/windows](https://git-scm.com/install/windows) and install it. The default installation options are fine.


## Installation Steps (Windows, using a Python `venv`)
> **Note:** The following commands are intended for use in the Windows Command Prompt (CMD).  
> If you are using PowerShell, some commands (like comments and activating the virtual environment) may differ.


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

> **Note:** If you have multiple versions of Python installed, use `py -3.11 -m venv wan2gp-env` instead of `python -m venv wan2gp-env` to ensure the correct version is used.

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

- **SageAttention V1** (Requires the `.post26` wheel or newer):

```cmd
pip install "sageattention <2"
```

- **FlashAttention-2** (Only the Triton backend is supported): 
```cmd
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install ninja
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE && python setup.py install
```

- **SDPA Flash**: Available by default in PyTorch on post-RDNA2 GPUs via AOTriton.

## Running Wan2GP

For future sessions, activate the environment every time if it isn't already activated, then run `python wgp.py`:

```cmd
cd \path-to\Wan2GP
wan2gp-env\Scripts\activate
python wgp.py
```

It is advised to set the following environment variables at the start of every new session (you can create a `.bat` file that activates your venv, sets these, then launches `wgp.py`):

```cmd
set ROCM_HOME=%ROCM_ROOT%
set PATH=%ROCM_ROOT%\lib\llvm\bin;%ROCM_BIN%;%PATH%
set CC=clang-cl
set CXX=clang-cl
set DISTUTILS_USE_SDK=1
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

MIOpen (AMD's cuDNN equivalent) is not yet stable; it frequently causes OOMs, crashes the display driver, and significantly increases generation times. Currently, it is recommended to use fast mode by setting `set MIOPEN_FIND_MODE=FAST`, or to disable it entirely by editing `wgp.py` and adding the following line below `import torch` (line 51):

```cmd
torch.backends.cudnn.enabled = False
```

To verify that it is disabled, or to enable verbose logging, you can set:

```cmd
set MIOPEN_ENABLE_LOGGING=1
set MIOPEN_ENABLE_LOGGING_CMD=1
set MIOPEN_LOG_LEVEL=5
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
- Ensure you're using Python 3.11, and not 3.10
- Try adding `--pre` flag if not already present

**"No matching distribution found":**
- Your GPU architecture may not be supported
- Check that you've activated your virtual environment

### Performance Issues

- **Monitor VRAM usage** - Reduce batch size or resolution if running out of memory
- **Close GPU-intensive apps** - Apps with hardware acceleration enabled (browsers, Discord etc.).

### Known Issues

Windows packages are new and may be unstable.

Known issues are tracked at: https://github.com/ROCm/TheRock/issues/808

## Additional Resources

- [TheRock GitHub Repository](https://github.com/ROCm/TheRock/)
- [Releases Documentation](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [Supported GPU Architectures](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [Roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)
