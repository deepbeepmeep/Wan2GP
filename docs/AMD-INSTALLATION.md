# Installation Guide

This guide covers installation for specific RDNA3 and RDNA3.5 AMD CPUs (APUs) and GPUs
running under Windows. 

tl;dr: Radeon RX 7900 GOOD, RX 9700 BAD, RX 6800 BAD. (I know, life isn't fair).

Currently supported (but not necessary tested):

**gfx110x**:

* Radeon RX 7600
* Radeon RX 7700 XT
* Radeon RX 7800 XT
* Radeon RX 7900 GRE
* Radeon RX 7900 XT
* Radeon RX 7900 XTX

**gfx1151**:

* Ryzen 7000 series APUs (Phoenix)
* Ryzen Z1 (e.g., handheld devices like the ROG Ally)

**gfx1201**:

* Ryzen 8000 series APUs (Strix Point) 
* A [frame.work](https://frame.work/au/en/desktop) desktop/laptop


## Requirements

- Python 3.11 (3.12 might work, 3.10 definately will not!)

## Installation Environment

This installation uses PyTorch 2.7.0 because that's what currently available in
terms of pre-compiled wheels.

### Installing Python

Download Python 3.11 from [python.org/downloads/windows](https://www.python.org/downloads/windows/). Hit Ctrl+F and search for "3.11". Dont use this direct link: [https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) -- that was an IQ test.

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

If that doesnt work, scream into a bucket.

### Installing Git

Get Git from [git-scm.com/downloads/win](https://git-scm.com/downloads/win). Default install is fine.


## Install (Windows, using `venv`)

### Step 1: Download and Set Up Environment

```cmd
:: Navigate to your desired install directory
cd \your-path-to-wan2gp

:: Clone the repository
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP

:: Create virtual environment using Python 3.10.9
python -m venv wan2gp-env

:: Activate the virtual environment
wan2gp-env\Scripts\activate
```

### Step 2: Install PyTorch

The pre-compiled wheels you need are hosted at [scottt's rocm-TheRock releases](https://github.com/scottt/rocm-TheRock/releases). Find the heading that says:

**Pytorch wheels for gfx110x, gfx1151, and gfx1201**

Don't click this link: [https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch-gfx110x](https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch-gfx110x). It's just here to check if you're skimming.

Copy the links of the closest binaries to the ones in the example below (adjust if you're not running Python 3.11), then hit enter.

```cmd
pip install ^
    https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch-gfx110x/torch-2.7.0a0+rocm_git3f903c3-cp311-cp311-win_amd64.whl ^
    https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch-gfx110x/torchaudio-2.7.0a0+52638ef-cp311-cp311-win_amd64.whl ^
    https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch-gfx110x/torchvision-0.22.0+9eb57cd-cp311-cp311-win_amd64.whl
```

### Step 3: Install Dependencies

```cmd
:: Install core dependencies
pip install -r requirements.txt
```

## Attention Modes

WanGP supports several attention implementations, only one of which will work for you:

- **SDPA** (default): Available by default with PyTorch.  This uses the built-in aotriton accel library, so is actually pretty fast.

## Performance Profiles

Choose a profile based on your hardware:

- **Profile 3 (LowRAM_HighVRAM)**: Loads entire model in VRAM, requires 24GB VRAM for 8-bit quantized 14B model
- **Profile 4 (LowRAM_LowVRAM)**: Default, loads model parts as needed, slower but lower VRAM requirement

## Running Wan2GP

In future, you will have to do this:

```cmd
cd \path-to\wan2gp
wan2gp\Scripts\activate.bat
python wgp.py
```

For now, you should just be able to type `python wgp.py` (because you're already in the virtual environment)

## Troubleshooting

- If you use a HIGH VRAM mode, don't be a fool.  Make sure you use VAE Tiled Decoding.

### Memory Issues

- Use lower resolution or shorter videos
- Enable quantization (default)
- Use Profile 4 for lower VRAM usage
- Consider using 1.3B models instead of 14B models

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 
