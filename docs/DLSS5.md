# DLSS 5 optional runtime installation

> [!CAUTION]
> **Copyright, license, and security warning:** this optional integration executes native Windows binaries with access to your GPU and files; several required runtime components are closed source. WanGP does not audit, authenticate, endorse, or redistribute NVIDIA, ReShade, RenoDX, or other third-party runtime binaries. The WanGP, Merserk, and DLSS5-Feeder MIT licenses do **not** grant rights to copy or redistribute those third-party components. Download only from sources you trust, verify signatures and hashes where available, scan every archive before use, and run it entirely at your own risk. You are responsible for complying with every component's license.

DLSS 5 support is optional and is not installed by the normal WanGP installer. It uses:

- a WanGP depth-aware worker derived from the MIT-licensed [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder);
- a buildable WanGP Frame Generation worker;
- the optional legacy no-depth worker from the MIT-licensed Merserk [dlss5-visual-enhancer](https://github.com/Merserk/dlss5-visual-enhancer);
- separately obtained NVIDIA DLSS runtimes, ReShade with full add-on support, and the RenoDX DLSS 5 add-on.

## Install the worker bundle

Download [WanGP-DLSS5-workers-v1.1.1.zip](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/download/wangp-v1.1.1/WanGP-DLSS5-workers-v1.1.1.zip) from the [DeepBeepMeep dlss5-visual-enhancer fork release](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/tag/wangp-v1.1.1). The ZIP contains the two buildable WanGP workers, the optional legacy Merserk no-depth worker, attribution/license notices, and the required directory structure. It intentionally does **not** contain NVIDIA, ReShade, or RenoDX binaries.

```text
Size:   132132 bytes
SHA256: 6A6288C304251AB0D21C17267F6FA3EE148F706555459859E753A6B1390C47FC
```

Create `WanGP/dlss5`, then extract the **contents** of the ZIP into that folder. Do not extract it into `postprocessing/dlss5`, which contains Python source only. The archive entries begin with `host/`, `dlss/`, and `dlssg/`, so the result must look like this after the separately sourced dependencies are added:

```text
WanGP/
|-- dlss5/
|   |-- host/
|   |   |-- nr-depth-worker.exe       # WanGP depth-aware worker
|   |   |-- nvngx.dll                 # legacy Merserk no-depth worker; optional
|   |   |-- dxgi.dll                  # ReShade full add-on build
|   |   |-- renodx-dlss5.addon64      # RenoDX DLSS 5 add-on
|   |   `-- nvngx_dlssnr.dll          # NVIDIA Neural Rendering runtime
|   |-- dlss/
|   |   `-- nvngx_dlss.dll            # NVIDIA DLSS Super Resolution runtime
|   `-- dlssg/
|       |-- dlssg-worker.exe           # WanGP open D3D12 Frame Generation worker
|       `-- nvngx_dlssg.dll            # NVIDIA Frame Generation runtime
```

Keep every runtime component in the indicated subfolder under the single root `dlss5` folder.

## Obtain the third-party dependencies separately

The worker ZIP is not a complete third-party runtime pack. Obtain and place these files yourself:

| Destination | Component | Authoritative source |
| --- | --- | --- |
| `dlss5/host/dxgi.dll` | ReShade 64-bit **with full add-on support** | [reshade.me](https://reshade.me/) |
| `dlss5/host/renodx-dlss5.addon64` | RenoDX DLSS 5 add-on | [RenoDX releases](https://github.com/clshortfuse/renodx/releases) |
| `dlss5/host/nvngx_dlssnr.dll` | NVIDIA DLSS Neural Rendering | A genuine NVIDIA SDK, driver, or licensed game distribution |
| `dlss5/dlss/nvngx_dlss.dll` | NVIDIA DLSS Super Resolution | [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) or another authorized NVIDIA distribution |
| `dlss5/dlssg/nvngx_dlssg.dll` | NVIDIA DLSS Frame Generation | [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) or another authorized NVIDIA distribution |

Do not download individual DLLs from unofficial mirrors. NVIDIA components are governed by the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt). Review ReShade and RenoDX licensing at their source before copying or redistributing their binaries.

## Build the WanGP workers yourself

Install Visual Studio 2022 C++ build tools and a Windows SDK, then clone NVIDIA's official DLSS SDK repository. From the WanGP root, run:

```powershell
git clone https://github.com/NVIDIA/DLSS C:\temp\NVIDIA-DLSS
powershell -ExecutionPolicy Bypass -File native\dlss5\build.ps1 -NgxSdk C:\temp\NVIDIA-DLSS
```

The script writes `dlss5/host/nr-depth-worker.exe` and `dlss5/dlssg/dlssg-worker.exe`. Pass `-Target nr` or `-Target dlssg` to build only one. See `native/dlss5/LICENSE-DLSS5-Feeder` for attribution.

## Tested component integrity

These hashes identify the exact files tested with the v1.1.1 worker bundle. They do not establish safety or redistribution rights.

| File | SHA-256 | Windows signature |
| --- | --- | --- |
| `dlss/nvngx_dlss.dll` | `C85F971CE023C9F3492FC7455F0B01A24BA18EA39636407A846902C4360B0B7E` | Valid, NVIDIA Corporation |
| `dlssg/nvngx_dlssg.dll` | `135EAF0733C1E37381A8C28ABCF7A862404A54132B81787C04E35D09EFC5E36F` | Valid, NVIDIA Corporation |
| `dlssg/dlssg-worker.exe` | `D93084633E0AAB4A08C43A5EE240176716EF73D87F06F35C2293509FBFC8BD00` | Unsigned, buildable from the fork source |
| `host/dxgi.dll` | `0CEE63F9C9F13F3AC909C5B4903F4DBB4B719A7AB3B4F13B0DEAF83C814B94F7` | Unsigned |
| `host/nvngx.dll` | `58191F4D38288C6BFBDA47EF56911D32052A9789E65714F4583F426E01464638` | Unsigned |
| `host/nvngx_dlssnr.dll` | `6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927` | Unsigned |
| `host/renodx-dlss5.addon64` | `D5ADF82EB44B065F4C590AC91FE824BAB07AFEA0EB9F994BDE936710C8593952` | Unsigned |

`nr-depth-worker.exe` is compiled from the included source and has a release-specific hash in the worker ZIP's `SHA256SUMS.txt`. Prefer rebuilding it yourself when you need source-to-binary assurance.

Before installation, scan the downloaded archives and the extracted directory with current security software. Microsoft Defender reported no detections for the development runtime on 3 September 2026; that result is informational, not a safety guarantee.

## Hardware and diagnostics

Neural Rendering requires Windows 11 and GeForce RTX 30 or newer; RTX 30 is experimental, while RTX 40/50 are the primary targets. Frame Generation requires GeForce RTX 40 or newer, a compatible driver, and Hardware-accelerated GPU scheduling (HAGS). WanGP offers 2x through 4x on compatible RTX 40/50 GPUs and only offers 5x and 6x on RTX 50 GPUs when supported by the installed runtime.

Restart WanGP after installing or replacing the runtime. Unavailable modes are labelled with the missing requirement in their dropdown. For additional Frame Generation diagnostics, run `dlss5/dlssg/dlssg-worker.exe --probe` from the `dlss5/dlssg` directory. Neural Rendering writes diagnostic information to `dlss5/host/ReShade.log`.

## Recorded-video depth and motion guides

Recorded videos do not contain the depth and motion information normally supplied by a game engine. WanGP estimates these guides automatically for DLSS 5 processing.

Open **Config > Extensions > Spatial Upsamplers / Visual Refiners** to configure both DLSS 5 paths:

- **DLSS 5 Depth Resolution Precision**: `Full Res`, `Half Res` (default), or `Quarter Res`. Lower resolutions reduce depth-estimation time and memory use, at a possible cost to fine depth detail.
- **DLSS 5 Motion Vector**: `Original` (default, faster) or `RAFT` (slower, generally better quality). This choice applies to both Neural Rendering and Frame Generation.

The Postprocessing, Late Postprocessing, and Media Flow controls expose **DLSS 5 NR Intensity** from `0.0` through `2.0`, with a default of `1.0`.

Because these guides are estimated from the video, results can differ from DLSS integrated directly into a game engine.
