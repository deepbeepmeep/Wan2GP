# WanGP Windows launcher

A small desktop front-end that turns WanGP into a normal Windows application:
one `WanGP-Setup-x.y.z.exe` to install, a Start-menu shortcut to run, and the
web UI inside a native window instead of a browser tab.

## Design principle

The launcher owns **no** installation logic. Everything about environments,
GPU detection, CUDA/ROCm profiles and acceleration kernels already lives in the
repository's `setup.py`, and the launcher is a graphical façade over it:

| The launcher needs to… | What it actually runs |
| --- | --- |
| find the active environment | `python setup.py get_env_info` → `ENV_INFO\|type\|path` |
| build the environment | `python setup.py install --env venv --auto` |
| update the code | `python setup.py update` |
| switch environments | `python setup.py manage` (in a real console) |
| upgrade Torch/Triton/Sage | `python setup.py upgrade` (in a real console) |
| start the app | `<env python> wgp.py --server-name 127.0.0.1 --server-port <free port>` |

It also mirrors `scripts/run.bat` where behaviour is observable: extra flags are
read from `scripts/args.txt`, and exit code `42` means "restart me".

Consequence: `setup.py` stays the single source of truth. A new GPU profile or
a new kernel added there is picked up by the launcher with no code change here.

## What the user experiences

1. **Install** — `WanGP-Setup.exe`, per-user, no admin prompt. Defaults to
   `%LOCALAPPDATA%\Programs\WanGP`, and the directory page is left enabled so a
   roomier drive can be chosen (models need tens of GB). The installer adds the
   Microsoft Edge WebView2 Runtime if Windows does not already have it.
2. **First launch** — a progress window with a live console:
   Python 3.11 is installed if missing (per-user, including the `py` launcher),
   a private copy of Git is unpacked if missing, then `setup.py --auto` detects
   the GPU and builds the environment.
3. **Every launch after that** — `wgp.py` starts on a free local port, the
   launcher waits for the port to answer, then swaps the progress page for the
   WanGP interface. A menu bar exposes Update, Repair, environment management
   and the installation / outputs / logs folders.

## Layout

```
launcher/
  main.py              entry point; --selftest verifies a frozen build
  config.py            paths, bundled configuration, logging
  process.py           windowless subprocesses, line streaming, process trees
  prereqs.py           Python 3.11 and Git detection plus unattended install
  environment.py       the setup.py bridge and args.txt handling
  server.py            runs wgp.py, waits for readiness, honours exit code 42
  ui.py                pywebview window, menu bar, JS bridge
  web/setup.html       the progress / console page
  build_assets.py      generates the .ico and the EXE version resource
  wangp.spec           PyInstaller one-folder bundle → launcher-bin/
  installer.iss        Inno Setup script → WanGP-Setup-x.y.z.exe
  tests/               offline tests, no GPU and no pywebview required
```

Runtime state lives outside the installation directory, in
`%LOCALAPPDATA%\WanGP`: `logs/launcher.log`, `logs/wangp-server.log`,
`downloads/` (cached Python and Git installers) and `tools/git` when a private
Git was needed.

## Building

CI does this on every tag `launcher-v*` and on demand via **Actions → Build
Windows installer**; the installer lands as a build artifact and, for a tag, on
the release. To reproduce it locally on a Windows machine:

```powershell
python -m pip install -r launcher\requirements-launcher.txt
python -m unittest launcher.tests.test_launcher -v
python launcher\build_assets.py --version 1.0.0
pyinstaller launcher\wangp.spec --noconfirm --distpath dist --workpath build\pyinstaller

# Payload = the tracked files only, so no caches or environments leak in.
mkdir build\payload
git archive --format=tar HEAD | tar -x -C build\payload

& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" `
    /DMyAppVersion=1.0.0 /DPayloadDir="$PWD\build\payload" `
    /DLauncherDir="$PWD\dist\launcher-bin" /DOutputDir="$PWD\dist" `
    launcher\installer.iss
```

Run it unfrozen during development — same behaviour, no packaging step:

```powershell
python launcher\main.py
```

## Configuration

`launcher_config.json` is bundled into the executable and can be overridden by a
file of the same name placed next to the installed launcher. CI stamps
`app_version` and `repo_url` at build time. Useful keys:

| Key | Purpose |
| --- | --- |
| `repo_url`, `repo_branch` | remote used to attach the install to git for updates |
| `python.installer_url`, `python.installer_sha256` | the Python bootstrap; set the checksum to pin it |
| `git.portable_url`, `git.portable_sha256` | the portable Git bootstrap |
| `env_type` | `venv` (default), `uv`, `conda` or `none` |
| `server.preferred_port` | tried first; a free port is chosen if it is busy |
| `server.startup_timeout_seconds` | how long to wait for the web UI on a cold start |

## Notes and limits

- **Updates.** The installer ships the tracked sources without a `.git`
  directory, so the first update attaches the tree to `repo_url` itself instead
  of relying on `setup.py`'s repair path, which hardcodes the upstream
  repository in its `git remote add origin`. Setting `origin` first also makes
  that path harmless if it ever runs: `git remote add` fails on an existing
  remote and `setup.py` ignores the error, so the fork's remote survives. After
  that, `setup.py update` handles the usual `git pull` plus requirements
  refresh.
- **Uninstalling** never deletes the environment, models or generated media on
  its own: it asks, and keeping them is the default answer.
- **Windows only.** `setup.py` builds its environment with `py -3.11 -m venv`
  on Windows, so the Python launcher is a hard prerequisite; that is why the
  launcher installs Python itself when `py -3.11` does not resolve.
- **Not covered by the offline tests:** the WebView2 window, the Inno Setup
  flow and the unattended Python install. The build workflow runs the frozen
  executable with `--selftest` to catch packaging mistakes, but the installer
  itself still needs a manual pass on a real Windows machine before a release.
