# Command Line Reference

This document covers current command line options for WanGP. Deprecated Wan model-selection shortcuts remain accepted for existing launch scripts, but are omitted from command help.

## Basic Usage

```bash
# Default launch
python wgp.py

```

## Deepy: Gradio, CLI or Web

Choose one access mode per launch, using your saved Deepy configuration:

```bash
# Gradio at / and the synchronized mobile Web app at /deepy/
python wgp.py --listen

# Interactive terminal chat, without Gradio
python wgp.py --ask-deepy

# Standalone browser app, accessible on the local network
python wgp.py --deepy-server --listen --server-port 7860
```

Gradio includes the Web app at `http://<PC-address>:7860/deepy/`; standalone Web serves it at `http://<PC-address>:7860/`. In a Gradio launch, both interfaces share chat, galleries, selections, settings and active work. Chat and generations continue with every browser closed. Authentication is disabled by default; `--auth` enables a shared password login for Gradio and Deepy. On smartphones the app uses native keyboard dictation. Separate launches do not attach to another process; use saved sessions when moving between processes.

Common Deepy launch options:

| Option | Purpose |
|---|---|
| `--config FOLDER` | Configuration folder containing `wgp_config.json` |
| `--deepy-sessions-dir FOLDER` | Persistent-session location; see [session storage](#deepy-session-location) |
| `--output-dir FOLDER` | Override image, video and audio output folders in CLI or Web mode |
| `--deepy-voice-language CODE` | Overrides the configured microphone transcription language in Gradio or Web; `fr`, `en`, etc., or `auto` for detection |
| `--debug-deepy FOLDER` | Deepy debug logs |
| `--llm-io FOLDER` | LLM input/output transcripts; see [LLM I/O Transcript](#llm-io-transcript) |

Shared network protection options:

| Option | Purpose |
|---|---|
| `--auth` | Enable password-only login for Gradio and Deepy; generate a password when none is supplied |
| `--no-auth` | Explicitly disable web authentication; this is the default |
| `--auth-password PASSPHRASE` | Fixed web passphrase; requires `--auth`. Environment alternative: `WANGP_AUTH_PASSWORD` |
| `--mcp-auth` | Enable separate OAuth authorization for network MCP |
| `--mcp-auth-password PASSPHRASE` | Fixed MCP approval passphrase; requires `--mcp-auth`. Environment alternative: `WANGP_MCP_AUTH_PASSWORD` |
| `--mcp-auth-url ORIGIN` | Public MCP server origin, such as `https://wangp.example.com:7866`; required with `--mcp-auth` |
| `--ssl-certfile FILE`, `--ssl-keyfile FILE` | Certificate and private key for any web/HTTP MCP launch; environment alternatives: `WANGP_SSL_CERT`, `WANGP_SSL_KEY` |
| `--https-port PORT` | Serve HTTPS on this port and redirect the main HTTP port; requires the certificate and key |

The authentication and certificate flags no longer use Deepy-specific names. Authentication remains off unless enabled. See [MCP OAuth setup](API.md#mcp-authentication-and-https) for external clients.

See the [Deepy guide](DEEPY.md) for initial configuration, [interactive CLI commands](DEEPY.md#cli-mode), and Web network, HTTPS, and authentication setup.

## CLI Queue Processing (Headless Mode)

Process saved queues without launching the web UI. Useful for batch processing or automated workflows.

### Quick Start
```bash
# Process a saved queue (ZIP with attachments)
python wgp.py --process my_queue.zip

# Process a settings file (JSON)
python wgp.py --process my_settings.json

# Validate without generating (dry-run)
python wgp.py --process my_queue.zip --dry-run

# Process with custom output directory
python wgp.py --process my_queue.zip --output-dir ./batch_outputs
```

### Supported File Formats
| Format | Description |
|--------|-------------|
| `.zip` | Full queue with embedded attachments (images, videos, audio). Created via "Save Queue" button. |
| `.json` | Settings file only. Media paths are used as-is (absolute or relative to WanGP folder). Created via "Export Settings" button. |

### Workflow
1. **Create your queue** in the web UI using the normal interface
2. **Save the queue** using the "Save Queue" button (creates a .zip file)
3. **Close the web UI** if desired
4. **Process the queue** via command line:
   ```bash
   python wgp.py --process saved_queue.zip --output-dir ./my_outputs
   ```

### CLI Queue Options
```bash
--process PATH          # Path to queue (.zip) or settings (.json) file (enables headless mode)
--dry-run               # Validate file without generating (use with --process)
--output-dir PATH       # Override output directory (use with --process)
--verbose LEVEL         # Verbosity level 0-2 for detailed logging
```

### Console Output
The CLI mode provides real-time feedback:
```
WanGP CLI Mode - Processing queue: my_queue.zip
Output directory: ./batch_outputs
Loaded 3 task(s)

[Task 1/3] A beautiful sunset over the ocean...
  [12/30] Prompt 1/3 - Denoising | Phase 2/2 Low Noise
  Video saved
  Task 1 completed

[Task 2/3] A cat playing with yarn...
  [30/30] Prompt 2/3 - VAE Decoding
  Video saved
  Task 2 completed

==================================================
Queue completed: 3/3 tasks in 5m 23s
```

### Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Success (all tasks completed) |
| 1 | Error (file not found, invalid queue, or task failures) |
| 130 | Interrupted by user (Ctrl+C) |

## MCP Server

```bash
--mcp                              # Start WanGP as an MCP server without the web UI
--mcp-transport TRANSPORT          # stdio, sse, or streamable-http
--mcp-host HOST                    # Host for HTTP transports
--mcp-port PORT                    # Port for HTTP transports
--mcp-api-version {1,2}            # WanGP API contract; latest (2) by default, 1 for compatibility
--mcp-async                        # Permit wait=false in v2; disabled by default
--mcp-console-output               # Mirror WanGP output while serving MCP
--mcp-allow-read-file-system       # Allow agents to submit arbitrary server file paths (disabled by default)
```

Media IDs returned by the Gallery remain usable when filesystem reads are disabled. Streamable HTTP and SSE servers also expose short-lived Gallery upload/download URLs; stdio does not provide HTTP media transfer.

The default MCP interface is now v2. Existing clients that use historical tool names or parameters must launch with `python wgp.py --mcp --mcp-api-version 1`. Use `--mcp-api-version 2` to pin v2 explicitly. Both `--mcp-api-version` and `--mcp-async` also work with `python -m shared.mcp_server`. The number selects the WanGP tool API, not the MCP protocol version; unsupported numbers are rejected. API v1 retains its original wait behavior. In v2, generation and post-processing wait by default; `--mcp-async` enables optional asynchronous calls without changing that default. See [API migration](API.md#mcp-api-v2-and-migration).

### Examples
```bash
# Overnight batch processing
python wgp.py --process overnight_jobs.zip --output-dir ./renders

# Quick validation before long run
python wgp.py --process big_queue.zip --dry-run

# Verbose mode for debugging
python wgp.py --process my_queue.zip --verbose 2

# Combined with other options
python wgp.py --process queue.zip --output-dir ./out --attention sage2
```

## LLM I/O Transcript

```bash
--llm-io FOLDER                     # Record local and remote LLM traffic as a readable plain-text transcript
```

WanGP creates a timestamped `.log` file in the supplied folder. Each record is labelled `[OUT → LLM]` or `[IN ← LLM]` and identifies the engine and stream. Text is preserved verbatim; known special tokens include their names and numeric IDs, while other token streams use numeric IDs. Binary media is described by its source, type, dimensions, and size instead of being written as encoded data.

The transcript can contain prompts, conversation history, tool arguments/results, and model output. Enable it only while diagnosing a problem and treat the resulting file as private data.

## Deepy Session Location

```bash
--deepy-sessions-dir FOLDER          # Override the persistent Deepy sessions folder
```

The default is `deepy_sessions` in the WanGP installation root. The folder is created just in time, only after multi-session mode is enabled and Deepy receives its first request (or when a session archive is explicitly imported).

Each materialized session keeps its canonical decoder context in `context.json` and an append-only `cards.jsonl` journal of consolidated Web-client commands. Streaming token fragments are not written: a thought, response section, or tool card is journaled once it reaches a safe completed boundary, then those commands are replayed when the session is resumed. During replay, the Web client keeps the transcript hidden and suppresses per-command layout, disclosure, animation, and scroll work before one final refresh.

## Model and Performance Options

### Model Configuration
```bash
--quantize-transformer BOOL   # Enable/disable transformer quantization (default: True)
--compile                     # Enable PyTorch compilation (requires Triton)
--attention MODE              # Force attention mode: sdpa, flash, sage, sage2
--profile NUMBER              # Performance profile 1-5 (default: 4)
--preload NUMBER              # Preload N MB of diffusion model in VRAM
--fp16                        # Force fp16 instead of bf16 models
--save-quantized              # Save an INT8 Quanto checkpoint during model loading
--convrot                     # With --save-quantized, save INT8 ConvRot instead of Quanto INT8
--gpu DEVICE                  # Run on specific GPU device (e.g., "cuda:1")
```

### Performance Profiles
- **Profile 1**: Load entire current model in VRAM and keep all unused models in reserved RAM for fast VRAM tranfers 
- **Profile 2**: Load model parts as needed, keep all unused models in reserved RAM for fast VRAM tranfers
- **Profile 3**: Load entire current model in VRAM (requires 24GB for 14B model)
- **Profile 4**: Default and recommended, load model parts as needed, most flexible option
- **Profile 4+** (4.5): Profile 4 variation, can save up to 1 GB of VRAM, but will be slighlty slower on some configs
- **Profile 5**: Minimum RAM usage

### Memory Management
```bash
--perc-reserved-mem-max FLOAT # Max percentage of RAM for reserved memory (< 0.5)
```

## Lora Configuration

```bash
--loras PATH                 # Root folder for default LoRA subfolders (default: loras)
--lora-config FILE           # Optional JSON mapping LoRA subfolder names to paths
--lora-preset PRESET         # Load lora preset file (.lset) on startup
--check-loras                # Filter incompatible loras (slower startup)
```

Use `python wgp.py --lora-config lora_paths.json` to override individual collections. JSON keys match the exact subfolder names in the default `loras/` folder, such as `wan`, `wan_5B`, or `flux2_klein_4b`. Values are complete directory paths; relative values resolve beside the JSON file.

JSON entries take precedence over `--loras`. Unlisted keys use the root from `--loras`, then `wgp_config.json`'s `loras_root`, then `loras/`. The former model-specific `--lora-dir*` flags have been removed. See the [LoRA guide](LORAS.md#custom-lora-directories) for a sample JSON and setup instructions.

## Generation Settings

### Basic Generation
```bash
--seed NUMBER                # Set default seed value
--frames NUMBER              # Set default number of frames to generate
--steps NUMBER               # Set default number of denoising steps
--advanced                   # Launch with advanced mode enabled
```

### Advanced Generation
```bash
--teacache MULTIPLIER        # TeaCache speed multiplier: 0, 1.5, 1.75, 2.0, 2.25, 2.5
```

## Interface and Server Options

### Server Configuration
```bash
--server-port PORT           # Gradio server port (default: 7860)
--server-name NAME           # Gradio server name (default: localhost)
--listen                     # Make server accessible on network
--share                      # Create shareable HuggingFace URL for remote access
--open-browser               # Open browser automatically when launching
```

### Interface Options
```bash
--lock-config                # Prevent modifying video engine configuration from interface
--theme THEME_NAME           # UI theme: "default" or "gradio"
```

## File and Directory Options

```bash
--settings PATH              # Path to folder containing default settings for all models
--config PATH                # Config folder for wgp_config.json and queue.zip
--workspaces-dir FOLDER       # Gallery workspaces for Gradio/Web (default: ./workspaces)
--verbose LEVEL              # Information level 0-2 (default: 1)
```

## Examples

### Basic Usage Examples
```bash
# Launch with specific model and loras
python wgp.py ----lora-preset mystyle.lset

# High-performance setup with compilation
python wgp.py --compile --attention sage2 --profile 3

# Low VRAM setup
python wgp.py --profile 4 --attention sdpa
```

### Server Configuration Examples
```bash
# Network accessible server
python wgp.py --listen --server-port 8080

# Shareable server with custom theme
python wgp.py --share --theme gradio --open-browser

# Locked configuration for public use
python wgp.py --lock-config --share
```

### Advanced Performance Examples
```bash
# Maximum performance (requires high-end GPU)
python wgp.py --compile --attention sage2 --profile 3 --preload 2000

# Optimized for RTX 2080Ti
python wgp.py --profile 4 --attention sdpa --teacache 2.0

# Memory-efficient setup
python wgp.py --fp16 --profile 4 --perc-reserved-mem-max 0.3
```

### TeaCache Configuration
```bash
# Different speed multipliers
python wgp.py --teacache 1.5   # 1.5x speed, minimal quality loss
python wgp.py --teacache 2.0   # 2x speed, some quality loss
python wgp.py --teacache 2.5   # 2.5x speed, noticeable quality loss
python wgp.py --teacache 0     # Disable TeaCache
```

## Attention Modes

### SDPA (Default)
```bash
python wgp.py --attention sdpa
```
- Available by default with PyTorch
- Good compatibility with all GPUs
- Moderate performance

### Sage Attention
```bash
python wgp.py --attention sage
```
- Requires Triton installation
- On RTX 20XX, install SageAttention 1.0.6
- 30% faster than SDPA
- Small quality cost

### Sage2 Attention
```bash
python wgp.py --attention sage2
```
- Requires Triton and SageAttention 2.x
- Requires RTX 30XX or newer (Ampere or newer)
- 40% faster than SDPA
- Best performance option

### Flash Attention
```bash
python wgp.py --attention flash
```
- May require CUDA kernel compilation
- Good performance
- Can be complex to install on Windows

## Troubleshooting Command Lines

### Fallback to Basic Setup
```bash
# If advanced features don't work
python wgp.py --attention sdpa --profile 4 --fp16
```

### Debug Mode
```bash
# Maximum verbosity for troubleshooting
python wgp.py --verbose 2 --check-loras
```

### Memory Issue Debugging
```bash
# Minimal memory usage
python wgp.py --profile 4 --attention sdpa --perc-reserved-mem-max 0.2
```



## Configuration Files

### Settings Files
Load custom settings:
```bash
python wgp.py --settings /path/to/settings/folder
```

### Config Folder
Use a separate folder for the UI config and autosaved queue:
```bash
python wgp.py --config /path/to/config
```
If missing, `wgp_config.json` or `queue.zip` are loaded once from the WanGP root and then written to the config folder.

### Lora Presets
Create and share lora configurations:
```bash
# Load specific preset
python wgp.py --lora-preset anime_style.lset

# With custom lora root
python wgp.py --loras /shared/loras --lora-preset mystyle.lset
```

## Environment Variables

While not command line options, these environment variables can affect behavior:
- `CUDA_VISIBLE_DEVICES` - Limit visible GPUs
- `PYTORCH_CUDA_ALLOC_CONF` - CUDA memory allocation settings
- `TRITON_CACHE_DIR` - Triton cache directory (for Sage attention) 
- `WAN2GP_DEEPY_TELEMETRY=1` - Enable detailed Deepy decode, MTP, CUDA-memory, and GPU telemetry when verbose level 2 is active (disabled by default)

---

> Applies to: WanGP startup options and saved-queue processing from a shell. Command-line flags configure the application process; generation settings and API tool arguments are specified separately.
