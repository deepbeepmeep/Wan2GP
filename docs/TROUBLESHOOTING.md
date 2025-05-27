# Troubleshooting Guide

This guide covers common issues and their solutions when using WanGP.

## Installation Issues

### PyTorch Installation Problems

#### CUDA Version Mismatch
**Problem**: PyTorch can't detect GPU or CUDA errors
**Solution**: 
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 12.4 (RTX 10XX-40XX)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124

# For CUDA 12.8 (RTX 50XX)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

#### Python Version Issues
**Problem**: Package compatibility errors
**Solution**: Ensure you're using Python 3.10.9
```bash
python --version  # Should show 3.10.9
conda create -n wan2gp python=3.10.9
```

### Dependency Installation Failures

#### Triton Installation (Windows)
**Problem**: `pip install triton-windows` fails
**Solution**:
1. Update pip: `pip install --upgrade pip`
2. Try pre-compiled wheel
3. Fallback to SDPA attention: `python wgp.py --attention sdpa`

#### SageAttention Compilation Issues
**Problem**: SageAttention installation fails
**Solution**:
1. Install Visual Studio Build Tools (Windows)
2. Use pre-compiled wheels when available
3. Fallback to basic attention modes

## Memory Issues

### CUDA Out of Memory

#### During Model Loading
**Problem**: "CUDA out of memory" when loading model
**Solutions**:
```bash
# Use smaller model
python wgp.py --t2v-1-3B

# Enable quantization (usually default)
python wgp.py --quantize-transformer True

# Use memory-efficient profile
python wgp.py --profile 4

# Reduce preloaded model size
python wgp.py --preload 0
```

#### During Video Generation
**Problem**: Memory error during generation
**Solutions**:
1. Reduce frame count (shorter videos)
2. Lower resolution in advanced settings
3. Use lower batch size
4. Clear GPU cache between generations

### System RAM Issues

#### High RAM Usage
**Problem**: System runs out of RAM
**Solutions**:
```bash
# Limit reserved memory
python wgp.py --perc-reserved-mem-max 0.3

# Use minimal RAM profile
python wgp.py --profile 5

# Enable swap file (OS level)
```

## Performance Issues

### Slow Generation Speed

#### General Optimization
```bash
# Enable compilation (requires Triton)
python wgp.py --compile

# Use faster attention
python wgp.py --attention sage2

# Enable TeaCache
python wgp.py --teacache 2.0

# Use high-performance profile
python wgp.py --profile 3
```

#### GPU-Specific Optimizations

**RTX 10XX/20XX Series**:
```bash
python wgp.py --attention sdpa --profile 4 --teacache 1.5
```

**RTX 30XX/40XX Series**:
```bash
python wgp.py --compile --attention sage --profile 3 --teacache 2.0
```

**RTX 50XX Series**:
```bash
python wgp.py --attention sage --profile 4 --fp16
```

### Attention Mechanism Issues

#### Sage Attention Not Working
**Problem**: Sage attention fails to compile or work
**Diagnostic Steps**:
1. Check Triton installation:
   ```python
   import triton
   print(triton.__version__)
   ```
2. Clear Triton cache:
   ```bash
   # Windows
   rmdir /s %USERPROFILE%\.triton
   # Linux
   rm -rf ~/.triton
   ```
3. Fallback solution:
   ```bash
   python wgp.py --attention sdpa
   ```

#### Flash Attention Issues
**Problem**: Flash attention compilation fails
**Solution**: 
- Windows: Often requires manual CUDA kernel compilation
- Linux: Usually works with `pip install flash-attn`
- Fallback: Use Sage or SDPA attention

## Model-Specific Issues

### Lora Problems

#### Loras Not Loading
**Problem**: Loras don't appear in the interface
**Solutions**:
1. Check file format (should be .safetensors, .pt, or .pth)
2. Verify correct directory:
   ```
   loras/          # For t2v models
   loras_i2v/      # For i2v models
   loras_hunyuan/  # For Hunyuan models
   ```
3. Click "Refresh" button in interface
4. Use `--check-loras` to filter incompatible files

#### Lora Compatibility Issues
**Problem**: Lora causes errors or poor results
**Solutions**:
1. Check model size compatibility (1.3B vs 14B)
2. Verify lora was trained for your model type
3. Try different multiplier values
4. Use `--check-loras` flag to auto-filter

### VACE-Specific Issues

#### Poor VACE Results
**Problem**: VACE generates poor quality or unexpected results
**Solutions**:
1. Enable Skip Layer Guidance
2. Use detailed prompts describing all elements
3. Ensure proper mask creation with Matanyone
4. Check reference image quality
5. Use at least 15 steps, preferably 30+

#### Matanyone Tool Issues
**Problem**: Mask creation difficulties
**Solutions**:
1. Use negative point prompts to refine selection
2. Create multiple sub-masks and combine them
3. Try different background removal options
4. Ensure sufficient contrast in source video

## Network and Server Issues

### Gradio Interface Problems

#### Port Already in Use
**Problem**: "Port 7860 is already in use"
**Solution**:
```bash
# Use different port
python wgp.py --server-port 7861

# Or kill existing process
# Windows
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Linux
lsof -i :7860
kill <PID>
```

#### Interface Not Loading
**Problem**: Browser shows "connection refused"
**Solutions**:
1. Check if server started successfully
2. Try `http://127.0.0.1:7860` instead of `localhost:7860`
3. Disable firewall temporarily
4. Use `--listen` flag for network access

### Remote Access Issues

#### Sharing Not Working
**Problem**: `--share` flag doesn't create public URL
**Solutions**:
1. Check internet connection
2. Try different network
3. Use `--listen` with port forwarding
4. Check firewall settings

## Quality Issues

### Poor Video Quality

#### General Quality Improvements
1. Increase number of steps (25-30+)
2. Use larger models (14B instead of 1.3B)
3. Enable Skip Layer Guidance
4. Improve prompt descriptions
5. Use higher resolution settings

#### Specific Quality Issues

**Blurry Videos**:
- Increase steps
- Check source image quality (i2v)
- Reduce TeaCache multiplier
- Use higher guidance scale

**Inconsistent Motion**:
- Use longer overlap in sliding windows
- Reduce window size
- Improve prompt consistency
- Check control video quality (VACE)

**Color Issues**:
- Check model compatibility
- Adjust guidance scale
- Verify input image color space
- Try different VAE settings

## Advanced Debugging

### Enable Verbose Output
```bash
# Maximum verbosity
python wgp.py --verbose 2

# Check lora compatibility
python wgp.py --check-loras --verbose 2
```

### Memory Debugging
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Reduce memory usage
python wgp.py --profile 4 --perc-reserved-mem-max 0.2
```

### Performance Profiling
```bash
# Test different configurations
python wgp.py --attention sdpa --profile 4  # Baseline
python wgp.py --attention sage --profile 3  # Performance
python wgp.py --compile --teacache 2.0      # Maximum speed
```

## Getting Help

### Before Asking for Help
1. Check this troubleshooting guide
2. Read the relevant documentation:
   - [Installation Guide](INSTALLATION.md)
   - [Getting Started](GETTING_STARTED.md)
   - [Command Line Reference](CLI.md)
3. Try basic fallback configuration:
   ```bash
   python wgp.py --attention sdpa --profile 4
   ```

### Community Support
- **Discord Server**: https://discord.gg/g7efUW9jGV
- Provide relevant information:
  - GPU model and VRAM amount
  - Python and PyTorch versions
  - Complete error messages
  - Command used to launch WanGP
  - Operating system

### Reporting Bugs
When reporting issues:
1. Include system specifications
2. Provide complete error logs
3. List the exact steps to reproduce
4. Mention any modifications to default settings
5. Include command line arguments used

## Emergency Fallback

If nothing works, try this minimal configuration:
```bash
# Absolute minimum setup
python wgp.py --t2v-1-3B --attention sdpa --profile 4 --teacache 0 --fp16

# If that fails, check basic PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
``` 