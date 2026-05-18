# Wan2GP → Apple Silicon (MPS) Migration Analysis

**Date:** 2026-04-26
**Project:** Wan2GP (deepbeepmeep/Wan2GP)
**Platform:** Apple Silicon (M-series) via PyTorch MPS backend

---

## 1. Scale Overview

| Metric | Count |
|--------|-------|
| Total Python files | ~200 |
| Files with `torch.cuda` references | 122 |
| `torch.cuda.*` calls | 413 |
| `device="cuda"` / `.cuda()` strings | 368 |
| CUDA-only package imports | 77 |
| Files with CUDA-only imports | 25 |

**Verdict:** This is a MASSIVE CUDA-dependent codebase. Individual find-and-replace is NOT viable. **The monkey-patch strategy is mandatory.**

---

## 2. CUDA API Classification

### 2.1 Easy Replacements (monkey-patch covers)

| CUDA API | Count | MPS Replacement | Difficulty |
|----------|-------|-----------------|------------|
| `torch.cuda.is_available()` | 137 | `torch.backends.mps.is_available()` | Easy |
| `torch.cuda.empty_cache()` | 67 | `torch.mps.empty_cache()` | Easy |
| `torch.cuda.synchronize()` | 62 | `torch.mps.synchronize()` | Easy |
| `torch.cuda.manual_seed_all()` | 10 | `torch.manual_seed()` | Easy |
| `torch.cuda.set_device()` | 14 | No-op (single device) | Easy |
| `torch.cuda.current_device()` | 16 | No-op (returns 0) | Easy |
| `torch.cuda.device_count()` | 13 | Return 1 | Easy |
| `torch.cuda.manual_seed()` | 7 | `torch.manual_seed()` | Easy |
| `torch.cuda.ipc_collect()` | 16 | No-op (no IPC on MPS) | Easy |
| `torch.cuda.device()` | 3 | context manager, no-op | Easy |
| `torch.cuda.Stream()` | 1 | No MPS equivalent | Skip |
| `torch.cuda.stream()` | 1 | No MPS equivalent | Skip |

### 2.2 Medium Replacements (need logic changes)

| CUDA API | Count | Issue | Solution |
|----------|-------|-------|----------|
| `torch.cuda.get_device_capability()` | 13 | No MPS equivalent | Hardcode (11,0) or detect M-chip |
| `torch.cuda.get_device_properties()` | 9 | VRAM detection fails | Use `sysctl -n hw.memsize` |
| `torch.cuda.amp` (autocast) | 8 | AMP works differently on MPS | Use `torch.autocast("mps")` |
| `torch.cuda.mem_get_info()` | 3 | No MPS equivalent | Estimate from system RAM |
| `torch.cuda.memory_allocated()` | 3 | No MPS equivalent | Skip or return 0 |
| `torch.cuda.memory_reserved()` | 2 | No MPS equivalent | Skip or return 0 |
| `torch.cuda.max_memory_reserved()` | 2 | No MPS equivalent | Skip |
| `torch.cuda.max_memory_allocated()` | 1 | No MPS equivalent | Skip |
| `torch.cuda.is_bf16_supported()` | 1 | M1/M2 don't support BF16 | Check chip version |
| `torch.cuda.Event` | 11 | No MPS equivalent | Use timing fallback |
| `torch.cuda.CUDAGraph` / `graph()` | 7 | CUDA Graphs not on MPS | Disable entirely |
| `torch.cuda.is_current_stream_capturing()` | 3 | Related to CUDA graphs | Always return False |
| `torch.cuda.graph_pool_handle()` | 1 | CUDA graphs only | Disable |
| `torch.cuda.current_stream()` | 1 | No MPS equivalent | Return dummy |
| `torch.cuda.memory_stats()` | 1 | No MPS equivalent | Return empty dict |

### 2.3 Hard Replacements (no MPS equivalent — must skip/fallback)

| Feature | Count | Issue | Solution |
|---------|-------|-------|----------|
| Flash Attention | 24 imports | CUDA kernel only | Fall back to SDPA |
| Sage Attention | 23 imports | CUDA kernel only | Fall back to SDPA |
| Triton | 23 imports | CUDA compiler only | Disable all Triton paths |
| xformers | 6 imports | CUDA only | Fall back to SDPA |
| bitsandbytes | 1 import | CUDA only | Use FP16 |
| Quanto INT8 Triton kernels | 2 files | Triton-dependent | Disable |
| GGUF / Nunchaku / NVFP4 quantizers | 5 files | CUDA kernels only | Disable |
| CUDAGraph integration | 2 files | CUDA only | Disable |
| Multi-GPU / distributed | ~10 files | No MPS multi-GPU | Force single-device |

---

## 3. Top Files by CUDA Density (Priority Order)

| Rank | File | CUDA Refs | Notes |
|------|------|-----------|-------|
| 1 | `models/TTS/index_tts2/infer_v2.py` | 57 | TTS model — NOT core Wan |
| 2 | `models/TTS/index_tts2/accel/accel_engine.py` | 23 | TTS acceleration — NOT core |
| 3 | `shared/llm_engines/nanovllm/engine/model_runner.py` | 20 | vLLM engine — needs MPS fallback |
| 4 | `models/kandinsky5/kandinsky/generation_utils.py` | 17 | Kandinsky model — separate model |
| 5 | `shared/kernels/quanto_int8_triton.py` | 15 | Triton quant kernels — DISABLE |
| 6 | `shared/kernels/quanto_int8_inject.py` | 13 | Quanto INT8 — DISABLE |
| 7 | `preprocessing/matanyone/app.py` | 13 | Segmentation preprocessing |
| 8 | `wgp.py` (main entry) | 12 | **CRITICAL** — needs MPS detection |
| 9 | `shared/qtypes/nvfp4.py` | 12 | CUDA quantization — DISABLE |
| 10 | `shared/qtypes/gguf.py` | 12 | GGUF quantization — DISABLE |
| 11 | `models/hyvideo/modules/placement.py` | 12 | Hunyuan model — separate |
| 12 | `shared/sage2_core.py` | 11 | Sage Attention 2 core — DISABLE |
| 13 | `models/wan/scail/__init__.py` | 10 | Wan scail variant |
| 14 | `models/wan/any2video.py` | 7 | **CORE** — Wan video pipeline |
| 15 | `shared/attention.py` | 7 | **CRITICAL** — attention dispatcher |
| 16 | `models/hyvideo/hunyuan.py` | 10 | Hunyuan model — separate |

---

## 4. Core Wan Model Files (the ones that matter most)

| File | CUDA Refs | Role |
|------|-----------|------|
| `models/wan/wan_handler.py` | 0 | Model handler (clean) |
| `models/wan/any2video.py` | 7 | Main video generation pipeline |
| `models/wan/modules/model.py` | ~5 | Transformer model |
| `models/wan/modules/vae.py` | ~2 | VAE encoder/decoder |
| `models/wan/modules/vae2_2.py` | ~2 | VAE 2.2 |
| `models/wan/modules/t5.py` | 1 | T5 text encoder |
| `shared/attention.py` | 7 | **Attention dispatcher** — MUST patch |
| `shared/sage2_core.py` | 11 | Sage2 attention — must disable |
| `wgp.py` | 12 | **Main entry** — device detection |

---

## 5. Dependency Chain

```
wgp.py (entry)
  ├── mmgp.offload (external library — CRITICAL dependency)
  ├── shared/attention.py (attention mode selection)
  ├── shared/sage2_core.py (Sage Attention 2)
  ├── shared/kernels/quanto_int8_* (quantization)
  ├── shared/qtypes/*.py (quantization types)
  ├── models/wan/*.py (Wan model code)
  ├── models/wan/modules/model.py (transformer)
  ├── models/wan/modules/vae.py (VAE)
  ├── shared/llm_engines/nanovllm/ (vLLM for prompt enhancement)
  └── preprocessing/*.py (depth, canny, pose, etc.)
```

**Critical external dependency:** `mmgp` (offload library, version 3.7.6 required). This library handles model offloading/quantization and likely has its own CUDA dependencies. **This must be checked separately.**

---

## 6. Recommended Migration Phases

### Phase 1: Monkey-Patch Module (1-2 hours)
- Create `shared/mps/device_patch.py` with full CUDA→MPS monkey-patch
- Import at top of `wgp.py` before any other imports
- Covers 80% of `torch.cuda.*` calls automatically

### Phase 2: Module-Level Fixes (2-3 hours)
- `wgp.py` line 2033: `torch.cuda.get_device_capability()` — direct edit needed
- `shared/attention.py` line 9: `torch.cuda.get_device_capability()` — direct edit
- `shared/sage2_core.py` — module-level CUDA calls
- Wrap all CUDA-only imports in try/except

### Phase 3: Core Wan Model (3-4 hours)
- `models/wan/any2video.py` — device strings, amp
- `models/wan/modules/model.py` — attention, device
- `models/wan/modules/vae.py` — device, amp
- Disable quantization paths (quanto, gguf, nvfp4)

### Phase 4: Attention System (2 hours)
- `shared/attention.py` — force SDPA as only mode on MPS
- Disable flash_attn, sageattention, xformers paths
- `shared/sage2_core.py` — skip on MPS

### Phase 5: Preprocessing & Postprocessing (2 hours)
- `preprocessing/` — depth, canny, pose models
- `postprocessing/` — RIFE, MMAudio
- Most use `torch.cuda.is_available()` — monkey-patch handles

### Phase 6: Quantization & vLLM (1-2 hours)
- `shared/qtypes/` — disable all CUDA quant types
- `shared/kernels/` — disable Triton kernels
- `shared/llm_engines/nanovllm/` — disable vLLM on MPS

### Phase 7: Setup & Configuration (1 hour)
- `setup_config.json` — add APPLE_MPS profile
- `setup.py` — Apple Silicon detection
- `requirements.txt` — MPS-compatible deps

### Phase 8: Testing & Validation (ongoing)
- Test model loading
- Test video generation
- Performance profiling
- Memory optimization

---

## 7. Critical Risks

1. **mmgp library**: The offload library (v3.7.6) is external and likely CUDA-specific. If it doesn't support MPS, the entire project is blocked. Must verify first.

2. **Memory constraints**: Wan 14B model requires ~28GB+ VRAM. M-series unified memory (16/32/64/128GB) may or may not be sufficient. May need aggressive offloading or smaller models (1.3B).

3. **BF16 support**: M1/M2 don't support BF16 natively. Must force FP16/FP32.

4. **Performance expectation**: MPS will be significantly slower than CUDA. 14B model inference could take 10-30 minutes per video on M2/M3 Max.

5. **No multi-GPU**: All distributed/multi-GPU code paths must be disabled.

---

## 8. Quick Start Checklist

- [ ] Verify `mmgp` library MPS compatibility
- [ ] Create `shared/mps/device_patch.py` monkey-patch
- [ ] Patch `wgp.py` entry point (line 2033, 2043)
- [ ] Patch `shared/attention.py` (line 9, force SDPA)
- [ ] Wrap CUDA-only imports in try/except
- [ ] Disable quantization (quanto, gguf, nvfp4, nunchaku)
- [ ] Disable Triton kernels
- [ ] Disable vLLM on MPS
- [ ] Add APPLE_MPS profile to `setup_config.json`
- [ ] Test with smallest model (Wan 1.3B t2v)
- [ ] Test with Wan 14B if memory allows

---

## 9. Estimated Total Effort

| Phase | Hours | Confidence |
|-------|-------|------------|
| Phase 1: Monkey-patch | 1-2 | High |
| Phase 2: Module-level fixes | 2-3 | High |
| Phase 3: Core Wan model | 3-4 | Medium |
| Phase 4: Attention system | 2 | High |
| Phase 5: Pre/post processing | 2 | Medium |
| Phase 6: Quantization/vLLM | 1-2 | High |
| Phase 7: Setup/config | 1 | High |
| Phase 8: Testing | 4-8 | Low |
| **Total** | **16-25 hours** | |

**Key blocker:** mmgp library compatibility. If it doesn't support MPS, additional work (or a fork) is needed.
