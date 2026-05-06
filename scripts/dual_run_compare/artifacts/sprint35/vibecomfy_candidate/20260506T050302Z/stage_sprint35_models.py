
from __future__ import annotations

import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

ARTIFACT_ROOT = Path('/workspace/reigh-worker/scripts/dual_run_compare/artifacts/sprint35/vibecomfy_candidate')
RUN_DIR = ARTIFACT_ROOT / sorted([p.name for p in ARTIFACT_ROOT.iterdir() if p.is_dir()])[-1] if ARTIFACT_ROOT.exists() and any(p.is_dir() for p in ARTIFACT_ROOT.iterdir()) else ARTIFACT_ROOT / 'staging'
RUN_DIR.mkdir(parents=True, exist_ok=True)
records = []

def materialize(repo: str, filename: str, targets: list[str], min_size: int) -> None:
    path = Path(hf_hub_download(repo_id=repo, filename=filename)).resolve(strict=True)
    size = path.stat().st_size
    if size < min_size:
        raise RuntimeError(f'{repo}/{filename} resolved to {path} with only {size} bytes')
    for raw in targets:
        target = Path(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(path, target)
            mode = 'hardlink'
        except OSError:
            os.symlink(path, target)
            mode = 'symlink'
        staged = target.resolve(strict=True)
        records.append({'repo': repo, 'filename': filename, 'source': str(path), 'target': str(target), 'mode': mode, 'bytes': staged.stat().st_size})

loras = [
    ('DeepBeepMeep/Wan2.1', 'loras_accelerators/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors'),
    ('DeepBeepMeep/Wan2.1', 'loras_accelerators/DetailEnhancerV1.safetensors'),
    ('DeepBeepMeep/Wan2.1', 'loras_accelerators/Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors'),
    ('DeepBeepMeep/Wan2.1', 'loras_accelerators/Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors'),
]
materialize('Kijai/WanVideo_comfy', 'umt5-xxl-enc-bf16.safetensors', ['models/text_encoders/umt5-xxl-enc-bf16.safetensors'], 1_000_000_000)
materialize('Kijai/WanVideo_comfy', 'Wan2_2_VAE_bf16.safetensors', ['models/vae/wanvideo/Wan2_2_VAE_bf16.safetensors', 'models/vae/wanvideo\\Wan2_2_VAE_bf16.safetensors'], 100_000_000)
materialize('DeepBeepMeep/Wan2.2', 'Wan2_2_Fun_VACE_A14B_HIGH_mbf16.safetensors', ['models/diffusion_models/WanVideo/Wan2_2_Fun_VACE_A14B_HIGH_mbf16.safetensors', 'models/diffusion_models/WanVideo\\Wan2_2_Fun_VACE_A14B_HIGH_mbf16.safetensors'], 1_000_000_000)
materialize('DeepBeepMeep/Wan2.2', 'Wan2_2_Fun_VACE_A14B_LOW_mbf16.safetensors', ['models/diffusion_models/WanVideo/Wan2_2_Fun_VACE_A14B_LOW_mbf16.safetensors', 'models/diffusion_models/WanVideo\\Wan2_2_Fun_VACE_A14B_LOW_mbf16.safetensors'], 1_000_000_000)
for repo, filename in loras:
    name = filename.rsplit('/', 1)[-1]
    materialize(repo, filename, [f'models/loras/WanVideo/loras/{name}', f'models/loras/WanVideo\\loras\\{name}', f'models/loras/{name}'], 10_000_000)
(RUN_DIR / 'model_staging.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps({'staged': len(records), 'record_path': str(RUN_DIR / 'model_staging.json')}))
