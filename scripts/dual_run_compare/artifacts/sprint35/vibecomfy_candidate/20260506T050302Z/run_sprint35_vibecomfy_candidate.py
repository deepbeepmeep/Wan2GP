
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path('/workspace')
VIBECOMFY_ROOT = REPO_ROOT / 'vibecomfy'
FIXTURE_ROOT = REPO_ROOT / 'reigh-worker/scripts/dual_run_compare/fixtures/sprint35'
ARTIFACT_ROOT = REPO_ROOT / 'reigh-worker/scripts/dual_run_compare/artifacts/sprint35/vibecomfy_candidate'
RUN_LABEL = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_DIR = ARTIFACT_ROOT / RUN_LABEL
OUTPUT_DIR = RUN_DIR / 'outputs'
RUN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RUN_DIR / 'vibecomfy_candidate.log'


def log(message: str) -> None:
    line = f'[{datetime.now(timezone.utc).isoformat()}] {message}'
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    proc = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration',
        '-of', 'json', str(path),
    ], text=True, capture_output=True)
    return {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr, 'parsed': json.loads(proc.stdout or '{}') if proc.stdout else {}}


def candidate_outputs(result_outputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in result_outputs:
        p = Path(item)
        if not p.is_absolute():
            p = VIBECOMFY_ROOT / p
        if p.exists() and p.is_file():
            paths.append(p)
    for root in [VIBECOMFY_ROOT / 'output', VIBECOMFY_ROOT / 'out/runs']:
        if root.exists():
            paths.extend(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4', '.webm', '.png', '.webp'})
    dedup: list[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup


def main() -> int:
    manifest = json.loads((FIXTURE_ROOT / 'manifest.json').read_text())
    input_target = VIBECOMFY_ROOT / 'input/sprint35'
    if input_target.exists():
        shutil.rmtree(input_target)
    shutil.copytree(FIXTURE_ROOT / 'inputs', input_target / 'inputs')
    os.chdir(VIBECOMFY_ROOT)
    os.environ['VIBECOMFY_COMFY_CONFIGURATION'] = json.dumps({
        'preview_method': 'none',
        'cache_none': True,
        'reserve_vram': 2,
    })
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    log('starting VibeComfy embedded run')
    started = time.time()
    status = 'failed'
    result_info = {}
    output_records = []
    selected_video = None
    try:
        from vibecomfy.registry.ready import workflow_from_ready
        from vibecomfy.runtime.run import run_embedded_sync

        workflow = workflow_from_ready('video/wanvideo_wrapper_22_14b_vace_cocktail_dry_run')
        result = run_embedded_sync(workflow, backend='api')
        elapsed = time.time() - started
        log(f'run returned run_id={result.run_id} prompt_id={result.prompt_id} elapsed={elapsed:.2f}')
        result_info = {
            'run_id': result.run_id,
            'prompt_id': result.prompt_id,
            'outputs': result.outputs,
            'metadata_path': result.metadata_path,
            'log_path': result.log_path,
            'wall_clock_seconds': elapsed,
        }
        outputs = candidate_outputs(result.outputs)
        for source in outputs:
            dest = OUTPUT_DIR / source.name
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            record = {
                'source_path': str(source),
                'artifact_path': str(dest),
                'artifact_relative_to_repo': str(dest.relative_to(REPO_ROOT)),
                'bytes': dest.stat().st_size,
                'sha256': sha256(dest),
            }
            if dest.suffix.lower() in {'.mp4', '.webm'}:
                record['ffprobe'] = ffprobe(dest)
            output_records.append(record)
        videos = [Path(r['artifact_path']) for r in output_records if Path(r['artifact_path']).suffix.lower() in {'.mp4', '.webm'}]
        if not videos:
            raise RuntimeError(f'No video output found; result outputs={result.outputs!r}')
        selected_video = max(videos, key=lambda p: p.stat().st_mtime)
        status = 'success'
    except Exception as exc:
        elapsed = time.time() - started
        log('candidate failed: ' + repr(exc))
        (RUN_DIR / 'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result_info.setdefault('wall_clock_seconds', elapsed)
    metadata = {
        'status': status,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'template_id': 'video/wanvideo_wrapper_22_14b_vace_cocktail_dry_run',
        'template_path': 'vibecomfy/ready_templates/video/wanvideo_wrapper_22_14b_vace_cocktail_dry_run.py',
        'route_key': manifest.get('route_key'),
        'threshold_route_key': manifest.get('threshold_route_key'),
        'fixture_manifest_path': 'reigh-worker/scripts/dual_run_compare/fixtures/sprint35/manifest.json',
        'model': manifest.get('model_name'),
        'seed': manifest.get('seed'),
        'resolution': f"{manifest.get('width')}x{manifest.get('height')}",
        'fps': manifest.get('fps'),
        'num_frames': manifest.get('num_frames'),
        'result': result_info,
        'outputs': output_records,
        'selected_video': str(selected_video) if selected_video else None,
        'selected_video_relative_to_repo': str(selected_video.relative_to(REPO_ROOT)) if selected_video else None,
        'model_staging_notes': json.loads((RUN_DIR / 'model_staging.json').read_text()) if (RUN_DIR / 'model_staging.json').exists() else [],
    }
    (RUN_DIR / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    log(f'finished status={status}')
    return 0 if status == 'success' else 1

if __name__ == '__main__':
    raise SystemExit(main())
