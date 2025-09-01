# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import os
import io
import numpy as np
import torch
import yaml
import matplotlib
import argparse

app = Flask(__name__, static_folder='static', template_folder='templates')


# ——— Arguments ———————————————————————————————————
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='videos_example')
args = parser.parse_args()


# ——— Configuration —————————————————————————————
BASE_DIR = args.save_dir
STATIC_BASE = os.path.join('static', BASE_DIR)
IMAGES_DIR = os.path.join(STATIC_BASE, 'images')
OVERLAY_DIR = os.path.join(STATIC_BASE, 'images_tracks')
TRACKS_DIR = os.path.join(BASE_DIR, 'tracks')
YAML_PATH = os.path.join(BASE_DIR, 'test.yaml')
IMAGES_DIR_OUT = os.path.join(BASE_DIR, 'images')

FIXED_LENGTH = 121
COLOR_CYCLE = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
QUANT_MULTI = 8

for d in (IMAGES_DIR, TRACKS_DIR, OVERLAY_DIR, IMAGES_DIR_OUT):
    os.makedirs(d, exist_ok=True)

# ——— Helpers ———————————————————————————————————————


def array_to_npz_bytes(arr, path, compressed=True, quant_multi=QUANT_MULTI):
    # pack into uint16 as before
    arr_q = (quant_multi * arr).astype(np.float32)
    bio = io.BytesIO()
    if compressed:
        np.savez_compressed(bio, array=arr_q)
    else:
        np.savez(bio, array=arr_q)
    torch.save(bio.getvalue(), path)


def load_existing_tracks(path):
    raw = torch.load(path)
    bio = io.BytesIO(raw)
    with np.load(bio) as npz:
        return npz['array']

# ——— Routes ———————————————————————————————————————


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    f = request.files['image']
    from PIL import Image
    img = Image.open(f.stream)
    orig_w, orig_h = img.size

    idx = len(os.listdir(IMAGES_DIR)) + 1
    ext = f.filename.rsplit('.', 1)[-1]
    fname = f"{idx:02d}.{ext}"
    img.save(os.path.join(IMAGES_DIR, fname))
    img.save(os.path.join(IMAGES_DIR_OUT, fname))

    return jsonify({
        'image_url': f"{STATIC_BASE}/images/{fname}",
        'image_id': idx,
        'ext': ext,
        'orig_width': orig_w,
        'orig_height': orig_h
    })


@app.route('/store_tracks', methods=['POST'])
def store_tracks():
    data = request.get_json()
    image_id = data['image_id']
    ext = data['ext']
    free_tracks = data.get('tracks', [])
    circ_trajs = data.get('circle_trajectories', [])

    # Debug lengths
    for i, tr in enumerate(free_tracks, 1):
        print(f"Freehand Track {i}: {len(tr)} points")
    for i, tr in enumerate(circ_trajs, 1):
        print(f"Circle/Static Traj {i}: {len(tr)} points")

    def pad_pts(tr):
        """Convert list of {x,y} to (FIXED_LENGTH,1,3) array, padding/truncating."""
        pts = np.array([[p['x'], p['y'], 1] for p in tr], dtype=np.float32)
        n = pts.shape[0]
        if n < FIXED_LENGTH:
            pad = np.zeros((FIXED_LENGTH - n, 3), dtype=np.float32)
            pts = np.vstack((pts, pad))
        else:
            pts = pts[:FIXED_LENGTH]
        return pts.reshape(FIXED_LENGTH, 1, 3)

    arrs = []

    # 1) Freehand tracks
    for i, tr in enumerate(free_tracks):
        pts = pad_pts(tr)
        arrs.append(pts,)

    # 2) Circle + Static combined
    for i, tr in enumerate(circ_trajs):
        pts = pad_pts(tr)

        arrs.append(pts)
    print(arrs)
    # Nothing to save?
    if not arrs:
        overlay_file = f"{image_id:02d}.png"
        return jsonify({
            'status': 'ok',
            'overlay_url': f"{STATIC_BASE}/images_tracks/{overlay_file}"
        })

    new_tracks = np.stack(arrs, axis=0)  # (T_new, FIXED_LENGTH,1,4)

    # Load existing .pth and pad old channels to 4 if needed
    track_path = os.path.join(TRACKS_DIR, f"{image_id:02d}.pth")
    if os.path.exists(track_path):
        # shape (T_old, FIXED_LENGTH,1,3) or (...,4)
        old = load_existing_tracks(track_path)
        if old.ndim == 4 and old.shape[-1] == 3:
            pad = np.zeros(
                (old.shape[0], old.shape[1], old.shape[2], 1), dtype=np.float32)
            old = np.concatenate((old, pad), axis=-1)
        all_tracks = np.concatenate([old, new_tracks], axis=0)
    else:
        all_tracks = new_tracks

    # Save updated track file
    array_to_npz_bytes(all_tracks, track_path, compressed=True)

    # Build overlay PNG
    img_path = os.path.join(IMAGES_DIR, f"{image_id:02d}.{ext}")
    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    for t in all_tracks:
        coords = t[:, 0, :]  # (FIXED_LENGTH,4)
        ax.plot(coords[:, 0][coords[:, 2] > 0.5], coords[:, 1]
                [coords[:, 2] > 0.5], marker='o', color=COLOR_CYCLE[0])
    ax.axis('off')
    overlay_file = f"{image_id:02d}.png"
    fig.savefig(os.path.join(OVERLAY_DIR, overlay_file),
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Update YAML (unchanged)
    entry = {
        "image": os.path.join(f"tools/trajectory_editor/{BASE_DIR}/images/{image_id:02d}.{ext}"),
        "text": None,
        "track": os.path.join(f"tools/trajectory_editor/{BASE_DIR}/tracks/{image_id:02d}.pth")
    }
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH) as yf:
            docs = yaml.safe_load(yf) or []
    else:
        docs = []

    for e in docs:
        if e.get("image", "").endswith(f"{image_id:02d}.{ext}"):
            e.update(entry)
            break
    else:
        docs.append(entry)

    with open(YAML_PATH, 'w') as yf:
        yaml.dump(docs, yf, default_flow_style=False)

    return jsonify({
        'status': 'ok',
        'overlay_url': f"{STATIC_BASE}/images_tracks/{overlay_file}"
    })


if __name__ == '__main__':
    app.run(debug=True)
