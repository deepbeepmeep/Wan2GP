# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
def query_download_def(metric=False):
    filename = "depth_anything_v3_metric_large_bf16.safetensors" if metric else "depth_anything_v3_vitl_bf16.safetensors"
    return {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": ["depth"], "fileList": [[filename]]}
