def query_download_def(metric=False):
    filename = "depth_anything_v3_metric_large_bf16.safetensors" if metric else "depth_anything_v3_vitl_bf16.safetensors"
    return {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": ["depth"], "fileList": [[filename]]}
