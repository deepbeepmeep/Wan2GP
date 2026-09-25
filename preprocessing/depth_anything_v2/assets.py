def query_download_def(variant="vitl"):
    filename = "depth_anything_v2_vitb.pth" if variant == "vitb" else "depth_anything_v2_vitl.pth"
    return {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": ["depth"], "fileList": [[filename]]}
