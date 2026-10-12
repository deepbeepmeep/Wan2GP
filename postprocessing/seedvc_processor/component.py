# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""SeedVC is GPL-3.0: WanGP does not ship it, it fetches its separate repository the first time SeedVC is used."""
import hashlib
import importlib
import os
import shutil
import sys
import tempfile
import zipfile

SEEDVC_COMPONENT_REPO = "deepbeepmeep/WanGP-SeedVC"
# A new component revision needs both values: python -m postprocessing.seedvc_processor.component <commit>
SEEDVC_COMPONENT_COMMIT = "f59018fd909e06266790a20f4c4a5d67613d6101"
SEEDVC_COMPONENT_DIGEST = "a923430ed55e137a7556dbc1fc48b8b48b63456493f33852e6ff70b3937cab4b"
SEEDVC_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seedvc")
_REVISION_FILE = ".wangp_revision"


def tree_digest(root: str) -> str:
    digest = hashlib.sha256()
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in (".git", "__pycache__"))
        for name in sorted(files):
            if name == _REVISION_FILE:
                continue
            path = os.path.join(folder, name)
            with open(path, "rb") as reader:
                digest.update(os.path.relpath(path, root).replace(os.sep, "/").encode() + b"\0" + hashlib.sha256(reader.read()).digest())
    return digest.hexdigest()


def installed() -> bool:
    # A git checkout of the component repository is a developer setup and is left untouched.
    if os.path.isdir(os.path.join(SEEDVC_COMPONENT_DIR, ".git")):
        return True
    try:
        with open(os.path.join(SEEDVC_COMPONENT_DIR, _REVISION_FILE), encoding="utf-8") as reader:
            return reader.read().strip() == SEEDVC_COMPONENT_COMMIT
    except FileNotFoundError:
        return False


def _download(commit: str, temp_dir: str, gen=None) -> str:
    from shared.utils.download import download_url

    archive = os.path.join(temp_dir, "WanGP-SeedVC.zip")
    download_url(f"https://codeload.github.com/{SEEDVC_COMPONENT_REPO}/zip/{commit}", archive, gen=gen)
    with zipfile.ZipFile(archive) as reader:
        reader.extractall(temp_dir)
        return os.path.join(temp_dir, reader.namelist()[0].split("/")[0])


def ensure_installed(gen=None) -> None:
    if installed():
        return
    parent = os.path.dirname(SEEDVC_COMPONENT_DIR)
    with tempfile.TemporaryDirectory(prefix="_seedvc_", dir=parent, ignore_cleanup_errors=True) as temp_dir:
        source = _download(SEEDVC_COMPONENT_COMMIT, temp_dir, gen)
        if tree_digest(source) != SEEDVC_COMPONENT_DIGEST:
            raise RuntimeError(f"The SeedVC component downloaded from github.com/{SEEDVC_COMPONENT_REPO} does not match revision {SEEDVC_COMPONENT_COMMIT}.")
        if os.path.exists(SEEDVC_COMPONENT_DIR):
            shutil.rmtree(SEEDVC_COMPONENT_DIR)
        shutil.move(source, SEEDVC_COMPONENT_DIR)  # a rename, or a copy while Windows briefly locks the freshly extracted files
    # The revision is written last: an interrupted install is fetched again.
    with open(os.path.join(SEEDVC_COMPONENT_DIR, _REVISION_FILE), "w", encoding="utf-8") as writer:
        writer.write(SEEDVC_COMPONENT_COMMIT)
    importlib.invalidate_caches()  # the running WanGP imports the new package without a restart
    print(f"SeedVC component installed from github.com/{SEEDVC_COMPONENT_REPO} (GPL-3.0)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        print(tree_digest(_download(sys.argv[1], temp_dir)))
