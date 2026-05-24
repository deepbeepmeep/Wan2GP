import os
import sys
import runpy


PACKAGE_NAME = "postprocessing.seedvc"


def _ensure_local_imports():
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)


def app():
    _ensure_local_imports()
    runpy.run_module(f"{PACKAGE_NAME}.app_vc", run_name="__main__")


def app_v2():
    _ensure_local_imports()
    runpy.run_module(f"{PACKAGE_NAME}.app_vc_v2", run_name="__main__")


def app_combined():
    _ensure_local_imports()
    runpy.run_module(f"{PACKAGE_NAME}.app", run_name="__main__")


def infer_v1():
    _ensure_local_imports()
    runpy.run_module(f"{PACKAGE_NAME}.inference", run_name="__main__")


def infer_v2():
    _ensure_local_imports()
    runpy.run_module(f"{PACKAGE_NAME}.inference_v2", run_name="__main__")


def train():
    _ensure_local_imports()
    runpy.run_module(f"{PACKAGE_NAME}.train", run_name="__main__")


def eval():
    _ensure_local_imports()
    runpy.run_module(f"{PACKAGE_NAME}.eval", run_name="__main__")
