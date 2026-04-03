from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


def _register_family_lora_args(parser, family_handlers: Sequence[str], lora_root: str) -> None:
    registered_families = set()
    for path in family_handlers:
        handler = importlib.import_module(path).family_handler
        family_name = handler.query_model_family()
        family_key = family_name or path
        if family_key in registered_families:
            continue
        if hasattr(handler, "register_lora_cli_args"):
            handler.register_lora_cli_args(parser, lora_root)
        registered_families.add(family_key)


def _arg_provided(argv: Sequence[str], name: str) -> bool:
    return any(one_arg == name or str(one_arg).startswith(f"{name}=") for one_arg in argv)


def parse_wgp_args(family_handlers: Sequence[str], config_filename: str, default_lora_root: str, argv: Sequence[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt or image using Gradio")
    add = parser.add_argument

    add("--save-masks", action="store_true", help="save proprocessed masks for debugging or editing")
    add("--save-speakers", action="store_true", help="save proprocessed audio track with extract speakers for debugging or editing")
    add("--debug-gen-form", action="store_true", help="View form generation / refresh time")
    add("--betatest", action="store_true", help="test unreleased features")
    add("--vram-safety-coefficient", type=float, default=0.8, help="max VRAM (between 0 and 1) that should be allocated to preloaded models")
    add("--share", action="store_true", help="Create a shared URL to access webserver remotely")
    add("--lock-config", action="store_true", help="Prevent modifying the configuration from the web interface")
    add("--lock-model", action="store_true", help="Prevent switch models")
    add("--save-quantized", action="store_true", help="Save a quantized version of the current model")
    add("--test", action="store_true", help="Load the model and exit generation immediately")
    add("--preload", type=str, default="0", help="Megabytes of the diffusion model to preload in VRAM")
    add("--multiple-images", action="store_true", help="Allow inputting multiple images with image to video")
    add("--loras", type=str, default="", help="Root folder for LoRAs (default: loras)")
    _register_family_lora_args(parser, family_handlers, default_lora_root)
    add("--check-loras", action="store_true", help="Filter Loras that are not valid")
    add("--lora-preset", type=str, default="", help="Lora preset to preload")
    add("--settings", type=str, default="settings", help="Path to settings folder")
    add("--config", type=str, default="", help=f"Path to config folder for {config_filename} and queue.zip")
    add("--profile", type=str, default=-1, help="Profile No")
    add("--verbose", type=str, default=1, help="Verbose level")
    add("--debug-deepy", type=str, default=None, help="Enable Deepy verbose debug logging and write it to the given folder")
    add("--steps", type=int, default=0, help="default denoising steps")
    add("--frames", type=int, default=0, help="default number of frames")
    add("--seed", type=int, default=-1, help="default generation seed")
    add("--advanced", action="store_true", help="Access advanced options by default")
    add("--fp16", action="store_true", help="For using fp16 transformer model")
    add("--bf16", action="store_true", help="For using bf16 transformer model")
    add("--server-port", type=str, default=0, help="Server port")
    add("--theme", type=str, default="", help="set UI Theme")
    add("--perc-reserved-mem-max", type=float, default=0, help="percent of RAM allocated to Reserved RAM")
    add("--server-name", type=str, default="", help="Server name")
    add("--gpu", type=str, default="", help="Default GPU Device")
    add("--open-browser", action="store_true", help="open browser")
    add("--t2v", action="store_true", help="text to video mode")
    add("--i2v", action="store_true", help="image to video mode")
    add("--t2v-14B", action="store_true", help="text to video mode 14B model")
    add("--t2v-1-3B", action="store_true", help="text to video mode 1.3B model")
    add("--vace-1-3B", action="store_true", help="Vace ControlNet 1.3B model")
    add("--i2v-1-3B", action="store_true", help="Fun InP image to video mode 1.3B model")
    add("--i2v-14B", action="store_true", help="image to video mode 14B model")
    add("--compile", action="store_true", help="Enable pytorch compilation")
    add("--listen", action="store_true", help="Server accessible on local network")
    add("--attention", type=str, default="", help="attention mode")
    add("--vae-config", type=str, default="", help="vae config mode")
    add("--process", type=str, default="", help="Process a saved queue (.zip) or settings file (.json) without launching the web UI")
    add("--ask-deepy", action="store_true", help="Start an interactive Deepy console session without launching the web UI")
    add("--dry-run", action="store_true", help="Validate file without generating (use with --process)")
    add("--output-dir", type=str, default="", help="Override output directory for CLI processing (use with --process)")
    add("--refresh-catalog", action="store_true", help="Refresh local plugin metadata for installed external plugins")
    add("--refresh-full-catalog", action="store_true", help="Refresh local plugin metadata for all catalog plugins")
    add("--merge-catalog", action="store_true", help="Merge plugins_local.json into plugins.json and remove plugins_local.json")

    args = parser.parse_args(argv)
    if args.ask_deepy and not _arg_provided(argv, "--verbose"):
        args.verbose = "0"
    return args
