import os
from copy import deepcopy


DEFAULT_KIWI_VARIANT = "instruct_reference"
KIWI_REPO_ID = "DeepBeepMeep/Wan2.2"

KIWI_VARIANT_MAP = {
    "instruct_only": {
        "any_kiwi_source": True,
        "any_kiwi_ref": False,
        "kiwi_ref_embedder": False,
        "kiwi_ref_pad_first": False,
        "kiwi_mllm_repo_id": KIWI_REPO_ID,
        "kiwi_mllm_folder": "kiwi_mllm_encoder_instruct_only",
        "kiwi_text_encoder_file": "instruct_only_mllm_encoder_bf16.safetensors",
        "kiwi_source_embedder_file": "wan2.2_kiwi_edit_5B_instruct_only_source_embedder.safetensors",
        "kiwi_ref_embedder_file": None,
        "config_file": "models/wan/configs/kiwi_edit_instruct_only.json",
    },
    "reference_only": {
        "any_kiwi_source": False,
        "any_kiwi_ref": True,
        "kiwi_ref_embedder": True,
        "kiwi_ref_pad_first": True,
        "kiwi_mllm_repo_id": KIWI_REPO_ID,
        "kiwi_mllm_folder": "kiwi_mllm_encoder_reference_only",
        "kiwi_text_encoder_file": "reference_only_mllm_encoder_bf16.safetensors",
        "kiwi_source_embedder_file": "wan2.2_kiwi_edit_5B_reference_only_source_embedder.safetensors",
        "kiwi_ref_embedder_file": "wan2.2_kiwi_edit_5B_reference_only_ref_embedder.safetensors",
        "config_file": "models/wan/configs/kiwi_edit_reference_only.json",
    },
    "instruct_reference": {
        "any_kiwi_source": True,
        "any_kiwi_ref": True,
        "kiwi_ref_embedder": True,
        "kiwi_ref_pad_first": False,
        "kiwi_mllm_repo_id": KIWI_REPO_ID,
        "kiwi_mllm_folder": "kiwi_mllm_encoder_instruct_reference",
        "kiwi_text_encoder_file": "instruct_reference_mllm_encoder_bf16.safetensors",
        "kiwi_source_embedder_file": "wan2.2_kiwi_edit_5B_instruct_reference_source_embedder.safetensors",
        "kiwi_ref_embedder_file": "wan2.2_kiwi_edit_5B_instruct_reference_ref_embedder.safetensors",
        "config_file": "models/wan/configs/kiwi_edit.json",
    },
}


def detect_kiwi_variant(model_def):
    urls = model_def.get("URLs", [])
    primary_url = ""
    if isinstance(urls, list) and len(urls) > 0:
        primary_url = os.path.basename(str(urls[0])).lower()
    elif isinstance(urls, str):
        primary_url = os.path.basename(urls).lower()
    if "instruct_only" in primary_url:
        return "instruct_only"
    if "reference_only" in primary_url:
        return "reference_only"
    return DEFAULT_KIWI_VARIANT


def get_kiwi_variant_model_def(model_def):
    variant = detect_kiwi_variant(model_def)
    props = deepcopy(KIWI_VARIANT_MAP[variant])
    props["kiwi_variant"] = variant
    props["kiwi_text_encoder_folder"] = props["kiwi_mllm_folder"]
    text_encoder_file = props.get("kiwi_text_encoder_file", "")
    if text_encoder_file.endswith("_bf16.safetensors"):
        props["kiwi_text_encoder_quanto_file"] = text_encoder_file.replace("_bf16.safetensors", "_quanto_bf16_int8.safetensors")
    else:
        props["kiwi_text_encoder_quanto_file"] = os.path.splitext(text_encoder_file)[0] + "_quanto_int8.safetensors"
    source_embedder_file = props.get("kiwi_source_embedder_file", None)
    if source_embedder_file is not None:
        source_stem = os.path.splitext(source_embedder_file)[0]
        props["kiwi_source_embedder_download_path"] = os.path.join("kiwi_embedders", source_stem, props["kiwi_mllm_folder"], "source_embedder", "diffusion_pytorch_model.safetensors")
    ref_embedder_file = props.get("kiwi_ref_embedder_file", None)
    if ref_embedder_file is not None:
        ref_stem = os.path.splitext(ref_embedder_file)[0]
        props["kiwi_ref_embedder_download_path"] = os.path.join("kiwi_embedders", ref_stem, props["kiwi_mllm_folder"], "ref_embedder", "diffusion_pytorch_model.safetensors")
    return props
