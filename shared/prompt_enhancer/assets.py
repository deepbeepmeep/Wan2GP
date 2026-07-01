from __future__ import annotations

from typing import Any


FLORENCE2_FOLDER = "Florence2"
LLAMA32_FOLDER = "Llama3_2"
LLAMAJOY_FOLDER = "llama-joycaption-beta-one-hf-llava"
PROMPT_ENHANCER_REPO = "DeepBeepMeep/LTX_Video"

FLORENCE2_FILES = ["config.json", "configuration_florence2.py", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"]
LLAMA32_FILES = ["config.json", "generation_config.json", "Llama3_2_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
LLAMAJOY_FILES = ["config.json", "llama_config.json", "llama_joycaption_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]

QWEN35_TEXT_GGUF_FILENAME = "Qwen3.5-9B-Abliterated-text-Q4_K_M_bis.gguf"
QWEN35_9B_ABLITERATED_TEXT_INT8_VERSION = 1
QWEN35_TEXT_INT8_FILENAME_V1 = "Qwen3.5-9B-Abliterated_quanto_bf16_int8.safetensors"
QWEN35_TEXT_INT8_FILENAME_V2 = "Qwen3.5-9B-Abliterated_v2_quanto_bf16_int8.safetensors"
QWEN35_TEXT_INT8_FILENAME = QWEN35_TEXT_INT8_FILENAME_V2 if QWEN35_9B_ABLITERATED_TEXT_INT8_VERSION == 2 else QWEN35_TEXT_INT8_FILENAME_V1
QWEN35_VISION_FILENAME = "Qwen3.5-9B-vision_bf16.safetensors"
QWEN35_ABLITERATED_REPO = "DeepBeepMeep/Wan2.1"
QWEN35_ABLITERATED_TEXT_REQUIRED_FILES = (
    "chat_template.jinja",
    "config.json",
)
QWEN35_4B_TEXT_GGUF_FILENAME = "Qwen3.5-4B-Abliterated-text-Q4_K_M.gguf"
QWEN35_4B_VISION_FILENAME = "Qwen3.5-4B-vision_bf16.safetensors"
QWEN35_4B_TEXT_INT8_FILENAME = "Qwen3.5-4B-Abliterated_quanto_bf16_int8.safetensors"
QWEN35_VARIANT_9B = "9b"
QWEN35_VARIANT_4B = "4b"
QWEN35_VARIANT_SPECS = {
    QWEN35_VARIANT_9B: {
        "display_name": "Qwen3.5-9B Abliterated",
        "assets_dir_name": "Qwen3_5_9B_Abliterated",
        "root_repo": QWEN35_ABLITERATED_REPO,
        "repo_subfolder": "Qwen3_5_9B_Abliterated",
        "root_files": [
            "chat_template.jinja",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
        "text_repo": QWEN35_ABLITERATED_REPO,
        "text_required_files": list(QWEN35_ABLITERATED_TEXT_REQUIRED_FILES),
        "text_int8_filename": QWEN35_TEXT_INT8_FILENAME,
        "text_int8_tie_word_embeddings": QWEN35_9B_ABLITERATED_TEXT_INT8_VERSION == 2,
        "gguf_repo": QWEN35_ABLITERATED_REPO,
        "text_gguf_filename": QWEN35_TEXT_GGUF_FILENAME,
        "vision_filename": QWEN35_VISION_FILENAME,
        "tie_word_embeddings": False,
    },
    QWEN35_VARIANT_4B: {
        "display_name": "Qwen3.5-4B Abliterated",
        "assets_dir_name": "Qwen3_5_4B_Abliterated",
        "root_repo": QWEN35_ABLITERATED_REPO,
        "repo_subfolder": "Qwen3_5_4B_Abliterated",
        "root_files": [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
        "text_repo": None,
        "text_required_files": [],
        "text_int8_filename": QWEN35_4B_TEXT_INT8_FILENAME,
        "gguf_repo": QWEN35_ABLITERATED_REPO,
        "text_gguf_filename": QWEN35_4B_TEXT_GGUF_FILENAME,
        "vision_filename": QWEN35_4B_VISION_FILENAME,
        "tie_word_embeddings": True,
    },
}


def _qwen35_variant_files(variant: str) -> list[str]:
    spec = QWEN35_VARIANT_SPECS[variant]
    return list(spec["root_files"]) + [spec["vision_filename"], spec["text_int8_filename"], spec["text_gguf_filename"]]


def query_prompt_enhancer_download_defs() -> list[dict[str, Any]]:
    return [
        {
            "repoId": PROMPT_ENHANCER_REPO,
            "sourceFolderList": [FLORENCE2_FOLDER, LLAMA32_FOLDER],
            "fileList": [FLORENCE2_FILES, LLAMA32_FILES],
        },
        {
            "repoId": PROMPT_ENHANCER_REPO,
            "sourceFolderList": [FLORENCE2_FOLDER, LLAMAJOY_FOLDER],
            "fileList": [FLORENCE2_FILES, LLAMAJOY_FILES],
        },
        {
            "repoId": QWEN35_ABLITERATED_REPO,
            "sourceFolderList": [
                QWEN35_VARIANT_SPECS[QWEN35_VARIANT_4B]["repo_subfolder"],
                QWEN35_VARIANT_SPECS[QWEN35_VARIANT_9B]["repo_subfolder"],
            ],
            "fileList": [_qwen35_variant_files(QWEN35_VARIANT_4B), _qwen35_variant_files(QWEN35_VARIANT_9B)],
        },
    ]
