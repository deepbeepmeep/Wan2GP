import os
import torch

__all__ = [
    "C_SCALE",
    "PROMPT_TEMPLATE",
    "MODEL_BASE",
    "PRECISIONS",
    "NORMALIZATION_TYPE",
    "ACTIVATION_TYPE",
    "VAE_PATH",
    "TEXT_ENCODER_PATH",
    "TOKENIZER_PATH",
    "TEXT_PROJECTION",
    "DATA_TYPE",
    "NEGATIVE_PROMPT",
    "NEGATIVE_PROMPT_I2V",
    "FLOW_PATH_TYPE",
    "FLOW_PREDICT_TYPE",
    "FLOW_LOSS_WEIGHT",
    "FLOW_SNR_TYPE",
    "FLOW_SOLVER",
]

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

# =================== Constant Values =====================
# Computation scale factor, 1P = 1_000_000_000_000_000. Tensorboard will display the value in PetaFLOPS to avoid
# overflow error when tensorboard logging values.
C_SCALE = 1_000_000_000_000_000

# When using decoder-only models, we must provide a prompt template to instruct the text encoder
# on how to generate the text.
# --------------------------------------------------------------------
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
) 
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)  

PROMPT_TEMPLATE_ENCODE_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

NEGATIVE_PROMPT = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
NEGATIVE_PROMPT_I2V = "deformation, a poor composition and deformed video, bad teeth, bad eyes, bad limbs"

PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere."},
    {"role": "user", "content": "{}"}
]

PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."},
    {"role": "user", "content": "{}"}
]


PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 95,
    },
    "dit-llm-encode-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_I2V,
        "crop_start": 36,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271
    },
    "dit-llm-encode-video-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO_I2V,
        "crop_start": 103,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271
    },
    "li-dit-encode-image-json": {"template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON, "crop_start": -1}, # auto-calculate crop_start
    "li-dit-encode-video-json": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON, "crop_start": -1}, # auto-calculate crop_start
}

# ======================= Model ======================
PRECISIONS = {"fp32", "fp16", "bf16"}
NORMALIZATION_TYPE = {"layer", "rms"}
ACTIVATION_TYPE = {"relu", "silu", "gelu", "gelu_tanh"}

# =================== Model Path =====================
MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")

# =================== Data =======================
DATA_TYPE = {"image", "video", "image_video"}

# 3D VAE
VAE_PATH = {"884-16c-hy": f"{MODEL_BASE}/hunyuan-video-t2v-720p/vae"}

# Text Encoder
TEXT_ENCODER_PATH = {
    "clipL": f"clip_vit_large_patch14",
    "llm": f"llava-llama-3-8b",
    "llm-i2v": f"llava-llama-3-8b",
}

# Tokenizer
TOKENIZER_PATH = {
    "clipL": f"clip_vit_large_patch14",
    "llm": f"llava-llama-3-8b",
    "llm-i2v": f"llava-llama-3-8b",
}

TEXT_PROJECTION = {
    "linear",  # Default, an nn.Linear() layer
    "single_refiner",  # Single TokenRefiner. Refer to LI-DiT
}

# Flow Matching path type
FLOW_PATH_TYPE = {
    "linear",               # Linear trajectory between noise and data
    "gvp",                  # Generalized variance-preserving SDE
    "vp",                   # Variance-preserving SDE
}

# Flow Matching predict type
FLOW_PREDICT_TYPE = {
    "velocity",             # Predict velocity
    "score",                # Predict score
    "noise",                # Predict noise
}

# Flow Matching loss weight
FLOW_LOSS_WEIGHT = {
    "velocity",             # Weight loss by velocity
    "likelihood",           # Weight loss by likelihood
}

# Flow Matching SNR type
FLOW_SNR_TYPE = {
    "lognorm",              # Log-normal SNR
    "uniform",              # Uniform SNR
}

# Flow Matching solvers
FLOW_SOLVER = {
    "euler",                # Euler solver
}