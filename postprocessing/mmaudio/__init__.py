MMAUDIO_MODE_OFF = 0
MMAUDIO_MODE_V2 = 1
MMAUDIO_MODE_NEW = 2
MMAUDIO_PERSIST_UNLOAD = 1
MMAUDIO_PERSIST_RAM = 2
MMAUDIO_STANDARD = "mmaudio_large_44k_v2.pth"
MMAUDIO_ALTERNATE = "mmaudio_large_44k_gold_8.5k_final_fp16.safetensors"
MMAUDIO_MODE_CHOICES = [("Standard", MMAUDIO_MODE_V2), ("NSFW", MMAUDIO_MODE_NEW)]
MMAUDIO_DEFAULT_MODE = MMAUDIO_MODE_CHOICES[0][1]


def validate_mmaudio_remux(config, video_source) -> str:
    if not get_mmaudio_settings(config)[0]:
        return "MMAudio is disabled in Configuration > Extensions"
    from shared.utils.utils import get_video_info
    fps, _, _, frames_count = get_video_info(video_source)
    return "" if frames_count >= round(fps) else "MMAudio can generate an Audio track only if the Video is at least 1s long"


def normalize_mmaudio_config(config):
    mode = config.get("mmaudio_mode", None)
    persistence = config.get("mmaudio_persistence", None)
    if mode is None:
        old = config.get("mmaudio_enabled", 0)
        mode = MMAUDIO_DEFAULT_MODE if old == 0 else MMAUDIO_MODE_V2
    if persistence is None:
        old = config.get("mmaudio_enabled", 0)
        persistence = MMAUDIO_PERSIST_RAM if old == MMAUDIO_PERSIST_RAM else MMAUDIO_PERSIST_UNLOAD
    if mode not in (MMAUDIO_MODE_V2, MMAUDIO_MODE_NEW):
        mode = MMAUDIO_DEFAULT_MODE
    if persistence not in (MMAUDIO_PERSIST_UNLOAD, MMAUDIO_PERSIST_RAM):
        persistence = MMAUDIO_PERSIST_UNLOAD
    config["mmaudio_mode"] = mode
    config["mmaudio_persistence"] = persistence
    return mode, persistence


def get_mmaudio_settings(config):
    mode, persistence = normalize_mmaudio_config(config)
    enabled = mode != MMAUDIO_MODE_OFF
    if mode == MMAUDIO_MODE_V2:
        model_name = "large_44k_v2"
        model_path = MMAUDIO_STANDARD
    elif mode == MMAUDIO_MODE_NEW:
        model_name = "large_44k"
        model_path = MMAUDIO_ALTERNATE
    else:
        model_name = None
        model_path = None
    return enabled, mode, persistence, model_name, model_path
