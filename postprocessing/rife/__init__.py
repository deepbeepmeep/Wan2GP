RIFE_TEMPORAL_UPSAMPLING_MODES = ("", "rife2", "rife4")


def validate_temporal_upsampling(temporal_upsampling, *, source_is_image: bool = False) -> str:
    temporal_upsampling = temporal_upsampling or ""
    if temporal_upsampling not in RIFE_TEMPORAL_UPSAMPLING_MODES:
        return f"Unknown temporal upsampling mode: {temporal_upsampling}"
    if source_is_image and len(temporal_upsampling) > 0:
        return "Temporal Upsampling can not be used with an Image"
    return ""
