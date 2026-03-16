# =============================================================================
# Diffusion Schedule
# =============================================================================

# Noise schedule for the distilled pipeline. These sigma values control noise
# levels at each denoising step and were tuned to match the distillation process.
from ...ltx_core.types import SpatioTemporalScaleFactors

DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# Reduced schedule for super-resolution stage 2 (subset of distilled values)
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


# =============================================================================
# Video Generation Defaults
# =============================================================================

DEFAULT_SEED = 10
DEFAULT_1_STAGE_HEIGHT = 512
DEFAULT_1_STAGE_WIDTH = 768
DEFAULT_2_STAGE_HEIGHT = DEFAULT_1_STAGE_HEIGHT * 2
DEFAULT_2_STAGE_WIDTH = DEFAULT_1_STAGE_WIDTH * 2
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_CFG_GUIDANCE_SCALE = 4.0


# =============================================================================
# Audio
# =============================================================================

AUDIO_SAMPLE_RATE = 24000


# =============================================================================
# LoRA
# =============================================================================

DEFAULT_LORA_STRENGTH = 1.0


# =============================================================================
# Video VAE Architecture
# =============================================================================

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
VIDEO_LATENT_CHANNELS = 128


# =============================================================================
# Image Preprocessing
# =============================================================================

# CRF (Constant Rate Factor) for H.264 encoding used in image conditioning.
# Lower = higher quality, 0 = lossless. This mimics compression artifacts.
DEFAULT_IMAGE_CRF = 33


# =============================================================================
# Prompts
# =============================================================================

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
