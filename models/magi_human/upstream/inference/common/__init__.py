from .arch import get_arch_memory, is_hopper_arch
from .cpu_offload_wrapper import CPUOffloadWrapper
from .sequence_schema import Modality, VarlenHandler


class EngineConfig:
    pass


class DataProxyConfig:
    def __init__(
        self,
        t_patch_size=1,
        patch_size=2,
        frame_receptive_field=-1,
        spatial_rope_interpolation="extra",
        ref_audio_offset=1000,
        text_offset=0,
        coords_style="v2",
    ):
        self.t_patch_size = t_patch_size
        self.patch_size = patch_size
        self.frame_receptive_field = frame_receptive_field
        self.spatial_rope_interpolation = spatial_rope_interpolation
        self.ref_audio_offset = ref_audio_offset
        self.text_offset = text_offset
        self.coords_style = coords_style


class EvaluationConfig:
    pass


def parse_config(*args, **kwargs):
    raise RuntimeError("The upstream Magi Human CLI config parser is not used in WanGP.")


__all__ = [
    "CPUOffloadWrapper",
    "DataProxyConfig",
    "EngineConfig",
    "EvaluationConfig",
    "Modality",
    "VarlenHandler",
    "get_arch_memory",
    "is_hopper_arch",
    "parse_config",
]
