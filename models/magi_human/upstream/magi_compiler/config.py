class _OffloadConfig:
    def __init__(self):
        self.gpu_resident_weight_ratio = 1.0


class CompileConfig:
    def __init__(self):
        self.offload_config = _OffloadConfig()
