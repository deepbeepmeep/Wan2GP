from __future__ import annotations

from .configuration_acestep_v15 import AceStepConfig
from .modeling_acestep_v15_turbo import AceStepConditionGenerationModel as _AceStepHFModel


class AceStepConditionGenerationModel(_AceStepHFModel):
    @classmethod
    def from_config(cls, config):
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        else:
            config = dict(config)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        return cls(AceStepConfig(**config))
