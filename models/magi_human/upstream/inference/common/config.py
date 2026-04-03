# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Literal, Tuple

import torch
from inference.utils import env_is_true, print_rank_0
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class EngineConfig(BaseModel):
    # Basic settings
    seed: int = Field(1234, description="Random seed used for python, numpy, pytorch, and cuda.")
    load: str | None = Field(None, description="Directory containing a model checkpoint.")

    # Parallelism strategy
    distributed_backend: Literal["nccl", "gloo"] = Field("nccl", description="Distributed backend. Choices: ['nccl', 'gloo'].")
    distributed_timeout_minutes: int = Field(10, description="Timeout minutes for torch.distributed.")
    sequence_parallel: bool = Field(False, description="Enable sequence parallel optimization.")
    tp_size: int = Field(1, description="Degree of tensor model parallelism.")
    pp_size: int = Field(1, description="Degree of pipeline model parallelism.")
    cp_size: int = Field(1, description="Degree of context parallelism.")
    dp_size: int = Field(1, description="Degree of data parallelism.")


class ModelConfig(BaseModel):
    """Model configuration class defining various parameters for video generation model"""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    num_layers: int = Field(default=40, description="Number of Transformer layers")
    hidden_size: int = Field(default=5120, description="Hidden size of the Transformer model")
    head_dim: int = Field(default=128, description="Dimension per attention head")
    num_query_groups: int = Field(default=8, description="Number of query groups for grouped-query attention")
    video_in_channels: int = Field(default=48 * 4, description="Number of video input channels after patch embedding")
    audio_in_channels: int = Field(default=64, description="Number of audio input channels")
    text_in_channels: int = Field(default=3584, description="Number of text input channels")
    checkpoint_qk_layernorm_rope: bool = Field(default=False, description="Enable checkpointing for QK layernorm + RoPE")
    params_dtype: torch.dtype | str = Field(default=torch.float32, description="Parameter dtype")
    tread_config: dict = Field(
        default=dict(
            selection_rate=0.5, start_layer_idx=2, end_layer_idx=25  # after forward of 0, 1  # before forward of 26 27 28 29
        ),
        description="TReAD (Token Routing and Early Drop) configuration",
    )
    mm_layers: list[int] = Field(default=[0, 1, 2, 3, 36, 37, 38, 39], description="Indices of multimodal fusion layers")
    local_attn_layers: list[int] = Field(default=[], description="Indices of local attention layers")
    enable_attn_gating: bool = Field(default=True, description="Enable attention gating")
    activation_type: str = Field(default="swiglu7", description="Activation type")
    gelu7_layers: list[int] = Field(default=[0, 1, 2, 3], description="Indices of gelu7 layers")

    # Add computed fields
    num_heads_q: int = Field(default=0, description="Number of query heads (calculated from hidden_size // head_dim)")
    num_heads_kv: int = Field(default=0, description="Number of key-value heads (calculated from num_query_groups)")
    post_norm_layers: list[int] = Field(default=[], description="Indices of post norm layers")

    @field_serializer("params_dtype")
    def serialize_dtype(self, value: torch.dtype | str) -> str:
        return str(value)

    @field_validator("params_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            if value == "torch.float32" or value == "float32":
                return torch.float32
            elif value == "torch.float16" or value == "float16":
                return torch.float16
            elif value == "torch.bfloat16" or value == "bfloat16":
                return torch.bfloat16
        raise ValueError(f"Unknown torch.dtype string: '{value}'")


class DataProxyConfig(BaseModel):
    t_patch_size: int = Field(default=1, description="Patch size for time dimension")
    patch_size: int = Field(default=2, description="Patch size for spatial dimensions")
    frame_receptive_field: int = Field(default=11, description="Frame receptive field")
    spatial_rope_interpolation: Literal["inter", "extra"] = Field(
        default="extra", description="Spatial rope interpolation method."
    )
    ref_audio_offset: int = Field(default=1000, description="Offset for reference audio.")
    text_offset: int = Field(default=0, description="Offset for text.")
    coords_style: Literal["v1", "v2"] = Field(default="v2", description="Coords style.")


class EvaluationConfig(BaseModel):
    """Evaluation configuration class defining parameters for model evaluation and inference"""

    model_config = ConfigDict(protected_namespaces=())

    data_proxy_config: DataProxyConfig = Field(default=DataProxyConfig(), description="Data proxy configuration")

    fps: int = Field(default=25, description="Frames per second for video generation")
    num_inference_steps: int = Field(default=32, description="Number of denoising steps during inference")
    video_txt_guidance_scale: float = Field(default=5.0, description="Video text guidance scale for text conditioning")
    audio_txt_guidance_scale: float = Field(default=5.0, description="Audio text guidance scale for text conditioning")
    txt_encoder_type: Literal["t5_gemma"] = Field(default="t5_gemma", description="Text encoder type.")
    t5_gemma_target_length: int = Field(default=640, description="Target length for T5-Gemma encoder.")
    support_ref_audio: bool = Field(default=True, description="Whether to support the ref_audio feature")
    shift: float = Field(default=5.0, description="Temporal shift parameter for video generation")
    exp_name: str = Field(default="exp_debug", description="Experiment name with evaluation suffix")
    audio_model_path: str = Field(default="", description="Path to the pretrained audio model")
    txt_model_path: str = Field(default="", description="Path to the pretrained txt model")
    vae_model_path: str = Field(default="", description="Path to the pretrained vae model")
    vae_stride: Tuple[int, int, int] = Field(default=(4, 16, 16), description="VAE stride in format (time, height, width)")
    z_dim: int = Field(default=48, description="Dimension of z space.")
    patch_size: Tuple[int, int, int] = Field(default=(1, 2, 2), description="Patch size in format (time, height, width)")
    cfg_number: int = Field(default=2, description="Classifier-free guidance number")
    sr_cfg_number: int = Field(default=2, description="SR Classifier-free guidance number")

    # flops recording
    enable_flops_recording: bool = Field(default=False, description="Whether to enable flops recording")

    # super resolution model configuration
    use_sr_model: bool = Field(default=False, description="Whether to use the super resolution model")
    sr_model_path: str = Field(default="", description="Path to the pretrained super resolution model")
    sr_num_inference_steps: int = Field(default=5, description="Number of denoising steps during super resolution inference")
    noise_value: int = Field(default=220, description="Noise value for the super resolution model")
    sr_video_txt_guidance_scale: float = Field(
        default=3.5, description="Super resolution video text guidance scale for text conditioning"
    )
    use_cfg_trick: bool = Field(default=True, description="Whether to use the cfg trick")
    cfg_trick_start_frame: int = Field(default=13, description="Start frame for the cfg trick")
    cfg_trick_value: float = Field(default=2.0, description="Value for the cfg trick")
    using_sde_flag: bool = Field(default=False, description="Whether to use the sde flag")
    sr_audio_noise_scale: float = Field(default=0.7, description="Noise scale for the super resolution audio")

    # turbo-vae config
    use_turbo_vae: bool = Field(default=True, description="Whether to use the turbo-vae")
    student_config_path: str = Field(default="", description="Path to the student config")
    student_ckpt_path: str = Field(default="", description="Path to the student checkpoint")


class MagiPipelineConfig(BaseSettings):
    engine_config: EngineConfig = Field(description="Engine configuration.", default_factory=EngineConfig)
    arch_config: ModelConfig = Field(default=ModelConfig(), description="Model configuration.")
    evaluation_config: EvaluationConfig = Field(default=EvaluationConfig(), description="Evaluation configuration.")
    sr_arch_config: ModelConfig = Field(default=ModelConfig(), description="Super resolution model configuration.")
    model_config = SettingsConfigDict(cli_parse_args=True, cli_ignore_unknown_args=True, cli_implicit_flags=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--config-load-path", type=str, default=None, help="Path to load the config.json from")
        args, _ = parser.parse_known_args()
        config_load_path = args.config_load_path
        sources = [env_settings, CliSettingsSource(settings_cls, cli_parse_args=True, cli_ignore_unknown_args=True)]
        if config_load_path:
            sources.append(JsonConfigSettingsSource(settings_cls, json_file=config_load_path))

        sources.extend([init_settings, dotenv_settings, file_secret_settings])
        return tuple(sources)

    def save_to_json(self, json_path: str, indent: int = 4):
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.__str__(indent=indent))

    def __str__(self, indent: int = 4):
        data = self.model_dump(mode="json")
        formatted = json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=False)
        class_name = self.__class__.__name__
        return f"{class_name}:\n{formatted}".replace('"', "")

    def __repr__(self, indent: int = 4):
        return self.__str__(indent=indent)

    @model_validator(mode="after")
    def validate_engine_config(self):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.engine_config.dp_size = world_size // (
            self.engine_config.tp_size * self.engine_config.pp_size * self.engine_config.cp_size
        )

        assert world_size % self.engine_config.tp_size == 0
        tp_pp_size = self.engine_config.tp_size * self.engine_config.pp_size
        assert world_size % tp_pp_size == 0
        tp_pp_cp_size = tp_pp_size * self.engine_config.cp_size
        assert world_size % tp_pp_cp_size == 0
        assert world_size == self.engine_config.dp_size * tp_pp_cp_size

        if self.engine_config.tp_size == 1:
            self.engine_config.sequence_parallel = False

        return self

    @model_validator(mode="after")
    def post_override_config(self):
        self.arch_config.num_heads_q = self.arch_config.hidden_size // self.arch_config.head_dim
        self.arch_config.num_heads_kv = self.arch_config.num_query_groups

        self.sr_arch_config = copy.deepcopy(self.arch_config)
        if env_is_true("SR2_1080"):
            self.sr_arch_config = copy.deepcopy(self.arch_config)
            # fmt: off
            self.sr_arch_config.local_attn_layers = [
                0, 1, 2,
                4, 5, 6,
                8, 9, 10,
                12, 13, 14,
                16, 17, 18,
                20, 21, 22,
                24, 25, 26,
                28, 29, 30,
                32, 33, 34,
                35, 36, 37,
                38, 39,
            ]
            # fmt: on
            self.evaluation_config.sr_video_txt_guidance_scale = 3.5

        return self


def prevent_unsupported_list_syntax():
    """
    Check sys.argv before Pydantic parsing to prevent using unsupported list syntax.
    """
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if i + 2 < len(args):
            value1, value2 = args[i + 1], args[i + 2]
            if not value1.startswith("-") and not value2.startswith("-"):
                error_msg = (
                    f"\n\nError: Detected list parameter '{arg}' using unsupported command line syntax.\n"
                    f"Error pattern: '{arg} {value1} {value2} ...'\n\n"
                    "Pydantic (or related libraries) do not support passing lists with space-separated multiple values.\n"
                    "Please use one of the following supported formats:\n\n"
                    f"1. JSON style:      {arg} '[{value1},{value2},...]'\n"
                    f"2. Argparse style:  {arg} {value1} {arg} {value2}\n"
                    f"3. Lazy style:      {arg} {value1},{value2}\n"
                )
                raise ValueError(error_msg)


def parse_config(verbose: bool = False) -> MagiPipelineConfig:
    parser = argparse.ArgumentParser(description="Load and optionally save config", allow_abbrev=False)
    parser.add_argument("--config-save-path", type=str, default=None, help="Path to save the config.json to")
    args, _ = parser.parse_known_args()

    prevent_unsupported_list_syntax()
    config = MagiPipelineConfig()

    if args.config_save_path is not None:
        config.save_to_json(args.config_save_path)

    if verbose:
        print_rank_0(config)

    return config
