# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
# Adapted from transformers v5.17.0, modeling_qwen3_vl.py for inference.
from __future__ import annotations
import itertools
import torch
class Qwen3VLModel:

    def get_vision_position_ids(self, start_position: int, grid_thw: list[int, int, int] | torch.Tensor, temp_merge_size: int=1, spatial_merge_size: int=1, time_interval: int=1, device: str | torch.device | None=None):
        llm_grid_t, llm_grid_h, llm_grid_w = (grid_thw[0].item() // temp_merge_size, grid_thw[1].item() // spatial_merge_size, grid_thw[2].item() // spatial_merge_size)
        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing='ij')
        vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
        vision_position_ids[0] += start_position
        return vision_position_ids

    def get_rope_index(self, input_ids: torch.LongTensor, mm_token_type_ids: torch.IntTensor, image_grid_thw: torch.LongTensor | None=None, video_grid_thw: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        mrope_position_deltas = []
        position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        grid_iters = {1: iter(image_grid_thw) if image_grid_thw is not None else None, 2: iter(video_grid_thw) if video_grid_thw is not None else None}
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]
            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))
            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device)
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return (position_ids, mrope_position_deltas)
