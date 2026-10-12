"""Wan2.2 video adapters mapped onto either Prism video expert."""


def map_video_lora(state, fused_layers, remaining_layers):
    converted = {}
    for key, tensor in state.items():
        key = key.removeprefix('diffusion_model.').removeprefix('transformer.')
        if key.startswith('blocks.'):
            _, index, suffix = key.split('.', 2)
            index = int(index)
            if not 0 <= index < fused_layers + remaining_layers:
                raise ValueError(f'Prism video LoRA block out of range: {key}')
            prefix = f'fusion_blocks.{index}.video_block' if index < fused_layers else f'remaining_video_blocks.{index - fused_layers}'
        elif key.startswith(('head.', 'patch_embedding.', 'text_embedding.', 'time_embedding.', 'time_projection.')):
            prefix, suffix = 'video_dit', key
        else:
            raise ValueError(f'Unsupported Prism video LoRA key: {key}')
        # MMGP owns dtype, alpha/rank, deltas, step multipliers and residency.
        converted[f'{prefix}.{suffix}'] = tensor
    return converted
