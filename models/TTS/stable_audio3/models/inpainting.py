import random
import torch
from enum import Enum
from typing import List, Optional, Tuple

class MaskType(Enum):
    RANDOM_SEGMENTS = 0  # Legacy: uncontrolled ratio, overlapping segments
    FULL_MASK = 1
    CAUSAL_MASK = 2
    RANDOM_SPANS = 3     # New: explicit ratio control, non-overlapping spans


def _generate_random_spans_mask(
    real_sequence_length: int,
    sequence_length: int,
    device: torch.device,
    max_spans: int = 4,
    mask_ratio_range: Tuple[float, float] = (0.2, 1.0),
    span_count_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Generate a mask with 1-N non-overlapping contiguous spans covering a target ratio.

    Samples a target mask ratio from U[mask_ratio_range], then allocates that budget
    across a variable number of spans with Dirichlet-like random allocation.

    Args:
        real_sequence_length: Number of valid (non-padding) tokens.
        sequence_length: Total sequence length including padding.
        device: Torch device.
        max_spans: Maximum number of masked spans.
        mask_ratio_range: (lo, hi) range for uniform sampling of target mask ratio.
        span_count_weights: Weights for sampling number of spans [1..max_spans].
            Defaults to geometric decay [4, 2, 1, 1] (biased toward fewer spans).
    """
    item_mask = torch.ones((1, 1, sequence_length), device=device, dtype=torch.float32)

    if real_sequence_length == 0:
        return item_mask

    # 1. Sample target mask ratio
    lo, hi = mask_ratio_range
    target_ratio = random.uniform(lo, hi)
    target_masked_tokens = max(1, int(target_ratio * real_sequence_length))

    # 2. Sample number of spans (biased toward fewer spans)
    if span_count_weights is None:
        span_count_weights = [4, 2, 1, 1]

    # Truncate/pad weights to max_spans
    weights = span_count_weights[:max_spans]
    while len(weights) < max_spans:
        weights.append(weights[-1])

    num_spans = random.choices(range(1, max_spans + 1), weights=weights, k=1)[0]
    num_spans = min(num_spans, real_sequence_length)

    # 3. Allocate target tokens across spans (Dirichlet-like)
    raw_weights = [random.random() for _ in range(num_spans)]
    total_weight = sum(raw_weights)
    span_lengths = [max(1, int(w / total_weight * target_masked_tokens)) for w in raw_weights]

    # Adjust to hit exact target
    diff = target_masked_tokens - sum(span_lengths)
    if diff > 0:
        span_lengths[0] += diff
    elif diff < 0:
        for i in range(len(span_lengths)):
            reduce = min(-diff, span_lengths[i] - 1)
            span_lengths[i] -= reduce
            diff += reduce
            if diff >= 0:
                break

    # 4. Place spans without overlap (largest first for better packing)
    span_lengths.sort(reverse=True)
    placed_spans = []

    for length in span_lengths:
        length = min(length, real_sequence_length)
        if length <= 0:
            continue

        max_start = real_sequence_length - length
        if max_start < 0:
            continue

        # Try random placement avoiding overlaps
        placed = False
        for _ in range(50):
            start = random.randint(0, max_start)
            end = start + length
            if not any(s < end and start < e for s, e in placed_spans):
                placed_spans.append((start, end))
                item_mask[:, :, start:end] = 0
                placed = True
                break

        # Fallback: scan for first valid position
        if not placed:
            for start in range(max_start + 1):
                end = start + length
                if not any(s < end and start < e for s, e in placed_spans):
                    placed_spans.append((start, end))
                    item_mask[:, :, start:end] = 0
                    break

    return item_mask


def random_inpaint_mask(
    sequence: torch.Tensor,
    padding_masks: torch.Tensor,
    max_mask_segments: int = 10,
    mask_type_probabilities: Optional[List[float]] = None,
    mask_padding: bool = False,
    force_mask_type: Optional[MaskType] = None,
    # RANDOM_SPANS parameters
    max_spans: int = 4,
    mask_ratio_range: Tuple[float, float] = (0.2, 1.0),
    span_count_weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random inpainting masks for a batch of latent audio sequences.
    The output inpainting mask has 0 where data should be inpainted, and 1 where data is provided.

    Supports two segment masking modes:
        - RANDOM_SEGMENTS (legacy): Uncontrolled mask ratio, overlapping segments allowed.
        - RANDOM_SPANS (new): Explicit target mask ratio sampled from mask_ratio_range,
          non-overlapping contiguous spans with controllable count distribution.

    Args:
        sequence: The input sequence tensor of shape (b, c, sequence_length).
        padding_masks: A tensor of shape (b, sequence_length)
                       where 1 indicates real data latents and 0 indicates latents encoding silence padding.
        max_mask_segments: (RANDOM_SEGMENTS only) Maximum number of segments.
        mask_type_probabilities: Probabilities for each mask type. Length must match the
                                 number of MaskType values used:
                                 - 3 elements: [P(RANDOM_SEGMENTS), P(FULL_MASK), P(CAUSAL_MASK)]
                                 - 4 elements: [P(RANDOM_SEGMENTS), P(FULL_MASK), P(CAUSAL_MASK), P(RANDOM_SPANS)]
                                 If None, defaults to [0.1, 0.8, 0.1] (legacy 3-type behavior).
        mask_padding: If True, zero out padding region in the inpaint mask for
                      RANDOM_SEGMENTS/RANDOM_SPANS/CAUSAL_MASK. Use when training with
                      attention masking for variable-length sequences.
        max_spans: (RANDOM_SPANS only) Maximum number of masked spans.
        mask_ratio_range: (RANDOM_SPANS only) (lo, hi) for uniform sampling of target mask ratio.
        span_count_weights: (RANDOM_SPANS only) Weights for sampling number of spans [1..max_spans].

    Returns:
        A tuple containing:
            - masked_sequence: The sequence with masks applied (original sequence where mask is 1,
                               and usually 0 or a placeholder where mask is 0).
            - inpaint_mask: The generated inpainting mask tensor (0 for inpaint, 1 for keep).
    """
    b, _, sequence_length = sequence.size()

    # Skip probability validation when forcing a specific mask type
    if force_mask_type is not None:
        mask_types_to_sample = None
    else:
        if mask_type_probabilities is None:
            mask_type_probabilities = [0.1, 0.8, 0.1]

        num_probs = len(mask_type_probabilities)

        # Determine which mask types are active based on probability list length
        if num_probs == 3:
            # Legacy: [RANDOM_SEGMENTS, FULL_MASK, CAUSAL_MASK]
            mask_types_to_sample = [MaskType.RANDOM_SEGMENTS.value, MaskType.FULL_MASK.value, MaskType.CAUSAL_MASK.value]
        elif num_probs == 4:
            # Extended: [RANDOM_SEGMENTS, FULL_MASK, CAUSAL_MASK, RANDOM_SPANS]
            mask_types_to_sample = [MaskType.RANDOM_SEGMENTS.value, MaskType.FULL_MASK.value, MaskType.CAUSAL_MASK.value, MaskType.RANDOM_SPANS.value]
        else:
            raise ValueError(
                f"mask_type_probabilities must have 3 or 4 elements. Got {num_probs}."
            )

        if not torch.isclose(torch.tensor(sum(mask_type_probabilities)), torch.tensor(1.0)):
            raise ValueError(
                f"mask_type_probabilities must sum to 1.0. "
                f"Current sum: {sum(mask_type_probabilities)}"
            )

    output_masks_list = []

    for i in range(b):
        padding_mask_single_item = padding_masks[i]
        real_sequence_length = (padding_mask_single_item == 1).sum().item()

        item_mask = torch.ones((1, 1, sequence_length), device=sequence.device, dtype=torch.float32)

        if force_mask_type is not None:
            current_mask_type = force_mask_type
        else:
            chosen_mask_value = random.choices(mask_types_to_sample, weights=mask_type_probabilities, k=1)[0]
            current_mask_type = MaskType(chosen_mask_value)

        if current_mask_type == MaskType.FULL_MASK:
            item_mask = torch.zeros((1, 1, sequence_length), device=sequence.device, dtype=torch.float32)
        elif real_sequence_length == 0:
            pass  # item_mask remains all ones
        else:
            if current_mask_type == MaskType.RANDOM_SEGMENTS:
                # Legacy behavior: uncontrolled ratio, overlapping segments
                num_segments = random.randint(1, max_mask_segments)
                max_len_per_segment_calc = max(1, real_sequence_length // num_segments)

                for _ in range(num_segments):
                    segment_length = random.randint(1, max_len_per_segment_calc)

                    if real_sequence_length - segment_length < 0:
                        continue
                    mask_start = random.randint(0, real_sequence_length - segment_length)
                    item_mask[:, :, mask_start : mask_start + segment_length] = 0

            elif current_mask_type == MaskType.RANDOM_SPANS:
                item_mask = _generate_random_spans_mask(
                    real_sequence_length=real_sequence_length,
                    sequence_length=sequence_length,
                    device=sequence.device,
                    max_spans=max_spans,
                    mask_ratio_range=mask_ratio_range,
                    span_count_weights=span_count_weights,
                )

            elif current_mask_type == MaskType.CAUSAL_MASK:
                unmasked_prefix_len = random.randint(0, real_sequence_length)

                if unmasked_prefix_len < real_sequence_length:
                    item_mask[:, :, unmasked_prefix_len:real_sequence_length] = 0

            # When using attention masking for variable-length training, zero out padding
            # so the model doesn't see it as provided context. Without attention masking,
            # padding is handled via mask_loss_weight and should remain as provided input.
            if mask_padding:
                item_mask[:, :, real_sequence_length:] = 0

        output_masks_list.append(item_mask)

    final_inpaint_mask = torch.cat(output_masks_list, dim=0).to(sequence.device)
    masked_sequence = sequence * final_inpaint_mask
    return masked_sequence, final_inpaint_mask
