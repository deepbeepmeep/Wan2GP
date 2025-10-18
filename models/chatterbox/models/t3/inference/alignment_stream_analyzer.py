# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
import contextlib
from dataclasses import dataclass
from types import MethodType
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# (layer_idx, head_idx) pairs to probe for alignment
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


def forward_eager(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask,
    past_key_value = None,
    cache_position= None,
    **kwargs,
):

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


    # ---- Manual attention (eager) ----
    # query scaling
    # (B, nH, T_q, dH) x (B, nH, dH, T_k) -> (B, nH, T_q, T_k)
    attn_scores = torch.matmul(query_states * self.scaling, key_states.transpose(-2, -1))

    # Mask (supports additive masks or boolean causal masks)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(
                attention_mask, torch.finfo(attn_scores.dtype).min
            )
        else:
            # expected shape broadcastable to (B, 1, T_q, T_k)
            attn_scores = attn_scores + attention_mask

    # Softmax in fp32 for numeric stability, then cast back
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Dropout (no-op in eval)
    if self.training and self.attention_dropout and self.attention_dropout > 0:
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)

    # Weighted sum: (B, nH, T_q, T_k) x (B, nH, T_k, dH) -> (B, nH, T_q, dH)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

class  AlignmentStreamAnalyzer:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        """
        Attaches lightweight spies to a few attention layers so we can read their attention maps
        *without* globally enabling output_attentions (which breaks SDPA).
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j - i, device="cpu")
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        # Track generated tokens for repetition detection
        self.generated_tokens = []

        # Collect per-probed-head attention from the last step
        self.last_aligned_attns = []
        self._orig_forwards = {}  # keep originals in case you want to restore

        for buf_idx, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns.append(None)
            self._add_attention_spy(tfmr, buf_idx, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx: int, layer_idx: int, head_idx: int):
        """
        Monkey-patches the target layer's self_attn.forward to:
          - force output_attentions=True just for that call,
          - run under a context that makes SDPA produce attention weights (math kernel),
          - capture the returned attn weights for our analyzer,
          - return the original outputs untouched for upstream callers.
        No weights/state are changed.
        """
        attn_mod = tfmr.layers[layer_idx].self_attn

        # Save original forward once
        if id(attn_mod) not in self._orig_forwards:
            self._orig_forwards[id(attn_mod)] = attn_mod.forward

        orig_forward = self._orig_forwards[id(attn_mod)]
        analyzer = self  # to close over


        # Keep a handle to the real sdpa module; we won't copy or move any params
        original_class = type(attn_mod)  # LlamaSdpaAttention subclass of LlamaAttention

        def wrapped_forward(self_attn, *args, **kwargs):
            # Force this call to produce attention weights, regardless of upstream flags
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True

            out = forward_eager(self_attn, *args, **kwargs)

            step_attention = out[1].detach().cpu()  # (B, H, Tq, Tk)
            analyzer.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]  # (Tq, Tk)
            return out

        # Bind our wrapper as a method of this module instance
        attn_mod.forward = MethodType(wrapped_forward, attn_mod)

        # IMPORTANT: We do NOT touch tfmr.config.output_attentions here.
        # Keeping it False avoids the global SDPA conflict.

    def step(self, logits, next_token=None):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # If we haven't captured all probed attentions yet, skip alignment logic this step.
        if any(x is None for x in self.last_aligned_attns):
            return logits

        # Average over probed heads
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)  # (Tq, Tk)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone().cpu()  # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone().cpu()  # (1, S)

        # TODO: monotonic masking; could have issue b/c spaces are often skipped.
        A_chunk[:, self.curr_frame_pos + 1:] = 0

        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)  # NOTE: very lenient!
        if not discontinuity:
            self.text_position = cur_text_posn

        # Hallucinations at the start of speech show up as activations at the bottom of the attention maps!
        # To mitigate this, we just wait until there are no activations far off-diagonal in the last 2 tokens,
        # and there are some strong activations in the first few tokens.
        if T >= 2 and S >= 4:
            false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        else:
            false_start = not self.started  # conservative very early
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Is generation likely complete?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # NOTE: EOS rarely assigned activations, and second-last token is often punctuation, so use last 3 tokens.
        # NOTE: due to the false-start behaviour, we need to make sure we skip activations for the first few tokens.
        last_text_token_duration = A[15:, -3:].sum()

        # Activations for the final token that last too long are likely hallucinations.
        long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5)  # ~200ms

        # If there are activations in previous tokens after generation has completed, assume repetition error.
        alignment_repetition = self.complete and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        # Track generated tokens for repetition detection
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        # Check for excessive token repetition (2x same token in a row)
        token_repetition = (
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-2:])) == 1
        )

        if token_repetition:
            repeated_token = self.generated_tokens[-1]
            logger.warning(f"🚨 Detected 2x repetition of token {repeated_token}")

        # Suppress EoS to prevent early termination
        if cur_text_posn < S - 3 and S > 5:  # Only suppress if text is longer than 5 tokens
            logits[..., self.eos_idx] = -2**15

        # If a bad ending is detected, force emit EOS by modifying logits
        # NOTE: this means logits may be inconsistent with latents!
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits
