# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 policy for the bundled Sol-Attn kernels."""

from __future__ import annotations

from mmgp import offload

from shared.attention import pay_attention


SOL_ATTN_THRESH_TYPE = "diag"
SOL_ATTN_MIN_TOKENS = 8192


class MiniMaxH3SolAttention:
    def __init__(self):
        self.enabled = False
        self._runtime_validated = False
        self._announced = False
        self.sink_tokens = 0
        self.tau = 1.0

    def begin_forward(self, layout, device, dtype, tau, grouped_video=False):
        self.sink_tokens = int(layout.video_indices[layout.num_condition_video_rows])
        self.tau = float(tau)
        self.enabled = offload.shared_state.get("_attention") == "sol"
        if self.enabled and grouped_video:
            raise ValueError("MiniMax H3 grouped masked denoising is not compatible with Sol Attention")
        if self.enabled and not self._runtime_validated:
            from shared.sol_attn import validate_runtime
            capability = validate_runtime(device, dtype)
            self._runtime_validated = True
            if not self._announced:
                print(f"[MiniMax H3] Sol-Attn enabled with Triton on SM{capability[0]}{capability[1]} (tau={self.tau:g}, {SOL_ATTN_THRESH_TYPE})")
                self._announced = True

    def use_for_layer(self, tokens):
        return self.enabled and tokens >= SOL_ATTN_MIN_TOKENS

    def __call__(self, qkv_list, use_sol):
        if not use_sol:
            return pay_attention(qkv_list, recycle_q=True)

        query, key, value = qkv_list
        qkv_list.clear()
        from shared.sol_attn import sol_attn
        output = sol_attn(query, key, value, tau=self.tau, thresh_type=SOL_ATTN_THRESH_TYPE,
                          sink_start=0, sink_tokens=self.sink_tokens, int8_qk=True)
        return output


__all__ = ["MiniMaxH3SolAttention"]
