"""Token-ID generation with request-local stopping and progress for audio LMs."""

import torch
from tqdm import tqdm

from .vllm_support import NanoVllmTextEngine


class TokenGenerationEngine(NanoVllmTextEngine):
    @torch.inference_mode()
    def generate_tokens(self, prefix, *, end_token, max_tokens, seed, logits_processor, negative=None, cfg_scale=1.0, callback=None, abort_fn=None, stop_fn=None, stop_min_tokens=0, progress_label="Generating tokens", initial_cache_tokens=0):
        prompt_tokens = max(len(prefix), len(negative) if negative is not None else 0)
        cache_options = dict(kv_cache_initial_tokens=min(initial_cache_tokens, max_tokens), kv_cache_max_tokens=max_tokens, kv_cache_prompt_tokens=prompt_tokens) if initial_cache_tokens else {}
        if cache_options != self._kv_cache_options or (initial_cache_tokens and self._max_num_seqs_hint != (2 if cfg_scale > 1 else 1)):
            self.close()
            self._kv_cache_options = cache_options
            self._max_model_len_hint = self._max_num_seqs_hint = self._max_num_batched_tokens_hint = None
        self.reserve_runtime(prompt_tokens, max_tokens, cfg_scale)
        self._ensure_llm()
        self._llm.scheduler.eos = end_token
        history = []
        with tqdm(total=max_tokens, desc=progress_label, unit="token") as progress:
            def update(token):
                if abort_fn is not None and abort_fn():
                    raise InterruptedError(f"{progress_label} interrupted")
                history.append(token)
                progress.update()
                if callback is not None:
                    callback(step_idx=len(history) - 1, override_num_inference_steps=max_tokens, denoising_extra=f"{progress_label}: {len(history)} tokens", progress_unit="tokens")

            def process(input_ids, logits):
                if abort_fn is not None and abort_fn():
                    raise InterruptedError(f"{progress_label} interrupted")
                if stop_fn is not None and len(history) >= stop_min_tokens and stop_fn():
                    scores = torch.full_like(logits, -torch.inf)
                    scores[:, end_token] = 0
                    return scores
                return logits_processor(logits, history)

            process._requires_input_ids = False

            params = self._sampling_params_cls(temperature=1.0, max_tokens=max_tokens, cfg_scale=cfg_scale, seed=seed, logits_processor=process, logits_processor_update_state=update)
            try:
                outputs = self._llm.generate([prefix], params, use_tqdm=False, unconditional_prompts=[negative] if negative is not None else None)
                tokens = outputs[0]["token_ids"]
                ended = bool(tokens and tokens[-1] == end_token)
                return tokens[:-1] if ended else tokens, not ended
            finally:
                self._llm.reset()
