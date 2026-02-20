import os
from typing import Any, Callable, Optional

_PROBE_CACHE = None
_WARNED_REQUESTED_VLLM_NOT_SUPPORTED = False
_TRITON_SMOKE_CACHE = None


def _env_enabled(name, default=True):
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _check_triton_runtime_smoke():
    global _TRITON_SMOKE_CACHE
    if _TRITON_SMOKE_CACHE is not None:
        return _TRITON_SMOKE_CACHE
    try:
        import torch
        import triton
        import triton.language as tl

        if not torch.cuda.is_available():
            _TRITON_SMOKE_CACHE = (False, "CUDA is not available")
            return _TRITON_SMOKE_CACHE

        @triton.jit
        def _smoke_add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(y_ptr + offs, x + 1.0, mask=mask)

        n_elements = 128
        block_size = 128
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.arange(n_elements, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        grid = (triton.cdiv(n_elements, block_size),)
        _smoke_add_one_kernel[grid](x, y, n_elements, BLOCK=block_size)
        torch.cuda.synchronize(device=device)
        if not torch.allclose(y, x + 1.0, atol=1e-5, rtol=1e-5):
            _TRITON_SMOKE_CACHE = (False, "Triton runtime smoke test failed: incorrect output from smoke kernel")
            return _TRITON_SMOKE_CACHE
    except Exception as exc:
        msg = str(exc).replace("\n", " ").strip()
        if len(msg) > 260:
            msg = msg[:260] + "..."
        _TRITON_SMOKE_CACHE = (False, f"Triton runtime smoke test failed: {msg}")
        return _TRITON_SMOKE_CACHE
    _TRITON_SMOKE_CACHE = (True, "ok")
    return _TRITON_SMOKE_CACHE


def _check_triton():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception as exc:
        return False, f"Triton import failed: {exc}"
    if _env_enabled("WGP_VLLM_TRITON_SMOKE", default=True):
        smoke_ok, smoke_msg = _check_triton_runtime_smoke()
        if not smoke_ok:
            return False, smoke_msg
    return True, "ok"


def _check_flash_attention_2():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        version = str(getattr(flash_attn, "__version__", ""))
    except ModuleNotFoundError:
        return False, "non installed"
    except Exception as exc:
        if "no module named 'flash_attn'" in str(exc).strip().lower():
            return False, "non installed"
        return False, f"FlashAttention import failed: {exc}"

    major = None
    if len(version) > 0:
        try:
            major = int(version.split(".", 1)[0])
        except Exception:
            major = None
    if major is not None and major < 2:
        return False, f"FlashAttention major version is {major}, expected >= 2"
    return True, "ok"


def probe_vllm_runtime(force=False):
    global _PROBE_CACHE
    if _PROBE_CACHE is not None and not force:
        return _PROBE_CACHE.copy()

    checks = {}

    triton_ok, triton_msg = _check_triton()
    checks["triton"] = {"ok": triton_ok, "message": triton_msg}

    flash_ok, flash_msg = _check_flash_attention_2()
    checks["flash_attention_2"] = {"ok": flash_ok, "message": flash_msg}

    supported = triton_ok and flash_ok
    result = {
        "supported": supported,
        "preferred_engine": "vllm" if supported else "legacy",
        "checks": checks,
    }

    _PROBE_CACHE = result.copy()
    return result


def resolve_lm_decoder_engine(requested_engine, engines_available = []):
    requested_engine = str(requested_engine or "").strip().lower()
    probe_result = probe_vllm_runtime()
    supported = bool(probe_result.get("supported", False))
    cg_available = "cg" in engines_available
    vllm_available= "vllm" in engines_available
    default_engine = "cg" if cg_available else "legacy"
    if requested_engine == "vllm":
        if supported:
            if vllm_available: return "vllm"
            requested_engine = default_engine
        elif not vllm_available:
            requested_engine = default_engine
        else:
            global _WARNED_REQUESTED_VLLM_NOT_SUPPORTED
            if not _WARNED_REQUESTED_VLLM_NOT_SUPPORTED:
                checks = probe_result.get("checks", {})
                reasons = []
                if isinstance(checks, dict):
                    for check_name, check_data in checks.items():
                        if isinstance(check_data, dict) and not check_data.get("ok", False):
                            msg = str(check_data.get("message", "failed")).replace("\n", " ").strip()
                            if len(msg) > 220:
                                msg = msg[:220] + "..."
                            reasons.append(f"{check_name}={msg}")
                reason_text = "; ".join(reasons) if len(reasons) > 0 else "unknown reason"
                print(f"[LM] Requested decoder engine 'vllm' is not supported at startup ({reason_text}).")
                _WARNED_REQUESTED_VLLM_NOT_SUPPORTED = True
            return default_engine
    if requested_engine == "":
        return "vllm" if supported and vllm_available else default_engine
    if requested_engine in ("legacy", "cg"):
        if not cg_available:
            return "legacy"
        return requested_engine
    print(f"[LM] Unknown decoder engine '{requested_engine}', falling back to 'legacy'.")
    return "legacy"


class NanoVllmTextEngine:
    keep_loaded_for_phase2 = True

    def __init__(self, model, model_path: str, tokenizer):
        self.model = model
        self.model_path = model_path
        self.tokenizer = tokenizer
        self._llm = None
        self._sampling_params_cls = None
        self._max_model_len_hint = None
        self._max_num_seqs_hint = None
        self._max_num_batched_tokens_hint = None
        self._last_failure_reason = ""

    @staticmethod
    def _compute_runtime_hints(prompt_len: int, max_tokens: int, cfg_scale: float):
        max_model_len = max(8, int(prompt_len) + int(max_tokens))
        max_num_seqs = 2 if cfg_scale and cfg_scale > 1.0 else 1
        max_num_batched_tokens = max_model_len * max_num_seqs
        return max_model_len, max_num_seqs, max_num_batched_tokens

    def _ensure_runtime_capacity(self, max_model_len: int, max_num_seqs: int, max_num_batched_tokens: int):
        if self._max_model_len_hint is None:
            self._max_model_len_hint = max_model_len
            self._max_num_seqs_hint = max_num_seqs
            self._max_num_batched_tokens_hint = max_num_batched_tokens
            return

        need_grow = (
            max_model_len > int(self._max_model_len_hint)
            or max_num_seqs > int(self._max_num_seqs_hint)
            or max_num_batched_tokens > int(self._max_num_batched_tokens_hint)
        )
        if not need_grow:
            return

        self._max_model_len_hint = max_model_len
        self._max_num_seqs_hint = max_num_seqs
        self._max_num_batched_tokens_hint = max_num_batched_tokens
        self.close()

    def reserve_runtime(self, prompt_len: int, max_tokens: int, cfg_scale: float):
        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)

    def _ensure_llm(self):
        if self._llm is not None:
            return
        try:
            import torch._inductor.config as inductor_config

            if bool(getattr(inductor_config, "split_reductions", False)):
                inductor_config.split_reductions = False
        except Exception:
            pass
        try:
            from shared.llm_engines.nanovllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"nano-vllm is not available for vllm engine: {exc}") from exc
        if not self.model_path:
            raise RuntimeError("vllm engine requires a model_path")
        max_model_len = self._max_model_len_hint or 4096
        max_num_seqs = self._max_num_seqs_hint or 1
        max_num_batched_tokens = self._max_num_batched_tokens_hint or (max_model_len * max_num_seqs)
        self._llm = LLM(
            model=self.model_path,
            enforce_eager=False,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            tokenizer=self.tokenizer,
            model_object=self.model,
        )
        self._sampling_params_cls = SamplingParams

    def release_runtime_allocations(self):
        if self._llm is None:
            return
        try:
            self._llm.reset_runtime_state()
        except Exception:
            pass

    def close(self):
        llm = getattr(self, "_llm", None)
        self._llm = None
        if llm is not None:
            try:
                close_fn = getattr(llm, "close", None)
                if callable(close_fn):
                    close_fn()
                else:
                    try:
                        llm.reset_runtime_state()
                    except Exception:
                        pass
                    try:
                        llm.clear_graph_cache()
                    except Exception:
                        pass
                    exit_fn = getattr(llm, "exit", None)
                    if callable(exit_fn):
                        exit_fn()
            except Exception:
                pass
            try:
                del llm
            except Exception:
                pass
        self._sampling_params_cls = None

    def __del__(self):
        self.close()

    @staticmethod
    def _extract_text_and_tokens(output_obj) -> tuple[str, list[int]]:
        if output_obj is None:
            return "", []
        if isinstance(output_obj, dict):
            text = str(output_obj.get("text", "") or "")
            token_ids = output_obj.get("token_ids", []) or []
            return text, [int(x) for x in token_ids]
        if hasattr(output_obj, "outputs"):
            outputs = getattr(output_obj, "outputs", None)
            if outputs and len(outputs) > 0:
                text = str(getattr(outputs[0], "text", "") or "")
                token_ids = getattr(outputs[0], "token_ids", None)
                if token_ids is None:
                    token_ids = getattr(outputs[0], "token_ids_list", []) or []
                return text, [int(x) for x in token_ids] if token_ids else []
        text = str(getattr(output_obj, "text", "") or "")
        token_ids = getattr(output_obj, "token_ids", None)
        if token_ids is None:
            token_ids = []
        return text, [int(x) for x in token_ids]

    def get_last_failure_reason(self) -> str:
        return self._last_failure_reason

    def generate_text(
        self,
        prompt: str,
        prompt_negative: str,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        callback=None,
        abort_fn: Optional[Callable[[], bool]] = None,
        logits_processor: Optional[Any] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
        release_vram_after: bool = True,
        ignore_eos: bool = False,
    ):
        del stop_checker
        if abort_fn is not None and abort_fn():
            if release_vram_after:
                self.release_runtime_allocations()
            return None
        try:
            prompt_len = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_len = 0
        if cfg_scale > 1.0 and prompt_negative:
            try:
                prompt_len = max(prompt_len, len(self.tokenizer.encode(prompt_negative)))
            except Exception:
                pass

        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)
        self._ensure_llm()
        if self._llm is None:
            return None
        try:
            self._llm.reset()
        except Exception:
            pass

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} 0/{max_tokens}",
                progress_unit="tokens",
            )

        seed_value = None
        if seed is not None:
            try:
                seed_value = int(seed)
            except Exception:
                seed_value = None
            if seed_value is not None and seed_value < 0:
                seed_value = None

        temp = temperature if temperature is not None and temperature > 0 else 1e-5
        sampling_params = self._sampling_params_cls(
            temperature=temp,
            max_tokens=max_tokens,
            cfg_scale=max(cfg_scale, 1.0),
            top_k=top_k if top_k is not None and top_k > 0 else None,
            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
            ignore_eos=bool(ignore_eos),
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            seed=seed_value,
        )

        text = ""
        token_ids: list[int] = []
        try:
            outputs = self._llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                use_tqdm=True,
                unconditional_prompts=[prompt_negative] if cfg_scale > 1.0 else None,
            )
            if outputs:
                text, token_ids = self._extract_text_and_tokens(outputs[0])
            if (not text) and token_ids:
                try:
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                except Exception:
                    text = ""
            self._last_failure_reason = ""
        except Exception as exc:
            self._last_failure_reason = str(exc)
            raise
        finally:
            if release_vram_after:
                self.release_runtime_allocations()

        if callback is not None:
            callback(
                step_idx=max(0, max_tokens - 1),
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} {max_tokens}/{max_tokens}",
                progress_unit="tokens",
            )

        return {"token_ids": token_ids, "text": text}
