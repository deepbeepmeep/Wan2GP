"""Parallel drafters using the established target verification/rejection sampler."""
import torch

from .model_runner import ModelRunner
from ..utils.context import reset_context


class BlockDraftRunner(ModelRunner):
    def _can_batch_dflash_acceptance(self, seq):
        from .speculative_sampling import can_batch_acceptance
        return (self.model.mtp.method == "dflash2"
                and not getattr(self, "_disable_dflash_gpu_acceptance", False)
                and can_batch_acceptance(self, seq))

    def _store_speculative_pending(self, seq, target_logits, hidden_states, positions):
        super()._store_speculative_pending(seq, target_logits, hidden_states, positions)
        self._speculative_pending[seq.seq_id]["features"] = self.model._draft_features[:, -1:].clone()

    def _prime_mtp_context(self, seq):
        self.model.mtp.reset_sequence_state()
        self._prepare_target_speculative_state()
        tokens, count = seq.token_ids, seq.num_tokens
        chunk = int(getattr(self.model, "_prefill_chunk_tokens", 0)) or 1024
        try:
            for start in range(0, count, chunk):
                end = min(count, start + chunk)
                seq.token_ids, seq.num_tokens, seq.num_cached_tokens = tokens[:end], end, start
                ids, positions, embeds = self.prepare_prefill([seq])
                hidden = self.model(input_ids=ids, positions=positions, inputs_embeds=embeds)
                keep = end - start - int(end == count)
                self.model.mtp.append_context(self.model._draft_features[:, :keep], positions[..., :keep])
                reset_context()
            self._store_speculative_pending(seq, self.model.compute_logits(hidden[:, -1:])[0], hidden, positions)
            seq.clear_prompt_data()
            self.speculative_stats["target_passes"] += 1
        finally:
            seq.token_ids, seq.num_tokens = tokens, count
            reset_context()

    @torch.inference_mode()
    def prefill_mtp_suffix(self, seqs, previous_token_count):
        self.ensure_runtime_ready()
        seq = seqs[0]
        pending = self._speculative_pending.pop(seq.seq_id, None)
        seq.num_cached_tokens = previous_token_count if pending is not None else previous_token_count - 1
        if pending is not None:
            self.model.mtp.append_context(pending["features"], pending["positions"])
        ids, positions, embeds = self.prepare_prefill([seq])
        hidden = self.model(input_ids=ids, positions=positions, inputs_embeds=embeds)
        self.model.mtp.append_context(self.model._draft_features[:, :-1], positions[..., :-1])
        self._store_speculative_pending(seq, self.model.compute_logits(hidden)[0], hidden, positions)
        seq.clear_prompt_data()
        reset_context()
        self.speculative_stats["target_passes"] += 1

    def _emit_pending_mtp_token(self, seq, sample_params):
        pending = self._speculative_pending.pop(seq.seq_id)
        token = self._sample_speculative_target(seq, pending["target_logits"], sample_params, [])
        self.model.mtp.append_context(pending["features"], pending["positions"])
        self._speculative_drafts[seq.seq_id] = {"anchor": torch.full((), token, dtype=torch.long, device=pending["features"].device)}
        self.speculative_stats["emitted_tokens"] += 1
        return [[token]]

    def _advance_mtp(self, seq, token_ids, positions, hidden_states):
        self.model.mtp.append_context(self.model._draft_features[:, :len(token_ids)], positions)
        self._speculative_drafts[seq.seq_id] = {"anchor": torch.full((), token_ids[-1], dtype=torch.long, device=hidden_states.device)}

    def _build_mtp_drafts(self, seq, sample_params, draft_count, start_position, profile=None):
        drafter = self.model.mtp
        hidden, unary = drafter.propose(seq.last_token, start_position, self.model.token_embd, self.model.output)
        if drafter.method == "dflash2":
            return self._build_dflash_drafts(seq, sample_params, draft_count, hidden, unary, profile)
        from .speculative_sampling import can_batch_acceptance
        processor = self._speculative_logits_processor(seq, True)
        if (hidden.is_cuda and can_batch_acceptance(self, seq) and not getattr(self, "_disable_dspark_gpu_draft", False)
                and (processor is None or getattr(processor, "_is_token_mask", False))
                and (not seq.predictive_penalty or seq.repetition_penalty in (None, 1.0))):
            return self._build_dspark_gpu_chain(seq, draft_count, hidden, unary)
        tokens, distributions = [], [] if seq.top_k != 1 else None
        previous = seq.last_token
        penalty = 1.0 if sample_params[-1] is None else float(sample_params[-1][0].item())
        confidence_threshold = float(self.model._prompt_enhancer_speculative_confidence)
        for index in range(draft_count):
            logits, confidence = drafter.proposal_logits(hidden[index], unary[index], previous)
            # Confidence scheduling only shortens a proposed block; verification
            # always applies the target distribution and existing rejection rule.
            if confidence is not None and index > 0 and confidence_threshold > 0 and confidence.item() < confidence_threshold:
                break
            logits = self._apply_speculative_logit_rules(seq, logits, penalty, tokens, predictive=True)
            if distributions is not None:
                probs = self._speculative_distribution(seq, logits, sample_params, tokens, predictive=True, profile=profile, profile_role=f"draft{index}", rules_applied=True)
                token = self._sample_distribution(probs)
                distributions.append(probs)
            else:
                token = int(logits.argmax().item())
            tokens.append(token)
            previous = token
        return tokens, drafter.get_cache_length(), distributions

    def _dspark_gpu_chain(self, seq, hidden, unary, anchor, bias, noise, threshold):
        from .dflash_sampling import target_probabilities
        tokens, probabilities, valid_prefix = [], [], []
        previous = anchor
        active = torch.ones((), dtype=torch.bool, device=hidden.device)
        top_k = None if seq.top_k is None else min(seq.top_k, unary.shape[-1])
        for index in range(noise.shape[0]):
            scores, confidence = self.model.mtp.proposal_logits(hidden[index], unary[index], previous)
            scores = scores + bias
            valid = torch.isfinite(scores).any()
            if index and threshold > 0:
                # Match the reference confidence.item() comparison with a Python
                # float, including thresholds just above a representable score.
                valid = valid & (confidence.double().reshape(()) >= threshold)
            active = active & valid
            valid_prefix.append(active)
            if seq.top_k != 1:
                # Sample exactly from the filtered q that acceptance uses (exponential race).
                probs = target_probabilities(scores[None], top_k, seq.top_p, seq.min_p, seq.temperature)[0][0]
                previous = (probs / noise[index]).argmax()
                probabilities.append(probs)
            else:
                previous = scores.argmax()  # lowest token ID among ties, as the reference argmax
            tokens.append(previous)
        return torch.stack(tokens), probabilities, torch.stack(valid_prefix).sum()

    def _build_dspark_gpu_chain(self, seq, draft_count, hidden, unary):
        # One graph for the dependent Markov chain. Confidence exits shorten the
        # valid prefix instead of synchronizing after every draft; inactive
        # suffix drafts are discarded before acceptance.
        threshold = float(self.model._prompt_enhancer_speculative_confidence)
        key = (draft_count, threshold, seq.top_k, seq.top_p, seq.min_p, seq.temperature)
        graphs = getattr(self, "_dspark_sampling_graphs", None)
        if graphs is None:
            graphs = self._dspark_sampling_graphs = {}
        bias = self._get_logits_bias(seq, unary)
        processor = self._speculative_logits_processor(seq, True)
        if processor is not None:
            rule_bias = torch.zeros_like(unary[:1])
            if bias is not None:
                rule_bias.copy_(bias)
            bias = processor(None, rule_bias)[0]
        state = graphs.get(key)
        if state is None:
            if len(graphs) >= 8:
                graphs.pop(next(iter(graphs)))
            inputs = hidden[:draft_count].clone()
            logits = unary[:draft_count].clone()
            anchor = torch.tensor(seq.last_token, dtype=torch.long, device=hidden.device)
            fixed_bias = torch.zeros_like(unary[0])
            if bias is not None:
                fixed_bias.copy_(bias.reshape(-1))
            noise = torch.ones((draft_count, unary.shape[-1]), dtype=torch.float32, device=hidden.device)
            self._dspark_gpu_chain(seq, inputs, logits, anchor, fixed_bias, noise, threshold)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tokens, probabilities, length = self._dspark_gpu_chain(seq, inputs, logits, anchor, fixed_bias, noise, threshold)
            state = (graph, inputs, logits, anchor, fixed_bias, noise, tokens, probabilities, length)
            graphs[key] = state
        graph, inputs, logits, anchor, fixed_bias, noise, tokens, probabilities, length = state
        inputs.copy_(hidden[:draft_count])
        logits.copy_(unary[:draft_count])
        anchor.fill_(seq.last_token)
        fixed_bias.zero_() if bias is None else fixed_bias.copy_(bias.reshape(-1))
        if seq.top_k != 1:
            noise.exponential_(1, generator=self._sampling_generator).clamp_min_(1e-10)
        graph.replay()
        self._draft_valid_length = length
        return tokens, self.model.mtp.get_cache_length(), probabilities if seq.top_k != 1 else None

    def _compact_draft_distribution(self, seq, scores, temperature):
        """Filter only the candidate support, including top-k boundary ties."""
        scores = scores.float().div(temperature)
        top_k = int(seq.top_k) if seq.top_k is not None else 0
        if 0 < top_k < scores.numel():
            threshold = scores.topk(top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        top_p = float(seq.top_p) if seq.top_p is not None and 0 < seq.top_p < 1 else None
        min_p = float(seq.min_p) if seq.min_p is not None and seq.min_p > 0 else None
        return self._filter_speculative_distribution(scores, scores, None, top_p, min_p)

    def _build_dflash_drafts(self, seq, sample_params, draft_count, hidden, unary, profile):
        drafter = self.model.mtp
        tokens, distributions = [], [] if seq.top_k != 1 else None
        previous = seq.last_token
        penalty = 1.0 if sample_params[-1] is None else float(sample_params[-1][0].item())
        processor = self._speculative_logits_processor(seq, True) if hidden.is_cuda else None
        if (hidden.is_cuda and (processor is None or getattr(processor, "_is_token_mask", False))
                and (not seq.predictive_penalty or penalty == 1.0)):
            return self._build_dflash_gpu_chain(seq, draft_count, hidden, unary)
        for index in range(draft_count):
            scores, candidates = drafter.proposal_candidates(hidden[index], unary[index], previous)
            # Process the full logits only for the existing arbitrary grammar,
            # bias and repetition interfaces. Sampling stays on the 16 entries.
            logits = torch.full_like(unary[index], -torch.inf)
            logits.scatter_(0, candidates, scores)
            logits = self._apply_speculative_logit_rules(seq, logits, penalty, tokens, predictive=True)
            scores = logits[candidates]
            if torch.isneginf(scores).all().item():
                break
            if distributions is not None:
                probabilities = self._compact_draft_distribution(seq, scores, sample_params[0][0])
                selected = self._sample_distribution_tensor(probabilities)
                token = int(candidates.gather(0, selected.reshape(1)).item())
                # Verification needs q over the target vocabulary for p-q.
                # Only this final scatter is dense; sorting and RNG are compact.
                full = torch.zeros_like(logits)
                full.scatter_(0, candidates, probabilities)
                distributions.append(full)
            else:
                # Match full-vocabulary argmax's lowest-token-ID tie break.
                winners = candidates.masked_fill(scores != scores.max(), logits.numel())
                token = int(winners.min().item())
            tokens.append(token)
            previous = token
        return tokens, drafter.get_cache_length(), distributions

    def _dflash_gpu_chain(self, seq, hidden, unary, anchor, bias, noise):
        tokens, probabilities, valid_prefix = [], [], []
        previous = anchor
        active = torch.ones((), dtype=torch.bool, device=hidden.device)
        for index in range(noise.shape[0]):
            scores, candidates = self.model.mtp.proposal_candidates(hidden[index], unary[index], previous)
            scores = scores + bias[candidates]
            valid = ~torch.isneginf(scores).all()
            active = active & valid
            valid_prefix.append(active)
            # Inactive suffix slots are discarded before target verification.
            # Keep their arithmetic defined while executing a fixed graph.
            scores = torch.where(valid, scores, torch.zeros_like(scores))
            if seq.top_k != 1:
                probs = self._compact_draft_distribution(seq, scores, seq.temperature)
                previous = candidates.gather(0, (probs / noise[index]).argmax().reshape(1))[0]
                full = torch.zeros_like(unary[index])
                full.scatter_(0, candidates, probs)
                probabilities.append(full)
            else:
                previous = candidates.masked_fill(scores != scores.max(), unary.shape[-1]).min()
            tokens.append(previous)
        return torch.stack(tokens), probabilities, torch.stack(valid_prefix).sum()

    def _build_dflash_gpu_chain(self, seq, draft_count, hidden, unary):
        # Evaluate known suppression/thinking masks once for this proposal.
        # Without history penalties or arbitrary Python grammar, the complete
        # dependent candidate chain can run on the GPU. Other processors retain
        # the general sparse path above.
        key = (draft_count, seq.top_k, seq.top_p, seq.min_p, seq.temperature)
        graphs = getattr(self, "_dflash_sampling_graphs", None)
        if graphs is None:
            graphs = self._dflash_sampling_graphs = {}
        state = graphs.get(key)
        bias = self._get_logits_bias(seq, unary)
        processor = self._speculative_logits_processor(seq, True)
        if processor is not None:
            rule_bias = torch.zeros_like(unary[:1])
            if bias is not None:
                rule_bias.copy_(bias)
            # Preserve processor-after-bias precedence, including a thinking
            # budget forcing its sole close token over an earlier suppression.
            bias = processor(None, rule_bias)[0]
        if state is None:
            if len(graphs) >= 8:
                graphs.pop(next(iter(graphs)))
            inputs = hidden[:draft_count].clone()
            logits = unary[:draft_count].clone()
            anchor = torch.tensor(seq.last_token, dtype=torch.long, device=hidden.device)
            fixed_bias = torch.zeros_like(unary[0])
            if bias is not None:
                fixed_bias.copy_(bias.reshape(-1))
            noise = torch.ones((draft_count, self.model.mtp.candidate_selector.top_k), dtype=torch.float32, device=hidden.device)
            self._dflash_gpu_chain(seq, inputs, logits, anchor, fixed_bias, noise)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tokens, probabilities, length = self._dflash_gpu_chain(seq, inputs, logits, anchor, fixed_bias, noise)
            state = (graph, inputs, logits, anchor, fixed_bias, noise, tokens, probabilities, length)
            graphs[key] = state
        graph, inputs, logits, anchor, fixed_bias, noise, tokens, probabilities, length = state
        inputs.copy_(hidden[:draft_count])
        logits.copy_(unary[:draft_count])
        anchor.fill_(seq.last_token)
        if bias is None:
            fixed_bias.zero_()
        else:
            fixed_bias.copy_(bias.reshape(-1))
        if seq.top_k != 1:
            noise.exponential_(1, generator=self._sampling_generator).clamp_min_(1e-10)
        graph.replay()
        if self._can_batch_dflash_acceptance(seq):
            self._draft_valid_length = length
            return tokens, self.model.mtp.get_cache_length(), probabilities if seq.top_k != 1 else None
        self._draft_valid_length = None
        count = int(length.item())
        return tokens[:count], self.model.mtp.get_cache_length(), probabilities[:count] if seq.top_k != 1 else None

    def _sample_verified_block(self, seq, logits, draft_tokens, draft_distributions, sample_params, profile=None):
        from .speculative_sampling import can_batch_acceptance, sample_verified_block
        if self.model.mtp.method == "dspark" and torch.is_tensor(draft_tokens) and logits.is_cuda:
            if can_batch_acceptance(self, seq) and not getattr(self, "_disable_dspark_gpu_acceptance", False):
                return sample_verified_block(self, seq, logits, draft_tokens, draft_distributions, sample_params, profile,
                                             valid_length=self._draft_valid_length, method="dspark")
            # The GPU chain returns a full block; the reference sampler takes its valid prefix.
            count = int(self._draft_valid_length.item())
            return self._sample_verified_block_reference(seq, logits[:count + 1], draft_tokens[:count],
                                                         None if draft_distributions is None else draft_distributions[:count], sample_params, profile)
        if not (torch.is_tensor(draft_tokens) and logits.is_cuda and self._can_batch_dflash_acceptance(seq)):
            return self._sample_verified_block_reference(seq, logits, draft_tokens, draft_distributions, sample_params, profile)
        return sample_verified_block(self, seq, logits, draft_tokens, draft_distributions, sample_params, profile,
                                     valid_length=self._draft_valid_length, method="dflash")

    def reset_runtime_state(self):
        super().reset_runtime_state()
        self._dflash_sampling_graphs = {}
        self._dspark_sampling_graphs = {}
        self._draft_valid_length = None
        self.model._draft_features = None
